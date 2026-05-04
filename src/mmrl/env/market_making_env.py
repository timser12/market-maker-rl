from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    class _Env:
        metadata = {}

        def reset(self, *, seed=None, options=None):
            return None

    class _Discrete:
        def __init__(self, n: int):
            self.n = n

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Dict(dict):
        pass

    class _Spaces:
        Discrete = _Discrete
        Box = _Box
        Dict = _Dict

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()

from mmrl.config import BotConfig
from mmrl.data.order_book import BookSnapshot, LocalOrderBook, decimal_to_int
from mmrl.data.trades import AggTrade
from mmrl.env.actions import DiscreteActionMapper
from mmrl.features.state_builder import ORDER_BOOK_CHANNELS, StateBuilder
from mmrl.features.trade_buckets import TradeWindowAggregator
from mmrl.sim.fill_model import ConservativeQueueFillModel
from mmrl.sim.paper_engine import PaperExecutionEngine
from mmrl.sim.portfolio import Portfolio
from mmrl.sim.reward import RewardModel
from mmrl.sim.risk import RiskManager


@dataclass(slots=True)
class ReplayEvent:
    kind: str
    payload: dict[str, Any]


class MarketMakingEnv(gym.Env):
    """Gymnasium-style market-making environment.

    Observation:
        order_book: [channels, time, levels]
        portfolio:  [inventory, cash, equity, unrealized]

    Each environment step applies one agent action, then processes market events
    until the configured decision interval is reached.

    The reward is equity-first:
        reward = delta mark-to-market equity - inventory/risk/cancel penalties
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: BotConfig,
        initial_snapshot: BookSnapshot,
        events: list[ReplayEvent],
    ):
        super().__init__()

        self.cfg = cfg
        self.initial_snapshot = initial_snapshot
        self.events = events

        self.event_idx = 0
        self.last_market_event_ms = 0

        self.book = LocalOrderBook.from_snapshot(
            initial_snapshot,
            cfg.tick_size,
            cfg.step_size,
        )
        self.trade_agg = TradeWindowAggregator(cfg.window_ms)
        self.state_builder = StateBuilder(
            cfg.levels,
            cfg.history,
            cfg.tick_size,
            cfg.step_size,
        )

        self.reward_model = RewardModel(max_inventory=cfg.max_inventory)
        self.mapper = DiscreteActionMapper(
            decimal_to_int(cfg.base_order_size, cfg.step_size)
        )

        self.engine = self._new_engine()

        starting_mid = self._mid_price_decimal()
        starting_equity = self.engine.portfolio.equity(starting_mid)
        self.reward_model.reset(starting_equity)

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict(
            {
                "order_book": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(ORDER_BOOK_CHANNELS, cfg.history, cfg.levels),
                    dtype=np.float32,
                ),
                "portfolio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )

    def _new_engine(self) -> PaperExecutionEngine:
        max_order_size_int = decimal_to_int(
            self.cfg.max_order_size,
            self.cfg.step_size,
        )

        risk = RiskManager(
            max_inventory=self.cfg.max_inventory,
            max_drawdown_quote=self.cfg.max_drawdown_quote,
            min_spread_ticks=self.cfg.min_spread_ticks,
            max_staleness_ms=self.cfg.max_staleness_ms,
            max_order_size_int=max_order_size_int,
        )

        fill_model = ConservativeQueueFillModel(
            tick_size=self.cfg.tick_size,
            step_size=self.cfg.step_size,
            maker_fee_bps=self.cfg.maker_fee_bps,
            latency_min_ms=self.cfg.latency_min_ms,
            latency_max_ms=self.cfg.latency_max_ms,
            queue_ahead_fraction=getattr(self.cfg, "queue_ahead_fraction", Decimal("1.0")),
            cancel_ahead_fraction=getattr(self.cfg, "cancel_ahead_fraction", Decimal("0.25")),
            trade_through_fill_fraction=getattr(
                self.cfg,
                "trade_through_fill_fraction",
                Decimal("1.0"),
            ),
        )

        return PaperExecutionEngine(
            book=self.book,
            fill_model=fill_model,
            portfolio=Portfolio(),
            risk_manager=risk,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        self.event_idx = 0
        self.last_market_event_ms = 0

        self.book = LocalOrderBook.from_snapshot(
            self.initial_snapshot,
            self.cfg.tick_size,
            self.cfg.step_size,
        )
        self.trade_agg = TradeWindowAggregator(self.cfg.window_ms)
        self.state_builder = StateBuilder(
            self.cfg.levels,
            self.cfg.history,
            self.cfg.tick_size,
            self.cfg.step_size,
        )

        self.engine = self._new_engine()

        self.reward_model = RewardModel(max_inventory=self.cfg.max_inventory)
        starting_mid = self._mid_price_decimal()
        starting_equity = self.engine.portfolio.equity(starting_mid)
        self.reward_model.reset(starting_equity)

        return self._observation(), {}

    def step(self, action: int):
        if self.event_idx >= len(self.events):
            return self._observation(), 0.0, True, False, {"reason": "end_of_replay"}

        previous_inventory = self.engine.portfolio.inventory
        previous_mid_price = self._mid_price_decimal()
        previous_equity = self.engine.portfolio.equity(previous_mid_price)

        proposal = self.mapper.to_proposal(int(action))

        first_event = self.events[self.event_idx]
        start_ms = self._event_time(first_event)

        risk_result = self.engine.propose_and_place(
            proposal=proposal,
            now_ms=start_ms,
            last_market_event_ms=self.last_market_event_ms or start_ms,
        )

        fills = []
        processed_events = 0
        depth_events = 0
        trade_events = 0

        decision_interval_ms = self._decision_interval_ms()

        while self.event_idx < len(self.events):
            event = self.events[self.event_idx]
            event_ms = self._event_time(event)

            if processed_events > 0 and event_ms - start_ms >= decision_interval_ms:
                break

            self.event_idx += 1
            processed_events += 1

            if event.kind == "depth":
                # Important: update queue position BEFORE applying the book update.
                self.engine.process_depth_update(event.payload, event_ms)

                self.book.apply_depth_event(event.payload)
                self.last_market_event_ms = int(event.payload.get("E", event_ms))
                depth_events += 1

            elif event.kind == "aggTrade":
                trade = AggTrade.from_binance(
                    event.payload,
                    self.cfg.tick_size,
                    self.cfg.step_size,
                )
                self.trade_agg.add(trade)

                new_fills = self.engine.process_trade(trade)
                fills.extend(new_fills)

                self.last_market_event_ms = trade.trade_time_ms
                trade_events += 1

            else:
                raise ValueError(f"unknown event kind {event.kind}")

        current_mid_price = self._mid_price_decimal()
        current_equity = self.engine.portfolio.equity(current_mid_price)

        reward = self.reward_model.compute(
            portfolio=self.engine.portfolio,
            previous_inventory=previous_inventory,
            previous_equity=previous_equity,
            current_equity=current_equity,
            current_mid_price=current_mid_price,
            risk_penalty=risk_result.penalty,
            agent_cancelled=risk_result.reason == "agent_requested_cancel_all",
        )

        terminated = self.event_idx >= len(self.events)
        observation = self._observation()

        info = {
            "risk": risk_result.reason,
            "risk_decision": str(risk_result.decision),
            "reward_breakdown": reward,
            "fills": len(fills),
            "processed_events": processed_events,
            "depth_events": depth_events,
            "trade_events": trade_events,
            "open_orders": len(self.engine.open_orders),
            "inventory": float(self.engine.portfolio.inventory),
            "cash": float(self.engine.portfolio.cash),
            "equity": float(current_equity),
            "mid_price": float(current_mid_price),
            "previous_equity": float(previous_equity),
            "current_equity": float(current_equity),
            "equity_delta": float(current_equity - previous_equity),
            "event_idx": self.event_idx,
        }

        return observation, reward.total, terminated, False, info

    def _observation(self) -> dict[str, np.ndarray]:
        buckets = (
            self.trade_agg.buckets(self.last_market_event_ms)
            if self.last_market_event_ms
            else {}
        )

        frame = self.state_builder.build_order_book_frame(self.book, buckets)
        order_book_state = self.state_builder.push_frame(frame)

        mid = self._mid_price_decimal()
        portfolio_state = self.state_builder.portfolio_vector(
            self.engine.portfolio,
            mid,
        )

        return {
            "order_book": order_book_state.astype(np.float32),
            "portfolio": portfolio_state,
        }

    def _mid_price_decimal(self) -> Decimal:
        mid_ticks = self.book.mid_price_int()

        if mid_ticks is None:
            return Decimal("0")

        return Decimal(str(mid_ticks)) * self.cfg.tick_size

    def _decision_interval_ms(self) -> int:
        """
        Supports cfg.decision_interval_ms if you added it.
        Falls back to cfg.window_ms so your existing config still works.
        """
        return int(getattr(self.cfg, "decision_interval_ms", self.cfg.window_ms))

    @staticmethod
    def _event_time(event: ReplayEvent) -> int:
        if event.kind == "depth":
            return int(event.payload.get("E", 0))

        if event.kind == "aggTrade":
            return int(event.payload.get("T", event.payload.get("E", 0)))

        return 0