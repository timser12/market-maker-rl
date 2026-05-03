from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # lets smoke tests run before optional RL deps are installed
    class _Env:
        metadata = {}

        def reset(self, *, seed=None):
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

    Observation is a dict:
    - order_book: [channels, time, levels]
    - portfolio: [inventory, cash, equity, unrealized]
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: BotConfig, initial_snapshot: BookSnapshot, events: list[ReplayEvent]):
        super().__init__()
        self.cfg = cfg
        self.initial_snapshot = initial_snapshot
        self.events = events
        self.event_idx = 0
        self.last_market_event_ms = 0

        self.book = LocalOrderBook.from_snapshot(initial_snapshot, cfg.tick_size, cfg.step_size)
        self.trade_agg = TradeWindowAggregator(cfg.window_ms)
        self.state_builder = StateBuilder(cfg.levels, cfg.history, cfg.tick_size, cfg.step_size)
        self.reward_model = RewardModel()
        self.mapper = DiscreteActionMapper(decimal_to_int(cfg.base_order_size, cfg.step_size))

        self.engine = self._new_engine()

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict(
            {
                "order_book": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(ORDER_BOOK_CHANNELS, cfg.history, cfg.levels),
                    dtype=np.float32,
                ),
                "portfolio": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            }
        )

    def _new_engine(self) -> PaperExecutionEngine:
        max_order_size_int = decimal_to_int(self.cfg.max_order_size, self.cfg.step_size)
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
        )
        return PaperExecutionEngine(self.book, fill_model, Portfolio(), risk)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.event_idx = 0
        self.last_market_event_ms = 0
        self.book = LocalOrderBook.from_snapshot(self.initial_snapshot, self.cfg.tick_size, self.cfg.step_size)
        self.trade_agg = TradeWindowAggregator(self.cfg.window_ms)
        self.state_builder = StateBuilder(self.cfg.levels, self.cfg.history, self.cfg.tick_size, self.cfg.step_size)
        self.engine = self._new_engine()
        return self._observation(), {}

    def step(self, action: int):
        if self.event_idx >= len(self.events):
            return self._observation(), 0.0, True, False, {"reason": "end_of_replay"}

        previous_mid_ticks = Decimal(str(self.book.mid_price_int() or 0))
        proposal = self.mapper.to_proposal(int(action))
        now_ms = self._event_time(self.events[self.event_idx])
        risk_result = self.engine.propose_and_place(proposal, now_ms, self.last_market_event_ms or now_ms)

        fills = []
        event = self.events[self.event_idx]
        self.event_idx += 1
        if event.kind == "depth":
            self.book.apply_depth_event(event.payload)
            self.last_market_event_ms = int(event.payload.get("E", now_ms))
        elif event.kind == "aggTrade":
            trade = AggTrade.from_binance(event.payload, self.cfg.tick_size, self.cfg.step_size)
            self.trade_agg.add(trade)
            fills = self.engine.process_trade(trade)
            self.last_market_event_ms = trade.trade_time_ms
        else:
            raise ValueError(f"unknown event kind {event.kind}")

        current_mid_ticks = Decimal(str(self.book.mid_price_int() or previous_mid_ticks))
        reward = self.reward_model.compute(
            portfolio=self.engine.portfolio,
            fills=fills,
            previous_mid=previous_mid_ticks,
            current_mid=current_mid_ticks,
            risk_penalty=risk_result.penalty,
        )
        terminated = self.event_idx >= len(self.events)
        info = {"risk": risk_result.reason, "reward_breakdown": reward}
        return self._observation(), reward.total, terminated, False, info

    def _observation(self) -> dict[str, np.ndarray]:
        buckets = self.trade_agg.buckets(self.last_market_event_ms) if self.last_market_event_ms else {}
        frame = self.state_builder.build_order_book_frame(self.book, buckets)
        order_book_state = self.state_builder.push_frame(frame)
        mid = self.engine.current_mid_decimal()
        portfolio_state = self.state_builder.portfolio_vector(self.engine.portfolio, mid)
        return {"order_book": order_book_state.astype(np.float32), "portfolio": portfolio_state}

    @staticmethod
    def _event_time(event: ReplayEvent) -> int:
        if event.kind == "depth":
            return int(event.payload.get("E", 0))
        if event.kind == "aggTrade":
            return int(event.payload.get("T", event.payload.get("E", 0)))
        return 0
