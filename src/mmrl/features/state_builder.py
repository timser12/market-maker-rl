from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np

from mmrl.data.order_book import LocalOrderBook, int_to_decimal
from mmrl.features.trade_buckets import TradeBucket
from mmrl.sim.orders import Side
from mmrl.sim.portfolio import Portfolio


ORDER_BOOK_CHANNELS = 12
AUX_FEATURE_DIM = 18


@dataclass(slots=True)
class StateBuilder:
    levels: int
    history: int
    tick_size: Decimal
    step_size: Decimal

    previous_bid_qty: np.ndarray | None = None
    previous_ask_qty: np.ndarray | None = None

    frames: deque[np.ndarray] = field(default_factory=deque)
    mid_history: deque[Decimal] = field(default_factory=lambda: deque(maxlen=64))

    def build_order_book_frame(
        self,
        book: LocalOrderBook,
        trade_buckets: dict[int, TradeBucket] | None = None,
    ) -> np.ndarray:
        trade_buckets = trade_buckets or {}
        rows = book.levels_around_mid(self.levels)

        frame = np.zeros((ORDER_BOOK_CHANNELS, self.levels), dtype=np.float32)

        bid_qty = np.array([r[1] for r in rows], dtype=np.float32)
        ask_qty = np.array([r[2] for r in rows], dtype=np.float32)

        bid_qty_dec = bid_qty * float(self.step_size)
        ask_qty_dec = ask_qty * float(self.step_size)

        frame[0] = bid_qty_dec
        frame[1] = ask_qty_dec
        frame[2] = np.log1p(bid_qty_dec)
        frame[3] = np.log1p(ask_qty_dec)
        frame[4] = np.cumsum(bid_qty_dec)
        frame[5] = np.cumsum(ask_qty_dec)

        denom = bid_qty_dec + ask_qty_dec + 1e-9
        frame[6] = (bid_qty_dec - ask_qty_dec) / denom

        frame[7] = np.arange(self.levels, dtype=np.float32) / max(1.0, float(self.levels))

        if self.previous_bid_qty is None:
            delta_bid = np.zeros_like(bid_qty_dec)
            delta_ask = np.zeros_like(ask_qty_dec)
        else:
            delta_bid = np.log1p(bid_qty_dec) - np.log1p(self.previous_bid_qty)
            delta_ask = np.log1p(ask_qty_dec) - np.log1p(self.previous_ask_qty)

        frame[8] = delta_bid
        frame[9] = delta_ask

        best_bid = book.best_bid()
        best_ask = book.best_ask()

        for i in range(self.levels):
            bid_price = best_bid - i if best_bid is not None else None
            ask_price = best_ask + i if best_ask is not None else None

            if ask_price is not None and ask_price in trade_buckets:
                frame[10, i] = float(
                    int_to_decimal(trade_buckets[ask_price].buy_qty, self.step_size)
                )

            if bid_price is not None and bid_price in trade_buckets:
                frame[11, i] = float(
                    int_to_decimal(trade_buckets[bid_price].sell_qty, self.step_size)
                )

        self.previous_bid_qty = bid_qty_dec.copy()
        self.previous_ask_qty = ask_qty_dec.copy()

        return frame

    def push_frame(self, frame: np.ndarray) -> np.ndarray:
        if len(self.frames) >= self.history:
            self.frames.popleft()

        self.frames.append(frame)

        while len(self.frames) < self.history:
            self.frames.appendleft(np.zeros_like(frame))

        return np.stack(list(self.frames), axis=1)

    def aux_vector(
        self,
        portfolio: Portfolio,
        book: LocalOrderBook,
        mid_price: Decimal,
        trade_buckets: dict[int, TradeBucket],
        max_inventory: Decimal,
        open_orders: dict[int, Any],
    ) -> np.ndarray:
        if mid_price > 0:
            self.mid_history.append(mid_price)

        max_inventory_notional = max_inventory * mid_price if mid_price > 0 else Decimal("1")
        if max_inventory_notional <= 0:
            max_inventory_notional = Decimal("1")

        inventory_ratio = self._safe_decimal_ratio(portfolio.inventory, max_inventory)
        cash_norm = self._safe_decimal_ratio(portfolio.cash, max_inventory_notional)
        equity_norm = self._safe_decimal_ratio(portfolio.equity(mid_price), max_inventory_notional)
        unrealized_norm = self._safe_decimal_ratio(
            portfolio.unrealized_pnl(mid_price),
            max_inventory_notional,
        )

        spread_ticks = book.spread_ticks() or 0
        spread_bps = 0.0

        if mid_price > 0:
            spread_price = Decimal(spread_ticks) * self.tick_size
            spread_bps = float(spread_price / mid_price * Decimal("10000"))

        returns = self._mid_returns()

        mid_return_1 = returns[-1] if len(returns) else 0.0
        mean_recent_return = float(np.mean(returns[-10:])) if len(returns) else 0.0
        realized_volatility = float(np.std(returns[-20:])) if len(returns) >= 2 else 0.0
        absolute_return = abs(mid_return_1)

        top_book_imbalance = self._top_book_imbalance(book, n=10)
        microprice_deviation_bps = self._microprice_deviation_bps(book, mid_price)

        buy_volume = 0
        sell_volume = 0
        buy_count = 0
        sell_count = 0

        for bucket in trade_buckets.values():
            buy_volume += bucket.buy_qty
            sell_volume += bucket.sell_qty
            buy_count += bucket.buy_count
            sell_count += bucket.sell_count

        buy_volume_dec = int_to_decimal(buy_volume, self.step_size)
        sell_volume_dec = int_to_decimal(sell_volume, self.step_size)

        buy_volume_norm = self._safe_decimal_ratio(buy_volume_dec, max_inventory)
        sell_volume_norm = self._safe_decimal_ratio(sell_volume_dec, max_inventory)
        flow_delta_norm = self._safe_decimal_ratio(buy_volume_dec - sell_volume_dec, max_inventory)

        trade_count_scaled = min(1.0, float(buy_count + sell_count) / 100.0)

        open_bid_orders = 0
        open_ask_orders = 0

        for order in open_orders.values():
            if getattr(order, "side", None) == Side.BUY:
                open_bid_orders += 1
            elif getattr(order, "side", None) == Side.SELL:
                open_ask_orders += 1

        vector = np.array(
            [
                float(inventory_ratio),
                float(cash_norm),
                float(equity_norm),
                float(unrealized_norm),
                float(spread_ticks) / 100.0,
                spread_bps / 100.0,
                mid_return_1,
                mean_recent_return,
                realized_volatility,
                absolute_return,
                top_book_imbalance,
                microprice_deviation_bps / 100.0,
                float(buy_volume_norm),
                float(sell_volume_norm),
                float(flow_delta_norm),
                trade_count_scaled,
                min(1.0, open_bid_orders / 10.0),
                min(1.0, open_ask_orders / 10.0),
            ],
            dtype=np.float32,
        )

        return np.nan_to_num(vector, nan=0.0, posinf=10.0, neginf=-10.0)

    def portfolio_vector(self, portfolio: Portfolio, mid_price: Decimal) -> np.ndarray:
        """
        Backward-compatible old method.

        The environment should now use aux_vector(), but this stays here so
        older scripts/tests do not immediately explode.
        """
        inv = float(portfolio.inventory)
        cash = float(portfolio.cash)
        equity = float(portfolio.equity(mid_price))
        unrealized = float(portfolio.unrealized_pnl(mid_price))

        return np.array([inv, cash, equity, unrealized], dtype=np.float32)

    def _mid_returns(self) -> np.ndarray:
        if len(self.mid_history) < 2:
            return np.array([], dtype=np.float32)

        mids = np.array([float(x) for x in self.mid_history], dtype=np.float64)

        prev = mids[:-1]
        curr = mids[1:]

        returns = np.zeros_like(curr)
        valid = prev > 0
        returns[valid] = np.log(curr[valid] / prev[valid])

        return returns.astype(np.float32)

    def _top_book_imbalance(self, book: LocalOrderBook, n: int = 10) -> float:
        bid_depth = int_to_decimal(book.total_depth("bid", n), self.step_size)
        ask_depth = int_to_decimal(book.total_depth("ask", n), self.step_size)

        denom = bid_depth + ask_depth

        if denom <= 0:
            return 0.0

        return float((bid_depth - ask_depth) / denom)

    def _microprice_deviation_bps(self, book: LocalOrderBook, mid_price: Decimal) -> float:
        best_bid = book.best_bid()
        best_ask = book.best_ask()

        if best_bid is None or best_ask is None or mid_price <= 0:
            return 0.0

        bid_qty = Decimal(book.bids.get(best_bid, 0)) * self.step_size
        ask_qty = Decimal(book.asks.get(best_ask, 0)) * self.step_size

        denom = bid_qty + ask_qty

        if denom <= 0:
            return 0.0

        bid_price = int_to_decimal(best_bid, self.tick_size)
        ask_price = int_to_decimal(best_ask, self.tick_size)

        microprice = (ask_price * bid_qty + bid_price * ask_qty) / denom

        return float((microprice - mid_price) / mid_price * Decimal("10000"))

    @staticmethod
    def _safe_decimal_ratio(numerator: Decimal, denominator: Decimal) -> float:
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)