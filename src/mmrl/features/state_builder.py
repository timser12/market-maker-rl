from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from mmrl.data.order_book import LocalOrderBook, int_to_decimal
from mmrl.features.trade_buckets import TradeBucket
from mmrl.sim.portfolio import Portfolio


ORDER_BOOK_CHANNELS = 12


@dataclass(slots=True)
class StateBuilder:
    levels: int
    history: int
    tick_size: Decimal
    step_size: Decimal
    previous_bid_qty: np.ndarray | None = None
    previous_ask_qty: np.ndarray | None = None
    frames: deque[np.ndarray] = field(default_factory=deque)

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
        frame[7] = np.arange(self.levels, dtype=np.float32)

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
                frame[10, i] = float(int_to_decimal(trade_buckets[ask_price].buy_qty, self.step_size))
            if bid_price is not None and bid_price in trade_buckets:
                frame[11, i] = float(int_to_decimal(trade_buckets[bid_price].sell_qty, self.step_size))

        self.previous_bid_qty = bid_qty_dec.copy()
        self.previous_ask_qty = ask_qty_dec.copy()
        return frame

    def push_frame(self, frame: np.ndarray) -> np.ndarray:
        if len(self.frames) >= self.history:
            self.frames.popleft()
        self.frames.append(frame)
        while len(self.frames) < self.history:
            self.frames.appendleft(np.zeros_like(frame))
        # Shape: [channels, time, levels] for CNN/temporal processing.
        return np.stack(list(self.frames), axis=1)

    def portfolio_vector(self, portfolio: Portfolio, mid_price: Decimal) -> np.ndarray:
        inv = float(portfolio.inventory)
        cash = float(portfolio.cash)
        equity = float(portfolio.equity(mid_price))
        unrealized = float(portfolio.unrealized_pnl(mid_price))
        return np.array([inv, cash, equity, unrealized], dtype=np.float32)
