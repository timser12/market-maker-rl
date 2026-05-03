from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque

from mmrl.data.trades import AggTrade


@dataclass(slots=True)
class TradeBucket:
    buy_qty: int = 0
    sell_qty: int = 0
    buy_count: int = 0
    sell_count: int = 0

    @property
    def total_qty(self) -> int:
        return self.buy_qty + self.sell_qty

    @property
    def delta(self) -> int:
        return self.buy_qty - self.sell_qty


@dataclass(slots=True)
class TradeWindowAggregator:
    """Aggregates aggTrade events into rolling price-level windows."""

    window_ms: int
    trades: Deque[AggTrade] = field(default_factory=deque)

    def add(self, trade: AggTrade) -> None:
        self.trades.append(trade)
        self.evict_older_than(trade.trade_time_ms)

    def evict_older_than(self, now_ms: int) -> None:
        cutoff = now_ms - self.window_ms
        while self.trades and self.trades[0].trade_time_ms < cutoff:
            self.trades.popleft()

    def buckets(self, now_ms: int | None = None) -> dict[int, TradeBucket]:
        if now_ms is not None:
            self.evict_older_than(now_ms)
        out: dict[int, TradeBucket] = defaultdict(TradeBucket)
        for trade in self.trades:
            bucket = out[trade.price_int]
            if trade.side == "buy":
                bucket.buy_qty += trade.qty_int
                bucket.buy_count += 1
            else:
                bucket.sell_qty += trade.qty_int
                bucket.sell_count += 1
        return dict(out)
