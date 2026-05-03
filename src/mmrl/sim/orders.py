from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from itertools import count


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    NEW = "new"
    LIVE = "live"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"


_ORDER_IDS = count(1)


@dataclass(slots=True)
class LimitOrder:
    side: Side
    price_int: int
    qty_int: int
    created_time_ms: int
    queue_ahead_int: int
    latency_until_ms: int = 0
    order_id: int = field(default_factory=lambda: next(_ORDER_IDS))
    filled_qty_int: int = 0
    status: OrderStatus = OrderStatus.NEW

    @property
    def remaining_qty_int(self) -> int:
        return max(0, self.qty_int - self.filled_qty_int)

    def mark_live_if_due(self, now_ms: int) -> None:
        if self.status == OrderStatus.NEW and now_ms >= self.latency_until_ms:
            self.status = OrderStatus.LIVE

    def cancel(self) -> None:
        if self.status not in {OrderStatus.FILLED, OrderStatus.CANCELLED}:
            self.status = OrderStatus.CANCELLED


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: int
    side: Side
    price_int: int
    qty_int: int
    time_ms: int
    fee_quote: Decimal
