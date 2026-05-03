from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Literal


class BookDesyncError(RuntimeError):
    """Raised when Binance depth sequence IDs imply missed updates."""


def decimal_to_int(value: str | Decimal, quantum: Decimal) -> int:
    """Convert a Binance decimal string to an exact fixed-point integer.

    This deliberately refuses non-grid values. We are not using float keys in a
    market-making book. That would be academic malpractice.
    """

    d = Decimal(str(value))
    scaled = d / quantum
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise ValueError(f"{value} is not aligned with quantum {quantum}")
    return int(integral)


def int_to_decimal(value: int, quantum: Decimal) -> Decimal:
    return Decimal(value) * quantum


@dataclass(slots=True)
class BookSnapshot:
    last_update_id: int
    bids: list[tuple[str, str]]
    asks: list[tuple[str, str]]


@dataclass(slots=True)
class LocalOrderBook:
    tick_size: Decimal
    step_size: Decimal
    bids: dict[int, int] = field(default_factory=dict)
    asks: dict[int, int] = field(default_factory=dict)
    last_update_id: int | None = None

    @classmethod
    def from_snapshot(
        cls,
        snapshot: BookSnapshot,
        tick_size: Decimal,
        step_size: Decimal,
    ) -> "LocalOrderBook":
        book = cls(tick_size=tick_size, step_size=step_size)
        book.last_update_id = snapshot.last_update_id
        for price, qty in snapshot.bids:
            book._set_level("bid", price, qty)
        for price, qty in snapshot.asks:
            book._set_level("ask", price, qty)
        return book

    def apply_depth_event(self, event: dict) -> Literal["applied", "ignored"]:
        """Apply a Binance diff-depth event with strict sequence checks.

        Rules implemented from Binance's local order book procedure:
        - ignore events with u <= local lastUpdateId
        - restart if U > local lastUpdateId + 1
        - otherwise set quantities; zero quantity removes the level
        """

        if self.last_update_id is None:
            raise RuntimeError("load a REST snapshot before applying stream events")

        first_update_id = int(event["U"])
        final_update_id = int(event["u"])

        if final_update_id <= self.last_update_id:
            return "ignored"
        if first_update_id > self.last_update_id + 1:
            raise BookDesyncError(
                f"missed depth event: U={first_update_id}, local={self.last_update_id}"
            )

        for price, qty in event.get("b", []):
            self._set_level("bid", price, qty)
        for price, qty in event.get("a", []):
            self._set_level("ask", price, qty)

        self.last_update_id = final_update_id
        return "applied"

    def _set_level(self, side: Literal["bid", "ask"], price: str, qty: str) -> None:
        price_i = decimal_to_int(price, self.tick_size)
        qty_i = decimal_to_int(qty, self.step_size)
        book_side = self.bids if side == "bid" else self.asks
        if qty_i == 0:
            book_side.pop(price_i, None)
        else:
            book_side[price_i] = qty_i

    def best_bid(self) -> int | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> int | None:
        return min(self.asks) if self.asks else None

    def mid_price_int(self) -> float | None:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def spread_ticks(self) -> int | None:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def top_n(self, side: Literal["bid", "ask"], n: int) -> list[tuple[int, int]]:
        data = self.bids if side == "bid" else self.asks
        reverse = side == "bid"
        return sorted(data.items(), reverse=reverse)[:n]

    def levels_around_mid(self, levels: int) -> list[tuple[int, int, int]]:
        """Return aligned rows around mid: (distance_ticks, bid_qty, ask_qty).

        Distance 0 means the first bucket away from mid on both sides. For a
        half-tick mid, the nearest bid and ask remain symmetric in distance.
        """

        mid = self.mid_price_int()
        if mid is None:
            return [(i, 0, 0) for i in range(levels)]

        rows: list[tuple[int, int, int]] = []
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid is None or best_ask is None:
            return [(i, 0, 0) for i in range(levels)]

        for i in range(levels):
            bid_price = best_bid - i
            ask_price = best_ask + i
            rows.append((i, self.bids.get(bid_price, 0), self.asks.get(ask_price, 0)))
        return rows

    def total_depth(self, side: Literal["bid", "ask"], n: int) -> int:
        return sum(q for _, q in self.top_n(side, n))

    def clone_shallow(self) -> "LocalOrderBook":
        return LocalOrderBook(
            tick_size=self.tick_size,
            step_size=self.step_size,
            bids=dict(self.bids),
            asks=dict(self.asks),
            last_update_id=self.last_update_id,
        )

    def apply_snapshot_levels(
        self,
        bids: Iterable[tuple[str, str]],
        asks: Iterable[tuple[str, str]],
        last_update_id: int,
    ) -> None:
        self.bids.clear()
        self.asks.clear()
        self.last_update_id = last_update_id
        for price, qty in bids:
            self._set_level("bid", price, qty)
        for price, qty in asks:
            self._set_level("ask", price, qty)
