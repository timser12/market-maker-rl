from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Deque

from mmrl.data.order_book import BookDesyncError, BookSnapshot, LocalOrderBook


@dataclass(slots=True)
class DepthSynchronizer:
    """Buffers Binance diff-depth events and joins them with a REST snapshot."""

    tick_size: Decimal
    step_size: Decimal
    buffer_maxlen: int = 20_000
    buffer: Deque[dict] = field(default_factory=deque)
    book: LocalOrderBook | None = None
    first_buffered_u: int | None = None

    def push_event(self, event: dict) -> None:
        if len(self.buffer) >= self.buffer_maxlen:
            self.buffer.popleft()
        if self.first_buffered_u is None:
            self.first_buffered_u = int(event["U"])
        self.buffer.append(event)

    def initialize_from_snapshot(self, snapshot: BookSnapshot) -> LocalOrderBook:
        if self.first_buffered_u is not None and snapshot.last_update_id < self.first_buffered_u:
            raise BookDesyncError(
                "snapshot older than first buffered stream event; fetch another snapshot"
            )

        book = LocalOrderBook.from_snapshot(snapshot, self.tick_size, self.step_size)

        valid = [e for e in self.buffer if int(e["u"]) > snapshot.last_update_id]
        if valid:
            first = valid[0]
            if not (int(first["U"]) <= snapshot.last_update_id + 1 <= int(first["u"])):
                # Binance wording: first event should have lastUpdateId within [U, u].
                # The +1 tolerance supports the common implementation where local id
                # advances to u after applying the first bridging event.
                raise BookDesyncError(
                    f"first valid event does not bridge snapshot: snapshot={snapshot.last_update_id}, "
                    f"U={first['U']}, u={first['u']}"
                )
            for event in valid:
                book.apply_depth_event(event)

        self.book = book
        self.buffer.clear()
        return book
