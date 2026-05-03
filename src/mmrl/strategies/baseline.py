from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from mmrl.data.order_book import LocalOrderBook, decimal_to_int
from mmrl.sim.portfolio import Portfolio
from mmrl.sim.risk import QuoteProposal


@dataclass(slots=True)
class InventorySkewBaseline:
    """Hardcoded benchmark: quote around mid, widen/skew on inventory."""

    base_size_int: int
    normal_offset_ticks: int = 2
    wide_offset_ticks: int = 6
    skew_threshold: Decimal = Decimal("0.005")

    @classmethod
    def from_sizes(cls, base_size: Decimal, step_size: Decimal) -> "InventorySkewBaseline":
        return cls(base_size_int=decimal_to_int(base_size, step_size))

    def propose(self, book: LocalOrderBook, portfolio: Portfolio) -> QuoteProposal:
        spread = book.spread_ticks()
        if spread is None:
            return QuoteProposal(None, None, 0, 0, cancel_all=True)
        offset = self.wide_offset_ticks if spread <= 1 else self.normal_offset_ticks
        bid_size = self.base_size_int
        ask_size = self.base_size_int
        bid_offset = offset
        ask_offset = offset
        if portfolio.inventory > self.skew_threshold:
            bid_offset = self.wide_offset_ticks
            ask_offset = 1
            bid_size = max(1, bid_size // 2)
        elif portfolio.inventory < -self.skew_threshold:
            bid_offset = 1
            ask_offset = self.wide_offset_ticks
            ask_size = max(1, ask_size // 2)
        return QuoteProposal(bid_offset, ask_offset, bid_size, ask_size)
