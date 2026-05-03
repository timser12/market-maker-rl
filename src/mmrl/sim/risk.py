from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from mmrl.data.order_book import LocalOrderBook, int_to_decimal
from mmrl.sim.orders import Side
from mmrl.sim.portfolio import Portfolio


class RiskDecision(str, Enum):
    ALLOW = "allow"
    BLOCK_BUY = "block_buy"
    BLOCK_SELL = "block_sell"
    CANCEL_ALL = "cancel_all"
    PAUSE = "pause"


@dataclass(frozen=True, slots=True)
class QuoteProposal:
    bid_offset_ticks: int | None
    ask_offset_ticks: int | None
    bid_size_int: int
    ask_size_int: int
    cancel_all: bool = False


@dataclass(frozen=True, slots=True)
class RiskResult:
    decision: RiskDecision
    proposal: QuoteProposal
    reason: str | None = None
    penalty: float = 0.0


@dataclass(slots=True)
class RiskManager:
    max_inventory: Decimal
    max_drawdown_quote: Decimal
    min_spread_ticks: int
    max_staleness_ms: int
    max_order_size_int: int
    starting_equity: Decimal = Decimal("0")

    def evaluate(
        self,
        proposal: QuoteProposal,
        book: LocalOrderBook,
        portfolio: Portfolio,
        now_ms: int,
        last_market_event_ms: int,
    ) -> RiskResult:
        if proposal.cancel_all:
            return RiskResult(RiskDecision.CANCEL_ALL, proposal, "agent_requested_cancel_all")

        spread = book.spread_ticks()
        if spread is None or spread < self.min_spread_ticks:
            return RiskResult(
                RiskDecision.PAUSE,
                QuoteProposal(None, None, 0, 0, cancel_all=True),
                "spread_too_small_or_book_empty",
                penalty=0.01,
            )

        if now_ms - last_market_event_ms > self.max_staleness_ms:
            return RiskResult(
                RiskDecision.CANCEL_ALL,
                QuoteProposal(None, None, 0, 0, cancel_all=True),
                "stale_market_data",
                penalty=0.05,
            )

        mid_i = book.mid_price_int()
        if mid_i is not None:
            equity = portfolio.equity(int_to_decimal(int(mid_i), book.tick_size))
            if self.starting_equity and self.starting_equity - equity > self.max_drawdown_quote:
                return RiskResult(
                    RiskDecision.CANCEL_ALL,
                    QuoteProposal(None, None, 0, 0, cancel_all=True),
                    "drawdown_limit_breached",
                    penalty=1.0,
                )

        bid_size = min(proposal.bid_size_int, self.max_order_size_int)
        ask_size = min(proposal.ask_size_int, self.max_order_size_int)
        bid_offset = proposal.bid_offset_ticks
        ask_offset = proposal.ask_offset_ticks

        if portfolio.inventory >= self.max_inventory:
            bid_offset = None
            bid_size = 0
            return RiskResult(
                RiskDecision.BLOCK_BUY,
                QuoteProposal(bid_offset, ask_offset, bid_size, ask_size),
                "inventory_too_long",
                penalty=0.1,
            )
        if portfolio.inventory <= -self.max_inventory:
            ask_offset = None
            ask_size = 0
            return RiskResult(
                RiskDecision.BLOCK_SELL,
                QuoteProposal(bid_offset, ask_offset, bid_size, ask_size),
                "inventory_too_short",
                penalty=0.1,
            )

        return RiskResult(
            RiskDecision.ALLOW,
            QuoteProposal(bid_offset, ask_offset, bid_size, ask_size),
            None,
            penalty=0.0,
        )

    @staticmethod
    def allows_side(decision: RiskDecision, side: Side) -> bool:
        if decision in {RiskDecision.CANCEL_ALL, RiskDecision.PAUSE}:
            return False
        if decision == RiskDecision.BLOCK_BUY and side == Side.BUY:
            return False
        if decision == RiskDecision.BLOCK_SELL and side == Side.SELL:
            return False
        return True
