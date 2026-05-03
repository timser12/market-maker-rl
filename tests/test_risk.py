from decimal import Decimal

from mmrl.data.order_book import BookSnapshot, LocalOrderBook, decimal_to_int
from mmrl.sim.portfolio import Portfolio
from mmrl.sim.risk import QuoteProposal, RiskDecision, RiskManager


def test_risk_blocks_more_buys_when_long():
    tick = Decimal("0.01")
    step = Decimal("0.00001")
    book = LocalOrderBook.from_snapshot(
        BookSnapshot(1, bids=[("99.99", "1.00000")], asks=[("100.01", "1.00000")]), tick, step
    )
    p = Portfolio(inventory=Decimal("0.03"))
    risk = RiskManager(
        max_inventory=Decimal("0.02"),
        max_drawdown_quote=Decimal("100"),
        min_spread_ticks=1,
        max_staleness_ms=1000,
        max_order_size_int=decimal_to_int("0.00100", step),
    )
    result = risk.evaluate(QuoteProposal(1, 1, 100, 100), book, p, now_ms=100, last_market_event_ms=100)
    assert result.decision == RiskDecision.BLOCK_BUY
    assert result.proposal.bid_size_int == 0
    assert result.proposal.ask_size_int == 100
