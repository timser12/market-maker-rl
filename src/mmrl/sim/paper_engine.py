from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from mmrl.data.order_book import LocalOrderBook, int_to_decimal
from mmrl.data.trades import AggTrade
from mmrl.sim.fill_model import ConservativeQueueFillModel
from mmrl.sim.orders import Fill, LimitOrder, OrderStatus, Side
from mmrl.sim.portfolio import Portfolio
from mmrl.sim.risk import QuoteProposal, RiskDecision, RiskManager, RiskResult


@dataclass(slots=True)
class PaperExecutionEngine:
    book: LocalOrderBook
    fill_model: ConservativeQueueFillModel
    portfolio: Portfolio
    risk_manager: RiskManager
    open_orders: dict[int, LimitOrder] = field(default_factory=dict)
    last_fills: list[Fill] = field(default_factory=list)

    def cancel_all(self) -> None:
        for order in self.open_orders.values():
            order.cancel()
        self.open_orders.clear()

    def place_quotes_from_risk_result(self, result: RiskResult, now_ms: int) -> None:
        if result.decision in {RiskDecision.CANCEL_ALL, RiskDecision.PAUSE} or result.proposal.cancel_all:
            self.cancel_all()
            return

        self.cancel_all()  # conservative cancel/replace each decision interval
        best_bid = self.book.best_bid()
        best_ask = self.book.best_ask()
        if best_bid is None or best_ask is None:
            return
        mid = (best_bid + best_ask) // 2

        p = result.proposal
        if p.bid_offset_ticks is not None and p.bid_size_int > 0:
            bid_price = min(best_bid, mid - p.bid_offset_ticks)
            order = self.fill_model.new_order(Side.BUY, bid_price, p.bid_size_int, now_ms, self.book)
            self.open_orders[order.order_id] = order
        if p.ask_offset_ticks is not None and p.ask_size_int > 0:
            ask_price = max(best_ask, mid + p.ask_offset_ticks)
            order = self.fill_model.new_order(Side.SELL, ask_price, p.ask_size_int, now_ms, self.book)
            self.open_orders[order.order_id] = order

    def process_trade(self, trade: AggTrade) -> list[Fill]:
        fills: list[Fill] = []
        for order_id, order in list(self.open_orders.items()):
            fill = self.fill_model.process_trade(order, trade)
            if fill is not None:
                self.portfolio.apply_fill(fill, self.book.tick_size, self.book.step_size)
                fills.append(fill)
            if order.status in {OrderStatus.FILLED, OrderStatus.CANCELLED}:
                self.open_orders.pop(order_id, None)
        self.last_fills = fills
        return fills

    def current_mid_decimal(self) -> Decimal:
        mid = self.book.mid_price_int()
        if mid is None:
            return Decimal("0")
        return int_to_decimal(int(mid), self.book.tick_size)

    def propose_and_place(self, proposal: QuoteProposal, now_ms: int, last_market_event_ms: int) -> RiskResult:
        result = self.risk_manager.evaluate(
            proposal=proposal,
            book=self.book,
            portfolio=self.portfolio,
            now_ms=now_ms,
            last_market_event_ms=last_market_event_ms,
        )
        self.place_quotes_from_risk_result(result, now_ms)
        return result
