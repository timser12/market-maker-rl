from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import random

from mmrl.data.order_book import LocalOrderBook
from mmrl.data.trades import AggTrade
from mmrl.sim.orders import Fill, LimitOrder, OrderStatus, Side


@dataclass(slots=True)
class ConservativeQueueFillModel:
    """Queue-aware fill model for paper trading.

    The simplistic backtester says: price touched, filled. The serious simulator
    says: there was quantity ahead of you, and only aggressive flow can eat it.
    """

    tick_size: Decimal
    step_size: Decimal
    maker_fee_bps: Decimal
    latency_min_ms: int = 50
    latency_max_ms: int = 250
    rng: random.Random = random.Random(7)

    def queue_ahead_for(self, book: LocalOrderBook, side: Side, price_int: int) -> int:
        book_side = book.bids if side == Side.BUY else book.asks
        return book_side.get(price_int, 0)

    def new_order(self, side: Side, price_int: int, qty_int: int, now_ms: int, book: LocalOrderBook) -> LimitOrder:
        delay = self.rng.randint(self.latency_min_ms, self.latency_max_ms)
        return LimitOrder(
            side=side,
            price_int=price_int,
            qty_int=qty_int,
            created_time_ms=now_ms,
            queue_ahead_int=self.queue_ahead_for(book, side, price_int),
            latency_until_ms=now_ms + delay,
        )

    def process_trade(self, order: LimitOrder, trade: AggTrade) -> Fill | None:
        order.mark_live_if_due(trade.trade_time_ms)
        if order.status in {OrderStatus.NEW, OrderStatus.CANCELLED, OrderStatus.FILLED}:
            return None
        if trade.price_int != order.price_int:
            return None

        # Our bid fills from aggressive sell volume. Our ask fills from aggressive buy volume.
        if order.side == Side.BUY and trade.side != "sell":
            return None
        if order.side == Side.SELL and trade.side != "buy":
            return None

        remaining_aggressive_qty = trade.qty_int
        if order.queue_ahead_int > 0:
            consumed = min(order.queue_ahead_int, remaining_aggressive_qty)
            order.queue_ahead_int -= consumed
            remaining_aggressive_qty -= consumed
        if remaining_aggressive_qty <= 0:
            return None

        fill_qty = min(order.remaining_qty_int, remaining_aggressive_qty)
        if fill_qty <= 0:
            return None

        order.filled_qty_int += fill_qty
        order.status = OrderStatus.FILLED if order.remaining_qty_int == 0 else OrderStatus.PARTIALLY_FILLED
        notional = Decimal(order.price_int) * self.tick_size * Decimal(fill_qty) * self.step_size
        fee_quote = notional * self.maker_fee_bps / Decimal("10000")
        return Fill(
            order_id=order.order_id,
            side=order.side,
            price_int=order.price_int,
            qty_int=fill_qty,
            time_ms=trade.trade_time_ms,
            fee_quote=fee_quote,
        )
