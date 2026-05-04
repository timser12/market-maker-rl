from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import random

from mmrl.data.order_book import LocalOrderBook, decimal_to_int
from mmrl.data.trades import AggTrade
from mmrl.sim.orders import Fill, LimitOrder, OrderStatus, Side


@dataclass(slots=True)
class ConservativeQueueFillModel:
    """Queue-aware fill model for paper trading.

    Improvements over naive touch-fill simulation:

    1. Queue position:
       The bot does not instantly fill when price touches its order.
       It must wait for visible queue ahead to be consumed.

    2. Trade-through handling:
       If aggressive flow trades through the order price, the order can fill.

    3. Depth-cancellation proxy:
       If visible size at our price decreases in the depth stream, a fraction
       of that decrease is assumed to be cancellations/removals ahead of us.

    4. Latency:
       New orders only become live after a random acceptance delay.

    Still not perfect. But much less childish than:
        "last price touched my quote, therefore I got filled."
    """

    tick_size: Decimal
    step_size: Decimal
    maker_fee_bps: Decimal

    latency_min_ms: int = 50
    latency_max_ms: int = 250

    # 1.0 means assume the full visible quantity at our price is ahead of us.
    # 0.5 means assume we are halfway through the visible queue.
    queue_ahead_fraction: Decimal = Decimal("1.0")

    # Fraction of visible depth reductions that reduce our queueAhead.
    # 0.0 = ignore cancellations.
    # 1.0 = every visible reduction helps our queue position.
    # 0.25 is conservative.
    cancel_ahead_fraction: Decimal = Decimal("0.25")

    # If price trades through our level, fill this fraction of remaining qty.
    # 1.0 = assume fully fill on trade-through.
    # 0.5 = more conservative.
    trade_through_fill_fraction: Decimal = Decimal("1.0")

    rng: random.Random = field(default_factory=lambda: random.Random(7))

    def queue_ahead_for(self, book: LocalOrderBook, side: Side, price_int: int) -> int:
        book_side = book.bids if side == Side.BUY else book.asks
        visible_qty = book_side.get(price_int, 0)

        queue_ahead = int(Decimal(visible_qty) * self.queue_ahead_fraction)
        return max(0, queue_ahead)

    def new_order(
        self,
        side: Side,
        price_int: int,
        qty_int: int,
        now_ms: int,
        book: LocalOrderBook,
    ) -> LimitOrder:
        delay = self.rng.randint(self.latency_min_ms, self.latency_max_ms)

        return LimitOrder(
            side=side,
            price_int=price_int,
            qty_int=qty_int,
            created_time_ms=now_ms,
            queue_ahead_int=self.queue_ahead_for(book, side, price_int),
            latency_until_ms=now_ms + delay,
        )

    def process_depth_update(
        self,
        order: LimitOrder,
        event: dict,
        book_before_update: LocalOrderBook,
        event_time_ms: int,
    ) -> None:
        """Update queue position from depth changes before applying the update.

        Binance depth updates give the new absolute quantity at a price level.
        If visible quantity at our price decreases, we treat part of that
        decrease as cancellations/removals ahead of us.

        This is not perfect because we cannot know whether the removed quantity
        was in front of us or behind us. So we use cancel_ahead_fraction.
        """

        order.mark_live_if_due(event_time_ms)

        if order.status not in {OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED}:
            return

        updates = event.get("b", []) if order.side == Side.BUY else event.get("a", [])
        book_side = book_before_update.bids if order.side == Side.BUY else book_before_update.asks

        old_visible_qty = book_side.get(order.price_int, 0)

        new_visible_qty: int | None = None

        for price, qty in updates:
            price_int = decimal_to_int(price, self.tick_size)
            if price_int == order.price_int:
                new_visible_qty = decimal_to_int(qty, self.step_size)
                break

        if new_visible_qty is None:
            return

        visible_reduction = max(0, old_visible_qty - new_visible_qty)

        if visible_reduction <= 0:
            return

        queue_reduction = int(Decimal(visible_reduction) * self.cancel_ahead_fraction)

        if queue_reduction > 0:
            order.queue_ahead_int = max(0, order.queue_ahead_int - queue_reduction)

    def process_trade(self, order: LimitOrder, trade: AggTrade) -> Fill | None:
        order.mark_live_if_due(trade.trade_time_ms)

        if order.status in {OrderStatus.NEW, OrderStatus.CANCELLED, OrderStatus.FILLED}:
            return None

        if order.remaining_qty_int <= 0:
            return None

        if not self._trade_can_hit_order(order, trade):
            return None

        exact_price_hit = trade.price_int == order.price_int
        trade_through = self._trade_through_order(order, trade)

        if exact_price_hit:
            fill_qty = self._fill_from_exact_price_trade(order, trade)

        elif trade_through:
            fill_qty = self._fill_from_trade_through(order, trade)

        else:
            fill_qty = 0

        if fill_qty <= 0:
            return None

        order.filled_qty_int += fill_qty

        if order.remaining_qty_int == 0:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        notional = (
            Decimal(order.price_int)
            * self.tick_size
            * Decimal(fill_qty)
            * self.step_size
        )
        fee_quote = notional * self.maker_fee_bps / Decimal("10000")

        return Fill(
            order_id=order.order_id,
            side=order.side,
            price_int=order.price_int,
            qty_int=fill_qty,
            time_ms=trade.trade_time_ms,
            fee_quote=fee_quote,
        )

    def _fill_from_exact_price_trade(self, order: LimitOrder, trade: AggTrade) -> int:
        aggressive_qty = trade.qty_int

        if order.queue_ahead_int > 0:
            consumed = min(order.queue_ahead_int, aggressive_qty)
            order.queue_ahead_int -= consumed
            aggressive_qty -= consumed

        if aggressive_qty <= 0:
            return 0

        return min(order.remaining_qty_int, aggressive_qty)

    def _fill_from_trade_through(self, order: LimitOrder, trade: AggTrade) -> int:
        """Fill when the market trades through our price.

        Example:
            Our bid = 100
            Aggressive sell trade prints at 99.99

        If trades happen below our bid, our bid level should have been consumed
        first. We therefore fill a configurable fraction of remaining size.
        """

        order.queue_ahead_int = 0

        fill_qty = int(Decimal(order.remaining_qty_int) * self.trade_through_fill_fraction)

        if fill_qty <= 0 and order.remaining_qty_int > 0:
            fill_qty = 1

        return min(order.remaining_qty_int, fill_qty)

    @staticmethod
    def _trade_can_hit_order(order: LimitOrder, trade: AggTrade) -> bool:
        # Our bid gets hit by aggressive sells.
        if order.side == Side.BUY:
            return trade.side == "sell" and trade.price_int <= order.price_int

        # Our ask gets hit by aggressive buys.
        if order.side == Side.SELL:
            return trade.side == "buy" and trade.price_int >= order.price_int

        return False

    @staticmethod
    def _trade_through_order(order: LimitOrder, trade: AggTrade) -> bool:
        if order.side == Side.BUY:
            return trade.side == "sell" and trade.price_int < order.price_int

        if order.side == Side.SELL:
            return trade.side == "buy" and trade.price_int > order.price_int

        return False