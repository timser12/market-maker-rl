from decimal import Decimal

from mmrl.data.order_book import BookSnapshot, LocalOrderBook, decimal_to_int
from mmrl.data.trades import AggTrade
from mmrl.sim.fill_model import ConservativeQueueFillModel
from mmrl.sim.orders import Side


def test_queue_must_be_consumed_before_fill():
    tick = Decimal("0.01")
    step = Decimal("0.00001")
    book = LocalOrderBook.from_snapshot(
        BookSnapshot(1, bids=[("99.99", "1.00000")], asks=[("100.01", "1.00000")]),
        tick,
        step,
    )
    model = ConservativeQueueFillModel(tick, step, Decimal("1.0"), latency_min_ms=1, latency_max_ms=1)
    order = model.new_order(Side.BUY, decimal_to_int("99.99", tick), decimal_to_int("0.50000", step), 0, book)

    # First aggressive sell consumes only the visible queue ahead.
    t1 = AggTrade(1, decimal_to_int("99.99", tick), decimal_to_int("0.40000", step), 10, "sell")
    assert model.process_trade(order, t1) is None
    assert order.queue_ahead_int == decimal_to_int("0.60000", step)

    # Second aggressive sell consumes the rest of queue and partially fills us.
    t2 = AggTrade(2, decimal_to_int("99.99", tick), decimal_to_int("1.00000", step), 20, "sell")
    fill = model.process_trade(order, t2)
    assert fill is not None
    assert fill.qty_int == decimal_to_int("0.40000", step)


def test_trade_through_fills_bid():
    tick = Decimal("0.01")
    step = Decimal("0.00001")

    book = LocalOrderBook.from_snapshot(
        BookSnapshot(
            1,
            bids=[("100.00", "1.00000")],
            asks=[("100.02", "1.00000")],
        ),
        tick,
        step,
    )

    model = ConservativeQueueFillModel(
        tick,
        step,
        Decimal("1.0"),
        latency_min_ms=1,
        latency_max_ms=1,
        trade_through_fill_fraction=Decimal("1.0"),
    )

    order = model.new_order(
        Side.BUY,
        decimal_to_int("100.00", tick),
        decimal_to_int("0.50000", step),
        0,
        book,
    )

    trade = AggTrade(
        1,
        decimal_to_int("99.99", tick),
        decimal_to_int("0.10000", step),
        10,
        "sell",
    )

    fill = model.process_trade(order, trade)

    assert fill is not None
    assert fill.qty_int == decimal_to_int("0.50000", step)


def test_depth_reduction_reduces_queue_ahead():
    tick = Decimal("0.01")
    step = Decimal("0.00001")

    book = LocalOrderBook.from_snapshot(
        BookSnapshot(
            1,
            bids=[("100.00", "1.00000")],
            asks=[("100.02", "1.00000")],
        ),
        tick,
        step,
    )

    model = ConservativeQueueFillModel(
        tick,
        step,
        Decimal("1.0"),
        latency_min_ms=1,
        latency_max_ms=1,
        cancel_ahead_fraction=Decimal("0.50"),
    )

    order = model.new_order(
        Side.BUY,
        decimal_to_int("100.00", tick),
        decimal_to_int("0.50000", step),
        0,
        book,
    )

    assert order.queue_ahead_int == decimal_to_int("1.00000", step)

    depth_event = {
        "U": 2,
        "u": 2,
        "E": 10,
        "b": [["100.00", "0.40000"]],
        "a": [],
    }

    model.process_depth_update(order, depth_event, book, event_time_ms=10)

    # Visible size dropped by 0.6 BTC. With cancel_ahead_fraction=0.5,
    # queue ahead should drop by 0.3 BTC.
    assert order.queue_ahead_int == decimal_to_int("0.70000", step)