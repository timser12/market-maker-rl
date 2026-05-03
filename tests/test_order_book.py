from decimal import Decimal

import pytest

from mmrl.data.order_book import BookDesyncError, BookSnapshot, LocalOrderBook, decimal_to_int


def make_book():
    return LocalOrderBook.from_snapshot(
        BookSnapshot(
            last_update_id=100,
            bids=[("99.99", "1.00000")],
            asks=[("100.01", "2.00000")],
        ),
        Decimal("0.01"),
        Decimal("0.00001"),
    )


def test_decimal_to_int_rejects_off_grid_values():
    with pytest.raises(ValueError):
        decimal_to_int("100.001", Decimal("0.01"))


def test_depth_event_applies_set_and_remove():
    book = make_book()
    result = book.apply_depth_event(
        {"U": 101, "u": 101, "b": [["99.98", "3.00000"]], "a": [["100.01", "0.00000"]]}
    )
    assert result == "applied"
    assert book.last_update_id == 101
    assert book.bids[9998] == 300000
    assert 10001 not in book.asks


def test_old_depth_event_is_ignored():
    book = make_book()
    assert book.apply_depth_event({"U": 90, "u": 100, "b": [], "a": []}) == "ignored"


def test_depth_gap_raises_desync():
    book = make_book()
    with pytest.raises(BookDesyncError):
        book.apply_depth_event({"U": 102, "u": 102, "b": [], "a": []})
