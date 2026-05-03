from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from mmrl.data.order_book import decimal_to_int


@dataclass(frozen=True, slots=True)
class AggTrade:
    agg_id: int
    price_int: int
    qty_int: int
    trade_time_ms: int
    side: Literal["buy", "sell"]

    @classmethod
    def from_binance(cls, raw: dict, tick_size: Decimal, step_size: Decimal) -> "AggTrade":
        # Binance: m=true means buyer was maker, therefore seller was taker.
        side = "sell" if raw["m"] else "buy"
        return cls(
            agg_id=int(raw["a"]),
            price_int=decimal_to_int(raw["p"], tick_size),
            qty_int=decimal_to_int(raw["q"], step_size),
            trade_time_ms=int(raw["T"]),
            side=side,
        )
