from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from mmrl.data.order_book import int_to_decimal
from mmrl.sim.orders import Fill, Side


@dataclass(slots=True)
class Portfolio:
    cash: Decimal = Decimal("0")
    inventory: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    avg_entry_price: Decimal | None = None
    fee_paid_quote: Decimal = Decimal("0")

    def apply_fill(self, fill: Fill, tick_size: Decimal, step_size: Decimal) -> None:
        price = int_to_decimal(fill.price_int, tick_size)
        qty = int_to_decimal(fill.qty_int, step_size)
        notional = price * qty
        self.fee_paid_quote += fill.fee_quote
        if fill.side == Side.BUY:
            old_inv = self.inventory
            self.cash -= notional + fill.fee_quote
            self.inventory += qty
            if self.inventory != 0:
                if self.avg_entry_price is None or old_inv <= 0:
                    self.avg_entry_price = price
                else:
                    self.avg_entry_price = ((self.avg_entry_price * old_inv) + notional) / self.inventory
        else:
            self.cash += notional - fill.fee_quote
            if self.avg_entry_price is not None and self.inventory > 0:
                closed_qty = min(qty, self.inventory)
                self.realized_pnl += (price - self.avg_entry_price) * closed_qty - fill.fee_quote
            self.inventory -= qty
            if self.inventory == 0:
                self.avg_entry_price = None
            elif self.inventory < 0:
                self.avg_entry_price = price

    def equity(self, mid_price: Decimal) -> Decimal:
        return self.cash + self.inventory * mid_price

    def unrealized_pnl(self, mid_price: Decimal) -> Decimal:
        if self.avg_entry_price is None:
            return Decimal("0")
        return (mid_price - self.avg_entry_price) * self.inventory
