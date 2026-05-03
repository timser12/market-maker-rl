from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from mmrl.sim.orders import Fill, Side
from mmrl.sim.portfolio import Portfolio


@dataclass(slots=True)
class RewardBreakdown:
    realized_pnl: float = 0.0
    spread_capture: float = 0.0
    inventory_penalty: float = 0.0
    adverse_selection_penalty: float = 0.0
    fee_penalty: float = 0.0
    risk_override_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.realized_pnl
            + self.spread_capture
            - self.inventory_penalty
            - self.adverse_selection_penalty
            - self.fee_penalty
            - self.risk_override_penalty
        )


@dataclass(slots=True)
class RewardModel:
    inventory_penalty_coef: float = 0.02
    adverse_selection_coef: float = 1.0

    def compute(
        self,
        portfolio: Portfolio,
        fills: list[Fill],
        previous_mid: Decimal,
        current_mid: Decimal,
        risk_penalty: float,
    ) -> RewardBreakdown:
        breakdown = RewardBreakdown()
        breakdown.realized_pnl = float(portfolio.realized_pnl)
        breakdown.inventory_penalty = self.inventory_penalty_coef * float(abs(portfolio.inventory))
        breakdown.fee_penalty = float(portfolio.fee_paid_quote)
        breakdown.risk_override_penalty = risk_penalty

        for fill in fills:
            fill_price = Decimal(fill.price_int)  # tick scaling cancels in sign proxy below
            mid_tick_prev = previous_mid
            if fill.side == Side.BUY:
                breakdown.spread_capture += float(mid_tick_prev - fill_price)
                if current_mid < previous_mid:
                    breakdown.adverse_selection_penalty += float(previous_mid - current_mid)
            else:
                breakdown.spread_capture += float(fill_price - mid_tick_prev)
                if current_mid > previous_mid:
                    breakdown.adverse_selection_penalty += float(current_mid - previous_mid)
        breakdown.adverse_selection_penalty *= self.adverse_selection_coef
        return breakdown
