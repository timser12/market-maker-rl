from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from mmrl.sim.portfolio import Portfolio


@dataclass(slots=True)
class RewardBreakdown:
    equity_delta: float = 0.0
    equity_delta_scaled: float = 0.0

    inventory_penalty: float = 0.0
    inventory_change_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    risk_override_penalty: float = 0.0
    cancel_penalty: float = 0.0

    current_equity: float = 0.0
    previous_equity: float = 0.0
    inventory_ratio: float = 0.0
    drawdown_scaled: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.equity_delta_scaled
            - self.inventory_penalty
            - self.inventory_change_penalty
            - self.drawdown_penalty
            - self.risk_override_penalty
            - self.cancel_penalty
        )


@dataclass(slots=True)
class RewardModel:
    """
    Equity-first reward.

    The main reward is incremental mark-to-market equity change.
    Penalties then discourage the agent from getting that equity by simply
    warehousing inventory like a degenerate directional trader.
    """

    max_inventory: Decimal

    inventory_penalty_coef: float = 0.01
    inventory_change_penalty_coef: float = 0.002
    drawdown_penalty_coef: float = 0.02
    risk_override_penalty_coef: float = 1.0
    agent_cancel_penalty: float = 0.001

    equity_high_water: Decimal | None = None

    def reset(self, starting_equity: Decimal) -> None:
        self.equity_high_water = starting_equity

    def compute(
        self,
        portfolio: Portfolio,
        previous_inventory: Decimal,
        previous_equity: Decimal,
        current_equity: Decimal,
        current_mid_price: Decimal,
        risk_penalty: float,
        agent_cancelled: bool,
    ) -> RewardBreakdown:
        if self.equity_high_water is None:
            self.equity_high_water = previous_equity

        self.equity_high_water = max(self.equity_high_water, current_equity)

        equity_delta = current_equity - previous_equity

        # Normalize reward by approximate max inventory notional.
        # Example: max_inventory=0.02 BTC and BTC=80k -> norm ≈ 1600 quote currency.
        equity_norm = abs(self.max_inventory * current_mid_price)
        if equity_norm <= Decimal("0"):
            equity_norm = Decimal("1")

        equity_delta_scaled = float(equity_delta / equity_norm)

        inventory_ratio_dec = abs(portfolio.inventory) / self.max_inventory
        inventory_ratio = float(inventory_ratio_dec)

        previous_inventory_ratio_dec = abs(previous_inventory) / self.max_inventory
        inventory_change_ratio = float(abs(inventory_ratio_dec - previous_inventory_ratio_dec))

        drawdown = self.equity_high_water - current_equity
        drawdown_scaled = float(drawdown / equity_norm) if drawdown > 0 else 0.0

        breakdown = RewardBreakdown(
            equity_delta=float(equity_delta),
            equity_delta_scaled=equity_delta_scaled,
            inventory_penalty=self.inventory_penalty_coef * inventory_ratio**2,
            inventory_change_penalty=self.inventory_change_penalty_coef * inventory_change_ratio,
            drawdown_penalty=self.drawdown_penalty_coef * drawdown_scaled**2,
            risk_override_penalty=self.risk_override_penalty_coef * float(risk_penalty),
            cancel_penalty=self.agent_cancel_penalty if agent_cancelled else 0.0,
            current_equity=float(current_equity),
            previous_equity=float(previous_equity),
            inventory_ratio=inventory_ratio,
            drawdown_scaled=drawdown_scaled,
        )

        return breakdown