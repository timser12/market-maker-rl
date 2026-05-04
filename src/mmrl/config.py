from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator


class BotConfig(BaseModel):
    """Central config for simulation-first market-making research."""

    symbol: str = "BTCUSDT"
    tick_size: Decimal = Decimal("0.01")
    step_size: Decimal = Decimal("0.00001")

    depth_limit: int = Field(default=1000, ge=5, le=5000)
    levels: PositiveInt = 200
    history: PositiveInt = 16
    window_ms: PositiveInt = 1000

    websocket_base: str = "wss://stream.binance.com:9443"
    rest_base: str = "https://api.binance.com"

    maker_fee_bps: Decimal = Decimal("1.0")
    max_inventory: Decimal = Decimal("0.02")
    max_drawdown_quote: Decimal = Decimal("100.0")
    min_spread_ticks: PositiveInt = 1
    max_staleness_ms: PositiveInt = 2500
    max_order_size: Decimal = Decimal("0.002")
    base_order_size: Decimal = Decimal("0.001")
    latency_min_ms: PositiveInt = 50
    latency_max_ms: PositiveInt = 250
    decision_interval_ms: PositiveInt = 1000
    queue_ahead_fraction: Decimal = Decimal("1.0")
    cancel_ahead_fraction: Decimal = Decimal("0.25")
    trade_through_fill_fraction: Decimal = Decimal("1.0")

    @field_validator("symbol")
    @classmethod
    def upper_symbol(cls, value: str) -> str:
        return value.upper()

    @field_validator("tick_size", "step_size", "maker_fee_bps", "max_inventory", "max_drawdown_quote", "max_order_size", "base_order_size")
    @classmethod
    def positive_decimal(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("must be positive")
        return value
    
    @field_validator(
        "queue_ahead_fraction",
        "cancel_ahead_fraction",
        "trade_through_fill_fraction",
    )
    @classmethod
    def fraction_between_zero_and_one(cls, value: Decimal) -> Decimal:
        if value < 0 or value > 1:
            raise ValueError("must be between 0 and 1")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BotConfig":
        with open(path, "rb") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls(**raw)
