from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import orjson
import websockets

from mmrl.config import BotConfig
from mmrl.data.order_book import BookSnapshot


@dataclass(slots=True)
class BinanceMarketDataClient:
    """Minimal public-market-data client.

    This client does not place orders. Good. We are building a research system,
    not a tuition-burning machine.
    """

    cfg: BotConfig

    async def fetch_depth_snapshot(self) -> BookSnapshot:
        params = {"symbol": self.cfg.symbol, "limit": self.cfg.depth_limit}
        async with httpx.AsyncClient(base_url=self.cfg.rest_base, timeout=10.0) as client:
            response = await client.get("/api/v3/depth", params=params)
            response.raise_for_status()
            raw = response.json()
        return BookSnapshot(
            last_update_id=int(raw["lastUpdateId"]),
            bids=[tuple(x) for x in raw["bids"]],
            asks=[tuple(x) for x in raw["asks"]],
        )

    def stream_names(self, kline_interval: str = "1s") -> list[str]:
        symbol = self.cfg.symbol.lower()
        return [
            f"{symbol}@depth@100ms",
            f"{symbol}@aggTrade",
            f"{symbol}@kline_{kline_interval}",
        ]

    async def combined_stream(self, kline_interval: str = "1s") -> AsyncIterator[dict]:
        streams = "/".join(self.stream_names(kline_interval))
        url = f"{self.cfg.websocket_base}/stream?streams={streams}"
        async for websocket in websockets.connect(url, ping_interval=20, ping_timeout=60):
            try:
                async for message in websocket:
                    yield orjson.loads(message)
            except websockets.ConnectionClosed:
                await asyncio.sleep(1.0)
                continue
