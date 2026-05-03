from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import orjson
import typer
from rich.console import Console

from mmrl.config import BotConfig
from mmrl.data.binance_client import BinanceMarketDataClient
from mmrl.data.order_book import BookDesyncError, BookSnapshot, LocalOrderBook
from mmrl.data.trades import AggTrade

app = typer.Typer(help="Collect Binance public market data for offline simulation.")
console = Console()


def snapshot_to_payload(snapshot: BookSnapshot) -> dict[str, Any]:
    """
    Stable JSON schema for replay snapshots.

    BookSnapshot uses slots=True, so snapshot.__dict__ does not exist.
    Do not serialize production-ish data objects by hoping __dict__ exists.
    """
    if is_dataclass(snapshot):
        payload = asdict(snapshot)
    else:
        payload = {
            "last_update_id": snapshot.last_update_id,
            "bids": snapshot.bids,
            "asks": snapshot.asks,
        }

    return {
        "last_update_id": int(payload["last_update_id"]),
        "bids": payload["bids"],
        "asks": payload["asks"],
    }


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), help="Path to YAML config."),
    out: Path = typer.Option(Path("data/raw/events.jsonl"), help="Output JSONL replay path."),
    max_events: int = typer.Option(100_000, help="Number of post-snapshot events to write."),
    max_seconds: int | None = typer.Option(None, help="Optional wall-clock collection limit."),
    append: bool = typer.Option(False, help="Append to output instead of replacing it."),
    kline_interval: str = typer.Option("1s", help="Binance kline interval."),
) -> None:
    asyncio.run(_run(config, out, max_events, max_seconds, append, kline_interval))


async def _run(
    config_path: Path,
    out: Path,
    max_events: int,
    max_seconds: int | None,
    append: bool,
    kline_interval: str,
) -> None:
    cfg = BotConfig.from_yaml(config_path)
    client = BinanceMarketDataClient(cfg)

    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if append else "wb"

    console.print(f"[bold]Collecting {cfg.symbol}[/bold] to {out}")
    console.print("Waiting for first depth event, then fetching REST snapshot...")

    book: LocalOrderBook | None = None
    n = 0
    depth_applied = 0
    depth_ignored = 0
    agg_trades = 0
    klines = 0
    start_monotonic = time.monotonic()

    with out.open(mode) as fh:
        async for wrapped in client.combined_stream(kline_interval=kline_interval):
            stream = wrapped["stream"]
            data = wrapped["data"]

            is_depth = stream.endswith("@depth@100ms") or stream.endswith("@depth")
            is_trade = stream.endswith("@aggTrade")
            is_kline = "@kline_" in stream

            if book is None:
                if not is_depth:
                    continue

                first_depth_u = int(data["U"])
                snapshot = await _fetch_snapshot_that_is_not_too_old(client, first_depth_u)
                book = LocalOrderBook.from_snapshot(snapshot, cfg.tick_size, cfg.step_size)

                fh.write(
                    orjson.dumps(
                        {
                            "kind": "snapshot",
                            "payload": snapshot_to_payload(snapshot),
                        }
                    )
                    + b"\n"
                )

                console.print(
                    f"Initialized snapshot last_update_id={snapshot.last_update_id}; "
                    f"first_stream_U={first_depth_u}"
                )

            if is_depth:
                try:
                    status = book.apply_depth_event(data)
                    if status == "applied":
                        depth_applied += 1
                    else:
                        depth_ignored += 1
                except BookDesyncError as exc:
                    console.print(f"[red]Depth stream desync detected: {exc}[/red]")
                    console.print("Stopped collection. Keep this replay or collect a fresh one.")
                    break
                kind = "depth"

            elif is_trade:
                AggTrade.from_binance(data, cfg.tick_size, cfg.step_size)
                agg_trades += 1
                kind = "aggTrade"

            elif is_kline:
                klines += 1
                kind = "kline"

            else:
                continue

            fh.write(orjson.dumps({"kind": kind, "payload": data}) + b"\n")
            n += 1

            if n % 1000 == 0:
                fh.flush()
                console.print(
                    f"collected={n:,} "
                    f"depth_applied={depth_applied:,} "
                    f"depth_ignored={depth_ignored:,} "
                    f"aggTrade={agg_trades:,} "
                    f"kline={klines:,}"
                )

            if n >= max_events:
                break

            if max_seconds is not None and time.monotonic() - start_monotonic >= max_seconds:
                break

    console.print(
        {
            "out": str(out),
            "events_written": n,
            "depth_applied": depth_applied,
            "depth_ignored": depth_ignored,
            "aggTrade": agg_trades,
            "kline": klines,
        }
    )


async def _fetch_snapshot_that_is_not_too_old(
    client: BinanceMarketDataClient,
    first_depth_u: int,
    max_attempts: int = 10,
) -> BookSnapshot:
    for attempt in range(1, max_attempts + 1):
        snapshot = await client.fetch_depth_snapshot()
        if snapshot.last_update_id >= first_depth_u:
            return snapshot

        console.print(
            f"snapshot last_update_id={snapshot.last_update_id} is older than "
            f"first_stream_U={first_depth_u}; retrying ({attempt}/{max_attempts})"
        )
        await asyncio.sleep(0.25)

    raise RuntimeError("could not fetch a snapshot new enough to initialize replay")


if __name__ == "__main__":
    app()