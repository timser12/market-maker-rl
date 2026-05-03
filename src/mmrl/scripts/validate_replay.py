from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from mmrl.config import BotConfig
from mmrl.data.order_book import BookDesyncError, LocalOrderBook
from mmrl.data.trades import AggTrade
from mmrl.scripts.replay_io import load_replay

app = typer.Typer(help="Validate that a JSONL replay can drive the local book.")
console = Console()


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml")),
    replay: Path = typer.Option(Path("data/raw/events.jsonl")),
) -> None:
    cfg = BotConfig.from_yaml(config)
    snapshot, events, counts = load_replay(replay)
    book = LocalOrderBook.from_snapshot(snapshot, cfg.tick_size, cfg.step_size)

    applied = 0
    ignored = 0
    trades = 0
    first_ts: int | None = None
    last_ts: int | None = None

    for idx, event in enumerate(events, start=1):
        try:
            if event.kind == "depth":
                status = book.apply_depth_event(event.payload)
                if status == "applied":
                    applied += 1
                else:
                    ignored += 1
                ts = int(event.payload.get("E", 0))

            elif event.kind == "aggTrade":
                trade = AggTrade.from_binance(event.payload, cfg.tick_size, cfg.step_size)
                trades += 1
                ts = trade.trade_time_ms

            else:
                continue

        except BookDesyncError as exc:
            raise typer.BadParameter(f"desync at replay event #{idx}: {exc}") from exc

        if ts:
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)

    duration_s = None if first_ts is None or last_ts is None else (last_ts - first_ts) / 1000.0
    best_bid = book.best_bid()
    best_ask = book.best_ask()

    console.print(
        {
            "file": str(replay),
            "raw_counts": counts,
            "events_loaded_for_env": len(events),
            "depth_applied": applied,
            "depth_ignored_as_stale": ignored,
            "agg_trades": trades,
            "duration_seconds": duration_s,
            "final_last_update_id": book.last_update_id,
            "final_spread_ticks": None if best_bid is None or best_ask is None else best_ask - best_bid,
            "status": "OK",
        }
    )


if __name__ == "__main__":
    app()