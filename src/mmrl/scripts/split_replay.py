from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import typer
from rich.console import Console

from mmrl.config import BotConfig
from mmrl.data.order_book import BookDesyncError, BookSnapshot, LocalOrderBook, int_to_decimal
from mmrl.scripts.replay_io import parse_snapshot_payload

app = typer.Typer(help="Chronologically split one replay into train/eval replays.")
console = Console()


def snapshot_payload_from_book(book: LocalOrderBook, cfg: BotConfig) -> dict[str, Any]:
    bids = [
        [
            str(int_to_decimal(price_int, cfg.tick_size)),
            str(int_to_decimal(qty_int, cfg.step_size)),
        ]
        for price_int, qty_int in sorted(book.bids.items(), reverse=True)
        if qty_int > 0
    ]

    asks = [
        [
            str(int_to_decimal(price_int, cfg.tick_size)),
            str(int_to_decimal(qty_int, cfg.step_size)),
        ]
        for price_int, qty_int in sorted(book.asks.items())
        if qty_int > 0
    ]

    return {
        "last_update_id": int(book.last_update_id),
        "bids": bids,
        "asks": asks,
    }


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml")),
    in_replay: Path = typer.Option(Path("data/raw/events.jsonl")),
    train_out: Path = typer.Option(Path("data/raw/train.jsonl")),
    eval_out: Path = typer.Option(Path("data/raw/eval.jsonl")),
    train_frac: float = typer.Option(0.8),
) -> None:
    if not 0.1 < train_frac < 0.95:
        raise typer.BadParameter("train_frac should be between 0.1 and 0.95")

    cfg = BotConfig.from_yaml(config)

    snapshot: BookSnapshot | None = None
    usable_records: list[dict[str, Any]] = []

    with in_replay.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue

            record = orjson.loads(line)
            kind = record["kind"]

            if kind == "snapshot" and snapshot is None:
                snapshot = parse_snapshot_payload(record["payload"])

            elif kind in {"depth", "aggTrade"}:
                usable_records.append(record)

    if snapshot is None:
        raise RuntimeError("input replay has no snapshot")

    split_idx = int(len(usable_records) * train_frac)
    train_records = usable_records[:split_idx]
    eval_records = usable_records[split_idx:]

    if not train_records or not eval_records:
        raise RuntimeError("split produced empty train or eval set")

    book = LocalOrderBook.from_snapshot(snapshot, cfg.tick_size, cfg.step_size)

    applied_depth = 0
    ignored_depth = 0

    for record in train_records:
        if record["kind"] != "depth":
            continue

        try:
            status = book.apply_depth_event(record["payload"])
            if status == "applied":
                applied_depth += 1
            else:
                ignored_depth += 1

        except BookDesyncError as exc:
            raise RuntimeError(f"desync while creating eval snapshot: {exc}") from exc

    train_out.parent.mkdir(parents=True, exist_ok=True)
    eval_out.parent.mkdir(parents=True, exist_ok=True)

    with train_out.open("wb") as fh:
        fh.write(
            orjson.dumps(
                {
                    "kind": "snapshot",
                    "payload": {
                        "last_update_id": snapshot.last_update_id,
                        "bids": snapshot.bids,
                        "asks": snapshot.asks,
                    },
                }
            )
            + b"\n"
        )

        for record in train_records:
            fh.write(orjson.dumps(record) + b"\n")

    with eval_out.open("wb") as fh:
        fh.write(
            orjson.dumps(
                {
                    "kind": "snapshot",
                    "payload": snapshot_payload_from_book(book, cfg),
                }
            )
            + b"\n"
        )

        for record in eval_records:
            fh.write(orjson.dumps(record) + b"\n")

    console.print(
        {
            "input": str(in_replay),
            "train_out": str(train_out),
            "eval_out": str(eval_out),
            "total_usable_records": len(usable_records),
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "train_frac": train_frac,
            "eval_snapshot_last_update_id": book.last_update_id,
            "depth_applied_to_make_eval_snapshot": applied_depth,
            "depth_ignored_to_make_eval_snapshot": ignored_depth,
        }
    )


if __name__ == "__main__":
    app()