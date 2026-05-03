from __future__ import annotations

from pathlib import Path

import orjson
import typer

app = typer.Typer(help="Create a tiny deterministic replay file for smoke tests.")


@app.command()
def main(out: Path = Path("data/raw/toy_replay.jsonl")) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "kind": "snapshot",
        "payload": {
            "last_update_id": 100,
            "bids": [["99.99", "1.00000"], ["99.98", "2.00000"], ["99.97", "3.00000"]],
            "asks": [["100.01", "1.00000"], ["100.02", "2.00000"], ["100.03", "3.00000"]],
        },
    }
    events = [snapshot]
    depth_id = 100
    for i in range(1, 200):
        t = 1_700_000_000_000 + i * 100
        if i % 2:
            depth_id += 1
            events.append(
                {
                    "kind": "depth",
                    "payload": {
                        "e": "depthUpdate",
                        "E": t,
                        "s": "BTCUSDT",
                        "U": depth_id,
                        "u": depth_id,
                        "b": [["99.99", "1.00000"]],
                        "a": [["100.01", "1.00000"]],
                    },
                }
            )
        else:
            events.append(
                {
                    "kind": "aggTrade",
                    "payload": {
                        "e": "aggTrade",
                        "E": t,
                        "s": "BTCUSDT",
                        "a": i,
                        "p": "99.99" if i % 4 == 0 else "100.01",
                        "q": "0.40000",
                        "f": i,
                        "l": i,
                        "T": t,
                        "m": i % 4 == 0,
                        "M": True,
                    },
                }
            )
    with out.open("wb") as fh:
        for event in events:
            fh.write(orjson.dumps(event) + b"\n")
    print(out)


if __name__ == "__main__":
    app()
