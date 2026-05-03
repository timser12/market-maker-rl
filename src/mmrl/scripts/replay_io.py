from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from mmrl.data.order_book import BookSnapshot
from mmrl.env.market_making_env import ReplayEvent


def parse_snapshot_payload(payload: dict[str, Any]) -> BookSnapshot:
    last_update_id = payload.get("last_update_id", payload.get("lastUpdateId"))
    if last_update_id is None:
        raise KeyError("snapshot payload needs last_update_id or lastUpdateId")

    return BookSnapshot(
        last_update_id=int(last_update_id),
        bids=[tuple(x) for x in payload["bids"]],
        asks=[tuple(x) for x in payload["asks"]],
    )


def load_replay(path: Path) -> tuple[BookSnapshot, list[ReplayEvent], dict[str, int]]:
    snapshot: BookSnapshot | None = None
    events: list[ReplayEvent] = []

    counts = {
        "snapshot": 0,
        "depth": 0,
        "aggTrade": 0,
        "kline": 0,
        "ignored": 0,
    }

    with path.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue

            raw = orjson.loads(line)
            kind = raw["kind"]
            payload = raw["payload"]

            if kind == "snapshot":
                if snapshot is None:
                    snapshot = parse_snapshot_payload(payload)
                    counts["snapshot"] += 1
                else:
                    counts["ignored"] += 1

            elif kind in {"depth", "aggTrade"}:
                events.append(ReplayEvent(kind, payload))
                counts[kind] += 1

            elif kind == "kline":
                counts["kline"] += 1

            else:
                counts["ignored"] += 1

    if snapshot is None:
        raise RuntimeError("replay file must contain a snapshot before market events")

    return snapshot, events, counts