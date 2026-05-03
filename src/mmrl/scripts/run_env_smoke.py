from __future__ import annotations

from pathlib import Path

import orjson
import typer
from rich.console import Console

from mmrl.config import BotConfig
from mmrl.data.order_book import BookSnapshot
from mmrl.env.market_making_env import MarketMakingEnv, ReplayEvent
from mmrl.agents.random_agent import RandomAgent

app = typer.Typer(help="Run a smoke simulation with a random agent.")
console = Console()


def load_replay(path: Path) -> tuple[BookSnapshot, list[ReplayEvent]]:
    snapshot: BookSnapshot | None = None
    events: list[ReplayEvent] = []
    with path.open("rb") as fh:
        for line in fh:
            raw = orjson.loads(line)
            if raw["kind"] == "snapshot":
                p = raw["payload"]
                snapshot = BookSnapshot(
                    last_update_id=int(p["last_update_id"]),
                    bids=[tuple(x) for x in p["bids"]],
                    asks=[tuple(x) for x in p["asks"]],
                )
            elif raw["kind"] in {"depth", "aggTrade"}:
                events.append(ReplayEvent(raw["kind"], raw["payload"]))
    if snapshot is None:
        raise RuntimeError("replay file must begin with a snapshot")
    return snapshot, events


@app.command()
def main(
    config: Path = Path("configs/default.yaml"),
    replay: Path = Path("data/raw/toy_replay.jsonl"),
) -> None:
    cfg = BotConfig.from_yaml(config)
    snapshot, events = load_replay(replay)
    env = MarketMakingEnv(cfg, snapshot, events)
    agent = RandomAgent(env.action_space.n)
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    steps = 0
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
    console.print({"steps": steps, "total_reward": total_reward, "final_info": str(info)})


if __name__ == "__main__":
    app()
