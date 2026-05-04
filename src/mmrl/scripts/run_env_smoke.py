from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from mmrl.agents.random_agent import RandomAgent
from mmrl.config import BotConfig
from mmrl.env.market_making_env import MarketMakingEnv
from mmrl.scripts.replay_io import load_replay

app = typer.Typer(help="Run a smoke simulation with a random agent.")
console = Console()


@app.command()
def main(
    config: Path = Path("configs/default.yaml"),
    replay: Path = Path("data/raw/toy_replay.jsonl"),
) -> None:
    cfg = BotConfig.from_yaml(config)
    snapshot, events, counts = load_replay(replay)

    console.print(
        {
            "replay_counts": counts,
            "events_loaded_for_env": len(events),
        }
    )

    env = MarketMakingEnv(cfg, snapshot, events)
    agent = RandomAgent(env.action_space.n)

    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    steps = 0
    last_info = {}

    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        last_info = info

    console.print(
        {
            "steps": steps,
            "total_reward": total_reward,
            "final_info": last_info,
        }
    )


if __name__ == "__main__":
    app()