from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
from rich.console import Console
from rich.table import Table

from mmrl.agents.networks import DuelingDQN
from mmrl.config import BotConfig
from mmrl.env.market_making_env import MarketMakingEnv
from mmrl.scripts.replay_io import load_replay

app = typer.Typer(help="Evaluate a trained DQN model on an offline replay.")
console = Console()


ACTION_NAMES = {
    0: "do_nothing",
    1: "quote_tight",
    2: "quote_normal",
    3: "quote_wide",
    4: "reduce_long",
    5: "reduce_short",
    6: "cancel_all",
    7: "only_bid",
    8: "only_ask",
}


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def obs_to_tensors(obs: dict[str, np.ndarray], device: torch.device):
    book = torch.from_numpy(obs["order_book"]).unsqueeze(0).float().to(device)
    portfolio = torch.from_numpy(obs["portfolio"]).unsqueeze(0).float().to(device)
    return book, portfolio

def json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, torch.device):
        return str(obj)

    return str(obj)

def reward_to_dict(reward_breakdown: Any) -> dict[str, float]:
    if reward_breakdown is None:
        return {
            "equity_delta": 0.0,
            "equity_delta_scaled": 0.0,
            "inventory_penalty": 0.0,
            "inventory_change_penalty": 0.0,
            "drawdown_penalty": 0.0,
            "risk_override_penalty": 0.0,
            "cancel_penalty": 0.0,
            "current_equity": 0.0,
            "previous_equity": 0.0,
            "inventory_ratio": 0.0,
            "drawdown_scaled": 0.0,
        }

    return {
        "equity_delta": float(getattr(reward_breakdown, "equity_delta", 0.0)),
        "equity_delta_scaled": float(getattr(reward_breakdown, "equity_delta_scaled", 0.0)),
        "inventory_penalty": float(getattr(reward_breakdown, "inventory_penalty", 0.0)),
        "inventory_change_penalty": float(
            getattr(reward_breakdown, "inventory_change_penalty", 0.0)
        ),
        "drawdown_penalty": float(getattr(reward_breakdown, "drawdown_penalty", 0.0)),
        "risk_override_penalty": float(getattr(reward_breakdown, "risk_override_penalty", 0.0)),
        "cancel_penalty": float(getattr(reward_breakdown, "cancel_penalty", 0.0)),
        "current_equity": float(getattr(reward_breakdown, "current_equity", 0.0)),
        "previous_equity": float(getattr(reward_breakdown, "previous_equity", 0.0)),
        "inventory_ratio": float(getattr(reward_breakdown, "inventory_ratio", 0.0)),
        "drawdown_scaled": float(getattr(reward_breakdown, "drawdown_scaled", 0.0)),
    }


def print_summary(summary: dict[str, Any]) -> None:
    table = Table(title="DQN Evaluation Summary")
    table.add_column("metric")
    table.add_column("value")

    keys = [
        "model",
        "replay",
        "device",
        "steps",
        "total_reward",
        "avg_reward",
        "final_inventory",
        "final_cash",
        "final_equity",
        "final_unrealized",
        "max_abs_inventory",
        "total_fills",
        "risk_override_count",
    ]

    for key in keys:
        table.add_row(key, str(summary.get(key)))

    console.print(table)

    action_table = Table(title="Action Counts")
    action_table.add_column("action")
    action_table.add_column("name")
    action_table.add_column("count")

    for action, count in summary["action_counts"].items():
        action_int = int(action)
        action_table.add_row(str(action_int), ACTION_NAMES.get(action_int, "?"), str(count))

    console.print(action_table)

    risk_table = Table(title="Risk Override Counts")
    risk_table.add_column("reason")
    risk_table.add_column("count")

    for reason, count in summary["risk_counts"].items():
        risk_table.add_row(str(reason), str(count))

    console.print(risk_table)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml")),
    replay: Path = typer.Option(Path("data/raw/events.jsonl")),
    model: Path = typer.Option(Path("models/dqn_demo.pt")),
    out: Path = typer.Option(Path("runs/evaluation/eval_summary.json")),
    step_log: Path | None = typer.Option(Path("runs/evaluation/eval_steps.csv")),
    max_steps: int = typer.Option(100_000),
    device_name: str = typer.Option("auto"),
    torch_threads: int = typer.Option(1),
) -> None:
    torch.set_num_threads(max(1, torch_threads))

    cfg = BotConfig.from_yaml(config)
    snapshot, events, replay_counts = load_replay(replay)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    checkpoint = load_checkpoint(model, device)
    n_actions = int(checkpoint.get("n_actions", 9))

    policy = DuelingDQN(book_channels=12, portfolio_dim=4, n_actions=n_actions).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    env = MarketMakingEnv(cfg, snapshot, events)
    obs, _ = env.reset()

    total_reward = 0.0
    action_counts: Counter[int] = Counter()
    risk_counts: Counter[str] = Counter()
    reward_component_sums: Counter[str] = Counter()

    max_abs_inventory = 0.0
    total_fills = 0
    done = False
    steps = 0

    step_writer = None
    step_file = None

    if step_log is not None:
        step_log.parent.mkdir(parents=True, exist_ok=True)
        step_file = step_log.open("w", newline="", encoding="utf-8")
        step_writer = csv.DictWriter(
            step_file,
            fieldnames=[
                "step",
                "action",
                "action_name",
                "reward",
                "q_max",
                "q_min",
                "inventory",
                "cash",
                "equity",
                "unrealized",
                "open_orders",
                "fills",
                "risk_reason",
                "equity_delta",
                "equity_delta_scaled",
                "inventory_penalty",
                "inventory_change_penalty",
                "drawdown_penalty",
                "risk_override_penalty",
                "cancel_penalty",
                "current_equity",
                "previous_equity",
                "inventory_ratio",
                "drawdown_scaled",
            ],
        )
        step_writer.writeheader()

    try:
        while not done and steps < max_steps:
            with torch.no_grad():
                book, portfolio = obs_to_tensors(obs, device)
                q_values = policy(book, portfolio)
                action = int(q_values.argmax(dim=1).item())
                q_max = float(q_values.max().item())
                q_min = float(q_values.min().item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            total_reward += float(reward)
            action_counts[action] += 1

            risk_reason = info.get("risk")
            if risk_reason:
                risk_counts[str(risk_reason)] += 1

            fills = len(env.engine.last_fills)
            total_fills += fills

            portfolio_vec = next_obs["portfolio"]
            inventory = float(portfolio_vec[0])
            max_abs_inventory = max(max_abs_inventory, abs(inventory))

            reward_parts = reward_to_dict(info.get("reward_breakdown"))
            for key, value in reward_parts.items():
                reward_component_sums[key] += float(value)

            if step_writer is not None:
                step_writer.writerow(
                    {
                        "step": steps,
                        "action": action,
                        "action_name": ACTION_NAMES.get(action, str(action)),
                        "reward": float(reward),
                        "q_max": q_max,
                        "q_min": q_min,
                        "inventory": float(portfolio_vec[0]),
                        "cash": float(portfolio_vec[1]),
                        "equity": float(portfolio_vec[2]),
                        "unrealized": float(portfolio_vec[3]),
                        "open_orders": len(env.engine.open_orders),
                        "fills": fills,
                        "risk_reason": risk_reason,
                        **reward_parts,
                    }
                )

            obs = next_obs
            steps += 1

    finally:
        if step_file is not None:
            step_file.close()

    final_portfolio = obs["portfolio"]

    summary = {
        "model": str(model),
        "replay": str(replay),
        "config": str(config),
        "device": str(device),
        "replay_counts": replay_counts,
        "steps": steps,
        "total_reward": total_reward,
        "avg_reward": total_reward / max(1, steps),
        "final_inventory": float(final_portfolio[0]),
        "final_cash": float(final_portfolio[1]),
        "final_equity": float(final_portfolio[2]),
        "final_unrealized": float(final_portfolio[3]),
        "max_abs_inventory": max_abs_inventory,
        "total_fills": total_fills,
        "risk_override_count": sum(risk_counts.values()),
        "action_counts": dict(action_counts),
        "risk_counts": dict(risk_counts),
        "reward_component_sums": dict(reward_component_sums),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "checkpoint_episode": checkpoint.get("episode"),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
    json.dumps(summary, indent=2, default=json_default),
    encoding="utf-8",
)

    print_summary(summary)

    console.print(
        {
            "summary_written": str(out),
            "step_log_written": None if step_log is None else str(step_log),
        }
    )


if __name__ == "__main__":
    app()