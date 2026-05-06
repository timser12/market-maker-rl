from __future__ import annotations

import csv
import json
import random
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
from rich.console import Console
from rich.table import Table
from torch import nn

from mmrl.agents.networks import DuelingDQN
from mmrl.config import BotConfig
from mmrl.env.market_making_env import MarketMakingEnv
from mmrl.scripts.replay_io import load_replay
from mmrl.representation.market_encoder import FrozenEncoderDQN, load_pretrained_encoder

app = typer.Typer(help="Train a logged discrete DQN prototype on an offline replay.")
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


@dataclass(slots=True)
class Transition:
    book: np.ndarray
    portfolio: np.ndarray
    action: int
    reward: float
    next_book: np.ndarray
    next_portfolio: np.ndarray
    done: bool


def clean_array(array: np.ndarray, dtype: np.dtype | type = np.float32) -> np.ndarray:
    return np.nan_to_num(array, nan=0.0, posinf=10.0, neginf=-10.0).astype(dtype, copy=False)


def clean_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "order_book": clean_array(obs["order_book"], np.float32),
        "portfolio": clean_array(obs["portfolio"], np.float32),
    }


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.data)

    def push(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        obs = clean_obs(obs)
        next_obs = clean_obs(next_obs)

        self.data.append(
            Transition(
                book=obs["order_book"].astype(np.float16, copy=True),
                portfolio=obs["portfolio"].astype(np.float32, copy=True),
                action=int(action),
                reward=float(reward),
                next_book=next_obs["order_book"].astype(np.float16, copy=True),
                next_portfolio=next_obs["portfolio"].astype(np.float32, copy=True),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.data, batch_size)


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames

        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        clean = {key: row.get(key) for key in self.fieldnames}
        self.writer.writerow(clean)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def obs_to_tensors(obs: dict[str, np.ndarray], device: torch.device):
    obs = clean_obs(obs)
    book = torch.from_numpy(np.ascontiguousarray(obs["order_book"])).unsqueeze(0).float().to(device)
    portfolio = torch.from_numpy(np.ascontiguousarray(obs["portfolio"])).unsqueeze(0).float().to(device)
    return book, portfolio


def batch_to_tensors(batch: list[Transition], device: torch.device):
    books = torch.from_numpy(np.stack([x.book for x in batch]).astype(np.float32)).to(device)
    portfolios = torch.from_numpy(np.stack([x.portfolio for x in batch]).astype(np.float32)).to(device)
    actions = torch.tensor([x.action for x in batch], dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=device).unsqueeze(1)
    next_books = torch.from_numpy(np.stack([x.next_book for x in batch]).astype(np.float32)).to(device)
    next_portfolios = torch.from_numpy(np.stack([x.next_portfolio for x in batch]).astype(np.float32)).to(device)
    dones = torch.tensor([x.done for x in batch], dtype=torch.float32, device=device).unsqueeze(1)
    return books, portfolios, actions, rewards, next_books, next_portfolios, dones


def choose_action(
    policy: DuelingDQN,
    obs: dict[str, np.ndarray],
    epsilon: float,
    n_actions: int,
    device: torch.device,
) -> tuple[int, float, float]:
    """
    Always compute Q-values for logging, even when epsilon chooses a random action.
    No more ugly q_max=nan unless the network is genuinely broken.
    """
    with torch.no_grad():
        book, portfolio = obs_to_tensors(obs, device)
        q_values = policy(book, portfolio)

        if not torch.isfinite(q_values).all():
            raise RuntimeError(f"Non-finite Q-values detected: {q_values}")

        greedy_action = int(q_values.argmax(dim=1).item())
        q_max = float(q_values.max().item())
        q_min = float(q_values.min().item())

    if random.random() < epsilon:
        return random.randrange(n_actions), q_max, q_min

    return greedy_action, q_max, q_min


def optimize(
    policy: DuelingDQN,
    target: DuelingDQN,
    memory: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> dict[str, float] | None:
    if len(memory) < batch_size:
        return None

    batch = memory.sample(batch_size)
    books, portfolios, actions, rewards, next_books, next_portfolios, dones = batch_to_tensors(
        batch, device
    )

    q_all = policy(books, portfolios)
    if not torch.isfinite(q_all).all():
        raise RuntimeError("Non-finite policy Q-values during optimization")

    q = q_all.gather(1, actions)

    with torch.no_grad():
        next_policy_q = policy(next_books, next_portfolios)
        next_target_q_all = target(next_books, next_portfolios)

        if not torch.isfinite(next_policy_q).all() or not torch.isfinite(next_target_q_all).all():
            raise RuntimeError("Non-finite next Q-values during optimization")

        next_actions = next_policy_q.argmax(dim=1, keepdim=True)
        next_q = next_target_q_all.gather(1, next_actions)
        target_q = rewards + gamma * (1.0 - dones) * next_q

    loss = nn.functional.smooth_l1_loss(q, target_q)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(nn.utils.clip_grad_norm_(policy.parameters(), 5.0).item())
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "q_mean": float(q.mean().item()),
        "target_q_mean": float(target_q.mean().item()),
        "grad_norm": grad_norm,
    }


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
    """
    Supports both your newer reward object names and the older scaffold names.
    This avoids CSV/logging crashes while you keep iterating on reward.py.
    """
    defaults = {
        "reward_equity_delta": 0.0,
        "reward_equity_delta_scaled": 0.0,
        "reward_equity_change": 0.0,
        "reward_spread_capture": 0.0,
        "reward_inventory_penalty": 0.0,
        "reward_inventory_change_penalty": 0.0,
        "reward_adverse_selection_penalty": 0.0,
        "reward_drawdown_penalty": 0.0,
        "reward_fee_penalty": 0.0,
        "reward_risk_override_penalty": 0.0,
        "reward_cancel_penalty": 0.0,
        "reward_current_equity": 0.0,
        "reward_previous_equity": 0.0,
        "reward_inventory_ratio": 0.0,
        "reward_drawdown_scaled": 0.0,
    }

    if reward_breakdown is None:
        return defaults

    return {
        "reward_equity_delta": float(getattr(reward_breakdown, "equity_delta", 0.0)),
        "reward_equity_delta_scaled": float(getattr(reward_breakdown, "equity_delta_scaled", 0.0)),
        "reward_equity_change": float(getattr(reward_breakdown, "equity_change", 0.0)),
        "reward_spread_capture": float(getattr(reward_breakdown, "spread_capture", 0.0)),
        "reward_inventory_penalty": float(getattr(reward_breakdown, "inventory_penalty", 0.0)),
        "reward_inventory_change_penalty": float(
            getattr(reward_breakdown, "inventory_change_penalty", 0.0)
        ),
        "reward_adverse_selection_penalty": float(
            getattr(reward_breakdown, "adverse_selection_penalty", 0.0)
        ),
        "reward_drawdown_penalty": float(getattr(reward_breakdown, "drawdown_penalty", 0.0)),
        "reward_fee_penalty": float(getattr(reward_breakdown, "fee_penalty", 0.0)),
        "reward_risk_override_penalty": float(
            getattr(reward_breakdown, "risk_override_penalty", 0.0)
        ),
        "reward_cancel_penalty": float(getattr(reward_breakdown, "cancel_penalty", 0.0)),
        "reward_current_equity": float(getattr(reward_breakdown, "current_equity", 0.0)),
        "reward_previous_equity": float(getattr(reward_breakdown, "previous_equity", 0.0)),
        "reward_inventory_ratio": float(getattr(reward_breakdown, "inventory_ratio", 0.0)),
        "reward_drawdown_scaled": float(getattr(reward_breakdown, "drawdown_scaled", 0.0)),
    }


def account_stats(env: MarketMakingEnv) -> dict[str, float]:
    """
    Important: obs['portfolio'] is now the 18-feature auxiliary vector.
    Do not read raw cash/equity from it. Pull true account state from env.engine.
    """
    mid = env.engine.current_mid_decimal()
    portfolio = env.engine.portfolio

    return {
        "inventory": float(portfolio.inventory),
        "cash": float(portfolio.cash),
        "equity": float(portfolio.equity(mid)),
        "unrealized": float(portfolio.unrealized_pnl(mid)),
    }


def save_checkpoint(
    path: Path,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: BotConfig,
    replay: Path,
    n_actions: int,
    portfolio_dim: int,
    book_channels: int,
    global_step: int,
    episode: int,
    model_type: str = "dueling_dqn",
    encoder_checkpoint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.model_dump(mode="json"),
            "portfolio_dim": int(portfolio_dim),
            "book_channels": int(book_channels),
            "replay": str(replay),
            "n_actions": int(n_actions),
            "global_step": int(global_step),
            "episode": int(episode),
            "model_type": model_type,
            "encoder_checkpoint": encoder_checkpoint,
            "extra": extra or {},
        },
        path,
    )


def print_progress(row: dict[str, Any]) -> None:
    table = Table(title=f"Training step {row['global_step']}")
    table.add_column("metric")
    table.add_column("value")

    keys = [
        "episode",
        "episode_step",
        "epsilon",
        "reward",
        "rolling_reward_100",
        "loss",
        "q_max",
        "inventory",
        "cash",
        "equity",
        "open_orders",
        "fills",
        "risk_reason",
        "action_name",
    ]

    for key in keys:
        table.add_row(key, str(row.get(key)))

    console.print(table)


def run_greedy_eval(
    cfg: BotConfig,
    snapshot,
    events,
    policy: DuelingDQN,
    device: torch.device,
    max_eval_steps: int,
) -> dict[str, Any]:
    env = MarketMakingEnv(cfg, snapshot, events)
    obs, _ = env.reset()
    obs = clean_obs(obs)

    total_reward = 0.0
    action_counts: Counter[int] = Counter()
    risk_counts: Counter[str] = Counter()
    max_abs_inventory = 0.0
    total_fills = 0

    policy.eval()

    steps = 0
    done = False

    while not done and steps < max_eval_steps:
        with torch.no_grad():
            book, portfolio = obs_to_tensors(obs, device)
            q_values = policy(book, portfolio)

            if not torch.isfinite(q_values).all():
                raise RuntimeError(f"Non-finite Q-values detected: {q_values}")

            action = int(q_values.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = clean_obs(next_obs)
        done = bool(terminated or truncated)

        total_reward += float(reward)
        action_counts[action] += 1

        risk_reason = info.get("risk")
        if risk_reason:
            risk_counts[str(risk_reason)] += 1

        stats = account_stats(env)
        max_abs_inventory = max(max_abs_inventory, abs(stats["inventory"]))
        total_fills += int(info.get("fills", len(env.engine.last_fills)))

        obs = next_obs
        steps += 1

    policy.train()

    stats = account_stats(env)

    return {
        "eval_steps": steps,
        "eval_total_reward": total_reward,
        "eval_avg_reward": total_reward / max(1, steps),
        "eval_final_inventory": stats["inventory"],
        "eval_final_cash": stats["cash"],
        "eval_final_equity": stats["equity"],
        "eval_final_unrealized": stats["unrealized"],
        "eval_max_abs_inventory": max_abs_inventory,
        "eval_total_fills": total_fills,
        "eval_action_counts": dict(action_counts),
        "eval_risk_counts": dict(risk_counts),
    }


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml")),
    replay: Path = typer.Option(Path("data/raw/events.jsonl")),
    out: Path = typer.Option(Path("models/dqn_demo.pt")),
    run_name: str | None = typer.Option(None),
    episodes: int = typer.Option(3),
    max_steps: int = typer.Option(50_000),
    replay_capacity: int = typer.Option(5_000),
    batch_size: int = typer.Option(32),
    learning_starts: int = typer.Option(5_000),
    train_every_steps: int = typer.Option(4),
    gamma: float = typer.Option(0.99),
    lr: float = typer.Option(1e-4),
    target_update_steps: int = typer.Option(1_000),
    checkpoint_every_steps: int = typer.Option(5_000),
    log_every: int = typer.Option(5000),
    eval_every_episodes: int = typer.Option(1),
    max_eval_steps: int = typer.Option(10_000),
    epsilon_start: float = typer.Option(1.0),
    epsilon_end: float = typer.Option(0.05),
    epsilon_decay_steps: int = typer.Option(50_000),
    seed: int = typer.Option(7),
    device_name: str = typer.Option("auto"),
    torch_threads: int = typer.Option(1),
    encoder_checkpoint: Path | None = typer.Option(
        None,
        help="Optional pretrained market encoder checkpoint. If provided, train an MLP DQN head on frozen encoder embeddings.",
    ),
) -> None:
    torch.set_num_threads(max(1, torch_threads))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = BotConfig.from_yaml(config)
    snapshot, events, counts = load_replay(replay)

    if not events:
        raise typer.BadParameter("replay contains no depth/aggTrade events")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cuda":
        console.print(
            {
                "cuda": True,
                "gpu": torch.cuda.get_device_name(0),
                "torch_cuda_version": torch.version.cuda,
            }
        )
    else:
        console.print("[yellow]Running on CPU. Use --device-name cuda if CUDA is available.[/yellow]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"dqn_{cfg.symbol.lower()}_{timestamp}"
    run_dir = Path("runs") / run_name
    checkpoint_dir = run_dir / "checkpoints"

    train_step_logger = CsvLogger(
        run_dir / "train_steps.csv",
        [
            "global_step",
            "episode",
            "episode_step",
            "epsilon",
            "action",
            "action_name",
            "reward",
            "rolling_reward_100",
            "loss",
            "q_mean",
            "target_q_mean",
            "grad_norm",
            "q_max",
            "q_min",
            "inventory",
            "cash",
            "equity",
            "unrealized",
            "open_orders",
            "fills",
            "risk_reason",
            "reward_equity_delta",
            "reward_equity_delta_scaled",
            "reward_equity_change",
            "reward_spread_capture",
            "reward_inventory_penalty",
            "reward_inventory_change_penalty",
            "reward_adverse_selection_penalty",
            "reward_drawdown_penalty",
            "reward_fee_penalty",
            "reward_risk_override_penalty",
            "reward_cancel_penalty",
            "reward_current_equity",
            "reward_previous_equity",
            "reward_inventory_ratio",
            "reward_drawdown_scaled",
        ],
    )

    episode_logger = CsvLogger(
        run_dir / "train_episodes.csv",
        [
            "episode",
            "steps",
            "episode_reward",
            "avg_reward",
            "epsilon",
            "buffer_size",
            "last_loss",
            "final_inventory",
            "final_cash",
            "final_equity",
            "final_unrealized",
            "action_counts_json",
            "risk_counts_json",
        ],
    )

    eval_logger = CsvLogger(
        run_dir / "eval_episodes.csv",
        [
            "episode",
            "eval_steps",
            "eval_total_reward",
            "eval_avg_reward",
            "eval_final_inventory",
            "eval_final_cash",
            "eval_final_equity",
            "eval_final_unrealized",
            "eval_max_abs_inventory",
            "eval_total_fills",
            "eval_action_counts_json",
            "eval_risk_counts_json",
        ],
    )

    env = MarketMakingEnv(cfg, snapshot, events)
    initial_obs, _ = env.reset()
    initial_obs = clean_obs(initial_obs)

    n_actions = int(env.action_space.n)
    portfolio_dim = int(initial_obs["portfolio"].shape[0])
    book_channels = int(initial_obs["order_book"].shape[0])

    if encoder_checkpoint is not None:
        encoder, encoder_metadata = load_pretrained_encoder(encoder_checkpoint, device)

        expected_book_channels = int(encoder_metadata["book_channels"])
        expected_aux_dim = int(encoder_metadata["aux_dim"])
        embedding_dim = int(encoder_metadata["embedding_dim"])

        if expected_book_channels != book_channels:
            raise RuntimeError(
                f"Encoder expects book_channels={expected_book_channels}, "
                f"but env has book_channels={book_channels}"
            )

        if expected_aux_dim != portfolio_dim:
            raise RuntimeError(
                f"Encoder expects aux_dim={expected_aux_dim}, "
                f"but env has portfolio_dim={portfolio_dim}"
            )

        policy = FrozenEncoderDQN(
            encoder=encoder,
            embedding_dim=embedding_dim,
            n_actions=n_actions,
            freeze_encoder=True,
        ).to(device)

        target_encoder, _ = load_pretrained_encoder(encoder_checkpoint, device)

        target = FrozenEncoderDQN(
            encoder=target_encoder,
            embedding_dim=embedding_dim,
            n_actions=n_actions,
            freeze_encoder=True,
        ).to(device)

        model_type = "frozen_encoder_dqn"

    else:
        policy = DuelingDQN(
            book_channels=book_channels,
            portfolio_dim=portfolio_dim,
            n_actions=n_actions,
        ).to(device)

        target = DuelingDQN(
            book_channels=book_channels,
            portfolio_dim=portfolio_dim,
            n_actions=n_actions,
        ).to(device)

        model_type = "dueling_dqn"

    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    memory = ReplayBuffer(replay_capacity)

    metadata = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config": str(config),
        "replay": str(replay),
        "replay_counts": counts,
        "events_loaded_for_env": len(events),
        "device": str(device),
        "n_actions": n_actions,
        "portfolio_dim": portfolio_dim,
        "book_channels": book_channels,
        "episodes": episodes,
        "max_steps": max_steps,
        "replay_capacity": replay_capacity,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        "train_every_steps": train_every_steps,
        "gamma": gamma,
        "lr": lr,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=json_default),
        encoding="utf-8",
    )

    console.print(metadata)

    global_step = 0
    last_loss: float | None = None
    rolling_rewards: deque[float] = deque(maxlen=100)

    try:
        for ep in range(1, episodes + 1):
            obs, _ = env.reset(seed=seed + ep)
            obs = clean_obs(obs)

            episode_reward = 0.0
            episode_action_counts: Counter[int] = Counter()
            episode_risk_counts: Counter[str] = Counter()

            steps = 0
            done = False

            while not done and steps < max_steps:
                decay = min(1.0, global_step / max(1, epsilon_decay_steps))
                epsilon = epsilon_start + decay * (epsilon_end - epsilon_start)

                action, q_max, q_min = choose_action(policy, obs, epsilon, n_actions, device)

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_obs = clean_obs(next_obs)
                done = bool(terminated or truncated)

                memory.push(obs, action, reward, next_obs, done)

                should_train = (
                    len(memory) >= learning_starts
                    and global_step % train_every_steps == 0
                )

                if should_train:
                    opt_stats = optimize(
                        policy,
                        target,
                        memory,
                        optimizer,
                        batch_size,
                        gamma,
                        device,
                    )
                else:
                    opt_stats = None

                if opt_stats is not None:
                    last_loss = opt_stats["loss"]
                else:
                    opt_stats = {
                        "loss": None,
                        "q_mean": None,
                        "target_q_mean": None,
                        "grad_norm": None,
                    }

                episode_reward += float(reward)
                rolling_rewards.append(float(reward))
                episode_action_counts[action] += 1

                risk_reason = info.get("risk")
                if risk_reason:
                    episode_risk_counts[str(risk_reason)] += 1

                stats = account_stats(env)
                reward_parts = reward_to_dict(info.get("reward_breakdown"))

                row = {
                    "global_step": global_step,
                    "episode": ep,
                    "episode_step": steps,
                    "epsilon": round(epsilon, 6),
                    "action": action,
                    "action_name": ACTION_NAMES.get(action, str(action)),
                    "reward": float(reward),
                    "rolling_reward_100": float(np.mean(rolling_rewards)) if rolling_rewards else 0.0,
                    "loss": opt_stats["loss"],
                    "q_mean": opt_stats["q_mean"],
                    "target_q_mean": opt_stats["target_q_mean"],
                    "grad_norm": opt_stats["grad_norm"],
                    "q_max": q_max,
                    "q_min": q_min,
                    "inventory": stats["inventory"],
                    "cash": stats["cash"],
                    "equity": stats["equity"],
                    "unrealized": stats["unrealized"],
                    "open_orders": len(env.engine.open_orders),
                    "fills": int(info.get("fills", len(env.engine.last_fills))),
                    "risk_reason": risk_reason,
                    **reward_parts,
                }

                if global_step % log_every == 0:
                    train_step_logger.write(row)
                    print_progress(row)

                if global_step > 0 and global_step % target_update_steps == 0:
                    target.load_state_dict(policy.state_dict())
                    console.print(f"[green]Updated target network at step {global_step}[/green]")

                if global_step > 0 and global_step % checkpoint_every_steps == 0:
                    ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
                    save_checkpoint(
                        ckpt_path,
                        policy,
                        optimizer,
                        cfg,
                        replay,
                        n_actions,
                        portfolio_dim,
                        book_channels,
                        global_step,
                        ep,
                        extra={"run_dir": str(run_dir)},
                        model_type=model_type,
                        encoder_checkpoint=None if encoder_checkpoint is None else str(encoder_checkpoint),
                    )
                    console.print(f"[cyan]Saved checkpoint: {ckpt_path}[/cyan]")

                obs = next_obs
                steps += 1
                global_step += 1

            stats = account_stats(env)

            episode_row = {
                "episode": ep,
                "steps": steps,
                "episode_reward": episode_reward,
                "avg_reward": episode_reward / max(1, steps),
                "epsilon": epsilon,
                "buffer_size": len(memory),
                "last_loss": last_loss,
                "final_inventory": stats["inventory"],
                "final_cash": stats["cash"],
                "final_equity": stats["equity"],
                "final_unrealized": stats["unrealized"],
                "action_counts_json": json.dumps(dict(episode_action_counts), default=json_default),
                "risk_counts_json": json.dumps(dict(episode_risk_counts), default=json_default),
            }

            episode_logger.write(episode_row)
            console.print("[bold green]Episode finished[/bold green]", episode_row)

            if eval_every_episodes > 0 and ep % eval_every_episodes == 0:
                eval_result = run_greedy_eval(
                    cfg=cfg,
                    snapshot=snapshot,
                    events=events,
                    policy=policy,
                    device=device,
                    max_eval_steps=max_eval_steps,
                )

                eval_row = {
                    "episode": ep,
                    **{k: v for k, v in eval_result.items() if not isinstance(v, dict)},
                    "eval_action_counts_json": json.dumps(
                        eval_result["eval_action_counts"],
                        default=json_default,
                    ),
                    "eval_risk_counts_json": json.dumps(
                        eval_result["eval_risk_counts"],
                        default=json_default,
                    ),
                }

                eval_logger.write(eval_row)
                console.print("[bold magenta]Greedy evaluation[/bold magenta]", eval_result)

    finally:
        train_step_logger.close()
        episode_logger.close()
        eval_logger.close()

    save_checkpoint(
        out,
        policy,
        optimizer,
        cfg,
        replay,
        n_actions,
        portfolio_dim,
        book_channels,
        global_step,
        episodes,
        extra={"run_dir": str(run_dir)},
        model_type=model_type,
        encoder_checkpoint=None if encoder_checkpoint is None else str(encoder_checkpoint),
    )

    console.print(
        {
            "saved_model": str(out),
            "run_dir": str(run_dir),
            "global_step": global_step,
            "portfolio_dim": portfolio_dim,
            "book_channels": book_channels,
            "train_steps_csv": str(run_dir / "train_steps.csv"),
            "train_episodes_csv": str(run_dir / "train_episodes.csv"),
            "eval_episodes_csv": str(run_dir / "eval_episodes.csv"),
        }
    )


if __name__ == "__main__":
    app()