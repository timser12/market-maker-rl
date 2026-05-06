from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from mmrl.config import BotConfig
from mmrl.env.market_making_env import MarketMakingEnv
from mmrl.scripts.replay_io import load_replay

app = typer.Typer(help="Build a supervised pretraining dataset from an offline replay.")
console = Console()


def mid_price_float(env: MarketMakingEnv) -> float:
    return float(env.engine.current_mid_decimal())


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml")),
    replay: Path = typer.Option(Path("data/raw/train.jsonl")),
    out: Path = typer.Option(Path("data/pretrain/market_pretrain.npz")),
    horizon_steps: int = typer.Option(5),
    flat_threshold_bps: float = typer.Option(1.0),
    max_samples: int = typer.Option(20_000),
    stride: int = typer.Option(1),
) -> None:
    cfg = BotConfig.from_yaml(config)
    snapshot, events, counts = load_replay(replay)

    env = MarketMakingEnv(cfg, snapshot, events)
    obs, _ = env.reset()

    order_books: list[np.ndarray] = []
    aux_vectors: list[np.ndarray] = []
    mids: list[float] = []
    flow_delta_norms: list[float] = []

    done = False
    step = 0

    while not done and len(order_books) < max_samples:
        if step % stride == 0:
            order_books.append(obs["order_book"].astype(np.float16, copy=True))
            aux_vectors.append(obs["portfolio"].astype(np.float32, copy=True))
            mids.append(mid_price_float(env))

            # Aux index 14 is flow_delta_norm from our aux_vector.
            flow_delta_norms.append(float(obs["portfolio"][14]) if obs["portfolio"].shape[0] > 14 else 0.0)

        obs, reward, terminated, truncated, info = env.step(0)  # do_nothing; market-only progression
        done = bool(terminated or truncated)
        step += 1

    n = len(order_books)

    if n <= horizon_steps + 2:
        raise RuntimeError(f"Not enough samples: got {n}, horizon={horizon_steps}")

    order_book_array = np.stack(order_books).astype(np.float16)
    aux_array = np.stack(aux_vectors).astype(np.float32)
    mid_array = np.array(mids, dtype=np.float64)
    flow_array = np.array(flow_delta_norms, dtype=np.float32)

    usable_n = n - horizon_steps

    future_return_bps = np.zeros(usable_n, dtype=np.float32)
    direction = np.zeros(usable_n, dtype=np.int64)
    future_volatility_bps = np.zeros(usable_n, dtype=np.float32)
    future_flow_delta = np.zeros(usable_n, dtype=np.float32)
    toxicity = np.zeros(usable_n, dtype=np.float32)

    for i in range(usable_n):
        now_mid = mid_array[i]
        future_mid = mid_array[i + horizon_steps]

        if now_mid <= 0 or future_mid <= 0:
            ret_bps = 0.0
        else:
            ret_bps = float(np.log(future_mid / now_mid) * 10_000.0)

        future_return_bps[i] = ret_bps

        if ret_bps > flat_threshold_bps:
            direction[i] = 2  # up
        elif ret_bps < -flat_threshold_bps:
            direction[i] = 0  # down
        else:
            direction[i] = 1  # flat

        path = mid_array[i : i + horizon_steps + 1]
        path_returns = np.diff(np.log(np.maximum(path, 1e-12))) * 10_000.0
        future_volatility_bps[i] = float(np.std(path_returns)) if len(path_returns) else 0.0

        flow_slice = flow_array[i + 1 : i + horizon_steps + 1]
        flow_delta = float(np.mean(flow_slice)) if len(flow_slice) else 0.0
        future_flow_delta[i] = flow_delta

        # Toxicity proxy:
        # strong one-sided flow + meaningful future price movement.
        toxicity_raw = abs(flow_delta) * abs(ret_bps) / 10.0
        toxicity[i] = float(np.clip(toxicity_raw, 0.0, 1.0))

    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        order_book=order_book_array[:usable_n],
        aux=aux_array[:usable_n],
        direction=direction,
        future_return_bps=future_return_bps,
        future_volatility_bps=future_volatility_bps,
        future_flow_delta=future_flow_delta,
        toxicity=toxicity,
        book_channels=np.array([order_book_array.shape[1]], dtype=np.int64),
        history=np.array([order_book_array.shape[2]], dtype=np.int64),
        levels=np.array([order_book_array.shape[3]], dtype=np.int64),
        aux_dim=np.array([aux_array.shape[1]], dtype=np.int64),
        horizon_steps=np.array([horizon_steps], dtype=np.int64),
        flat_threshold_bps=np.array([flat_threshold_bps], dtype=np.float32),
    )

    console.print(
        {
            "out": str(out),
            "raw_replay_counts": counts,
            "samples": usable_n,
            "order_book_shape": order_book_array[:usable_n].shape,
            "aux_shape": aux_array[:usable_n].shape,
            "direction_counts": {
                "down": int((direction == 0).sum()),
                "flat": int((direction == 1).sum()),
                "up": int((direction == 2).sum()),
            },
            "mean_abs_return_bps": float(np.mean(np.abs(future_return_bps))),
            "mean_toxicity": float(np.mean(toxicity)),
        }
    )


if __name__ == "__main__":
    app()