from __future__ import annotations

import argparse
from pathlib import Path

from mmrl.config import BotConfig
from mmrl.env.market_making_env import MarketMakingEnv
from mmrl.scripts.replay_io import load_replay


ACTION_NAMES = {
    0: "do_nothing",
    1: "quote_both_tight",
    2: "quote_both_normal",
    3: "quote_both_wide",
    4: "skew_reduce_long",
    5: "skew_reduce_short",
    6: "cancel_all",
    7: "only_bid",
    8: "only_ask",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--replay", default="data/raw/events.jsonl")
    parser.add_argument("--action", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--print-every", type=int, default=250)
    args = parser.parse_args()

    cfg = BotConfig.from_yaml(Path(args.config))
    snapshot, events, counts = load_replay(Path(args.replay))

    print("=" * 80, flush=True)
    print("FILL AUDIT STARTED", flush=True)
    print(f"config: {args.config}", flush=True)
    print(f"replay: {args.replay}", flush=True)
    print(f"action: {args.action} / {ACTION_NAMES.get(args.action, 'unknown')}", flush=True)
    print(f"raw replay counts: {counts}", flush=True)
    print(f"events loaded for env: {len(events)}", flush=True)
    print("=" * 80, flush=True)

    if not events:
        raise RuntimeError("Replay has no usable depth/aggTrade events.")

    env = MarketMakingEnv(cfg, snapshot, events)
    obs, _ = env.reset()

    total_reward = 0.0
    total_fills = 0
    total_trade_events = 0
    total_depth_events = 0
    total_processed_events = 0
    total_steps = 0
    last_info = {}

    done = False

    while not done and total_steps < args.max_steps:
        obs, reward, terminated, truncated, info = env.step(args.action)
        done = terminated or truncated

        fills_this_step = int(info.get("fills", len(getattr(env.engine, "last_fills", []))))
        trade_events = int(info.get("trade_events", 0))
        depth_events = int(info.get("depth_events", 0))
        processed_events = int(info.get("processed_events", 1))

        total_reward += float(reward)
        total_fills += fills_this_step
        total_trade_events += trade_events
        total_depth_events += depth_events
        total_processed_events += processed_events
        total_steps += 1
        last_info = info

        if total_steps % args.print_every == 0 or fills_this_step > 0:
            print(
                {
                    "step": total_steps,
                    "reward": reward,
                    "fills_this_step": fills_this_step,
                    "total_fills": total_fills,
                    "open_orders": len(env.engine.open_orders),
                    "inventory": float(env.engine.portfolio.inventory),
                    "cash": float(env.engine.portfolio.cash),
                    "risk": info.get("risk"),
                    "processed_events": processed_events,
                    "trade_events": trade_events,
                    "depth_events": depth_events,
                },
                flush=True,
            )

    print("=" * 80, flush=True)
    print("FILL AUDIT FINISHED", flush=True)
    print(
        {
            "steps": total_steps,
            "total_reward": total_reward,
            "total_fills": total_fills,
            "fills_per_1000_steps": total_fills / max(1, total_steps) * 1000,
            "total_processed_events": total_processed_events,
            "total_trade_events": total_trade_events,
            "total_depth_events": total_depth_events,
            "final_open_orders": len(env.engine.open_orders),
            "final_inventory": float(env.engine.portfolio.inventory),
            "final_cash": float(env.engine.portfolio.cash),
            "last_info": last_info,
        },
        flush=True,
    )
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()