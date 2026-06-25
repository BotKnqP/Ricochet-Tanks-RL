"""Deterministic evaluation for a trained Ricochet Tanks policy.

Runs N episodes on a FIXED held-out seed schedule (base_eval_seed + i), with
seed-increment/randomization disabled, so the same model+config always yields
the same numbers and different models are compared on identical situations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

try:
    from .tank_env import TankEnv
except ImportError:
    from tank_env import TankEnv


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic policy evaluation.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--arena-mode", choices=["open", "maze"], default="open")
    parser.add_argument("--opponent", default="stationary")
    parser.add_argument("--spawn-powerups", action="store_true")
    parser.add_argument("--spawn-jitter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--base-eval-seed", type=int, default=900_000)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of greedy (default: deterministic).")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    model = PPO.load(str(model_path), device=args.device)
    env = TankEnv(
        arena_mode=args.arena_mode,
        opponent=args.opponent,
        spawn_powerups=args.spawn_powerups,
        spawn_jitter=args.spawn_jitter,
        seed_increment=False,
        randomize_seed=False,
        max_steps=args.max_steps,
    )

    deterministic = not args.stochastic
    results = {"win": 0, "loss": 0, "draw": 0, "timeout": 0}
    rewards: list[float] = []
    lengths: list[int] = []

    try:
        for i in range(args.n_episodes):
            obs, info = env.reset(seed=args.base_eval_seed + i)
            done = False
            ep_reward = 0.0
            ep_len = 0
            last_info = info
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward += float(reward)
                ep_len += 1
                last_info = info
                done = terminated or truncated
            result = str(last_info.get("result", "unknown"))
            results[result] = results.get(result, 0) + 1
            rewards.append(ep_reward)
            lengths.append(ep_len)
    finally:
        env.close()

    n = max(1, args.n_episodes)
    try:
        model_label = str(model_path.relative_to(ROOT))
    except ValueError:
        model_label = str(model_path)

    print(json.dumps({
        "model": model_label,
        "arena_mode": args.arena_mode,
        "opponent": args.opponent,
        "spawn_jitter": bool(args.spawn_jitter),
        "n_episodes": args.n_episodes,
        "deterministic": deterministic,
        "win_rate": round(results.get("win", 0) / n, 4),
        "mean_reward": round(float(np.mean(rewards)), 4) if rewards else None,
        "mean_length": round(float(np.mean(lengths)), 2) if lengths else None,
        "wins": results.get("win", 0),
        "losses": results.get("loss", 0),
        "draws": results.get("draw", 0),
        "timeouts": results.get("timeout", 0),
    }, indent=2))


if __name__ == "__main__":
    main()
