"""Evaluate a lesson-1 (nav_powerup_poison) policy over fixed held-out seeds.

Default: 100 episodes on seeds 900000..900099, deterministic actions.
Writes runs/nav_powerup_poison_v1/eval_nav.json and prints the summary.

  python train/evaluate_nav_powerup_poison.py --model-path models/nav_powerup_poison_v1_best.zip --episodes 100
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
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, default=Path("models/nav_powerup_poison_v1_best.zip"))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=900_000)
    p.add_argument("--max-steps", type=int, default=2200)
    p.add_argument("--out", type=Path, default=Path("runs/nav_powerup_poison_v1/eval_nav.json"))
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    out_path = args.out if args.out.is_absolute() else ROOT / args.out

    model = PPO.load(model_path, device=args.device)
    env = TankEnv(
        arena_mode="survival",
        scenario="nav_powerup_poison",
        opponent="none",
        spawn_powerups=True,
        max_steps=args.max_steps,
        seed=args.seed_base,
    )

    rows = []
    try:
        for i in range(args.episodes):
            seed = args.seed_base + i
            obs, info = env.reset(seed=seed)
            done = False
            total_r = 0.0
            last = info
            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, r, term, trunc, info = env.step(int(action))
                total_r += float(r)
                last = info
                done = term or trunc
            res = str(last.get("result", ""))
            rows.append({
                "seed": seed,
                "result": res,
                "success": bool(last.get("navSuccess", False)),
                "success_with_pickup": res == "nav_success_with_pickup",
                "success_no_pickup": res == "nav_success_no_pickup",
                "death": res == "nav_death",
                "timeout": res == "nav_timeout",
                "reward": total_r,
                "length": int(last.get("steps", 0)),
                "survival_time": float(last.get("elapsed", 0.0)),
                "wall_hits": int(last.get("wallHits", 0)),
                "pickups": int(last.get("pickups", 0)),
                "pickups_when_empty": int(last.get("pickupsWhenEmpty", 0)),
                "poison_damage_taken": float(last.get("poisonDamageTaken", 0.0)),
            })
    finally:
        env.close()

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    def rate(key: str) -> float:
        return float(np.mean([1.0 if r[key] else 0.0 for r in rows])) if rows else 0.0

    summary = {
        "episodes": len(rows),
        "model": str(model_path),
        "success_rate": rate("success"),
        "success_with_pickup_rate": rate("success_with_pickup"),
        "success_no_pickup_rate": rate("success_no_pickup"),
        "death_rate": rate("death"),
        "timeout_rate": rate("timeout"),
        "avg_reward": mean("reward"),
        "avg_episode_length": mean("length"),
        "avg_survival_time": mean("survival_time"),
        "avg_wall_hits": mean("wall_hits"),
        "avg_pickups": mean("pickups"),
        "avg_pickups_when_empty": mean("pickups_when_empty"),
        "avg_poison_damage_taken": mean("poison_damage_taken"),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "episodes_detail": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
