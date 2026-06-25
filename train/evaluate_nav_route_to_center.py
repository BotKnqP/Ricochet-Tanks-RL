"""Evaluate a lesson-1b (nav_route_to_center) policy over fixed held-out seeds.

Default: 100 episodes on seeds 900000..900099, deterministic actions.
Writes runs/nav_route_to_center_v1/eval_route.json and prints the summary.

  python train/evaluate_nav_route_to_center.py --model-path models/nav_route_to_center_v1_best.zip --episodes 100
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
    p.add_argument("--model-path", type=Path, default=Path("models/nav_route_to_center_v1_best.zip"))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=900_000)
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument("--run-dir", type=Path, default=Path("runs/nav_route_to_center_v1"))
    p.add_argument("--out", type=Path, default=None, help="defaults to <run-dir>/eval_route.json")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--scenario", default="nav_route_to_center", choices=["nav_route_to_center", "moba_poison_run"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    out_path = args.out if args.out is not None else (args.run_dir / "eval_route.json")
    out_path = out_path if out_path.is_absolute() else ROOT / out_path

    model = PPO.load(model_path, device=args.device)
    env = TankEnv(
        arena_mode="survival",
        scenario=args.scenario,
        opponent="none",
        spawn_powerups=False,
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
                "success": bool(last.get("routeSuccess", False)),
                "death": res == "route_death",
                "timeout": res == "route_timeout",
                "reward": total_r,
                "length": int(last.get("steps", 0)),
                "elapsed": float(last.get("elapsed", 0.0)),
                "wall_hits": int(last.get("wallHits", 0)),
                "new_cells": int(last.get("newCells", 0)),
                "stuck_events": int(last.get("stuckEvents", 0)),
                "no_progress_events": int(last.get("noProgressEvents", 0)),
                "final_path_dist": float(last.get("pathDist", -1)),
                "best_path_dist": float(last.get("bestPathDist", -1)),
                "center_stay_time": float(last.get("centerStayTime", 0.0)),
            })
    finally:
        env.close()

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    def rate(key: str) -> float:
        return float(np.mean([1.0 if r[key] else 0.0 for r in rows])) if rows else 0.0

    def submean(subset, key: str) -> float:
        return float(np.mean([r[key] for r in subset])) if subset else 0.0

    timeouts = [r for r in rows if r["timeout"]]
    successes = [r for r in rows if r["success"]]

    summary = {
        "episodes": len(rows),
        "model": str(model_path),
        "success_rate": rate("success"),
        "death_rate": rate("death"),
        "timeout_rate": rate("timeout"),
        "avg_reward": mean("reward"),
        "avg_episode_length": mean("length"),
        "avg_elapsed": mean("elapsed"),
        "avg_wall_hits": mean("wall_hits"),
        "avg_new_cells": mean("new_cells"),
        "avg_stuck_events": mean("stuck_events"),
        "avg_no_progress_events": mean("no_progress_events"),
        "avg_final_path_dist": mean("final_path_dist"),
        "avg_best_path_dist": mean("best_path_dist"),
        "avg_center_stay_time": mean("center_stay_time"),
        # split success vs timeout so the failure mode is visible
        "timeout_avg_stuck_events": submean(timeouts, "stuck_events"),
        "timeout_avg_new_cells": submean(timeouts, "new_cells"),
        "timeout_avg_no_progress_events": submean(timeouts, "no_progress_events"),
        "timeout_avg_best_path_dist": submean(timeouts, "best_path_dist"),
        "success_avg_elapsed": submean(successes, "elapsed"),
        "success_avg_new_cells": submean(successes, "new_cells"),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "episodes_detail": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
