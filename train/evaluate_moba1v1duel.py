"""Evaluate a moba1v1duel combat policy vs a scripted opponent; writes eval_duel.json.

Example:
  python train/evaluate_moba1v1duel.py --model-path models/moba1v1duel_vs_easy_laika_best.zip \
      --opponent easy_laika --episodes 100 --run-dir runs/moba1v1duel_vs_easy_laika_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

try:
    from .tank_env import TankEnv
    from .train_moba1v1duel import DUEL_REWARD
except ImportError:
    from tank_env import TankEnv
    from train_moba1v1duel import DUEL_REWARD


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--opponent", default="easy_laika")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=900_000)
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument("--run-dir", type=Path, default=Path("runs/moba1v1duel_vs_easy_laika_v1"))
    p.add_argument("--out", type=Path, default=None, help="defaults to <run-dir>/eval_duel.json")
    p.add_argument("--spawn-powerups", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    out = args.out or (run_dir / "eval_duel.json")
    out = out if out.is_absolute() else ROOT / out

    model = PPO.load(model_path, device=args.device)
    env = TankEnv(
        arena_mode="survival", scenario="moba1v1duel", opponent=args.opponent,
        spawn_powerups=args.spawn_powerups, max_steps=args.max_steps,
        run_dir=run_dir, seed=args.seed_base, reward=DUEL_REWARD,
    )

    counts = {"win": 0, "loss": 0, "draw": 0, "timeout": 0}
    rew, length, elapsed = [], [], []
    hits_dealt, hits_taken, self_hits, powerups = [], [], [], []
    final_health, opp_health, poison_dmg = [], [], []
    outcome_elapsed = {"win": [], "loss": [], "draw": []}

    try:
        for i in range(args.episodes):
            obs, info = env.reset(seed=args.seed_base + i)
            done, total_r, steps = False, 0.0, 0
            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, r, term, trunc, info = env.step(int(action))
                total_r += r
                steps += 1
                done = term or trunc
            res = info.get("result", "draw")
            counts[res] = counts.get(res, 0) + (1 if res in counts else 0)
            if res not in counts:
                counts["draw"] += 1
                res = "draw"
            rew.append(total_r)
            length.append(steps)
            elapsed.append(float(info.get("elapsed", 0.0)))
            hits_dealt.append(int(info.get("hitsDealt", 0)))
            hits_taken.append(int(info.get("hitsTaken", 0)))
            self_hits.append(int(info.get("selfHits", 0)))
            powerups.append(int(info.get("powerups", 0)))
            final_health.append(float(info.get("learnerHealth", 0.0)))
            opp_health.append(float(info.get("opponentHealth", 0.0)))
            poison_dmg.append(float(info.get("poisonDamageTaken", 0.0)))
            if res in outcome_elapsed:
                outcome_elapsed[res].append(float(info.get("elapsed", 0.0)))
    finally:
        env.close()

    n = max(1, args.episodes)
    mean = lambda a: float(np.mean(a)) if a else 0.0
    summary = {
        "episodes": args.episodes,
        "model": str(model_path),
        "opponent": args.opponent,
        "scenario": "moba1v1duel",
        "win_rate": counts["win"] / n,
        "loss_rate": counts["loss"] / n,
        "draw_rate": counts["draw"] / n,
        "timeout_rate": counts["timeout"] / n,
        "avg_reward": mean(rew),
        "avg_episode_length": mean(length),
        "avg_elapsed": mean(elapsed),
        "avg_hits_dealt": mean(hits_dealt),
        "avg_hits_taken": mean(hits_taken),
        "avg_self_hits": mean(self_hits),
        "avg_powerups": mean(powerups),
        "avg_final_health": mean(final_health),
        "avg_opponent_final_health": mean(opp_health),
        "avg_poison_damage_taken": mean(poison_dmg),
        "win_avg_elapsed": mean(outcome_elapsed["win"]),
        "loss_avg_elapsed": mean(outcome_elapsed["loss"]),
        "draw_avg_elapsed": mean(outcome_elapsed["draw"]),
    }
    print(json.dumps(summary, indent=2))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
