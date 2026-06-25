"""Head-to-head: model A (blue/learner) vs frozen model B (red) — A's win rate over N episodes.

Measures self-play progress: a new generation should beat the previous one >0.5 (while also still
beating the scripts, checked separately by eval_all_scripts.py). Uses the self-play env so B is
driven from obs1 exactly as during training.

  python train/eval_head2head.py --model-a models/auto/selfplay_v1_best.zip \
      --model-b models/auto/bc_dagger_allscripts_v2.zip --episodes 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

try:
    from .tank_selfplay_env import TankSelfPlayVecEnv
    from .train_moba1v1duel import DUEL_REWARD
except ImportError:
    from tank_selfplay_env import TankSelfPlayVecEnv
    from train_moba1v1duel import DUEL_REWARD

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-a", type=Path, required=True, help="learner (plays blue/player-0).")
    p.add_argument("--model-b", type=Path, required=True, help="frozen opponent (plays red).")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=300_000)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--opp-deterministic", action="store_true", default=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    a = PPO.load(args.model_a if args.model_a.is_absolute() else ROOT / args.model_a, device=args.device)
    b = PPO.load(args.model_b if args.model_b.is_absolute() else ROOT / args.model_b, device=args.device)
    env = TankSelfPlayVecEnv(num_envs=args.n_envs, opponents=[0] * args.n_envs, opp_models=[b],
                             opp_deterministic=args.opp_deterministic, max_steps=args.max_steps,
                             base_seed=args.seed_base, reward=DUEL_REWARD)
    obs = env.reset()
    counts = {"win": 0, "loss": 0, "draw": 0, "timeout": 0, "other": 0}
    done_total = 0
    try:
        while done_total < args.episodes:
            act, _ = a.predict(obs, deterministic=True)
            obs, _r, dones, infos = env.step(act)
            for i, inf in enumerate(infos):
                if dones[i]:
                    res = inf.get("result", "other")
                    counts[res if res in counts else "other"] += 1
                    done_total += 1
    finally:
        env.close()

    n = max(1, done_total)
    wr = counts["win"] / n
    print(f"=== {Path(args.model_a).name} (blue) vs {Path(args.model_b).name} (red) ===")
    print(f"episodes={done_total}  A win={wr:.2f}  loss={counts['loss']/n:.2f}  "
          f"draw={counts['draw']/n:.2f}  timeout={counts['timeout']/n:.2f}")
    print(f"VERDICT: A {'BEATS' if wr > 0.5 else 'does NOT beat'} B ({wr:.2f})")


if __name__ == "__main__":
    main()
