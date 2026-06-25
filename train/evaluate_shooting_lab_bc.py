"""Evaluate a shooting-lab BC policy: OPEN arena, random-position turret, scenario=battle.

The general evaluate_bc.py hardcodes arena_mode='survival' and cannot request a random
turret, so it can't reproduce the shooting lab. This evaluator drives TankEnv with
arena_mode='open' + randomTurret=True so the BC faces the SAME varied-aim task it was
trained on (laika-aggressive-pro vs a randomly-placed turret). The model is an SB3 PPO
.zip; spaces must match the 101-d obs / Discrete(18) action env.

  python train/evaluate_shooting_lab_bc.py --model models/bc_shooting_lab_turret_v1.zip \
      --episodes 200 --arena open --scenario battle --opponent turret --random-turret --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

try:
    from .tank_env import TankEnv
    from .train_bc import FIRE_ACTIONS, N_ACTIONS
except ImportError:
    from tank_env import TankEnv
    from train_bc import FIRE_ACTIONS, N_ACTIONS


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--arena", default="open")
    p.add_argument("--scenario", default="battle")
    p.add_argument("--opponent", default="turret")
    p.add_argument("--random-turret", dest="random_turret", action=argparse.BooleanOptionalAction, default=False,
                   help="Drop the turret at a random position each episode (open arena). --no-random-turret to disable.")
    p.add_argument("--spawn-powerups", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable powerups (set for moba scenarios; off for the open shooting lab).")
    p.add_argument("--seed-base", type=int, default=900_000)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--device", default="cpu")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-dir", type=Path, default=Path("runs/bc_shooting_lab_turret"))
    p.add_argument("--out", type=Path, default=None, help="defaults to <run-dir>/eval_shooting_lab.json")
    return p.parse_args()


def rollout(env, policy_fn, episodes, seed_base):
    counts = {"win": 0, "loss": 0, "draw": 0, "timeout": 0}
    length = []
    hd, ht, sh, fh, oh = [], [], [], [], []
    act_counts = [0] * N_ACTIONS
    fire, total = 0, 0
    for i in range(episodes):
        obs, info = env.reset(seed=seed_base + i)        # distinct seed -> distinct turret position
        done, steps = False, 0
        while not done:
            a = policy_fn(obs)
            obs, r, term, trunc, info = env.step(a)
            if 0 <= a < N_ACTIONS:
                act_counts[a] += 1
                if a in FIRE_ACTIONS:
                    fire += 1
            total += 1
            steps += 1
            done = term or trunc
        res = info.get("result", "draw")
        counts[res if res in counts else "draw"] += 1
        length.append(steps)
        hd.append(float(info.get("hitsDealt", 0.0)))
        ht.append(float(info.get("hitsTaken", 0.0)))
        sh.append(float(info.get("selfHits", 0.0)))
        fh.append(float(info.get("learnerHealth", 0.0)))
        oh.append(float(info.get("opponentHealth", 0.0)))
    n = max(1, episodes)
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "episodes": episodes,
        "win_rate": counts["win"] / n,
        "loss_rate": counts["loss"] / n,
        "draw_rate": counts["draw"] / n,
        "timeout_rate": counts["timeout"] / n,
        "avg_length": mean(length),
        "avg_hits_dealt": mean(hd),
        "avg_hits_taken": mean(ht),
        "avg_self_hits": mean(sh),
        "avg_final_health": mean(fh),
        "avg_opponent_final_health": mean(oh),
        "fire_pct": 100.0 * fire / max(1, total),
        "action_distribution": act_counts,
    }


def main() -> None:
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    out = args.out or (run_dir / "eval_shooting_lab.json")
    out = out if out.is_absolute() else ROOT / out

    model = PPO.load(model_path, device=args.device)
    # sanity: BC model spaces must match the eval env (== the project OBS/ACTION invariants)
    assert getattr(model.observation_space, "shape", None) == (105,), \
        f"obs shape mismatch: got {getattr(model.observation_space, 'shape', None)}, expected (105,)"
    assert getattr(model.action_space, "n", None) == N_ACTIONS, \
        f"action mismatch: got n={getattr(model.action_space, 'n', None)}, expected {N_ACTIONS}"

    env = TankEnv(
        arena_mode=args.arena, scenario=args.scenario, opponent=args.opponent,
        random_turret=args.random_turret, spawn_powerups=args.spawn_powerups,
        max_steps=args.max_steps, run_dir=run_dir, seed=args.seed_base,
    )
    assert env.obs_size == model.observation_space.shape[0], \
        f"TankEnv obs_size {env.obs_size} != model obs {model.observation_space.shape[0]}"
    try:
        bc_policy = lambda obs: int(np.asarray(model.predict(obs, deterministic=args.deterministic)[0]).item())
        bc = rollout(env, bc_policy, args.episodes, args.seed_base)
    finally:
        env.close()

    summary = {
        "model": str(model_path), "arena": args.arena, "scenario": args.scenario,
        "opponent": args.opponent, "random_turret": bool(args.random_turret),
        "deterministic": bool(args.deterministic),
        **bc,
    }
    print(json.dumps(summary, indent=2))
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
