"""Evaluate a Behaviour-Cloned policy on moba1v1duel vs a scripted opponent.

Loads the BC .zip (an SB3 PPO model), rolls out deterministic episodes, and reports
win/loss/draw/timeout rates, action distribution, and fire %. By default it also runs a
random-policy baseline so you can confirm the BC policy is clearly better than random.

Example:
  python train/evaluate_bc.py --model models/bc_moba1v1duel_easy_laika.zip \
      --episodes 100 --opponent easy_laika --scenario moba1v1duel --device cpu
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
    from .train_bc import FIRE_ACTIONS, N_ACTIONS
except ImportError:
    from tank_env import TankEnv
    from train_moba1v1duel import DUEL_REWARD
    from train_bc import FIRE_ACTIONS, N_ACTIONS


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--opponent", default="easy_laika")
    p.add_argument("--scenario", default="moba1v1duel")
    p.add_argument("--seed-base", type=int, default=900_000)
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument("--device", default="cpu")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compare-random", action=argparse.BooleanOptionalAction, default=True,
                   help="Also run a random-policy baseline (min(episodes,30)) for comparison.")
    p.add_argument("--run-dir", type=Path, default=Path("runs/bc_moba1v1duel_easy_laika"))
    p.add_argument("--out", type=Path, default=None, help="defaults to <run-dir>/eval_bc.json")
    return p.parse_args()


def rollout(env, policy_fn, episodes, seed_base):
    counts = {"win": 0, "loss": 0, "draw": 0, "timeout": 0}
    rew, length, elapsed = [], [], []
    hd, ht, sh, fh, oh = [], [], [], [], []          # combat stats (from info, moba1v1duel)
    act_counts = [0] * N_ACTIONS
    fire, total = 0, 0
    for i in range(episodes):
        obs, info = env.reset(seed=seed_base + i)
        done, total_r, steps = False, 0.0, 0
        while not done:
            a = policy_fn(obs)
            obs, r, term, trunc, info = env.step(a)
            if 0 <= a < N_ACTIONS:
                act_counts[a] += 1
                if a in FIRE_ACTIONS:
                    fire += 1
            total += 1
            total_r += r
            steps += 1
            done = term or trunc
        res = info.get("result", "draw")
        counts[res if res in counts else "draw"] += 1
        rew.append(total_r)
        length.append(steps)
        elapsed.append(float(info.get("elapsed", 0.0)))
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
        "avg_reward": mean(rew),
        "avg_length": mean(length),
        "avg_elapsed": mean(elapsed),
        "avg_hits_dealt": mean(hd),
        "avg_hits_taken": mean(ht),
        "avg_self_hits": mean(sh),
        "avg_final_health": mean(fh),
        "avg_opponent_final_health": mean(oh),
        "action_distribution": act_counts,
        "fire_pct": 100.0 * fire / max(1, total),
    }


def main() -> None:
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    out = args.out or (run_dir / "eval_bc.json")
    out = out if out.is_absolute() else ROOT / out

    model = PPO.load(model_path, device=args.device)
    # sanity: BC model spaces must match the eval env (== resume-compat with the PPO trainer)
    assert getattr(model.observation_space, "shape", None) == (101,), \
        f"obs shape mismatch: got {getattr(model.observation_space, 'shape', None)}, expected (101,)"
    assert getattr(model.action_space, "n", None) == N_ACTIONS, \
        f"action mismatch: got n={getattr(model.action_space, 'n', None)}, expected {N_ACTIONS}"

    env = TankEnv(
        arena_mode="survival", scenario=args.scenario, opponent=args.opponent,
        spawn_powerups=True, max_steps=args.max_steps, run_dir=run_dir,
        seed=args.seed_base, reward=DUEL_REWARD,
    )
    assert env.obs_size == model.observation_space.shape[0], \
        f"TankEnv obs_size {env.obs_size} != model obs {model.observation_space.shape[0]}"
    try:
        bc_policy = lambda obs: int(np.asarray(model.predict(obs, deterministic=args.deterministic)[0]).item())
        bc = rollout(env, bc_policy, args.episodes, args.seed_base)

        rand = None
        if args.compare_random:
            rng = np.random.default_rng(args.seed_base)
            n_act = model.action_space.n
            rand_policy = lambda obs: int(rng.integers(0, n_act))
            rand = rollout(env, rand_policy, min(args.episodes, 30), args.seed_base)
    finally:
        env.close()

    summary = {
        "model": str(model_path), "scenario": args.scenario, "opponent": args.opponent,
        "deterministic": bool(args.deterministic),
        **{k: bc[k] for k in ("episodes", "win_rate", "loss_rate", "draw_rate", "timeout_rate",
                              "avg_reward", "avg_length", "avg_elapsed",
                              "avg_hits_dealt", "avg_hits_taken", "avg_self_hits",
                              "avg_final_health", "avg_opponent_final_health",
                              "fire_pct", "action_distribution")},
    }
    if rand is not None:
        summary["random_baseline"] = {k: rand[k] for k in ("episodes", "win_rate", "avg_reward")}
        summary["beats_random"] = bc["win_rate"] > rand["win_rate"]

    print(json.dumps(summary, indent=2))
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
