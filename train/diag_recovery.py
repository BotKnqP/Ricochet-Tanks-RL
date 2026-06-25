"""Task A: expert-recovery diagnostic -- are the learner's failure states recoverable?

A failed policy plays blue vs laika; at a takeover point we hand blue's control to the
laika-aggressive expert (via info["expertAction"]) and see if it can still win from the state
the learner drifted into. Takeover is triggered by the learner's health dropping to a threshold
(i.e. "the learner got into trouble") or at a fixed step.

  policy_only ~ the learner's normal laika win-rate (baseline)
  expert_only ~ the expert playing the whole game (ceiling, ~0.99)
  hp<=X       ~ expert takes over once blue health <= X

Reading:
  rescue stays high as the takeover gets later/sicker  -> states are RECOVERABLE; DAgger could fix
    it with better aggregation/training (it's a data/coverage problem).
  rescue collapses as takeover gets later              -> learner reaches UNRECOVERABLE states;
    ordinary DAgger labels there can't bootstrap wins -> need RL / earlier intervention / curriculum.

  python train/diag_recovery.py --model models/bc_dagger_moba_full_v1.zip --opponent laika --episodes 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

try:
    from .tank_env import TankEnv
    from .train_moba1v1duel import DUEL_REWARD
    from .train_bc import N_ACTIONS
except ImportError:
    from tank_env import TankEnv
    from train_moba1v1duel import DUEL_REWARD
    from train_bc import N_ACTIONS

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("models/bc_dagger_moba_full_v1.zip"))
    p.add_argument("--opponent", default="laika")
    p.add_argument("--expert", default="laika-aggressive")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--seed-base", type=int, default=300_000)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def use_expert_now(cond, taken, hp, step):
    if taken:
        return True
    if cond == "policy_only":
        return False
    if cond == "expert_only":
        return True
    if cond.startswith("hp<="):
        return hp <= float(cond[4:])
    if cond.startswith("step>="):
        return step >= int(cond[6:])
    return False


def run(model, env, cond, episodes, seed_base):
    pol = lambda o: int(np.asarray(model.predict(o, deterministic=True)[0]).item())
    wins = 0
    hd = fh = oh = takeover_step = 0.0
    for i in range(episodes):
        obs, info = env.reset(seed=seed_base + i)
        taken, done, step = False, False, 0
        while not done:
            ea = info.get("expertAction")
            hp = float(info.get("learnerHealth", 3.0))
            ue = use_expert_now(cond, taken, hp, step)
            if ue and not taken:
                taken = True
                takeover_step += step
            a = int(ea) if (ue and ea is not None) else pol(obs)
            obs, _r, term, trunc, info = env.step(a)
            step += 1
            done = term or trunc
        if info.get("result") == "win":
            wins += 1
        hd += float(info.get("hitsDealt", 0.0))
        fh += float(info.get("learnerHealth", 0.0))
        oh += float(info.get("opponentHealth", 0.0))
    n = max(1, episodes)
    return {"win": wins / n, "hits_dealt": hd / n, "my_hp": fh / n, "opp_hp": oh / n,
            "avg_takeover_step": takeover_step / n}


def main():
    args = parse_args()
    model = PPO.load(args.model if args.model.is_absolute() else ROOT / args.model, device=args.device)
    assert getattr(model.observation_space, "shape", None) == (101,), "this diagnostic expects a 101-d model"
    env = TankEnv(arena_mode="survival", scenario="moba1v1duel", opponent=args.opponent,
                  expert=args.expert, spawn_powerups=True, max_steps=args.max_steps,
                  run_dir=ROOT / "runs/_recovery", seed=args.seed_base, reward=DUEL_REWARD)
    conds = ["policy_only", "hp<=2.5", "hp<=2.0", "hp<=1.5", "hp<=1.0", "step>=120", "expert_only"]
    try:
        print(f"recovery diagnostic: model={args.model.name} vs {args.opponent}, expert={args.expert}, "
              f"{args.episodes} ep (seed {args.seed_base})")
        print("condition     win   hits_dealt  my_hp  opp_hp  takeover@step")
        for c in conds:
            r = run(model, env, c, args.episodes, args.seed_base)
            print("%-12s  %.2f  %5.2f       %.2f   %.2f    %.0f"
                  % (c, r["win"], r["hits_dealt"], r["my_hp"], r["opp_hp"], r["avg_takeover_step"]))
    finally:
        env.close()


if __name__ == "__main__":
    main()
