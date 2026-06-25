"""Evaluate one model vs the full scripted-opponent pool; report per-opponent win + maximin summary.

Same eval path (evaluate_shooting_lab_bc.rollout, deterministic) as the training gate, so numbers are
consistent with promotion decisions. "beat all scripts" target = min win across opponents.

  python train/eval_all_scripts.py --model models/auto/bc_dagger_allscripts_v1.zip --episodes 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

try:
    from .tank_env import TankEnv
    from .train_moba1v1duel import DUEL_REWARD
    from .evaluate_shooting_lab_bc import rollout as eval_rollout
except ImportError:
    from tank_env import TankEnv
    from train_moba1v1duel import DUEL_REWARD
    from evaluate_shooting_lab_bc import rollout as eval_rollout

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPPS = "stationary,easy_laika,laika,laika-aggressive,laika-aggressive-pro"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--opponents", default=DEFAULT_OPPS)
    p.add_argument("--episodes", type=int, default=100, help="episodes PER seed base.")
    p.add_argument("--seed-bases", default="300000",
                   help="comma list of seed bases; win rates are averaged over them (robust to the "
                        "strong spawn/powerup seed-dependence of moba1v1duel).")
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    opponents = args.opponents.split(",")
    seed_bases = [int(s) for s in args.seed_bases.split(",")]
    model = PPO.load(model_path, device=args.device)
    run_dir = ROOT / "runs/auto_campaign/_eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = TankEnv(arena_mode="survival", scenario="moba1v1duel", opponent=opponents[0], spawn_powerups=True,
                  max_steps=args.max_steps, run_dir=run_dir, seed=seed_bases[0], reward=DUEL_REWARD)
    pol = lambda obs: int(model.predict(obs, deterministic=True)[0])

    # per-opponent win rate per seed base, then averaged
    per_seed = {o: {} for o in opponents}
    try:
        for opp in opponents:
            env.opponent = opp
            for sb in seed_bases:
                per_seed[opp][sb] = eval_rollout(env, pol, args.episodes, sb)["win_rate"]
    finally:
        env.close()

    wins = {o: sum(per_seed[o].values()) / len(seed_bases) for o in opponents}
    mn = min(wins.values())
    mean = sum(wins.values()) / len(wins)
    print(f"=== {model_path.name} vs all scripts ({args.episodes}ep x {len(seed_bases)} seed bases {seed_bases}) ===")
    print("%-22s %6s   per-seed" % ("opponent", "avg"))
    for o in opponents:
        ps = "  ".join("%d:%.2f" % (sb, per_seed[o][sb]) for sb in seed_bases)
        print("%-22s %6.2f   %s" % (o, wins[o], ps))
    print("-" * 50)
    print(f"MAXIMIN (avg min win) = {mn:.2f}   MEAN = {mean:.2f}   ALL>0.5: {all(w > 0.5 for w in wins.values())}")
    out = args.out or (run_dir / f"{model_path.stem}_allscripts.json")
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps({"model": str(model_path), "episodes": args.episodes, "seed_bases": seed_bases,
                               "wins_avg": wins, "per_seed": per_seed, "min": mn, "mean": mean}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
