"""Uniform per-stage metrics: win-rate / fire% / hits-dealt / self-hits / ttk vs the 4 laika, under a
given spawn mode. Appends a markdown block + a JSONL row to the run dir.
  python train/_stage_eval.py --model models/auto/fixed_bc.zip --label "S1 naked-BC" --episodes 60
"""
import sys, os, argparse, json
from pathlib import Path
HERE = Path(__file__).resolve().parent          # train/
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from stable_baselines3 import PPO
from tank_env import TankEnv
from train_dagger import eval_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--episodes", type=int, default=60)        # per opponent
    ap.add_argument("--spawn-mode", default="tri_fixed")
    ap.add_argument("--ruleset", default="survival_v1")
    ap.add_argument("--opponents", default="laika,easy_laika,stationary,laika-aggressive-pro")
    ap.add_argument("--seed-base", type=int, default=900000)
    ap.add_argument("--out-dir", default=str(ROOT / "runs/tri_pipeline"))
    a = ap.parse_args()

    pool = [o.strip() for o in a.opponents.split(",") if o.strip()]
    model = PPO.load(a.model, device="cpu")
    env = TankEnv(arena_mode="survival", scenario="moba1v1duel", opponent=pool[0],
                  spawn_powerups=True, max_steps=3000, seed=a.seed_base,
                  run_dir=Path(a.out_dir), ruleset=a.ruleset, spawn_mode=a.spawn_mode)
    res = eval_pool(model, env, a.episodes, a.seed_base, pool)   # {opp: {win_rate,avg_hits_dealt,avg_self_hits,fire_pct,avg_length}}

    wins = [res[o]["win_rate"] for o in pool]
    mean_w, min_w = sum(wins) / len(wins), min(wins)
    cleared = sum(1 for w in wins if w > 0.5)

    # Print FIRST so the (expensive) numbers always surface even if a file write fails.
    print(f"[{a.label}] mean_win={mean_w:.3f} min_win={min_w:.3f} cleared={cleared}/4 | " +
          " | ".join(f"{o}={res[o]['win_rate']:.2f}(fire{res[o]['fire_pct']:.0f}/hd{res[o]['avg_hits_dealt']:.1f})" for o in pool),
          flush=True)

    out_md = Path(a.out_dir) / "metrics.md"
    out_js = Path(a.out_dir) / "metrics.jsonl"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    if not out_md.exists():
        out_md.write_text(f"# Pipeline metrics vs the 4 laika "
                          f"(ruleset {a.ruleset}, spawnMode {a.spawn_mode}, obs 105)\n\n"
                          f"Per-stage win / fire% / hits-dealt / self-hits / ttk; each of the 4 laika.\n",
                          encoding="utf-8")
    with out_md.open("a", encoding="utf-8") as f:    # utf-8: Windows defaults to GBK and chokes on non-ascii
        f.write(f"\n## {a.label}  ({a.episodes} ep/opp, {a.spawn_mode})\n\n")
        f.write("| opponent | win | fire% | hits-dealt | self-hits | ttk(steps) | >0.5? |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for o in pool:
            e = res[o]
            f.write(f"| {o} | {e['win_rate']:.2f} | {e['fire_pct']:.0f} | {e['avg_hits_dealt']:.1f} "
                    f"| {e['avg_self_hits']:.2f} | {e['avg_length']:.0f} | {'yes' if e['win_rate'] > 0.5 else 'NO'} |\n")
        f.write(f"\n**mean win {mean_w:.2f} · min win {min_w:.2f} · cleared {cleared}/4 cells >0.5**\n")
    with out_js.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": a.label, "spawn": a.spawn_mode, "episodes": a.episodes,
                            "mean_win": round(mean_w, 4), "min_win": round(min_w, 4), "cleared": cleared,
                            "per_opp": {o: {k: (round(v, 4) if isinstance(v, (int, float)) else v)
                                            for k, v in res[o].items()} for o in pool}}) + "\n")


if __name__ == "__main__":
    main()
