"""Script-vs-script win matrix via eval_script_bot.js (blue=row expert, red=col opponent).

Used to design the per-opponent expert map for DAgger: for each opponent (column), pick the
blue expert (row) with the highest win_rate as that opponent's label-time oracle.

  python train/recon_matrix.py --episodes 40 --blues laika,laika-aggressive,laika-aggressive-pro \
      --reds stationary,easy_laika,laika,laika-aggressive,laika-aggressive-pro
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def one(blue, red, episodes, seed, max_steps, scenario):
    out = subprocess.run(
        ["node", "eval_script_bot.js", "--blue", blue, "--red", red, "--scenario", scenario,
         "--episodes", str(episodes), "--seed", str(seed), "--max-steps", str(max_steps)],
        cwd=str(ROOT), capture_output=True, text=True)
    m = re.search(r"\{.*\}", out.stdout, re.S)
    if not m:
        return None, None
    d = json.loads(m.group(0))
    return float(d["win_rate"]), float(d.get("avg_hits_dealt", -1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blues", default="laika,laika-aggressive,laika-aggressive-pro")
    p.add_argument("--reds", default="stationary,easy_laika,laika,laika-aggressive,laika-aggressive-pro")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=300_000)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--scenario", default="moba1v1duel")
    args = p.parse_args()
    blues = args.blues.split(",")
    reds = args.reds.split(",")

    print(f"WIN MATRIX (blue beats red), {args.episodes}ep seed={args.seed} scenario={args.scenario}")
    hdr = "blue\\red".ljust(24) + "".join(r[:14].rjust(15) for r in reds)
    print(hdr)
    best_for = {}
    for blue in blues:
        cells = []
        for red in reds:
            w, _ = one(blue, red, args.episodes, args.seed, args.max_steps, args.scenario)
            cells.append(w)
            if w is not None and w > best_for.get(red, (-1, ""))[0]:
                best_for[red] = (w, blue)
        print(blue.ljust(24) + "".join(("%.2f" % c if c is not None else "  ? ").rjust(15) for c in cells))
    print("-" * len(hdr))
    print("BEST expert per opponent (DAgger expert-map):")
    for red in reds:
        w, blue = best_for.get(red, (None, "?"))
        print(f"  {red:24s} -> {blue:24s} ({w:.2f})" if w is not None else f"  {red}: ?")


if __name__ == "__main__":
    main()
