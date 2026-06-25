"""Script-vs-script evaluator (Python wrapper around eval_script_bot.js).

Evaluates a scripted blue bot vs a scripted red bot on moba1v1duel and prints/saves the
combat metrics. No torch/PPO — it just runs the node engine.

  python train/evaluate_script_bot.py --blue laika-aggressive-pro --red easy_laika \
      --scenario moba1v1duel --episodes 200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--blue", default="laika-aggressive-pro")
    p.add_argument("--red", default="easy_laika")
    p.add_argument("--scenario", default="moba1v1duel")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=1307)
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument("--node-bin", default="node")
    p.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [args.node_bin, "eval_script_bot.js",
           "--blue", args.blue, "--red", args.red, "--scenario", args.scenario,
           "--episodes", str(args.episodes), "--seed", str(args.seed), "--max-steps", str(args.max_steps)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"eval_script_bot.js failed ({proc.returncode})")
    data = json.loads(proc.stdout)
    print(json.dumps(data, indent=2))
    if args.out is not None:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
