"""Analyze demo JSONL datasets: per-source breakdown + overall data-quality stats.

  python train/analyze_demos.py --data-glob "data/expert_demos/**/*.jsonl,data/human_demos/*.jsonl"

Supports comma-separated globs and ** recursion. Source buckets come from the `expert`
field (aggressive / pro / human / other), the same as train_bc.py --source-mix.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
from collections import defaultdict
from pathlib import Path

try:
    from .train_bc import ACTION_TABLE, N_ACTIONS, FIRE_ACTIONS, classify_source
except ImportError:
    from train_bc import ACTION_TABLE, N_ACTIONS, FIRE_ACTIONS, classify_source

ROOT = Path(__file__).resolve().parents[1]


def expand_globs(spec: str):
    files = []
    for pat in spec.split(","):
        pat = pat.strip()
        if not pat:
            continue
        matched = globmod.glob(pat, recursive=True)
        if not matched:
            matched = globmod.glob(str(ROOT / pat), recursive=True)
        files.extend(matched)
    return sorted(set(f for f in files if f.endswith(".jsonl")))


def _src_acc():
    return {"episodes": 0, "wins": 0, "transitions": 0, "fire": 0, "act": [0] * N_ACTIONS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", required=True)
    args = ap.parse_args()

    files = expand_globs(args.data_glob)
    if not files:
        raise SystemExit(f"no .jsonl files matched --data-glob {args.data_glob!r}")

    src = defaultdict(_src_acc)
    o = {"episodes": 0, "transitions": 0, "fire": 0, "act": [0] * N_ACTIONS,
         "results": defaultdict(int), "invalid_actions": 0, "control_mismatch": 0,
         "obs_dims": set(), "act_min": 10 ** 9, "act_max": -1, "good": 0, "bad": 0}

    for path in files:
        ep_meta, rows = {}, []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") == "episode_summary":
                    ep_meta[r.get("episode")] = (r.get("expert"), r.get("result"), bool(r.get("good_demo", True)))
                elif r.get("type") == "transition":
                    rows.append(r)
        for _ep, (expert, result, good) in ep_meta.items():
            s = classify_source(expert)
            src[s]["episodes"] += 1
            o["episodes"] += 1
            o["results"][result or "?"] += 1
            if result == "win":
                src[s]["wins"] += 1
            o["good" if good else "bad"] += 1
        for r in rows:
            s = classify_source(r.get("expert"))
            ac, ob = r.get("action"), r.get("obs")
            if not isinstance(ac, int) or ac < 0 or ac >= N_ACTIONS:
                o["invalid_actions"] += 1
                continue
            if isinstance(ob, list):
                o["obs_dims"].add(len(ob))
            ctrl = r.get("control") or {}
            if (ctrl.get("throttle"), ctrl.get("turn"), bool(ctrl.get("fire"))) != ACTION_TABLE[ac]:
                o["control_mismatch"] += 1
            src[s]["transitions"] += 1
            src[s]["act"][ac] += 1
            o["transitions"] += 1
            o["act"][ac] += 1
            o["act_min"], o["act_max"] = min(o["act_min"], ac), max(o["act_max"], ac)
            if ac in FIRE_ACTIONS:
                src[s]["fire"] += 1
                o["fire"] += 1

    total_tx = max(1, o["transitions"])
    topn = lambda counts, n: ", ".join("%d:%d" % (i, counts[i]) for i in sorted(range(N_ACTIONS), key=lambda j: -counts[j])[:n])

    print("=" * 74)
    print("files=%d  episodes=%d  transitions=%d" % (len(files), o["episodes"], o["transitions"]))
    print("-" * 74)
    print("SOURCE BREAKDOWN (by transition):")
    for s in sorted(src):
        d = src[s]
        wr = d["wins"] / max(1, d["episodes"])
        fp = 100.0 * d["fire"] / max(1, d["transitions"])
        pct = 100.0 * d["transitions"] / total_tx
        print("  %-11s episodes=%-5d transitions=%-8d (%4.1f%%)  win_rate=%.2f  fire%%=%4.1f  top=[%s]"
              % (s, d["episodes"], d["transitions"], pct, wr, fp, topn(d["act"], 5)))
    print("-" * 74)
    print("OVERALL:")
    print("  obs_dim=%s  action_min=%d action_max=%d" % (sorted(o["obs_dims"]), o["act_min"], o["act_max"]))
    print("  invalid_actions=%d  control_mismatch=%d" % (o["invalid_actions"], o["control_mismatch"]))
    print("  fire%%=%.1f" % (100.0 * o["fire"] / total_tx))
    print("  top actions: [%s]" % topn(o["act"], 6))
    print("  result counts: %s" % dict(o["results"]))
    print("  good/bad episodes (by good_demo): %d/%d" % (o["good"], o["bad"]))
    print("=" * 74)


if __name__ == "__main__":
    main()
