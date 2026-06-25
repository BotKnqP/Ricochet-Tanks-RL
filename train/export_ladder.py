"""Export every ladder model (SB3 .zip) into one combined browser-style JS weights file
(window.RICOCHET_POLICIES[name] = {layers:[...]}) so the pure-JS Elo harness can run them.
  python train/export_ladder.py
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import json
from stable_baselines3 import PPO
from export_policy import extract_policy

# (zip path relative to ROOT, policy name) — the fixed-spawn lineage + the deployed champion + the v1.5 agent.
LADDER = [
    ("models/auto/fixed_bc.zip",                       "fixed_bc"),
    ("models/auto/fixed_dagger.zip",                   "fixed_dagger"),
    ("runs/fixed_pipeline/fixed_league_best.zip",      "fixed_league"),     # gitignored; from _fixed_pipeline.sh
    ("runs/fixed_pipeline/fixed_dr_league_best.zip",   "fixed_dr_league"),  # the committed ladder_weights.js already has these
    ("models/auto/league_robust_v3_best_105.zip",      "v3champ"),
    ("models/auto/v15_league_agent.zip",               "v15"),
]

out = ROOT / "ladder_weights.js"
chunks = ["window.RICOCHET_POLICIES = window.RICOCHET_POLICIES || {};\n"]
for rel, name in LADDER:
    p = ROOT / rel
    if not p.exists():
        print(f"  SKIP {name}: missing {rel}", flush=True)
        continue
    model = PPO.load(str(p), device="cpu")
    pol = extract_policy(model)
    in_dim = len(pol["layers"][0]["w"][0])
    out_dim = len(pol["layers"][-1]["b"])
    print(f"  {name:16s} in={in_dim} out={out_dim} layers={len(pol['layers'])}  <- {rel}", flush=True)
    chunks.append(f'window.RICOCHET_POLICIES["{name}"] = {json.dumps(pol, separators=(",", ":"))};\n')

out.write_text("".join(chunks), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size // 1024} KB)", flush=True)
