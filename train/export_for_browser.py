"""Export the campaign agents to one browser policy file (window.RICOCHET_POLICIES).

Writes runs/auto_live/live_policy.js exposing (the in-page watch UI dropdown switches which drives blue):
  .latest  -> league_robust_v2_latest  (the v2 GENERALIST: best held-out, default blue)
  .best    -> league_robust_v2_best    (the v2 gated model: highest uniform floor)
  .anchor  -> selfplay_v1_latest        (the OLD self-play agent, for A/B comparison)
  .specialist -> dagger_aggro_specialist (the laika-aggressive killer)
  .agent/.live aliased to .latest for backward compat. Also writes live_status.js.
"""
from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3 import PPO

try:
    from .export_policy import extract_policy
except ImportError:
    from export_policy import extract_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/auto_live"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "v2": "models/auto/survival_v2_stage1b_best.zip",     # survival_v2 neural generalist (best v2-trained policy; DAgger imitation of the script failed -> obs lacks velocity)
    "v3": "models/auto/league_robust_v3_best.zip",        # old-rule CHAMPION (held-out 0.526, maximin 0.20)
    "v3max": "models/auto/league_robust_v3_latest.zip",   # old-rule max-generalist (held-out 0.586)
    "anchor": "models/auto/selfplay_v1_latest.zip",       # the ORIGINAL self-play agent (held-out 0.284) for A/B
    "specialist": "models/auto/dagger_aggro_specialist.zip",
}
dumped = {}
for k, p in MODELS.items():
    fp = ROOT / p
    if not fp.exists():
        print(f"  skip {k}: {p} (not found)"); continue
    dumped[k] = json.dumps(extract_policy(PPO.load(fp, device="cpu")), separators=(",", ":"))

lines = ["window.RICOCHET_POLICIES = window.RICOCHET_POLICIES || {};"]
for k, j in dumped.items():
    lines.append(f"window.RICOCHET_POLICIES.{k} = {j};")
lines.append("window.RICOCHET_POLICIES.agent = window.RICOCHET_POLICIES.v3;")    # default = the v3 champion
lines.append("window.RICOCHET_POLICIES.live = window.RICOCHET_POLICIES.v3;")
(OUT / "live_policy.js").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "live_policy.json").write_text(dumped["v3"], encoding="utf-8")  # fetch fallback -> v3 champion
(OUT / "live_status.js").write_text(
    "window.RICOCHET_LIVE_STATUS = {\"phase\":\"v3 champion (pick model + opponent below)\","
    "\"opponent\":\"pick below\",\"timesteps\":420000,\"ep_rew_mean\":null,\"ep_len_mean\":null};\n",
    encoding="utf-8")
print(f"wrote {OUT/'live_policy.js'} (" + ", ".join(f"{k} {len(j)//1024}KB" for k, j in dumped.items()) + ")")
