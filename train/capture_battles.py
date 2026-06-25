"""Capture real battle trajectories of a neural agent vs each scripted opponent -> battles.json.

Drives the agent through rl_bridge_capture.js (which returns tank/shell/poison positions). For each
opponent it scans a few seeds and keeps a representative battle (prefers a WIN), downsampling frames.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "train" / "rl_bridge_capture.js"

# (label, model path, opponent script, candidate seeds)
JOBS = [
    ("vs stationary",        "models/auto/selfplay_v1_latest.zip",      "stationary",           [300000, 300003, 300007]),
    ("vs easy_laika",        "models/auto/selfplay_v1_latest.zip",      "easy_laika",           [300000, 300001, 300004]),
    ("vs laika",             "models/auto/selfplay_v1_latest.zip",      "laika",                [300000, 300002, 300005, 300009]),
    ("vs laika-aggressive-pro","models/auto/selfplay_v1_latest.zip",    "laika-aggressive-pro", [300000, 300001, 300006]),
    ("vs laika-aggressive",  "models/auto/selfplay_v1_latest.zip",      "laika-aggressive",     [300000, 300002, 300008, 300011, 300014]),
    ("SPECIALIST vs laika-aggressive", "models/auto/dagger_aggro_specialist.zip", "laika-aggressive", [900001, 900004, 900002, 900007, 900010]),
]
MAX_STEPS = 900
STRIDE = 3            # keep every 3rd frame (~2 sim steps each -> ~100ms/frame at 30fps*2)


def bridge():
    return subprocess.Popen([ "node", str(BRIDGE) ], cwd=str(ROOT),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)


def req(proc, payload):
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def run_episode(proc, model, opp, seed):
    r = req(proc, {"cmd": "reset", "seed": seed})
    walls = r["view"].get("walls"); ww = r["view"].get("worldW"); wh = r["view"].get("worldH")
    frames = [r["view"]]
    obs = np.asarray(r["obs"], dtype=np.float32)
    done, result, k = False, None, 0
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        r = req(proc, {"cmd": "step", "action": int(a), "opponent": opp})
        obs = np.asarray(r["obs"], dtype=np.float32)
        k += 1
        if k % STRIDE == 0:
            frames.append(r["view"])
        done = r["done"]; result = r.get("result")
    if frames[-1] is not r["view"]:
        frames.append(r["view"])
    return {"frames": frames, "result": result, "seed": seed, "walls": walls, "worldW": ww, "worldH": wh}


def main():
    out = {"arena": None, "battles": []}
    cache = {}
    for label, mpath, opp, seeds in JOBS:
        if mpath not in cache:
            cache[mpath] = PPO.load(ROOT / mpath, device="cpu")
        model = cache[mpath]
        proc = bridge()
        req(proc, {"cmd": "init", "scenario": "moba1v1duel", "arenaMode": "survival",
                   "spawnPowerups": True, "maxSteps": MAX_STEPS})
        chosen = None
        for seed in seeds:
            ep = run_episode(proc, model, opp, seed)
            if out["arena"] is None and ep["walls"] is not None:
                out["arena"] = {"walls": ep["walls"], "worldW": ep["worldW"], "worldH": ep["worldH"]}
            if chosen is None or (ep["result"] == "win" and chosen["result"] != "win"):
                chosen = ep
            if ep["result"] == "win":
                break
        proc.terminate()
        chosen.pop("walls", None); chosen.pop("worldW", None); chosen.pop("worldH", None)
        chosen["label"] = label; chosen["opponent"] = opp; chosen["model"] = Path(mpath).name
        chosen["nframes"] = len(chosen["frames"])
        out["battles"].append(chosen)
        print(f"{label:34s} seed={chosen['seed']} result={chosen['result']:8s} frames={chosen['nframes']}")
    dest = ROOT / "runs/auto_campaign/battles.json"
    dest.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {dest} ({dest.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
