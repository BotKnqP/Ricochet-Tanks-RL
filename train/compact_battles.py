"""Compress battles.json -> battles_min.json: downsample frames + round to small ints for inline embed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAXF = 46  # max frames per battle
WR = {"stationary": 0.69, "easy_laika": 0.62, "laika": 0.64,
      "laika-aggressive-pro": 0.46, "laika-aggressive": 0.04}

src = json.load(open(ROOT / "runs/auto_campaign/battles.json"))
a = src["arena"]
walls = []
for w in a["walls"]:
    walls += [w["x"], w["y"], w["w"], w["h"]]

import math
def deg(r):
    return int(round((r % (2 * math.pi)) * 180 / math.pi))

battles = []
for b in src["battles"]:
    F = b["frames"]
    n = len(F)
    idx = sorted(set([round(i * (n - 1) / (MAXF - 1)) for i in range(min(MAXF, n))]))
    frames = []
    for i in idx:
        f = F[i]
        t = f["tanks"]
        fr = {"t": [[t[0]["x"], t[0]["y"], deg(t[0]["a"]), t[0]["h"], 1 if t[0]["alive"] else 0],
                    [t[1]["x"], t[1]["y"], deg(t[1]["a"]), t[1]["h"], 1 if t[1]["alive"] else 0]],
              "s": [[s["x"], s["y"], s["o"]] for s in f["shells"]]}
        pz = f.get("poison")
        if pz and pz.get("rect"):
            r = pz["rect"]
            fr["z"] = [int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])] if isinstance(r, dict) else [int(v) for v in r]
        frames.append(fr)
    # special label override for the specialist
    is_spec = b["label"].startswith("SPECIALIST")
    battles.append({
        "label": b["label"], "opp": b["opponent"], "result": b["result"],
        "model": "specialist" if is_spec else "self-play agent",
        "wr": (0.88 if is_spec else WR.get(b["opponent"], 0.0)),
        "frames": frames,
    })

out = {"w": a["worldW"], "h": a["worldH"], "walls": walls, "battles": battles}
dest = ROOT / "runs/auto_campaign/battles_min.json"
dest.write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote {dest} ({dest.stat().st_size} bytes); battles={len(battles)} "
      f"total_frames={sum(len(b['frames']) for b in battles)}")
