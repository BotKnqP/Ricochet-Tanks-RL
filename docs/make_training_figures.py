#!/usr/bin/env python3
"""Plot the REAL training curves from the fresh re-run logs (the originals were deleted in the repo cleanup).
Sources:
  runs/sp_reduced/monitor.csv         -> per-episode reward (the literal reward curve)
  runs/repro/sp_reduced.log [gate]    -> per-opponent win-rate vs timesteps (self-play learning curve)
  runs/repro/dagger.log    iter N ... -> per-opponent win-rate vs DAgger iteration (imitation learning curve)
Run: python docs/make_training_figures.py  ->  docs/figures/fig7..9_*.png
"""
import os, re, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)
C = {"laika": "#e15759", "easy_laika": "#4e79a7", "stationary": "#59a14f",
     "laika-aggressive-pro": "#f1a340", "mean": "#222222", "min": "#9c755f", "reward": "#4e79a7"}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "font.family": "DejaVu Sans"})

def save(fig, name):
    fig.tight_layout(); p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", p)

def roll(xs, k):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k + 1); out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out

# ---------- Figure 7: self-play episode-reward curve (monitor.csv) ----------
mon = os.path.join(ROOT, "runs", "sp_reduced", "monitor.csv")
if os.path.exists(mon):
    rs, ls = [], []
    with open(mon) as f:
        for line in f:
            if line.startswith("#") or line.startswith("r,l,t"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try: rs.append(float(parts[0])); ls.append(int(float(parts[1])))
                except ValueError: pass
    steps, acc = [], 0
    for l in ls: acc += l; steps.append(acc)
    if rs:
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        ax.scatter(steps, rs, s=6, color=C["reward"], alpha=0.18, label="per-episode reward")
        ax.plot(steps, roll(rs, 40), color=C["reward"], lw=2.4, label="rolling mean (40 ep)")
        ax.axhline(0, color="#888", lw=1, ls="--")
        ax.set_xlabel("environment steps"); ax.set_ylabel("episode reward  (terminal ±3 = win/loss)")
        ax.set_title("Figure 7.  Self-play training reward (reduced-pool, warm-started from the deployed agent)", fontweight="bold", fontsize=10.8)
        ax.legend(frameon=False, fontsize=9.5, loc="lower right")
        save(fig, "fig7_selfplay_reward.png")
else:
    print("WARN: monitor.csv not found yet:", mon)

# ---------- Figure 8: self-play win-rate vs timesteps ([gate] lines) ----------
splog = os.path.join(ROOT, "runs", "repro", "sp_reduced.log")
gate = re.compile(r"\[gate\]\s*@(\d+)\s+cheap\[min=([\d.]+)\s+mean=([\d.]+)\s*\|\s*laik=([\d.]+)\s+easy=([\d.]+)\s+stat=([\d.]+)\s+laik=([\d.]+)\]")
base = re.compile(r"anchor baselines.*'laika':\s*([\d.]+).*'easy_laika':\s*([\d.]+).*'stationary':\s*([\d.]+).*'laika-aggressive-pro':\s*([\d.]+)")
if os.path.exists(splog):
    steps = []; series = {k: [] for k in ["laika", "easy_laika", "stationary", "laika-aggressive-pro", "min", "mean"]}
    with open(splog) as f:
        txt = f.read()
    b = base.search(txt)
    if b:  # step 0 baseline
        v = [float(x) for x in b.groups()]
        steps.append(0); series["laika"].append(v[0]); series["easy_laika"].append(v[1])
        series["stationary"].append(v[2]); series["laika-aggressive-pro"].append(v[3])
        series["min"].append(min(v)); series["mean"].append(sum(v) / 4)
    seen = set()
    for m in gate.finditer(txt):
        st = int(m.group(1))
        if st in seen: continue
        seen.add(st)
        steps.append(st); series["min"].append(float(m.group(2))); series["mean"].append(float(m.group(3)))
        series["laika"].append(float(m.group(4))); series["easy_laika"].append(float(m.group(5)))
        series["stationary"].append(float(m.group(6))); series["laika-aggressive-pro"].append(float(m.group(7)))
    if len(steps) > 1:
        order = sorted(range(len(steps)), key=lambda i: steps[i])
        steps = [steps[i] for i in order]
        for k in series: series[k] = [series[k][i] for i in order]
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        for k in ["stationary", "easy_laika", "laika-aggressive-pro", "laika"]:
            ax.plot(steps, series[k], "-o", color=C[k], lw=1.8, ms=4, label=k)
        ax.plot(steps, series["mean"], "--", color=C["mean"], lw=2.2, label="mean")
        ax.set_ylim(0, 1.0); ax.set_xlabel("environment steps"); ax.set_ylabel("win-rate (gate eval)")
        ax.set_title("Figure 8.  Self-play win-rate over training: the lift, and the flat laika wall", fontweight="bold", fontsize=10.8)
        ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower right")
        save(fig, "fig8_selfplay_winrate.png")
    else:
        print("WARN: <2 gate points parsed from", splog)

# ---------- Figure 9: DAgger imitation learning curve (per-iter win-rates) ----------
dlog = os.path.join(ROOT, "runs", "repro", "dagger.log")
itre = re.compile(r"iter\s+(\d+).*?mean_win=([\d.]+)\s*\|\s*(.*)")
oppre = re.compile(r"([a-z][a-z_\-]+)\s+win=([\d.]+)")
if os.path.exists(dlog):
    its = []; mean = []; per = {k: [] for k in ["laika", "easy_laika", "stationary", "laika-aggressive-pro"]}
    with open(dlog) as f:
        for line in f:
            m = itre.search(line)
            if not m: continue
            its.append(int(m.group(1))); mean.append(float(m.group(2)))
            d = {o: float(w) for o, w in oppre.findall(m.group(3))}
            for k in per: per[k].append(d.get(k, float("nan")))
    if len(its) > 1:
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        for k in ["stationary", "easy_laika", "laika-aggressive-pro", "laika"]:
            ax.plot(its, per[k], "-o", color=C[k], lw=1.8, ms=5, label=k)
        ax.plot(its, mean, "--", color=C["mean"], lw=2.4, label="mean")
        ax.set_ylim(0, 1.0); ax.set_xlabel("DAgger iteration"); ax.set_xticks(its)
        ax.set_ylabel("win-rate (held-out eval seeds)")
        ax.set_title("Figure 9.  DAgger imitation learning curve (clone of the committed-aggression script)", fontweight="bold", fontsize=10.8)
        ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower right")
        save(fig, "fig9_dagger_winrate.png")
    else:
        print("WARN: <2 dagger iters parsed from", dlog)

print("done.")
