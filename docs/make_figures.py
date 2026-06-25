#!/usr/bin/env python3
"""Generate the statistical figures from the project's VERIFIED result numbers.
No raw training curves survived the repo cleanup, so these chart end-state metrics (win-rates) that are
each cross-referenced in docs/TRAINING_HISTORY.md / docs/JOURNAL.md.
Run: python docs/make_figures.py  ->  writes docs/figures/*.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# consistent academic palette
C_NEURAL = "#4e79a7"   # neural / learned
C_SCRIPT = "#e15759"   # hand-coded script / baseline
C_DELIV  = "#59a14f"   # the deliverable / good outcome
C_MUTE   = "#bab0ac"   # context / weak
C_GOLD   = "#f1a340"   # highlight

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})

def barlabels(ax, bars, fmt="{:.2f}", dy=0.012):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)

# ---- Figure 1: the laika fire-timing wall (the central result) ----
labels = ["RL\nself-play", "DAgger\nclone", "Proximity\nreward", "Aim\nreward",
          "Lead/intercept\nobs feature", "Cornering\nreward", "Hand-coded\nscript"]
vals   = [0.25, 0.30, 0.30, 0.28, 0.32, 0.26, 0.63]
cols   = [C_NEURAL]*6 + [C_SCRIPT]
fig, ax = plt.subplots(figsize=(8.4, 4.3))
bars = ax.bar(labels, vals, color=cols, edgecolor="white", linewidth=0.6)
barlabels(ax, bars)
ax.axhspan(0.26, 0.32, color=C_NEURAL, alpha=0.10)
ax.axhline(0.5, color="#555", lw=1, ls="--")
ax.text(0.15, 0.515, "0.5", color="#555", fontsize=9)
ax.annotate("the wall: every neural\napproach caps at 0.26–0.32",
            xy=(2.5, 0.30), xytext=(2.2, 0.46), fontsize=9.5, color=C_NEURAL,
            ha="center", arrowprops=dict(arrowstyle="->", color=C_NEURAL, lw=1.1))
ax.set_ylim(0, 0.72); ax.set_ylabel("win-rate vs evasive  laika  (random spawn, ≥200 ep)")
ax.set_title("Figure 1.  The fire-timing wall: six neural approaches vs. one hand-coded script", fontweight="bold", fontsize=11.5)
save(fig, "fig1_laika_wall.png")

# ---- Figure 2: imitation beats RL (mean win-rate over the opponent family) ----
labels = ["Plain PPO\n(unanchored)", "RL self-play\n(full pool)", "DAgger clone", "DAgger + self-play\n(deliverable)"]
vals   = [0.27, 0.33, 0.50, 0.59]
cols   = [C_MUTE, C_NEURAL, C_NEURAL, C_DELIV]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
bars = ax.bar(labels, vals, color=cols, edgecolor="white", linewidth=0.6)
barlabels(ax, bars)
ax.axhline(0.5, color="#555", lw=1, ls="--"); ax.text(-0.45, 0.515, "0.5", color="#555", fontsize=9)
ax.set_ylim(0, 0.7); ax.set_ylabel("mean win-rate vs the opponent family")
ax.set_title("Figure 2.  Imitation > RL: a cloned script beats every pure-RL run", fontweight="bold", fontsize=11.5)
ax.annotate("pure RL seesaws to the\nconflict equilibrium", xy=(1, 0.33), xytext=(0.7, 0.50),
            fontsize=9, color="#666", ha="center", arrowprops=dict(arrowstyle="->", color="#888", lw=1))
save(fig, "fig2_imitation_vs_rl.png")

# ---- Figure 3: the multi-task conflict (specialist vs combined, vs laika-aggressive) ----
fig, ax = plt.subplots(figsize=(5.6, 4.2))
bars = ax.bar(["Single combined\npolicy", "Specialist\n(this opponent only)"], [0.04, 0.88],
              color=[C_SCRIPT, C_DELIV], edgecolor="white", linewidth=0.6)
barlabels(ax, bars)
ax.set_ylim(0, 1.0); ax.set_ylabel("win-rate vs  laika-aggressive  (fixed spawn, robust 3-seed)")
ax.set_title("Figure 3.  Every opponent is individually winnable —\nthe wall is the multi-task conflict, not a precision ceiling", fontweight="bold", fontsize=11)
save(fig, "fig3_multitask_conflict.png")

# ---- Figure 4: domain randomization -> held-out generalization ----
versions = ["Anchor", "v2 pilot", "v3 champion"]
held = [0.284, 0.414, 0.526]
mm   = [0.02, 0.12, 0.20]
x = range(len(versions))
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(x, held, "-o", color=C_DELIV, lw=2.2, ms=8, label="held-out mean (never-trained opponents)")
ax.plot(x, mm,   "-s", color=C_NEURAL, lw=2.2, ms=8, label="held-out maximin (worst case)")
for xi, v in zip(x, held): ax.text(xi, v+0.022, f"{v:.3f}", ha="center", fontsize=9.5, color=C_DELIV, fontweight="bold")
for xi, v in zip(x, mm):   ax.text(xi, v+0.022, f"{v:.2f}",  ha="center", fontsize=9.5, color=C_NEURAL, fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(versions)
ax.set_ylim(0, 0.62); ax.set_ylabel("win-rate on held-out opponent set")
ax.set_title("Figure 4.  Domain randomization generalizes to never-trained opponents", fontweight="bold", fontsize=11.5)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
save(fig, "fig4_domain_randomization.png")

# ---- Figure 5: the deliverable, per-opponent (random spawn, reduced pool) ----
labels = ["stationary", "easy_laika", "pro\n(skilled)", "laika\n(evasive)"]
vals   = [0.90, 0.68, 0.47, 0.32]
cols   = [C_DELIV, C_DELIV, C_GOLD, C_SCRIPT]
fig, ax = plt.subplots(figsize=(6.8, 4.2))
bars = ax.bar(labels, vals, color=cols, edgecolor="white", linewidth=0.6)
barlabels(ax, bars)
ax.axhline(0.59, color="#333", lw=1.4, ls="--")
ax.text(3.05, 0.60, "mean 0.59", color="#333", fontsize=9.5, ha="right", fontweight="bold")
ax.set_ylim(0, 1.0); ax.set_ylabel("win-rate (random spawn, 72 ep/opp)")
ax.set_title("Figure 5.  The deployed v1.5 league agent across the four-opponent family", fontweight="bold", fontsize=11.5)
save(fig, "fig5_deliverable_winrates.png")

# ---- Figure 6: seat x spawn (the OOD / fixed-spawn-specialist verification) ----
labels = ["v3champ\nBLUE · fixed", "v3champ\nBLUE · random", "v3champ\nRED · random",
          "v15clone\nBLUE · random", "laika vs laika\n(mirror)"]
vals   = [0.78, 0.167, 0.113, 0.287, 0.503]
cols   = [C_DELIV, C_NEURAL, C_MUTE, C_NEURAL, C_GOLD]
fig, ax = plt.subplots(figsize=(8.0, 4.3))
bars = ax.bar(labels, vals, color=cols, edgecolor="white", linewidth=0.6)
barlabels(ax, bars, fmt="{:.3f}")
ax.axhline(0.5, color="#555", lw=1, ls="--")
ax.set_ylim(0, 0.9); ax.set_ylabel("win-rate vs laika  (survival_v1, 300 ep)")
ax.set_title("Figure 6.  A fixed-spawn / seat-0 specialist runs out-of-distribution elsewhere\n(and why the browser ‘≈0.5’ was a laika-vs-laika mirror)", fontweight="bold", fontsize=10.8)
save(fig, "fig6_seat_spawn.png")

print("done.")
