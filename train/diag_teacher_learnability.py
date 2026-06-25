"""Experiment 2: is the EXPERT's action recoverable from the student's obs? (offline)

If laika-aggressive leads shots using a quantity the student can't reconstruct from K frames,
BC/DAgger will always struggle. This trains a plain classifier (expert obs -> expert action) on
the gallery demos under several input encodings and reports held-out accuracy + fire recall/
precision. The decisive comparison:
  - does accuracy rise with K (more frames)?            -> memory helps
  - does K=1+delta (explicit first-difference/velocity) match or beat K=4? -> it's a MOTION-FEATURE
    EXTRACTION problem (the small MLP can't difference frames itself), not a DAgger/PPO problem.
  - does a WIDE net help where the small net fails?     -> capacity matters.

Episode-level train/val split (no adjacent-frame leakage).

  python train/diag_teacher_learnability.py --data-glob "data/expert_demos/gallery/*.jsonl"
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .train_bc import N_ACTIONS, FIRE_ACTIONS, ACTION_TABLE
except ImportError:
    from train_bc import N_ACTIONS, FIRE_ACTIONS, ACTION_TABLE

ROOT = Path(__file__).resolve().parents[1]
OBS = 101
FIRE = np.array(sorted(FIRE_ACTIONS))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-glob", required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--cap-episodes", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_episodes(files, rng, cap):
    """Return list of per-episode (obs_seq [T,101], act_seq [T]) for good WON episodes."""
    eps = []
    for path in files:
        meta, rows = {}, defaultdict(list)
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
                    meta[r.get("episode")] = (r.get("result"), bool(r.get("good_demo", True)))
                elif r.get("type") == "transition":
                    rows[r.get("episode")].append(r)
        for ep, seq in rows.items():
            if ep not in meta:
                continue
            result, good = meta[ep]
            if not (good and result == "win"):
                continue
            obs, act = [], []
            for r in seq:
                ob, ac = r.get("obs"), r.get("action")
                if not isinstance(ob, list) or len(ob) != OBS or not isinstance(ac, int) or not (0 <= ac < N_ACTIONS):
                    continue
                ctrl = r.get("control") or {}
                if (ctrl.get("throttle"), ctrl.get("turn"), bool(ctrl.get("fire"))) != ACTION_TABLE[ac]:
                    continue
                obs.append(ob)
                act.append(ac)
            if len(obs) >= 4:
                eps.append((np.asarray(obs, np.float32), np.asarray(act, np.int64)))
    rng.shuffle(eps)
    return eps[:cap] if cap else eps


def featurize(eps, K, use_delta):
    """Build (X, Y) for a set of episodes. K-frame stack (oldest..newest); +delta appends obs_t-obs_{t-1}."""
    Xs, Ys = [], []
    for obs, act in eps:
        T = len(obs)
        for t in range(T):
            frames = [obs[max(0, t - k)] if t - k >= 0 else np.zeros(OBS, np.float32) for k in range(K - 1, -1, -1)]
            feat = np.concatenate(frames)
            if use_delta:
                prev = obs[t - 1] if t >= 1 else obs[t]
                feat = np.concatenate([obs[t], obs[t] - prev])
            Xs.append(feat)
            Ys.append(act[t])
    return np.asarray(Xs, np.float32), np.asarray(Ys, np.int64)


class MLP(nn.Module):
    def __init__(self, din, hidden):
        super().__init__()
        layers, d = [], din
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers += [nn.Linear(d, N_ACTIONS)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_eval(Xtr, Ytr, Xva, Yva, hidden, epochs, batch, lr, device, rng):
    net = MLP(Xtr.shape[1], hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xtr_t = torch.as_tensor(Xtr, device=device)
    Ytr_t = torch.as_tensor(Ytr, device=device)
    for _ in range(epochs):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            opt.zero_grad()
            loss = F.cross_entropy(net(Xtr_t[b]), Ytr_t[b])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.as_tensor(Xva, device=device)).argmax(1).cpu().numpy()
    acc = float((pred == Yva).mean())
    fire_true = np.isin(Yva, FIRE)
    fire_pred = np.isin(pred, FIRE)
    rec = float((fire_pred & fire_true).sum() / max(1, fire_true.sum()))      # fire recall
    prec = float((fire_pred & fire_true).sum() / max(1, fire_pred.sum()))     # fire precision
    return acc, rec, prec, float(fire_true.mean())


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    files = sorted(f for pat in args.data_glob.split(",")
                   for f in (globmod.glob(pat.strip(), recursive=True) or globmod.glob(str(ROOT / pat.strip()), recursive=True))
                   if f.endswith(".jsonl"))
    if not files:
        raise SystemExit(f"no .jsonl matched {args.data_glob!r}")
    eps = load_episodes(files, rng, args.cap_episodes)
    n_val = max(1, int(len(eps) * args.val_frac))
    val_eps, tr_eps = eps[:n_val], eps[n_val:]
    print(f"episodes: train={len(tr_eps)} val={len(val_eps)}  (files={len(files)})")

    configs = [
        ("K1            ", 1, False, [64, 64]),
        ("K3            ", 3, False, [64, 64]),
        ("K4            ", 4, False, [64, 64]),
        ("K6            ", 6, False, [64, 64]),
        ("K1+delta      ", 1, True, [64, 64]),
        ("K4  wide      ", 4, False, [256, 256]),
        ("K1+delta wide ", 1, True, [256, 256]),
    ]
    print("config          dim    val_acc  fire_recall  fire_prec   (fire base=%.2f)" % 0.0)
    for name, K, use_delta, hidden in configs:
        Xtr, Ytr = featurize(tr_eps, K, use_delta)
        Xva, Yva = featurize(val_eps, K, use_delta)
        acc, rec, prec, fbase = train_eval(Xtr, Ytr, Xva, Yva, hidden, args.epochs, args.batch_size,
                                           args.lr, args.device, np.random.default_rng(args.seed + 1))
        print("%s  %-5d  %.3f    %.3f        %.3f      (fire base=%.2f)" % (name, Xtr.shape[1], acc, rec, prec, fbase))


if __name__ == "__main__":
    main()
