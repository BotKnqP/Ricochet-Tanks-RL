"""Behaviour Cloning trainer: human-demo JSONL -> SB3 PPO .zip (resume-compatible).

Reads browser-recorded human demonstrations (human_demo_moba1v1duel_easy_laika_*.jsonl),
trains an imitation policy with supervised CrossEntropy(obs -> action id), and saves a
Stable-Baselines3 PPO `.zip`. The output is a drop-in warm start for the existing PPO
trainer:  python train/train_moba1v1duel.py --resume models/bc_moba1v1duel_easy_laika.zip

BC is an 18-class classification problem (action space = Discrete(18)):
    input : obs,    shape [N, 101]   (the same observation schema the PPO agent sees)
    target: action, shape [N]        (the recorded Discrete(18) action id — NOT raw keys)
    loss  : CrossEntropyLoss(logits, action)

Example:
  python train/train_bc.py --data-glob "data/human_demos/*.jsonl" \
      --out models/bc_moba1v1duel_easy_laika.zip --epochs 30 --batch-size 256 \
      --lr 3e-4 --device cpu --filter good_wins
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "moba1v1duel"
OPPONENT = "easy_laika"


# Read-only mirror of game_core.js makeActionTable() — the canonical Discrete(18) order.
# (idle, then throttle{-1,0,1} x turn{-1,0,1} x fire{F,T}, skipping the idle duplicate.)
# This is NOT a change to ACTION_TABLE; it lets us validate recorded controls and derive
# the fire-action set without spawning node.
def _build_action_table():
    table = [(0.0, 0.0, False)]
    for thr in (-1.0, 0.0, 1.0):
        for turn in (-1.0, 0.0, 1.0):
            for fire in (False, True):
                if thr == 0.0 and turn == 0.0 and not fire:
                    continue
                table.append((thr, turn, fire))
    return table


ACTION_TABLE = _build_action_table()
N_ACTIONS = len(ACTION_TABLE)                                  # 18
FIRE_ACTIONS = frozenset(i for i, (_, _, f) in enumerate(ACTION_TABLE) if f)


def classify_source(expert):
    """Map a transition's `expert` field to a mixing-source bucket."""
    if expert == "laika-aggressive":
        return "aggressive"
    if expert in ("laika-aggressive-pro", "pro"):
        return "pro"
    if expert is None or expert == "human":
        return "human"
    return "other"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-glob", required=True,
                   help='Glob for demo JSONL files, e.g. "data/human_demos/*.jsonl". Supports multiple files.')
    p.add_argument("--out", type=Path, default=Path("models/bc_moba1v1duel_easy_laika.zip"),
                   help="Output SB3 PPO .zip (resume-compatible with train_moba1v1duel.py).")
    p.add_argument("--epochs", type=positive_int, default=30)
    p.add_argument("--batch-size", type=positive_int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--filter", default="all", choices=["all", "wins", "good", "good_wins"],
                   help="Which episodes to use: all / wins (result==win) / good (good_demo) / good_wins (both).")
    p.add_argument("--fire-weight", type=float, default=1.0,
                   help="CE class-weight multiplier on the 9 fire-action classes in TRAINING only "
                        "(counteracts rare-fire under-prediction; eval loss/acc stay unweighted). 1.0=off.")
    p.add_argument("--allowed-scenarios", default="moba1v1duel",
                   help="Comma-separated scenarios to accept (transitions from others are skipped).")
    p.add_argument("--allowed-opponents", default="easy_laika,stationary,laika,laika-aggressive,laika-aggressive-pro,turret",
                   help="Comma-separated opponents to accept. Default accepts human (easy_laika) + script-demo opponents.")
    p.add_argument("--allowed-experts", default="",
                   help='Comma-separated demonstrators to accept (the "expert" field). Empty = accept any '
                        "(incl. human demos, which have no expert field).")
    p.add_argument("--source-mix", default="",
                   help='Resample TRANSITIONS to source fractions, e.g. "aggressive=0.60,pro=0.30,human=0.10". '
                        "Source = the expert field bucket (aggressive/pro/human/other). Empty = no resampling.")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1307)
    return p.parse_args()


class _SpaceEnv(gym.Env):
    """Minimal env carrying ONLY the observation/action spaces, so PPO builds a default
    MlpPolicy whose SAVED spaces equal TankEnv's exactly — Box(-1,1,(obs,),float32) and
    Discrete(n) — making the .zip resume-compatible. It is never stepped for real."""

    def __init__(self, obs_size: int, n_actions: int):
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(n_actions)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def load_demos(files, filt, allowed_scenarios, allowed_opponents, allowed_experts):
    """Read JSONL demos: build per-(file,episode) metadata from episode_summary lines,
    then keep transitions per the filter. Returns (obs_list, action_list, stats).
    allowed_experts=None means accept any demonstrator (human demos have no expert field)."""
    obs, actions, sources = [], [], []
    stats = {
        "files": 0, "episodes": set(), "transitions": 0,
        "results": {"win": 0, "loss": 0, "draw": 0, "timeout": 0, "other": 0},
        "skipped": 0, "obs_size": None,
        "act_counts": [0] * N_ACTIONS, "fire_count": 0, "control_mismatch": 0,
    }
    for path in files:
        stats["files"] += 1
        ep_meta = {}        # episode id -> (result, good_demo)
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["skipped"] += 1
                    continue
                t = row.get("type")
                if t == "episode_summary":
                    ep_meta[row.get("episode")] = (row.get("result"), bool(row.get("good_demo", True)))
                elif t == "transition":
                    rows.append(row)
        # tally outcomes from the summaries we saw in this file
        for _ep, (res, _gd) in ep_meta.items():
            stats["results"][res if res in stats["results"] else "other"] += 1
        # keep transitions per the filter + validation
        for row in rows:
            ep = row.get("episode")
            if ep not in ep_meta:                 # unfinalised / crashed episode (no summary) -> drop
                stats["skipped"] += 1
                continue
            res, good = ep_meta[ep]
            keep = (filt == "all"
                    or (filt == "wins" and res == "win")
                    or (filt == "good" and good)
                    or (filt == "good_wins" and good and res == "win"))
            if not keep:
                continue
            ob = row.get("obs")
            ac = row.get("action")
            if (not isinstance(ob, list) or len(ob) == 0 or not isinstance(ac, int)
                    or ac < 0 or ac >= N_ACTIONS
                    or row.get("scenario") not in allowed_scenarios
                    or row.get("opponent") not in allowed_opponents
                    or (allowed_experts is not None and row.get("expert") not in allowed_experts)):
                stats["skipped"] += 1
                continue
            if stats["obs_size"] is None:
                stats["obs_size"] = len(ob)
            if len(ob) != stats["obs_size"]:
                stats["skipped"] += 1
                continue
            ob_arr = np.asarray(ob, dtype=np.float32)
            if not np.all(np.isfinite(ob_arr)) or ob_arr.min() < -1.0001 or ob_arr.max() > 1.0001:
                stats["skipped"] += 1             # NaN/Inf or out of [-1,1] -> reject (obs invariant)
                continue
            ctrl = row.get("control") or {}
            if (ctrl.get("throttle"), ctrl.get("turn"), bool(ctrl.get("fire"))) != ACTION_TABLE[ac]:
                stats["control_mismatch"] += 1    # control disagrees with ACTION_TABLE[action] -> reject
                continue
            obs.append(ob)
            actions.append(ac)
            sources.append(classify_source(row.get("expert")))
            stats["episodes"].add((path, ep))
            stats["transitions"] += 1
            stats["act_counts"][ac] += 1
            if ac in FIRE_ACTIONS:
                stats["fire_count"] += 1
    stats["episodes"] = len(stats["episodes"])
    stats["sources"] = sources
    return obs, actions, stats


def main() -> None:
    args = parse_args()
    # Supports comma-separated patterns and ** recursion (e.g. "data/**/*.jsonl,data/human_demos/*.jsonl").
    files = []
    for pat in args.data_glob.split(","):
        pat = pat.strip()
        if not pat:
            continue
        matched = globmod.glob(pat, recursive=True) or globmod.glob(str(ROOT / pat), recursive=True)
        files.extend(matched)
    files = sorted(set(f for f in files if f.endswith(".jsonl")))
    if not files:
        raise SystemExit(f"no .jsonl files matched --data-glob {args.data_glob!r}")

    allowed_scenarios = frozenset(s.strip() for s in args.allowed_scenarios.split(",") if s.strip())
    allowed_opponents = frozenset(s.strip() for s in args.allowed_opponents.split(",") if s.strip())
    allowed_experts = frozenset(s.strip() for s in args.allowed_experts.split(",") if s.strip()) or None
    obs_list, act_list, stats = load_demos(files, args.filter, allowed_scenarios, allowed_opponents, allowed_experts)
    if not obs_list:
        raise SystemExit(f"0 transitions after filter={args.filter!r}. "
                         f"(files={stats['files']}, results={stats['results']}) — try --filter all.")

    obs = np.asarray(obs_list, dtype=np.float32)
    actions = np.asarray(act_list, dtype=np.int64)
    obs_size = obs.shape[1]
    sources = np.asarray(stats["sources"])
    src_counts = {s: int((sources == s).sum()) for s in sorted(set(sources.tolist()))}

    resample_note = ""
    if args.source_mix:
        mix = {}
        for kv in args.source_mix.split(","):
            k, _, v = kv.partition("=")
            if k.strip():
                mix[k.strip()] = float(v)
        pools = {s: np.where(sources == s)[0] for s in mix}
        base = sum(len(pools[s]) for s in mix) or len(obs)
        rng_mix = np.random.default_rng(args.seed)
        parts, resampled = [], {}
        for s, w in mix.items():
            avail = len(pools[s])
            target = int(round(base * w))
            resampled[s] = target if avail else 0
            if avail and target:
                parts.append(rng_mix.choice(pools[s], size=target, replace=target > avail))
        if parts:
            keep = np.concatenate(parts)
            rng_mix.shuffle(keep)
            obs, actions = obs[keep], actions[keep]
            resample_note = f"  ->source-mix {resampled} (total {len(obs)})"

    fire_pct = 100.0 * float(np.isin(actions, list(FIRE_ACTIONS)).mean()) if len(actions) else 0.0

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(obs))
    n_val = min(max(0, int(len(obs) * args.val_split)), max(0, len(obs) - 1))   # keep >=1 train; disjoint
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    # ---- stats ----
    print("=" * 64)
    print(f"BC data  filter={args.filter}")
    print(f"  files={stats['files']}  episodes={stats['episodes']}  transitions={stats['transitions']}")
    print(f"  outcomes(by episode_summary): {stats['results']}")
    print(f"  source breakdown(transitions): {src_counts}{resample_note}")
    print(f"  obs shape={tuple(obs.shape)}  action shape={tuple(actions.shape)}  (dtype {actions.dtype})")
    print(f"  action distribution: {stats['act_counts']}")
    print(f"  fire-action %: {fire_pct:.1f}%   (weak opponent -> low fire is expected; NOT upweighting)")
    print(f"  skipped(invalid)={stats['skipped']}  control_mismatch={stats['control_mismatch']}")
    print(f"  train/val split: train={len(tr_idx)}  val={len(val_idx)}  (val_split={args.val_split})")
    print("=" * 64)
    if obs_size != 105:
        raise SystemExit(f"obs_size={obs_size} != 101 — the .zip would not resume into the 101-d PPO env; "
                         f"all demos must carry length-105 observations (the project OBS_SIZE).")

    # ---- model (default MlpPolicy; saved spaces match TankEnv -> PPO.load resume works) ----
    env = _SpaceEnv(obs_size, N_ACTIONS)
    model = PPO("MlpPolicy", env, device=args.device, seed=args.seed, verbose=0)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.lr)

    # Fire-action upweighting: the 9 fire classes are rare (~6%) and only fire well when
    # ALIGNED+reloaded; vanilla CE under-predicts them and the rollout policy stops firing.
    # Weighting their TRAINING loss pushes the net to fire when it should. Eval stays unweighted.
    class_w = torch.ones(N_ACTIONS, device=next(model.policy.parameters()).device)
    if args.fire_weight != 1.0:
        for i in FIRE_ACTIONS:
            class_w[i] = args.fire_weight

    def _logits(obs_np):
        # verified recipe (SB3 2.9): obs_to_tensor -> get_distribution -> Categorical.logits.
        # obs_to_tensor takes the float32 ndarray and puts it on the policy device.
        ot, _ = model.policy.obs_to_tensor(obs_np)
        return model.policy.get_distribution(ot).distribution.logits

    def _eval_loss(idx):
        if len(idx) == 0:
            return float("nan"), float("nan")
        model.policy.set_training_mode(False)
        with torch.no_grad():
            logits = _logits(obs[idx])
            tgt = torch.as_tensor(actions[idx], dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, tgt).item()
            acc = (logits.argmax(1) == tgt).float().mean().item()
        return loss, acc

    print(f"training BC: epochs={args.epochs} batch={args.batch_size} lr={args.lr} device={args.device} fire_weight={args.fire_weight}")
    for epoch in range(args.epochs):
        model.policy.set_training_mode(True)
        order = tr_idx[rng.permutation(len(tr_idx))]
        ep_loss, n_batches = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            b = order[start:start + args.batch_size]
            optimizer.zero_grad()
            logits = _logits(obs[b])
            tgt = torch.as_tensor(actions[b], dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, tgt, weight=class_w)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        tr_loss = ep_loss / max(1, n_batches)
        va_loss, va_acc = _eval_loss(val_idx)
        if epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % max(1, args.epochs // 10) == 0:
            msg = f"  epoch {epoch + 1:3d}/{args.epochs}  train_loss={tr_loss:.4f}"
            if not np.isnan(va_loss):
                msg += f"  val_loss={va_loss:.4f}  val_acc={va_acc:.3f}"
            print(msg)

    tr_loss_final, tr_acc_final = _eval_loss(tr_idx)
    va_loss_final, va_acc_final = _eval_loss(val_idx)

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    model.policy.set_training_mode(False)
    model.save(out)

    print("-" * 64)
    print(f"final  train_loss={tr_loss_final:.4f} train_acc={tr_acc_final:.3f}"
          + ("" if np.isnan(va_loss_final) else f"  val_loss={va_loss_final:.4f} val_acc={va_acc_final:.3f}"))
    print(f"saved SB3 PPO model: {out}")
    print(f"resume:  python train/train_moba1v1duel.py --opponent easy_laika --resume {out} ...")


if __name__ == "__main__":
    main()
