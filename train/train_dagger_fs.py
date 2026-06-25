"""Stage-2: FRAME-STACK DAgger -- give the policy motion perception so it can lead-shoot.

Every prior laika attempt capped at hits_dealt ~0.7-1.0 because the 101-d obs is a single
frame with the opponent's position+facing but NO velocity. You cannot learn to lead a moving
target from a snapshot. Here the policy input is the last K frames concatenated (K*101), so a
plain MLP can infer velocity/dodge from the frame differences.

Reuses the existing demos directly: each episode's obs sequence is windowed into K-stacks
(zero-padded at the episode start) -- no re-generation needed. The expert oracle still labels
on-policy states (DAgger), optionally per-opponent via --expert-map. Fresh policy (the new
K*101 input cannot warm-start a 101-d model).

  python train/train_dagger_fs.py --n-stack 3 \
      --data-glob "data/expert_demos/moba_perop/*.jsonl" \
      --scenario moba1v1duel --opponents laika,easy_laika,stationary --spawn-powerups \
      --expert-map "laika=laika-aggressive,easy_laika=laika-aggressive-pro,stationary=laika-aggressive-pro" \
      --out models/bc_dagger_fs_moba_v1.zip --device cpu
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
from collections import deque, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

try:
    from .tank_env import TankEnv
    from .train_bc import FIRE_ACTIONS, N_ACTIONS, ACTION_TABLE, classify_source, _SpaceEnv
    from .train_moba1v1duel import DUEL_REWARD
except ImportError:
    from tank_env import TankEnv
    from train_bc import FIRE_ACTIONS, N_ACTIONS, ACTION_TABLE, classify_source, _SpaceEnv
    from train_moba1v1duel import DUEL_REWARD

ROOT = Path(__file__).resolve().parents[1]
OBS = 101


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-stack", type=int, default=3, help="Frames concatenated into the obs (motion perception).")
    p.add_argument("--data-glob", required=True)
    p.add_argument("--out", type=Path, default=Path("models/bc_dagger_fs_v1.zip"))
    p.add_argument("--scenario", default="moba1v1duel")
    p.add_argument("--arena", default="survival")
    p.add_argument("--opponent", default="laika")
    p.add_argument("--opponents", default="", help="Comma-sep opponent pool (round-robin collect, per-opponent eval).")
    p.add_argument("--expert", default="laika-aggressive-pro")
    p.add_argument("--expert-map", default="", help="LABEL-TIME oracle: opponent=expert,... (policy obs has no identity).")
    p.add_argument("--spawn-powerups", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--rollout-episodes", type=int, default=45)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--bc-epochs", type=int, default=14, help="Initial BC epochs on the stacked demos.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--fire-weight", type=float, default=4.0)
    p.add_argument("--beta0", type=float, default=0.3)
    p.add_argument("--beta-decay", type=float, default=0.5)
    p.add_argument("--initial-cap", type=int, default=80000)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1307)
    return p.parse_args()


def expand_globs(spec):
    files = []
    for pat in spec.split(","):
        pat = pat.strip()
        if pat:
            files.extend(globmod.glob(pat, recursive=True) or globmod.glob(str(ROOT / pat), recursive=True))
    return sorted(set(f for f in files if f.endswith(".jsonl")))


def stack_window(frames):
    """Concatenate a deque of K frames (oldest..newest) into one vector (matches VecFrameStack)."""
    return np.concatenate(list(frames)).astype(np.float32)


def load_stacked_demos(files, K, allowed_scenarios, allowed_opponents, allowed_experts, cap, rng):
    """Window each WON+good episode's obs sequence into K-stacks (zero-pad at start). good_wins filter."""
    X, Y, src = [], [], []
    skipped = 0
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
                    meta[r.get("episode")] = (r.get("expert"), r.get("result"), bool(r.get("good_demo", True)))
                elif r.get("type") == "transition":
                    rows[r.get("episode")].append(r)
        for ep, seq in rows.items():
            if ep not in meta:
                continue
            expert, result, good = meta[ep]
            if not (good and result == "win"):                  # good_wins filter
                continue
            if (expert not in allowed_experts or seq[0].get("scenario") not in allowed_scenarios
                    or seq[0].get("opponent") not in allowed_opponents):
                continue
            buf = deque([np.zeros(OBS, np.float32)] * K, maxlen=K)
            s = classify_source(expert)
            for r in seq:                                       # rows are in step order
                ob, ac = r.get("obs"), r.get("action")
                if (not isinstance(ob, list) or len(ob) != OBS or not isinstance(ac, int)
                        or ac < 0 or ac >= N_ACTIONS):
                    skipped += 1
                    continue
                ctrl = r.get("control") or {}
                if (ctrl.get("throttle"), ctrl.get("turn"), bool(ctrl.get("fire"))) != ACTION_TABLE[ac]:
                    skipped += 1
                    continue
                arr = np.asarray(ob, np.float32)
                if not np.all(np.isfinite(arr)) or arr.min() < -1.0001 or arr.max() > 1.0001:
                    skipped += 1
                    continue
                buf.append(arr)
                X.append(stack_window(buf))
                Y.append(ac)
                src.append(s)
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.int64)
    if cap and len(X) > cap:
        keep = rng.choice(len(X), cap, replace=False)
        X, Y = X[keep], Y[keep]
        src = [src[i] for i in keep]
    return X, Y, src, skipped


class Stacker:
    """Maintains the last K frames during a live rollout."""
    def __init__(self, K):
        self.K = K
        self.buf = deque([np.zeros(OBS, np.float32)] * K, maxlen=K)

    def reset(self, obs):
        self.buf = deque([np.zeros(OBS, np.float32)] * self.K, maxlen=self.K)
        self.buf.append(np.asarray(obs, np.float32))
        return stack_window(self.buf)

    def push(self, obs):
        self.buf.append(np.asarray(obs, np.float32))
        return stack_window(self.buf)


def predict(model, sobs, deterministic=True):
    return int(np.asarray(model.predict(sobs, deterministic=deterministic)[0]).item())


def train_on(model, optimizer, X, Y, epochs, batch_size, class_w, rng):
    model.policy.set_training_mode(True)
    n = len(X)
    last = float("nan")
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            b = order[s:s + batch_size]
            optimizer.zero_grad()
            ot, _ = model.policy.obs_to_tensor(X[b])
            logits = model.policy.get_distribution(ot).distribution.logits
            tgt = torch.as_tensor(Y[b], dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, tgt, weight=class_w)
            loss.backward()
            optimizer.step()
            last = loss.item()
    return last


def collect(model, env, episodes, beta, seed_base, rng, opponents, expert_map, default_expert, K):
    model.policy.set_training_mode(False)
    X, Y, nfire = [], [], 0
    for i in range(episodes):
        opp = opponents[i % len(opponents)]
        env.opponent = opp
        env.expert = expert_map.get(opp, default_expert)        # label-time oracle (NOT in policy obs)
        obs, info = env.reset(seed=seed_base + i)
        st = Stacker(K)
        sobs = st.reset(obs)
        ea = info.get("expertAction")
        done = False
        while not done:
            if ea is not None:
                X.append(sobs.copy())
                Y.append(int(ea))
                if int(ea) in FIRE_ACTIONS:
                    nfire += 1
            a = int(ea) if (ea is not None and rng.random() < beta) else predict(model, sobs)
            obs, _r, term, trunc, info = env.step(a)
            sobs = st.push(obs)
            ea = info.get("expertAction")
            done = term or trunc
    return X, Y, 100.0 * nfire / max(1, len(Y))


def rollout_stacked(model, env, episodes, seed_base, K):
    counts = {"win": 0, "loss": 0, "draw": 0, "timeout": 0}
    length = []
    hd = ht = sh = fh = oh = 0.0
    fire = total = 0
    for i in range(episodes):
        obs, info = env.reset(seed=seed_base + i)
        st = Stacker(K)
        sobs = st.reset(obs)
        done, steps = False, 0
        while not done:
            a = predict(model, sobs)
            if a in FIRE_ACTIONS:
                fire += 1
            total += 1
            steps += 1
            obs, _r, term, trunc, info = env.step(a)
            sobs = st.push(obs)
            done = term or trunc
        res = info.get("result", "draw")
        counts[res if res in counts else "draw"] += 1
        length.append(steps)
        hd += float(info.get("hitsDealt", 0.0)); ht += float(info.get("hitsTaken", 0.0))
        sh += float(info.get("selfHits", 0.0))
        fh += float(info.get("learnerHealth", 0.0)); oh += float(info.get("opponentHealth", 0.0))
    n = max(1, episodes)
    return {"win_rate": counts["win"] / n, "avg_length": sum(length) / n, "fire_pct": 100.0 * fire / max(1, total),
            "avg_hits_dealt": hd / n, "avg_hits_taken": ht / n, "avg_self_hits": sh / n,
            "avg_final_health": fh / n, "avg_opponent_final_health": oh / n}


def eval_pool(model, env, episodes_per, seed_base, opponents, K):
    out = {}
    for opp in opponents:
        env.opponent = opp
        out[opp] = rollout_stacked(model, env, episodes_per, seed_base, K)
    return out


def pool_line(res):
    mw = sum(e["win_rate"] for e in res.values()) / max(1, len(res))
    parts = ["%s win=%.2f hd=%.2f sh=%.2f" % (o, e["win_rate"], e["avg_hits_dealt"], e["avg_self_hits"])
             for o, e in res.items()]
    return "mean_win=%.2f | " % mw + " || ".join(parts)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    K = int(args.n_stack)
    EVAL_SEED = 900_000
    pool = [o.strip() for o in args.opponents.split(",") if o.strip()] or [args.opponent]
    expert_map = {}
    for kv in args.expert_map.split(","):
        k, _, v = kv.partition("=")
        if k.strip() and v.strip():
            expert_map[k.strip()] = v.strip()
    allowed_experts = frozenset({args.expert, "pro"} | set(expert_map.values()))

    files = expand_globs(args.data_glob)
    if not files:
        raise SystemExit(f"no .jsonl matched {args.data_glob!r}")
    X, Y, src, skipped = load_stacked_demos(
        files, K, frozenset([args.scenario]), frozenset(pool), allowed_experts, args.initial_cap, rng)
    if len(X) == 0:
        raise SystemExit("0 stacked demo transitions after good_wins + filters.")
    src_counts = {s: src.count(s) for s in sorted(set(src))}

    space = _SpaceEnv(K * OBS, N_ACTIONS)
    model = PPO("MlpPolicy", space, device=args.device, seed=args.seed, verbose=0)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.lr)
    class_w = torch.ones(N_ACTIONS, device=next(model.policy.parameters()).device)
    for i in FIRE_ACTIONS:
        class_w[i] = args.fire_weight

    env = TankEnv(arena_mode=args.arena, scenario=args.scenario, opponent=pool[0], expert=args.expert,
                  spawn_powerups=args.spawn_powerups, max_steps=args.max_steps, seed=EVAL_SEED,
                  run_dir=ROOT / "runs/dagger_fs", reward=DUEL_REWARD)
    try:
        print("=" * 78)
        print(f"FRAME-STACK DAgger  K={K}  obs={K * OBS}  pool={pool}  expert_map={expert_map or args.expert}")
        print(f"  stacked demos={len(X)} (skipped {skipped})  sources={src_counts}")
        print("NOTE: expert selected by opponent at LABEL time only; policy obs has no opponent identity.")
        bc_loss = train_on(model, optimizer, X, Y, args.bc_epochs, args.batch_size, class_w, rng)
        print(f"  initial BC: {args.bc_epochs} epochs, final loss={bc_loss:.3f}")
        base = eval_pool(model, env, args.eval_episodes, EVAL_SEED, pool, K)
        print(f"iter 0 (BC)           |D|={len(X):6d}  {pool_line(base)}")
        history = [("bc", base)]
        D_X, D_Y = X, Y
        for it in range(args.iters):
            beta = args.beta0 * (args.beta_decay ** it)
            nx, ny, cf = collect(model, env, args.rollout_episodes, beta, 500_000 + it * 10_000,
                                 rng, pool, expert_map, args.expert, K)
            if nx:
                D_X = np.concatenate([D_X, np.asarray(nx, np.float32)], axis=0)
                D_Y = np.concatenate([D_Y, np.asarray(ny, np.int64)], axis=0)
            loss = train_on(model, optimizer, D_X, D_Y, args.epochs, args.batch_size, class_w, rng)
            ev = eval_pool(model, env, args.eval_episodes, EVAL_SEED, pool, K)
            history.append((f"it{it + 1}", ev))
            print(f"iter {it + 1} beta={beta:.3f} +{len(ny):5d} (label fire%={cf:.1f}) "
                  f"|D|={len(D_X):6d} loss={loss:.3f}  {pool_line(ev)}")
    finally:
        env.close()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    model.policy.set_training_mode(False)
    model.save(out)
    # stash n_stack alongside the model so evaluators know how to stack
    (out.parent / (out.stem + "_meta.json")).write_text(json.dumps({"n_stack": K, "obs": K * OBS}))
    print("-" * 78)
    print("FS-DAgger curve (mean win):")
    for tag, ev in history:
        mw = sum(e["win_rate"] for e in ev.values()) / max(1, len(ev))
        per = "  ".join("%s=%.2f" % (o, e["win_rate"]) for o, e in ev.items())
        print(f"  {tag:5s} mean={mw:.2f}   {per}")
    print(f"saved: {out}  (n_stack={K}; eval must stack K frames)")


if __name__ == "__main__":
    main()
