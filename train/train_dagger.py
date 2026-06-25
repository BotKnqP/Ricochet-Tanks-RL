"""DAgger for the tank duels: fix BC's covariate shift with on-policy expert labels.

Plain BC collapses at rollout (small aim errors -> off-distribution states with no fire
signal). DAgger fixes exactly this: roll out the CURRENT learner (visit ITS states), ask
the laika-aggressive-pro script (a cheap JS oracle via rl_bridge -> info["expertAction"])
what it would do there, aggregate those (state, expert action) pairs, retrain. Repeat.
Pure imitation, no reward design.

Supports an OPPONENT POOL: collection round-robins over the pool (so the learner is
corrected against every opponent), and evaluation reports per-opponent metrics.

  # fresh-from-moba DAgger vs a pool, expert = laika-aggressive-pro:
  python train/train_dagger.py --warm-start none \
      --data-glob "data/expert_demos/moba_transfer/*.jsonl" \
      --scenario moba1v1duel --opponents laika,easy_laika,stationary --spawn-powerups \
      --out models/bc_dagger_moba_pool_v1.zip --device cpu
"""

from __future__ import annotations

import argparse
import glob as globmod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

try:
    from .tank_env import TankEnv
    from .train_bc import load_demos, FIRE_ACTIONS, N_ACTIONS, _SpaceEnv
    from .evaluate_shooting_lab_bc import rollout as eval_rollout
except ImportError:
    from tank_env import TankEnv
    from train_bc import load_demos, FIRE_ACTIONS, N_ACTIONS, _SpaceEnv
    from evaluate_shooting_lab_bc import rollout as eval_rollout


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--warm-start", default="models/bc_shooting_lab_turret_fw4.zip",
                   help="BC .zip to warm-start from, or 'none' to train a fresh initial BC on the demos.")
    p.add_argument("--data-glob", default="data/expert_demos/shooting_lab/pro_vs_turret_open_2000.jsonl",
                   help="Initial expert demos to seed the aggregated dataset (comma-sep + ** ok).")
    p.add_argument("--out", type=Path, default=Path("models/bc_dagger_v1.zip"))
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--rollout-episodes", type=int, default=30, help="On-policy episodes collected per iter.")
    p.add_argument("--epochs", type=int, default=6, help="Retrain epochs per iter (on the full aggregate).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--fire-weight", type=float, default=4.0, help="CE class-weight on the 9 fire classes.")
    p.add_argument("--beta0", type=float, default=0.4, help="Initial expert-mix prob during rollout (DAgger beta).")
    p.add_argument("--beta-decay", type=float, default=0.5, help="beta_i = beta0 * beta_decay**i.")
    p.add_argument("--initial-cap", type=int, default=60000,
                   help="Subsample seed demos to this many transitions so on-policy data carries weight.")
    p.add_argument("--arena", default="open")
    p.add_argument("--scenario", default="battle")
    p.add_argument("--opponent", default="turret")
    p.add_argument("--opponents", default="",
                   help="Comma-separated opponent POOL: round-robin in collection, per-opponent in eval. "
                        "Empty -> use the single --opponent.")
    p.add_argument("--random-turret", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--spawn-powerups", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable powerups (set for moba scenarios; off for the open shooting lab).")
    p.add_argument("--expert", default="laika-aggressive-pro")
    p.add_argument("--filter", default="good_wins", choices=["all", "wins", "good", "good_wins"],
                   help="Which seed demos to keep: all / wins (result==win) / good (good_demo flag) / good_wins (both). "
                        "Use 'wins' for demo sets that carry no good_demo flag (e.g. the v15 set).")
    p.add_argument("--expert-map", default="",
                   help="LABEL-TIME oracle selection only: comma-sep opponent=expert pairs, e.g. "
                        "'laika=laika-aggressive,easy_laika=laika-aggressive-pro'. Picks WHICH expert labels "
                        "actions per rollout opponent. The policy's observation NEVER includes opponent identity; "
                        "this only changes the teacher, not the student's input.")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--ruleset", default="survival_v1", help="survival_v2 -> HP x2 / slower regen / random spawn.")
    p.add_argument("--spawn-mode", default="", help="fixed/half_random/full_random (override ruleset default).")
    p.add_argument("--tank-max-hp", type=float, default=0.0, help="override tankMaxHp (0 -> ruleset default).")
    p.add_argument("--eval-episodes", type=int, default=20, help="Eval episodes PER opponent each iter.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1307)
    return p.parse_args()


def expand_globs(spec: str):
    files = []
    for pat in spec.split(","):
        pat = pat.strip()
        if not pat:
            continue
        files.extend(globmod.glob(pat, recursive=True) or globmod.glob(str(ROOT / pat), recursive=True))
    return sorted(set(f for f in files if f.endswith(".jsonl")))


def make_policy_fn(model, deterministic=True):
    return lambda obs: int(np.asarray(model.predict(obs, deterministic=deterministic)[0]).item())


def train_on(model, optimizer, obs_np, act_np, epochs, batch_size, class_w, rng):
    """CE(obs -> expert action) on the full aggregated dataset, fire-weighted."""
    model.policy.set_training_mode(True)
    n = len(obs_np)
    last = float("nan")
    for _ep in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            b = order[start:start + batch_size]
            optimizer.zero_grad()
            ot, _ = model.policy.obs_to_tensor(obs_np[b])
            logits = model.policy.get_distribution(ot).distribution.logits
            tgt = torch.as_tensor(act_np[b], dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, tgt, weight=class_w)
            loss.backward()
            optimizer.step()
            last = loss.item()
    return last


def collect(model, env, episodes, beta, seed_base, rng, opponents, expert_map, default_expert):
    """Round-robin the opponent pool; roll out (beta*expert + (1-beta)*deterministic learner);
    label EVERY visited state with the (per-opponent) expert's action. The expert is chosen by
    opponent at LABEL time only -- it is never part of the policy's observation."""
    model.policy.set_training_mode(False)
    learner = make_policy_fn(model, deterministic=True)
    obs_buf, act_buf, n_fire = [], [], 0
    for i in range(episodes):
        opp = opponents[i % len(opponents)]                # round-robin opponent
        env.opponent = opp
        env.expert = expert_map.get(opp, default_expert)   # label-time oracle (NOT in policy obs)
        obs, info = env.reset(seed=seed_base + i)
        ea = info.get("expertAction")
        done = False
        while not done:
            if ea is not None:
                obs_buf.append(np.asarray(obs, dtype=np.float32))
                act_buf.append(int(ea))
                if int(ea) in FIRE_ACTIONS:
                    n_fire += 1
            a = int(ea) if (ea is not None and rng.random() < beta) else learner(obs)
            obs, _r, term, trunc, info = env.step(a)
            ea = info.get("expertAction")
            done = term or trunc
    fire_pct = 100.0 * n_fire / max(1, len(act_buf))
    return obs_buf, act_buf, fire_pct


def eval_pool(model, env, episodes_per, seed_base, opponents):
    """Evaluate the deterministic policy against each opponent separately."""
    pol = make_policy_fn(model, deterministic=True)
    out = {}
    for opp in opponents:
        env.opponent = opp
        out[opp] = eval_rollout(env, pol, episodes_per, seed_base)
    return out


def pool_line(res):
    mean_win = sum(ev["win_rate"] for ev in res.values()) / max(1, len(res))
    parts = ["%s win=%.2f hd=%.1f sh=%.2f fire=%.1f len=%.0f"
             % (opp, ev["win_rate"], ev["avg_hits_dealt"], ev["avg_self_hits"], ev["fire_pct"], ev["avg_length"])
             for opp, ev in res.items()]
    return "mean_win=%.2f | " % mean_win + " || ".join(parts)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    EVAL_SEED = 900_000                                    # held-out eval seeds, fixed across iters
    pool = [o.strip() for o in args.opponents.split(",") if o.strip()] or [args.opponent]
    expert_map = {}                                        # label-time oracle: opponent -> expert
    for kv in args.expert_map.split(","):
        k, _, v = kv.partition("=")
        if k.strip() and v.strip():
            expert_map[k.strip()] = v.strip()
    allowed_experts = frozenset({args.expert, "pro"} | set(expert_map.values()))

    files = expand_globs(args.data_glob)
    if not files:
        raise SystemExit(f"no .jsonl matched --data-glob {args.data_glob!r}")
    obs_list, act_list, stats = load_demos(
        files, args.filter,
        frozenset([args.scenario]), frozenset(pool), allowed_experts)
    if not obs_list:
        raise SystemExit(f"0 demo transitions (results={stats['results']}).")
    D_obs = np.asarray(obs_list, dtype=np.float32)
    D_act = np.asarray(act_list, dtype=np.int64)
    if args.initial_cap and len(D_obs) > args.initial_cap:
        keep = rng.choice(len(D_obs), args.initial_cap, replace=False)
        D_obs, D_act = D_obs[keep], D_act[keep]
    if D_obs.shape[1] != 105:
        raise SystemExit(f"obs_size={D_obs.shape[1]} != 101")

    space_env = _SpaceEnv(105, N_ACTIONS)
    class_w = torch.ones(N_ACTIONS)
    for i in FIRE_ACTIONS:
        class_w[i] = args.fire_weight

    ws = str(args.warm_start).strip()
    warm = None if ws.lower() in ("none", "") else (Path(ws) if Path(ws).is_absolute() else ROOT / ws)
    if warm is not None and warm.exists():
        model = PPO.load(warm, env=space_env, device=args.device)
        print(f"warm-started from {warm}")
    else:
        model = PPO("MlpPolicy", space_env, device=args.device, seed=args.seed, verbose=0)
        print("fresh model -> initial BC on the seed demos")
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.lr)
    class_w = class_w.to(next(model.policy.parameters()).device)
    if warm is None or not warm.exists():
        train_on(model, optimizer, D_obs, D_act, max(args.epochs, 10), args.batch_size, class_w, rng)

    env = TankEnv(
        arena_mode=args.arena, scenario=args.scenario, opponent=pool[0],
        random_turret=args.random_turret, expert=args.expert, spawn_powerups=args.spawn_powerups,
        max_steps=args.max_steps, seed=EVAL_SEED, run_dir=ROOT / "runs/bc_dagger_pool",
        ruleset=args.ruleset, spawn_mode=(args.spawn_mode or None),
        tank_max_hp=(args.tank_max_hp or None),
    )
    try:
        print("=" * 78)
        print(f"DAgger: iters={args.iters} rollout_eps={args.rollout_episodes} epochs/iter={args.epochs} "
              f"fire_weight={args.fire_weight} beta0={args.beta0} pool={pool} "
              f"expert_map={expert_map or args.expert} init|D|={len(D_obs)}")
        print("NOTE: expert selected by opponent at LABEL time only; policy obs has no opponent identity.")
        base = eval_pool(model, env, args.eval_episodes, EVAL_SEED, pool)
        print(f"iter 0 (init)         |D|={len(D_obs):6d}  {pool_line(base)}")
        history = [("init", base)]

        # BEST = most BALANCED so far: maximize (min win across pool, then mean win). This refuses
        # a laika-skewed iter (e.g. stationary collapsed) in favour of one that holds all opponents.
        balance_key = lambda ev: (round(min(e["win_rate"] for e in ev.values()), 3),
                                  round(sum(e["win_rate"] for e in ev.values()) / max(1, len(ev)), 3))
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        model.policy.set_training_mode(False)
        model.save(out)                                  # the init BC is the current best
        best_key, best_tag = balance_key(base), "init"

        for it in range(args.iters):
            beta = args.beta0 * (args.beta_decay ** it)
            new_obs, new_act, coll_fire = collect(
                model, env, args.rollout_episodes, beta, 500_000 + it * 10_000, rng, pool,
                expert_map, args.expert)
            if new_obs:
                D_obs = np.concatenate([D_obs, np.asarray(new_obs, dtype=np.float32)], axis=0)
                D_act = np.concatenate([D_act, np.asarray(new_act, dtype=np.int64)], axis=0)
            loss = train_on(model, optimizer, D_obs, D_act, args.epochs, args.batch_size, class_w, rng)
            ev = eval_pool(model, env, args.eval_episodes, EVAL_SEED, pool)
            history.append((f"it{it + 1}", ev))
            bk = balance_key(ev)
            star = ""
            if bk > best_key:                            # new most-balanced -> overwrite `out`
                best_key, best_tag = bk, f"it{it + 1}"
                model.policy.set_training_mode(False)
                model.save(out)
                star = " *BEST(balanced)"
            print(f"iter {it + 1} beta={beta:.3f} +{len(new_act):5d} (label fire%={coll_fire:.1f}) "
                  f"|D|={len(D_obs):6d} loss={loss:.3f}  {pool_line(ev)}{star}")
    finally:
        env.close()

    # `out` holds the best-balanced checkpoint; also stash the final iter for reference.
    final_path = out.with_name(out.stem + "_final.zip")
    model.policy.set_training_mode(False)
    model.save(final_path)
    print("-" * 78)
    print("DAgger curve (min/mean win across pool):")
    for tag, ev in history:
        mean_win = sum(e["win_rate"] for e in ev.values()) / max(1, len(ev))
        mn = min(e["win_rate"] for e in ev.values())
        per = "  ".join("%s=%.2f" % (opp, e["win_rate"]) for opp, e in ev.items())
        print(f"  {tag:6s}  min={mn:.2f} mean={mean_win:.2f}   {per}")
    print(f"BEST balanced = {best_tag} (min={best_key[0]:.2f} mean={best_key[1]:.2f}) -> saved {out}")
    print(f"final iter -> {final_path}")


if __name__ == "__main__":
    main()
