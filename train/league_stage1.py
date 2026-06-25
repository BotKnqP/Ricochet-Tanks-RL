"""Stage-1 league: AlphaStar-style safe RL improvement of a frozen DAgger anchor.

The lesson from the earlier failures: single-opponent vanilla PPO from a BC/DAgger warm
start ERODES it (vs laika 16->8; vs easy 70->53) and forgets untrained opponents (round-2
forgot stationary). The league fix here is the minimal-but-real version:

  1. ANCHOR is never overwritten -- we branch off it and only ever write new files.
  2. MIXED opponent pool: each of the N vec-envs is pinned to one opponent from the pool,
     so every PPO rollout batches gradient from ALL opponents at once -> no forgetting.
  3. GATES: a checkpoint is promoted to `_best` ONLY if it improves laika WITHOUT eroding
     easy_laika (>= baseline-drop) or stationary (>= floor). Otherwise the anchor stays best.

This is NOT self-play yet (frozen-checkpoint opponents = Stage 1.5). It is the script pool
{stationary, easy_laika, laika, laika-aggressive-pro, laika-aggressive}, which is what
prevents forgetting. Motion perception (frame-stack / recurrent) for laika lead-shooting is
Stage 2 -- this stage verifies the league mechanics protect the anchor's abilities.

  python train/league_stage1.py --anchor models/bc_dagger_moba_full_v1.zip \
      --total-timesteps 300000 --n-envs 16 --lr 1e-4 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

try:
    from .tank_env import TankEnv
    from .tank_vec_env import TankVecEnv
    from .train_moba1v1duel import DUEL_REWARD
    from .train_bc import N_ACTIONS
    from .evaluate_shooting_lab_bc import rollout as eval_rollout
except ImportError:
    from tank_env import TankEnv
    from tank_vec_env import TankVecEnv
    from train_moba1v1duel import DUEL_REWARD
    from train_bc import N_ACTIONS
    from evaluate_shooting_lab_bc import rollout as eval_rollout


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "moba1v1duel"
# round-3 DAgger anchor baselines (100-ep): the gates are relative to these.
BASELINES = {"stationary": 1.00, "easy_laika": 0.69, "laika": 0.14}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", type=Path, default=Path("models/bc_dagger_moba_full_v1.zip"),
                   help="Frozen DAgger anchor to branch from (NEVER overwritten).")
    p.add_argument("--out-prefix", default="models/league_branch_001",
                   help="Branch output prefix -> <prefix>_latest.zip / _best.zip (distinct from the anchor).")
    p.add_argument("--mix", default="easy_laika=0.30,laika=0.30,laika-aggressive-pro=0.20,laika-aggressive=0.10,stationary=0.10",
                   help="Opponent pool sampling weights -> per-env allocation.")
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--eval-interval", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=25, help="Cheap per-interval monitor eval.")
    p.add_argument("--confirm-episodes", type=int, default=100,
                   help="Promotion-gate eval size (anchor baseline + candidate confirm); big enough to beat 25-ep noise.")
    p.add_argument("--eval-seed-base", type=int, default=300_000)
    p.add_argument("--gate-easy-drop", type=float, default=0.10, help="easy_laika must stay >= baseline - this.")
    p.add_argument("--gate-stationary-min", type=float, default=0.90, help="stationary must stay >= this.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1307)
    return p.parse_args()


def allocate(mix: dict[str, float], n: int) -> list[str]:
    """Allocate n envs across the weighted opponent pool.

    Guarantees every weight>0 opponent at least 1 env WHEN n >= #opponents, so a small weight is
    never SILENTLY dropped to zero (the old largest-remainder floor did exactly that: e.g. weight
    0.04 over 10 envs -> int(0.43) = 0, opponent never trained). Surplus envs beyond the 1-each
    floor are shared out in proportion to weight (largest remainder). When n < #opponents it keeps
    the n highest-weight opponents and WARNS loudly about the rest instead of dropping silently."""
    items = [(k, w) for k, w in mix.items() if w > 0]
    if not items:
        return []
    if len(items) > n:
        items.sort(key=lambda kv: -kv[1])
        print(f"[allocate] WARNING: n_envs={n} < #opponents={len(items)} -> dropping lowest-weight "
              f"{[k for k, _ in items[n:]]} (raise --n-envs / lower --self-frac / trim --script-mix)")
        items = items[:n]
    total = sum(w for _, w in items) or 1.0
    surplus = n - len(items)                                  # envs above the 1-each floor
    raw = {k: (w / total) * surplus for k, w in items}
    counts = {k: 1 + int(raw[k]) for k, _ in items}
    short = n - sum(counts.values())
    for k, _ in sorted(items, key=lambda kv: -(raw[kv[0]] - int(raw[kv[0]])))[:max(0, short)]:
        counts[k] += 1
    alloc: list[str] = []
    for k, _ in items:
        alloc += [k] * counts[k]
    return alloc[:n]


class LeagueCallback(BaseCallback):
    """Two-tier gate. The earlier single-tier gate self-calibrated on a noisy 25-ep eval and
    promoted a checkpoint that actually eroded easy/stationary at 100 ep. Now: a cheap eval
    runs every interval for monitoring, and a checkpoint is PROMOTED to `_best` ONLY after a
    CONFIRM eval at `confirm_episodes` passes the gate vs the anchor's OWN confirm-episodes
    baseline (measured once at t=0). Same eval path for baseline + candidate -> self-consistent."""

    def __init__(self, eval_env, eval_opponents, interval, episodes, confirm_episodes, seed_base,
                 best_path, latest_path, easy_drop, stat_min, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_opponents = eval_opponents
        self.interval = int(interval)
        self.episodes = int(episodes)
        self.confirm_episodes = int(confirm_episodes)
        self.seed_base = int(seed_base)
        self.best_path = best_path
        self.latest_path = latest_path
        self.easy_drop = float(easy_drop)
        self.stat_min = float(stat_min)
        self._last_eval = 0
        self.best_laika = -1.0
        self.baselines = None

    def _policy_fn(self):
        return lambda obs: int(np.asarray(self.model.predict(obs, deterministic=True)[0]).item())

    def _run(self, episodes):
        pol = self._policy_fn()
        out = {}
        for opp in self.eval_opponents:
            self.eval_env.opponent = opp
            out[opp] = eval_rollout(self.eval_env, pol, episodes, self.seed_base)
        return out

    def _gate(self, w):
        easy_ok = w.get("easy_laika", 0) >= self.baselines["easy_laika"] - self.easy_drop
        stat_ok = w.get("stationary", 0) >= self.stat_min
        laika_up = w.get("laika", 0) > self.baselines["laika"]
        return easy_ok and stat_ok and laika_up

    def _on_training_start(self):
        res = self._run(self.confirm_episodes)        # anchor @ confirm-N -> the TRUE gate baselines
        self.baselines = {o: res[o]["win_rate"] for o in res}
        if self.verbose:
            print("[league] anchor baselines @%dep: %s | gate: laika>%.2f, easy>=%.2f, stationary>=%.2f"
                  % (self.confirm_episodes, {k: round(v, 2) for k, v in self.baselines.items()},
                     self.baselines["laika"], self.baselines["easy_laika"] - self.easy_drop, self.stat_min))
        self._last_eval = 0

    def _evaluate(self):
        w = {o: r["win_rate"] for o, r in self._run(self.episodes).items()}   # cheap monitor eval
        self.model.save(self.latest_path)
        # cheap pre-filter: only pay for a confirm eval when laika improves and nothing obviously crashed
        candidate = (w.get("laika", 0) > self.baselines["laika"]
                     and w.get("easy_laika", 0) >= self.baselines["easy_laika"] - self.easy_drop - 0.05
                     and w.get("stationary", 0) >= self.stat_min - 0.05)
        promoted, conf = False, None
        if candidate:
            conf = {o: r["win_rate"] for o, r in self._run(self.confirm_episodes).items()}
            if self._gate(conf) and conf["laika"] > self.best_laika:
                self.best_laika = conf["laika"]
                self.model.save(self.best_path)
                promoted = True
        if self.verbose:
            cheap = " ".join("%s=%.2f" % (o[:4], w[o]) for o in self.eval_opponents)
            line = "[league] @%-7d cheap[%s]" % (self.num_timesteps, cheap)
            if conf is not None:
                cc = " ".join("%s=%.2f" % (o[:4], conf[o]) for o in self.eval_opponents)
                line += " | confirm@%d[%s]%s" % (self.confirm_episodes, cc,
                                                 " *PROMOTED" if promoted else " (gate fail)")
            print(line)
        self._export(w, conf, promoted)
        self._last_eval = int(self.num_timesteps)

    def _export(self, w, conf, promoted):
        status = {"timesteps": int(self.num_timesteps), "cheap": w, "confirm": conf,
                  "baselines": self.baselines, "best_laika": self.best_laika, "promoted": promoted}
        (self.eval_env.run_dir / "league_status.json").write_text(json.dumps(status, indent=2))

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.interval:
            self._evaluate()
        return True

    def _on_training_end(self):
        self._evaluate()


def main() -> None:
    args = parse_args()
    mix = {}
    for kv in args.mix.split(","):
        k, _, v = kv.partition("=")
        if k.strip():
            mix[k.strip()] = float(v)
    alloc = allocate(mix, args.n_envs)
    counts = {o: alloc.count(o) for o in dict.fromkeys(alloc)}

    anchor = args.anchor if args.anchor.is_absolute() else ROOT / args.anchor
    prefix = args.out_prefix
    best_path = ROOT / f"{prefix}_best.zip"
    latest_path = ROOT / f"{prefix}_latest.zip"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / "runs/league_stage1"
    run_dir.mkdir(parents=True, exist_ok=True)

    venv = TankVecEnv(
        num_envs=args.n_envs, arena_mode="survival", scenario=SCENARIO,
        opponents=alloc, spawn_powerups=True, max_steps=args.max_steps,
        base_seed=args.seed, seed_increment=True, reward=DUEL_REWARD,
    )
    env = VecMonitor(venv, filename=str(run_dir / "monitor.csv"))

    model = PPO.load(anchor, env=env, device=args.device)   # set_env resizes the rollout buffer to n_envs
    model.ent_coef = args.ent_coef                          # safe post-load (read in the loss)
    model.learning_rate = args.lr
    model.lr_schedule = lambda _p: args.lr                  # PPO.load otherwise ignores the new lr
    model.tensorboard_log = str(run_dir / "tb")
    assert model.observation_space.shape == (105,) and model.action_space.n == N_ACTIONS

    eval_env = TankEnv(
        arena_mode="survival", scenario=SCENARIO, opponent="laika", spawn_powerups=True,
        max_steps=args.max_steps, run_dir=run_dir, seed=args.eval_seed_base, reward=DUEL_REWARD,
    )

    print("=" * 78)
    print(f"LEAGUE Stage-1  anchor={anchor.name} (frozen)  branch -> {prefix}_best/_latest.zip")
    print(f"  env mix ({args.n_envs} envs): {counts}")
    print(f"  lr={args.lr} ent={args.ent_coef} steps={args.total_timesteps}")
    callback = LeagueCallback(
        eval_env, ["stationary", "easy_laika", "laika"],
        args.eval_interval, args.eval_episodes, args.confirm_episodes, args.eval_seed_base,
        best_path, latest_path, args.gate_easy_drop, args.gate_stationary_min,
    )
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
    finally:
        model.save(latest_path)
        eval_env.close()
        venv.close()
    if callback.best_laika >= 0:
        print(f"saved: {best_path.name} (best gated laika={callback.best_laika:.3f}), {latest_path.name}")
    else:
        print(f"NO gated improvement over the anchor -- anchor remains best. latest: {latest_path.name}")


if __name__ == "__main__":
    main()
