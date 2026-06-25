"""Task B: league PPO v2 -- anchored, safe RL handoff (the AlphaStar-faithful version).

Every prior plain-PPO attempt eroded the warm start (cold critic + drift off the learnable
region). Diagnostics established the obs/action are fine and the teacher is ~98% learnable on
the expert distribution; the failure was closed-loop drift. v2 addresses that directly:

  1. CRITIC WARM-UP  -- freeze the actor, train only the value head for `--warmup-steps` so the
     advantage estimates are sane before the actor ever moves (no cold-critic blow-up).
  2. BC/CE ANCHOR    -- after each PPO update, a few CE steps pull the policy back toward the
     demo actions, so RL explores but cannot drift out of the ~98%-learnable region.
  3. MIXED POOL      -- per-env fixed opponents, so every rollout batches all opponents (anti-forgetting).
  4. FIXED 100-ep GATE -- promote `_best` only if laika improves while easy/stationary hold (reused).
  5. The anchor (bc_dagger_moba_canon_v2) is NEVER overwritten.

  python train/league_ppo_v2.py --anchor models/bc_dagger_moba_canon_v2.zip \
      --data-glob "data/expert_demos/moba_perop/*.jsonl" --total-timesteps 300000 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

try:
    from .tank_env import TankEnv
    from .tank_vec_env import TankVecEnv
    from .train_moba1v1duel import DUEL_REWARD
    from .train_bc import N_ACTIONS, load_demos
    from .league_stage1 import LeagueCallback, allocate
except ImportError:
    from tank_env import TankEnv
    from tank_vec_env import TankVecEnv
    from train_moba1v1duel import DUEL_REWARD
    from train_bc import N_ACTIONS, load_demos
    from league_stage1 import LeagueCallback, allocate

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "moba1v1duel"


class AnchoredPPO(PPO):
    """PPO whose train() = standard update + a BC/CE anchor toward the demo actions."""

    def set_anchor(self, demo_X, demo_Y, ce_steps, lambda_ce, batch, seed):
        self.demo_X = demo_X
        self.demo_Y = demo_Y
        self.ce_steps = int(ce_steps)
        self._ce_target = int(ce_steps)      # restored after critic warm-up
        self.lambda_ce = float(lambda_ce)
        self.ce_batch = int(batch)
        self._ce_rng = np.random.default_rng(seed)

    def train(self):
        super().train()                      # standard PPO update (actor frozen during warm-up)
        if getattr(self, "ce_steps", 0) <= 0:
            return
        self.policy.set_training_mode(True)
        for _ in range(self.ce_steps):
            idx = self._ce_rng.integers(0, len(self.demo_X), self.ce_batch)
            ot, _ = self.policy.obs_to_tensor(self.demo_X[idx])
            logits = self.policy.get_distribution(ot).distribution.logits
            tgt = torch.as_tensor(self.demo_Y[idx], dtype=torch.long, device=logits.device)
            loss = self.lambda_ce * F.cross_entropy(logits, tgt)
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()


def set_actor_frozen(model, frozen: bool):
    """Freeze/unfreeze the ACTOR (policy branch + action head); critic stays trainable."""
    p = model.policy
    mods = [p.mlp_extractor.policy_net, p.action_net]
    for m in mods:
        for param in m.parameters():
            param.requires_grad = not frozen


class CriticWarmup(BaseCallback):
    """Freeze the actor (value-only) for `warmup_steps`, then unfreeze and turn the CE-anchor on."""

    def __init__(self, warmup_steps, verbose=1):
        super().__init__(verbose)
        self.warmup_steps = int(warmup_steps)
        self._unfrozen = False

    def _on_training_start(self):
        set_actor_frozen(self.model, True)
        self.model.ce_steps = 0
        if self.verbose:
            print(f"[warmup] actor FROZEN (value-only), no CE, for {self.warmup_steps} steps")

    def _on_step(self) -> bool:
        if not self._unfrozen and self.num_timesteps >= self.warmup_steps:
            set_actor_frozen(self.model, False)
            self.model.ce_steps = self.model._ce_target
            self._unfrozen = True
            if self.verbose:
                print(f"[warmup] actor UNFROZEN @ {self.num_timesteps}; CE-anchor on "
                      f"(ce_steps={self.model.ce_steps}, lambda={self.model.lambda_ce})")
        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", type=Path, default=Path("models/bc_dagger_moba_canon_v2.zip"))
    p.add_argument("--data-glob", default="data/expert_demos/moba_perop/*.jsonl")
    p.add_argument("--out-prefix", default="models/league_ppo_v2")
    p.add_argument("--mix", default="laika=0.30,easy_laika=0.25,stationary=0.15,laika-aggressive-pro=0.15,laika-aggressive=0.15")
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--warmup-steps", type=int, default=40_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--clip-range", type=float, default=0.05)
    p.add_argument("--ent-coef", type=float, default=0.0005)
    p.add_argument("--ce-steps", type=int, default=8, help="CE-anchor mini-batches after each PPO update.")
    p.add_argument("--lambda-ce", type=float, default=1.0)
    p.add_argument("--ce-batch", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--eval-interval", type=int, default=30_000)
    p.add_argument("--eval-episodes", type=int, default=25)
    p.add_argument("--confirm-episodes", type=int, default=100)
    p.add_argument("--gate-easy-drop", type=float, default=0.10)
    p.add_argument("--gate-stationary-min", type=float, default=0.50)
    p.add_argument("--eval-seed-base", type=int, default=300_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1307)
    return p.parse_args()


def main():
    args = parse_args()
    mix = {}
    for kv in args.mix.split(","):
        k, _, v = kv.partition("=")
        if k.strip():
            mix[k.strip()] = float(v)
    alloc = allocate(mix, args.n_envs)
    counts = {o: alloc.count(o) for o in dict.fromkeys(alloc)}

    anchor = args.anchor if args.anchor.is_absolute() else ROOT / args.anchor
    best_path = ROOT / f"{args.out_prefix}_best.zip"
    latest_path = ROOT / f"{args.out_prefix}_latest.zip"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / "runs/league_ppo_v2"
    run_dir.mkdir(parents=True, exist_ok=True)

    # CE-anchor demos (true-physics moba_perop)
    files = sorted(__import__("glob").glob(str(ROOT / args.data_glob)) or __import__("glob").glob(args.data_glob))
    obs_list, act_list, _ = load_demos(
        files, "good_wins", frozenset([SCENARIO]),
        frozenset(["laika", "easy_laika", "stationary"]),
        frozenset(["laika-aggressive", "laika-aggressive-pro", "pro"]))
    demo_X = np.asarray(obs_list, dtype=np.float32)
    demo_Y = np.asarray(act_list, dtype=np.int64)

    venv = TankVecEnv(num_envs=args.n_envs, arena_mode="survival", scenario=SCENARIO,
                      opponents=alloc, spawn_powerups=True, max_steps=args.max_steps,
                      base_seed=args.seed, seed_increment=True, reward=DUEL_REWARD)
    env = VecMonitor(venv, filename=str(run_dir / "monitor.csv"))

    model = AnchoredPPO.load(anchor, env=env, device=args.device)
    model.learning_rate = args.lr
    model.lr_schedule = lambda _p: args.lr
    model.clip_range = lambda _p: args.clip_range
    model.ent_coef = args.ent_coef
    model.tensorboard_log = str(run_dir / "tb")
    model.set_anchor(demo_X, demo_Y, args.ce_steps, args.lambda_ce, args.ce_batch, args.seed)
    assert model.observation_space.shape == (105,) and model.action_space.n == N_ACTIONS

    eval_env = TankEnv(arena_mode="survival", scenario=SCENARIO, opponent="laika", spawn_powerups=True,
                       max_steps=args.max_steps, run_dir=run_dir, seed=args.eval_seed_base, reward=DUEL_REWARD)

    print("=" * 80)
    print(f"LEAGUE PPO v2 (anchored)  anchor={anchor.name} (frozen)  branch -> {args.out_prefix}_best/_latest")
    print(f"  pool({args.n_envs} envs)={counts}  demos={len(demo_X)}")
    print(f"  lr={args.lr} clip={args.clip_range} ent={args.ent_coef} warmup={args.warmup_steps} "
          f"ce_steps={args.ce_steps} lambda_ce={args.lambda_ce} steps={args.total_timesteps}")
    callback = [
        CriticWarmup(args.warmup_steps),
        LeagueCallback(eval_env, ["stationary", "easy_laika", "laika"],
                       args.eval_interval, args.eval_episodes, args.confirm_episodes, args.eval_seed_base,
                       best_path, latest_path, args.gate_easy_drop, args.gate_stationary_min),
    ]
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
    finally:
        model.save(latest_path)
        eval_env.close()
        venv.close()
    print(f"done. latest -> {latest_path.name}; best (if any gated promotion) -> {best_path.name}")


if __name__ == "__main__":
    main()
