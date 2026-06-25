"""Lesson 1 PPO: nav_powerup_poison.

The agent learns to walk, avoid walls, grab a power-up, reach the safe zone and
survive 12s after the poison ring shrinks to the centre. No combat: red is
removed and the reward is the dedicated nav profile (game_core.js
NAV_REWARD_DEFAULTS) -- battle win/loss/hit rewards are not used.

Train:    python train/train_nav_powerup_poison.py --total-timesteps 500000
Evaluate: python train/evaluate_nav_powerup_poison.py --model-path models/nav_powerup_poison_v1_best.zip --episodes 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

try:
    from .export_policy import export_model
    from .tank_env import TankEnv
except ImportError:
    from export_policy import export_model
    from tank_env import TankEnv


ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=positive_int, default=500_000)
    p.add_argument("--run-dir", type=Path, default=Path("runs/nav_powerup_poison_v1"))
    p.add_argument("--model-path", type=Path, default=Path("models/nav_powerup_poison_v1.zip"))
    p.add_argument("--seed", type=int, default=1307)
    p.add_argument("--max-steps", type=positive_int, default=2200)
    p.add_argument("--action-repeat", type=positive_int, default=2)
    p.add_argument("--step-dt", type=float, default=1.0 / 30.0)
    # Longer horizon than the battle lessons -> larger gamma and rollout.
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=positive_int, default=2048)
    p.add_argument("--batch-size", type=positive_int, default=256)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.003)
    # Best-checkpoint evaluation (success_rate is the selection metric). Eval runs full
    # deterministic episodes and can rival training wall-clock; keep it infrequent.
    p.add_argument("--eval-interval", type=positive_int, default=50_000)
    p.add_argument("--eval-episodes", type=positive_int, default=8)
    p.add_argument("--eval-seed-base", type=int, default=800_000)
    p.add_argument("--n-envs", type=positive_int, default=8,
                   help="Parallel envs via the batched single-process Node bridge (default 8). Use 1 to debug.")
    p.add_argument("--device", default="cpu", help="MlpPolicy trains faster on CPU; pass cuda to force GPU.")
    return p.parse_args()


def _nav_env(seed: int, max_steps: int, action_repeat: int, step_dt: float, run_dir: Path) -> TankEnv:
    return TankEnv(
        arena_mode="survival",
        scenario="nav_powerup_poison",
        opponent="none",
        spawn_powerups=True,
        seed_increment=True,
        max_steps=max_steps,
        action_repeat=action_repeat,
        step_dt=step_dt,
        run_dir=run_dir,
        seed=seed,
    )


class NavBestCheckpoint(BaseCallback):
    """Periodically roll out deterministic episodes on held-out seeds, log
    success_rate, and keep both a `_best` (highest success_rate) and `_latest`
    checkpoint. Eval episodes are long (~112s of sim) so keep the interval coarse."""

    def __init__(self, interval: int, episodes: int, seed_base: int,
                 best_path: Path, latest_path: Path, max_steps: int, run_dir: Path, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.interval = int(interval)
        self.episodes = int(episodes)
        self.seed_base = int(seed_base)
        self.best_path = best_path
        self.latest_path = latest_path
        self.max_steps = int(max_steps)
        self.run_dir = run_dir
        self._last_eval = 0
        self.best_rate = -1.0

    def _evaluate(self) -> float:
        env = _nav_env(self.seed_base, self.max_steps, 2, 1.0 / 30.0, Path("."))
        successes = 0
        try:
            for i in range(self.episodes):
                obs, info = env.reset(seed=self.seed_base + i)
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _r, term, trunc, info = env.step(int(action))
                    done = term or trunc
                if info.get("navSuccess"):
                    successes += 1
        finally:
            env.close()
        rate = successes / max(1, self.episodes)
        self.model.save(self.latest_path)
        self.logger.record("nav/success_rate", rate)
        self.logger.record("nav/best_success_rate", max(self.best_rate, rate))
        if rate > self.best_rate:
            self.best_rate = rate
            self.model.save(self.best_path)
            if self.verbose:
                print(f"[nav] new best success_rate={rate:.3f} @ {self.num_timesteps} steps -> {self.best_path.name}")
        # Export the live policy for the browser watch mode (unchanged exporter).
        export_model(self.model, self.run_dir, {
            "timesteps": int(self.num_timesteps),
            "phase": "nav_powerup_poison_v1",
            "scenario": "nav_powerup_poison",
            "arenaMode": "survival",
            "opponent": "none",
            "success_rate": rate,
        })
        self._last_eval = int(self.num_timesteps)
        return rate

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.interval:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        self._evaluate()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    stem = model_path.stem
    for suffix in ("_latest", "_best"):   # tolerate a --model-path that already carries the suffix
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    best_path = model_path.with_name(stem + "_best.zip")
    latest_path = model_path.with_name(stem + "_latest.zip")

    if args.n_envs > 1:
        # Batched single-process bridge: one node process steps all envs per round-trip.
        from stable_baselines3.common.vec_env import VecMonitor
        try:
            from .tank_vec_env import TankVecEnv
        except ImportError:
            from tank_vec_env import TankVecEnv
        venv = TankVecEnv(
            num_envs=args.n_envs,
            arena_mode="survival",
            scenario="nav_powerup_poison",
            opponent="none",
            spawn_powerups=True,
            seed_increment=True,
            max_steps=args.max_steps,
            action_repeat=args.action_repeat,
            step_dt=args.step_dt,
            base_seed=args.seed,
        )
        env = VecMonitor(venv, filename=str(run_dir / "monitor.csv"))
    else:
        env = Monitor(
            _nav_env(args.seed, args.max_steps, args.action_repeat, args.step_dt, run_dir),
            filename=str(run_dir / "monitor.csv"),
            info_keywords=("navSuccess",),
        )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(run_dir / "tb"),
    )

    callback = NavBestCheckpoint(
        args.eval_interval, args.eval_episodes, args.eval_seed_base,
        best_path, latest_path, args.max_steps, run_dir,
    )
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
    finally:
        model.save(latest_path)   # _best and _latest are the artifacts; no redundant no-suffix dump
        env.close()
    print(f"saved: {latest_path.name}, {best_path.name} (best success_rate={callback.best_rate:.3f})")


if __name__ == "__main__":
    main()
