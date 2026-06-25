"""Train Level 1 PPO: open arena, stationary opponent, no power-ups."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
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
PHASE = "level1b_open_stationary_jitter"


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def make_status(model: PPO, phase: str, opponent: str, arena_mode: str) -> dict[str, Any]:
    episodes = list(model.ep_info_buffer or [])
    return {
        "timesteps": int(model.num_timesteps),
        "phase": phase,
        "opponent": opponent,
        "arenaMode": arena_mode,
        "ep_rew_mean": _mean_or_none([float(ep["r"]) for ep in episodes if "r" in ep]),
        "ep_len_mean": _mean_or_none([float(ep["l"]) for ep in episodes if "l" in ep]),
    }


class LiveExportCallback(BaseCallback):
    def __init__(self, run_dir: Path, export_interval: int, phase: str, opponent: str, arena_mode: str) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.export_interval = int(export_interval)
        self.phase = phase
        self.opponent = opponent
        self.arena_mode = arena_mode
        self._last_export = -1

    def _export(self) -> None:
        export_model(self.model, self.run_dir, make_status(self.model, self.phase, self.opponent, self.arena_mode))
        self._last_export = int(self.model.num_timesteps)

    def _on_training_start(self) -> None:
        self._export()

    def _on_step(self) -> bool:
        if self.model.num_timesteps - self._last_export >= self.export_interval:
            self._export()
        return True

    def _on_training_end(self) -> None:
        self._export()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=positive_int, default=300_000)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/ppo_level1b_open_stationary_jitter"))
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_level1b_open_stationary_jitter.zip"))
    parser.add_argument("--seed", type=int, default=1307)
    parser.add_argument("--arena-mode", choices=["open", "maze", "survival"], default="open")
    parser.add_argument("--scenario", default="battle",
                        choices=["battle", "fixed_moba", "moba1v1duel", "nav_powerup_poison", "nav_route_to_center"],
                        help="Map/scenario. fixed_moba = fixed symmetric combat arena; moba1v1duel adds "
                             "poison ring + duel mechanics (regen / 5-shot 2x / slow turn). Both need --arena-mode survival.")
    parser.add_argument("--opponent", default="stationary")
    parser.add_argument("--spawn-powerups", action="store_true")
    parser.add_argument("--shell-decay", action=argparse.BooleanOptionalAction, default=True,
                        help="CANONICAL 2x-launch, decay-to-zero shells (matches game_core default + demos). "
                             "--no-shell-decay only for the legacy constant-speed experiments.")
    parser.add_argument("--spawn-jitter", action=argparse.BooleanOptionalAction, default=True,
                        help="Level 1B open-arena spawn/heading jitter (default on; --no-spawn-jitter for 1A).")
    parser.add_argument("--seed-increment", action=argparse.BooleanOptionalAction, default=True,
                        help="Vary the episode seed each reset for diversity (default on; --no-seed-increment for 1A).")
    parser.add_argument("--randomize-seed", action="store_true",
                        help="Sample a random episode seed each reset instead of incrementing.")
    parser.add_argument("--n-envs", type=positive_int, default=1,
                        help="Parallel envs; >1 uses SubprocVecEnv (one node bridge each).")
    parser.add_argument("--max-steps", type=positive_int, default=500)
    parser.add_argument("--action-repeat", type=positive_int, default=2)
    parser.add_argument("--step-dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=positive_int, default=1024)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument("--export-interval", type=positive_int, default=10_000)
    parser.add_argument("--phase", default=None,
                        help="Status label; defaults to the run-dir name minus the 'ppo_' prefix.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def make_env_fn(rank: int, args: argparse.Namespace, run_dir: Path):
    """Factory for one TankEnv. Each parallel env gets a disjoint seed band
    (base + rank*10000) so SubprocVecEnv workers don't replay the same episodes."""
    def _init() -> Monitor:
        env = TankEnv(
            arena_mode=args.arena_mode,
            scenario=args.scenario,
            opponent=args.opponent,
            spawn_powerups=args.spawn_powerups,
            spawn_jitter=args.spawn_jitter,
            shell_decay=args.shell_decay,
            seed_increment=args.seed_increment,
            randomize_seed=args.randomize_seed,
            max_steps=args.max_steps,
            action_repeat=args.action_repeat,
            step_dt=args.step_dt,
            run_dir=run_dir,
            seed=args.seed + rank * 10000,
        )
        monitor_name = "monitor.csv" if rank == 0 else f"monitor_{rank}.csv"
        return Monitor(env, filename=str(run_dir / monitor_name))
    return _init


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    phase = args.phase or (run_dir.name[4:] if run_dir.name.startswith("ppo_") else run_dir.name)

    if args.n_envs > 1:
        # Batched single-process bridge: one node process steps all envs per
        # round-trip. Avoids the Windows SubprocVecEnv spawn/torch page-file blowup.
        from stable_baselines3.common.vec_env import VecMonitor
        try:
            from .tank_vec_env import TankVecEnv
        except ImportError:
            from tank_vec_env import TankVecEnv
        venv = TankVecEnv(
            num_envs=args.n_envs,
            arena_mode=args.arena_mode,
            scenario=args.scenario,
            opponent=args.opponent,
            spawn_powerups=args.spawn_powerups,
            spawn_jitter=args.spawn_jitter,
            shell_decay=args.shell_decay,
            seed_increment=args.seed_increment,
            randomize_seed=args.randomize_seed,
            max_steps=args.max_steps,
            action_repeat=args.action_repeat,
            step_dt=args.step_dt,
            base_seed=args.seed,
        )
        env = VecMonitor(venv, filename=str(run_dir / "monitor.csv"))
    else:
        env = make_env_fn(0, args, run_dir)()

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
    )

    export_model(model, run_dir, make_status(model, phase, args.opponent, args.arena_mode))
    callback = LiveExportCallback(run_dir, args.export_interval, phase, args.opponent, args.arena_mode)
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
    finally:
        model.save(model_path)
        export_model(model, run_dir, make_status(model, phase, args.opponent, args.arena_mode))
        env.close()


if __name__ == "__main__":
    main()
