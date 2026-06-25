"""moba1v1duel PPO: combat on the fixed moba map + poison ring, vs a scripted opponent.

Stage-1 combat curriculum: warm-start a poison-run model and learn to beat easy_laika
(a heavily-dampened laika), then later step up to turret / laika.

Train:    python train/train_moba1v1duel.py --opponent easy_laika --resume models/moba_poison_run_v1_best.zip --total-timesteps 200000 --n-envs 8
Evaluate: python train/evaluate_moba1v1duel.py --model-path models/moba1v1duel_vs_easy_laika_best.zip --opponent easy_laika --episodes 100
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
SCENARIO = "moba1v1duel"

# Battle reward profile for the duel (stronger win/loss magnitudes; no fancy shaping
# yet — we watch win_rate / hits / timeout first, per the curriculum plan).
DUEL_REWARD = {
    "win": 3.0, "loss": -3.0, "draw": -0.5, "winBySelfHit": 0.3, "timeoutPenalty": 1.0,
    "hit": 0.35, "hitShield": 0.18, "selfHit": -0.20, "selfHitShield": -0.12, "selfDefeat": -0.5,
    "powerup": 0.08, "timePenalty": 0.001,
    "backwardPenalty": 0.0, "approachCoef": 0.0, "aimBonus": 0.0,
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=positive_int, default=200_000)
    p.add_argument("--run-dir", type=Path, default=Path("runs/moba1v1duel_vs_easy_laika_v1"))
    p.add_argument("--model-path", type=Path, default=Path("models/moba1v1duel_vs_easy_laika.zip"))
    p.add_argument("--opponent", default="easy_laika",
                   help="Scripted opponent: easy_laika (stage 1) / turret / laika / laika-aggressive / stationary.")
    p.add_argument("--seed", type=int, default=1307)
    p.add_argument("--max-steps", type=positive_int, default=1800)
    p.add_argument("--action-repeat", type=positive_int, default=2)
    p.add_argument("--step-dt", type=float, default=1.0 / 30.0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=positive_int, default=2048)
    p.add_argument("--batch-size", type=positive_int, default=256)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--eval-interval", type=positive_int, default=50_000)
    p.add_argument("--eval-episodes", type=positive_int, default=30)
    p.add_argument("--eval-seed-base", type=int, default=900_000)
    p.add_argument("--n-envs", type=positive_int, default=8)
    p.add_argument("--spawn-powerups", action=argparse.BooleanOptionalAction, default=True,
                   help="Spawn the fixed symmetric power-ups (default on).")
    p.add_argument("--resume", type=Path, default=None,
                   help="Warm-start from a .zip (e.g. models/moba_poison_run_v1_best.zip). OBS_SIZE must still be 101.")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _duel_env(seed, max_steps, action_repeat, step_dt, run_dir, opponent, spawn_powerups) -> TankEnv:
    return TankEnv(
        arena_mode="survival", scenario=SCENARIO, opponent=opponent,
        spawn_powerups=spawn_powerups, seed_increment=True,
        max_steps=max_steps, action_repeat=action_repeat, step_dt=step_dt,
        run_dir=run_dir, seed=seed, reward=DUEL_REWARD,
    )


# ACTION_TABLE indices with fire=True (idle + throttle{-1,0,1} x turn{-1,0,1} x fire{F,T}).
FIRE_ACTIONS = frozenset({2, 4, 6, 8, 9, 11, 13, 15, 17})


class DuelBestCheckpoint(BaseCallback):
    """Roll out deterministic episodes on held-out seeds; log win_rate + combat metrics
    (fire%, hits_dealt/taken, self_hits, final/opponent health, length), optionally probe a
    held-out opponent for forgetting, keep `_best` (win_rate, tie-break lower avg_elapsed) +
    `_latest`, and export live_*."""

    def __init__(self, interval, episodes, seed_base, best_path, latest_path,
                 max_steps, run_dir, opponent, spawn_powerups, probe_opponent=None,
                 probe_episodes=12, verbose=1):
        super().__init__(verbose)
        self.interval = int(interval)
        self.episodes = int(episodes)
        self.seed_base = int(seed_base)
        self.best_path = best_path
        self.latest_path = latest_path
        self.max_steps = int(max_steps)
        self.run_dir = run_dir
        self.opponent = opponent
        self.spawn_powerups = spawn_powerups
        self.probe_opponent = probe_opponent           # held-out opponent for a forgetting check
        self.probe_episodes = int(probe_episodes)
        self._last_eval = 0
        self.best_key = (-1.0, 0.0)   # (win_rate, -avg_elapsed)

    def _rollout(self, opponent, episodes):
        """Deterministic rollout vs `opponent`; return win_rate + combat metrics from info."""
        env = _duel_env(self.seed_base, self.max_steps, 2, 1.0 / 30.0, Path("."), opponent, self.spawn_powerups)
        wins = 0
        elapsed = length = hd = ht = sh = fh = oh = 0.0
        fire = total = 0
        try:
            for i in range(episodes):
                obs, info = env.reset(seed=self.seed_base + i)
                done, steps = False, 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    a = int(action)
                    if a in FIRE_ACTIONS:
                        fire += 1
                    total += 1
                    steps += 1
                    obs, _r, term, trunc, info = env.step(a)
                    done = term or trunc
                if info.get("result") == "win":
                    wins += 1
                elapsed += float(info.get("elapsed", 0.0))
                length += steps
                hd += float(info.get("hitsDealt", 0.0)); ht += float(info.get("hitsTaken", 0.0))
                sh += float(info.get("selfHits", 0.0))
                fh += float(info.get("learnerHealth", 0.0)); oh += float(info.get("opponentHealth", 0.0))
        finally:
            env.close()
        n = max(1, episodes)
        return {
            "win_rate": wins / n, "avg_elapsed": elapsed / n, "avg_length": length / n,
            "fire_pct": 100.0 * fire / max(1, total),
            "hits_dealt": hd / n, "hits_taken": ht / n, "self_hits": sh / n,
            "final_health": fh / n, "opponent_final_health": oh / n,
        }

    def _status(self, m, probe, live="latest"):
        s = {
            "timesteps": int(self.num_timesteps), "phase": self.run_dir.name,
            "scenario": SCENARIO, "arenaMode": "survival", "opponent": self.opponent,
            "live": live, "best_win_rate": self.best_key[0], **m,
        }
        if probe is not None:
            s["probe_opponent"] = self.probe_opponent
            s["probe_win_rate"] = probe["win_rate"]
        return s

    def _evaluate(self) -> float:
        m = self._rollout(self.opponent, self.episodes)
        probe = self._rollout(self.probe_opponent, self.probe_episodes) if self.probe_opponent else None
        self.model.save(self.latest_path)
        for k, v in m.items():
            self.logger.record(f"duel/{k}", v)
        if probe is not None:
            self.logger.record(f"duel/probe_{self.probe_opponent}_win", probe["win_rate"])
        key = (m["win_rate"], -m["avg_elapsed"])
        is_best = key > self.best_key
        if is_best:
            self.best_key = key
            self.model.save(self.best_path)
        if self.verbose:
            probe_s = (" || %s win=%.2f" % (self.probe_opponent, probe["win_rate"])) if probe else ""
            print("[duel] @%-7d %s win=%.3f fire%%=%.1f hd=%.2f sh=%.2f ht=%.2f len=%.0f my_hp=%.2f opp_hp=%.2f%s%s"
                  % (self.num_timesteps, self.opponent, m["win_rate"], m["fire_pct"], m["hits_dealt"],
                     m["self_hits"], m["hits_taken"], m["avg_length"], m["final_health"],
                     m["opponent_final_health"], probe_s, " *BEST" if is_best else ""))
        export_model(self.model, self.run_dir, self._status(m, probe))
        self._last_eval = int(self.num_timesteps)
        return m["win_rate"]

    def _on_training_start(self) -> None:
        # so live_status exists at t=0 (minimal, before the first real eval)
        export_model(self.model, self.run_dir,
                     {"timesteps": 0, "phase": self.run_dir.name, "live": "latest", "win_rate": 0.0})

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
        from stable_baselines3.common.vec_env import VecMonitor
        try:
            from .tank_vec_env import TankVecEnv
        except ImportError:
            from tank_vec_env import TankVecEnv
        venv = TankVecEnv(
            num_envs=args.n_envs, arena_mode="survival", scenario=SCENARIO,
            opponent=args.opponent, spawn_powerups=args.spawn_powerups, seed_increment=True,
            max_steps=args.max_steps, action_repeat=args.action_repeat, step_dt=args.step_dt,
            base_seed=args.seed, reward=DUEL_REWARD,
        )
        env = VecMonitor(venv, filename=str(run_dir / "monitor.csv"))
    else:
        env = Monitor(
            _duel_env(args.seed, args.max_steps, args.action_repeat, args.step_dt, run_dir, args.opponent, args.spawn_powerups),
            filename=str(run_dir / "monitor.csv"),
        )

    if args.resume:
        resume_path = args.resume if args.resume.is_absolute() else ROOT / args.resume
        model = PPO.load(resume_path, env=env, device=args.device)
        model.tensorboard_log = str(run_dir / "tb")
        model.ent_coef = args.ent_coef     # re-tune exploration for the combat phase
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _progress_remaining: args.learning_rate   # PPO.load otherwise ignores --learning-rate
        print(f"resumed from {resume_path} (lr={args.learning_rate}, ent_coef={args.ent_coef})")
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size,
            gamma=args.gamma, gae_lambda=args.gae_lambda, ent_coef=args.ent_coef,
            verbose=1, seed=args.seed, device=args.device, tensorboard_log=str(run_dir / "tb"),
        )

    callback = DuelBestCheckpoint(
        args.eval_interval, args.eval_episodes, args.eval_seed_base,
        best_path, latest_path, args.max_steps, run_dir, args.opponent, args.spawn_powerups,
        probe_opponent=("stationary" if args.opponent != "stationary" else "easy_laika"),
    )
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
    finally:
        model.save(latest_path)
        env.close()
    print(f"saved: {latest_path.name}, {best_path.name} (best win_rate={callback.best_key[0]:.3f})")


if __name__ == "__main__":
    main()
