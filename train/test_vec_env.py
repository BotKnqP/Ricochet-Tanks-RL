"""Throughput + correctness check for the batched Node bridge (TankVecEnv).

Steps N envs with random actions and reports env-steps/sec, so you can compare
against the ~485 fps single-env baseline.
"""

from __future__ import annotations

import json
import time

import numpy as np

try:
    from .tank_vec_env import TankVecEnv
except ImportError:
    from tank_vec_env import TankVecEnv


def main() -> None:
    num_envs = 8
    steps = 1500
    env = TankVecEnv(
        num_envs=num_envs,
        arena_mode="open",
        opponent="turret",
        spawn_jitter=True,
        seed_increment=True,
        max_steps=500,
        base_seed=1307,
    )
    obs_dim = env.observation_space.shape[0]
    obs = env.reset()
    assert obs.shape == (num_envs, obs_dim), f"bad reset shape {obs.shape}"
    assert np.all(np.isfinite(obs)), "reset obs not finite"

    rng = np.random.default_rng(0)
    episodes = 0
    seen_seeds: set[int] = set()
    t0 = time.time()
    try:
        for _ in range(steps):
            actions = rng.integers(env.action_space.n, size=num_envs)
            obs, rewards, dones, infos = env.step(actions)
            assert obs.shape == (num_envs, obs_dim)
            assert np.all(np.isfinite(obs)), "step obs not finite"
            assert np.all(np.isfinite(rewards)), "reward not finite"
            for i, done in enumerate(dones):
                if done:
                    episodes += 1
                    assert "terminal_observation" in infos[i], "missing terminal_observation"
                    assert infos[i]["terminal_observation"].shape == (obs_dim,)
    finally:
        elapsed = time.time() - t0
        env.close()

    env_steps = steps * num_envs
    print(json.dumps({
        "ok": True,
        "num_envs": num_envs,
        "env_steps": env_steps,
        "wall_seconds": round(elapsed, 2),
        "fps_env_steps_per_sec": round(env_steps / elapsed),
        "episodes_finished": episodes,
    }, indent=2))


if __name__ == "__main__":
    main()
