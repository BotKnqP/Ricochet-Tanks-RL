"""Random-action smoke test for the Node-backed TankEnv."""

from __future__ import annotations

import json

import numpy as np

try:
    from .tank_env import TankEnv
except ImportError:
    from tank_env import TankEnv


def assert_obs(obs: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    assert obs.shape == expected_shape, f"bad obs shape: {obs.shape}, expected {expected_shape}"
    assert np.all(np.isfinite(obs)), "obs contains NaN or Inf"
    assert np.all(obs >= -1.000001) and np.all(obs <= 1.000001), "obs outside [-1, 1]"


def run_phase(label: str, *, steps: int = 2000, **env_kwargs: object) -> dict:
    rng = np.random.default_rng(1307)
    env = TankEnv(opponent="stationary", spawn_powerups=False, seed=1307, **env_kwargs)
    expected_shape = env.observation_space.shape
    obs, info = env.reset()
    assert_obs(obs, expected_shape)

    total_reward = 0.0
    episodes = 1
    terminations = 0
    truncations = 0
    last_info = info
    seeds_seen = {int(info.get("seed", -1))}

    try:
        for _ in range(steps):
            action = int(rng.integers(env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            assert_obs(obs, expected_shape)
            assert np.isfinite(reward), "reward is NaN or Inf"
            assert isinstance(terminated, bool), "terminated must be bool"
            assert isinstance(truncated, bool), "truncated must be bool"

            total_reward += reward
            last_info = info
            if terminated or truncated:
                terminations += int(terminated)
                truncations += int(truncated)
                episodes += 1
                obs, last_info = env.reset()
                assert_obs(obs, expected_shape)
                seeds_seen.add(int(last_info.get("seed", -1)))
    finally:
        env.close()

    return {
        "phase": label,
        "steps": steps,
        "obs_size": int(expected_shape[0]),
        "action_size": int(env.action_space.n),
        "episodes": int(episodes),
        "terminations": int(terminations),
        "truncations": int(truncations),
        "distinct_episode_seeds": len(seeds_seen),
        "spawn_jitter": bool(last_info.get("spawnJitter", False)),
        "arena_mode": str(last_info.get("arenaMode", "")),
        "total_reward": round(total_reward, 6),
    }


def main() -> None:
    # Level 1A: fixed open spawn, no seed variation -> every episode identical.
    level1a = run_phase("level1a_fixed", arena_mode="open", spawn_jitter=False, seed_increment=False)
    assert level1a["distinct_episode_seeds"] == 1, "Level 1A must reuse a single fixed seed"
    assert not level1a["spawn_jitter"], "Level 1A must not jitter"

    # Level 1B: jittered spawn + per-episode seed increment -> diverse episodes.
    level1b = run_phase("level1b_jitter", arena_mode="open", spawn_jitter=True, seed_increment=True)
    assert level1b["distinct_episode_seeds"] >= 2, "Level 1B must vary the seed across episodes"
    assert level1b["spawn_jitter"], "Level 1B must report jitter on"

    print(json.dumps({"ok": True, "level1a": level1a, "level1b": level1b}, indent=2))


if __name__ == "__main__":
    main()
