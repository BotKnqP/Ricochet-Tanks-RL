"""SB3 VecEnv backed by a single batched Node bridge (rl_bridge_vec.js).

One node process steps all N games per round-trip, so the Python<->Node latency
is paid once per batch instead of once per env. Stays in ONE python + ONE node
process, avoiding the Windows SubprocVecEnv spawn/torch page-file blowup.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "train" / "rl_bridge_vec.js"


class TankVecEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        arena_mode: str = "open",
        opponent: str = "stationary",
        opponents: Sequence[str] | None = None,
        spawn_powerups: bool = False,
        spawn_jitter: bool = False,
        shell_decay: bool = True,   # CANONICAL physics (matches game_core default + demos)
        scenario: str = "battle",
        seed_increment: bool = True,
        randomize_seed: bool = False,
        max_steps: int = 500,
        action_repeat: int = 2,
        step_dt: float = 1.0 / 30.0,
        base_seed: int = 1307,
        seed_stride: int = 10_000,
        reward: dict[str, float] | None = None,
        node_bin: str = "node",
    ) -> None:
        self.node_bin = node_bin
        self._closed = False
        self._actions: list[int] | None = None
        self._proc = subprocess.Popen(
            [node_bin, str(BRIDGE)],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        core_cfg: dict[str, Any] = {
            "arenaMode": arena_mode,
            "scenario": scenario,
            "spawnPowerups": bool(spawn_powerups),
            "spawnJitter": bool(spawn_jitter),
            "shellDecay": bool(shell_decay),
            "maxSteps": int(max_steps),
            "actionRepeat": int(action_repeat),
            "stepDt": float(step_dt),
        }
        if reward is not None:
            core_cfg["reward"] = reward

        init_payload: dict[str, Any] = {
            "cmd": "init",
            "n": int(num_envs),
            "baseSeed": int(base_seed),
            "seedStride": int(seed_stride),
            "seedIncrement": bool(seed_increment),
            "randomizeSeed": bool(randomize_seed),
            "opponent": opponent,
            "core": core_cfg,
        }
        if opponents is not None:
            init_payload["opponents"] = [str(o) for o in opponents]   # league: per-env fixed opponent
        init = self._request(init_payload)
        obs_size = int(init["obsSize"])
        action_size = int(init["actionSize"])
        observation_space = spaces.Box(-1.0, 1.0, shape=(obs_size,), dtype=np.float32)
        action_space = spaces.Discrete(action_size)
        super().__init__(int(num_envs), observation_space, action_space)
        self.render_mode = None
        assert obs_size == 105 and action_size == 18, f"obs/action invariant broken: {obs_size}/{action_size}"
        _opp = list(opponents) if opponents is not None else opponent
        print(f"[env] TankVecEnv n={num_envs} scenario={scenario} arena={arena_mode} opponents={_opp} "
              f"shell_decay={bool(shell_decay)} spawn_powerups={bool(spawn_powerups)} obs={obs_size} "
              f"action={action_size} base_seed={base_seed}")

    # ------------------------------------------------------------------ IPC
    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("TankVecEnv is closed")
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            err = ""
            if self._proc.stderr is not None and self._proc.poll() is not None:
                try:
                    err = self._proc.stderr.read()
                except Exception:
                    err = ""
            raise RuntimeError(f"vec bridge exited without a response: {err.strip()}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"vec bridge error: {response['error']}")
        return response

    def _to_obs(self, raw: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise RuntimeError("vec obs contains NaN or Inf")
        return np.clip(arr, -1.0, 1.0)

    # ------------------------------------------------------------- VecEnv API
    def reset(self) -> np.ndarray:
        return self._to_obs(self._request({"cmd": "reset"})["obs"])

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = [int(a) for a in np.asarray(actions).reshape(-1)]

    def step_wait(self):
        resp = self._request({"cmd": "step", "actions": self._actions})
        obs = self._to_obs(resp["obs"])
        rewards = np.asarray(resp["reward"], dtype=np.float32)
        dones = np.asarray(resp["done"], dtype=bool)
        infos: list[dict[str, Any]] = []
        for info in resp["info"]:
            info = dict(info)
            if "terminal_observation" in info:
                info["terminal_observation"] = self._to_obs([info["terminal_observation"]])[0]
            infos.append(info)
        return obs, rewards, dones, infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # --------- abstract stubs (sufficient for MlpPolicy PPO training) ---------
    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * self.num_envs

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        n = self.num_envs if indices is None else len(self._indices(indices))
        if attr_name == "render_mode":
            return [None] * n
        return [None] * n

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        return None

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list[Any]:
        raise NotImplementedError(f"env_method({method_name}) not supported by TankVecEnv")

    def _indices(self, indices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)
