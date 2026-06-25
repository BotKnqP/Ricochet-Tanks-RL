"""SB3 VecEnv for SELF-PLAY: each env's opponent is either a script OR a frozen policy.

Backed by rl_bridge_selfplay.js, which returns obs1 (red perspective) every step. moba1v1duel is
symmetric, so a policy trained as blue (obs0) plays red directly from obs1. Per-step the wrapper
runs each policy-controlled env's frozen opponent on its cached obs1 and passes the resulting red
action back as `opponentActions`; script envs fall back to the bridge's scripted control. This is
what lets a single rollout MIX scripts and a league of past selves (anti-forgetting + self-play).

Opponent spec per env (`opponents` list): a str = scripted opponent; an int = index into
`opp_models` (a frozen SB3 policy). `opp_models` are loaded once and never trained.
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
BRIDGE = ROOT / "train" / "rl_bridge_selfplay.js"


class TankSelfPlayVecEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        opponents: Sequence[Any],          # per-env: str (script) or int (index into opp_models)
        opp_models: Sequence[Any] | None = None,   # frozen SB3 policies (predict only)
        opp_deterministic: bool = False,   # sample the opponent policy (diversity) vs argmax
        arena_mode: str = "survival",
        scenario: str = "moba1v1duel",
        spawn_powerups: bool = True,
        shell_decay: bool = True,
        ruleset: str = "survival_v1",      # "survival_v2" -> HP x2 / slower regen / random spawn
        spawn_mode: str | None = None,     # override spawnMode (fixed/half_random/full_random)
        tank_max_hp: float | None = None,  # override tankMaxHp
        seed_increment: bool = True,
        randomize_seed: bool = False,
        max_steps: int = 900,
        action_repeat: int = 2,
        step_dt: float = 1.0 / 30.0,
        base_seed: int = 1307,
        seed_stride: int = 10_000,
        reward: dict[str, float] | None = None,
        node_bin: str = "node",
    ) -> None:
        if len(opponents) != num_envs:
            raise ValueError(f"opponents must have {num_envs} entries, got {len(opponents)}")
        self.opp_models = list(opp_models or [])
        self.opp_deterministic = bool(opp_deterministic)
        # split spec: script name for the bridge init (policy envs get a harmless placeholder that
        # is always overridden by opponentActions), and a (kind, payload) per env for stepping.
        self._spec: list[tuple[str, Any]] = []
        init_opponents: list[str] = []
        for o in opponents:
            if isinstance(o, (int, np.integer)):
                idx = int(o)
                if not (0 <= idx < len(self.opp_models)):
                    raise ValueError(f"policy opponent index {idx} out of range ({len(self.opp_models)} models)")
                self._spec.append(("policy", idx))
                init_opponents.append("stationary")     # placeholder, never used (overridden)
            else:
                self._spec.append(("script", str(o)))
                init_opponents.append(str(o))

        self.node_bin = node_bin
        self._closed = False
        self._actions: list[int] | None = None
        self._obs1: np.ndarray | None = None
        self._proc = subprocess.Popen(
            [node_bin, str(BRIDGE)], cwd=str(ROOT),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        core_cfg: dict[str, Any] = {
            "arenaMode": arena_mode, "scenario": scenario, "spawnPowerups": bool(spawn_powerups),
            "shellDecay": bool(shell_decay), "maxSteps": int(max_steps),
            "actionRepeat": int(action_repeat), "stepDt": float(step_dt),
        }
        if ruleset and ruleset != "survival_v1": core_cfg["ruleset"] = str(ruleset)
        if spawn_mode is not None: core_cfg["spawnMode"] = str(spawn_mode)
        if tank_max_hp is not None: core_cfg["tankMaxHp"] = float(tank_max_hp)
        if reward is not None:
            core_cfg["reward"] = reward
        init_payload: dict[str, Any] = {
            "cmd": "init", "n": int(num_envs), "baseSeed": int(base_seed), "seedStride": int(seed_stride),
            "seedIncrement": bool(seed_increment), "randomizeSeed": bool(randomize_seed),
            "opponent": "stationary", "opponents": init_opponents, "core": core_cfg,
        }
        init = self._request(init_payload)
        obs_size = int(init["obsSize"]); action_size = int(init["actionSize"])
        observation_space = spaces.Box(-1.0, 1.0, shape=(obs_size,), dtype=np.float32)
        action_space = spaces.Discrete(action_size)
        super().__init__(int(num_envs), observation_space, action_space)
        self.render_mode = None
        assert obs_size == 105 and action_size == 18, f"obs/action invariant broken: {obs_size}/{action_size}"
        n_pol = sum(1 for k, _ in self._spec if k == "policy")
        print(f"[env] TankSelfPlayVecEnv n={num_envs} scenario={scenario} arena={arena_mode} "
              f"script_envs={num_envs - n_pol} policy_envs={n_pol} models={len(self.opp_models)} "
              f"opp_det={self.opp_deterministic} shell_decay={bool(shell_decay)} base_seed={base_seed}")

    # ------------------------------------------------------------------ IPC
    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("TankSelfPlayVecEnv is closed")
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
            raise RuntimeError(f"selfplay bridge exited without a response: {err.strip()}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"selfplay bridge error: {response['error']}")
        return response

    def _to_obs(self, raw) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise RuntimeError("selfplay obs contains NaN or Inf")
        return np.clip(arr, -1.0, 1.0)

    def _opponent_actions(self) -> list[Any]:
        """Red action per env: int for policy envs (from cached obs1), None for script envs."""
        out: list[Any] = [None] * self.num_envs
        if self._obs1 is None:
            return out
        # batch by model index to amortize predict()
        by_model: dict[int, list[int]] = {}
        for i, (kind, payload) in enumerate(self._spec):
            if kind == "policy":
                by_model.setdefault(int(payload), []).append(i)
        for midx, idxs in by_model.items():
            batch = self._obs1[idxs]
            acts, _ = self.opp_models[midx].predict(batch, deterministic=self.opp_deterministic)
            acts = np.asarray(acts).reshape(-1)
            for j, env_i in enumerate(idxs):
                out[env_i] = int(acts[j])
        return out

    # ------------------------------------------------------------- VecEnv API
    def reset(self) -> np.ndarray:
        resp = self._request({"cmd": "reset"})
        self._obs1 = self._to_obs(resp["obs1"])
        return self._to_obs(resp["obs"])

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = [int(a) for a in np.asarray(actions).reshape(-1)]

    def step_wait(self):
        opp_actions = self._opponent_actions()
        resp = self._request({"cmd": "step", "actions": self._actions, "opponentActions": opp_actions})
        obs = self._to_obs(resp["obs"])
        self._obs1 = self._to_obs(resp["obs1"])
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

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --------- abstract stubs (sufficient for MlpPolicy PPO training) ---------
    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * self.num_envs

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        n = self.num_envs if indices is None else len(self._indices(indices))
        return [None] * n

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        return None

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list[Any]:
        raise NotImplementedError(f"env_method({method_name}) not supported")

    def _indices(self, indices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)
