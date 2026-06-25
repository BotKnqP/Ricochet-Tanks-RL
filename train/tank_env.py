"""Gymnasium wrapper for the Node-backed Ricochet Tanks core."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "train" / "rl_bridge.js"


class TankEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        arena_mode: str = "open",
        opponent: str = "stationary",
        scenario: str = "battle",
        seed: int = 1307,
        max_steps: int = 500,
        node_bin: str = "node",
        spawn_powerups: bool = False,
        spawn_jitter: bool = False,
        random_turret: bool = False,
        expert: str | None = None,
        shell_decay: bool = True,   # CANONICAL physics: matches game_core default + demos + eval_script_bot
        ruleset: str = "survival_v1",      # "survival_v2" -> HP x2 / slower regen / random spawn
        spawn_mode: str | None = None,     # override spawnMode (fixed/half_random/full_random)
        tank_max_hp: float | None = None,  # override tankMaxHp
        seed_increment: bool = False,
        randomize_seed: bool = False,
        action_repeat: int = 2,
        step_dt: float = 1.0 / 30.0,
        run_dir: str | Path = "runs/ppo_level1_open_stationary",
        reward: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        self.arena_mode = arena_mode
        self.opponent = opponent
        self.scenario = str(scenario)
        self.seed_value = int(seed)
        self.max_steps = int(max_steps)
        self.spawn_powerups = bool(spawn_powerups)
        # Episode diversity (Level 1B). Defaults off so Level 1A's fixed-target
        # behaviour is unchanged; the trainer opts in explicitly.
        self.spawn_jitter = bool(spawn_jitter)
        # Shooting-lab: random turret spawn each episode (open arena only). Off by default.
        self.random_turret = bool(random_turret)
        # DAgger oracle: if set, reset/step responses carry the expert's action for the
        # observed state (info["expertAction"]) so a learner rollout can be expert-labelled.
        self.expert = str(expert) if expert else None
        # Experimental 2x-then-decay shells. Off by default so RL training keeps the
        # original constant-speed physics this round; opt in explicitly to train on it.
        self.shell_decay = bool(shell_decay)
        self.ruleset = str(ruleset)
        self.spawn_mode = str(spawn_mode) if spawn_mode is not None else None
        self.tank_max_hp = float(tank_max_hp) if tank_max_hp is not None else None
        self.seed_increment = bool(seed_increment)
        self.randomize_seed = bool(randomize_seed)
        self.episode_index = 0
        self._seed_rng = np.random.default_rng(self.seed_value)
        self.action_repeat = int(action_repeat)
        self.step_dt = float(step_dt)
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_absolute():
            self.run_dir = ROOT / self.run_dir
        self.reward = dict(reward) if reward is not None else None
        self.node_bin = node_bin
        self._proc: subprocess.Popen[str] | None = None
        self._closed = False
        self.obs_size: int | None = None
        self.action_size: int | None = None

        self._start_bridge()
        probe = self._request(self._reset_payload(self.seed_value))
        info = dict(probe.get("info", {}))
        self.obs_size = int(info["obsSize"])
        self.action_size = int(info["actionSize"])
        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_size,), dtype=np.float32)
        self._obs(probe)
        # env-config record + invariants (every run logs exactly what physics/spaces it ran under)
        assert self.obs_size == 105 and self.action_size == 18, \
            f"obs/action invariant broken: obs={self.obs_size} action={self.action_size}"
        print(f"[env] TankEnv scenario={self.scenario} arena={self.arena_mode} opponent={self.opponent} "
              f"expert={self.expert} shell_decay={self.shell_decay} spawn_powerups={self.spawn_powerups} "
              f"random_turret={self.random_turret} obs={self.obs_size} action={self.action_size} seed={self.seed_value}")

    def _start_bridge(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [self.node_bin, str(BRIDGE)],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("TankEnv is closed")
        self._start_bridge()
        assert self._proc is not None
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("bridge pipes are not available")

        try:
            self._proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        except BrokenPipeError as exc:
            raise RuntimeError(self._bridge_error("bridge pipe broke")) from exc

        if not line:
            raise RuntimeError(self._bridge_error("bridge exited without a response"))
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"bridge error: {response['error']}")
        return response

    def _bridge_error(self, message: str) -> str:
        err = ""
        if self._proc is not None and self._proc.stderr is not None:
            if self._proc.poll() is not None:
                try:
                    err = self._proc.stderr.read()
                except Exception:
                    err = ""
        return f"{message}{': ' + err.strip() if err.strip() else ''}"

    def _reset_payload(self, seed: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cmd": "reset",
            "seed": int(seed),
            "arenaMode": self.arena_mode,
            "scenario": self.scenario,
            "maxSteps": self.max_steps,
            "spawnPowerups": self.spawn_powerups,
            "spawnJitter": self.spawn_jitter,
            "randomTurret": self.random_turret,
            "shellDecay": self.shell_decay,
            "actionRepeat": self.action_repeat,
            "stepDt": self.step_dt,
        }
        if self.ruleset and self.ruleset != "survival_v1": payload["ruleset"] = self.ruleset
        if self.spawn_mode is not None: payload["spawnMode"] = self.spawn_mode
        if self.tank_max_hp is not None: payload["tankMaxHp"] = self.tank_max_hp
        if self.reward is not None:
            payload["reward"] = self.reward
        if self.expert is not None:
            payload["expert"] = self.expert
        return payload

    def _obs(self, response: dict[str, Any]) -> np.ndarray:
        obs = np.asarray(response["obs"], dtype=np.float32)
        expected = (self.obs_size,)
        if obs.shape != expected:
            raise RuntimeError(f"bad obs shape: {obs.shape}")
        if not np.all(np.isfinite(obs)):
            raise RuntimeError("obs contains NaN or Inf")
        if np.any(obs < -1.000001) or np.any(obs > 1.000001):
            raise RuntimeError("obs outside [-1, 1]")
        return np.clip(obs, -1.0, 1.0).astype(np.float32, copy=False)

    def _next_episode_seed(self, explicit_seed: bool) -> int:
        # An explicitly passed seed is honoured exactly (eval / tests rely on this)
        # and restarts the increment schedule. Otherwise pick per the diversity mode.
        if explicit_seed:
            self.episode_index = 1
            return int(self.seed_value) & 0x7FFFFFFF
        if self.randomize_seed:
            return int(self._seed_rng.integers(0, 2**31 - 1))
        if self.seed_increment:
            episode_seed = (self.seed_value + self.episode_index) & 0x7FFFFFFF
            self.episode_index += 1
            return int(episode_seed)
        return int(self.seed_value) & 0x7FFFFFFF

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        explicit_seed = seed is not None
        if explicit_seed:
            self.seed_value = int(seed)
            self._seed_rng = np.random.default_rng(self.seed_value)
        if options:
            self.opponent = str(options.get("opponent", self.opponent))
            self.arena_mode = str(options.get("arena_mode", self.arena_mode))
            self.scenario = str(options.get("scenario", self.scenario))
            if "spawn_powerups" in options:
                self.spawn_powerups = bool(options["spawn_powerups"])
            if "spawn_jitter" in options:
                self.spawn_jitter = bool(options["spawn_jitter"])
            if "random_turret" in options:
                self.random_turret = bool(options["random_turret"])
            if "shell_decay" in options:
                self.shell_decay = bool(options["shell_decay"])
            if "max_steps" in options:
                self.max_steps = int(options["max_steps"])

        episode_seed = self._next_episode_seed(explicit_seed)
        response = self._request(self._reset_payload(episode_seed))
        return self._obs(response), dict(response.get("info", {}))

    def step(self, action: int):
        msg = {
            "cmd": "step",
            "action": int(action),
            "opponent": self.opponent,
        }
        if self.expert is not None:
            msg["expert"] = self.expert
        response = self._request(msg)
        obs = self._obs(response)
        reward = float(response["reward"])
        if not np.isfinite(reward):
            raise RuntimeError("reward is NaN or Inf")
        terminated = bool(response["done"]) and not bool(response["truncated"])
        truncated = bool(response["truncated"])
        info = dict(response.get("info", {}))
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        self._proc = None
        if proc is None:
            return
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
