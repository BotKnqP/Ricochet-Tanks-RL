"""Behaviour-Cloning pipeline smoke test.  Run:  python smoke_bc.py

Generates synthetic demos, trains BC for 2 epochs, and verifies the spec's checks:
data reads, obs [N,101] float32, actions [N] int64 all in [0,18), trains+saves an SB3
PPO .zip, PPO.load reloads it, evaluate_bc runs, and the .zip is resume-compatible with
the real moba1v1duel duel env. Self-contained; cleans up its own temp artifacts.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "train"))
from train_bc import load_demos, ACTION_TABLE, N_ACTIONS          # noqa: E402

DATA = ROOT / "data" / "human_demos" / "_smoke_bc.jsonl"
MODEL = ROOT / "models" / "_smoke_bc.zip"


def run(cmd):
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main():
    try:
        # (1) generate synthetic demos (laika vs easy_laika -> wins -> good_wins has data)
        run(["node", "gen_synthetic_demo.js", "--episodes", "4",
             "--out", "data/human_demos/_smoke_bc.jsonl", "--blue", "laika"])
        assert DATA.exists(), "demo file not generated"

        # (1-4, 9-10) load + shape / dtype / range checks
        obs_l, act_l, stats = load_demos([str(DATA)], "good_wins",
                                         frozenset(["moba1v1duel"]), frozenset(["easy_laika"]), None)
        obs = np.asarray(obs_l, dtype=np.float32)
        acts = np.asarray(act_l, dtype=np.int64)
        assert obs.ndim == 2 and obs.shape[1] == 101, f"obs shape {obs.shape}"
        assert acts.ndim == 1 and acts.dtype == np.int64, f"actions {acts.shape} {acts.dtype}"
        assert (acts >= 0).all() and (acts < N_ACTIONS).all(), "actions out of [0,18)"
        assert np.all(np.isfinite(obs)) and obs.min() >= -1.0001 and obs.max() <= 1.0001, "obs out of [-1,1]"
        assert stats["control_mismatch"] == 0 and stats["skipped"] == 0, f"data issues {stats}"
        # every transition's control round-trips through ACTION_TABLE[action]
        with open(DATA, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("type") == "transition":
                    c = r["control"]
                    assert (c["throttle"], c["turn"], bool(c["fire"])) == ACTION_TABLE[r["action"]], "control != ACTION_TABLE[action]"
        print(f"OK data: obs {obs.shape} {obs.dtype}  actions {acts.shape} {acts.dtype}  all in [0,{N_ACTIONS})")

        # (5-6) train 2 epochs -> save an SB3 PPO .zip
        run([sys.executable, "train/train_bc.py", "--data-glob", "data/human_demos/_smoke_bc.jsonl",
             "--out", "models/_smoke_bc.zip", "--epochs", "2", "--filter", "good_wins", "--device", "cpu"])
        assert MODEL.exists(), ".zip not saved"

        # (7) PPO.load reloads it with the right spaces
        from stable_baselines3 import PPO
        m = PPO.load(MODEL, device="cpu")
        assert m.observation_space.shape == (101,) and m.action_space.n == N_ACTIONS, \
            f"loaded spaces {m.observation_space} {m.action_space}"
        print(f"OK PPO.load: {m.observation_space}  {m.action_space}")

        # (8) evaluate_bc runs (3 episodes vs easy_laika)
        run([sys.executable, "train/evaluate_bc.py", "--model", "models/_smoke_bc.zip",
             "--episodes", "3", "--no-compare-random", "--device", "cpu"])

        # resume-compat: PPO.load INTO the real duel env (the train_moba1v1duel --resume path)
        from tank_env import TankEnv
        from train_moba1v1duel import DUEL_REWARD
        env = TankEnv(arena_mode="survival", scenario="moba1v1duel", opponent="easy_laika",
                      spawn_powerups=True, max_steps=120, reward=DUEL_REWARD,
                      run_dir=str(ROOT / "runs" / "_smoke_bc"))
        PPO.load(MODEL, env=env, device="cpu")     # raises ValueError if spaces mismatch
        env.close()
        print("OK resume-compat: PPO.load(bc.zip, env=duel_env)")
        print("BC SMOKE OK")
    finally:
        for p in (DATA, MODEL):
            try:
                p.unlink()
            except OSError:
                pass
        shutil.rmtree(ROOT / "runs" / "_smoke_bc", ignore_errors=True)
        shutil.rmtree(ROOT / "runs" / "bc_moba1v1duel_easy_laika", ignore_errors=True)


if __name__ == "__main__":
    main()
