"""Environment-parity check: pure-JS (eval_script_bot.js) vs TankEnv/bridge, same seeds.

After the shell_decay=True canonical reset, the SAME expert matchup must produce ~the same
win_rate and hits_dealt whether run in the JS core directly or through the python<->node
bridge. If they disagree, the training/eval env still diverges from the demo env -> do NOT
proceed to any training.

  python train/parity_check.py --episodes 50
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

try:
    from .tank_env import TankEnv
    from .train_moba1v1duel import DUEL_REWARD
except ImportError:
    from tank_env import TankEnv
    from train_moba1v1duel import DUEL_REWARD

ROOT = Path(__file__).resolve().parents[1]
MATCHUPS = [("laika-aggressive", "laika"),
            ("laika-aggressive-pro", "easy_laika"),
            ("laika-aggressive-pro", "stationary")]


def js_side(blue, red, episodes, seed, max_steps):
    out = subprocess.run(
        ["node", "eval_script_bot.js", "--blue", blue, "--red", red, "--scenario", "moba1v1duel",
         "--episodes", str(episodes), "--seed", str(seed), "--max-steps", str(max_steps)],
        cwd=str(ROOT), capture_output=True, text=True)
    m = re.search(r"\{.*\}", out.stdout, re.S)
    if not m:
        raise RuntimeError(f"could not parse eval_script_bot output: {out.stdout[-300:]} {out.stderr[-300:]}")
    d = json.loads(m.group(0))
    return float(d["win_rate"]), float(d.get("avg_hits_dealt", -1))


def env_side(blue, red, episodes, seed, max_steps):
    env = TankEnv(arena_mode="survival", scenario="moba1v1duel", opponent=red, expert=blue,
                  spawn_powerups=True, max_steps=max_steps, run_dir=ROOT / "runs/_parity",
                  seed=seed, reward=DUEL_REWARD)            # shell_decay defaults True now
    wins, hd = 0, 0.0
    try:
        for i in range(episodes):
            obs, info = env.reset(seed=seed + i)
            done = False
            while not done:
                a = int(info.get("expertAction"))
                obs, _r, term, trunc, info = env.step(a)
                done = term or trunc
            if info.get("result") == "win":
                wins += 1
            hd += float(info.get("hitsDealt", 0.0))
    finally:
        env.close()
    return wins / max(1, episodes), hd / max(1, episodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=300_000)
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument("--tol-win", type=float, default=0.10)
    p.add_argument("--tol-hd", type=float, default=0.40)
    args = p.parse_args()

    print(f"PARITY CHECK  episodes={args.episodes} seed={args.seed} max_steps={args.max_steps}  (canonical shell_decay=True)")
    print("matchup                              JS_win  Env_win  d_win   JS_hd  Env_hd  d_hd   PASS")
    ok_all = True
    for blue, red in MATCHUPS:
        jw, jh = js_side(blue, red, args.episodes, args.seed, args.max_steps)
        ew, eh = env_side(blue, red, args.episodes, args.seed, args.max_steps)
        passed = abs(ew - jw) <= args.tol_win and abs(eh - jh) <= args.tol_hd
        ok_all = ok_all and passed
        print("%-34s   %.2f    %.2f    %+.2f   %.2f   %.2f   %+.2f   %s"
              % (f"{blue} vs {red}", jw, ew, ew - jw, jh, eh, eh - jh, "ok" if passed else "FAIL"))
    print("=" * 70)
    print("PARITY %s  (tol: win<=%.2f, hits<=%.2f)" % ("PASS -> safe to proceed" if ok_all else "FAIL -> do NOT train",
                                                       args.tol_win, args.tol_hd))


if __name__ == "__main__":
    main()
