"""Self-play trainer: anchored PPO vs a MIX of scripts + frozen self/past-checkpoints.

Why self-play might succeed where plain PPO-vs-scripts eroded laika: the opponent pool always
includes strong selves, so there is no "cheap" opponent to over-exploit — beating a copy of
yourself is ~50% by construction, which keeps the gradient balanced. We keep the same guards that
made league_ppo_v2 safe: critic warm-up, CE/BC anchor to demos, and a fixed 100-ep gate on the
SCRIPTS (self-play must not erode scripted-opponent performance). The frozen self is a separate
loaded copy of the anchor; the training model never overwrites it (or any user model).

  python train/train_selfplay.py --anchor models/bc_dagger_moba_canon_v2.zip \
      --self-pool models/bc_dagger_moba_canon_v2.zip --out-prefix models/auto/selfplay_v1 \
      --total-timesteps 240000 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

try:
    from .tank_env import TankEnv
    from .tank_selfplay_env import TankSelfPlayVecEnv
    from .train_moba1v1duel import DUEL_REWARD
    from .train_bc import N_ACTIONS, load_demos
    from .league_stage1 import allocate
    from .league_ppo_v2 import AnchoredPPO, CriticWarmup
    from .evaluate_shooting_lab_bc import rollout as eval_rollout
except ImportError:
    from tank_env import TankEnv
    from tank_selfplay_env import TankSelfPlayVecEnv
    from train_moba1v1duel import DUEL_REWARD
    from train_bc import N_ACTIONS, load_demos
    from league_stage1 import allocate
    from league_ppo_v2 import AnchoredPPO, CriticWarmup
    from evaluate_shooting_lab_bc import rollout as eval_rollout

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "moba1v1duel"


class BalancedGateCallback(BaseCallback):
    """Generalized MAXIMIN gate for 'beat all scripts': a checkpoint is promoted to `_best` only
    if, at confirm-N, its MINIMUM win across ALL gate opponents exceeds the best min so far AND its
    mean does not regress below the anchor's mean. This protects every opponent (laika-aggressive
    + pro included), unlike the laika-only LeagueCallback. Two-tier (cheap monitor -> confirm)."""

    def __init__(self, eval_env, eval_opponents, interval, episodes, confirm_episodes, seed_base,
                 best_path, latest_path, mean_floor_drop=0.05, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_opponents = list(eval_opponents)
        self.interval = int(interval)
        self.episodes = int(episodes)
        self.confirm_episodes = int(confirm_episodes)
        self.seed_base = int(seed_base)
        self.best_path = best_path
        self.latest_path = latest_path
        self.mean_floor_drop = float(mean_floor_drop)
        self._last_eval = 0
        self.baselines = None
        self.base_min = -1.0
        self.base_mean = -1.0
        self.best_min = -1.0

    def _policy_fn(self):
        return lambda obs: int(np.asarray(self.model.predict(obs, deterministic=True)[0]).item())

    def _run(self, episodes):
        pol = self._policy_fn()
        out = {}
        for opp in self.eval_opponents:
            self.eval_env.opponent = opp
            out[opp] = eval_rollout(self.eval_env, pol, episodes, self.seed_base)
        return {o: r["win_rate"] for o, r in out.items()}

    def _on_training_start(self):
        w = self._run(self.confirm_episodes)
        self.baselines = w
        self.base_min = min(w.values())
        self.base_mean = sum(w.values()) / len(w)
        self.best_min = self.base_min
        if self.verbose:
            print("[gate] anchor baselines @%dep: %s  min=%.2f mean=%.2f"
                  % (self.confirm_episodes, {k: round(v, 2) for k, v in w.items()}, self.base_min, self.base_mean))
        self._last_eval = 0

    def _evaluate(self):
        w = self._run(self.episodes)
        self.model.save(self.latest_path)
        cheap_min = min(w.values())
        cheap_mean = sum(w.values()) / len(w)
        conf = None
        promoted = False
        # pre-filter: confirm only when the cheap eval looks like a balance improvement
        if cheap_min >= self.best_min - 0.05 and cheap_mean >= self.base_mean - self.mean_floor_drop:
            conf = self._run(self.confirm_episodes)
            cmin = min(conf.values())
            cmean = sum(conf.values()) / len(conf)
            if cmin > self.best_min and cmean >= self.base_mean - self.mean_floor_drop:
                self.best_min = cmin
                self.model.save(self.best_path)
                promoted = True
        if self.verbose:
            cheap = " ".join("%s=%.2f" % (o[:4], w[o]) for o in self.eval_opponents)
            line = "[gate] @%-7d cheap[min=%.2f mean=%.2f | %s]" % (self.num_timesteps, cheap_min, cheap_mean, cheap)
            if conf is not None:
                cc = " ".join("%s=%.2f" % (o[:4], conf[o]) for o in self.eval_opponents)
                line += " | confirm[min=%.2f %s]%s" % (min(conf.values()), cc, " *PROMOTED" if promoted else " (gate fail)")
            print(line)
        status = {"timesteps": int(self.num_timesteps), "cheap": w, "confirm": conf,
                  "baselines": self.baselines, "best_min": self.best_min, "promoted": promoted}
        (self.eval_env.run_dir / "selfplay_status.json").write_text(json.dumps(status, indent=2))
        self._last_eval = int(self.num_timesteps)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.interval:
            self._evaluate()
        return True

    def _on_training_end(self):
        self._evaluate()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", type=Path, default=Path("models/bc_dagger_moba_canon_v2.zip"))
    p.add_argument("--self-pool", default="models/bc_dagger_moba_canon_v2.zip",
                   help="comma list of frozen-self checkpoints (the league of past selves).")
    p.add_argument("--data-glob", default="data/expert_demos/moba_perop/*.jsonl")
    p.add_argument("--out-prefix", default="models/auto/selfplay_v1")
    p.add_argument("--script-mix", default="laika=0.25,easy_laika=0.2,stationary=0.15,laika-aggressive=0.25,laika-aggressive-pro=0.15")
    p.add_argument("--self-frac", type=float, default=0.5, help="fraction of envs whose opponent is a frozen self.")
    p.add_argument("--gate-opponents", default="stationary,easy_laika,laika,laika-aggressive,laika-aggressive-pro")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--total-timesteps", type=int, default=240_000)
    p.add_argument("--warmup-steps", type=int, default=30_000)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--clip-range", type=float, default=0.05)
    p.add_argument("--ent-coef", type=float, default=0.0008)
    p.add_argument("--ce-steps", type=int, default=8)
    p.add_argument("--lambda-ce", type=float, default=1.0)
    p.add_argument("--ce-batch", type=int, default=256)
    p.add_argument("--anchor-to-policy", action="store_true",
                   help="CE-anchor toward the START policy's actions (relabel demo states with the warm-started "
                        "model) instead of the laika-family demo actions -- prevents drift WITHOUT pulling toward "
                        "the narrow expert strategy (GPT: anchor-to-policy, not anchor-to-demo).")
    p.add_argument("--ruleset", default="survival_v1", help="'survival_v2' -> long-form combat (HP x2 / slower regen / random spawn).")
    p.add_argument("--spawn-mode", default=None, help="override spawnMode: fixed / half_random / full_random.")
    p.add_argument("--tank-max-hp", type=float, default=None, help="override tankMaxHp (default: ruleset value).")
    p.add_argument("--powerup-reward", type=float, default=None, help="reward for the learner picking up a powerup (default DUEL_REWARD 0.08; raise to teach resource play).")
    p.add_argument("--poison-hurt", type=float, default=None, help="penalty per HP of poison damage the learner takes (default 0; teaches poison-survival).")
    p.add_argument("--close-range-hit-penalty", type=float, default=None, help="extra penalty when the learner takes an ENEMY hit at close range (<200px) -- teaches the evasive 1-shot-counter vs reckless rushers.")
    p.add_argument("--clean-trade-bonus", type=float, default=None, help="bonus when the learner lands a hit without having been hit in the last 0.5s (rewards clean trades, not blood-trades).")
    p.add_argument("--approach-coef", type=float, default=None, help="dense reward per unit of distance CLOSED toward a visible enemy (DUEL_REWARD default 0; >0 teaches the evasive-pursuit opening AND biases toward aggression-commitment — the diagnosis's b-reward fix).")
    p.add_argument("--aim-bonus", type=float, default=None, help="dense reward for aiming at a visible enemy (DUEL_REWARD default 0; rewards keeping the gun on a dodger).")
    p.add_argument("--proximity-bonus", type=float, default=None, help="per-step bonus for staying within proximity-range of a VISIBLE enemy (NON-potential -> changes the optimum toward cornering evasive dodgers; the laika-wall fix).")
    p.add_argument("--proximity-range", type=float, default=None, help="px radius for --proximity-bonus (default 200).")
    p.add_argument("--corner-coef", type=float, default=None, help="ESCAPE-ANGLE cornering reward: per-step bonus * fraction of the enemy's escape directions BLOCKED by walls/the agent (cut off a reactive dodger's retreat). Janosov/SciRep 'pressure as reward'.")
    p.add_argument("--corner-range", type=float, default=None, help="px wall-proximity radius for the --corner-coef escape rays (default 120).")
    p.add_argument("--opp-deterministic", action="store_true", help="argmax opponent (default: sample, for diversity).")
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--eval-interval", type=int, default=30_000)
    p.add_argument("--eval-episodes", type=int, default=25)
    p.add_argument("--confirm-episodes", type=int, default=100)
    p.add_argument("--gate-easy-drop", type=float, default=0.12)
    p.add_argument("--gate-stationary-min", type=float, default=0.45)
    p.add_argument("--eval-seed-base", type=int, default=300_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=2099)
    return p.parse_args()


def main():
    args = parse_args()
    anchor = args.anchor if args.anchor.is_absolute() else ROOT / args.anchor
    best_path = ROOT / f"{args.out_prefix}_best.zip"
    latest_path = ROOT / f"{args.out_prefix}_latest.zip"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / f"runs/{Path(args.out_prefix).name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # frozen-self league (separate loaded copies; never trained)
    self_paths = [s.strip() for s in args.self_pool.split(",") if s.strip()]
    opp_models = [PPO.load(p if Path(p).is_absolute() else ROOT / p, device=args.device) for p in self_paths]

    # opponent assignment: script envs + self envs (self uses model indices round-robin)
    mix = {}
    for kv in args.script_mix.split(","):
        k, _, v = kv.partition("=")
        if k.strip():
            mix[k.strip()] = float(v)
    n_self = max(1, round(args.self_frac * args.n_envs))
    n_script = args.n_envs - n_self
    script_alloc = allocate(mix, n_script) if n_script > 0 else []
    self_alloc = [i % len(opp_models) for i in range(n_self)]   # int -> opp_models index
    opponents = list(script_alloc) + self_alloc
    counts = {}
    for o in opponents:
        key = f"self[{o}]" if isinstance(o, int) else o
        counts[key] = counts.get(key, 0) + 1

    # CE-anchor demos
    import glob as _glob
    files = []                                            # comma-split like train_bc (multi-pattern globs)
    for pat in args.data_glob.split(","):
        files += _glob.glob(str(ROOT / pat.strip())) or _glob.glob(pat.strip())
    files = sorted(set(files))
    obs_list, act_list, _ = load_demos(
        files, "good_wins", frozenset([SCENARIO]),
        frozenset(["laika", "easy_laika", "stationary"]),
        frozenset(["laika-aggressive", "laika-aggressive-pro", "pro"]))
    demo_X = np.asarray(obs_list, dtype=np.float32)
    demo_Y = np.asarray(act_list, dtype=np.int64)

    # Resource-play reward shaping (survival_v2): the base DUEL_REWARD's powerup=0.08 is too small vs hit/win,
    # so the policy ignores powerups; poison damage had no signal. Raise powerup + add a poison-damage penalty.
    duel_reward = dict(DUEL_REWARD)
    if args.powerup_reward is not None: duel_reward["powerup"] = float(args.powerup_reward)
    if args.poison_hurt is not None: duel_reward["poisonHurt"] = float(args.poison_hurt)
    if args.close_range_hit_penalty is not None: duel_reward["closeRangeHit"] = float(args.close_range_hit_penalty)
    if args.clean_trade_bonus is not None: duel_reward["cleanTrade"] = float(args.clean_trade_bonus)
    if args.approach_coef is not None: duel_reward["approachCoef"] = float(args.approach_coef)
    if args.aim_bonus is not None: duel_reward["aimBonus"] = float(args.aim_bonus)
    if args.proximity_bonus is not None: duel_reward["proximityBonus"] = float(args.proximity_bonus)
    if args.proximity_range is not None: duel_reward["proximityRange"] = float(args.proximity_range)
    if args.corner_coef is not None: duel_reward["cornerCoef"] = float(args.corner_coef)
    if args.corner_range is not None: duel_reward["cornerRange"] = float(args.corner_range)
    print(f"  reward: powerup={duel_reward.get('powerup')} poisonHurt={duel_reward.get('poisonHurt', 0.0)} "
          f"closeRangeHit={duel_reward.get('closeRangeHit', 0.0)} cleanTrade={duel_reward.get('cleanTrade', 0.0)} "
          f"hit={duel_reward.get('hit')} win={duel_reward.get('win', 1.0)}")

    venv = TankSelfPlayVecEnv(num_envs=args.n_envs, opponents=opponents, opp_models=opp_models,
                              opp_deterministic=args.opp_deterministic, arena_mode="survival",
                              scenario=SCENARIO, spawn_powerups=True, max_steps=args.max_steps,
                              ruleset=args.ruleset, spawn_mode=args.spawn_mode, tank_max_hp=args.tank_max_hp,
                              base_seed=args.seed, reward=duel_reward)
    from stable_baselines3.common.vec_env import VecMonitor
    env = VecMonitor(venv, filename=str(run_dir / "monitor.csv"))

    model = AnchoredPPO.load(anchor, env=env, device=args.device)
    model.learning_rate = args.lr
    model.lr_schedule = lambda _p: args.lr
    model.clip_range = lambda _p: args.clip_range
    model.ent_coef = args.ent_coef
    if args.anchor_to_policy:
        # GPT round-2: anchor toward the STARTING POLICY, not the laika-family demo actions. Relabel the
        # CE targets on the demo STATES with the warm-started (= start) policy's own argmax actions. The
        # CE-anchor then resists catastrophic drift from the good starting point while letting RL push the
        # diverse-opponent generalization, WITHOUT dragging the policy back toward the narrow expert strategy.
        demo_Y = model.predict(demo_X, deterministic=True)[0].astype(np.int64)
        print(f"[anchor] anchor-to-POLICY: relabeled {len(demo_Y)} CE targets with the start policy "
              f"({anchor.name}) — not laika demos")
    model.set_anchor(demo_X, demo_Y, args.ce_steps, args.lambda_ce, args.ce_batch, args.seed)
    assert model.observation_space.shape == (105,) and model.action_space.n == N_ACTIONS

    eval_env = TankEnv(arena_mode="survival", scenario=SCENARIO, opponent="laika", spawn_powerups=True,
                       max_steps=args.max_steps, run_dir=run_dir, seed=args.eval_seed_base, reward=duel_reward,
                       ruleset=args.ruleset, spawn_mode=args.spawn_mode, tank_max_hp=args.tank_max_hp)

    print("=" * 80)
    print(f"SELF-PLAY  anchor={anchor.name} (frozen, warm-start)  selves={[Path(p).name for p in self_paths]}")
    print(f"  pool({args.n_envs})={counts}  demos={len(demo_X)}  opp_det={args.opp_deterministic}")
    print(f"  lr={args.lr} clip={args.clip_range} ent={args.ent_coef} warmup={args.warmup_steps} "
          f"ce_steps={args.ce_steps} steps={args.total_timesteps}")
    print(f"  gate opponents={args.gate_opponents} (self-play must NOT erode scripts)")
    callback = [
        CriticWarmup(args.warmup_steps),
        BalancedGateCallback(eval_env, args.gate_opponents.split(","),
                             args.eval_interval, args.eval_episodes, args.confirm_episodes,
                             args.eval_seed_base, best_path, latest_path),
    ]
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
    finally:
        model.save(latest_path)
        eval_env.close()
        venv.close()
    print(f"done. latest -> {latest_path}; best (gated) -> {best_path}")


if __name__ == "__main__":
    main()
