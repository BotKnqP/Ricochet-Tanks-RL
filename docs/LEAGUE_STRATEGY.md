# LEAGUE_STRATEGY.md — Mini-AlphaStar league to break the Ricochet-Tanks seesaw

> RESEARCH/DESIGN ONLY. This document specifies edits and *command recipes*. Nothing here has
> been executed. Every claim about an external method is cited; every file/asset named below was
> verified to exist in this checkout on 2026-06-21.

---

## 1. Diagnosis — our seesaw IS AlphaStar's catastrophic-forgetting failure mode

A single identity-blind PPO policy in `moba1v1duel` **seesaws**: whenever it learns the
charger/laika-aggressive counter (charge-and-trade) it forgets the laika counter (evasive
lead-shoot), and vice-versa, because the two counters are **mutually contradictory** and the 105-d
obs carries **no opponent identity** — so both behaviours must live in one shared weight set and the
PPO gradient simply spends itself where wins are cheapest. That is exactly AlphaStar's documented
self-play pathology: plain self-play "may chase cycles (A beats B, B beats C, A loses to C)
indefinitely without making progress," and a win-rate-maximizing learner forgets the hard opponent
to bank the easy one (Vinyals et al., *Grandmaster level in StarCraft II*, Nature 2019;
`AlphaStar_unformatted.pdf` lines 122-133). Our own runs reproduce this at toy scale: **JOURNAL J.4**
records that velocity-obs tripled the neural base (mean 0.11→0.35) but single-policy self-play
**still seesawed and never beat the fixed pure-aggression SCRIPT** (maximin 0.46-0.50; the script
wins precisely because it is a *fixed* behaviour that never reacts and therefore never seesaws).
AlphaStar's cure is not better-tuned PPO — it is (a) **reweight the opponent distribution** (PFSP)
so the agent is forced to spend games where it is barely-winning/losing, (b) **hold the agent
against a frozen archive** of all past players so any regression is detected and corrected, and
(c) **add reset-on-add exploiters** that hunt the current product's specific holes. Those are the
three levers we have never pulled, and they map cleanly onto infra we already built.

---

## 2. Mini-league architecture mapped to our EXISTING files + assets

**Scale collapse.** We are one 16-core box, not 12 agents × 32 TPU × 44 days. We keep AlphaStar's
*algorithm* (PFSP weighting + frozen archive + exploiter resets + non-regression gate) and drop its
*scale* (no distributed actor-learner split, no Nash solver in the minimal version, agents run
**sequentially as time-slices**, not concurrently — the JS bridge at ~3.9 ms/decision already
saturates one 16-env PPO, so concurrency would only halve each agent's throughput).

### Roster (3 trainable slots, run one-at-a-time per generation)

| Role | AlphaStar analogue | Warm-start seed (asset) | Opponent pool | PFSP | Gate | Reset |
|---|---|---|---|---|---|---|
| **M (main / product)** | main agent | `models/auto/vel_bc_anchor2.zip` (105-d, mean 0.35) — gen 0 only; then carries its own `_latest` | full script roster ∪ frozen archive | `f_hard`, p≈1.5 | extended (§4) | **NEVER** |
| **E_reck (reckless exploiter)** | main exploiter | `models/auto/v2_reckless_specialist_best.zip` | current-M snapshot + aggression scripts {laika, laika-aggressive, charger, ace} | `f_var` | target-cluster only | on every add |
| **E_evas (evasive exploiter)** | main exploiter | `models/auto/v2_movers_specialist_latest.zip` | current-M snapshot + kite scripts {laika-aggressive-pro, p-kiter, baiter, turtle, precision} | `f_var` | target-cluster only | on every add |

Only **M** ever ships (its gated `_best.zip`). Exploiters forget everything outside their cluster
**on purpose** — that is what an exploiter is for. Every member **must** inherit `AnchoredPPO` +
`CriticWarmup` (`train/league_ppo_v2.py:48` / `:85`) or it erodes on contact — documented repeatedly
(JOURNAL; the warmup+CE-anchor stack is what let R2 self-play lift laika 0.30→0.64).

### Archive

- **Directory** `models/auto/league/` + **registry** `train/league_registry.json` (NEW data file):
  `{id, path, kind: main|exploiter|script, gen, parent, last_winrates}`. Scripts are virtual
  (name-only) entries. This persistent league metadata is the thing the codebase currently lacks
  (today there are only flat `*.zip` + JOURNAL prose).
- **Snapshot-into-pool (the anti-cycling guarantee).** Every generation, `model.save()` M into
  `models/auto/league/` and **hot-append** it to `TankSelfPlayVecEnv.opp_models` so M must keep
  beating its **past selves**. Today `opp_models` is **load-once** in `tank_selfplay_env.py:54`
  (`__init__`), and R2/F faked growth by **manual relaunch** with `--self-pool [prev_latest,
  prev_best,...]`. The frozen-self mechanism already works: `tank_selfplay_env.py:143-148` runs each
  frozen policy via `opp_models[midx].predict(self._obs1[idxs], deterministic=...)` on the cached
  red-perspective obs1, and `moba1v1duel` is symmetric so a blue-trained policy plays red directly.
- **Cap** the archive selves at ~6-8 (newest M + both latest exploiters + 2-3 spaced older M) so
  `opp_models` stays small and the bridge stays fast.

### Training loop (one generation ≈ one 6 h block)

```
1. E_reck ~120k steps  vs {current-M snapshot} (f_var, aggression scripts).
     if win(E_reck vs M) > 0.6 on its cluster -> freeze into archive (a real hole in M). RESET E_reck.
2. E_evas ~120k steps  vs {current-M snapshot} (f_var, kite scripts).
     if win(E_evas vs M) > 0.6 -> freeze into archive. RESET E_evas.
3. Snapshot current M into archive (anti-cycling); cap selves at ~8; rebuild PFSP weights from the
     gate's last per-opponent win rates (the gate's _run() already computes these).
4. M ~200k steps vs the PFSP-sampled pool (full script roster + archive selves incl the 2 fresh
     exploiters). Promote M_best ONLY if the extended gate (§4) passes.
5. Repeat. ~1 generation per 6 h block; M gets the majority of wall-clock (200k of ~440k steps).
```

### Files this touches (existing assets, verified present)

- **REUSE unchanged:** `league_ppo_v2.AnchoredPPO` + `CriticWarmup` (mandatory for every member),
  `evaluate_shooting_lab_bc.rollout` (the gate engine, imported at `train_selfplay.py:32/40`),
  `eval_v2_agent.js` (TRAIN/HELD maximin/mean reporting), `eval_router.js` (the fallback router),
  `script_matrix.js` (optional payoff matrix for a later RPP/Nash gate upgrade),
  `league_stage1.allocate` (`league_stage1.py:81`, already guarantees ≥1 env/opponent).
- **Opponent ecology already on disk:** the script roster {stationary, easy_laika, laika,
  laika-aggressive, laika-aggressive-pro/pro, charger, ace, p-kiter, randomized, …} via
  `game_core.js` `scriptedControl` (`PARAM_PRESETS` ~L1859, `sampleRandomParams` ~L1874).
- **Exploiter seeds already trained:** `dagger_aggro_specialist.zip`, `v2_reckless_specialist_best.zip`,
  `v2_movers_specialist_latest.zip` (these are 101-d — see STEP ZERO below).

---

## 3. STEP ZERO — obs101→105 weight-transfer surgery (the strong warm-start the last run lacked)

**Why first.** J.4 shows the previous self-play run started from a *weak* anchor (the from-scratch
105-d vel net, mean 0.11 evasive-only / 0.35 full-suite). The strong combat policies
(`league_robust_v3_best`, `dagger_aggro_specialist`, `v2_reckless_specialist_best`, …) are **101-d**
and cannot load into the 105-d env. Surgery gives the league a competent 105-d starting policy that
**holds combat from step 0** instead of re-paying a from-scratch train. This is OpenAI-Five
"Changing the Observation Space" (Eq.8/Eq.9 `W_hat=[W|0]`; arXiv:1912.06680 App. B): the new input
columns are zero-init so the network output is **byte-identical** at the instant of surgery; the new
dims only move off zero if gradients prove them useful.

**The one subtlety (verified in `game_core.js`).** The 4 velocity dims are **interior, not
appended**. `OBS_SIZE = 73 + RAY_OFFSETS.length = 105` (`game_core.js:84`); `buildObservation`
puts motion at indices **90-93** (`mvf, mvl, ovf, ovl`, `game_core.js:2030-2040`) and **keeps the
11-wide poison/safe block trailing at 94-104** (comment `game_core.js:2033`: "Kept BEFORE the poison
block so the survival poison/safe block stays the trailing 11 dims"). So in the 101-d models the
poison block was columns 90-100, and surgery is a **column-INSERT at 90 with a right-shift of the
11 poison columns to 94-104** — a naïve `[W|0]` *append* would mis-map every poison feature and
destroy the champion.

The only two obs-consuming matrices are `mlp_extractor.policy_net.0.weight (64,101)` and
`mlp_extractor.value_net.0.weight (64,101)` (default SB3 `MlpPolicy`, net_arch [64,64], Tanh,
separate pi/vf heads — confirmed from the champions' state_dicts). Everything downstream
(`.2.*`, `action_net`, `value_net`) is obs-dim-agnostic and copies **verbatim**.

**NEW FILE `train/surgery_obs101to105.py`** (TEXT recipe, do NOT run):

```python
OBS_OLD, OBS_NEW, VEL_AT, N_VEL = 101, 105, 90, 4
def insert_cols(W_old):                 # (64,101) -> (64,105); cols 90:94 = 0
    out = W_old.new_zeros((W_old.shape[0], OBS_NEW))
    out[:, :VEL_AT]          = W_old[:, :VEL_AT]      # [0..89]   copy
    # out[:, 90:94] stays ZERO                        # [90..93]  new velocity
    out[:, VEL_AT+N_VEL:]    = W_old[:, VEL_AT:]      # [94..104] <- old [90..100] (poison shift)
    return out
# Build a fresh 105-d default-MlpPolicy PPO via the _SpaceEnv(105,18) trick (train_bc.py:107) so the
# saved spaces are Box(-1,1,(105,))/Discrete(18) and PPO.load-resume stays valid; copy the two first
# layers through insert_cols(), copy ALL other tensors verbatim, load_state_dict(strict=True).
# AUDIT (Surgery-with-Sets, arXiv:1912.06719): assert new[:,90:94]==0, new[:,:90]==old[:,:90],
# new[:,94:105]==old[:,90:101]. Do NOT transplant the (64,101) Adam moments — fresh Adam state.
```

**Command recipe (run later; surgery the champion + every frozen-self that will be a self-pool/eval opponent):**

```bash
python train/surgery_obs101to105.py models/auto/league_robust_v3_best.zip       models/auto/league_robust_v3_best_105.zip
python train/surgery_obs101to105.py models/auto/dagger_aggro_specialist.zip     models/auto/dagger_aggro_specialist_105.zip
python train/surgery_obs101to105.py models/auto/v2_reckless_specialist_best.zip models/auto/v2_reckless_specialist_best_105.zip
python train/surgery_obs101to105.py models/auto/v2_movers_specialist_latest.zip models/auto/v2_movers_specialist_latest_105.zip
python train/surgery_obs101to105.py models/auto/selfplay_v1_latest.zip          models/auto/selfplay_v1_latest_105.zip
```

After surgery, on the **first** resume run a brief **LR=0 / tiny-LR warm window** (overridable at
`train_selfplay.py:231-232` `model.learning_rate` / `model.lr_schedule`) so SB3/Adam re-estimate
the moments under the new (64,105) shape (OpenAI-Five "smooth restart"; matches our documented
Adam/resume fragility). **League hygiene:** surgery EVERY frozen-self in `--self-pool` and every
frozen-self eval opponent identically, or `opp_models[midx].predict` (`tank_selfplay_env.py:145`)
hits a 101-vs-105 shape mismatch / silently-weak opponent and poisons the maximin gate.

> **Note on the warm-start choice for M.** `vel_bc_anchor2.zip` is already 105-d (mean 0.35) and
> needs no surgery; it is the safe gen-0 seed for **M**. The surgered `league_robust_v3_best_105`
> and `dagger_aggro_specialist_105` are stronger *combat* policies and are the better seeds for the
> **exploiters** (and candidate alternates for M if Phase 1 shows vel_bc_anchor2 too weak). Do NOT
> mix any un-surgered 101-d ckpt into a 105-d pool.

---

## 4. Exploitability promotion gate + PFSP sampler

### 4a. Extended gate (replaces / extends `BalancedGateCallback`, `train/train_selfplay.py:46`)

Today's gate (`train_selfplay.py:92-107`) promotes on **"min over scripts strictly up AND mean not
regressed below anchor-mean − 0.05"** and evals **only scripts**. Two holes the campaign fell into:
(1) **mean** is a tie-breaker that still lets a checkpoint trade laika→0 while mean rises (the
seesaw); (2) the cheap pre-filter runs at **25 ep** and the campaign was demonstrably fooled by
25-ep noise (JOURNAL: +0.05 deltas are inside 60-ep noise; +0.20 is real). It also never checks
**exploitability** — nothing tests whether a fresh best-response crushes the candidate.

**New `ExploitabilityGateCallback`** — same constructor surface + new args (`archive_paths`,
`exploiter_paths`, `collapse_floor=0.10`, `collapse_eps=0.06`, `maximin_margin=0.07`,
`exploiter_max_win=0.65`, `p_pfsp=1.5`). Eval the candidate vs the **WHOLE roster** = the full fixed
script suite (TRAIN ∪ HELD from `eval_v2_agent.js`: laika, easy_laika, stationary, laika-aggressive,
charger, laika-aggressive-pro + held precision, counter, turtle, baiter, p-kiter) **∪ every frozen
archive ckpt ∪ every exploiter on disk**. Scripts use the existing `TankEnv.opponent` string path
(`train_selfplay.py:77`); frozen-policy opponents reuse the `obs1` mechanism (a small
`TankSelfPlayVecEnv` whose `opp_models` are the archive/exploiter ckpts). This closes the
"gate only sees scripts" hole — it is AlphaStar's **archive-as-permanent-test**: a candidate that
wins laterally by sacrificing laika now also loses to its own ancestor on laika, so the regression
is **visible**.

**Promotion rule** (all clauses, at CONFIRM = ≥90 ep × 2 seed-bases; the 25-ep cheap eval may
pre-filter but NEVER promote; **mean is logged, never a criterion**):

```python
def should_promote(win, prior_win, best_maximin,
                   floor=0.10, eps=0.06, margin=0.07, exploiter_max=0.65, pool_payoff=None):
    roster = list(win)
    maximin = min(win[o] for o in roster)
    if maximin < best_maximin + margin:                            return False, "maximin not up"       # (1)
    for o in roster:                                                                                     # (2) anti-forget
        if win[o] < max(floor, prior_win.get(o, 0.0) - eps):       return False, f"collapse:{o}"
    if max(1.0 - win[o] for o in roster) > exploiter_max:          return False, "exploitable"           # (3)
    if pool_payoff is not None:                                                                           # (4) RPP, optional
        p = fictitious_play_nash(pool_payoff, iters=1000)          # pure numpy, N~20
        if (p @ pool_payoff @ p) < best_rpp:                       return False, "rpp down"
    return True, f"PROMOTE maximin={maximin:.2f}"
```

- **(1) MAXIMIN UP** beyond eval noise (Δ ≥ 0.07 — +0.05 is inside 60-ep noise).
- **(2) NO-COLLAPSE** (the anti-seesaw term, = AlphaStar's "min win-rate vs all past versions"):
  every opponent `win[o] ≥ max(0.10, prior_win[o] − 0.06)`, where `prior_win` is the win vs `o` at
  the **last promoted** checkpoint (persisted in `runs/<prefix>/league.json`). This is what makes
  laika→0 a **hard reject** instead of a mean-offsetting trade.
- **(3) NOT EXPLOITABLE** — no roster opponent (script, archive ancestor, or fresh exploiter) beats
  the candidate by > ~0.65 (mirrors AlphaStar's >70% exploiter threshold). Tier the
  exploitability probe cheapest-first: (c1) fixed-roster maximin (free, already in the matrix);
  (c2) empirical-RPP over the ~20×20 pool payoff via 1000-iter fictitious play (microseconds);
  (c3) the periodically-reset trained exploiter as the honest best-response oracle (run as an
  occasional audit, not every interval).
- **(4) RPP not worse** (optional) — candidate's expected win vs the empirical-Nash mixture of the
  pool ≥ prior best (catches a candidate that beats each pool member alone but is exploitable by a
  *mixture*).

### 4b. PFSP sampler (replaces the static `allocate(--script-mix)` split)

**NEW FILE `train/pfsp.py`** (pure numpy, reuses the floor-safe allocator):

```python
import numpy as np
from league_stage1 import allocate
def f_hard(x, p=1.5): return (1.0 - x) ** p     # main agent: grind what you LOSE to; f_hard(1)=0 -> solved opp gets 0 envs
def f_var(x):         return x * (1.0 - x)       # exploiter / weak-target curriculum: peak at x=0.5
def pfsp_alloc(winrates, opponents, mode, n_envs, p=1.5, eps=1e-3, weak_thresh=0.2):
    def wf(o):
        x = float(winrates.get(o, 0.5))
        if mode == 'hard' and x >= weak_thresh: return f_hard(x, p)
        return f_var(x)                          # f_var for exploiters AND for struggling targets (x<0.2)
    w = {o: max(eps, wf(o)) for o in opponents}
    s = sum(w.values()) or 1.0
    return allocate({o: w[o]/s for o in opponents}, n_envs)   # reuses >=1-env floor / no floor-to-0
```

**Mixture (AlphaStar 35/50/15, scaled down).** Each generation, split the env budget:
~**35%** naive self-play (M vs newest-M snapshot — fast learning) + ~**50%** PFSP over
{scripts + frozen selves} + ~**15%** reserved for **"forgotten"** opponents (any whose current win
dropped > 0.15 below its value at the last promoted M — the explicit anti-forgetting term our
DAgger/PPO runs lacked). **M uses `f_hard` p≈1.5; exploiters use `f_var`.** Win rates come from the
gate's `_run()`, **EMA-smoothed and averaged over ≥2 seed bases** (outcomes are seed-noisy — don't
let the sampler chase noise). **Critical:** route weights through the FIXED `allocate()`
(`league_stage1.py:81`) and **assert pool counts ≥1 each generation** — a PFSP weight that rounds
to <1 env would otherwise silently zero an opponent (the documented floor-to-0 gotcha).

> **`f_hard(1)=0` starves a fully-beaten opponent of TRAINING games — but it MUST stay in the
> GATE/eval suite or a regression there goes unseen.** Sampling is sparse; the eval roster is
> fixed and FULL.

### 4c. Code changes summary

- `train/train_selfplay.py`: add `--pfsp-mode {hard,var,none}`, `--pfsp-p`, `--winrates-json`;
  derive the per-env allocation from PFSP weights instead of the static `--script-mix`
  (the parse/allocate block is at `train_selfplay.py:185-195`). Add `--collapse-eps`,
  `--exploiter-paths`, `--archive-paths`; swap `BalancedGateCallback` for
  `ExploitabilityGateCallback`. Persist `runs/<prefix>/league.json` (prior_win baseline). Forbid the
  25-ep cheap eval from promoting (confirm ≥90 ep × 2 seeds). (`selfplay_status.json` is already
  written at `train_selfplay.py:117`.)
- `train/tank_selfplay_env.py`: add `hot_add_opponent(model)` (append to `self.opp_models`) and
  `set_opponents(list)` (mutate `self._spec` + `init_opponents` in place) so the orchestrator can
  grow the frozen-self pool and re-weight per-env opponents **without** a full bridge restart
  (today `opponents` is fixed at construction, `tank_selfplay_env.py:58-69`).
- `train/league.py` (NEW, ~150 lines of glue): the orchestrator. Loads `league_registry.json`; per
  generation runs the exploiter→main passes by importing `train_selfplay.main()` with per-pass argv
  (or subprocess), snapshots M, appends to the registry, rebuilds PFSP weights from the prior gate
  win-rates.
- `train/league_registry.json` (NEW data file).
- `train/pfsp.py`, `train/surgery_obs101to105.py` (NEW, §3).

**Orchestrator command recipe (one generation; run later):**

```bash
python train/league.py --gens 1 --n-envs 16 --device cpu \
    --main-anchor models/auto/vel_bc_anchor2.zip \
    --reck-seed   models/auto/v2_reckless_specialist_best_105.zip \
    --evas-seed   models/auto/v2_movers_specialist_latest_105.zip \
    --main-steps 200000 --exploiter-steps 120000 \
    --pfsp-p 1.5 --collapse-eps 0.06
```

---

## 5. Memory / architecture upgrade — minimal-viable-first ordering

**Core diagnosis:** the seesaw is a **commitment / non-transitivity** failure, not a perception
failure. "Recognize the style + commit to the counter" has two sub-problems — RECOGNIZE (infer the
fixed-per-episode style from history) and COMMIT (lock the matching counter with no gradient
interference). The three options differ in **where the recognized-style state lives**, and that is
the whole ballgame.

1. **(C) Hard-router + experts — FIRST. Already built, zero retrain, directly delivers commitment.**
   `train/eval_router.js` already does classify-then-lock: a **45-step warmup** (generalist drives
   while it accumulates the enemy's **approach speed** — `--approach 8`, `--close-frac 0.30`,
   computed from public game state **outside** the 105-d obs, so no obs change and no retrain), then
   it **locks** reckless→specialist / evasive→generalist for the rest of the episode. It puts the
   style state in an **external latch**, so the two contradictory counters live in **separate**
   expert nets and never share a gradient. **Measured: maximin 0.12→0.35, mean 0.40→0.57**
   (JOURNAL I.2; battles table line 514). The only gap is the laika floor (the generalist route
   bottoms out at 0.35) → the one piece of NEW work is **one clean 105-d evasive specialist** for
   the evasive route (a normal DAgger run, reuses the expert-map oracle, no architecture change).

   ```bash
   node train/eval_router.js --spec-script laika-aggressive \
     --generalist models/auto/v2_movers_specialist_latest_105.zip \
     --opps laika,laika-aggressive,charger,easy_laika,stationary,laika-aggressive-pro,randomized \
     --ruleset survival_v1 --warmup 45 --approach 8 --close-frac 0.30 --episodes 30 --seeds 300000,900000
   ```

2. **(A) VecFrameStack K=4 — SECOND, ONLY for the velocity/lead-shooting wall (a SEPARATE problem).**
   Concat last 4 frames → 420-d gives acceleration/charging perception. But it **breaks the 105-d
   warm-start invariant** (the `assert obs_size==105` at `tank_selfplay_env.py:100` /
   `tank_env.py:92`), forces a fresh net + full re-BC, and per our own teacher-learnability
   diagnostic (K=1 ~98%; K>1 +1-2%) it barely helps the per-step decision and adds **no commitment
   latch**. So it will likely still seesaw. Frame-stack **one laggard expert**, not the router. (If
   used, `train/train_dagger_fs.py` exists but its module-level `OBS=101` is stale and must be fixed
   to 105 first.)

3. **(B) RecurrentPPO / GRU — LAST / probably never.** True memory (could infer-and-latch the
   style), but `sb3-contrib` is **not installed**, `MlpLstmPolicy` is weight-incompatible with the
   MLP anchor (lose the warm-start), the CE/BC anchor (`AnchoredPPO.set_anchor`, `league_ppo_v2.py:51`)
   must be rewritten to replay **sequences** with hidden-state carry, BPTT is fragile on CPU, and it
   is still **one weight set** so commitment is not guaranteed. Overkill for a style that is
   fixed-per-episode and cleanly classifiable in ~1-2 s by a 3-line heuristic.

---

## 6. Phased plan — ordered by effort vs expected payoff

**Phase 1 (~½ day edits + 1 short run) — the SMALLEST experiment that validates "league/surgery
beats single-run self-play" BEFORE committing 10 h.**
Do not build the orchestrator, the exploiters, or the registry yet. Build only:
1. `train/surgery_obs101to105.py` and surgery **one** model (`league_robust_v3_best` → `_105.zip`)
   with the three audit asserts (§3). This is pure offline weight-editing, no training.
2. `train/pfsp.py` + wire `--pfsp-mode hard` into `train_selfplay.py` (derive the allocation from
   `(1-winrate)^1.5` over the existing scripts + selves, reusing `allocate()`).
3. The single cheap gate upgrade: in `BalancedGateCallback._evaluate`, **delete `mean` as a
   promotion criterion** and add the **no-collapse** clause (`win[o] ≥ max(0.10, prior_win[o]−0.06)`),
   keeping `prior_win` in a tiny `league.json`. (~1 line of logic change + a small bookkeeping read.)

**Phase-1 A/B test (the validation):** run **one** `train_selfplay.py` (~200k steps, ~one 6 h block)
with `--anchor models/auto/league_robust_v3_best_105.zip --pfsp-mode hard` and the no-collapse gate,
**vs** the existing single-run self-play baseline (the J.4 run, vel_bc_anchor2 + static mix + old
gate). **Decision rule:** if the surgered+PFSP+no-collapse run lifts the gate **maximin by ≥ 0.07
over the J.4 baseline at the honest 90 ep × 2-seed confirm** (and laika does not collapse), the
league mechanics are validated → proceed to Phase 2. If it does not, **stop and skip to the router
fallback** — do not pour 10 h into the full league.

**Phase 2 (~1-2 days) — full mini-league** *only if Phase 1 passed.* Add `train/league.py`, the
registry, `hot_add_opponent` / `set_opponents`, snapshot-into-pool growth, and the two reset-on-add
exploiters (E_reck, E_evas) seeded from the surgered specialists. Extend the gate to the full
`ExploitabilityGateCallback` (whole-roster eval incl. archive + exploiters, clauses 1-3; clause 4
RPP optional). Run ~4-6 generations.

**Phase 3 (optional polish) — exploitability ceiling & PBT.** Add the RPP/Nash gate clause via
`script_matrix.js` + `fictitious_play_nash`; optionally a small (4-8 worker) PBT layer over
{CE-anchor λ, ent-coef, self-frac, p} ranked by **maximin fitness** (not reward). These are
microsecond-cost at our pool size; only the sim throughput is real.

**Router track (runs in PARALLEL with all phases; the 16-core-affordable floor and the likely
deliverable).** Independently of the league, train **one clean 105-d evasive specialist** and ship
`eval_router.js` with reckless = `laika-aggressive` script + evasive = that specialist. This already
measured **maximin 0.35** with zero new training infra. If the league seesaws (J.4 warns a
memoryless league may never beat the pure-aggression script, maximin ~0.50, because the residual gap
is recurrence/opponent-modelling not league mechanics), the league becomes a **specialist factory**
for the router (PFSP/RPP tells you which specialists have non-zero Nash support) — the
16-core-affordable realization of the mixed Nash.

| Phase | Effort | Expected payoff | Gate to proceed |
|---|---|---|---|
| 1 (surgery + PFSP + no-collapse gate) | ½ day + 1 block | validates the whole thesis cheaply | maximin Δ ≥ 0.07 vs J.4 |
| 2 (full league + exploiters + archive) | 1-2 days | the principled anti-seesaw ceiling | Phase 1 passed |
| 3 (RPP/Nash gate + PBT) | ~1 day | last few % + auto-tuning | optional |
| Router (parallel) | ~1 DAgger run | the reliable floor / shippable deliverable | always do |

---

## 7. Single-machine scale caveats — we are NOT DeepMind

1. **No distributed actor-learner, no concurrency.** Agents run **sequentially** as time-slices on
   one 16-env PPO; the JS bridge (~3.9 ms/decision) is the throughput bottleneck, so running M and
   an exploiter concurrently would only halve each one's throughput. AlphaStar's 12 agents × 32 TPU
   × 44 days only bought a *harder game* (StarCraft) — the **value was the algorithm** (PFSP
   weighting + exploiter resets + archive + non-regression gate), which is **scale-free**.
2. **No Nash solver in the minimal version.** RPP/`fictitious_play_nash` over a ~20×20 matrix is
   microseconds and is a Phase-3 polish, not a dependency.
3. **Noise dominates at our eval budget.** Win-rates swing with seed (documented). EMA-smooth the
   PFSP win-rates, average over ≥2 seed bases, and require gate Δ ≥ 0.07 (60-ep noise); never
   promote on the 25-ep cheap eval.
4. **`f_hard` with large p degenerates** to single-opponent overfit → the others become "forgotten"
   and the seesaw reappears one level up. Keep p≈1.5 and keep the self-play + 15% anti-forget slices
   nonzero.
5. **Reset cadence is load-bearing.** An exploiter that never resets converges to a stale counter and
   stops finding new holes. Reset to its specialist seed (or current M) on every freeze.
6. **Every member MUST inherit `AnchoredPPO` + `CriticWarmup`** or it erodes on contact (documented
   repeatedly).
7. **Obs-spec hygiene.** Frozen archive selves must all be 105-d; the `vel_bc_anchor2` lineage is
   already 105-d (no surgery), but do NOT mix any 101-d ckpt into a 105-d pool. Surgery every
   self-pool / eval opponent identically (OpenAI-Five's "frozen past agents forever play worse"
   trap).
8. **BIGGEST — ruleset confound + memoryless ceiling.** Under **survival_v2** the non-transitivity
   **dissolves**: the pure-aggression SCRIPT already hits maximin ~0.50 and beats held-out, so the
   league is **unnecessary there** and the cheap deliverable is the script (already deployed,
   JOURNAL J.x). The seesaw/league work matters for the **v1-rules multi-strategy regime**.
   **CONFIRM which ruleset you are optimizing before building any of this**, and budget the
   script+router (maximin 0.35-0.50) as the likely actual deliverable — even a *correct* memoryless
   league may not beat the script because the residual gap is recurrence/opponent-modelling, not
   league mechanics (J.4).

---

### Sources (external)
- Vinyals et al., *Grandmaster level in StarCraft II using multi-agent RL*, Nature 575 (2019):
  https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf
  (three roles; PFSP `f_hard=(1-x)^p` / `f_var=x(1-x)`; 35/50/15 mixture; >70% / reset cadences;
  FSP/archive anti-cycle; Relative Population Performance = Nash(P)ᵀ P Nash(P)). Numeric reset
  thresholds (0.7, 2e9 steps) are from the Nature Supplementary pseudocode — **not re-verified this
  session**; confirm against the Nature SI or arXiv:2104.06890 (mini-AlphaStar) before hard-coding.
- DeepMind blog: https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
- Zhang et al., *A Survey on Self-Play Methods in RL*, arXiv:2408.01072 (Sec 3.2.5 + Algo 5: PFSP
  `σ = f(P)/Σf(P)`).
- OpenAI et al., *Dota 2 with Large Scale Deep RL*, arXiv:1912.06680 (Sec 3.3 + App. B surgery
  Eq.1/Eq.6/Eq.8/Eq.9 `[W|0]`, LR=0 smooth restart, frozen-opponent compatibility).
- Raiman et al., *Neural Network Surgery with Sets*, NeurIPS 2019, arXiv:1912.06719 (feature→param
  interaction map; the surgery audit).
- Rolnick et al., *Experience Replay for Continual Learning* (CLEAR), arXiv:1811.11682 (value-head
  anchor + policy-distillation on replay).
- Jaderberg et al., *Population Based Training*, arXiv:1711.09846 (truncation select + perturb ×0.8/×1.2).
- Lanctot et al., PSRO, arXiv:1711.00832; Timbers et al., *Approximate Exploitability*,
  arXiv:2004.09677 (exploitability/NashConv ≥ 0, = 0 iff Nash).

### Sources (local, verified this session)
`runs/auto_campaign/JOURNAL.md` (I.2 router maximin 0.12→0.35; J.4 single-policy self-play seesaws,
script champion maximin 0.46-0.50); `train/train_selfplay.py` (BalancedGateCallback L46, allocate
reuse L193, anchor-to-policy L235, `assert obs==(105,)` L244); `train/league_stage1.py` (allocate
L81, LeagueCallback L110); `train/league_ppo_v2.py` (AnchoredPPO L48, CriticWarmup L85);
`train/tank_selfplay_env.py` (frozen-self opp_models.predict L143-148, load-once `__init__`,
`assert obs==105` L100); `train/eval_router.js` (classify-then-lock, --spec-script/--generalist,
warmup/approach/close-frac); `train/eval_v2_agent.js` (TRAIN/HELD maximin/mean); `game_core.js`
(OBS_SIZE=105 L84, motion dims 90-93 + trailing poison 94-104 L2030-2040); `models/auto/` (assets:
vel_bc_anchor2, league_robust_v3_best, dagger_aggro_specialist, v2_reckless_specialist_best,
v2_movers_specialist_latest, selfplay_v1_latest).

---

## Adversarial critique & minimal path

> Critique appended 2026-06-21. RESEARCH/DESIGN ONLY — nothing run. Code/asset facts below were
> re-verified in this checkout; external claims are cited inline. The thesis above is **internally
> coherent and the AlphaStar mapping is fair**, but for a single 16-core hobby box it is ~3x larger
> than the evidence justifies, and it rests on one confound that, if resolved the likely way, **deletes
> the whole league**. Read this section before writing a line of `league.py`.

### 0. The load-bearing confound — RESOLVE THIS FIRST, IT MAY END THE PROJECT

The document's own caveat #8 admits it but then proceeds anyway; that ordering is backwards. The
evidence base is **ruleset-mixed**:

- The router's headline **maximin 0.35 / mean 0.57** (I.2, the "central result") was measured **under
  survival_v2** — `eval_router.js` was invoked with `--ruleset survival_v2 ... v2 half_random`
  (JOURNAL I.2/I.3, lines 454-475). The §5 router command recipe above instead passes
  `--ruleset survival_v1`, so **it does not reproduce the 0.35 number it cites.**
- The script champion's **maximin 0.50** is **under survival_v2** (`eval_v2_agent.js` hardcodes
  `ruleset:"survival_v2"`, train/eval_v2_agent.js:31; JOURNAL I.7 line 531).
- J.4's headline "single-policy self-play **still seesaws**" run (`vel_selfplay2.log`) shows
  **no `ruleset=survival_v2` and no HP=6 markers** in its env banner (the survival_v2 stage logs and
  `eval_v2_agent` both carry those markers; this run does not) — yet J.4's *prose* concludes "cannot
  beat the script **under v2**." So the seesaw was demonstrated on a **v1-ish training run** and then
  rhetorically attached to a **v2 script ceiling**. The two halves of the motivating argument are not
  the same experiment.

Why this is fatal to the league case: caveat #8 itself states that **under survival_v2 the
non-transitivity dissolves** — the pure-aggression *script* already hits maximin ~0.50 and beats
held-out, "so the league is unnecessary there." The league is therefore only justified for a
**v1 multi-strategy regime whose seesaw has not actually been demonstrated at the velocity-obs stage**
in this checkout. **Mandatory gate-zero (about 20 min, no training):** run `eval_v2_agent.js`-style
maximin for `vel_bc_anchor2` AND the pure-aggression script under **survival_v1** (the regime the
league claims to fix). Two outcomes: (a) if the script already wins under v1 too then the deliverable
is the script + router and **you do not build the league at all**; (b) only if a real, reproducible v1
seesaw with a sub-0.3 maximin exists does any of §2-§4 earn its keep. Building Phase-1 before this
check risks spending a 6 h block validating mechanics for a regime that does not need them.

### 1. Over-engineered / cargo-culted from DeepMind scale — CUT THESE

- **The two reset-on-add exploiters (E_reck, E_evas), the registry, `hot_add_opponent`/`set_opponents`,
  snapshot-into-pool growth (all of §2 + Phase 2).** This is the single largest build and the most
  cargo-culted item. AlphaStar needs exploiters because StarCraft's strategy space is huge and
  unknown; **here the holes are already known and enumerable** (laika vs aggressive — the journal
  names them). A reset-on-add exploiter is an *automated discovery* mechanism for holes you have
  already discovered by hand. Replace the entire exploiter apparatus with a **fixed adversarial
  opponent set** (the scripts that already expose each hole) in a PFSP pool. You lose ~nothing and
  delete ~400 lines + the registry + the env-mutation API.
- **Clause (4) RPP / `fictitious_play_nash` (§4a, Phase 3).** Microsecond cost, yes — but it is
  decision-theoretic theater on a ~7-script + handful-of-selves pool. A 20x20 empirical-Nash adds no
  actionable signal a human cannot read off the maximin row. Cut entirely; it is résumé-driven
  complexity at this scale.
- **PBT layer (Phase 3).** 4-8 PBT workers on one 16-core box means each worker gets ~2-4 cores and
  the JS bridge (the real bottleneck, ~3.9 ms/decision) is shared — PBT here is *slower* serial
  hyperparameter search wearing a distributed-systems costume. Cut; hand-tune `lambda_ce`/`ent_coef`.
- **The "15% forgotten-opponent" reserved slice + EMA-smoothed PFSP + ≥2-seed-base averaging in the
  sampler (§4b).** The **no-collapse gate clause already fully covers anti-forgetting** — if an
  opponent regresses, the gate refuses promotion, which is the behavior you want. A separate sampler
  term that chases noisy per-opponent deltas is redundant and is itself a noise-chasing risk (the doc
  even warns of this in caveat #4). Keep PFSP `f_hard` as a *static-ish* reweight recomputed once per
  block; drop the EMA/multi-seed sampler machinery.
- **Sequential 3-slot roster framing.** Even kept, the "roster of 3 trainable slots run one-at-a-time"
  is just "train M against a fixed adversarial pool." The roster abstraction adds vocabulary, not
  capability, at N=1 concurrent learner.

**Corrections to the doc's own claims (factual):** §5.3 asserts "`sb3-contrib` is **not installed**."
It IS — `import sb3_contrib` -> **2.8.0** in this env. The install barrier against RecurrentPPO does
not exist; the *real* objections to GRU (weight-incompatible with the MLP anchor, BPTT-on-CPU
fragility, still one weight set) stand, but the doc should not lean on a false blocker. Also: the §3
surgery is correctly cheap because `policy.pth` is only **95 KB**; but each champion `.zip` is
**128 MB** (a 128 MB `data` blob — replay/rollout artifact). Capping the archive at "6-8 selves"
therefore means **~1 GB of disk per generation of frozen selves**, and every one must be loaded into
`opp_models` for the bridge — a real RAM/IO cost the "keep the pool small" note understates. Strip the
`data` blob (save policy-only) before archiving selves.

### 2. Most-likely failure mode, per phase

- **STEP ZERO (surgery).** Fails **silently**. The poison-block right-shift is the kind of off-by-N
  that produces a network that *loads* (strict=True passes — shapes are right) and runs but is subtly
  miswired, manifesting only as a mysteriously weak champion that the maximin gate then quietly
  rejects, costing a 6 h block before anyone suspects the surgery. The three audit asserts catch a
  *zeroing* error but **not a wrong `VEL_AT`/`N_VEL`**. Mitigation must be **behavioral**: after
  surgery, do a real rollout-parity check — feed a batch of genuine obs vectors through both nets with
  the 4 velocity columns held at the env's zero-velocity value and assert **identical argmax actions**
  before trusting the warm-start. Structural zero-asserts alone are insufficient.
- **Phase 1.** Fails by being **under-powered to detect Δ≥0.07** at the very budget it prescribes. The
  doc itself says +0.05 is inside 60-ep noise and the script champion sits *exactly* on a laika
  16/32 knife-edge (I.7). A single 200k run vs a single J.4 baseline, both seed-noisy and with **no
  seed replication on either arm**, cannot cleanly resolve a 0.07 maximin difference — you will get an
  ambiguous result and rationalize proceeding.
- **Phase 2.** Fails by **seesaw-one-level-up** (the doc's own caveat #4): with `f_hard` starving
  beaten opponents and a small pool, M cycles among the adversarial scripts exactly as the single
  policy cycled among opponents — the league mechanics do not add *commitment*, and J.4 already proved
  the gap is commitment/recurrence, not opponent-distribution. Most likely Phase 2 reproduces J.4 at
  higher engineering cost.
- **Phase 3.** Fails by being irrelevant (see §1) — RPP/PBT polish a model that the §0 confound says
  should not exist.
- **Router track.** The *least* likely to fail and the only arm with a **measured** win (0.35), but
  its one stated gap (laika floor 0.35 from the generalist) needs the "one clean 105-d evasive
  specialist" — itself a normal DAgger run that could underperform; low risk, known recipe.

### 3. Is Phase-1 the smallest + most decisive experiment? No — sharper one below

Phase 1 bundles **three** independent changes (surgery, PFSP, no-collapse gate) into one run, so a
null result **cannot attribute blame** — was it surgery mis-wiring, PFSP starving an opponent, or the
gate? That is the opposite of decisive. It is also not the smallest: it requires writing surgery +
pfsp + a gate edit (about ½ day) and burns a 6 h block before any signal.

**Smaller + more decisive sequence (each step gates the next, total < 1 block of compute):**

1. **§0 gate-zero (20 min, zero training, zero code):** script vs `vel_bc_anchor2` maximin under
   **survival_v1**. If the script already wins -> STOP, ship script+router. The single most decisive
   measurement in the whole program, and it costs nothing.
2. **Surgery rollout-parity (30 min, zero training):** surgery `league_robust_v3_best` -> 105-d, then
   the behavioral parity check from §2. Decisive for "is the warm-start even valid," isolated from
   PFSP and the gate. If parity holds you have a *free competent 105-d combat policy* regardless of
   whether the league ever happens.
3. **One-variable run (1 block):** take the strongest warm-start from step 2 and run plain self-play
   **with only the no-collapse gate added** (no PFSP, no surgery-of-the-pool, default mix). The
   no-collapse clause is a ~3-line change and is the *one* idea here that directly targets the
   documented failure (mean-offsetting trades). If even the cleanest no-collapse run still seesaws,
   PFSP and exploiters will not save it (the gap is commitment), and you skip straight to the router.
   If it *holds* laika without collapsing, THEN add PFSP as a second isolated variable.

This makes each result attributable and front-loads the two zero-cost decisive checks.

### 4. The simpler non-league path that gets ~80% (recommended)

**Yes — and the document already contains it; it just buries it as a "parallel track" instead of the
primary plan.** The router (§5 option C) is **already built, already measured at maximin 0.35 / mean
0.57 (the campaign's best multi-strategy result), and needs zero new training infra.** The entire
80%-path is:

1. **Surgery `league_robust_v3_best` -> 105-d** (offline, ~30 min, the one genuinely useful piece of
   §3) for a competent 105-d combat policy on the reckless route.
2. **One DAgger run** for a clean 105-d evasive specialist (the doc's own stated single gap; reuses
   the expert-map oracle, no architecture change).
3. **Ship `eval_router.js`** with reckless=`laika-aggressive` script + evasive=that specialist.

That is **one offline edit + one DAgger run** vs the league's ½-day Phase-1 + 1-2 day Phase-2 +
multiple 6 h blocks, and it delivers the *already-measured* 0.35 maximin — the league's own §6 table
lists the router as "the reliable floor / shippable deliverable" and admits a *correct* memoryless
league "may not beat the script because the residual gap is recurrence/opponent-modelling, not league
mechanics." If 0.35 must be pushed higher, the **cheapest** lever is not the league but the explicit
external commitment latch the router already provides — add a *third* route or tune the classifier,
not a 12-agent population. Reframe the league: **build it only if (a) §0 proves a real v1 seesaw AND
(b) the router's per-route specialists are themselves seesawing** — at which point a *minimal* PFSP
pool (no exploiters, no registry, no RPP) becomes a **specialist factory** for the router, the only
honest 16-core justification for any league code at all.

### 5. One-line scorecard

| Component | Verdict |
|---|---|
| AlphaStar diagnosis / mapping | Sound, fair |
| §0 ruleset confound | **Blocking — resolve before any build** |
| Surgery (one model, policy-only, rollout-parity) | Keep — genuinely useful, cheap |
| No-collapse gate clause | Keep — the one idea that targets the real failure |
| PFSP `f_hard` (static, once/block) | Keep, minimal form |
| Exploiters + registry + env-mutation API + snapshot growth | **Cut** (holes already known) |
| RPP/Nash clause, PBT, EMA sampler, 15% forget-slice | **Cut** (scale theater) |
| Router (already 0.35, already built) | **Promote to primary deliverable** |
