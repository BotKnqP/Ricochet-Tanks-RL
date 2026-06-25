# Autonomous Campaign Journal — Ricochet Tanks RL

**Mandate (user, 2026-06-20):** run autonomously ~6h. Goal: a policy that beats ALL scripted
opponents, then reaches self-play capability. Constraints: NEVER overwrite the user's existing
models (canon_v2, historical_*); use NEW names. Self-debug + self-strategize each round. Record
every action here. May consult literature and try bold new strategies.

**Starting point (canonical, parity-verified physics shell_decay=True):**
- Final user model `models/bc_dagger_moba_canon_v2.zip` (DO NOT OVERWRITE): 100-ep laika 0.53 /
  stationary 0.60 / easy_laika 0.38 / laika-aggressive 0.01 (unseen) / pro untested.
- Established: RL (PPO, even anchored) erodes laika → imitation-first for script coverage.
  Self-play is the explicit end-goal lever, applied last with anchor+gate protection.
- Combat scripts: stationary, easy_laika, laika, laika-aggressive(=aggressive), laika-aggressive-pro(=pro).

**Plan (adaptive):**
- R0 recon: script win-matrix → per-opponent expert map (esp. who beats laika-aggressive).
- R1: generate any missing canonical demos (expert that beats laika-aggressive / pro).
- R2: all-scripts DAgger (full pool + per-op experts + maximin best-ckpt) → 100-ep eval all scripts.
- R3+: patch weakest opponents (weights/experts/iters), iterate until all scripts >~0.5.
- R_sp: build policy-opponent (frozen .zip as red) in env; self-play league (anchor+gate).
- R_sp+: self-play vs past selves + scripts mixed.

New-model namespace: `models/auto/` (campaign outputs only; user models untouched).

---

## R0 — recon + self-play infra (2026-06-20)

**Self-play infrastructure BUILT + smoke-verified (the hard part of the self-play goal):**
- `train/rl_bridge_selfplay.js` (NEW, separate from proven vec bridge): per-env opponent driven by
  script OR by `opponentActions[i]` (external red action); returns `obs1` (red perspective) every step.
- `train/tank_selfplay_env.py` (NEW): SB3 VecEnv that caches obs1, runs frozen "self"/past-checkpoint
  policies on it (batched per model), feeds red actions back. moba1v1duel is SYMMETRIC so a blue-trained
  policy plays red from obs1. Smoke PASS: n=4 (2 script + 2 policy=canon_v2), reset+60 steps, episodes
  cycle, obs (4,101) clean. → self-play training is now just a trainer script away.

**Perf finding (important budget constraint):** sim is ~3.9ms/decision (moba physics + laika A*
pathfinding); `eval_script_bot.js` adds per-episode subprocess/IPC idle → ~9s/episode (unusable for
matrices). Fix: `train/script_matrix.js` (NEW) runs in-process via `core.advance` (no obs raycasts).
Passive-blue matchups still hit the 900-step timeout (slow); aggressive experts resolve fast. Budget:
~4-8 substantial training runs fit in 6h → be economical, prefer warm-start + on-policy DAgger labels
over re-generating demos.

**Matrix recon (20ep, blue=expert beats red=opponent), canonical physics:**
- laika-aggressive(aggro): stationary 0.95, easy_laika 0.40, laika 0.95, aggro-mirror 0.00, pro 1.00.
- laika-aggressive-pro(pro): stationary 0.70(to .30), easy_laika 1.00, laika 0.60, [aggro ?], [pro ?].
- Takeaways: aggro dominates {stationary, laika, pro}; pro dominates {easy_laika}; aggro LOSES to
  easy_laika (0.40) and the player-0 aggro mirror (0.00 — possible spawn/first-shot asymmetry favoring
  player 1; the learner sits at player 0, so laika-aggressive is structurally the hardest opponent).
- KEY pending cell: pro vs laika-aggressive (the counter to the reckless bot). If pro beats it, pro is
  the laika-aggressive expert; else easy_laika (which beat aggro ~0.60) is the defensive counter.

**Built + smoke-verified train_selfplay.py** (NEW): anchored PPO vs scripts + frozen self, critic
warm-up + CE-anchor + NEW `BalancedGateCallback` (maximin: promote `_best` only if MIN win across ALL
gate opponents improves and mean doesn't regress — generalizes the laika-only LeagueCallback to "beat
all scripts"). Smoke PASS (4 envs, 2 script + 2 self). Both trainers ready.

**Plan refinement:** all-scripts DAgger warm-started from canon_v2 (read-only), pool = 5 scripts,
per-opponent expert-map from matrix, maximin best-ckpt (already in train_dagger.py). Output
`models/auto/bc_dagger_allscripts_v1.zip`. Then self-play from that.

**KEY discovery — player-0 (blue) advantage:** matrix is asymmetric. blue=aggro beats red=pro 1.00,
AND blue=pro beats red=aggro 0.95; pro-vs-pro mirror 0.75 (blue wins). The learner is ALWAYS player 0
(blue) in train+eval, so it has this structural advantage → every script opponent has a ≥0.95 blue
expert and is therefore beatable. Final expert-map (blue expert per red opponent):
  stationary→laika-aggressive(.95), easy_laika→pro(1.0), laika→laika-aggressive(.95),
  laika-aggressive→pro(.95), laika-aggressive-pro→laika-aggressive(1.0).
Note: aggro→stationary (.95, no timeout) should fix canon_v2's stationary 37%-timeout (it used pro→stat).

Built `train/eval_all_scripts.py` (NEW): one model vs all 5 opponents, reports maximin (min win) +
mean + ALL>0.5. Same eval path as the gate.

## R1 — all-scripts DAgger (DONE) + a MAJOR confound discovered
warm-start canon_v2, 5-opp pool, expert-map above, iters=6 rollout=36 eval=20, maximin best-ckpt.
Result: best = iter6 (most on-policy), `models/auto/bc_dagger_allscripts_v1.zip` (20-ep in-training):
laika 0.50 / easy 0.55 / stationary 0.65 / laika-aggressive 0.20 / pro 0.50 — min 0.20 mean 0.48.
The aggressive pair lifted only at low beta (on-policy), still the weak floor.

**MAJOR CONFOUND — moba1v1duel outcome is strongly SEED-dependent (spawn/powerup placement):**
pro-vs-laika-aggressive = 0.95 at seed 300000 but 0.00 at seed 20240618 (verified identical under
step AND advance, so NOT a code bug — it's the seed). Consequences:
- `pro_vs_aggressive_moba_300.jsonl` was gen'd at the default seed 20240618 → **0 winning episodes**
  → the DAgger's laika-aggressive expert (pro) was effectively a LOSING expert at the demo seed.
- Every project result (canon_v2 "laika 0.53", the "blue advantage") is at ONE eval seed (300000)
  and may be spawn-lucky. Need multi-seed eval for robust numbers.
- Fix direction: characterize seed-dependence (running), then eval across multiple seed bases +
  regenerate demos at a winning, eval-disjoint seed range (or diverse seeds keeping only good_wins).

**CORRECTION — the real root cause was the QUALITY FILTER, not the seed.** seed-sensitivity test:
pro beats laika-aggressive robustly across ALL seeds (0.70/0.85/0.95/1.00/0.70 over 5 bases, avg ~0.84);
easy_laika ~0.73; laika ~0.00; aggro-mirror 0.00. So the expert-map (pro→laika-aggressive) was CORRECT.
`pro_vs_aggressive` actually had 266 WINS / 300 — but good_demo=0 because the `selfHits<=hitsDealt` gate
rejected every win (pro ricochets into itself in the chaotic duel vs reckless aggro). Fix: added
`--lenient` to gen_script_demos.js (keeps win+hitsDealt>=2, drops the self-hit rule); regenerating.
Seed-dependence IS real for some matchups → eval_all_scripts.py now averages over multiple seed bases.
Lesson for memory: the moba self-hit quality gate silently nukes valid aggressive-duel demos.

## R1b — corrected all-scripts DAgger
Enriched demos confirmed (all 6 files have good_demo: aggro_vs_laika 288, aggro_vs_pro 300,
aggro_vs_stationary 197, pro_vs_aggressive 266 [FIXED], pro_vs_easy 254, pro_vs_stationary 300).

- **v2 (fresh, --warm-start none): ABORTED.** iter0 mean 0.14, iter1 laika 0.15/easy 0.30 (vs
  warm-started v1's 0.85/0.55) — fresh start discarded canon_v2's skill; the low-fire pro_vs_aggressive
  demos taught minimal shooting. Misjudgment (I reasoned fresh avoids skew, but it threw away ability).
- **v3 (warm-start from allscripts_v1 + enriched demos):** v1 already had all 5 nonzero (min 0.20);
  warm-start + the now-correct aggressive demos should push the pair higher. iters 8, beta0 0.25,
  epochs 5. Output models/auto/bc_dagger_allscripts_v3.zip. (running, monitored)

Also built `train/eval_head2head.py` (model A blue vs frozen model B red, A win-rate) to measure
self-play generations. All campaign scripts compile.

**v3 RESULT: best = init (= v1), min 0.20 mean 0.48.** No iter beat the warm-start. Full curve shows
a relentless SEESAW: every iter, either laika OR the aggressive pair crashes below 0.20 (it5 mean 0.58
but laika 0.10; it7 pro 1.00 but laika-aggressive 0.00; it8 stationary 1.00 but easy 0.15). The DAgger
ceiling for ONE identity-blind policy = min ~0.20.

**Root-cause analysis (the crux of "beat all scripts"):** the per-opponent expert-map gives
CONTRADICTORY labels — vs laika the expert (aggro) says "charge+fire", vs laika-aggressive the expert
(pro) says "evade+1-shot". For similar early-episode states (before the opponent reveals its style)
these labels conflict, so the policy can't satisfy both and oscillates. DAgger imitates fixed labels →
stuck. **RL with an unambiguous win-reward could instead learn the CONDITIONAL policy** (read opponent
behavior from obs → adapt) that DAgger can't express. So R2 self-play is the principled next lever, not
just a capability demo. Anchor for R2 = bc_dagger_allscripts_v3 (min 0.20, all 5 nonzero).

## R2 — self-play (anchored PPO vs scripts + frozen self), maximin gate — WORKS
Hard scripts weighted in the pool (laika-aggressive 0.35, pro 0.20) so RL gets direct win-signal on the
weak opponents; CE-anchor + critic warm-up + maximin gate guard against the erosion seen in league_ppo_v2.

**RESULT — self-play is the FIRST method to improve here (gates, confirm @40ep, seed 300000):**
- baseline (v3): stat 0.70 easy 0.62 laika 0.30 laika-aggressive 0.00 pro 0.50 (min 0.00).
- @120k PROMOTED: laika 0.30→0.70; @210k PROMOTED: laika-aggressive 0.00→0.07 (min 0.07).
- The maximin gate twice promoted REAL gains — laika lifted hugely (0.30→~0.70) and the laika-aggressive
  FLOOR ticked up 0.00→0.03→0.07. Contrast: DAgger plateaued (oscillation), plain PPO/league eroded.
  Self-play (self in the pool + maximin gate) finds the conditional gains DAgger's contradictory labels
  and reward-greedy PPO could not. laika-aggressive is still the wall (~0.05-0.07) but the TREND is up.
- Saved: `models/auto/selfplay_v1_best.zip` (@210k, gated), `_latest.zip` (@229k).
- **Robust eval (50ep×2seed):** selfplay_v1_latest = stat 0.69 / easy 0.62 / laika 0.64 / laika-aggressive
  0.04 / pro 0.46 (mean **0.49**, best overall). selfplay_v1_best = min 0.06 (maximin pick). vs v3 the win
  is laika 0.30→0.64; the others rebalanced slightly down. Head-to-head selfplay_best vs v3 = 0.44 (a
  rebalance toward laika, NOT a uniform strength gain). laika-aggressive still ~0.04-0.06 (the wall).

## R3 — laika-aggressive specialist diagnostic (running)
warm-start selfplay_v1_latest, DAgger on laika-aggressive ALONE (pro expert, no competing opponents) to
find the achievable ceiling. If it reaches ~0.5 → the all-scripts conflict is the blocker (a conditional/
multi-policy design would solve it). If stuck ~0.06 → a fundamental imitation-precision wall in the fast
1-shot duel. Output models/auto/dagger_aggro_specialist.zip (DIAGNOSTIC — forgets other opponents).

**RESULT — DECISIVE:** the specialist beats laika-aggressive **0.88 robust (40ep×3 seeds:
0.85/0.93/0.85)**, while forgetting the rest (laika 0.05, stationary 0.18, easy 0.15, pro 0.01).
⇒ laika-aggressive is fully winnable by the learner; the combined model's 0.04 is the MULTI-TASK
CONFLICT, not an imitation-precision wall. (The specialist also oscillated 0.93↔0.00 across iters —
the precision duel is a knife-edge — but the 0.93 peak is repeatable + saved.)

## ROBUSTNESS CAMPAIGN (2026, user mandate: shift from countering fixed scripts to GENERALIZATION)
User observations: agent wins via 2-3 fixed per-script counter-routines, not general skill; it (1) camps
the OLD safe zone and gets poisoned, (2) ignores powerups (high-pressure 换血 style), (3) lost the pro
overlap-exploit. Goal: improve robustness to UNSEEN behaviour. Plan = small league self-play pilot (GPT).

- **A regression: rules verified** (collision, poison closes, no timeout/draw, death-ends, 8 smokes green).
  headless == browser (both use core.step repeat=2; verify_browser_cadence Δ=0).
- **B rich metrics:** added core stats shotsFired / enemyPowerups / contactSteps / deathCause / loserId
  (in info()); `train/eval_metrics.js` reports win, poison_death (learner died to poison = the camp-the-
  -safe-zone failure), time_to_kill, poison_dmg, powerups vs enemy_powerups, hit_per_shot, self_hits,
  contact_frac, ep_len. Already confirms the user's read: vs laika hit/shot 0.17, powerups 0, contact high.
- **C parameterized weird opponents** (game_core.js `paramBot` + `PARAM_PRESETS` + `randomized`): p-rusher,
  p-kiter, wall-sniper, charger, precision, spammer, counter, baiter, turtle, randomized. Knobs:
  aggression/preferredDist/fireTol/fireProb/dodge/retreatHealth/powerupPriority/kite/wallBounce/strafe.
  Deterministic per-seed (own LCG, no gameRng side-effects). No opponent id in the policy obs.
  TRAIN-weird = {p-rusher, wall-sniper, charger, spammer, baiter, randomized}; HELD-OUT (test-only) =
  {p-kiter, precision, counter, turtle} -> measures generalization to UNSEEN styles.
- **C2 baseline DONE — the generalization gap is real + quantified.** Two paramBot bugs fixed first:
  bots didn't lead shots (instakilled), then fired into walls point-blank -> self_hit suicide; fix =
  only fire on a CLEAR `findLaikaShot` solution. After fixing, the pool is a great benchmark.
  Standard: stationary 0.98 / easy 0.87 / laika 0.68 / laika-aggressive 0.10 / pro 0.37 (pro DROPPED
  from ~0.53 — the lost overlap-exploit, as the user noted).
  Weird (UNSEEN styles): agent mean 0.31, MAXIMIN 0.00 — charger 0.00, spammer 0.00, turtle 0.03,
  wall-sniper 0.13, counter 0.17, precision 0.33, p-kiter 0.37, baiter 0.53, randomized 0.63, p-rusher 0.87.
  pro (hand-coded general logic) on the same: mean 0.58 — it GENERALIZES better than the trained agent.
  Metrics confirm the user's read: powerups 0.00 vs everyone; self-hits high vs evasive (p-kiter 0.97,
  turtle 0.90); hit/shot 0.02-0.35. => narrow 2-3-routine play, fails unseen behaviour. baseline saved.

## D — small league self-play pilot (mixed pool, anchored, robust gate)
GPT round-2 advice (用户转述, 关键修正): don't make a FIXED list of weird scripts — the policy would just
"counter 12 scripts instead of 5". Make a PARAMETERIZED opponent ECOLOGY (per-episode random params =
domain randomization) + a clean HELD-OUT test set. Also: don't force powerup pickup — instead measure
"when the enemy grabs a powerup, does our win rate collapse?" (powerup vulnerability).

Adopted:
- `sampleRandomParams` widened + `ignorePoison` added; `randomized` = per-seed random opponent (verified
  varied: aggr 0.28-0.56, dist 89-221, kite/wallBounce/ignorePoison toggle).
- eval_metrics.js: added `win_enemy_pup` (win rate when enemy got >=1 powerup) = the vulnerability metric.
- **Pilot v1 ABORTED @30k** (preset-heavy: charger=0.12, spammer=0.12, randomized only 0.10 -> risks
  counter-memorization). Restarted as **v2 (`models/auto/league_robust_v2`)** with the parameterized family
  DOMINANT: script-mix randomized=0.35, presets {charger,spammer,wall-sniper,p-rusher,p-kiter}=0.05-0.06 each,
  standard scripts ~0.28. self-frac 0.35 (frozen selves [selfplay_v1_latest, canon_v2, allscripts_v3]).
- TRAIN opponents: standard + {charger,spammer,wall-sniper,p-rusher,p-kiter} + randomized.
  HELD-OUT (NOT trained, NOT gated): {precision, counter, turtle, baiter} -> the clean generalization test.
- gate (maximin) = stationary,laika,laika-aggressive-pro,charger,spammer,randomized. critic warm-up 30k,
  light CE-anchor (ce4/lambda0.5), 400k steps, eval every 50k. Promotion judged post-hoc on the FULL
  multi-criteria gate incl. held-out improvement (E).

### Harness AUDIT (5-agent adversarial review, 24 confirmed findings, run while pilot trains)
HARNESS IS SOUND — all 4 hard constraints verified: (1) obs is identity-blind (buildObservation = 101 dims
of physical state only, zero opponent name/id/preset); (2) held-out {precision,counter,turtle,baiter} absent
from training pool AND gate; (3) determinism intact (sampleRandomParams owns its LCG, botRand=sin(elapsed),
no Math.random/Date.now); (4) metrics attributed to learner id 0 (shotsFired/hitsDealt/selfHits/powerups/
enemyPowerups all guarded). paramBot fire-gate confirmed self-hit-safe. 9 presets confirmed distinct.
3 flags assessed: poison_death "simultaneous death" = NOT a bug (opponent-first-death => learner WON =>
correctly excluded). Gate pre-filter 0.05 variance slack = minor (post-hoc eval w/ larger samples is the
real gate). **ALLOCATION REALITY (real, acted on):** train_selfplay allocate() floors script-mix weights to
integer env counts over 10 script envs, so weights <0.06 -> 0 envs. REALIZED v2 pool =
{randomized:4, charger:1, spammer:1, wall-sniper:1, p-rusher:1, laika:1, laika-aggressive:1} + self[0..2]x2.
=> dropped stationary/easy_laika/pro/p-kiter from DIRECT training. This is FINE (even good) for generalization:
pool is 37.5% self + 25% randomized + 25% weird + 12.5% laika; standard scripts retained via frozen selves +
CE-anchor + protected by the gate (50k: stationary 0.97, laika 0.70, pro 0.40 holding). NO RESTART.
Bookkeeping fix: **p-kiter is now HELD-OUT too** (never trained by anchor or pilot) -> held-out =
{precision,counter,turtle,baiter,p-kiter} (compare_robust.js default updated). For FUTURE gens, use a
script-mix that allocates cleanly to the env count (e.g. each weight a multiple of 1/n_script_envs).

## E — RESULT: the league pilot GENERALIZED (both checkpoints PROMOTE on all 6 criteria)
Full-suite eval (15 opps, 30ep x 2 seeds), anchor selfplay_v1_latest vs league_robust_v2 _best(300k)/_latest(426k):
                        anchor   _best    _latest
  MEAN win              0.38     0.45     0.46
  MAXIMIN win           0.02     0.12     0.07
  HELD-OUT mean         0.284    0.358    0.414   (the untrained generalization test)
  self-hits mean        0.58     0.40     0.31
  powerup-vuln(win|pup) 0.14     0.23     0.26
HELD-OUT per-opp (anchor -> _latest): precision 0.27->0.50 (+0.23), p-kiter 0.38->0.67 (+0.28),
  turtle 0.07->0.27 (+0.20), counter 0.13->0.23 (+0.10), baiter 0.57->0.40 (-0.17, the lone regression).
Aggressive floor cracked: charger 0.02->0.17, spammer 0.02->0.13. Standard held: laika 0.68->0.73/0.82,
  pro 0.37->0.45/0.48, stationary ~1.00. _best also lifts laika-aggressive 0.10->0.25 + wall-sniper 0.12->0.23.
compare_robust verdict = PROMOTE for BOTH (held-out dWin +0.07 _best / +0.13 _latest; fixed worst-drop 0.07/0.05).
INTERPRETATION: this is REAL generalization, not rebalancing — the held-out (never-trained, never-gated) styles
rose +0.13, with multiple +0.20 deltas well beyond 60-ep noise. The user's 3 symptoms all improved: (1) narrow
2-3-routine play -> broader (held-out up); (2) high-pressure 换血 self-hits 0.58->0.31; (3) powerup vulnerability
0.14->0.26 (survives enemy powerups better). Domain-randomization league (heavy `randomized` + frozen selves +
anchor + maximin gate) is the method that worked where bare DAgger plateaued and bare PPO eroded.
TRADE-OFF: _latest = max generalization (held-out 0.414) but a wall-sniper hole (0.07) + weaker laika-aggressive
(0.12). _best = higher uniform floor (maximin 0.12, laika-aggressive 0.25, wall-sniper 0.23) but less held-out
(0.358). Both kept in models/auto/. NEXT (v3 ideas): anchor to the starting POLICY not laika demos (GPT) to
push held-out further + fix baiter/wall-sniper regressions; or combine via opponent-routing (the standing
multi-task-conflict solution). NB charger/spammer (reckless-aggressive) only reached ~0.15 — still the hardest.

## CAMPAIGN CONCLUSION (prior phase)
1. Self-play capability: ACHIEVED + verified + it improved the model (laika 0.30→0.64). Infra:
   rl_bridge_selfplay.js, tank_selfplay_env.py, train_selfplay.py (all smoke-passed + used in R2).
2. Beat all scripts: best single model `selfplay_v1_latest` clears 4/5 (mean 0.49); laika-aggressive
   0.04 in-combined but 0.88 as a specialist ⇒ the only blocker is the identity-blind single policy.
3. Clear path forward (future work): opponent-routing / conditional policy (gating net over
   specialists, or a behavior feature), using the existing specialist (0.88) + generalist.
Models in models/auto/ (user models untouched). Memory + STATUS.md updated.

## F — GPT round-3 plan executed (deploy + 3 code fixes + v3 pilot)
1. DEPLOY: browser watch now exposes a blue-model dropdown [latest / best / anchor(old) / specialist] +
   an opponent dropdown with the train-weird AND held-out styles (★). inputForRed passes script names
   straight to core.scriptedControl; export_for_browser.py emits all 4 policies; index.html cache->survival15.
   Verified via preview DOM: all 4 policies load, both dropdowns correct, 0 console errors. Default blue=latest.
2. (human-watch is the user's step — 10-20 held-out games in the browser.)
3. ALLOCATOR FIXED (league_stage1.allocate): old largest-remainder SILENTLY floored sub-1/n weights to 0
   envs. Now guarantees >=1 env per listed opponent when n>=#opponents, shares surplus by weight, and WARNS
   loudly + drops only lowest-weight when n<#opponents. (verified: v2 mix now warns, clean mix keeps all.)
4. ANCHOR-TO-POLICY (train_selfplay `--anchor-to-policy`): relabels the CE-anchor targets on the demo STATES
   with the warm-started START policy's own argmax (resists drift from the good start) instead of the narrow
   laika-family demo actions. (verified: relabeled 233128 targets w/ league_robust_v2_latest.)
5. RANDOMIZED FAMILY EXPANDED (sampleRandomParams): draws an ARCHETYPE so it frequently hits the hard styles
   — spammer-like ~26%, baiter-like ~22%, wall-sniper-like ~16%, broad ~36% — plus a seed avalanche-mix so
   SEQUENTIAL arena seeds don't draw correlated archetypes (was: 6 consecutive seeds all baiter). smokes green.
6. v3 PILOT LAUNCHED `models/auto/league_robust_v3` (600k): warm-start + anchor-to-policy from v2_latest;
   self-pool = [v2_latest, v2_best, selfplay_v1_latest] (league growth); script-mix randomized=0.34 + charger/
   spammer/wall-sniper + laika variants + stationary (8 opps, fits 10 script envs, no drop); gate = standard +
   hard weird + randomized; HELD-OUT still {precision,counter,turtle,baiter,p-kiter}. Goal: push held-out past
   v2's 0.414 + fix baiter/wall-sniper regressions, without eroding the v2 gains. Eval every 60k.

## F RESULT — v3 (anchor-to-policy + expanded family) GENERALIZED FURTHER
Full-suite eval (15 opps, 30ep x 2 seeds), v2_latest(baseline, current game_core) vs v3_best(420k)/v3_latest(622k):
                        v2_latest   v3_best   v3_latest
  MEAN win              0.45        0.55      0.59
  MAXIMIN win           0.07        0.20      0.18
  HELD-OUT mean         0.414       0.526     0.586
  self-hits mean        0.31        0.32      0.38
compare_robust: v3_BEST = PROMOTE on ALL 6 (held-out +0.11, fixed worst-drop 0.05, train-weird +0.11,
  poison_death down, powerup-vuln held 0.39, self-hits stable). v3_LATEST = HOLD (held-out +0.17 but
  powerup-vuln 0.39->0.15 [noisy 2-opp metric] + p-rusher 0.80->0.63 / randomized 0.58->0.47 dips).
HELD-OUT per-opp (v2->v3_best / v3_latest): counter 0.23->0.57/0.70, baiter 0.40->0.57/0.68 (v2 regression
  FIXED), turtle 0.27->0.37/0.43, precision 0.50->0.50/0.62, p-kiter 0.67->0.62/0.50.
v2 REGRESSIONS REPAIRED by v3: wall-sniper 0.07->0.30/0.27, baiter back up. laika-aggressive (the campaign's
  long wall) 0.12->0.32 (v3_best) / 0.63 (v3_latest!); pro 0.45->0.48/0.82.
CUMULATIVE generalization across the campaign: held-out 0.284 (anchor selfplay_v1) -> 0.414 (v2) -> 0.526
  (v3_best); maximin 0.02 -> 0.07 -> 0.20. The reckless `charger` (0.17->0.27/0.18) stays the single hardest.
DECISION: **v3_best = the deploy champion** (all-criteria PROMOTE, highest uniform floor, fixes regressions).
  v3_latest = the max-held-out / laika-family variant (cracks laika-aggressive 0.63 + pro 0.82) but with
  p-rusher/randomized/powerup trade-offs. Both kept in models/auto/. Anchor-to-policy + expanded family worked:
  it lifted the v2 weak spots (wall-sniper, spammer) AND the untrained held-out, faster (first promote @60k).

## G — survival_v2 / combat_v2 rule rework (implemented + verified; NO training yet, per user gate)
FROZEN (untouched): league_robust_v3_best (old-rule deploy champion), league_robust_v3_latest (old-rule max-gen).
Goal: long-form combat (拉扯/poison/powerups/positioning) instead of 西部对决/开局秒杀.
- S1 config (NON-DESTRUCTIVE — all knobs default to old survival_v1, so every existing scenario/model is
  byte-identical): cfg.tankMaxHp (default MAX_HEALTH=3), cfg.regenDelay (default 5s), cfg.spawnMode (default
  "fixed"). Preset `ruleset:"survival_v2"` -> HP 6 (x2), regenDelay 7.5s (x1.5), spawnMode half_random;
  individual knobs override (config.tankMaxHp / regenCooldownTicks / spawnMode). Per-tank maxHealth field;
  obs health normalizes by me/other.maxHealth so it stays [0,1] under HP x2. (verified: v1 byte-identical;
  v2 HP=6 obs healthObs=1.00 no out-of-range.)
- S2 half/full_random spawn (seeded, legal = boundary+walls+clearance, min-sep worldW*0.32; half_random
  blue LEFT / red RIGHT; at spawn poison inactive + powerups not yet placed so wall+boundary+min-dist is the
  full legality). verified 60/60 seeds: blue-left, red-right, min-sep, obs-finite.
- S3 scripts adapted: laika-pro / easy_laika / paramBot retreatHealth scaled by maxHealth/MAX_HEALTH ->
  retreat at the same HP FRACTION under v2 (identity at HP=3). HP x2 confirmed via hits-to-kill 2.3->3.8.
- S4 invariants PASS under v2 + random spawn: 40/40 combat + 8/8 idle resolve by DEATH (0 draws), deepOverlaps
  0, nanObs 0, poison fully closes (idle resolves by poison ~44s), shell_decay on. headless cadence unchanged.
- S5 sanity eval (4 OLD models under v2, no training): mean win drops to 0.22-0.33 (v3best 0.33, v3latest
  0.33, specialist 0.22, anchor 0.22) — all off-distribution + instakill crutch broken. poison_dmg jumped
  ~0 -> 0.5-1.9, poison_death ~0 -> 0.07-0.20, ttk ~1.5x (stationary 18.6->29s). Models still pick up 0
  powerups (a learned behaviour gap, not a rules gap). => confirms the rules WORK and retraining is needed.
- S6 acceptance: poison damage/death NOT ~0 (MET, ~10x up); instakill-dependence BROKEN (MET, win rates
  collapse, ttk up). powerup EVENTS happen (enemy contests, e.g. 0.27 vs randomized) but model PICKUP needs
  training. ep-length 1.5-2.5x NOT met by HP alone -> HP-sweep proves fights are AGGRESSION-limited not
  HP-limited (reckless aggro-vs-pro ~9s at HP 3/6/9 = x0.9-1.0; balanced pro-vs-laika x1.1@HP6 / x1.3@HP9).
  => longer episodes require long-form OPPONENTS (the parameterized ecology + training, Stage 3), not bigger
  HP. The CORE intent (reach the poison/resource phase, kill the instakill crutch) is achieved.
- S7 carpet regression GREEN: 5 smokes, verify_rules (v1 clean), browser survival17 loads 0 console errors
  (v3 models intact), spawn legality 60/60, collision/NaN/obs-action invariants all pass.
DECISION: rules implemented + verified + sanity-eval'd. Per the user's gate ("只有在回归测试和sanity eval都
通过后再训练"), PAUSING before training to confirm direction. Training order when greenlit: Stage1 1v1
half_random -> Stage2 full_random -> Stage3 random + parameterized ecology -> later 1v2/1v3.

## H — survival_v2 TRAINING wired + Stage 1 launched (user said 继续)
Plumbed survival_v2 options through the whole training stack (env -> bridge -> core):
- tank_selfplay_env.py + tank_env.py: new ctor args ruleset / spawn_mode / tank_max_hp -> coreCfg/reset payload.
- rl_bridge.js: forwards ruleset/spawnMode/tankMaxHp (rl_bridge_selfplay.js already spreads coreCfg). sameConfig
  is JSON.stringify so the new fields trigger core re-create.
- train_selfplay.py: --ruleset / --spawn-mode / --tank-max-hp -> BOTH the training VecEnv AND the gate eval_env
  (so the maximin gate measures survival_v2 performance). Smoke confirmed: gate baseline under v2 shows v3_best
  laika 0.00 (off-distribution, matches the sanity eval), anchor-to-policy relabel works, basic-script pool.
- **Stage 1 LAUNCHED** `models/auto/survival_v2_stage1` (500k): ruleset survival_v2, spawn half_random, warm-start
  + anchor-to-policy from v3_best; self-pool [v3_best, v3_latest]; script-mix = the 5 STANDARD scripts only (the
  parameterized weird ecology is held for Stage 3); gate = the 5 standard; self-frac 0.35, n-envs 16, max-steps
  1200 (v2 episodes run ~340-780 steps), warmup 30k (critic must re-warm for v2 value scale), eval every 60k.
  Goal: re-learn long-form combat (survive poison, contest powerups, beat reckless aggro WITHOUT instakill).
Training order ahead: Stage1 half_random -> Stage2 full_random -> Stage3 random + parameterized ecology -> 1v2/1v3.

## H.1 — v2 browser-watch toggle (while Stage 1 trains)
game_render.js: `?ruleset=survival_v2` (+ optional `?spawn=fixed|half_random|full_random`) makes the watch
core use survival_v2 (HP x2 / slower regen / random spawn). cache -> survival18. Verified: page loads clean
(0 console errors), URL param applied. Watch v2: append `&ruleset=survival_v2` to the watch URL. (Deployed
blue model is still old-rule v3 until Stage 1 produces a v2 model -> then redeploy + the user sees long-form play.)

## H.2 — Stage 1 result + RESOURCE-REWARD re-run (Stage 1b) [user chose: add resource reward]
Stage 1 (survival_v2_stage1_best, 420k) vs v3_best-under-v2 baseline: MAXIMIN 0.00->0.13 (laika 0.07->0.27),
but MEAN flat (0.33->0.32), pro REGRESSED 0.40->0.13, and **powerups picked up still 0.00**. Diagnosis:
the win-reward alone doesn't teach resource play — DUEL_REWARD powerup=0.08 is tiny vs hit=0.35/win=3.0, and
poison damage had NO reward signal (only the terminal loss). So the model fights longer but ignores powerups/poison.
FIX (reward shaping, NON-DESTRUCTIVE — fields default off): game_core REWARD_DEFAULTS += `poisonHurt: 0.0`; the
combat poison-damage tick now does `rewardDelta -= amount * reward.poisonHurt` for the learner (mirrors the
existing nav penalty). train_selfplay: `--powerup-reward` / `--poison-hurt` build a per-run reward from DUEL_REWARD.
**Stage 1b LAUNCHED** `models/auto/survival_v2_stage1b` (500k): continue from stage1_best (+ anchor-to-policy),
self-pool [stage1_best, stage1_latest, v3_best], --powerup-reward 0.3 (~1 hit) --poison-hurt 0.15, same v2 +
half_random + standard-script pool/gate. Goal: keep the v2 combat skill + finally learn to grab powerups +
avoid sitting in poison. The win-gate won't show pickup — the post-train full eval (pups/poison metrics) is the test.

## H.3 — survival_v2 review (3-agent, while Stage 1b trained) + max_steps fix
Review verdict: reward shaping CORRECT (powerup credits only the picker, poison penalty is learner-only + right
sign + proportional, poisonHurt defaults 0 so v1 unchanged, both training+eval envs share duel_reward); core-v2
CORRECT (v1 byte-identical, obs per-tank normalized, pickSpawn legal+seeded+sides correct); plumbing propagation
CORRECT. ONE CRITICAL catch: **max_steps=1200 (=80s) truncates BEFORE the poison ring fully closes (~91s + ~17s
to drain 6 HP = ~108s)** — a model trained to AVOID poison (the new poison-hurt reward!) can survive past 80s ->
episode truncates (resolve-by-health / possible draw) instead of death, biasing toward passive HP-hoarding. The
bias grows exactly as the resource reward bites. FIX: aborted Stage 1b @60k, relaunched with **--max-steps 1800
(120s)** so poison always forces death before the cap. (Minor low-risk: pickSpawn fallback point unvalidated —
rarely hit, 60/60 legality test passed; left as-is.)

## H.4 — Stage 1b verdict + v2 DEPLOY + Stage 2 launch [user: accept powerups, proceed]
Stage 1b eval (hardened: +pups +trunc_rate): trunc_rate=0.000 (max_steps 1800 fix WORKS, all resolve by death);
win MEAN 0.32->0.35(best)/0.37(latest); maximin gate 0.23 (best v2 floor: laika-aggr 0.23). BUT pups still ~0.01
-- the resource reward (powerup 0.3) was too weak to overcome combat-optimality (model wins without detouring;
trajectory rarely passes powerups). User chose: ACCEPT (model engages poison + survives enemy powerups
win|pup 0.39; proactive pickup is a nice-to-have for Stage 3's ecology) and proceed.
DEPLOY: browser now serves `v2` = survival_v2_stage1b_best; `?ruleset=survival_v2` defaults blue to the v2
long-form model (dropdown [v2,v3,v3max,anchor,specialist]). cache survival19, 0 console errors. Watch:
`...&watch=runs/auto_live&ruleset=survival_v2&red=<opp>` -> v2 model in long-form combat (switch to `v3` for A/B).
**Stage 2 LAUNCHED** `models/auto/survival_v2_stage2` (400k): survival_v2 + **full_random** spawn (both tanks
anywhere, min-sep), warm-start + anchor-to-policy from s1b_best, self-pool [s1b_best, s1b_latest, stage1_best],
resource reward kept, standard scripts. Goal: positional robustness to arbitrary spawns. Next: Stage 3 (+ ecology).

## H.5 — Reckless-wall RESEARCH (5-agent panel + synthesis) + reward-shaping impl [Ultracode, while Stage 2 trained]
The laika-aggressive/charger floor has pinned the maximin at EVERY stage (v1, v2, half/full random). Ran a
research+judge workflow (5 proposals: routing / reward-shaping / curriculum / dual-head / breed-reckless-self).
RANKED PLAN (synthesis, all infra-verified):
- WINNER (zero new code): add the SPECIALIST (dagger_aggro_specialist.zip, 0.88 vs laika-aggressive) as a FROZEN
  policy opponent via --self-pool (train_selfplay already loads .zip selves) + heavily weight scripted charger +
  laika-aggressive; maximin gate prevents the seesaw. The specialist-as-opponent is a black-box seed-robust policy,
  harder to frame-exploit than the script -> forces general evasion.
- COMBINE (the key novel lever, NOW IMPLEMENTED): range-dependent reward shaping. game_core REWARD_DEFAULTS +=
  `closeRangeHit` (penalty when the learner takes an ENEMY hit at <200px -> stop blood-trading vs rushers) +
  `cleanTrade` (bonus for landing a hit without being hit in the last 0.5s). train_selfplay flags
  --close-range-hit-penalty / --clean-trade-bonus. Both default 0 (v1 byte-identical, smoke-verified). Identity-blind.
- DEFER (research bets, most code): dual-head behaviour-gated policy (Prop4) then router+distillation (Prop1) —
  only if the cheap gate-protected mixed-pool approach stalls under ~0.30.
- SUCCESS BAR: laika-aggressive AND charger >=0.30 with laika >=0.55, stationary >=0.90 (no seesaw); verify at
  50ep x 2 seed bases (laika-aggressive is the most seed-sensitive matchup).
PLAN: apply as the reckless-focused **Stage 3** (it already adds the parameterized ecology incl. charger) +
specialist-in-pool + close-range shaping + heavy reckless weighting; or a dedicated `aggro_floor_v1` run.

## H.6 — Stage 2 DONE (negative, informative) + Stage 3 launched (reckless-focused, combines the plan)
Stage 2 (full_random, standard scripts, 400k): NET NEGATIVE — laika collapsed 0.15->0.00, mean 0.45->0.39,
nothing promoted (_best stays s1b_best). LESSON: full_random in isolation is counterproductive; it needs the
reckless reward shaping to be tractable. Stage 2_latest collapsed laika -> not used as a base.
**Stage 3 LAUNCHED** `models/auto/survival_v2_stage3` (500k) = the reckless-wall plan + the user's ecology, on
HALF_random (tractable; the plan's choice) from the strong s1b_best:
- self-pool = [s1b_best, s1b_latest, dagger_aggro_specialist (the 0.88 reckless-counter, as a frozen policy
  opponent for diversity), v3_best]; self-frac 0.40.
- script-mix = HEAVY reckless (laika-aggressive 0.18, charger 0.16) + ecology (randomized 0.18, spammer 0.10,
  wall-sniper 0.08) + standard (laika 0.12, easy 0.06, stationary 0.06, pro 0.06). 9 opps fit 10 envs (allocator).
- reward = powerup 0.3, poisonHurt 0.15, **closeRangeHit 0.12 + cleanTrade 0.15** (the close-range shaping that
  teaches the evasive 1-shot-counter vs rushers). max_steps 1800, anchor-to-policy.
- gate = stationary, laika, laika-aggressive, charger, pro, randomized. HELD-OUT (eval-only) =
  {precision, counter, turtle, baiter, p-kiter}. SUCCESS BAR = laika-aggressive & charger >=0.30, laika >=0.55,
  stationary >=0.90 (no seesaw); + held-out generalization. smoke-verified (specialist loads, reward plumbs).

## ============ 8-HOUR AUTONOMOUS CAMPAIGN (user: bold/aggressive, crack the core problem) ============
## I.0 — BIG PIVOT: opponent-ROUTING / conditional policy (the campaign's standing, never-built solution)
Stage 3 (reckless+ecology+close-range-shaping) PLATEAUED at the multi-task-conflict equilibrium: the close-range
shaping lifted reckless (charger/laika-aggressive 0.17->0.22) but laika dropped to 0.06 (120k), then laika 0.22 /
reckless 0.11 (180k) -- a SEESAW. mean ~0.39 horizontal. CONCLUSIVE across the whole campaign: one identity-blind
policy CANNOT hold both aggressive-pursuit (vs laika) AND evasive-counter (vs reckless). Stopped @240k.
=> Bold pivot to the OPPONENT ROUTER: infer the opponent's aggression ONLINE (closing-rate + close-contact from
the public state, OUTSIDE the fixed 101 obs) and route each step to a reckless-SPECIALIST or a GENERALIST.
- Built `train/eval_router.js` (inference-time router: EMA aggression score -> hard switch spec/gen, warmup default gen).
- ROUTER TEST 1 (spec=v1 dagger_aggro, gen=s1b_best, v2 half_random): the CLASSIFIER WORKS -- laika-aggressive
  routeAcc 1.00, charger 0.93 (correctly routes reckless->spec). BUT win 0.00 vs both: the v1 specialist is
  off-distribution under v2 (3HP-trained). Also over-routes for laika (routeAcc 0.13 -> classifier needs tuning).
- => Training a **v2 reckless-specialist** `models/auto/v2_reckless_specialist` (focused PPO: pool=laika-aggressive+
  charger ONLY, gate on reckless only so it over-specializes, strong close-range shaping 0.22/0.20, low CE 0.3,
  warm-start s1b_best, 400k). The router handles the rest, so forgetting laika/etc. is FINE.
- Also running: a WIDE research workflow (opponent-modeling / MoE / inference-time composition / AlphaStar league).
PLAN: v2-specialist -> tune the router classifier (separate laika from reckless) -> eval the routed policy on the
FULL suite + held-out. SUCCESS = beat BOTH evasive laika AND reckless laika-aggressive in ONE deployable system
(>=0.4 each) holding the rest. If hard-routing is insufficient -> a trained gating net (research's deeper option).

## I.1 — Router classifier FIXED + the 2-specialist insight
- The distance-EMA classifier had a FEEDBACK LOOP (routing to the aggressive specialist closes distance -> stays
  stuck on spec). FIX: CLASSIFY-THEN-LOCK -- observe during a warmup (generalist drives blue, consistent), classify
  reckless-vs-evasive, then LOCK the expert (opponent behaviour is fixed per episode). Signal = enemy APPROACH SPEED
  (its velocity toward blue, blue-position-robust): reckless 5.2-6.2 px/step vs evasive 0-3.4 (closeFrac is useless
  at the warmup window -- tanks still approaching at 3s). classify@35, approach>4.5. routeAcc now: laika 0.75,
  easy/stationary 1.0 (->gen), charger 0.83, laika-aggressive 0.67 (->spec). Research-backed (OPS-DeMo / BPR / DRON-MoE).
- INSIGHT: the single policy gets laika only 0.25 under v2 BECAUSE of the conflict (can't fully commit to aggressive
  pursuit while also holding evasive). So route between TWO specialists, each great at its half (no conflict):
  reckless->RECKLESS-spec (evasive 1-shot-counter), evasive->LAIKA/movers-spec (aggressive pursuit). Need to ALSO
  train a v2 LAIKA-specialist (focused on laika/easy/stationary, aggressive). Then the router's maximin ~= min over
  the two specialists' OWN-half win rates, not the compromised generalist's. Plan: reckless-spec (running) ->
  laika-spec -> 2-specialist router eval on the full suite + held-out.

## I.2 — *** ROUTING CRACKS THE MULTI-TASK CONFLICT *** (the campaign's central result)
KEY DISCOVERY (changed everything): under v2 the reckless-counter is NOT the evasive 1-shot (that's v1) -- it's
the AGGRESSIVE MIRROR. Script search vs reckless under v2: laika-aggressive(blue) beats laika-aggressive 0.70 /
charger 0.60; pro 0.40/0.40; laika 0.50/0.00; easy 0.20/0.10. So at 2xHP you out-slug the rusher (blue-player
advantage), you don't dodge it. (=> my close-range-shaping reckless-spec was learning the WRONG strategy, plateaued
0.23; stopped it.)
ROUTER (classify-then-lock on enemy APPROACH-SPEED@35steps; reckless-route = laika-aggressive SCRIPT, evasive-route
= s1b_best generalist), v2 half_random, 20ep x 2 seeds:
  laika 0.35 | easy 0.55 | stationary 0.93 | laika-aggressive 0.60 | charger 0.53 | pro 0.40 | randomized 0.63
  MAXIMIN 0.35  MEAN 0.57   <-- vs any SINGLE policy: maximin ~0.12, mean ~0.40 (and the eternal seesaw 0.06-0.22).
*** The router beats BOTH the reckless AND the evasive at once. Maximin nearly TRIPLED. The multi-task conflict
that blocked v1+v2+every stage is SOLVED by an inference-time opponent router. ***
NEXT (push the maximin up): the floor is now laika 0.35 (via the generalist). Train a v2 LAIKA/movers-specialist
(focused aggressive, no evasive shaping) for the evasive route. Also: DAgger a NEURAL reckless-spec from
laika-aggressive (for a pure-neural agent vs the script). Then deploy the router to the browser + full-suite + held-out eval.

## I.3 — Router DEPLOYED to browser + eval_router supports script-experts
- eval_router.js: --spec-script / --gen-script let a route use a SCRIPT (e.g. laika-aggressive) instead of a policy.
- game_render.js: `routerAction()` (classify-then-lock on enemy approach-speed@35 -> reckless=aggressive script,
  evasive=generalist policy `app.routerGen` default "v2"=s1b_best). Blue dropdown adds "★ ROUTER (beats both)",
  DEFAULT for `?ruleset=survival_v2` watch; app.router resets each round. cache survival20, 0 console errors.
  Watch the router crack both: `...&watch=runs/auto_live&ruleset=survival_v2&red=charger` (try red=laika too).
- Training a v2 MOVERS-specialist (laika/easy/stationary, aggressive, no evasive shaping) to raise the router's
  EVASIVE route above laika 0.35; then re-point app.routerGen at it + full-suite + held-out router eval.

## I.4 — *** THE CONFLICT DISSOLVES UNDER v2: PURE AGGRESSION IS A UNIVERSAL EXPERT ***
Tested the laika-aggressive SCRIPT (blue) vs every opponent under v2 half_random (12ep):
  laika 0.50 | easy 0.92 | stationary 0.92 | laika-aggressive 0.67 | charger 0.58 | pro 0.92 |
  precision 0.75 | counter 0.58 | turtle 0.50 | randomized 0.50   ==>  MAXIMIN 0.50  MEAN 0.68
*** Pure aggression beats EVERYTHING >=0.50 under v2, INCLUDING the held-out opponents. ***
This is BETTER than the router (maximin 0.35) and 4x the best self-play stage (0.12). REINTERPRETATION of the
whole campaign: the multi-task "conflict" was a v1 phenomenon (at 3 HP, reckless chargers die to evasive 1-shot).
At 2x HP + the structural blue-player advantage, the slugfest favours the AGGRESSOR -- one purely-aggressive
behaviour out-trades every style. The self-play agents never found this: the CE-anchor to mixed laika-family demos
+ the maximin gate trapped them in a COMPROMISED (under-aggressive) local optimum (laika 0.25, reckless 0.11-0.22),
and my Stage-3 close-range shaping actively pushed them the WRONG way (toward evasion). PIVOT: stop chasing a
router / two specialists; build a single PURE-AGGRESSIVE neural agent (imitate laika-aggressive under v2) -- it
should reach ~0.50 maximin, and RL fine-tuning could exceed the script. Stopped the movers-spec (obsolete).

## I.5 — Script-agent baseline (the guaranteed v2 deliverable) + DAgger seed fix
SCRIPT-AGENT (laika-aggressive for ALL opponents) on the full v2 suite (eval_v2_agent.js, 20ep/opp, half_random):
  TRAIN  laika 0.60 easy 0.95 stationary 0.95 laika-aggressive 0.65 charger 0.60 pro 0.80   -> maximin 0.60 mean 0.76
  HELD   precision 0.75 counter 0.65 turtle 0.60 baiter 0.75 (p-kiter ~)                     -> held-out maximin ~0.60
  trunc_rate 0.00 everywhere; ttk 7-17s. => pure aggression is a robust v2 agent, generalizes to held-out. This is
  the FLOOR we can deliver even with zero training (deploy the script as blue).
NEURAL version (DAgger imitate laika-aggressive under v2): first attempt warm-started s1b_best + reused the old V1
  laika-aggressive demos as seed -> REGRESSED (iter0 mean .42 -> iter2 .36, reckless .07-.13). Cause: 60k stale v1
  demos (fixed-spawn, v1 HP-timing) dominated the aggregate and taught mistimed aggression. FIX: wrote
  train/record_v2_demos.js -> fresh laika-aggressive demos recorded UNDER survival_v2/half_random (control matches
  ACTION_TABLE, obs in [-1,1], only wins kept). Restarting DAgger as a FRESH BC on the v2 demos (no compromised
  warm-start) + on-policy laika-aggressive relabel. Also: train_dagger.py now takes --ruleset/--spawn-mode/
  --tank-max-hp; export_for_browser.py skips missing models + has a "v2agent" slot; eval_v2_agent.js = the
  definitive train-vs-held-out v2 harness.

## I.6 — Neural agents are obs-capped under v2; the SCRIPT is the champion; DEPLOYED
DEFINITIVE comparison on the SAME harness (eval_v2_agent.js, v2 half_random, 12-20ep, full suite + held-out):
  selfplay1 (orig)        OVERALL maximin 0.00 mean 0.25
  oldspec (v1 aggro spec) OVERALL maximin 0.00 mean 0.26
  s1bbest (v2 generalist) OVERALL maximin 0.04 mean 0.39
  v3best  (old champion)  OVERALL maximin 0.13 mean 0.37   <- best NEURAL single policy
  router  (script+gen)    OVERALL maximin 0.35 mean 0.57
  *** script-agent (pure aggression) OVERALL maximin 0.60 mean 0.73  <- v2 CHAMPION (4.6x the best neural) ***
WHY neural caps out: 3 DAgger attempts to imitate laika-aggressive all failed -- warm-start+stale-v1-demos
regressed (.42->.36), fresh-BC under-fired (fire 2.6% -> .19), warm-start+fresh-demos+fw5 DIVERGED (loss .3->2.0).
Root cause = the documented "needs motion obs" wall (task #33): the 101-dim obs has positions but no enemy VELOCITY,
so a neural policy can't reproduce the script's precise pursuit of an EVASIVE target (laika neural ~0.20 vs script
0.60). The SCRIPT wins because it reads full game state. => under v2 the answer is BEHAVIOURAL (pure aggression),
and the script embodies it best; the neural ceiling is the obs, not the strategy.
DEPLOYED to browser (survival21): blue dropdown now leads with "★ pure-aggression (v2 champ)" and it's the DEFAULT
for ?ruleset=survival_v2 watch (then router, neural v2 agent, generalists). game_render inputForBlue returns the
"aggressive" script for liveModel==="aggro". Verified the code path IN-BROWSER (drove core.step synchronously):
blue="aggressive" vs stationary = 6/6 under v2 (matches headless) -> browser==headless parity holds. (The live rAF
watch only animates when the preview tab is FOREGROUND; document.hidden=true pauses it -- environment artifact, not a bug.)
A gentle DAgger (lr 1e-4, fw 2.5) runs as a best-effort neural attempt; the script-agent is the deliverable regardless.

## I.7 — CAPSTONE: adversarial verification (workflow) + MORNING_REPORT.md
Ran a 5-agent verification workflow (fresh held-out seeds 500000/650000) that SHARPENED the optimistic numbers:
- SCRIPT champion (pure aggression): OVERALL maximin 0.50 (NOT 0.60), mean 0.73, HELD-OUT maximin 0.56. trunc 0.00
  everywhere. CAVEAT: maximin sits EXACTLY on 0.50 (laika 16/32 tie -> knife-edge, fragile to seed). "Beats
  everything" is the LOOSE sense (>=~0.5), not a blowout.
- ADVERSARIAL worst-case: hardest = charger (0.56/0.53/0.41/0.51 across runs ~0.50 coin-flip; one 120ep dip to 0.41,
  200ep back to 0.51) then spammer (0.53-0.63). NO style sustains <0.4 vs pure aggression.
- BEST NEURAL (s1bbest) fresh-seed: OVERALL maximin 0.19 mean 0.39, collapsing opponent charger 0.19. ~2.6x weaker
  than the script. Confirms the neural cap.
- OBS-BLIND AUDIT: obs = 101-dim, identity-blind confirmed (9 self +6 power +10 enemy +32 rays +21 shell +10 powerup
  +2 round +11 poison). TRAIN/HELD split is a reporting label only. Shell owner flag is self/other, not opponent id. CLEAN.
Full writeup -> runs/auto_campaign/MORNING_REPORT.md (headline, the v2-dissolves-conflict discovery, comparison table,
deploy, honest caveats, next steps). DELIVERABLE = pure-aggression script as the survival_v2 champion, deployed
(survival22 default). Neural agents obs-capped (need enemy velocity, task #33). Carpet regression green (smokes +
verify_rules 7-1 by death, deepOverlaps 0). Frozen models untouched (v3_best/_latest, user canon_v2).

## I.8 — PLAY-VS-AGENT: you can now battle the agent live (survival_v2)
Added a human-vs-agent mode to game_render.js (cache survival23). Open from the index "🎮 对战智能体 Play vs the
Agent" links, or `?...&ruleset=survival_v2&play=1` (you=RED, Arrows+Enter) / `&play=1&side=blue` (you=BLUE, WASD).
Design: play mode = the watch path with botEnabled=false, so the AGENT drives one tank (inputForBlue/inputForRed ->
new agentAction(seat,model): aggro->"aggressive" script, neural->policyForward(buildObs(seat))) and YOU drive the
other via humanControl(seat). In-page panel: "agent" picker (aggro/router/v2/v3/v3max/anchor/specialist) + "swap side"
button + restart + a "You X : Y Agent" score; speed locked to 1x (real-time); banner shows your color+keys. Champion
(script) runs at full 60Hz (great feel); neural agents keep their RL cadence (15Hz). VERIFIED in-browser: UI correct
(agent picker, swap, no opponent/speed dropdowns), watch mode unaffected (still blue/opponent/speed), 0 console errors,
neural policies loaded (v2/v3/...); core.step accepts BOTH seats as either "aggressive" or a {throttle,turn,fire}
control object (what humanControl returns) -> agent-blue vs idle-human-red 3/3 and swapped human-blue vs agent-red 3/3
resolve cleanly under v2. (Live rAF only animates in a FOREGROUND tab; the preview tab is backgrounded.) Carpet smokes
+ verify_rules green.

## I.9 — v2 HP-bar visualization fix + laser buff (1.5x)
- HP BAR (game_render.js): was hardcoded to MAX_HEALTH(3) -> under survival_v2 (HP x2 = 6) the bar overflowed/mis-ticked.
  Now uses `tank.maxHealth || MAX_HEALTH` for both the fill ratio AND the per-HP tick marks (6 ticks under v2, 3 under
  v1), plus a green->amber->red fill by remaining fraction for the long v2 fights. VERIFIED: v2 state reports
  tanks[].maxHealth=6 (v1=3), so the bar divides correctly.
- LASER BUFF (game_core.js): all hits dealt a flat 1 dmg (weapons differed only by fire-rate). Added LASER_DAMAGE=1.5
  and damageTank now removes `hitType==="laser" ? 1.5 : 1` -> a laser hit kills in 4 under v2 (6/1.5) / 2 under v1.
  HP is now fractional after a laser; bar/death(<=0)/regen(min)/obs(clamp /maxHealth) all already handle that. index
  Power-Ups note updated ("Laser: ... (1.5x damage)"). Verified by code path (fireLaser->damageTank(...,"laser")) +
  smoke_core/moba1v1duel/script_bot + verify_rules all GREEN (no regression; fractional HP safe). cache survival24.

## J.1 — CONSTRAINTS LIFTED: velocity obs (OBS_SIZE 101->105) + ace opponent + powerup reward + self-play
User removed the OBS_SIZE freeze + gave 4h: goal = an agent that beats ALL scripts AND pressures a human.
Attacking the campaign's two root limits: (1) "needs motion obs" wall + (2) imitation can't exceed its teacher.
- OBS: added 4 motion dims (self+enemy velocity in the agent's HEADING frame, normalized) -> OBS_SIZE 101->105.
  Tanks now track per-step velocity in updateTank (after wall/tank blocking, so a blocked rusher reads ~0).
  Kept BEFORE the poison block so poison stays the trailing 11. Verified: obs len 105, in [-1,1], moving fwd reads
  self_fwd=1.0/lat=0, idle reads 0, enemy idle=0. All 8 smokes + verify_rules GREEN; env 105-dim roundtrip OK.
  Updated every 101->105 hardcode (tank_env/selfplay_env/vec_env asserts, train_bc/train_dagger/train_selfplay,
  evaluate_shooting_lab_bc, league_*, 8 smoke files).
- OPPONENT: added "ace" = aggressive point-blank powerup-brawler (the human-style threat: aggression beats evasion
  under v2, so the tough opp is aggressive+resourceful, NOT a kiter). aggressive script only beats ace 0.50 (vs
  charger 0.69) -> ace is the new HARDEST opponent. + powerup-reward 0.15 so the agent contests powerups.
- PIPELINE: re-recorded 109 laika-aggressive demos in the 105-d obs -> BC anchor (val_acc 0.51) -> PPO self-play
  (train_selfplay) warm-started from it, pool weighted to ace 0.27/laika-aggressive 0.2/charger 0.15 + 30% self,
  CE-anchor lambda 0.3 to the demos, maximin gate on {laika,laika-aggressive,charger,ace,pro}, 1.3M steps. RUNNING.

## J.2 — velocity-obs self-play did NOT converge in 4h; infra is the deliverable + continuation recipe
The velocity-obs PPO self-play (3 configs tried: gentle, aggressive, rebalanced+CE0.2) oscillated at mean ~0.10-0.17
on the HARD gate opponents and PROMOTED NOTHING (every gate failed). The agent did best vs aggressive opps
(ace 0.22-0.28, charger 0.17-0.22) and worst vs evasive laika (0.00-0.11) + skilled pro -> it had not yet learned
to USE the velocity to lead shots against an evasive target. ROOT CAUSE = training wall-clock: RL from a near-scratch
BC anchor (the BC only saw evasive-opponent demos + under-fired, so it started at min 0.05/mean 0.11) needs many hours
to learn precise aiming; the campaign's working self-play (R2: laika 0.30->0.64) started from a DECENT anchor. 4h
covered BUILDING + VERIFYING the infrastructure, not training a champion on top of it from scratch.
WHAT'S SOLID (reusable): OBS_SIZE 101->105 with ego-relative self+enemy velocity (removes the "needs motion obs"
wall); "ace" aggressive-powerup-brawler opponent; powerup reward; full 105-d pipeline (every 101 hardcode fixed,
all smokes green). The pure-aggression SCRIPT champion (maximin 0.50) is unaffected (script-based) + still deployed.

### CONTINUATION RECIPE (to finish the agent with more wall-clock)
1. Better anchor: record laika-aggressive demos vs ALL 7 opponents (incl ace/charger/pro) in the 105-d obs
   (`node train/record_v2_demos.js --opps laika,easy_laika,stationary,laika-aggressive,charger,laika-aggressive-pro,ace
   --episodes 80 --out data/expert_demos/v2vel/aggro_all.jsonl`), then a fire-weighted BC (fw 6-8, epochs 20).
   A stronger anchor (~0.3-0.4) is the single biggest lever -- self-play bootstraps from it, it can't from 0.11.
2. Then `train/train_selfplay.py --anchor <bc> --data-glob <demos> --ruleset survival_v2 --spawn-mode half_random
   --script-mix "<balanced, laika 0.2>" --gate-opponents laika,laika-aggressive,charger,ace,laika-aggressive-pro
   --self-frac 0.3 --lr 5e-5 --ent-coef 0.01 --lambda-ce 0.2 --powerup-reward 0.12 --total-timesteps 3-5M` and let
   it run 6-10h. Watch for the gate to PROMOTE (min rises). The velocity obs should let it finally beat evasive laika.
3. Deploy: export the 105-d model, drop the old 101-d browser models (obs-incompatible), wire it as the play-vs-agent
   neural option.

## J.3 — ENDGAME: champion confirmed + browser obs-fix deployed (survival25)
Fresh-seed eval (eval_v2_agent, 14ep x 2 seeds, 105-d obs): laika-aggressive CHAMPION maximin 0.46 mean 0.74
held-out 0.61; ace-as-agent 0.07 (powerup-detour + kite HURTS under v2's aggression-favoring dynamics); vel_agent
(self-play) 0.04. => laika-aggressive stays the champion by a wide margin; neither ace nor the under-trained
self-play agent beats it. DEPLOY: the 105-d obs makes the old 101-d neural browser models incompatible, so the
agent options are now SCRIPT-based (obs-independent): play-vs-agent + watch offer [aggro champion, ace, charger,
laika, pro]; agentAction returns the script name; default=aggro; cache survival25. VERIFIED in-browser: obs=105,
champion beats idle 3/3, 0 console errors, agent picker correct. Carpet regression green.
NET FOR THIS 4h: removed the OBS_SIZE freeze + built/verified the velocity-obs + ace + powerup-reward + 105-d
pipeline (the real reusable unlock); the trained agent on top needs the longer run in the J.2 recipe. The
pure-aggression champion (maximin ~0.46-0.50, human-intermediate) remains deployed + playable.

## J.4 — DEFINITIVE: velocity obs improved the neural BASE 3x, but single-policy self-play still SEESAWS (can't beat the script)
Executed the J.2 recipe fully: parallel-recorded 314 laika-aggressive demos vs ALL 7 opponents (105-d) -> strong BC
anchor (full-suite mean 0.35, vs the evasive-only anchor's 0.11; gate-opponent baseline min 0.12/mean 0.21, DOUBLE
the failed run's 0.05/0.11) -> long self-play (3M-step budget, charger/ace-weighted pool, lambda-ce 0.2).
RESULT: 3 gates @60k/@120k/@180k all SEESAWED -- mean flat ~0.19, no promotion, confirm-min 0.10 (even slightly
below the anchor). Each gate a different opponent is the weak one (laika 0.00->0.22, laika-aggressive 0.28->0.11,
ace 0.44->0.22, charger 0.22->0.11): the policy keeps trading the evasive half for the aggressive half instead of
committing to the one behaviour (pure aggression) that beats BOTH. This is EXACTLY the Stage-3 "conflict equilibrium"
(task #69) -- now CONFIRMED to persist even WITH velocity obs + a 3x-better anchor.
CONCLUSION: a single identity-blind NEURAL policy cannot beat the pure-aggression SCRIPT (maximin 0.46-0.50) under v2.
The script wins because it's a FIXED behaviour that never reacts/seesaws; an RL policy, by reacting to the obs,
over-adapts per-opponent and seesaws. Velocity removed the *aiming* ceiling (the base tripled) but not the
*commitment* problem. To actually exceed the script would need a different axis: recurrence/opponent-modelling (so it
can recognise + commit), a much longer (10h+) run that *might* consolidate, or just keeping the script (already the
best). Stopped the run. Script champion stays deployed (survival25, play-vs-agent works). Best neural artifact =
vel_bc_anchor2.zip (105-d, full-suite mean 0.35) for any future routing/research.

## J.5 — CHECK ① (the critique's gate-zero): the league is the WRONG TOOL for the v2 game
Ran pure-aggression SCRIPT + vel_bc_anchor2 maximin under survival_v1 (fixed spawn, HP=3), full suite + held-out
(eval_v2_agent + new --ruleset flag, 16ep x 2 seeds):
- NEURAL vel_bc_anchor2 under v1: 0.00 on EVERY opponent -> it is v2-trained (HP=6/random spawn), totally off-dist
  under v1; the "neural v1 seesaw" arm is moot (we have no 105-d v1-capable net without surgery).
- SCRIPT under v1: laika 0.97, stationary 0.97, pro 1.00, precision 0.91, p-kiter 0.97 (CRUSHES evasive) BUT
  laika-aggressive 0.00, charger 0.03, easy 0.31, counter 0.41, turtle 0.28 -> OVERALL maximin 0.00, mean 0.60.
  Under v1 the aggressive rusher DIES to other aggressors (3 HP -> mirror is lethal); no single fixed behaviour wins.
=> RULESET DICHOTOMY: v2 (the DEPLOYED game) script maximin 0.46-0.50 -> conflict DISSOLVES, a fixed behaviour wins;
   v1 script maximin 0.00 -> REAL multi-strategy conflict. The league/PFSP/exploiter machinery addresses the
   MULTI-STRATEGY case (v1), NOT the v2 case. The v2 neural SEESAW (J.4) is therefore a COMMITMENT problem (PPO won't
   lock the one winning behaviour=pure aggression), not a multi-strategy problem -- and a league does not fix commitment.
DECISION: do NOT build the league for v2. For a NEURAL v2 agent the right tool is SCRIPT DISTILLATION + commitment
(strong fire-weighted DAgger that LOCKS pure aggression; caps at ~script level 0.46, cheap = 1 run), or keep the
deployed script. The league is justified ONLY if optimizing the v1 ruleset (a different game than what's deployed).
The 20-min check saved building the whole league for the wrong regime -- exactly what gate-zero is for.

## J.6 — v1.5 test (v1 numbers + half_random spawn): v3_best does NOT transfer well; my "adapts well" call was wrong
Built surgery_obs101to105.py (insert 4 ZERO velocity cols @ idx 90, right-shift poison to 94-104; parity 100%
argmax over 400 states -> velocity proven ignored -> surgered net == original v3_best behaviourally). Surgered
league_robust_v3_best -> _105.zip. Evaled under v1(fixed) vs v1.5(half_random), laika-family pool:
  v1 FIXED (training regime): laika 0.72 easy 0.72 laika-aggressive 0.38 stationary 1.00 -> maximin 0.38 mean 0.64
  v1.5 RANDOM spawn:          laika 0.16 easy 0.53 laika-aggressive 0.16 stationary 0.97 -> maximin 0.16 mean 0.42
=> SIGNIFICANT degradation: laika 0.72->0.16 (collapses), maximin 0.38->0.16 (more than halved), mean 0.64->0.42.
PREDICTION WAS WRONG: I expected "adapts well, modest dip, retains v1 level" because I assumed the spawn was a minor
ego-relative shift and HP (not spawn) was what broke models under v2. ACTUAL: a FIXED-spawn-trained model OVER-FITS
the opening -- v3_best learned opening maneuvers tuned to the always-same start positions; random spawn miscalibrates
them, and the EVASIVE laika matchup (where opening positioning is decisive) collapses hardest. The ego-relative obs
helps the ongoing fight but the OPENING is off-distribution. LESSON: for v1.5 you must TRAIN under half_random, not
reuse a fixed-spawn model. CHEAP FIX (untested): a short continual-train of v3_best_105 under v1.5 (OpenAI-Five
post-surgery warm window) should re-adapt the opening + recover most of it -- same numerics, only the spawn to re-learn.

## J.7 — v1.5 continual-train ERODED v3 (2nd wrong prediction); reusing fixed-spawn models for v1.5 is a dead end
Short PPO continual-train of v3_best_105 under v1.5 (self-anchor --anchor-to-policy, laika pool, lr 3e-5, lambda-ce 0.3):
  baseline(v3_105 @v1.5): min 0.15 mean 0.29 (laika 0.28, laika-aggressive 0.15)
  @40k: min 0.00 mean 0.24   @80k: min 0.06 mean 0.21  -> DECLINING, not recovering. Stopped.
So the "cheap fix" (surgery + short continual-train re-adapts the opening) ALSO failed -- PPO eroded v3's combat
faster than it re-learned the v1.5 opening (the campaign's recurring PPO-erodes-warm-start). The self-anchor to V2
demo STATES + a gentle lr did not hold it. Two predictions wrong in a row (v3 adapts well -> no; continual-train
recovers -> no). HONEST CONCLUSION: reusing a fixed-spawn model for v1.5 is a dead end -- the opening over-fit is real
and a naive continual pass erodes rather than adapts. v1.5 = v1 (the MULTI-STRATEGY conflict regime, script maximin 0.00
under v1) + random spawn (harder, models over-fit the fixed opening). So v1.5 inherits the v1 conflict (needs
league/routing) AND the spawn-robustness challenge -- a HARDER regime than v1 for a single policy. A real v1.5 agent
must be trained UNDER half_random from the start (not transferred), and likely with routing (since the multi-strategy
conflict is present). The surgery tool itself is validated (parity 100%) and reusable; the transfer-by-continual-train hope is not.

## J.8 — v1.5 laika-family campaign (GOAL: neural maximin >0.5 under random spawn)
User set a FOCUSED goal + unlimited autonomous time: a NEURAL agent under v1.5 (survival_v1 + half_random) beating the
laika family {laika,easy_laika,stationary,laika-aggressive,laika-aggressive-pro} with maximin >0.5 (script ref ~0.5-0.56,
seed-dependent). Restricting to the laika family removes the extreme charger/ace seesaw -> milder conflict.
Pipeline: (1) recorded 638 win-demos UNDER v1.5 (laika-aggressive expert vs laika family, half_random) so the BC learns
RANDOM-SPAWN openings; added --ruleset to record_v2_demos.js. (2) BC anchor v15_bc_anchor (val_acc 0.527, fire_w5, excl
stationary bulk): v1.5 laika-family maximin 0.22 mean 0.26 -- WEAK (imperfect clone) but ALREADY > surgered v3_best (0.16),
confirming the opening WAS the gap. (3) Self-play Option B: anchor=v3_best_105 (strong combat) + CE-anchor to the v1.5
laika+easy DEMOS (teaches the random-spawn opening; ALIGNED with the objective, unlike the self-anchor-to-v2-states that
eroded in J.7) + maximin gate on the laika family. lr 1e-4, lambda-ce 0.4, 400k steps. This mirrors the R2 setup that
WORKED (laika 0.30->0.64). BUG FIXED en route: train_selfplay --data-glob didn't comma-split (demos=0); now splits like train_bc.

## J.9 — DIAGNOSIS workflow verdict (WHY_RANDOM_SPAWN.md, 4-agent + adversarial critic, all numbers re-verified)
The fixed-spawn "PPO beats script" was an ARTIFACT on BOTH sides: script's laika-aggressive 0.00 @fixed = a DETERMINISM
artifact (mirror-symmetric start game_core.js:2230 + id-phased tiebreak :1544/1601/1730/1900 -> blue id0 loses the
aggressive mirror EVERY seed); random spawn breaks the symmetry -> script maximin 0.00->0.50. v3's 0.38 was opening-overfit
(->0.16 random). RANKED VERDICT: PRIMARY=(c) COMMITMENT/multi-task-conflict — a single identity-blind feedforward PPO can't
COMMIT to the one fixed behaviour (pure aggression) the script wins with; it reacts+seesaws (J.4). But SOFT (specialist 0.88,
router 0.12->0.35). KEY: pure aggression (script) beats the WHOLE laika family (laika 0.63 AND laika-aggressive 0.67) -> NO
real conflict for a COMMITTED-aggressive policy; the conflict is an artifact of the net reacting instead of committing.
CONTRIBUTING: (c-2) PPO erodes warm-starts, (c-3) opening-overfit under DR, (b-reward) ZERO opening shaping (approachCoef=
aimBonus=backwardPenalty=0 train_moba1v1duel.py:32) -> sparse terminal credit can't teach the spawn-conditional opening.
REJECTED: (a) no-mother (script IS the mother, 0.50; imitation caps AT it), (b-state) obs (absolute board pos in obs[0:2],
K=1 98% learnable), (b-action) Discrete(18) (script lives in it). DECISIVE EXPERIMENT: single-opponent laika under random
spawn from scratch -> beats script's laika 0.63? YES=conflict(c), fix=composition; NO=spawn/reward(b) independent blocker.

## J.10 — Option B SEESAWED as diagnosed; pivot to the two robust paths (DAgger-clone + specialists)
Option B (single policy, multi-opponent laika family, v3 anchor + CE-anchor) @50k confirm: laika 0.28->0.15 DOWN,
laika-aggressive 0.15->0.25 UP, min flat 0.15 (gate fail). The EXACT seesaw the diagnosis predicted -- traded the
evasive half for the aggressive half, the conflict equilibrium. KILLED (350k more steps = the diagnosed dead-end).
Pivoted CPU to the two robust paths the diagnosis endorses:
  TRACK 2 (v15_laikaspec): laika-ONLY specialist under random spawn (the decisive experiment + a router component).
    @50k PROMOTED laika 0.10->0.20-0.30 (climbing toward script's 0.63) -> a single net CAN commit vs ONE opponent.
  TRACK 3 (v15_dagger_clone): DAgger-clone the COMMITTED-aggression script (laika-aggressive) under random spawn vs
    the whole laika family, warm-started from v15_bc_anchor. The DIRECT path: imitation caps AT the script's ~0.59,
    which CLEARS the >0.5 goal. fire_weight 6, 8 iters, beta0 0.4. Fixes the BC's covariate shift (0.22 -> toward 0.59).

## J.11 — v1.5 laika-family: the NEAR-NASH ceiling (maximin >0.5 infeasible; mean ~0.5 is the neural ceiling)
Mapped the achievable ceiling with script-vs-script under v1.5 (half_random):
  Best COUNTER to the aggressive rushers (across laika/easy_laika/ace as blue): laika @ 0.47 vs laika-aggressive, 0.38 vs pro.
  No script beats the aggressive opponents >0.5 -> laika-aggressive & pro are NEAR-NASH coin-flips under random spawn (the
  fixed-spawn determinism that made them lopsided is gone). So ANY single fixed strategy caps at maximin ~0.42-0.50 on the family.
=> maximin >0.5 ("don't lose to any") is INFEASIBLE on this family under random spawn -- a GAME property, not a neural limit.
   mean (总体胜率) >0.5 IS achievable (script mean 0.62). Best NEURAL so far = the DAgger clone v15_dagger_clone (it6):
   PROPER eval (72ep/opp): laika 0.38 easy 0.67 laika-aggressive 0.24 stationary 0.85 -> mean 0.49, maximin 0.24 (mirror-capped).
   The clone commits to aggression (right vs evasive) but a degraded aggression-copy LOSES the mirror to the original (0.24 vs script's ~0.5).
   RL erodes/seesaws (Option B 0.33, Track4 shaped laika 0.05); IMITATION beats RL here (committed strategy > reactive-optimized).
   Pro is NOT a better expert (maximin 0.42<0.50). Next: 2nd DAgger round (lower beta) to nudge the clone clearly past mean 0.5 + deploy.

## J.12 — DEPLOYED the v1.5 neural clone (playable) + the honest conclusion
Wired v15_dagger_clone (105-d, mean ~0.49-0.51 on the laika family @ random spawn) into the browser: model_weights_v15clone.js
(window.RICOCHET_POLICIES["v15clone"]), agentAction routes loaded neural policies through neuralAction/policyForward (105-d obs),
default agent = v15clone when loaded, dropdown lists it first, new v1.5 play link (?spawn=half_random&play=1 -> v1 numbers + random
spawn), cache survival26. VERIFIED in-browser (preview): policy loads (3 layers, 105-d in), banner "agent (v15clone) drives BLUE",
0 console errors. The deployed JSON == the evaluated clone.
CONCLUSION of the v1.5 campaign: the goal "总体胜率(mean) >0.5 vs the laika family @ random spawn" is at the NEURAL CEILING and the
clone meets it (~0.5). maximin >0.5 ("beat every opponent") is INFEASIBLE -- the aggressive opponents are near-Nash coin-flips even
for the best script (best counter laika: laika-aggressive 0.47, pro 0.38). Imitation (DAgger clone) BEAT every RL approach (which
seesawed to ~0.33) because the script's committed aggression beats RL's reactive optimization. 2nd DAgger round did not beat round-1.

## J.13 — User pivot: clone aggression, REMOVE laika-aggressive from the pool/demo -> dissolves the wall
User directive: clone the aggressive playstyle, get high win rate vs the OTHER scripts, league/improve, and at open-source/demo
time REMOVE laika-aggressive (the near-Nash mirror AND the cloned expert -> circular). KEY: removing it removes BOTH the
coin-flip cell AND the seesaw-driving conflict. On the reduced pool {laika,easy_laika,stationary,laika-aggressive-pro}:
  SCRIPT (laika-aggressive as blue): laika 0.63 easy 0.88 stationary 0.97 pro 0.54 -> maximin 0.54 (achievable!).
  CLONE (v15_dagger_clone): laika 0.38 easy 0.67 stationary 0.85 pro ~0.30 -> mean ~0.55 (MEAN GOAL MET), maximin ~0.30 (pro=weak cell).
Two tracks from the clone, reduced pool (gate excludes laika-aggressive): (A) DAgger to lift pro/laika toward the script;
(B) self-play/league -- the conflict that seesawed Option B is GONE, so RL may now lift instead of erode. Keep best.

## J.14 — Reduced-pool result: LEAGUE (self-play) worked once the conflict was removed; deployed spB_latest (mean 0.59)
Proper eval (72ep/opp) on the reduced pool {laika,easy_laika,stationary,laika-aggressive-pro}:
  clone(==dagA):   laika 0.38 easy 0.67 stat 0.85 pro 0.35 -> maximin 0.35 mean 0.56
  spB_best:        laika 0.21 easy 0.76 stat 0.89 pro 0.33 -> maximin 0.21 mean 0.55
  spB_latest:      laika 0.32 easy 0.68 stat 0.90 pro 0.47 -> maximin 0.32 mean 0.59  <- BEST mean, DEPLOYED
Removing laika-aggressive removed the seesaw enough that self-play LIFTED the mean 0.56->0.59 (pro 0.35->0.47, the first time
RL improved on the clone). Track A's reduced DAgger added nothing (_best == the clone init; DAgger was outpaced, only 5 iters).
laika stays the neural wall (~0.32-0.38) -> maximin still <0.5. Deployed spB_latest as v15clone (also saved v15_league_agent.zip).
Honest: mean 0.56->0.59 is ~1 SE (modest), the pro 0.35->0.47 lift is ~2 SE (real). For >0.5 MAXIMIN you'd need the router/laika-specialist; for "总体胜率" this clears it.

## J.15 — 4h autonomous: attack the laika wall (the only cell <0.5). Added NON-potential proximity reward.
Visualization DONE first: 👁 Watch mode (static deployed-agent spectate) — banner/win-rate/opponent+speed dropdowns, verified
in-browser (banner "v15clone (BLUE) vs laika (RED)", rounds score, 0 errors, screenshot). Bug fixed: app.watch.runDir defaulted
to "runs/live_demo" -> ?watch=1 now nulls it for static spectate (no live-training poll). index.html: 👁 watch links; cache survival28.
Then the laika-wall campaign: added REWARD_DEFAULTS.proximityBonus/proximityRange (game_core denseReward) — a per-step bonus for
staying within proximityRange of a VISIBLE enemy. NON-potential (changes the optimum toward cornering dodgers, unlike approachCoef
which preserved it). Verified: core reads it (-0.28 -> 1.22 with bonus), v1 byte-identical when 0. Plumbed --proximity-bonus/-range
into train_selfplay. TWO laika-specialist tracks from v15_league_agent (vs LAIKA only): (A) prox-RL self-play (proximityBonus 0.012,
range 180, gate on laika); (B) laika-focused DAgger (clone aggressive vs laika). Bar to break: laika 0.32 -> >0.5. If either breaks
it -> build router (league agent + laika-spec) for maximin >0.5. Else wrap up honestly (laika may be a true neural wall under random spawn).

## J.16 — proximity reward did NOT break the laika wall (gates were seed-luck); one more aim+prox attempt then wrap
HONEST 120-ep eval (seeds 300/500/700/900k) of all laika-specialists vs laika: league 0.28, dagger 0.26, prox_best 0.30,
prox_latest 0.28. The prox-RL gates that touched confirm-0.45 were SMALL-SAMPLE SEED-LUCK; the real laika number is ~0.28-0.30,
unchanged from baseline. So the proximity reward (NON-potential point-blank pressure) did NOT break the wall. laika is now a
ROBUST neural wall at ~0.30 across RL / clone / league / shaped / proximity -- the precise predictive-aim-vs-dodger skill is
unlearnable from sparse reward + uncloneable. Final attempt: stronger proximity 0.018 + aim 0.005 together (aim targets the
gun-on-dodger problem directly). If the honest eval still <0.5 -> wrap up: maximin >0.5 is infeasible, deployed league agent (mean 0.59) stands.

## J.17 — User's obs idea: added a predictive-AIM LEAD feature (obs 105->107) to attack the laika wall at its root
The aim+prox shaping failed (@100k laika 0.23, killed). User asked: is the limited shell obs (MAX_SHELL_FEATURES=3) the cause?
Verified obs: 3-shell limit is real but NOT the laika bottleneck (laika doesn't spam; obs already has enemy velocity + reload).
The real gap = COMPUTING the projectile-lead (intercept) from raw pos+vel. So added a DERIVED feature instead of more shells:
2 dims = sin/cos of the angular error between heading and the INTERCEPT angle (solve the lead quadratic; direct-shot, exact at
point-blank). Fair (the script computes this internally). game_core OBS_SIZE 73->75 (+2), feature after velocity / before poison
(poison stays trailing 11). Verified: obs 107-d, finite, intercept computes. Updated env asserts 105->107 (tank_env/selfplay_env/
vec_env/train_bc/train_selfplay/train_dagger). Wrote surgery_obs105to107.py (insert 2 zero cols @94; parity 100% -> league agent
-> v15_league_107.zip). Recorded 107-d laika demos (165 wins) so the CE-anchor teaches USING the lead feature. Training the 107-d
lead-feature laika-specialist (self-play vs laika + proximity 0.015/range170 + aim 0.004 + CE-anchor, gate 30/60ep). Decisive: large-sample
laika >0.5? If YES -> the obs was the wall, build router -> deploy 107-d. If NO -> revert obs to 105, the league agent (mean 0.59) stands.

## J.18 — Lead feature FAILED honest eval; reverted to clean 105-d; the laika wall is exhaustively confirmed. WRAP-UP.
The 107-d lead-feature laika-specialist looked promising on noisy gates (@50k cheap 0.57 / confirm 0.42) but the HONEST
120-ep eval of the saved _best = laika 0.32 (latest 0.28) -- within noise of the 0.28 baseline. The lead feature did NOT
break the wall. WHY: laika's evasion is REACTIVE (it dodges AFTER you commit a shot), so handing the agent the intercept
angle doesn't help (target moves after you fire; maze blocks direct shots). The script only beats laika via point-blank
PRESSURE (close enough it can't dodge in time), which the neural agent can't execute consistently.
=> EXHAUSTIVE: laika ~0.28-0.32 across RL / clone / proximity-reward / aim-reward / lead-feature. maximin >0.5 INFEASIBLE.
REVERTED cleanly: game_core OBS_SIZE 75->73, removed the lead-feature block; env asserts 107->105 (6 files); cache survival29.
Verified: obs=105 all modes, episodes resolve, smoke_core/smoke_moba1v1duel/verify_rules GREEN, browser play (v15clone 105-d) 0 errors.
Kept as reusable artifacts: surgery_obs105to107.py, proximityBonus reward (dormant, off by default). FINAL DELIVERABLE: v15_league_agent
(mean 0.59 reduced pool) deployed + playable + watchable. Per "如果不行就收尾": wrapped up. The laika wall is a real property of reactive evasion under random spawn.

## J.19 — Tank Academy workflow -> distilled to a cheap go/no-go (user chose it). RECONCILED the laika number first.
6-agent design workflow produced TANK_ACADEMY.md (8-stage start-state curriculum grounded in MPE/MADDPG, GRF Academy, Janosov
predator-prey, SciRep wall-pinning, Hide-and-Seek) + a BRUTAL adversarial critique. Critique verdict: DON'T build the 8-stage academy
(over-built; A1/A2 collapse; randomTurret is open-arena-only; the spawn-seed split is weak — spawns are i.i.d.+fully-observed so the gap
measures sampling noise; the single-laika+proximity config the academy converges to was ALREADY run = capped 0.30; the "longer ttk not
more wins" fingerprint = a fire-timing CONVERSION wall not reachability). RECONCILED CRITIQUE-0: v15_league_agent vs laika 200ep =
seed-base 300k 0.15 / 500k 0.23 / 700k 0.25 / 900k 0.42 / 1300k 0.35 -> avg 0.28, HUGE seed-variance (0.15-0.42). The "0.64" = the OLD
selfplay_v1 (101-d, fixed-spawn regime), NOT comparable. So the ~0.28 wall is real; always use >=200ep here.
Built the genuinely-NEW lever (the one untried thing): ESCAPE-ANGLE cornering reward (REWARD_DEFAULTS.cornerCoef/cornerRange + denseReward:
fraction of the enemy's 16 escape rays BLOCKED by walls/the agent; Janosov formation-score / SciRep 'Intercept = block the front'). Verified
fires + v1 byte-identical off. GO/NO-GO A/B from v15_league_agent vs laika ONLY: Arm A = cornering(0.015)+proximity, Arm B = proximity-only
(control = known 0.30). Decision rule: Arm A >0.40 (200ep) AND > Arm B -> cornering is the unlock, continue to full build (anneal + intercept-WITH-cornering);
else -> fire-timing wall confirmed, stop/pivot to composition. Honest odds ~25-30%.

## J.20 — GO/NO-GO = NO-GO. Cornering reward failed (eroded). The laika wall is DEFINITIVE across 6 approaches. FINAL.
200ep eval: Arm A (cornering+proximity) laika 0.26 (per-base 0.40/0.23/0.23/0.25/0.20) -- BELOW Arm B (proximity-only control) 0.31
and below baseline 0.28. The @50k cheap-0.40 "peak" had confirm 0.28 = seed-luck; PPO then ERODED the policy with the cornering reward.
Decision rule (Arm A >0.40 AND >Arm B) -> NO-GO. The predator-prey literature's cornering mechanism (Janosov/SciRep), the single most
principled untried lever, does NOT transfer. DEFINITIVE: laika ~0.26-0.32 across RL self-play / DAgger clone / proximity reward / aim
reward / lead-intercept OBS feature / escape-angle cornering reward. The binding constraint is FIRE-TIMING CONVERSION (the critique's
prediction): the proximity/cornering specialists have 2-3x longer ttk (they stay ON laika) but can't land the shot in the 1-2 frames
before a reactive dodge -- a closed-loop control-latency ceiling vs a survival-rewarded shooter-evader, unfixable by reward/obs.
The intercept-WITH-cornering combo (the critique's only remaining idea) is dead too: BOTH components independently failed/eroded.
Composition/router can't help (no laika-specialist >0.40 exists to route to). WRAP. cornerCoef reward left dormant (v1 byte-identical off);
TANK_ACADEMY.md + critique + surgery tools kept as artifacts. FINAL DELIVERABLE unchanged: v15_league_agent (mean 0.59 reduced pool), playable + watchable.
