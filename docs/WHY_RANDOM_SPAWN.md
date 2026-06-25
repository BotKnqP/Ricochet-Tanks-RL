# WHY can't the neural agent beat the script under RANDOM spawn? — causal diagnosis

Question being answered (human's exact words): *"can't beat the script under random spawn — is it
(a) no clonable mother, (b) state/action/reward problem, or (c) algorithm limitation?"*

Scope/honesty note: the human has made **2 over-confident wrong predictions in a row** (J.6 "v3 adapts
well" → wrong; J.7 "continual-train recovers it" → wrong). This report VERIFIES every load-bearing number
against code (`file:line`) and live re-evals run THIS session, and flags residual uncertainty explicitly.
It launches NO training (analysis-only mandate).

---

## 1. CORRECTED FRAMING — what actually happened (the fixed-spawn "win" was an ARTIFACT)

The comparison the human is reasoning from is **distorted on BOTH sides by the fixed opening.** The clean
fact is the opposite of the intuitive story.

**1a. "Random spawn removes the teacher" is backwards — random spawn CREATES the teacher.**
Live re-eval this session (`node train/eval_v2_agent.js --script laika-aggressive`, laika family,
12ep × 2 seed-bases 300000/900000):

| regime | laika | easy | stationary | **laika-aggressive** | overall maximin |
|---|---|---|---|---|---|
| **v1 FIXED spawn** (`--ruleset survival_v1 --spawn fixed`) | 0.92 | 0.42 | 0.96 | **0.00** (ttk −1.00) | **0.00** |
| **v1.5 RANDOM spawn** (`--ruleset survival_v1 --spawn half_random`) | 0.63 | 0.88 | 0.96 | **0.67** | **0.50** |

The script's laika-aggressive cell is **0.00 under fixed spawn — zero wins, every seed** (ttk −1.00 is the
"no win recorded" sentinel; `eval_v2_agent.js:37` only sums ttk on a win). That 0.00 is a **determinism
artifact, not weakness:** `game_core.js:2230-2236` places both tanks at a **mirror-symmetric** fixed start
(blue x=0.12·W, red x=0.88·W, facing each other), and the scripted bots break the blue/red tie with a
`bot.id`-phased term — `Math.sin(state.elapsed·k + bot.id·φ)` at `game_core.js:1544, 1601, 1730` and the
`me.id*78.233` fire/strafe RNG at `game_core.js:1900-1901`. So in the aggressive-vs-aggressive mirror blue
(id 0) loses **deterministically every seed**. Random spawn destroys that symmetry → the same matchup
becomes a real ~0.67 coin-flip → the script's laika-family maximin jumps **0.00 → 0.50.** (My 0.50/0.67
matches KEY-FACT-2's 0.56 within seed noise; the prompt's 0.56 came from a wider seed set.)

**1b. The neural "win" under fixed spawn was real but spawn-OVERFIT and tiny.**
v3_best (the fixed-spawn-trained net) on the laika family (`runs/auto_campaign/v3_v1fixed.txt` /
`v3_v15.txt`, corroborating J.6 lines 660-662):

| regime | laika | laika-aggressive | maximin |
|---|---|---|---|
| v1 FIXED (its training regime) | 0.72 | 0.38 | **0.38** |
| v1.5 RANDOM | 0.16 | 0.16 | **0.16** |

So under fixed spawn the net (0.38) "beats" the crippled-0.00 script in the laika-aggressive cell — but
**both numbers are fixed-opening artifacts.** Move to random spawn and the net **collapses 0.38 → 0.16**
(laika 0.72 → 0.16), worst exactly in the evasive-laika matchup "where opening positioning is decisive"
(J.6:664-667). Mechanistically a fixed always-identical start lets a net fit **shortcuts on the constant
opening** (open-loop early-trajectory memorization); random spawn removes those shortcuts.

**Honest verdict on the premise:** v3_best did NOT robustly beat the script. It won ONE degenerate
fixed-opening mirror cell while the script was crippled by the symmetry artifact, and its own edge is
opening-overfit. Under random spawn (the regime the human is asking about) the best *single* neural policy
reaches **laika-aggressive 0.04 / overall maximin 0.04** (`cmp_s1bbest.txt`, live-confirmed) versus the
**script's ~0.50** — a ~10× gap. THAT gap is the real phenomenon to explain.

---

## 2. RANKED VERDICT on (a) / (b) / (c)

### PRIMARY CAUSE → (c), and specifically its *commitment / multi-task-conflict* limb.
The dominant blocker is that a **single identity-blind feed-forward PPO policy cannot COMMIT** to the one
fixed behaviour (pure aggression) that the script uses to win. By reacting to the obs it over-adapts
per-opponent and **seesaws.** Verified evidence:
- J.4 (JOURNAL:622-637): with velocity obs + a 3×-stronger BC anchor (full-suite mean 0.35), 3 self-play
  gates @60k/120k/180k all **SEESAWED**, mean flat ~0.19, never promoted; each gate a *different* opponent
  is weak (laika 0.00→0.22, laika-aggressive 0.28→0.11, ace 0.44→0.22, charger 0.22→0.11). "The policy
  keeps trading the evasive half for the aggressive half."
- The seesaw FLOOR is real and saved: best single v2 policy = laika-aggressive **0.04**, overall maximin
  **0.04** (`cmp_s1bbest.txt`, this session).
- It is provably a CONFLICT, not a precision wall: the laika-aggressive **specialist** hits **0.88 robust**
  (0.85/0.93/0.85) while the combined net gets 0.04 (R3, JOURNAL:148-152) → the matchup is fully winnable
  in isolation; and an inference-time **router** that composes two specialists **triples maximin 0.12 → 0.35**
  (I.2, JOURNAL:454-465). So it is a **SOFT** limit of ONE net, dissolved by composition — not a hard
  architectural impossibility.

### CONTRIBUTING → (c)'s other two limbs + (b)'s reward limb.
- **(c-2) PPO erodes warm-starts** (tuning pathology, real): continual-train of v3 under v1.5 ERODED it
  **0.29 → 0.21** (min 0.15 → 0.06), declining not recovering (J.7:671-683; model
  `models/auto/v3_v15_adapt_latest.zip`). Also tasks #31/#32/#37.
- **(c-3) opening over-fit under domain randomization** (sample/coverage, real): v3 0.38 → 0.16 (J.6). This
  is the standard "DR turns one memorizable MDP into a distribution of MDPs → higher return/gradient
  variance → slower, more conservative convergence." Generalization costs optimization the closed-loop
  script never pays.
- **(b-reward) the reward is too sparse for the random-spawn OPENING** (real but SECONDARY, fixable):
  there is **zero opening/positioning shaping anywhere** — `DUEL_REWARD` has approachCoef = aimBonus =
  backwardPenalty = **0.0** (`train_moba1v1duel.py:32-37`); `REWARD_DEFAULTS` keeps all shaping 0.0
  (`game_core.js:466-472`). The only opening signal is the terminal ±3.0 win/loss credited back through
  gamma 0.995 (`train_moba1v1duel.py:61`) over hundreds of steps. Under fixed spawn the opening is constant
  so even sparse terminal credit memorizes one good opening (v3 did); under random spawn the good opening
  is **spawn-conditional and varies every episode**, so the lone late reward is hard to attribute to the
  early positioning action → a genuine credit-assignment gap. It is SECONDARY because a denser opening
  reward would NOT fix the commitment/seesaw, which dominates.

### REJECTED.
- **(a) "no clonable mother under random spawn."** FALSE on its own premise AND mislabels the mechanism.
  A random-spawn-robust mother **demonstrably exists** — the laika-aggressive script under v1.5 scores
  laika-family maximin **0.50** (live, §1a). The causal story is **inverted**: random spawn doesn't destroy
  the mother, it creates one by breaking the degenerate fixed mirror (§1a). And (a) conflates two
  mechanisms: imitation (BC/DAgger) **caps AT the mother** — `train_dagger.py:112` is pure cross-entropy on
  expert labels with no reward term, labeling every visited state with the expert's action
  (`train_dagger.py:119-141`); empirically the R3 specialist **matched** the ~0.84 teacher (0.88), never
  exceeded it. RL needs no mother but must EXCEED the teacher via reward — and the reason it fails is the
  commitment/conflict equilibrium, NOT a missing teacher (J.4 seesawed even WITH a 3× anchor).
- **(b-state) obs insufficient.** REJECTED. The first 2 obs dims are **absolute board position**
  (`game_core.js:1972-1974`, me.x/worldW, me.y/worldH) → a random spawn is **fully observed**, not a hidden
  ego shift; plus enemy dx/dy/bearing/heading, 32 wall rays, shells, powerup, safe-rect, and 4-d ego
  velocity (`game_core.js:1986-2042`). Per-step decision is K=1 single-frame **~98% learnable**
  (`LEAGUE_STRATEGY.md:316`). OBS_SIZE = 73 + 32 rays = **105** (`game_core.js:79-84`).
- **(b-action) Discrete(18) inexpressive.** REJECTED. `ACTION_TABLE` = idle + throttle{−1,0,1} ×
  turn{−1,0,1} × fire{F,T} (`game_core.js:134-146`). The winning **script lives entirely inside this action
  space** and wins random-spawn ricochet (maximin 0.50, §1a). A fixed function of the public state suffices;
  neither obs nor action is the blocker.

---

## 3. THE DEEPEST SINGLE ROOT CAUSE (one sentence)

**A reactive single feed-forward identity-blind PPO policy cannot replicate what makes the script win — an
unwavering COMMITMENT to one fixed behaviour — because, by reacting to the obs, it over-adapts per opponent
and seesaws; and random spawn makes this strictly harder by additionally destroying the memorizable fixed
opening, converting one MDP into a distribution of start geometries that the sparse, opening-blind terminal
reward cannot teach in one PPO pass.**

(I.e. it is dominantly an *algorithm/commitment* limit (c), aggravated by a *spawn-generalization/sparse-
reward* limit (b-reward); it is NOT (a) and NOT a state/action expressiveness limit.)

---

## 4. THE SINGLE MOST DECISIVE EXPERIMENT (isolates commitment vs spawn/reward)

The two surviving causes — **(c) commitment/conflict** vs **(b) spawn-generalization/reward** — are
confounded in every run so far because every random-spawn run also trains against MULTIPLE opponents (so a
seesaw could be conflict OR spawn-overfit). **Collapse the multi-task axis to isolate the spawn/reward
axis.**

**Run (NOT launched here — analysis-only):** train a **SINGLE-opponent** agent vs **laika ONLY**, **from
scratch under `--spawn half_random`** (survival_v1), no other opponents in the pool. Concretely:
`train/train_moba1v1duel.py --opponent laika --spawn-mode half_random --ruleset survival_v1
--total-timesteps ~3M` (or the equivalent single-opponent DAgger/self-play config), then
`node train/eval_v2_agent.js --policy <out>.json --script "" --ruleset survival_v1 --spawn half_random`
restricted to laika, vs the **script's laika cell under the same regime (~0.63, §1a)** as the bar to beat.

**Decision rule (one variable = multi-task conflict; spawn/reward held fixed at "present"):**
- **If the single-opponent agent BEATS the script vs laika under random spawn (win ≳ 0.63, robust across
  ≥3 seeds):** spawn + reward are **NOT** the blocker (a single net CAN solve one matchup under random
  spawn with the existing obs/action/reward) → the wall is the **MULTI-TASK COMMITMENT CONFLICT (c)**. The
  fix is composition (router/league/conditioning), not richer obs or denser reward.
- **If it STILL can't (win materially < the script's ~0.63, or it collapses like v3's laika 0.16):** then
  even with the conflict removed a single net fails under random spawn → the **spawn-generalization /
  opening credit-assignment gap (b-reward + c-3)** is a real independent blocker → add spawn-conditioned
  opening shaping (e.g. approachCoef > 0 or a first-strike-positioning bonus) and/or more half_random
  samples before any composition work.

This cleanly separates the two: it holds spawn-mode = half_random and the reward fixed, and toggles ONLY
the number of opponents (conflict on/off).

**Uncertainty flag (the human's last 2 predictions were wrong — so will I hedge):** my *prior* leans
toward the first branch (the R3 specialist hit 0.88 vs laika-aggressive, and laika in isolation is "fully
winnable"), which would say **conflict, not spawn**. BUT that 0.88 specialist was trained under **fixed**
spawn — there is **no existing data point of a single-opponent agent trained under random spawn from
scratch**, so this is genuinely the un-run experiment and the outcome is NOT a foregone conclusion. J.6's
v3-collapse (0.72→0.16 vs laika under random spawn) is weak counter-evidence that spawn alone can hurt the
laika matchup. Do not pre-judge; run it.

---

## 5. WHAT EACH ANSWER IMPLIES FOR THE FIX

- **If (a) were the cause (it is NOT):** you'd need to *find/record a robust teacher*. Already done — the
  laika-aggressive script IS the random-spawn-robust mother (maximin 0.50). But note: cloning it via
  BC/DAgger can only **match ~0.50, never beat it** (imitation caps at the teacher, `train_dagger.py:112`).
  So even under (a)'s framing the deliverable is "distill the script," which is exactly the cheap fallback.
- **If (b) is the cause (CONTRIBUTING, secondary):** add the **currently-absent opening shaping** — set
  `approachCoef > 0` for a dense early-closing signal, or a spawn-conditioned first-strike-positioning
  bonus, so the opening action gets credit without waiting for the terminal ±3.0. Low-cost, isolated to
  `DUEL_REWARD` / `REWARD_DEFAULTS`; would help spawn-generalization but will NOT by itself fix the seesaw.
- **If (c) is the cause (PRIMARY):** stop trying to make ONE net do everything. Use **composition** —
  the inference-time **router** already triples maximin 0.12→0.35 (I.2); push it with a random-spawn-trained
  laika/movers specialist for the evasive route, or add **recurrence / opponent-modelling** so a single net
  can *recognize and commit*, or a **fire-weighted DAgger that LOCKS pure aggression** (J.5 decision, caps
  at ~script level but cheap). For the c-2 erosion: gentler lr / stronger anchor / promotion gate. For c-3
  opening over-fit: train **under half_random from scratch** (never transfer a fixed-spawn model — J.7
  proved transfer is a dead end). The decisive experiment in §4 tells you whether (c) alone suffices or
  whether (b)'s shaping must be added first.

---

### Provenance (every number re-verified this session unless noted)
Live re-evals: script v1-fixed (laika-aggr 0.00 / maximin 0.00) & v1.5-random (laika-aggr 0.67 / maximin
0.50), `train/eval_v2_agent.js`, seeds 300000+900000. Saved-file confirms: `cmp_s1bbest.txt` (single-net
floor 0.04), `v1check_script.txt` (script fixed 0.00), `v3_v1fixed.txt`/`v3_v15.txt` (v3 0.38→0.16). Code:
`game_core.js:79-84,134-146,466-472,1544/1601/1730,1900-1901,1972-2042,2230-2258`;
`train_dagger.py:112,119-141`; `train_moba1v1duel.py:32-37,61`. Journal: J.4/J.5/J.6/J.7, R3, I.2;
`LEAGUE_STRATEGY.md:316`.

---

## Adversarial critique

*(Appended by an analysis-only adversarial reviewer. Mandate: re-verify every load-bearing number against
code/data, look for confounds, attack the "decisive experiment," and steelman the least-favored hypothesis.
All four checks below were re-run live this session. Bottom line up front: the central diagnosis SURVIVES —
in fact the confound it worried about cuts in its favour and is even stronger than stated — but the report
**over-claims §1's framing, mislabels its own confound, and §4's "single decisive experiment" is neither
single nor minimal.** Two of the four sub-claims I was asked to test came back AGAINST the report's wording.)*

### (1) Is the ranked verdict supported, or merely asserted? — MOSTLY SUPPORTED, with one asserted leap.
Re-verified the load-bearing evidence chain and it holds at `file:line`:
- The seesaw is **real and triangulated**, not asserted: J.4 (JOURNAL:622-637, gates trade laika 0.00↔0.22 for
  laika-aggressive 0.28↔0.11) **plus** an independent run the report under-cites — Stage-3 / I.0
  (JOURNAL:414-425) where **combat reward shaping was actually added** (`closeRangeHit 0.12 + cleanTrade 0.15`)
  and the policy STILL seesawed (reckless 0.17→0.22 while laika dropped to 0.06). Two different recipes, same
  equilibrium ⇒ the (c)-commitment claim is empirically earned.
- The "fully winnable in isolation" pillar is real: R3 specialist **0.85/0.93/0.85** (JOURNAL:148-152) and the
  router **0.12→0.35** maximin (I.2, JOURNAL:454-465) both verify.
- The single-net floor is real: `cmp_s1bbest.txt` laika-aggressive **0.04**, train maximin **0.04** (re-read
  this session, exact match).
- **The asserted leap:** §2's headline mechanism — *"by reacting to the obs it over-adapts per-opponent"* — is
  a plausible STORY, not a measured one. The data prove a *multi-task conflict* (specialist wins, combined net
  doesn't); they do **not** isolate "reactivity to obs" as the mechanism. An identity-blind net could equally
  fail by **mode-averaging / gradient interference** between two reward basins — no obs-reactivity required. The
  report treats "reactive ⇒ seesaw" as established when only "multi-task ⇒ seesaw" is. Minor, but it is the one
  place the prose outruns the evidence.

### (2) The confound — does "PPO beats script under fixed" survive the determinism artifact? — IT NEVER EXISTED; AND THE REPORT MIS-SCOPES ITS OWN CONFOUND.
This is the strongest part of the report and I can make it stronger. Live re-eval this session
(`eval_v2_agent.js --script laika-aggressive --ruleset survival_v1`, 12ep×2 seeds):
- FIXED spawn: laika-aggressive **0.00**, **charger 0.00** → maximin 0.00.
- RANDOM (half_random): laika-aggressive **0.67**, charger **0.58**, laika 0.63, pro 0.75 → laika-family
  maximin **0.63**, overall (incl. charger) **0.58**.

So the premise the human is reasoning from — *"the net beat the script under fixed spawn"* — is an artifact on
**both** sides and the report is right to demolish it. BUT the report's §1a only flags the **laika-aggressive
mirror** as the determinism artifact. The data show **charger is ALSO 0.00→0.58** — a *second* fixed-spawn
artifact the report missed. That means the script's fixed-spawn maximin=0.00 is driven by **at least two**
symmetry/opening-determinism cells, not "the one mirror cell." This *strengthens* the report's thesis (fixed
spawn cripples the script even harder than claimed, so the net's "win" is even more degenerate), but it also
means §1a's clean "mirror ⇒ 0.00" story is **incomplete** — present it as "fixed spawn deterministically loses
multiple aggressive-rush cells," not just the mirror. **Verdict: the confound is correctly identified and the
"PPO beats script under fixed" premise does NOT survive — it was never a real win.** No change to the ranking.

### (3) Is §4's "single decisive experiment" actually single + decisive + minimal? — NO. IT BUNDLES ≥2 VARIABLES AND ITS DECISION RULE IS MIS-CALIBRATED.
The report claims it "toggles ONLY the number of opponents." It does not. The proposed run —
*single-opponent vs laika, **from scratch**, under half_random, ~3M steps* — silently changes **three** things
at once relative to the s1b_best baseline it is implicitly compared against: (i) opponent count 6→1, (ii)
from-scratch vs warm-started, (iii) a fresh 3M-step optimization budget. A win could then be credited to
"conflict removed" when it was actually "from-scratch under DR finally had enough budget" — the very (c-3)
spawn-generalization axis the report says it is holding fixed. Worse, the **bar to beat is mis-set**: §4 says
"beat the script's laika cell ~0.63," but the report's OWN §1a/this-session number is the script winning laika
at **0.63** — i.e. the experiment asks the net to *match a strong teacher in its best matchup*, which even a
successful single-task agent may only tie, yielding an ambiguous result.
**Sharper, genuinely minimal experiment (one variable):** take the EXISTING saved `s1bbest_policy.json` and
fine-tune it **vs laika ONLY** under the **identical** spawn/reward/budget it already trained under — i.e. hold
spawn=half_random, reward, warm-start, and a *fixed modest* step budget constant, and toggle ONLY the opponent
pool {6 → 1}. If collapsing to one opponent lets laika climb off 0.29 (its current s1bbest laika cell) toward
the specialist's ~0.85 *without* a budget/from-scratch change, the conflict is isolated as causal. Compare
against **the net's own 0.29 laika cell**, not the script's 0.63 — the question is "does removing conflict help
THIS net," not "does it out-duel the teacher." That is single-variable and decisive; §4's version is neither.

### (4) Steelman the least-favored hypothesis — could (b) reward/credit-assignment be the HIDDEN PRIMARY cause?
Best case for (b)-as-primary: (i) the reward is genuinely degenerate for the opening — `DUEL_REWARD` has
`approachCoef=aimBonus=backwardPenalty=0.0` (train_moba1v1duel.py:37) and `REWARD_DEFAULTS` is all-zero shaping
(game_core.js:471-472), so the ONLY opening signal is the terminal ±3.0 discounted through gamma 0.995 over
hundreds of steps — verified, and a textbook sparse-credit setup; (ii) under random spawn the good opening is
spawn-conditional and varies every episode, so the lone terminal reward genuinely cannot attribute credit to
the early positioning move; (iii) J.6's v3 collapse (laika 0.72→0.16 vs laika under random spawn, re-confirmed
in `v3_v15.txt`) shows spawn ALONE can wreck the laika matchup with multi-task conflict held *constant* (same
net, only spawn changed) — the single cleanest natural experiment in the whole campaign, and it points at
spawn, not conflict. If you weight (iii) heavily, (b) looks primary.
**Why the steelman ultimately FAILS as *primary* (but earns more than "secondary"):** the decisive counter is
Stage-3 (JOURNAL:414-425) — when combat shaping WAS added, the seesaw **persisted and merely relocated** its
weak cell. Reward shaping was empirically tried and did not dissolve the conflict; it cannot be the primary
blocker of a phenomenon that survives it. AND the conflict reproduces under **fixed** spawn too (the v1 regime,
script maximin 0.00, J.7:679) where the credit-assignment-under-DR problem does **not** exist — so the conflict
is spawn-independent while (b) is spawn-specific. A cause present in both regimes outranks one present in only
one. **HOWEVER**, the report's "(b) is SECONDARY because denser reward wouldn't fix the seesaw" is slightly too
dismissive: the shaping that was tried (Stage-3) was *combat/close-range* shaping, NOT the *spawn-conditioned
opening* shaping the report itself proposes in §5 — so the specific (b)-fix the report recommends is **un-tested**,
and J.6(iii) is live evidence it might independently matter for the laika matchup. Correct rank: **(b-reward) is
a genuine independent CONTRIBUTING cause for the laika-under-random-spawn cell specifically — stronger than the
report's "secondary, dominated" framing — but it is NOT primary**, because the seesaw (the thing that produces
the 0.04 floor) is conflict-driven and survives both reward shaping and the removal of random spawn.

### Net verdict on the report
Ranking **(c) primary / (b-reward) contributing / (a) and state-action rejected** is **correct and the
rejections are airtight** (obs[0:2] are absolute board position, game_core.js:1972-1974, so spawn is fully
observed; the winning script lives inside Discrete(18); imitation caps at the teacher, train_dagger.py:112 —
all re-verified). Required corrections: (1) §1a undercounts the determinism artifact (charger 0.00→0.58 is a
second one); (2) §2's "reacts to obs" mechanism is asserted beyond the evidence (the data show conflict, not
reactivity); (3) §4's experiment bundles opponent-count with from-scratch+budget and mis-sets its bar — use the
warm-start single-toggle version above; (4) promote (b-reward) from "dominated secondary" to "independent
contributor for the laika cell," since its specific proposed fix (opening shaping) was never tried and J.6 shows
spawn alone hurts laika.

*Provenance (this session): live `eval_v2_agent.js` script runs — FIXED {laika-aggr 0.00, charger 0.00,
maximin 0.00}, RANDOM {laika-aggr 0.67, charger 0.58, laika 0.63, pro 0.75, maximin 0.63/0.58}. Re-read:
`cmp_s1bbest.txt` (0.04 floor), `v1check_script.txt`, `v3_v1fixed.txt`/`v3_v15.txt`. Code re-verified:
game_core.js:1972-1974 (abs pos obs), 1544/1601/1730 + 1900-1901 (id-phased RNG), 2230-2236 (mirror spawn),
466-472 (zero shaping); train_moba1v1duel.py:32-38 (zero opening reward); train_dagger.py:108-141 (pure CE).
Journal re-read: J.4 (622-637), J.6/J.7 (655-685), R3 (148-152), I.2 (454-465), **I.0/Stage-3 (405-434) — the
combat-shaping-still-seesawed run the original report under-cited.**)*
