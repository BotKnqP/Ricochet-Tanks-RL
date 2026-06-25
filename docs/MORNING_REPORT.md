# Morning Report — Ricochet Tanks RL (survival_v2)

**Run:** 8-hour autonomous campaign, overnight 2026-06-20 → 2026-06-21
**Ruleset:** survival_v2 (HP ×2, ~1.5× slower regen, randomized `half_random` spawn)
**Bottom line:** The night's deliverable is a robust agent that beats every opponent under
survival_v2. The surprise is that it is **not a neural network** — it is the pure-aggression
script (`laika-aggressive`). All numbers below are from **independent fresh-seed verification**
runs (seeds 500000, 650000 — disjoint from the script defaults 300000/900000), not from
in-training/optimistic evals.

---

## 1. Headline result

Under survival_v2, the **multi-task conflict dissolves**: a single purely-aggressive behaviour
beats *every* opponent we have, including the held-out styles it was never tuned against.

| Agent | OVERALL maximin | OVERALL mean | Verdict |
|---|---|---|---|
| **Pure-aggression SCRIPT** (`laika-aggressive`) | **0.50** | **0.73** | beats all 11 opponents (deployed champion) |
| Best NEURAL single policy (`s1b_best` generalist) | 0.19 | 0.39 | collapsing opponent; well below the script |

The pure-aggression script is the night's champion and what we deployed. The best neural
policy is **~2.6× weaker on maximin** (0.19 vs 0.50) and has a genuinely collapsing opponent
(charger at 0.19).

Two caveats are baked into the headline and detailed in §5:
- The script's maximin lands **exactly** on 0.50 (it wins `laika` precisely 16/32 — a tie, not a
  majority). The `>= 0.5` bar is met but the margin is zero.
- "Beats everything" is true in the **loose** sense (above the 0.4 floor everywhere, above 50%
  almost everywhere). It is **not** a blowout: `charger` and `spammer` are near coin-flips (~0.5).

---

## 2. The conceptual discovery — why v2 dissolves the conflict

For the entire prior campaign the blocker was a **multi-task conflict**: no single identity-blind
policy could beat *both* the evasive `laika` *and* the reckless chargers at once. The per-opponent
expert labels were contradictory (vs `laika`: "charge + fire"; vs reckless: "evade + 1-shot"), so
every single policy seesawed and the maximin sat at ~0.12. Routing between two specialists lifted
this to maximin 0.35 (a real result earlier in the night).

Then the key realization: **the conflict was a survival_v1 (3-HP) artifact.** At 3 HP, a reckless
charger dies to a precise evasive 1-shot, so the counter to reckless is *evasion* — which directly
opposes the aggressive pursuit needed against `laika`.

Under **survival_v2 the physics changes the optimal strategy**:
- **HP ×2** turns every fight into a slugfest rather than a 西部对决 instakill. You can no longer
  one-shot a charger, so dodging buys nothing.
- The **structural player-0 (blue) advantage** (the learner always spawns/acts as blue) means that
  in a straight slug-trade, blue out-trades red.
- Therefore the reckless-counter under v2 is the **aggressive mirror**, not evasion. And aggressive
  pursuit is *also* what beats `laika`. **The two halves of the old conflict collapse into one
  behaviour: pure aggression.** One script satisfies both, so there is nothing left to route.

This is why pure aggression is a near-universal expert under v2 and why the router (which was the
correct v1/early-v2 answer) became unnecessary.

---

## 3. Comparison table (same harness: `eval_v2_agent.js`, survival_v2, half_random)

All rows are on the identical 101-dim identity-blind observation and the same eval path. Neural
rows are the prior-stage `cmp_*.txt` runs (12ep × 2 seeds, seeds 300000/900000). The script and
the s1b_best headline rows are the **fresh-seed verification** runs (16ep × 2 seeds, seeds
500000/650000 — genuinely held-out seeds).

| Stage / agent | Type | OVERALL maximin | OVERALL mean | HELD-OUT maximin | HELD-OUT mean |
|---|---|---|---|---|---|
| `selfplay_v1` (R2 self-play) | neural | 0.00 | 0.25 | 0.08 | 0.20 |
| `oldspec` (v1 aggro specialist) | neural | 0.00 | 0.26 | 0.04 | 0.28 |
| `s1b_best` (v2 self-play generalist) | neural | **0.19** | 0.39 | 0.22 | 0.34 |
| `v3_best` (old-rules champion) | neural | 0.13 | 0.37 | 0.29 | 0.33 |
| Opponent **ROUTER** (script+gen) | hybrid | ~0.35 | ~0.57 | — | — |
| **Pure-aggression SCRIPT** | script | **0.50** | **0.73** | **0.56** | 0.74 |

Notes:
- The `s1b_best` maximin shows as 0.04 in the older 12ep `cmp_s1bbest.txt` run and 0.19 in the
  fresh 16ep×2-seed verification — both confirm it is far under the script; the floor opponent is
  the reckless `charger`/`laika-aggressive` cluster either way.
- The router number (~0.35) is from an earlier-in-the-night eval (20ep × 2 seeds, partial suite),
  reported here for trajectory; it was **not** re-verified on the full 11-opponent fresh-seed suite
  this workflow, so treat it as indicative, not audited.
- All four neural `cmp_*` rows and both fresh-seed champion rows show **trunc = 0.00 on every
  opponent** — no episodes hit the 3000-step cap, so none of these numbers are inflated by
  truncation/draw bookkeeping.

---

## 4. What was deployed to the browser

The browser watch (`game_render.js`, cache-busted to `?v=survival22`) now serves the verified
champion as the default for survival_v2:

- **Default blue model for `?ruleset=survival_v2` watch = `aggro` = the pure-aggression script**
  (`inputForBlue()` returns the `"aggressive"` script when `liveModel==="aggro"`;
  `app.liveModel` defaults to `"aggro"` under survival_v2).
- Blue dropdown order: **★ pure-aggression (v2 champ)** → router (script+gen) → v2 neural
  generalist → v3 champion (old rules) → v3 max-generalist → original agent (A/B) → laika-agg
  specialist. So a human can A/B the champion against the neural agents and the router live.
- Opponent dropdown exposes the train-weird and held-out (★) styles for held-out watching.
- The router (`routerAction()`, classify-then-lock on enemy approach-speed @35 steps) remains
  available as a selectable blue model for comparison.

Watch the champion crack a reckless opponent:
`...&watch=runs/auto_live&ruleset=survival_v2&red=charger` (try `red=laika` for the evasive case).

User models (`bc_dagger_moba_canon_v2`, `historical_*`) were **never overwritten**; all campaign
outputs live under `models/auto/`.

---

## 5. Honest limitations & caveats

1. **The maximin sits exactly on the 0.50 knife-edge.** In the verification run the OVERALL maximin
   is `0.500000` exactly, set by `laika` winning precisely 16/32. A one-episode swing on a different
   seed would drop it below 0.5. The `>= 0.5` claim holds, but the margin is **zero** and it is
   fragile to seed choice. "Beats `laika`" is literally a 50% tie, not a strict majority.

2. **"Beats everything" is loose, not dominant.** The genuinely hardest matchup is **`charger`**
   (head-pressure rusher), which scored 0.56 / 0.53 / 0.41 / 0.51 across four independent fresh-seed
   runs — averaging **~0.50**, a near coin-flip. A 120-episode run dipped it to **0.41**; a 200-episode
   follow-up pulled it back to 0.51, so it never *sustained* below the 0.4 floor, but it is close.
   `spammer` (0.53–0.63) is the next-hardest. So pure aggression "wins" against reckless styles only
   in the sense of staying ~at-or-above 50%, not by any margin.

3. **The neural ceiling is the observation, not the strategy.** Three separate DAgger attempts to
   imitate `laika-aggressive` into a neural net failed (warm-start + stale v1 demos regressed
   .42→.36; fresh BC under-fired at ~2.6% fire rate; warm-start + fresh demos + frame-weighting
   diverged, loss .3→2.0). Root cause: the 101-dim obs has enemy **positions but no enemy velocity**,
   so a neural policy cannot reproduce the script's precise pursuit of an *evasive* target (neural
   `laika` ~0.20–0.25 vs script 0.50). The script wins because it reads full game state. A best-effort
   gentle DAgger (`v2_aggro_dagger.zip`) was left running but is **not** the deliverable.

4. **Verification scope.** The deployed code path was confirmed (`?v=survival22`, default `aggro`,
   `inputForBlue` returns the script), but the live rAF watch only animates when the preview tab is
   foreground (`document.hidden` pauses it — environment artifact, not a bug). Headless == browser
   parity was confirmed earlier (blue=aggressive vs stationary 6/6 in-browser matches headless).

5. **Unverified-this-workflow numbers.** The router ~0.35 maximin is carried from an earlier eval,
   not re-audited on the full fresh-seed suite. The held-out generalization of the script is
   genuinely strong (held-out maximin 0.56), but note this is still only the 4–5 held-out families
   we have built, not arbitrary unseen behaviour.

6. **No truncation concerns.** Across every verification and `cmp_*` run, `trunc = 0.00` for all
   opponents — all episodes resolved cleanly by death. There is no draw/timeout inflation anywhere
   in these numbers (the `--max-steps 1800` fix from Stage 1b is holding).

---

## 6. Recommended next steps

1. **Harden the knife-edge.** Re-run the champion at higher N (e.g. 100ep × 3–4 seed bases) to get a
   robust point estimate of the `laika` and `charger` matchups. If maximin proves to sit reliably at
   ~0.50, state it as "≈0.5 (coin-flip on the two hardest)" rather than "≥0.5", which over-claims.

2. **Attack the two soft spots directly** (`charger`, `spammer`). These are the only near-coin-flips.
   A small reckless-focused tweak to the aggression script (or a thin script-level counter for
   head-pressure rushers) could lift the floor above 0.55 without touching anything else.

3. **Add enemy velocity to the observation** and only then revisit a neural agent. The documented
   "needs motion obs" wall (task #33) is the real cap. With velocity in the 101→~107-dim obs, a
   neural DAgger of the script should finally be able to track evasive targets, potentially matching
   or exceeding the script and enabling RL fine-tuning beyond it.

4. **Keep the script as the shipping default** until a neural agent provably beats it on the same
   fresh-seed harness (maximin and per-opponent). The script is the honest champion under v2 today.

5. **Broaden the held-out test set.** Current held-out families are hand-built; the strongest claim
   ("near-universal expert") would be more convincing against a larger, more adversarial randomized
   ecology.

---

*All evaluation numbers in this report come from `train/eval_v2_agent.js` on survival_v2 /
half_random with fresh held-out seeds (500000, 650000) unless a row is explicitly labelled as an
older `cmp_*` or earlier-night eval. maximin = min win rate across opponents; mean = average win
rate. Identity-blindness of the 101-dim obs was independently audited and confirmed.*
