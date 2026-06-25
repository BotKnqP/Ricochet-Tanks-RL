# How the fixed-spawn champion (v3champ) was trained, step by step

> [中文版](TRAINING_HISTORY.md)

> A lot of reward-curve data and several historical agent versions were accidentally deleted late in the
> project, so the record here is partial. Rigorous head-to-head scoring is in [`LADDER.md`](LADDER.md).

> Model file: `models/auto/league_robust_v3_best_105.zip` (browser name **v3champ / fixed-spawn champion**)
> obs 105-d · action 18 discrete · ruleset survival_v1 · fixed spawn (blue-left / red-right, symmetric)
> This documents every step from zero to champion: what changed, how much it helped, and the reward used.

## Final result (120 ep/opponent × 4 seed-bases)

| # | opponent | win rate | fire % | hits/ep | time-to-kill | >0.5? |
|---|---|---|---|---|---|---|
| 1 | laika (evasive) | **0.775** | 6.4% | 2.52 | 10.4s | ✅ |
| 2 | easy_laika | **0.775** | 3.0% | 1.93 | 28.9s | ✅ |
| 3 | stationary | **0.983** | 4.8% | 2.73 | 22.1s | ✅ |
| 4 | laika-aggressive-pro (reckless) | **0.458** | 7.3% | 1.51 | 8.9s | coin-flip |

**Mean 0.748 · worst 0.458 · 3/4 cleared >0.5.** The three evasive laika are solidly beaten; the reckless
pro is a coin-flip (it is a separate "reckless head-on" wall, not the evasion wall).

---

## Training pipeline at a glance (each step's change + gain)

| # | stage | core algorithm | key change | quantified gain |
|---|---|---|---|---|
| 0 | environment + scripted opponents | — | moba1v1duel scenario, 5 scripted bots, poison/regen/shield | provides learnable experts & opponents |
| 1 | **behavior cloning (BC)** | supervised (fire-weighted cross-entropy) | win-only demos from scripted experts; **shell_decay physics parity** | **laika 14% → 53%** (the biggest single fix) |
| 2 | **all-scripts DAgger** | DAgger + maximin gate | script-vs-script matrix → expert map (each opponent gets a ≥0.95 blue expert) | maximin **→ 0.20** (4/5 scripts cracked) |
| 3 | **self-play** | anchored PPO + frozen self in the pool | obs is symmetric → blue policy plays red directly | **laika 0.30 → 0.64** (first method to lift laika) |
| 4 | **domain-randomization league v2** | anchored PPO + multi-criteria gate | paramBot ecology (9 presets + per-episode random params); 426k steps | held-out **0.284 → 0.414**; maximin 0.02 → 0.12; pro 0.38 → 0.53 |
| 5 | **domain-randomization league v3** | + anchor-to-policy | expanded randomized family + seed avalanche-mix; 600k steps | held-out **0.414 → 0.526**; maximin 0.07 → **0.20** (all 6 criteria pass = champion) |
| 6 | **observation surgery** | zero-pad transfer | obs 101 → 105 (add 4 velocity dims); 100% parity | deployed `_best_105`, fixed-spawn laika ~0.77 |

Each stage expanded below.

---

### Stage 0 — environment & scripted opponents
Built the engine `game_core.js` and the `1v1duel` gun-duel scenario (closing poison ring, breathing regen,
5-shot 2× weapon, −30% turn), plus 5 scripted opponents (weak → strong):
- **stationary** (sitting target) · **easy_laika** (slow dodger) · **laika** (defensive evasive sniper) ·
  **laika-aggressive** (reckless) · **laika-aggressive-pro** (disciplined aggressor).

No training here — just preparing the "experts" and the "opponents".

### Stage 1 — behavior cloning (BC): laika 14% → 53%
The scripted expert (laika-aggressive) plays blue and we record **winning trajectories** (obs→action), then
clone with **fire-weighted cross-entropy** (fire actions weighted ×4–16, so it doesn't learn to wander without
shooting).

> **The biggest breakthrough here was not the network — it was physics:** early on the training bridge's
> `shell_decay` (shells weaken with distance) disagreed with the browser, so the same clone scored only **14%**
> vs laika under real physics. Setting `shell_decay=True` as the single standard took **the same clone from
> 14% → 53%** — the project's first and highest-yield fix, confirming "make the simulation consistent before
> blaming the algorithm".

### Stage 2 — all-scripts DAgger: cracks 4/5 scripts, hits the multi-task wall
BC is open-loop: as soon as it drifts off the expert's trajectory it falls apart (covariate shift). **DAgger**
lets the student roll out and the expert label the states it actually reaches, iterating.

- Ran a **script-vs-script matrix** and found a **structural player-0 (blue) advantage**: every opponent has a
  blue expert that beats it ≥0.95 (stationary←aggro, easy←pro, laika←aggro, aggro←pro, pro←aggro). From this
  we built an **expert map**, labeling each opponent with its counter.
- A **maximin gate** (promote a checkpoint only if the *worst* opponent also improves) selected
  `bc_dagger_allscripts_v3`.
- **Result: maximin 0.20** — laika/easy/stationary/pro all decent, but **laika-aggressive stuck at 0.04**.

> **Key diagnosis:** a dedicated **specialist** vs laika-aggressive reaches **0.88** (but forgets everything
> else = catastrophic forgetting). So 0.04 is not "unbeatable" — it is a **multi-task conflict in one
> identity-blind policy**: vs laika you must snipe steadily, vs aggro you must dodge and let it self-destruct;
> two opposite policies in one network see-saw.

### Stage 3 — self-play: laika 0.30 → 0.64
The scenario is left-right symmetric, so a **blue-trained policy can play the opponent directly from the red
view (obs1)**. Put a **frozen older self** in the opponent pool and run anchored PPO with the maximin gate.

- DAgger's labels are *contradictory* (different opponents demand opposite actions); RL's **win/loss reward is
  unambiguous**, so RL learns a **conditional policy** DAgger cannot.
- **laika 0.30 → 0.64**, robust across seeds. This is the first method that genuinely lifts laika — earlier
  pure PPO only eroded the warm-start, and pure DAgger plateaued.

### Stage 4 — domain-randomization league v2: generalization 0.284 → 0.414
To avoid overfitting the 5 fixed scripts, introduced the **paramBot ecology**: 9 archetype presets
(rusher/kiter/sniper/charger/precision/spammer/counter/baiter/turtle) + **per-episode random parameters**
(domain randomization). The opponent pool is randomized-dominant.

- The gate became **multi-criteria** (not just win rate): ① held-out opponent generalization, ② self-hit rate,
  ③ vulnerability to enemy power-ups, plus a maximin floor and "fixed scripts don't regress".
- 426k steps: **held-out generalization 0.284 → 0.414**; maximin 0.02 → 0.12 (charger/spammer cracked);
  pro 0.38 → 0.53; laika held at 0.90.

### Stage 5 — domain-randomization league v3: the champion (maximin 0.07 → 0.20)
Three changes on top of v2:
1. **anchor-to-policy** — the CE anchor no longer pulls toward the narrow laika demos but toward "the
   warm-started start policy's own argmax", resisting drift without dragging the policy back to old habits.
2. **expanded randomized family** — added spammer/baiter/wall-sniper archetypes + **seed avalanche-mix**
   (decorrelated, so different envs don't draw the same random params).
3. fixed the allocator (guarantee ≥1 env per listed opponent; no silently zeroing small-weight opponents).

- 600k steps: **held-out 0.414 → 0.526**; maximin 0.07 → **0.20**; `v3_best` improved on **all 6 criteria**.
- This is **`league_robust_v3_best`** — the champion.

### Stage 6 — observation surgery: 101 → 105
Late in the project we added **4 velocity dims** (better predictive aim), lifting obs 101 → 105. Used
`surgery_obs101to105.py` to **insert 4 zero-weight columns** into the champion's input layer (100% parity
verified), producing the deployed **`league_robust_v3_best_105`** — i.e. the browser's v3champ. It holds
~0.77 vs the three evasive laika at fixed spawn.

---

## Reward parameters (`game_core.js` `REWARD_DEFAULTS`, lines 466–480)

The champion used the **base sparse reward** throughout; all "shaping" terms are 0.0 (dormant). It is pushed
out by win/loss + hit + self-hit + a small time penalty:

```
// terminal
win:            +1.0     // win
loss:           -1.0     // loss
winBySelfHit:   +0.15    // win because the opponent self-destructed (discounted, so it can't just wait)
timeoutPenalty: -0.7     // timeout / draw penalty (forces active resolution)
draw:            0.0

// combat (per step)
hit:            +0.35    // hit the enemy
hitShield:      +0.18    // hit a shielded enemy (half value)
selfHit:        -0.16    // ricochet hits yourself
selfHitShield:  -0.12
selfDefeat:     -0.12    // dying from a self-hit

// resource / tempo
powerup:        +0.04    // pick up a power-up
timePenalty:    -0.001   // mild per-step urgency
poisonHurt:      0.0     // (dormant) standing-in-poison penalty

// shaping terms — all 0.0 (dormant, unused by the champion)
closeRangeHit:   0.0     // point-blank-trade penalty
cleanTrade:      0.0     // hit-without-being-hit bonus
backwardPenalty: 0.0     // reversing penalty
approachCoef:    0.0     // potential-based approach reward
aimBonus:        0.0     // aim-on-target bonus
proximityBonus:  0.0  (proximityRange 200)   // point-blank pressure
cornerCoef:      0.0  (cornerRange 120)       // cutting off escape angles
```

**Two extra PPO-side pieces** (not in the table above — they are trainer logic):
- **CE-anchor (λ_ce)** — adds a cross-entropy term to the PPO loss pulling the policy toward the BC/start base,
  resisting catastrophic forgetting.
- **critic warm-up** — freeze the actor and warm up only the critic for a while, then release the policy once
  the value function is stable, so early noisy updates don't destroy the warm-start.

> Design stance: **keep the reward sparse** (win/loss primary, hits secondary) and leave "how to win" for RL to
> explore; every shaping term that would change the optimum (point-blank, corner-cutting, approach) is turned
> off — they were tried in the later "break the evasion wall" experiments and pulled the policy off course, so
> the champion does not use them.

## References

**Imitation & behavior cloning**
- Pomerleau, D. A. (1988). *ALVINN: An Autonomous Land Vehicle in a Neural Network.* NIPS 1988. https://papers.nips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html
- Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger). AISTATS 2011. https://arxiv.org/abs/1011.0686

**Reinforcement learning algorithms**
- Mnih, V., et al. (2013). *Playing Atari with Deep Reinforcement Learning* (DQN). https://arxiv.org/abs/1312.5602
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature 518:529–533. https://www.nature.com/articles/nature14236
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms* (PPO). https://arxiv.org/abs/1707.06347
- Schulman, J., et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (GAE). https://arxiv.org/abs/1506.02438
- Raffin, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR 22(268). https://jmlr.org/papers/v22/20-1364.html

**Multi-agent, self-play & league training**
- Lowe, R., et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (MADDPG). https://arxiv.org/abs/1706.02275
- Heinrich, J., & Silver, D. (2016). *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games* (NFSP). https://arxiv.org/abs/1603.01121
- Bansal, T., et al. (2017). *Emergent Complexity via Multi-Agent Competition.* https://arxiv.org/abs/1710.03748
- Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning* (AlphaStar). Nature 575:350–354. https://www.nature.com/articles/s41586-019-1724-z
- OpenAI, et al. (2019). *Dota 2 with Large Scale Deep Reinforcement Learning* (OpenAI Five; observation surgery). https://arxiv.org/abs/1912.06680
- Lanctot, M., et al. (2017). *A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning* (PSRO). https://arxiv.org/abs/1711.00832
- Timbers, F., et al. (2020). *Approximate Exploitability: Learning a Best Response.* https://arxiv.org/abs/2004.09677
- Zhang, R., et al. (2024). *A Survey on Self-Play Methods in Reinforcement Learning.* https://arxiv.org/abs/2408.01072

**Curriculum, generalization & emergent strategy**
- Florensa, C., et al. (2017). *Reverse Curriculum Generation for Reinforcement Learning.* CoRL 2017. https://arxiv.org/abs/1707.05300
- Florensa, C., et al. (2018). *Automatic Goal Generation for Reinforcement Learning Agents* (GoalGAN). https://arxiv.org/abs/1705.06366
- Kurach, K., et al. (2019). *Google Research Football: A Novel Reinforcement Learning Environment* (incl. the "Football Academy"). https://arxiv.org/abs/1907.11180
- Baker, B., et al. (2019). *Emergent Tool Use From Multi-Agent Autocurricula* (hide-and-seek). https://arxiv.org/abs/1909.07528
- Cobbe, K., et al. (2018). *Quantifying Generalization in Reinforcement Learning* (CoinRun). https://arxiv.org/abs/1812.02341

**Pursuit–evasion / predator–prey (the cornering hypothesis)**
- Janosov, M., Virágh, C., Vásárhelyi, G., & Vicsek, T. (2017). *Group chasing tactics: how to catch a faster prey.* New J. Phys. 19:053003. https://arxiv.org/abs/1701.00284
- de Souza, C., et al. (2020). *Decentralized Multi-Agent Pursuit using Deep Reinforcement Learning.* https://arxiv.org/abs/2010.08193  *(note: the repo's earlier notes mis-cited this id for the Janosov paper above)*
- Xu, S., & Dang, Z. (2025). *Emergent behaviors in multiagent pursuit evasion games within a bounded 2D grid world.* Sci. Rep. 15:29376. https://www.nature.com/articles/s41598-025-15057-x

**Tooling / multi-agent environments & continual learning**
- Terry, J. K., et al. (2021). *PettingZoo: Gym for Multi-Agent Reinforcement Learning.* https://arxiv.org/abs/2009.14471
- Raiman, J., et al. (2019). *Neural Network Surgery with Sets.* https://arxiv.org/abs/1912.06719
- Rolnick, D., et al. (2019). *Experience Replay for Continual Learning* (CLEAR). https://arxiv.org/abs/1811.11682

**Same-domain corroboration**
- Ackermann, T., Spang, M., & Gardi, H. A. (2025). *Reinforcement Learning Agent for a 2D Shooter Game.* https://arxiv.org/abs/2509.15042
