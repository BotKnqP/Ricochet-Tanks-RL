#!/usr/bin/env bash
# LONG phase: domain-randomization league (the v3-champion recipe) under PURE FIXED spawn, warm-started from
# the fixed-spawn DAgger. Randomized-dominant opponent ecology + anchor-to-policy + many steps — the real path
# that lifted laika/easy to ~0.7 at fixed (vs the quick BC->DAgger->league which stalls at laika 0.05).
# Records: reward (monitor.csv), per-opponent win-rate (gate), fire% / hits-dealt / ttk (final _stage_eval).
cd "$(dirname "$0")" || exit 1
DEMOS="data/expert_demos/fixed/aggro_fixed.jsonl"
OPPS="laika,easy_laika,stationary,laika-aggressive-pro"
RD="runs/fixed_pipeline"
M="$RD/metrics.md"
ts() { date +%H:%M:%S; }

echo "=== LONG DR-league ($(ts)): warm-start fixed_dagger, randomized-dominant pool, 350k steps ==="
python -u train/train_selfplay.py \
  --anchor models/auto/fixed_dagger.zip --self-pool models/auto/fixed_dagger.zip --anchor-to-policy \
  --data-glob "$DEMOS" --out-prefix "$RD/fixed_dr_league" --ruleset survival_v1 --spawn-mode fixed \
  --gate-opponents "$OPPS" \
  --script-mix "randomized=0.45,laika=0.2,easy_laika=0.1,stationary=0.05,laika-aggressive-pro=0.2" \
  --self-frac 0.25 --n-envs 8 --total-timesteps 350000 --warmup-steps 512 \
  --eval-interval 20000 --eval-episodes 20 --confirm-episodes 40 --device cpu > "$RD/dr_league.log" 2>&1
echo "DR-league promotions: $(grep -c PROMOTED "$RD/dr_league.log")"

# record the gate win-rate progression + the reward-curve summary
{ echo; echo "## LONG domain-randomization league (fixed spawn, randomized-dominant, 350k steps)"; echo;
  echo "### win-rate over training (gate; positions = laika | easy | stationary | pro)"; echo '```';
  grep -E "anchor baselines|\[gate\]" "$RD/dr_league.log"; echo '```';
  echo "### reward (monitor.csv per-episode): first/last 3 + count";
  echo '```'; head -4 runs/fixed_dr_league/monitor.csv 2>/dev/null; echo "...";
  tail -3 runs/fixed_dr_league/monitor.csv 2>/dev/null;
  echo "episodes: $(($(wc -l < runs/fixed_dr_league/monitor.csv 2>/dev/null)-2))"; echo '```'; } >> "$M"

echo "=== FINAL eval ($(ts)): DR-league best, 60 ep/opp (win / fire% / hits / ttk) ==="
BEST="$RD/fixed_dr_league_best.zip"; [ -s "$BEST" ] || BEST="$RD/fixed_dr_league_latest.zip"
python -u train/_stage_eval.py --model "$BEST" --label "LONG DR-league final (fixed best)" --episodes 60 \
  --opponents "$OPPS" --spawn-mode fixed --ruleset survival_v1 --out-dir "$RD" 2>&1

echo "=== LONG TRAIN DONE ($(ts)) ==="
