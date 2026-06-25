#!/usr/bin/env bash
# Full "retrace the path" pipeline under PURE FIXED spawn (survival_v1), recording EVERY stage's data.
# S0 record (>=1000 wins, >=100/opp) -> S1 BC -> S2 DAgger -> S3 league PPO -> S4 final eval.
# Each stage writes its own log under runs/fixed_pipeline/ and appends a summary to metrics.md.
cd "$(dirname "$0")" || exit 1
DEMOS="data/expert_demos/fixed/aggro_fixed.jsonl"
OPPS="laika,easy_laika,stationary,laika-aggressive-pro"
RD="runs/fixed_pipeline"
M="$RD/metrics.md"
mkdir -p data/expert_demos/fixed "$RD"
ts() { date +%H:%M:%S; }
echo "# Fixed-spawn pipeline metrics (survival_v1, spawnMode fixed, obs 105) — vs the 4 laika" > "$M"

echo "=== STAGE S0 record ($(ts)): fixed-spawn demos until >=1000 wins, >=100/opp ==="
node train/record_v2_demos.js --expert laika-aggressive --opps "$OPPS" \
  --total-wins 1000 --min-wins 100 --spawn fixed --ruleset survival_v1 --out "$DEMOS" > "$RD/record.log" 2>&1
tail -2 "$RD/record.log"
[ -s "$DEMOS" ] || { echo "FAILED: no demos"; exit 1; }
{ echo; echo "## S0 demos (fixed spawn)"; grep -E "wins kept|wins by opponent" "$RD/record.log" | tail -2; } >> "$M"

echo "=== STAGE S1 naked-BC ($(ts)) ==="
python -u train/train_bc.py --data-glob "$DEMOS" --filter wins --allowed-scenarios moba1v1duel \
  --allowed-opponents "$OPPS" --allowed-experts laika-aggressive --out models/auto/fixed_bc.zip \
  --epochs 16 --fire-weight 4.0 --device cpu > "$RD/bc.log" 2>&1
[ -s models/auto/fixed_bc.zip ] || { echo "FAILED: no BC model"; exit 1; }
echo "S1 BC: $(grep -E 'final .*val_acc' "$RD/bc.log" | tail -1)"
{ echo; echo "## S1 naked-BC"; echo '```'; grep -E 'transitions|epoch .*val_acc|final .*val_acc' "$RD/bc.log" | tail -6; echo '```'; } >> "$M"

echo "=== STAGE S2 DAgger ($(ts)): warm-start BC, fixed spawn (iter 0 = BC baseline) ==="
python -u train/train_dagger.py --warm-start models/auto/fixed_bc.zip --data-glob "$DEMOS" --filter wins \
  --scenario moba1v1duel --spawn-powerups --spawn-mode fixed --ruleset survival_v1 \
  --opponents "$OPPS" --expert laika-aggressive --iters 8 --rollout-episodes 12 --epochs 3 \
  --eval-episodes 20 --out models/auto/fixed_dagger --device cpu > "$RD/dagger.log" 2>&1
[ -s models/auto/fixed_dagger.zip ] || { echo "FAILED: no DAgger model"; exit 1; }
echo "S2 DAgger best: $(grep 'BEST balanced' "$RD/dagger.log" | tail -1)"
{ echo; echo "## S2 DAgger (per-iteration win/fire%/hits; iter0 = BC baseline)"; echo '```'; \
  grep -E 'iter [0-9].*mean_win|BEST balanced' "$RD/dagger.log"; echo '```'; } >> "$M"

echo "=== STAGE S3 league-PPO ($(ts)): warm-start DAgger, reduced 4-laika pool + self ==="
python -u train/train_selfplay.py --anchor models/auto/fixed_dagger.zip --self-pool models/auto/fixed_dagger.zip \
  --data-glob "$DEMOS" --out-prefix "$RD/fixed_league" --ruleset survival_v1 --spawn-mode fixed \
  --gate-opponents "$OPPS" --script-mix "laika=0.3,easy_laika=0.2,stationary=0.2,laika-aggressive-pro=0.3" \
  --self-frac 0.34 --n-envs 6 --total-timesteps 150000 --warmup-steps 256 \
  --eval-interval 12000 --eval-episodes 18 --confirm-episodes 30 --device cpu > "$RD/league.log" 2>&1
echo "S3 promotions: $(grep -c PROMOTED "$RD/league.log")"
{ echo; echo "## S3 league-PPO (gate win-rate over training; positions = laika|easy|stationary|pro)"; echo '```'; \
  grep -E '\[gate\]' "$RD/league.log"; echo '```'; } >> "$M"

echo "=== STAGE S4 final-eval ($(ts)): best ckpt, 60 ep/opp ==="
BEST="$RD/fixed_league_best.zip"; [ -s "$BEST" ] || BEST="models/auto/fixed_dagger.zip"
python -u train/_stage_eval.py --model "$BEST" --label "S4 final (fixed-league best)" --episodes 60 \
  --opponents "$OPPS" --spawn-mode fixed --ruleset survival_v1 --out-dir "$RD" 2>&1

echo "=== PIPELINE DONE ($(ts)) ==="
