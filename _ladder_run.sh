#!/usr/bin/env bash
# Run the fixed-spawn ELO ladder in parallel: NSLICES node workers play disjoint ordered-pair slices,
# then _ladder_rank.js aggregates + fits Bradley-Terry stable Elo. Usage: bash _ladder_run.sh [GAMES] [NSLICES]
cd "$(dirname "$0")" || exit 1
G="${1:-60}"; NS="${2:-12}"
mkdir -p runs/ladder
rm -f runs/ladder/slice_*.json
ts() { date +%H:%M:%S; }
echo "=== ladder start $(ts): $NS workers x ${G} games/cell ==="
for k in $(seq 0 $((NS-1))); do
  node _ladder_worker.js --slice "$k" --nslices "$NS" --games "$G" &
done
wait
echo "=== workers done $(ts); ranking ==="
node _ladder_rank.js
echo "=== ladder done $(ts) ==="
