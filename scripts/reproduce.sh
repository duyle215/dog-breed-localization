#!/usr/bin/env bash
# Full experiment: baselines, three training seeds, aggregated metrics and figures.
# Usage: PYTHON=path/to/python bash scripts/reproduce.sh
set -euo pipefail
PY="${PYTHON:-python}"

"$PY" -m dogloc.baselines --config configs/default.yaml
for seed in 42 43 44; do
  "$PY" -m dogloc.train --config configs/default.yaml --name "largest_seed${seed}" "seed=${seed}"
done
"$PY" -m dogloc.aggregate runs/largest_seed42 runs/largest_seed43 runs/largest_seed44 \
  --out runs/summary/largest.json
cp runs/baselines_largest.json runs/summary/
"$PY" -m dogloc.visualize runs/largest_seed42 --out assets
