#!/usr/bin/env bash
# v123 無 L0：先 @25% CNF+SAT，完成後接 @50% CNF（表一兩欄 CNF）
set -euo pipefail
cd "$(dirname "$0")"
MODEL=../baselines/learned_shape/ranker_train_v1_v2_v3.joblib
OUT=../baselines/v4/l0_benchmark
LOG25="$OUT/run_v123_no_l0_k25.log"
LOG50="$OUT/run_v123_no_l0_k50.log"

if [[ ! -f "$OUT/v4_v123_no_l0_k25.json" ]]; then
  echo "=== v123 no-L0 @25% CNF+SAT ===" | tee -a "$LOG25"
  python3 run_v4_l0_benchmark.py --puzzle v4 --all --learned-only --no-l0 \
    --top-k 25 --model "$MODEL" \
    --learned-tag learned_k25_v123_no_l0 \
    --report-name v4_v123_no_l0_k25.json \
    --force-regen-cnf 2>&1 | tee -a "$LOG25"
fi

echo "=== v123 no-L0 @50% CNF only ===" | tee "$LOG50"
python3 run_v4_l0_benchmark.py --puzzle v4 --cnf-only --learned-only --no-l0 \
  --top-k 50 --model "$MODEL" \
  --learned-tag learned_k50_v123_no_l0 \
  --report-name v4_v123_no_l0_k50.json \
  --force-regen-cnf 2>&1 | tee -a "$LOG50"

echo "完成 → $OUT/v4_v123_no_l0_k25.json + $OUT/v4_v123_no_l0_k50.json"
