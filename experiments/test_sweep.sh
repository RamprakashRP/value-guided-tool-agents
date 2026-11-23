#!/usr/bin/env bash
# Final test runs on validation[:300] (same questions as the GPT-4o B_token baseline).
# Set DP/DA to the thresholds chosen by tune.sh. Override SPLIT to shrink if Qwen is too slow.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SPLIT=${SPLIT:-"validation[:300]"}
DP=${DP:?set DP to the tuned delta_prune}
DA=${DA:?set DA to the tuned delta_abort}
SEEDS=${SEEDS:-"0 1 2"}
OUT=results/test

for seed in $SEEDS; do
  common=(--split "$SPLIT" --seed $seed --sampling diverse --delta-prune $DP --delta-abort $DA)
  $PY -m experiments.run_search --method beam --scorer vtheta --K 3 "${common[@]}" --output $OUT/beam_K3_s${seed}.jsonl
  $PY -m experiments.run_search --method mcts --scorer vtheta --K 3 --N 3 "${common[@]}" --output $OUT/mcts_K3_N3_s${seed}.jsonl
  $PY -m experiments.run_search --method phs --scorer vtheta --K 3 "${common[@]}" --output $OUT/phs_K3_s${seed}.jsonl
done

# LATS baseline on a smaller subset, one seed.
$PY -m experiments.run_search --method lats --K 5 --N 3 --sampling diverse --split "${LATS_SPLIT:-validation[:100]}" --output $OUT/lats_K5_N3_s0.jsonl

$PY -m experiments.summarize "$OUT/*.jsonl" --csv $OUT/summary.csv
