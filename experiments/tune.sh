#!/usr/bin/env bash
# Grid over delta_prune x delta_abort for Beam (K=3) on the tuning split.
# Pick the best (EM, then lowest mean C(q)) and reuse it for MCTS and PHS.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SPLIT=${SPLIT:-"validation[300:450]"}
OUT=results/tune

for dp in 0.05 0.1 0.2; do
  for da in 0.1 0.2 0.3; do
    $PY -m experiments.run_search --method beam --scorer vtheta --K 3 --sampling diverse --split "$SPLIT" \
      --delta-prune $dp --delta-abort $da --no-value-log \
      --output $OUT/beam_K3_dp${dp}_da${da}.jsonl
  done
done

$PY -m experiments.summarize "$OUT/*.jsonl" --csv $OUT/summary.csv
