#!/usr/bin/env bash
# Grid over delta_prune x delta_abort for the gated width-1 Beam (K=3) on the tuning split.
# The tuning split is disjoint from the test split (validation[:300]).
# Pick the best (EM, then lowest mean C(q)) and reuse it for MCTS and PHS in test_sweep.sh.
# SHARDS=4 runs each config as 4 parallel workers (pair with OLLAMA_NUM_PARALLEL=4).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SPLIT=${SPLIT:-"validation[300:450]"}
OUT=results/tune
source experiments/_shards.sh

for dp in 0.05 0.1 0.2; do
  for da in 0.1 0.2 0.3; do
    run_sharded $OUT/beam_K3_W1_gate_dp${dp}_da${da}.jsonl \
      --method beam --scorer vtheta --K 3 --beam-width 1 \
      --sampling diverse --gate either --gate-value 0.33 --gate-entropy 0.08 --split "$SPLIT" \
      --delta-prune $dp --delta-abort $da --no-value-log
  done
done

$PY -m experiments.summarize "$OUT/*.jsonl" --csv $OUT/summary.csv
