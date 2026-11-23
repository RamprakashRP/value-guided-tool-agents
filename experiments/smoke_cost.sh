#!/usr/bin/env bash
# Cheaper-per-level variants vs greedy and the ungated smoke runs (results/smoke/).
# Usage: bash experiments/smoke_cost.sh   (Ollama must be running: OLLAMA_CONTEXT_LENGTH=4096 ollama serve)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=results/smoke_cost
SPLIT="validation[:20]"

run() {
  $PY -m experiments.run_search --split "$SPLIT" --scorer vtheta --sampling diverse --K 3 \
    --delta-prune 0.1 --delta-abort 0.15 "$@"
}

run --method beam --beam-width 1 --gate none --output $OUT/beam_W1_gate-none.jsonl
run --method beam --beam-width 3 --gate either --output $OUT/beam_W3_gate-either.jsonl
run --method beam --beam-width 1 --gate either --output $OUT/beam_W1_gate-either.jsonl
run --method beam --beam-width 2 --gate either --output $OUT/beam_W2_gate-either.jsonl
run --method mcts --N 3 --gate either --limit 10 --output $OUT/mcts_gate-either.jsonl
run --method phs --gate either --limit 10 --output $OUT/phs_gate-either.jsonl

$PY -m experiments.summarize "$OUT/*.jsonl" results/smoke/greedy.jsonl \
  results/smoke/beam_vtheta.jsonl results/smoke/mcts_vtheta.jsonl results/smoke/phs_vtheta.jsonl \
  --csv $OUT/summary.csv
