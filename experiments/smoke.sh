#!/usr/bin/env bash
# Smoke test of every search method against local Ollama qwen2.5:7b.
# Usage: bash experiments/smoke.sh   (Ollama must be running: OLLAMA_CONTEXT_LENGTH=4096 ollama serve)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=results/smoke
SPLIT="validation[:20]"
DP=0.1
DA=0.15

run() { $PY -m experiments.run_search --split "$SPLIT" --sampling diverse --delta-prune $DP --delta-abort $DA "$@"; }

# Quick pass: 2 questions per method to catch integration bugs.
for m in beam mcts phs lats; do
  run --method $m --scorer vtheta --limit 2 --output $OUT/quick_${m}.jsonl
done

# Smoke: dummy (heuristic) scorer, then V_theta.
run --method beam --scorer heuristic --output $OUT/beam_heuristic.jsonl
run --method beam --scorer vtheta --output $OUT/beam_vtheta.jsonl
run --method mcts --scorer vtheta --limit 10 --output $OUT/mcts_vtheta.jsonl
run --method phs --scorer vtheta --limit 10 --output $OUT/phs_vtheta.jsonl
run --method lats --limit 5 --output $OUT/lats.jsonl
run --method greedy --output $OUT/greedy.jsonl

$PY -m experiments.summarize "$OUT/beam_*.jsonl" "$OUT/mcts_*.jsonl" "$OUT/phs_*.jsonl" "$OUT/lats.jsonl" "$OUT/greedy.jsonl"
