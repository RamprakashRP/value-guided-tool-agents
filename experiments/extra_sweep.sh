#!/usr/bin/env bash
# Extra runs on validation[:300], seed 0, same DP/DA and gate as test_sweep.sh:
#   1. Qwen greedy baseline (Table 1 "SLM Baseline").
#   2. Scorer ablation: gated width-1 beam with random / heuristic scores instead of V_theta.
#   3. Search budget: K in {2, 5} for beam/MCTS/PHS, and N=5 for MCTS.
# SHARDS=4 runs each config as 4 parallel workers (pair with OLLAMA_NUM_PARALLEL=4).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SPLIT=${SPLIT:-"validation[:300]"}
DP=${DP:?set DP to the tuned delta_prune}
DA=${DA:?set DA to the tuned delta_abort}
OUT=results/extra
source experiments/_shards.sh

gate=(--sampling diverse --gate either --gate-value 0.33 --gate-entropy 0.08 --delta-prune $DP --delta-abort $DA)

run_sharded $OUT/greedy.jsonl --method greedy --split "$SPLIT"

for scorer in random heuristic; do
  run_sharded $OUT/beam_K3_W1_gate_${scorer}.jsonl --method beam --beam-width 1 --K 3 --scorer $scorer \
    "${gate[@]}" --split "$SPLIT"
done

for k in 2 5; do
  run_sharded $OUT/beam_K${k}_W1_gate.jsonl --method beam --beam-width 1 --K $k --scorer vtheta "${gate[@]}" --split "$SPLIT"
  run_sharded $OUT/mcts_K${k}_N3_gate.jsonl --method mcts --N 3 --K $k --scorer vtheta "${gate[@]}" --split "$SPLIT"
  run_sharded $OUT/phs_K${k}_gate.jsonl --method phs --K $k --scorer vtheta "${gate[@]}" --split "$SPLIT"
done
run_sharded $OUT/mcts_K3_N5_gate.jsonl --method mcts --N 5 --K 3 --scorer vtheta "${gate[@]}" --split "$SPLIT"

$PY -m experiments.summarize "$OUT/*.jsonl" --csv $OUT/summary.csv
