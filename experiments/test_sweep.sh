#!/usr/bin/env bash
# Final test runs on validation[:300] (same questions as the GPT-4o B_token baseline).
# Every method runs on the same questions. Set DP/DA to the thresholds chosen by tune.sh.
# SEEDS can be split across machines (e.g. SEEDS="1 2" on another box); summarize.py merges files.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SPLIT=${SPLIT:-"validation[:300]"}
DP=${DP:?set DP to the tuned delta_prune}
DA=${DA:?set DA to the tuned delta_abort}
SEEDS=${SEEDS:-"0 1 2"}
OUT=results/test

for seed in $SEEDS; do
  common=(--split "$SPLIT" --seed $seed --scorer vtheta --K 3 --sampling diverse
          --gate either --gate-value 0.33 --gate-entropy 0.08 --delta-prune $DP --delta-abort $DA)
  $PY -m experiments.run_search --method beam --beam-width 1 "${common[@]}" --output $OUT/beam_K3_W1_gate_s${seed}.jsonl
  $PY -m experiments.run_search --method mcts --N 3 "${common[@]}" --output $OUT/mcts_K3_N3_gate_s${seed}.jsonl
  $PY -m experiments.run_search --method phs "${common[@]}" --output $OUT/phs_K3_gate_s${seed}.jsonl
done

# LATS baseline as in Zhou et al. (no V_theta, no gate), one seed, same 300 questions.
if [[ "${RUN_LATS:-1}" == "1" ]]; then
  $PY -m experiments.run_search --method lats --K 5 --N 3 --sampling diverse --split "$SPLIT" \
    --output $OUT/lats_K5_N3_s0.jsonl
fi

$PY -m experiments.summarize "$OUT/*.jsonl" --csv $OUT/summary.csv
