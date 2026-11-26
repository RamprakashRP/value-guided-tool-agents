# Sourced by tune.sh / test_sweep.sh.
# run_sharded OUT.jsonl ARGS...: runs experiments.run_search with ARGS, split over $SHARDS parallel
# workers (one output file per shard; summarize.py merges them). SHARDS=1 runs a single process.
# Point at a non-default Ollama server with OLLAMA_BASE_URL (e.g. http://127.0.0.1:11500/v1).
SHARDS=${SHARDS:-1}

run_sharded() {
  local out=$1
  shift
  if [[ "$SHARDS" -le 1 ]]; then
    $PY -m experiments.run_search "$@" --output "$out"
    return
  fi
  local pids=() i
  for ((i = 0; i < SHARDS; i++)); do
    $PY -m experiments.run_search "$@" --shard "$i/$SHARDS" \
      --output "${out%.jsonl}.shard${i}of${SHARDS}.jsonl" &
    pids+=($!)
  done
  local pid status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  return $status
}
