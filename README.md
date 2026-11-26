# Value-Guided Trajectory Search for Multi-Step Tool Agents

**CS 680 Project** — An empirical framework for value-guided trajectory search, evaluation, and tool-use in language agent rollouts on multi-hop question answering (HotpotQA).

---

## 📁 Repository Structure

```text
value-guided-tool-agents/
├── shared/
│   ├── schema.py         # StepRecord & Trajectory dataclasses (agreed contract)
│   └── interfaces.py     # score_prefix(trajectory) -> float signature
├── envs/
│   ├── hotpot_mcp.py     # [Track A] MCP 2.x stdio server for HotpotQA
│   ├── hotpot_env.py     # [Track B] Same tools in-process, with snapshot/restore for branching
│   └── test_hotpot_mcp.py# End-to-end smoke test for MCP server
├── agent/
│   ├── rollout_runner.py # [Track A] Greedy Qwen-2.5-7B & GPT-4o rollout runner
│   └── search/           # [Track B] Informed search algorithms (Parthiv)
│       ├── common.py     # Search tree, Eq. 1 TokenLedger, policies, scorers, prune/backtrack
│       ├── beam.py       # Beam search guided by V_theta
│       ├── mcts.py       # Monte Carlo Tree Search with V_theta leaf values
│       ├── phs.py        # Policy-guided Heuristic Search, phi(n) = g / (pi * V)
│       └── lats.py       # LATS baseline: LLM self-evaluation + reflection
├── experiments/          # [Track B] run_search.py, summarize.py, smoke/tune/test_sweep scripts
├── tests/                # pytest: env parity with MCP server, search + ledger tests (mock LLM)
├── value_model/
│   ├── features.py       # [Track A] Feature extractor phi(tau_{1:t})
│   └── train_value.py    # [Track A] Value model training, calibration & ECE
├── data/
│   ├── value_model.joblib# Serialized trained value model V_theta (committed)
│   └── .gitignore        # Ignores large .jsonl/.parquet rollout files
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup Instructions

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install & Start Local Ollama Model
```bash
# Pull Qwen 2.5 7B Instruct
ollama pull qwen2.5:7b

# Ensure Ollama daemon is running
ollama list
```

---

## 📊 Track A Baseline & Calibration Results

Evaluated on **HotpotQA `distractor` (`validation[:300]`)**:

| Model / Approach | Exact Match (EM) | Avg Budget ($B_{\text{token}}$) | Validation AUC-ROC | Step 4+ AUC-ROC | ECE (10 bins) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPT-4o Greedy Baseline** | **51.7%** | 2,511 tokens/query | — | — | — | — |
| **Value Model $V_\theta$** (`value_model.joblib`) | — | — | **0.7369** | **0.8038** | **0.0634** | **0.1755** |

### Key Findings
- **Calibration**: The value model achieves an Expected Calibration Error of $0.0634$, indicating well-calibrated probabilities for branch pruning.
- **Deeper Trajectory Discrimination**: The model's AUC-ROC increases from $0.647$ at Step 1 to **$0.8038$ at Step 4+**, demonstrating strong ability to discern successful vs. failing tool trajectories as reasoning depth increases.
- **Top Predictive Features**:
  1. `normalized_total_tokens` (negative weight: high token consumption correlates with spinning in loops)
  2. `total_errors` (negative weight: frequent lookup/search misses indicate wrong direction)
  3. `repeat_action_count` (negative weight: loop detection signal)
  4. `logit_margin` / `token_entropy` (positive correlation with policy confidence)

---

## 🔌 Track B Integration Guide

Track B algorithms (Beam Search, MCTS, PHS, LATS) can directly evaluate any trajectory prefix $\tau_{1:t}$ using `score_prefix` from `shared.interfaces` without retraining.

### 1. Scoring a Trajectory Prefix
`score_prefix` automatically lazy-loads the serialized model `data/value_model.joblib`:

```python
from shared.interfaces import score_prefix
from shared.schema import StepRecord

# Can pass a list of dicts, a list of StepRecord objects, or a Trajectory instance
steps = [
    StepRecord(
        step_idx=1,
        action="search",
        action_input={"entity": "Scott Derrickson"},
        observation="[Scott Derrickson] American director...",
        info={"token_entropy": 0.04, "logit_margin": 13.5, "is_error": False, "total_tokens": 480}
    ),
    StepRecord(
        step_idx=2,
        action="search",
        action_input={"entity": "Ed Wood"},
        observation="[Ed Wood] American filmmaker...",
        info={"token_entropy": 0.03, "logit_margin": 14.1, "is_error": False, "total_tokens": 1050}
    )
]

# Returns expected success probability P(success | tau_{1:t}) in [0.0, 1.0]
prob = score_prefix(steps)
print(f"Prefix Success Probability: {prob:.4f}")
```

### 2. Interfacing with HotpotQA MCP Server
Spawn the server via stdio transport and execute tools in search trees:

```python
import asyncio
import json
import os
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def search_step():
    server_script = Path("envs/hotpot_mcp.py").resolve()
    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", str(server_script)],
        env=os.environ.copy()
    )
    
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # 1. Load episode context
            context_json = json.dumps({"Article Title": ["Sentence 1.", "Sentence 2."]})
            await session.call_tool("load_episode", arguments={"context_json": context_json})
            
            # 2. Execute tools during search tree expansion
            obs = await session.call_tool("search", arguments={"entity": "Article Title"})
            print("Observation:", obs)

asyncio.run(search_step())
```

---

## 🌲 Track B: Value-Guided Search

All search methods share one ReAct expansion step (`SearchContext.expand` in `agent/search/common.py`): sample K actions from Qwen, execute each on a forked `HotpotEnv`, and score the child prefix with a scorer (`vtheta`, or the dummy `random` / `heuristic`).

- **Eq. 1 accounting:** `TokenLedger` records prompt + completion tokens of *every* LLM call made for a query, including samples whose branch is later pruned, and LATS self-evaluation / reflection calls. `C_q` in the results is this total; `pruned_tokens` is the part spent on pruned branches.
- **Per-path tokens:** `info["total_tokens"]` on each step is the cumulative count along that branch only, matching how V_theta's training rollouts were logged.
- **Prompt parity:** system prompt and tool schemas are identical to `rollout_runner.py` (enforced by `tests/test_env_parity.py`). `--method greedy` reproduces `rollout_runner` exactly (same actions and per-step tokens).
- **Proposal sampling (`--sampling`):** Qwen puts ~96% probability on its top tool call, so K i.i.d. samples are usually identical. `diverse` (default) makes sample 1 greedy (the greedy path is always in the tree) and asks samples 2..K for an action different from those already proposed; the note is added to the request only, never to the stored history. `iid` keeps K independent samples for ablation. All calls are charged.
- **Cheaper levels:** `--beam-width W` keeps W beam nodes while still sampling K actions per node (default W = K). `--gate {none,value,entropy,either}` makes every non-root expansion take the greedy action alone (1 call) unless the node looks uncertain: `V < --gate-value` (default 0.33) or the greedy action's mean token entropy `> --gate-entropy` (default 0.08). Only then are the K-1 alternatives sampled. The root always branches. Defaults were set from smoke-run value logs (148 expansions): V's 30th percentile is 0.34 and entropy's 85th percentile is ~0.08, so `either` branches on ~35% of non-root expansions. Results report the branch rate.
- **Controls:** children with `V < delta_prune` are dropped; if no child reaches `delta_abort`, search backtracks to the best unexpanded node with `V >= delta_abort`, otherwise aborts (Y = 0). `--max-query-tokens` (default 3 x B_token) is a hard per-query cap.
- **Forced finish:** if the cap is hit before any `finish`, one extra greedy call asks the best open branch to answer now (charged as `force_finish`; the row has `forced_finish: true`). `--no-force-finish` restores strict abort.
- **Answer selection:** the finished trajectory with the highest V; the ground truth is only used after search to score it.

```bash
# Ollama with a 4k context (one-time: brew install ollama && ollama pull qwen2.5:7b)
OLLAMA_CONTEXT_LENGTH=4096 ollama serve

pytest                                        # offline tests (mock LLM)
python -m experiments.run_search --method beam --scorer vtheta --K 3 \
    --delta-prune 0.1 --delta-abort 0.15 --split "validation[:20]"
python -m experiments.run_search --method mcts --K 3 --N 3 ...   # also: phs, lats, greedy
python -m experiments.run_search --method beam --policy mock ...  # no LLM, for debugging
python -m experiments.summarize "results/**/*.jsonl"             # EM, mean C(q), bootstrap CIs

bash experiments/smoke.sh                     # every method on a few questions
bash experiments/tune.sh                      # delta grid, gated width-1 beam, validation[300:450] (disjoint from test)
DP=0.1 DA=0.15 bash experiments/test_sweep.sh # gated beam/MCTS/PHS x 3 seeds + LATS, all on validation[:300]
SEEDS="1 2" RUN_LATS=0 DP=.. DA=.. bash experiments/test_sweep.sh   # e.g. split seeds across machines
```

**On a shared GPU server**, run your own Ollama on a private port and split each config over parallel workers (each worker writes its own `*.shardIofN.jsonl`; `summarize.py` merges them):

```bash
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11500 OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_NUM_PARALLEL=4 ollama serve
export OLLAMA_BASE_URL=http://127.0.0.1:11500/v1   # read by run_search.py as the default --base-url
SHARDS=4 bash experiments/tune.sh
SHARDS=4 DP=.. DA=.. bash experiments/test_sweep.sh
python -m experiments.run_search --shard 0/4 ...   # single worker by hand: questions with index = 0 mod 4
```

Runs append to JSONL and resume where they stopped. Each row holds `is_correct`, `C_q`, `pruned_tokens`, `n_llm_calls`, `stop_reason`, the chosen trajectory and a per-node `value_log` (every expanded prefix with its V), which can be used for step-wise analysis or to refit V_theta on search-time prefixes.

---

## 🏃 Running Track A Pipelines

### Collect Greedy Rollouts
```bash
python -m agent.rollout_runner --split "train[:500]" --output "data/train_rollouts.jsonl" --model "qwen2.5:7b"
```

### Retrain / Re-calibrate Value Model
```bash
python value_model/train_value.py --train-file data/train_rollouts.jsonl --val-file data/val_rollouts.jsonl --model-out data/value_model.joblib
```
