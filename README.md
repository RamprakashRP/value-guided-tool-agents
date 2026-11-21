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
│   └── test_hotpot_mcp.py# End-to-end smoke test for MCP server
├── agent/
│   ├── rollout_runner.py # [Track A] Greedy Qwen-2.5-7B & GPT-4o rollout runner
│   └── search/           # [Track B] Informed search algorithms (Parthiv)
│       ├── beam.py       # Beam search guided by V_theta
│       ├── mcts.py       # Monte Carlo Tree Search with V_theta priors
│       ├── phs.py        # Policy-guided Heuristic Search
│       └── lats.py       # Language Agent Tree Search with reflection
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

## 🏃 Running Track A Pipelines

### Collect Greedy Rollouts
```bash
python -m agent.rollout_runner --split "train[:500]" --output "data/train_rollouts.jsonl" --model "qwen2.5:7b"
```

### Retrain / Re-calibrate Value Model
```bash
python value_model/train_value.py --train-file data/train_rollouts.jsonl --val-file data/val_rollouts.jsonl --model-out data/value_model.joblib
```
