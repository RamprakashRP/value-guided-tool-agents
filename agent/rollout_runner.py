"""Greedy ReAct Rollout Runner on HotpotQA using local Ollama and HotpotQA MCP Server.

Logs interaction steps into shared.schema.Trajectory objects saved in JSONL format.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import datasets
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from tqdm import tqdm

from shared.schema import StepRecord, Trajectory


# ---------------------------------------------------------------------------
# Exact Match (EM) Evaluation
# ---------------------------------------------------------------------------
def normalize_answer(s: Optional[str]) -> str:
    """Normalize text for HotpotQA evaluation:
    - Lowercase
    - Remove punctuation
    - Remove articles ('a', 'an', 'the')
    - Collapse whitespace
    """
    if s is None:
        return ""
    text = str(s).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).strip()


def compute_exact_match(prediction: Optional[str], ground_truth: Optional[str]) -> int:
    """Compute HotpotQA Exact Match (EM) score (1 or 0)."""
    return 1 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0


# ---------------------------------------------------------------------------
# Logprob & Uncertainty Metric Extraction
# ---------------------------------------------------------------------------
def extract_logprob_metrics(choice: Any) -> Tuple[float, float, float]:
    """Compute action_logprob, logit_margin, and token_entropy from choice logprobs.

    - action_logprob: sum of token logprob across generated tokens.
    - logit_margin: mean of (top[0].logprob - top[1].logprob) where >= 2 top logprobs exist.
    - token_entropy: mean approximate entropy -sum(p * log(p)) over top_logprobs (p = exp(lp)), clamped >= 0.0.
    """
    action_logprob = 0.0
    logit_margin = 0.0
    token_entropy = 0.0

    if not choice or not getattr(choice, "logprobs", None) or not choice.logprobs.content:
        return action_logprob, logit_margin, token_entropy

    tokens = choice.logprobs.content
    valid_tokens = [t for t in tokens if getattr(t, "logprob", None) is not None]
    if not valid_tokens:
        return action_logprob, logit_margin, token_entropy

    # 1. Action logprob: sum of token logprobs
    action_logprob = float(sum(t.logprob for t in valid_tokens))

    # 2. Logit margin & Token entropy
    margins: List[float] = []
    entropies: List[float] = []

    for t in valid_tokens:
        top_lps = getattr(t, "top_logprobs", None) or []
        if len(top_lps) >= 2:
            lp0 = top_lps[0].logprob
            lp1 = top_lps[1].logprob
            if lp0 is not None and lp1 is not None:
                margins.append(float(lp0 - lp1))

        if top_lps:
            terms = []
            for item in top_lps:
                lp = getattr(item, "logprob", None)
                if lp is not None:
                    p = math.exp(lp)
                    if p > 0.0:
                        terms.append(p * math.log(p))
            if terms:
                entropies.append(max(0.0, -sum(terms)))

    if margins:
        logit_margin = float(sum(margins) / len(margins))

    if entropies:
        token_entropy = float(sum(entropies) / len(entropies))

    return action_logprob, logit_margin, token_entropy


def extract_result_text(call_result: Any) -> str:
    """Extract plain text from MCP tool call response."""
    if hasattr(call_result, "content"):
        texts = [item.text for item in call_result.content if hasattr(item, "text")]
        if texts:
            return "\n".join(texts)
    return str(call_result)


# ---------------------------------------------------------------------------
# Rollout Runner
# ---------------------------------------------------------------------------
class RolloutRunner:
    """Orchestrates greedy ReAct rollouts on HotpotQA."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        max_steps: int = 8,
    ):
        self.model = model
        self.max_steps = max_steps
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def run_episode(
        self,
        session: ClientSession,
        row: Dict[str, Any],
        llm_tools: List[Dict[str, Any]],
    ) -> Trajectory:
        """Execute a single greedy ReAct episode."""
        task_id = str(row.get("id", ""))
        question = row.get("question", "")
        ground_truth = row.get("answer", "")

        # 1. Format and load context into MCP Server
        titles = row.get("context", {}).get("title", [])
        sentences_list = row.get("context", {}).get("sentences", [])
        context_dict = {
            title: sentences for title, sentences in zip(titles, sentences_list)
        }
        context_json = json.dumps(context_dict)
        await session.call_tool("load_episode", arguments={"context_json": context_json})

        # 2. Initialize Trajectory & conversation history
        trajectory = Trajectory(
            task_id=task_id,
            question=question,
            ground_truth=ground_truth,
            metadata={"model": self.model, "query_id": task_id, "mode": "greedy"},
        )

        system_prompt = (
            "You are an intelligent multi-hop question answering agent.\n"
            "You have access to the following tools:\n"
            "- search(entity: str): Search for an article by entity/title.\n"
            "- lookup(keyword: str): Scan the current article for the next sentence containing keyword.\n"
            "- finish(answer: str): Submit the final answer.\n\n"
            "Solve the question step-by-step. Gather facts using search and lookup.\n"
            "When calling finish(answer), provide ONLY the exact, concise answer phrase (e.g. 'yes', 'no', an entity name, or a number) without conversational filler or full sentences."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]

        total_tokens = 0
        final_answer: Optional[str] = None
        is_correct = False

        for step_idx in range(1, self.max_steps + 1):
            # Call LLM with logprobs and tools
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=llm_tools,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=2,
                )
            except Exception as e:
                # Safe fallback if endpoint errors on logprobs
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=llm_tools,
                    temperature=0.0,
                )

            choice = response.choices[0]
            message = choice.message
            tool_calls = getattr(message, "tool_calls", None)

            # Record token usage
            usage = getattr(response, "usage", None)
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            step_tokens = prompt_tokens + completion_tokens
            total_tokens += step_tokens

            # Extract uncertainty metrics
            action_logprob, logit_margin, token_entropy = extract_logprob_metrics(choice)

            # Case A: Model issued tool calls
            if tool_calls:
                call = tool_calls[0]
                tool_name = call.function.name
                raw_args = call.function.arguments

                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except Exception:
                    arguments = {"raw_input": raw_args}

                # Normalize argument keys for common variations
                if tool_name == "search" and "entity" not in arguments:
                    first_val = next(iter(arguments.values())) if arguments else question
                    arguments = {"entity": str(first_val)}
                elif tool_name == "lookup" and "keyword" not in arguments:
                    first_val = next(iter(arguments.values())) if arguments else ""
                    arguments = {"keyword": str(first_val)}
                elif tool_name == "finish" and "answer" not in arguments:
                    first_val = next(iter(arguments.values())) if arguments else ""
                    arguments = {"answer": str(first_val)}

                # Execute tool call on MCP server
                mcp_res = await session.call_tool(tool_name, arguments=arguments)
                observation = extract_result_text(mcp_res)

                # Check observation error status
                obs_stripped = observation.strip()
                is_error = obs_stripped.startswith("Error:") or obs_stripped.startswith("Could not find")

                thought_content = message.content or ""

                if tool_name == "finish":
                    final_answer = str(arguments.get("answer", "")).strip()
                    em_score = compute_exact_match(final_answer, ground_truth)
                    is_correct = bool(em_score == 1)
                    reward = 1.0 if is_correct else 0.0

                    step = StepRecord(
                        step_idx=step_idx,
                        thought=thought_content,
                        action="finish",
                        action_input=arguments,
                        observation=observation,
                        reward=reward,
                        done=True,
                        info={
                            "action_logprob": action_logprob,
                            "logit_margin": logit_margin,
                            "token_entropy": token_entropy,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "is_error": is_error,
                        },
                    )
                    trajectory.add_step(step)
                    break
                else:
                    # search or lookup tool step
                    step = StepRecord(
                        step_idx=step_idx,
                        thought=thought_content,
                        action=tool_name,
                        action_input=arguments,
                        observation=observation,
                        reward=0.0,
                        done=False,
                        info={
                            "action_logprob": action_logprob,
                            "logit_margin": logit_margin,
                            "token_entropy": token_entropy,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "is_error": is_error,
                        },
                    )
                    trajectory.add_step(step)

                    # Update history
                    messages.append({
                        "role": "assistant",
                        "content": thought_content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                        ],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": observation,
                    })

            # Case B: Model returned plain text without a tool call
            else:
                thought_content = message.content or ""
                reminder_obs = (
                    "You responded with text instead of calling a tool. "
                    "If you have determined the final answer, please call finish(answer=...). "
                    "Otherwise, call search(entity=...) or lookup(keyword=...) to gather facts."
                )

                step = StepRecord(
                    step_idx=step_idx,
                    thought=thought_content,
                    action="reminder",
                    action_input="",
                    observation=reminder_obs,
                    reward=0.0,
                    done=(step_idx >= self.max_steps),
                    info={
                        "action_logprob": action_logprob,
                        "logit_margin": logit_margin,
                        "token_entropy": token_entropy,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "is_error": False,
                    },
                )
                trajectory.add_step(step)

                messages.append({"role": "assistant", "content": thought_content})
                messages.append({"role": "user", "content": reminder_obs})

        # Finalize episode status
        trajectory.final_answer = final_answer
        trajectory.is_correct = is_correct
        trajectory.metadata["total_tokens"] = total_tokens
        trajectory.metadata["em_score"] = 1 if is_correct else 0

        return trajectory

    async def close(self) -> None:
        """Close client sessions."""
        await self.client.close()


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(
        description="Greedy ReAct rollouts on HotpotQA using local Ollama and MCP."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train[:500]",
        help="Hugging Face split string (default: 'train[:500]')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/train_rollouts.jsonl",
        help="Output JSONL file path (default: 'data/train_rollouts.jsonl')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Model name (default: 'qwen2.5:7b')",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434/v1",
        help="OpenAI-compatible base URL (default: 'http://localhost:11434/v1')",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "ollama"),
        help="API key (default: os.getenv('OPENAI_API_KEY') or 'ollama')",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Max tool steps per episode (default: 8)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read existing query_ids for checkpointing / resumption
    existing_ids: Set[str] = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    qid = (
                        data.get("task_id")
                        or data.get("query_id")
                        or data.get("metadata", {}).get("query_id")
                    )
                    if qid:
                        existing_ids.add(str(qid))
                except Exception:
                    pass
        print(f"Loaded {len(existing_ids)} existing trajectory IDs from {output_path}")

    # 2. Load dataset
    print(f"Loading HotpotQA split '{args.split}'...")
    dataset = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    print(f"Dataset contains {len(dataset)} examples.")

    # 3. Setup MCP server process parameters
    server_script = Path(__file__).resolve().parent.parent / "envs" / "hotpot_mcp.py"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", str(server_script)],
        env=os.environ.copy(),
    )

    runner = RolloutRunner(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_steps=args.max_steps,
    )

    # 4. Connect to MCP server and run rollouts
    total_evaluated = 0
    total_em = 0
    cumulative_tokens = 0

    try:
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Discover tools and filter to search, lookup, finish
                tools_response = await session.list_tools()
                llm_tools = []
                for t in tools_response.tools:
                    if t.name in ["search", "lookup", "finish"]:
                        llm_tools.append({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description or "",
                                "parameters": t.input_schema or {"type": "object", "properties": {}},
                            },
                        })

                print(f"Initialized MCP session. LLM tools available: {[t['function']['name'] for t in llm_tools]}")

                pbar = tqdm(dataset, desc="Rollouts")
                for row in pbar:
                    row_id = str(row.get("id", ""))
                    if row_id in existing_ids:
                        continue

                    traj = await runner.run_episode(session, row, llm_tools)

                    # Append to output JSONL immediately
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(traj.to_json() + "\n")
                        f.flush()

                    existing_ids.add(row_id)
                    total_evaluated += 1
                    if traj.is_correct:
                        total_em += 1
                    cumulative_tokens += traj.metadata.get("total_tokens", 0)

                    em_rate = (total_em / total_evaluated) if total_evaluated > 0 else 0.0
                    avg_tok = (cumulative_tokens / total_evaluated) if total_evaluated > 0 else 0.0

                    pbar.set_postfix({
                        "EM": f"{em_rate * 100:.1f}%",
                        "AvgTok": f"{avg_tok:.0f}",
                    })

    finally:
        await runner.close()

    print(f"\nDone! Evaluated {total_evaluated} trajectories. Saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
