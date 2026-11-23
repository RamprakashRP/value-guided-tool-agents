"""Language Agent Tree Search (LATS) baseline [Parthiv's Track B].

MCTS whose node values come from the policy LLM scoring its own trajectory
(Zhou et al. 2024), plus self-reflections after low-scoring answers. Unlike
the original HotpotQA setup, no ground-truth reward is used during search.
Every self-evaluation and reflection call is charged to the Eq. 1 ledger.
"""

from __future__ import annotations

import re
from typing import Any, List

from agent.search.common import Node, SearchContext, SearchResult
from agent.search.mcts import mcts_search

SELF_EVAL_INSTRUCTION = (
    "Analyze the trajectory of an agent answering a multi-hop question with search and lookup tools. "
    "Judge whether the actions and observations so far are on track to reach the correct answer, "
    "and whether any final answer is correct and concise. "
    "Respond with a single integer from 1 (hopeless) to 10 (certainly correct) and nothing else."
)

REFLECTION_INSTRUCTION = (
    "The agent's attempt below was judged unlikely to be correct. In two sentences, diagnose what went wrong "
    "and give a concrete plan for a better attempt (which articles to search, what to look up)."
)


def format_trajectory(question: str, steps: List[Any], obs_chars: int = 300) -> str:
    lines = [f"Question: {question}"]
    for s in steps:
        if s.thought:
            lines.append(f"Thought {s.step_idx}: {s.thought.strip()}")
        lines.append(f"Action {s.step_idx}: {s.action}({s.action_input})")
        obs = s.observation if len(s.observation) <= obs_chars else s.observation[:obs_chars] + "..."
        lines.append(f"Observation {s.step_idx}: {obs}")
    return "\n".join(lines)


def parse_score(text: str) -> float:
    match = re.search(r"\d+(\.\d+)?", text)
    if not match:
        return 0.5
    return min(max(float(match.group()) / 10.0, 0.0), 1.0)


class SelfEvalScorer:
    """V(n) from an extra LLM call per node, charged to the ledger."""

    def __init__(self, ctx: SearchContext, max_tokens: int = 8):
        self.ctx = ctx
        self.max_tokens = max_tokens

    def score_prefix(self, trajectory: Any) -> float:
        steps = getattr(trajectory, "steps", trajectory)
        if not steps:
            return 0.5
        messages = [
            {"role": "system", "content": SELF_EVAL_INSTRUCTION},
            {"role": "user", "content": format_trajectory(self.ctx.question, steps)},
        ]
        text, p_tok, c_tok = self.ctx.policy.generate_text(messages, self.max_tokens, self.ctx._next_seed(1))
        self.ctx.ledger.charge(p_tok, c_tok, None, kind="self_eval")
        return parse_score(text)


def lats_search(ctx: SearchContext, reflect_below: float = 0.5, max_reflections: int = 2) -> SearchResult:
    """Run MCTS with LLM self-evaluation as the value and reflections on weak answers.

    LATS has no value-threshold pruning, so delta_prune / delta_abort should be 0.
    """
    ctx.scorer = SelfEvalScorer(ctx)

    def reflect(child: Node) -> None:
        if not child.is_finished or child.value >= reflect_below or len(ctx.extra_context) >= max_reflections:
            return
        if ctx.ledger.over_budget():
            return
        messages = [
            {"role": "system", "content": REFLECTION_INSTRUCTION},
            {"role": "user", "content": format_trajectory(ctx.question, child.steps)},
        ]
        text, p_tok, c_tok = ctx.policy.generate_text(messages, 96, ctx._next_seed(1))
        ctx.ledger.charge(p_tok, c_tok, None, kind="reflection")
        ctx.extra_context.append(f"Reflection from a previous failed attempt: {text.strip()}")

    ctx.on_child_scored = reflect
    return mcts_search(ctx)
