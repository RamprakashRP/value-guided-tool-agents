"""Shared machinery for value-guided search over ReAct trajectories.

Holds the search tree node, the Eq. 1 token ledger, LLM policies (Ollama and a
mock for offline tests), prefix scorers, and expand/prune/backtrack helpers
used by beam, MCTS, PHS and LATS.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from agent.rollout_runner import compute_exact_match, extract_logprob_metrics
from envs.hotpot_env import EnvState, HotpotEnv
from shared.interfaces import score_prefix
from shared.schema import StepRecord, Trajectory

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALUE_MODEL = REPO_ROOT / "data" / "value_model.joblib"

# Must stay byte-identical to agent/rollout_runner.py so token counts match the greedy baselines.
SYSTEM_PROMPT = (
    "You are an intelligent multi-hop question answering agent.\n"
    "You have access to the following tools:\n"
    "- search(entity: str): Search for an article by entity/title.\n"
    "- lookup(keyword: str): Scan the current article for the next sentence containing keyword.\n"
    "- finish(answer: str): Submit the final answer.\n\n"
    "Solve the question step-by-step. Gather facts using search and lookup.\n"
    "When calling finish(answer), provide ONLY the exact, concise answer phrase (e.g. 'yes', 'no', an entity name, or a number) without conversational filler or full sentences."
)

REMINDER_OBS = (
    "You responded with text instead of calling a tool. "
    "If you have determined the final answer, please call finish(answer=...). "
    "Otherwise, call search(entity=...) or lookup(keyword=...) to gather facts."
)

FORCE_FINISH_NOTE = "Token budget exhausted. Call finish(answer=...) now with your best answer."


def exclusion_note(proposed: List[Candidate]) -> str:
    """Request-only instruction asking for an action not yet proposed at this node."""
    seen = ", ".join(dict.fromkeys(c.describe() for c in proposed))
    return f"Propose a different next action than: {seen}. Call exactly one tool."


def _tool(name: str, description: str, arg: str) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "properties": {arg: {"title": arg.capitalize(), "type": "string"}},
                "required": [arg],
                "type": "object",
                "title": f"{name}Arguments",
            },
        },
    }


# Must match what envs/hotpot_mcp.py advertises via list_tools (checked in tests/test_env_parity.py).
LLM_TOOLS: List[Dict[str, Any]] = [
    _tool(
        "search",
        "Search for an article by entity/title.\n\n"
        "Performs a case-insensitive exact title match in CONTEXT_PAGES.\n"
        "If no exact match is found, falls back to a case-insensitive substring match.\n\n"
        "Args:\n    entity: The entity or title name to search for.\n\n"
        "Returns:\n    The title and concatenated text of the article if found, or an error listing available titles.\n",
        "entity",
    ),
    _tool(
        "lookup",
        "Scan the current article for the next sentence containing keyword.\n\n"
        "Scans starting from LOOKUP_INDEX and updates the pointer to the next sentence.\n\n"
        "Args:\n    keyword: The keyword to search for within the current article.\n\n"
        "Returns:\n    The matched sentence with its 1-indexed number, or a message indicating no more occurrences.\n",
        "keyword",
    ),
    _tool(
        "finish",
        "Submit the final answer for the current episode.\n\n"
        "Args:\n    answer: The predicted answer string.\n\n"
        "Returns:\n    Formatted final answer string.\n",
        "answer",
    ),
]


# ---------------------------------------------------------------------------
# Token ledger (Eq. 1)
# ---------------------------------------------------------------------------
class TokenLedger:
    """Counts prompt + completion tokens of every LLM call made for one query.

    C(q) = total over all explored branches, including pruned ones.
    """

    def __init__(self, max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens
        self.calls: List[Dict[str, Any]] = []
        self.pruned_ids: set = set()

    def charge(self, prompt_tokens: int, completion_tokens: int, node_id: Optional[int], kind: str = "policy") -> None:
        self.calls.append({
            "node_id": node_id,
            "kind": kind,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
        })

    def assign(self, call_index: int, node_id: int) -> None:
        self.calls[call_index]["node_id"] = node_id

    def mark_pruned(self, node_id: int) -> None:
        self.pruned_ids.add(node_id)

    @property
    def total(self) -> int:
        return sum(c["prompt_tokens"] + c["completion_tokens"] for c in self.calls)

    @property
    def prompt_total(self) -> int:
        return sum(c["prompt_tokens"] for c in self.calls)

    @property
    def completion_total(self) -> int:
        return sum(c["completion_tokens"] for c in self.calls)

    @property
    def pruned_tokens(self) -> int:
        return sum(
            c["prompt_tokens"] + c["completion_tokens"]
            for c in self.calls
            if c["node_id"] in self.pruned_ids
        )

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def tokens_by_kind(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self.calls:
            out[c["kind"]] = out.get(c["kind"], 0) + c["prompt_tokens"] + c["completion_tokens"]
        return out

    def over_budget(self) -> bool:
        return self.max_tokens is not None and self.total >= self.max_tokens

    def summary(self) -> Dict[str, Any]:
        return {
            "C_q": self.total,
            "prompt_tokens": self.prompt_total,
            "completion_tokens": self.completion_total,
            "pruned_tokens": self.pruned_tokens,
            "n_llm_calls": self.n_calls,
            "tokens_by_kind": self.tokens_by_kind(),
        }


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """One sampled LLM response at a node."""

    tool_name: Optional[str]
    arguments: Dict[str, Any]
    raw_arguments: str
    thought: str
    call_id: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    action_logprob: float = 0.0
    logit_margin: float = 0.0
    token_entropy: float = 0.0

    @property
    def key(self) -> Tuple[str, str]:
        """Identity used to merge duplicate samples."""
        if self.tool_name is None:
            return ("reminder", "")
        return (self.tool_name, json.dumps(self.arguments, sort_keys=True))


    def describe(self) -> str:
        if self.tool_name is None:
            return "a plain-text reply without a tool call"
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.tool_name}({args})"


class Policy(Protocol):
    def sample(self, messages: List[Dict[str, Any]], temperature: float, seed: int) -> Candidate:
        """One LLM call proposing the next action."""
        ...

    def generate_text(self, messages: List[Dict[str, Any]], max_tokens: int, seed: int) -> Tuple[str, int, int]:
        """Plain completion (used by LATS self-evaluation); returns (text, prompt_tokens, completion_tokens)."""
        ...


def normalize_arguments(tool_name: str, arguments: Dict[str, Any], question: str) -> Dict[str, Any]:
    """Same argument-key repair as rollout_runner."""
    if tool_name == "search" and "entity" not in arguments:
        first_val = next(iter(arguments.values())) if arguments else question
        return {"entity": str(first_val)}
    if tool_name == "lookup" and "keyword" not in arguments:
        first_val = next(iter(arguments.values())) if arguments else ""
        return {"keyword": str(first_val)}
    if tool_name == "finish" and "answer" not in arguments:
        first_val = next(iter(arguments.values())) if arguments else ""
        return {"answer": str(first_val)}
    return arguments


class OllamaPolicy:
    """Samples actions from an OpenAI-compatible endpoint (Ollama by default).

    Ollama ignores n>1, so K samples are K separate calls; each is charged separately.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        max_completion_tokens: int = 256,
        question: str = "",
    ):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        for name in ("httpx", "httpx2", "openai"):
            logging.getLogger(name).setLevel(logging.WARNING)
        self.max_completion_tokens = max_completion_tokens
        self.question = question
        self.max_retries = 2
        self.n_failed_calls = 0
        self._failed_prompt_tokens = 0
        self._failed_completion_tokens = 0

    def _create(self, logprobs: bool = True, **kwargs: Any) -> Any:
        """Ollama returns HTTP 500 when generation degenerates ("token repeat limit reached").
        Retry with a new seed at temperature >= 0.7 (greedy would repeat the same loop).
        Failed attempts are still charged to C(q), at max_tokens completion tokens each."""
        from openai import InternalServerError

        for attempt in range(self.max_retries + 1):
            if attempt:
                kwargs = {**kwargs, "seed": kwargs["seed"] + 7919 * attempt,
                          "temperature": max(kwargs["temperature"], 0.7)}
            try:
                if not logprobs:
                    return self.client.chat.completions.create(**kwargs)
                try:
                    return self.client.chat.completions.create(logprobs=True, top_logprobs=2, **kwargs)
                except InternalServerError:
                    raise
                except Exception:
                    return self.client.chat.completions.create(**kwargs)
            except InternalServerError:
                self.n_failed_calls += 1
                self._failed_prompt_tokens += sum(len(json.dumps(m, default=str)) for m in kwargs["messages"]) // 4
                self._failed_completion_tokens += kwargs["max_tokens"]
                if attempt == self.max_retries:
                    raise

    def _take_failed_tokens(self) -> Tuple[int, int]:
        out = (self._failed_prompt_tokens, self._failed_completion_tokens)
        self._failed_prompt_tokens = self._failed_completion_tokens = 0
        return out

    def sample(self, messages: List[Dict[str, Any]], temperature: float, seed: int) -> Candidate:
        response = self._create(
            model=self.model,
            messages=messages,
            tools=LLM_TOOLS,
            temperature=temperature,
            max_tokens=self.max_completion_tokens,
            seed=seed,
        )
        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)
        failed_p, failed_c = self._take_failed_tokens()
        p_tok = (usage.prompt_tokens if usage else 0) + failed_p
        c_tok = (usage.completion_tokens if usage else 0) + failed_c
        action_logprob, logit_margin, token_entropy = extract_logprob_metrics(choice)
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            call = tool_calls[0]
            raw_args = call.function.arguments
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                arguments = {"raw_input": raw_args}
            if not isinstance(arguments, dict):
                arguments = {"raw_input": arguments}
            return Candidate(
                tool_name=call.function.name,
                arguments=normalize_arguments(call.function.name, arguments, self.question),
                raw_arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
                thought=message.content or "",
                call_id=call.id,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                action_logprob=action_logprob,
                logit_margin=logit_margin,
                token_entropy=token_entropy,
            )
        return Candidate(
            tool_name=None,
            arguments={},
            raw_arguments="",
            thought=message.content or "",
            call_id=None,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            action_logprob=action_logprob,
            logit_margin=logit_margin,
            token_entropy=token_entropy,
        )

    def generate_text(self, messages: List[Dict[str, Any]], max_tokens: int, seed: int) -> Tuple[str, int, int]:
        response = self._create(
            logprobs=False,
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            seed=seed,
        )
        usage = getattr(response, "usage", None)
        failed_p, failed_c = self._take_failed_tokens()
        return (
            response.choices[0].message.content or "",
            (usage.prompt_tokens if usage else 0) + failed_p,
            (usage.completion_tokens if usage else 0) + failed_c,
        )


class MockPolicy:
    """Offline stand-in for the LLM: random tool calls over the episode's titles.

    Token counts are approximated as len(text) // 4 so ledger tests are meaningful.
    """

    def __init__(self, titles: List[str], answers: Optional[List[str]] = None, seed: int = 0, finish_rate: float = 0.25):
        self.titles = list(titles)
        self.answers = list(answers or titles)
        self.rng = random.Random(seed)
        self.finish_rate = finish_rate
        self._call_counter = 0

    @staticmethod
    def _prompt_tokens(messages: List[Dict[str, Any]]) -> int:
        return sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or "")) for m in messages) // 4 + 50

    def sample(self, messages: List[Dict[str, Any]], temperature: float, seed: int) -> Candidate:
        depth = sum(1 for m in messages if m["role"] == "tool" or m.get("content") == REMINDER_OBS)
        self._call_counter += 1
        r = self.rng.random()
        if messages[-1].get("content") == FORCE_FINISH_NOTE:
            tool_name, args = "finish", {"answer": self.rng.choice(self.answers)}
        elif r < 0.05:
            tool_name, args = None, {}
        elif r < self.finish_rate + 0.1 * depth:
            tool_name, args = "finish", {"answer": self.rng.choice(self.answers)}
        elif r < 0.75:
            tool_name, args = "search", {"entity": self.rng.choice(self.titles + ["Nonexistent Page"])}
        else:
            tool_name, args = "lookup", {"keyword": self.rng.choice(["the", "born", "film", "zzz"])}
        return Candidate(
            tool_name=tool_name,
            arguments=args,
            raw_arguments=json.dumps(args),
            thought=f"Thinking about step {depth + 1}.",
            call_id=f"call_{self._call_counter}" if tool_name else None,
            prompt_tokens=self._prompt_tokens(messages),
            completion_tokens=self.rng.randint(15, 60),
            action_logprob=-self.rng.uniform(0.5, 15.0),
            logit_margin=self.rng.uniform(0.0, 15.0),
            token_entropy=self.rng.uniform(0.0, 0.5),
        )

    def generate_text(self, messages: List[Dict[str, Any]], max_tokens: int, seed: int) -> Tuple[str, int, int]:
        return str(self.rng.randint(1, 10)), self._prompt_tokens(messages), 2


# ---------------------------------------------------------------------------
# Scorers (all satisfy shared.interfaces.ValueEstimator)
# ---------------------------------------------------------------------------
class RandomScorer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def score_prefix(self, trajectory: Any) -> float:
        return self.rng.random()


class ConstantScorer:
    """No value model: every prefix gets the same score (used for greedy)."""

    def __init__(self, value: float = 0.5):
        self.value = value

    def score_prefix(self, trajectory: Any) -> float:
        return self.value


class HeuristicScorer:
    """Cheap hand-written prior: penalize tool errors, repeated actions and depth."""

    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps

    def score_prefix(self, trajectory: Any) -> float:
        steps = getattr(trajectory, "steps", trajectory)
        if not steps:
            return 0.5
        seen = set()
        errors = repeats = 0
        for s in steps:
            info = s.info if isinstance(s, StepRecord) else s.get("info", {})
            action = s.action if isinstance(s, StepRecord) else s.get("action", "")
            action_input = s.action_input if isinstance(s, StepRecord) else s.get("action_input")
            errors += int(bool(info.get("is_error", False)) or action == "reminder")
            key = (action, json.dumps(action_input, sort_keys=True, default=str))
            repeats += int(key in seen)
            seen.add(key)
        logit = 0.5 - 0.8 * errors - 0.8 * repeats - 1.5 * len(steps) / self.max_steps
        return 1.0 / (1.0 + math.exp(-logit))


class VthetaScorer:
    """Track A's trained value model V_theta via shared.interfaces.score_prefix."""

    def __init__(self, model_path: Path = DEFAULT_VALUE_MODEL, max_steps: int = 8):
        self.model_path = str(model_path)
        self.max_steps = max_steps
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Value model not found: {self.model_path}")

    def score_prefix(self, trajectory: Any) -> float:
        return score_prefix(trajectory, max_steps=self.max_steps, model_path=self.model_path)


def make_scorer(name: str, seed: int = 0, max_steps: int = 8, model_path: Optional[str] = None) -> Any:
    if name == "random":
        return RandomScorer(seed)
    if name == "heuristic":
        return HeuristicScorer(max_steps)
    if name == "vtheta":
        return VthetaScorer(Path(model_path) if model_path else DEFAULT_VALUE_MODEL, max_steps)
    if name == "none":
        return ConstantScorer()
    raise ValueError(f"Unknown scorer: {name}")


# ---------------------------------------------------------------------------
# Search tree
# ---------------------------------------------------------------------------
@dataclass
class Node:
    node_id: int
    steps: List[StepRecord]
    messages: List[Dict[str, Any]]
    env_state: EnvState
    path_tokens: int = 0
    log_pi: float = 0.0
    value: float = 0.5
    parent: Optional[Node] = None
    children: List[Node] = field(default_factory=list)
    done: bool = False
    final_answer: Optional[str] = None
    expanded: bool = False
    pruned: bool = False
    visits: int = 0
    value_sum: float = 0.0

    @property
    def depth(self) -> int:
        return len(self.steps)

    def is_terminal(self, max_steps: int) -> bool:
        return self.done or self.depth >= max_steps

    @property
    def is_finished(self) -> bool:
        """Reached finish() with an answer (a goal node)."""
        return self.done and self.final_answer is not None

    def alive_children(self) -> List[Node]:
        return [c for c in self.children if not c.pruned]


@dataclass
class SearchConfig:
    k: int = 3
    n_sims: int = 3
    delta_prune: float = 0.0
    delta_abort: float = 0.0
    max_steps: int = 8
    max_query_tokens: Optional[int] = 7533
    temperature: float = 0.7
    c_uct: float = 1.0
    seed: int = 0
    sampling: str = "diverse"
    force_finish: bool = True
    beam_width: Optional[int] = None
    gate: str = "none"
    gate_value: float = 0.33
    gate_entropy: float = 0.08


@dataclass
class SearchResult:
    trajectory: Trajectory
    final_answer: Optional[str]
    stop_reason: str
    ledger: TokenLedger
    n_nodes: int
    n_backtracks: int
    value_log: List[Dict[str, Any]]
    forced_finish: bool = False
    n_expansions: int = 0
    n_branched: int = 0

    @property
    def aborted(self) -> bool:
        return self.final_answer is None


class SearchContext:
    """Per-query state shared by all search algorithms."""

    def __init__(
        self,
        question: str,
        env: HotpotEnv,
        policy: Any,
        scorer: Any,
        config: SearchConfig,
        task_id: str = "",
        ground_truth: Optional[str] = None,
    ):
        self.question = question
        self.env = env
        self.policy = policy
        self.scorer = scorer
        self.config = config
        self.task_id = task_id
        self.ground_truth = ground_truth
        self.ledger = TokenLedger(config.max_query_tokens)
        self.nodes: List[Node] = []
        self.value_log: List[Dict[str, Any]] = []
        self._log_by_id: Dict[int, Dict[str, Any]] = {}
        self.n_backtracks = 0
        self.n_expansions = 0
        self.n_branched = 0
        self._call_seed = config.seed * 1_000_003
        self.extra_context: List[str] = []
        self.on_child_scored: Optional[Callable[[Node], None]] = None

    # -- construction -------------------------------------------------------
    def root(self) -> Node:
        node = Node(
            node_id=0,
            steps=[],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {self.question}"},
            ],
            env_state=self.env.snapshot(),
            value=float(self.scorer.score_prefix([])),
        )
        self.nodes = [node]
        return node

    def _next_seed(self, k: int) -> int:
        seed = self._call_seed
        self._call_seed += k
        return seed

    def _messages_for_policy(self, node: Node) -> List[Dict[str, Any]]:
        if not self.extra_context:
            return node.messages
        note = {"role": "system", "content": "\n\n".join(self.extra_context)}
        return node.messages[:1] + [note] + node.messages[1:]

    # -- expansion ----------------------------------------------------------
    def _propose(self, node: Node, k: int) -> List[Candidate]:
        """Draw k candidate actions at node.

        "iid": k independent samples at config.temperature (temperature 0 if k == 1).
        "diverse": sample 0 is greedy; later samples see a request-only note listing
        the actions already proposed, so near-deterministic policies still branch.
        """
        base = self._messages_for_policy(node)
        seed = self._next_seed(k)
        self.n_expansions += 1
        if k == 1:
            return [self.policy.sample(base, 0.0, seed)]
        if self.config.sampling == "iid" and self.config.gate == "none":
            self.n_branched += 1
            return [self.policy.sample(base, self.config.temperature, seed + i) for i in range(k)]
        out = [self.policy.sample(base, 0.0, seed)]
        if not self._should_branch(node, out[0]):
            return out
        self.n_branched += 1
        for i in range(1, k):
            if self.config.sampling == "iid":
                out.append(self.policy.sample(base, self.config.temperature, seed + i))
                continue
            request = base + [{"role": "user", "content": exclusion_note(out)}]
            out.append(self.policy.sample(request, self.config.temperature, seed + i))
        return out

    def _should_branch(self, node: Node, greedy: Candidate) -> bool:
        """Uncertainty gate: spend K-1 extra calls only where the search is unsure.

        The root always branches (its V is a placeholder). "value" branches when
        V(node) < gate_value; "entropy" when the greedy action's mean token entropy
        exceeds gate_entropy; "either" when any of the two holds.
        """
        gate = self.config.gate
        if gate == "none" or node.parent is None:
            return True
        low_value = node.value < self.config.gate_value
        uncertain = greedy.token_entropy > self.config.gate_entropy
        if gate == "value":
            return low_value
        if gate == "entropy":
            return uncertain
        if gate == "either":
            return low_value or uncertain
        raise ValueError(f"Unknown gate: {gate}")

    def expand(self, node: Node, k: Optional[int] = None) -> List[Node]:
        """Sample k actions at node, execute them, score children with the scorer.

        Every sampled call is charged to the ledger; duplicate actions are merged
        into one child (their calls are attributed to that child).
        """
        node.expanded = True
        if node.is_terminal(self.config.max_steps) or self.ledger.over_budget():
            return []
        k = k or self.config.k
        candidates = self._propose(node, k)

        groups: Dict[Tuple[str, str], List[Tuple[Candidate, int]]] = {}
        for cand in candidates:
            self.ledger.charge(cand.prompt_tokens, cand.completion_tokens, None)
            groups.setdefault(cand.key, []).append((cand, len(self.ledger.calls) - 1))

        # Sibling-renormalized action probabilities for PHS's pi(n).
        group_logmass = {
            key: _logsumexp([c.action_logprob for c, _ in members]) for key, members in groups.items()
        }
        total_logmass = _logsumexp(list(group_logmass.values()))

        children: List[Node] = []
        for key, members in groups.items():
            cand = members[0][0]
            child = self._make_child(node, cand)
            child.log_pi = node.log_pi + (group_logmass[key] - total_logmass)
            for _, call_idx in members:
                self.ledger.assign(call_idx, child.node_id)
            child.value = float(self.scorer.score_prefix(child.steps))
            node.children.append(child)
            children.append(child)
            self._log_node(child, n_samples=len(members))
            self._log_by_id[child.node_id]["branched"] = len(candidates) > 1
            if self.on_child_scored is not None:
                self.on_child_scored(child)
        return children

    def _make_child(self, parent: Node, cand: Candidate) -> Node:
        step_idx = parent.depth + 1
        step_tokens = cand.prompt_tokens + cand.completion_tokens
        path_tokens = parent.path_tokens + step_tokens
        messages = list(parent.messages)
        done = False
        final_answer = None

        if cand.tool_name is not None:
            self.env.restore(parent.env_state)
            observation = self.env.step(cand.tool_name, cand.arguments)
            env_state = self.env.snapshot()
            obs_stripped = observation.strip()
            is_error = obs_stripped.startswith("Error:") or obs_stripped.startswith("Could not find")
            action, action_input = cand.tool_name, cand.arguments
            if cand.tool_name == "finish":
                done = True
                final_answer = str(cand.arguments.get("answer", "")).strip()
            else:
                messages.append({
                    "role": "assistant",
                    "content": cand.thought,
                    "tool_calls": [{
                        "id": cand.call_id,
                        "type": "function",
                        "function": {"name": cand.tool_name, "arguments": cand.raw_arguments},
                    }],
                })
                messages.append({"role": "tool", "tool_call_id": cand.call_id, "content": observation})
        else:
            observation = REMINDER_OBS
            env_state = parent.env_state
            is_error = False
            action, action_input = "reminder", ""
            messages.append({"role": "assistant", "content": cand.thought})
            messages.append({"role": "user", "content": REMINDER_OBS})

        step = StepRecord(
            step_idx=step_idx,
            thought=cand.thought,
            action=action,
            action_input=action_input,
            observation=observation,
            reward=0.0,
            done=done or (action == "reminder" and step_idx >= self.config.max_steps),
            info={
                "action_logprob": cand.action_logprob,
                "logit_margin": cand.logit_margin,
                "token_entropy": cand.token_entropy,
                "prompt_tokens": cand.prompt_tokens,
                "completion_tokens": cand.completion_tokens,
                "total_tokens": path_tokens,
                "is_error": is_error,
            },
        )
        child = Node(
            node_id=len(self.nodes),
            steps=parent.steps + [step],
            messages=messages,
            env_state=env_state,
            path_tokens=path_tokens,
            parent=parent,
            done=done,
            final_answer=final_answer,
        )
        self.nodes.append(child)
        return child

    def _log_node(self, node: Node, n_samples: int) -> None:
        entry = {
            "node_id": node.node_id,
            "parent_id": node.parent.node_id if node.parent else None,
            "depth": node.depth,
            "value": node.value,
            "log_pi": node.log_pi,
            "n_samples": n_samples,
            "pruned": False,
            "final_answer": node.final_answer,
            "step": node.steps[-1].to_dict(),
        }
        self.value_log.append(entry)
        self._log_by_id[node.node_id] = entry

    # -- pruning / backtracking --------------------------------------------
    def prune(self, children: List[Node]) -> List[Node]:
        """Drop children with V < delta_prune; their tokens stay in C(q) as pruned tokens."""
        kept = []
        for c in children:
            if c.value < self.config.delta_prune:
                c.pruned = True
                self.ledger.mark_pruned(c.node_id)
                self._log_by_id[c.node_id]["pruned"] = True
            else:
                kept.append(c)
        return kept

    def frontier(self) -> List[Node]:
        """Unexpanded, unpruned, non-terminal nodes (backtrack candidates)."""
        return [
            n for n in self.nodes
            if not n.expanded and not n.pruned and not n.is_terminal(self.config.max_steps)
        ]

    def backtrack_target(self) -> Optional[Node]:
        """Highest-value unexpanded node with V >= delta_abort, or None (abort)."""
        if self.ledger.over_budget():
            return None
        eligible = [n for n in self.frontier() if n.value >= self.config.delta_abort]
        if not eligible:
            return None
        self.n_backtracks += 1
        return max(eligible, key=lambda n: n.value)

    def stop_reason(self) -> str:
        """Why search ended when no backtrack target was available."""
        if self.ledger.over_budget():
            return "budget"
        if self.finished_nodes():
            return "finished"
        if self.frontier():
            return "abort_value"
        return "exhausted"

    def finished_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.is_finished and not n.pruned]

    def force_finish(self) -> Optional[Node]:
        """Budget ran out with no answer: ask the best open branch to answer now.

        One extra greedy call, charged to C(q) as kind="force_finish". A plain-text
        reply is taken as the answer; any other tool call leaves the query unanswered.
        """
        open_nodes = [n for n in self.nodes if n.steps and not n.pruned and not n.done]
        if not open_nodes:
            open_nodes = [self.nodes[0]]
        node = max(open_nodes, key=lambda n: (n.value, n.depth))
        request = self._messages_for_policy(node) + [{"role": "user", "content": FORCE_FINISH_NOTE}]
        cand = self.policy.sample(request, 0.0, self._next_seed(1))
        self.ledger.charge(cand.prompt_tokens, cand.completion_tokens, None, kind="force_finish")
        if cand.tool_name is None:
            answer = answer_from_text(cand.thought)
            if answer is not None:
                cand.tool_name, cand.arguments, cand.raw_arguments = "finish", {"answer": answer}, json.dumps({"answer": answer})
        if cand.tool_name != "finish":
            return None
        child = self._make_child(node, cand)
        self.ledger.assign(len(self.ledger.calls) - 1, child.node_id)
        child.value = node.value
        node.children.append(child)
        self._log_node(child, n_samples=1)
        self._log_by_id[child.node_id]["forced_finish"] = True
        return child

    # -- result -------------------------------------------------------------
    def result(self, chosen: Optional[Node], stop_reason: str) -> SearchResult:
        """Build the SearchResult; ground truth is only used here, after search."""
        forced = False
        if chosen is None:
            finished = self.finished_nodes()
            chosen = max(finished, key=lambda n: n.value) if finished else None
        if chosen is None and stop_reason == "budget" and self.config.force_finish:
            chosen = self.force_finish()
            forced = True
        if chosen is None:
            frontier_or_all = [n for n in self.nodes if n.steps] or self.nodes
            path_node = max(frontier_or_all, key=lambda n: (n.depth, n.value))
            final_answer = None
            if stop_reason == "finished":
                stop_reason = "no_answer"
        else:
            path_node = chosen
            final_answer = chosen.final_answer

        is_correct = bool(compute_exact_match(final_answer, self.ground_truth)) if final_answer is not None else False
        traj = Trajectory(
            task_id=self.task_id,
            question=self.question,
            steps=[StepRecord.from_dict(s.to_dict()) for s in path_node.steps],
            final_answer=final_answer,
            ground_truth=self.ground_truth,
            is_correct=is_correct,
            metadata={"chosen_node_id": path_node.node_id, "chosen_value": path_node.value},
        )
        if traj.steps and final_answer is not None:
            traj.steps[-1].reward = 1.0 if is_correct else 0.0
            traj.total_reward = traj.steps[-1].reward
        return SearchResult(
            trajectory=traj,
            final_answer=final_answer,
            stop_reason=stop_reason,
            ledger=self.ledger,
            n_nodes=len(self.nodes),
            n_backtracks=self.n_backtracks,
            value_log=self.value_log,
            forced_finish=forced,
            n_expansions=self.n_expansions,
            n_branched=self.n_branched,
        )


def answer_from_text(text: str) -> Optional[str]:
    """Answer from a plain-text reply to the forced-finish prompt.

    Qwen sometimes writes tool calls as JSON in the message text: a finish call
    yields its answer, any other tool call yields None. Otherwise the first
    non-empty line is the answer.
    """
    decoder = json.JSONDecoder()
    saw_tool_call = False
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments") or {}
            if obj["name"] == "finish" and isinstance(args, dict) and "answer" in args:
                return str(args["answer"]).strip()
            saw_tool_call = True
    if saw_tool_call or '"name"' in text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else None


def _logsumexp(xs: List[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))
