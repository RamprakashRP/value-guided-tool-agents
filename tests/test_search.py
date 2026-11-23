"""Offline tests for search + Eq. 1 accounting using MockPolicy (no LLM needed)."""

from __future__ import annotations

import pytest

from agent.search import beam_search, lats_search, mcts_search, phs_search
from agent.search.common import (
    FORCE_FINISH_NOTE,
    Candidate,
    ConstantScorer,
    answer_from_text,
    exclusion_note,
    HeuristicScorer,
    MockPolicy,
    RandomScorer,
    SearchConfig,
    SearchContext,
    VthetaScorer,
)
from agent.search.phs import phs_priority
from envs.hotpot_env import HotpotEnv

CONTEXT = {
    "Scott Derrickson": ["Scott Derrickson (born July 16, 1966) is an American director.", "He lives in LA."],
    "Ed Wood": ["Edward Davis Wood Jr. was an American filmmaker.", "Wood was born in New York."],
    "Doctor Strange (2016 film)": ["Doctor Strange is a 2016 film directed by Scott Derrickson."],
}
TITLES = list(CONTEXT)
METHODS = {"beam": beam_search, "mcts": mcts_search, "phs": phs_search}


class RecordingPolicy(MockPolicy):
    """MockPolicy that remembers every token it reported, to check the ledger independently."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reported = []
        self.requests = []

    def sample(self, messages, temperature, seed):
        cand = super().sample(messages, temperature, seed)
        self.reported.append(cand.prompt_tokens + cand.completion_tokens)
        self.requests.append((list(messages), temperature))
        return cand

    def generate_text(self, messages, max_tokens, seed):
        text, p, c = super().generate_text(messages, max_tokens, seed)
        self.reported.append(p + c)
        return text, p, c


class DepthScorer:
    def __init__(self, by_depth, default=0.1):
        self.by_depth = by_depth
        self.default = default

    def score_prefix(self, steps):
        return self.by_depth.get(len(steps), self.default) if steps else 0.5


def make_ctx(policy, scorer, **cfg):
    config = SearchConfig(**{"k": 3, "n_sims": 3, "max_query_tokens": None, **cfg})
    return SearchContext("Were Scott Derrickson and Ed Wood of the same nationality?", HotpotEnv(CONTEXT), policy,
                         scorer, config, task_id="q1", ground_truth="yes")


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("scorer_name", ["random", "heuristic", "vtheta"])
@pytest.mark.parametrize("seed", range(5))
def test_ledger_counts_every_call_including_pruned(method, scorer_name, seed):
    scorer = {"random": RandomScorer(seed), "heuristic": HeuristicScorer(), "vtheta": VthetaScorer()}[scorer_name]
    policy = RecordingPolicy(TITLES, answers=TITLES + ["yes", "no"], seed=seed)
    ctx = make_ctx(policy, scorer, delta_prune=0.3, delta_abort=0.4, seed=seed)
    result = METHODS[method](ctx)

    assert result.ledger.total == sum(policy.reported)
    assert result.ledger.n_calls == len(policy.reported)
    assert 0 <= result.ledger.pruned_tokens <= result.ledger.total
    assert all(c["node_id"] is not None for c in result.ledger.calls)
    if result.final_answer is not None:
        assert result.trajectory.steps[-1].action == "finish"
        assert result.trajectory.is_correct == (result.final_answer.lower() == "yes")


def test_pruned_tokens_counted():
    policy = RecordingPolicy(TITLES, seed=1, finish_rate=-1.0)
    ctx = make_ctx(policy, DepthScorer({1: 0.05}), delta_prune=0.1)
    result = beam_search(ctx)
    assert result.ledger.pruned_tokens == result.ledger.total > 0
    assert result.final_answer is None


@pytest.mark.parametrize("method", list(METHODS))
def test_budget_cap_respected(method):
    policy = RecordingPolicy(TITLES, seed=3, finish_rate=-1.0)
    cap = 2000
    ctx = make_ctx(policy, RandomScorer(0), max_query_tokens=cap, k=3)
    result = METHODS[method](ctx)
    max_expansion = 4 * max(policy.reported)
    assert cap <= result.ledger.total < cap + max_expansion
    assert result.stop_reason == "budget"
    assert result.ledger.total == sum(policy.reported)


@pytest.mark.parametrize("method", list(METHODS))
def test_forced_finish_on_budget(method):
    policy = RecordingPolicy(TITLES, answers=["yes"], seed=3, finish_rate=-1.0)
    ctx = make_ctx(policy, RandomScorer(0), max_query_tokens=1500)
    result = METHODS[method](ctx)
    assert result.forced_finish
    assert result.final_answer == "yes" and result.trajectory.is_correct
    assert result.trajectory.steps[-1].action == "finish"
    assert result.ledger.tokens_by_kind()["force_finish"] > 0
    assert result.ledger.calls[-1]["kind"] == "force_finish"
    assert policy.requests[-1][0][-1]["content"] == FORCE_FINISH_NOTE
    assert result.ledger.total == sum(policy.reported)


def test_no_force_finish_aborts():
    policy = RecordingPolicy(TITLES, answers=["yes"], seed=3, finish_rate=-1.0)
    ctx = make_ctx(policy, RandomScorer(0), max_query_tokens=1500, force_finish=False)
    result = beam_search(ctx)
    assert not result.forced_finish and result.final_answer is None
    assert "force_finish" not in result.ledger.tokens_by_kind()


def test_diverse_sampling_note_is_request_only():
    policy = RecordingPolicy(TITLES, seed=0, finish_rate=-1.0)
    ctx = make_ctx(policy, RandomScorer(0), k=3, temperature=0.7, sampling="diverse")
    children = ctx.expand(ctx.root())
    msgs = [m for m, _ in policy.requests]
    assert [t for _, t in policy.requests] == [0.0, 0.7, 0.7]
    assert all(m["content"] != "" for m in msgs[0])
    assert not any(str(m.get("content", "")).startswith("Propose a different") for m in msgs[0])
    assert msgs[1][-1]["content"].startswith("Propose a different next action than: ")
    assert msgs[2][-1]["content"].count("(") >= 1
    for child in children:
        assert not any(str(m.get("content", "")).startswith("Propose a different") for m in child.messages)


def test_iid_sampling_has_no_note():
    policy = RecordingPolicy(TITLES, seed=0, finish_rate=-1.0)
    ctx = make_ctx(policy, RandomScorer(0), k=3, temperature=0.7, sampling="iid")
    ctx.expand(ctx.root())
    assert [t for _, t in policy.requests] == [0.7, 0.7, 0.7]
    assert all(m[-1]["content"].startswith("Question:") for m, _ in policy.requests)


class EntropyPolicy(RecordingPolicy):
    """RecordingPolicy whose greedy samples carry a fixed token entropy."""

    def __init__(self, *args, entropy=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy = entropy

    def sample(self, messages, temperature, seed):
        cand = super().sample(messages, temperature, seed)
        cand.token_entropy = self.entropy
        return cand


def _calls_for_child_expansion(gate, value, entropy, k=3):
    policy = EntropyPolicy(TITLES, seed=0, finish_rate=-1.0, entropy=entropy)
    ctx = make_ctx(policy, ConstantScorer(value), k=k, gate=gate, gate_value=0.33, gate_entropy=0.08)
    root_children = ctx.expand(ctx.root())
    root_calls = ctx.ledger.n_calls
    ctx.expand(root_children[0])
    return root_calls, ctx.ledger.n_calls - root_calls, ctx


@pytest.mark.parametrize("gate,value,entropy,expected", [
    ("either", 0.9, 0.0, 1),
    ("either", 0.1, 0.0, 3),
    ("either", 0.9, 0.5, 3),
    ("value", 0.9, 0.5, 1),
    ("value", 0.1, 0.0, 3),
    ("entropy", 0.1, 0.0, 1),
    ("entropy", 0.9, 0.5, 3),
    ("none", 0.9, 0.0, 3),
])
def test_gate_controls_branching(gate, value, entropy, expected):
    root_calls, child_calls, ctx = _calls_for_child_expansion(gate, value, entropy)
    assert root_calls == 3
    assert child_calls == expected
    assert ctx.n_expansions == 2
    assert ctx.n_branched == 1 + (expected > 1)
    assert ctx.ledger.total == sum(ctx.policy.reported)
    logged = [e for e in ctx.value_log if e["parent_id"] != 0]
    assert all(e["branched"] == (expected > 1) for e in logged)


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("seed", range(3))
def test_gated_search_ledger_and_cost(method, seed):
    costs = {}
    for gate in ("none", "either"):
        policy = RecordingPolicy(TITLES, answers=TITLES + ["yes"], seed=seed)
        ctx = make_ctx(policy, VthetaScorer(), gate=gate, seed=seed, max_query_tokens=6000)
        result = METHODS[method](ctx)
        assert result.ledger.total == sum(policy.reported)
        assert result.n_branched <= result.n_expansions
        costs[gate] = result
    assert costs["either"].n_branched <= costs["either"].n_expansions


@pytest.mark.parametrize("width", [1, 2])
def test_beam_width_limits_expansions_per_level(width):
    policy = RecordingPolicy(TITLES, seed=1, finish_rate=-1.0)
    ctx = make_ctx(policy, RandomScorer(0), k=3, beam_width=width, max_query_tokens=4000)
    beam_search(ctx)
    expanded_by_depth = {}
    for n in ctx.nodes:
        if n.expanded and not n.is_terminal(ctx.config.max_steps):
            expanded_by_depth[n.depth] = expanded_by_depth.get(n.depth, 0) + 1
    assert ctx.n_backtracks == 0
    assert expanded_by_depth[0] == 1
    assert all(count <= width for count in expanded_by_depth.values())
    assert max(expanded_by_depth) >= 2


@pytest.mark.parametrize("text,expected", [
    ("Chief of Protocol", "Chief of Protocol"),
    ("\n  yes \nbecause both are American", "yes"),
    ('{"name": "finish", "arguments": {"answer": "Chief of Protocol"}}', "Chief of Protocol"),
    ('I will answer.\n{"name": "finish", "arguments": {"answer": "no"}}', "no"),
    ('brtc\n{"name": "search", "arguments": {"entity": "Kiss and Tell (1992 film)"}}', None),
    ("", None),
])
def test_answer_from_text(text, expected):
    assert answer_from_text(text) == expected


def test_exclusion_note_lists_unique_actions():
    a = Candidate("search", {"entity": "Ed Wood"}, "", "", "c1", 1, 1)
    b = Candidate(None, {}, "", "", None, 1, 1)
    note = exclusion_note([a, a, b])
    assert note == (
        "Propose a different next action than: search(entity='Ed Wood'), "
        "a plain-text reply without a tool call. Call exactly one tool."
    )


def test_abort_when_all_children_below_delta_abort():
    policy = RecordingPolicy(TITLES, seed=0, finish_rate=-1.0)
    ctx = make_ctx(policy, ConstantScorer(0.3), delta_abort=0.5, k=3)
    result = beam_search(ctx)
    assert result.stop_reason == "abort_value"
    assert result.final_answer is None
    assert result.ledger.n_calls == 3


def test_beam_backtracks_to_best_unexpanded_node():
    policy = RecordingPolicy(TITLES, seed=0, finish_rate=-1.0)
    ctx = make_ctx(policy, DepthScorer({1: 0.9, 2: 0.6}, default=0.1), delta_abort=0.5, k=3)
    result = beam_search(ctx)
    assert result.n_backtracks >= 1
    assert result.stop_reason in ("abort_value", "exhausted")


def test_mcts_backtracks_and_terminates():
    policy = RecordingPolicy(TITLES, seed=0, finish_rate=-1.0)
    ctx = make_ctx(policy, DepthScorer({1: 0.9, 2: 0.6}, default=0.1), delta_abort=0.5, k=2, n_sims=2)
    result = mcts_search(ctx)
    assert result.n_backtracks >= 1
    assert result.ledger.total == sum(policy.reported)


def test_greedy_one_call_per_step():
    for seed in range(10):
        policy = RecordingPolicy(TITLES, answers=["yes"], seed=seed)
        ctx = make_ctx(policy, ConstantScorer(), k=1)
        result = beam_search(ctx)
        assert result.ledger.n_calls == result.trajectory.length
        assert result.n_nodes == result.trajectory.length + 1
        tokens = [s.info["total_tokens"] for s in result.trajectory.steps]
        assert tokens[-1] == result.ledger.total
        assert tokens == sorted(tokens)


def test_path_tokens_are_per_branch_not_global():
    policy = RecordingPolicy(TITLES, seed=2, finish_rate=-1.0)
    ctx = make_ctx(policy, RandomScorer(0), k=3, max_steps=3)
    beam_search(ctx)
    for node in ctx.nodes[1:]:
        own = sum(s.info["prompt_tokens"] + s.info["completion_tokens"] for s in node.steps)
        assert node.steps[-1].info["total_tokens"] == own == node.path_tokens
    assert max(n.path_tokens for n in ctx.nodes) < ctx.ledger.total


def test_env_state_is_forked_per_branch():
    policy = RecordingPolicy(TITLES, seed=4)
    ctx = make_ctx(policy, RandomScorer(1), k=3)
    phs_search(ctx)
    for node in ctx.nodes[1:]:
        env = HotpotEnv(CONTEXT)
        for s in node.steps:
            if s.action in ("search", "lookup", "finish"):
                obs = env.step(s.action, s.action_input)
                assert obs == s.observation


def test_phs_priority_prefers_cheap_likely_valuable():
    policy = RecordingPolicy(TITLES, seed=0)
    ctx = make_ctx(policy, RandomScorer(0))
    root = ctx.root()
    kids = ctx.expand(root)
    a = kids[0]
    base = phs_priority(a)
    a.value *= 2
    assert phs_priority(a) < base
    a.path_tokens *= 2
    a.log_pi -= 1.0
    assert phs_priority(a) > base - 0.7


def test_lats_charges_self_eval_and_reflection():
    policy = RecordingPolicy(TITLES, answers=TITLES, seed=5)
    ctx = make_ctx(policy, ConstantScorer(), k=3, n_sims=2)
    result = lats_search(ctx)
    kinds = result.ledger.tokens_by_kind()
    assert kinds.get("self_eval", 0) > 0
    assert result.ledger.total == sum(policy.reported)
