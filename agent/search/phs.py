"""Policy-guided Heuristic Search (PHS) [Parthiv's Track B]."""

from __future__ import annotations

import heapq
import math

from agent.search.common import Node, SearchContext, SearchResult

_MIN_VALUE = 1e-6


def phs_priority(node: Node) -> float:
    """log phi(n) = log g(n) - log pi(n) - log V(n)  (Orseau & Lelis 2021, with h = 1/V).

    g(n) is the tokens consumed along the path, pi(n) the product of
    sibling-renormalized action probabilities, V(n) the value estimate.
    """
    g = max(node.path_tokens, 1)
    return math.log(g) - node.log_pi - math.log(max(node.value, _MIN_VALUE))


def phs_search(ctx: SearchContext) -> SearchResult:
    """Best-first search ordered by phi(n); returns when a finish() node is popped.

    Children with V < delta_prune are never queued. Aborts when the queue
    empties or the token budget is exhausted.
    """
    root = ctx.root()
    counter = 0
    heap = [(0.0, counter, root)]

    while heap:
        if ctx.ledger.over_budget():
            return ctx.result(None, "budget")
        _, _, node = heapq.heappop(heap)
        if node.is_finished:
            return ctx.result(node, "finished")
        if node.is_terminal(ctx.config.max_steps):
            continue
        for child in ctx.prune(ctx.expand(node)):
            if child.is_terminal(ctx.config.max_steps) and not child.is_finished:
                continue
            counter += 1
            heapq.heappush(heap, (phs_priority(child), counter, child))

    return ctx.result(None, ctx.stop_reason())
