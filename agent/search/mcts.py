"""Monte Carlo Tree Search (MCTS) guided by value function [Parthiv's Track B]."""

from __future__ import annotations

import math
from typing import List

from agent.search.common import Node, SearchContext, SearchResult


def _leaf_value(ctx: SearchContext, node: Node) -> float:
    """Terminal nodes: V of the finished prefix, or 0 for a max-depth dead end."""
    if node.is_finished:
        return node.value
    if node.is_terminal(ctx.config.max_steps):
        return 0.0
    return node.value


def _select_child(ctx: SearchContext, parent: Node, kids: List[Node]) -> Node:
    log_n = math.log(max(parent.visits, 1))

    def uct(c: Node) -> float:
        if c.visits == 0:
            return float("inf")
        return c.value_sum / c.visits + ctx.config.c_uct * math.sqrt(log_n / c.visits)

    return max(kids, key=uct)


def _simulate(ctx: SearchContext, start: Node) -> None:
    """One selection -> expansion -> evaluation -> backup pass from start.

    Expansion samples K actions; each new child is initialised with its V
    (no LLM rollouts), and the best child value is backed up the path.
    """
    node = start
    path = [node]
    while node.expanded and not node.is_terminal(ctx.config.max_steps):
        kids = node.alive_children()
        if not kids:
            break
        node = _select_child(ctx, node, kids)
        path.append(node)

    if node.is_terminal(ctx.config.max_steps):
        value = _leaf_value(ctx, node)
    elif not node.expanded:
        kids = ctx.prune(ctx.expand(node))
        for c in kids:
            c.visits = 1
            c.value_sum = _leaf_value(ctx, c)
        value = max((c.value_sum for c in kids), default=0.0)
    else:
        value = 0.0

    for n in path:
        n.visits += 1
        n.value_sum += value


def mcts_search(ctx: SearchContext) -> SearchResult:
    """Run N simulations per decision step, then commit to the most-visited child.

    If every child of the committed node is pruned or below delta_abort,
    backtrack to the best unexpanded node with V >= delta_abort, else abort.
    Returns the finished trajectory with the highest V.
    """
    cfg = ctx.config
    current = ctx.root()

    while True:
        for _ in range(cfg.n_sims):
            if ctx.ledger.over_budget():
                break
            _simulate(ctx, current)
        if ctx.ledger.over_budget():
            return ctx.result(None, "budget")

        kids = current.alive_children()
        if not kids or max(c.value for c in kids) < cfg.delta_abort:
            target = ctx.backtrack_target()
            if target is None:
                return ctx.result(None, ctx.stop_reason())
            current = target
            continue

        current = max(kids, key=lambda c: (c.visits, c.value))
        if current.is_finished:
            return ctx.result(None, "finished")
        if current.is_terminal(cfg.max_steps):
            target = ctx.backtrack_target()
            if target is None:
                return ctx.result(None, ctx.stop_reason())
            current = target
