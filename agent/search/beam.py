"""Beam Search guided by value function [Parthiv's Track B]."""

from __future__ import annotations

from agent.search.common import SearchContext, SearchResult


def beam_search(ctx: SearchContext) -> SearchResult:
    """Beam of width K; each beam node samples K actions.

    Children with V < delta_prune are dropped. If no surviving child reaches
    delta_abort, jump to the best unexpanded node anywhere in the tree with
    V >= delta_abort; if none exists (or the budget is spent), abort.
    Stops once the best finished trajectory scores at least as high as every
    open branch, and returns the finished trajectory with the highest V.
    """
    cfg = ctx.config
    beam = [ctx.root()]

    while beam:
        children = []
        for node in beam:
            if ctx.ledger.over_budget():
                break
            children.extend(ctx.expand(node))
        if ctx.ledger.over_budget():
            return ctx.result(None, "budget")

        kept = ctx.prune(children)
        alive = sorted(
            (c for c in kept if not c.is_terminal(cfg.max_steps)),
            key=lambda c: c.value,
            reverse=True,
        )

        finished = ctx.finished_nodes()
        if finished and max(f.value for f in finished) >= max((n.value for n in alive), default=-1.0):
            return ctx.result(None, "finished")

        if not alive or alive[0].value < cfg.delta_abort:
            target = ctx.backtrack_target()
            if target is None:
                return ctx.result(None, ctx.stop_reason())
            beam = [target]
            continue

        beam = alive[: cfg.k]

    return ctx.result(None, "finished")
