"""Aggregate run_search.py outputs (and rollout_runner baselines) into a results table.

Exact match and mean C(q) are first averaged over seeds per question, then
95% CIs come from a bootstrap over questions.

Example:
    python -m experiments.summarize "results/*.jsonl" \
        --baseline "GPT-4o greedy=data/gpt4o_val300.jsonl" --csv results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

B_TOKEN = 2511


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def group_key(rec: Dict[str, Any]) -> Tuple:
    c = rec["config"]
    if rec["method"] == "greedy":
        return (rec["method"], rec.get("model"), rec["scorer"], "-", 1, "-", "-", "-", "-", "-")
    n = c["n_sims"] if rec["method"] in ("mcts", "lats") else "-"
    sampling = c.get("sampling", "iid")
    width = (c.get("beam_width") or c["k"]) if rec["method"] == "beam" else "-"
    gate = c.get("gate", "none")
    if gate != "none":
        gate = f"{gate}(v{c['gate_value']},e{c['gate_entropy']})"
    return (rec["method"], rec.get("model"), rec["scorer"], sampling, c["k"], n, width, gate,
            c["delta_prune"], c["delta_abort"])


def load_search_rows(patterns: List[str]) -> Dict[Tuple, List[Dict[str, Any]]]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        rec["_seed"] = rec["config"]["seed"]
                        key = group_key(rec)
                        if (key, rec["_seed"], rec["task_id"]) in seen:
                            continue
                        seen.add((key, rec["_seed"], rec["task_id"]))
                        groups[key].append(rec)
    return groups


def load_baseline(name: str, path: str) -> List[Dict[str, Any]]:
    """Read rollout_runner.py output (Trajectory JSONL) into run_search-like rows."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            rows.append({
                "task_id": t["task_id"],
                "is_correct": bool(t.get("is_correct")),
                "C_q": t.get("metadata", {}).get("total_tokens", 0),
                "pruned_tokens": 0,
                "aborted": t.get("final_answer") is None,
                "n_llm_calls": len(t.get("steps", [])),
                "_seed": 0,
            })
    return rows


def summarize_rows(rows: List[Dict[str, Any]], b_token: int) -> Dict[str, Any]:
    by_q: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_q[r["task_id"]].append(r)
    em = np.array([np.mean([r["is_correct"] for r in rs]) for rs in by_q.values()], dtype=float)
    cost = np.array([np.mean([r["C_q"] for r in rs]) for rs in by_q.values()], dtype=float)
    all_cost = np.array([r["C_q"] for r in rows], dtype=float)
    pruned = np.array([r.get("pruned_tokens", 0) for r in rows], dtype=float)
    em_lo, em_hi = bootstrap_ci(em)
    c_lo, c_hi = bootstrap_ci(cost)
    return {
        "n_questions": len(by_q),
        "n_seeds": len({r["_seed"] for r in rows}),
        "em": float(em.mean()),
        "em_ci": (em_lo, em_hi),
        "mean_cq": float(cost.mean()),
        "cq_ci": (c_lo, c_hi),
        "mean_within_budget": bool(cost.mean() <= b_token),
        "frac_q_within_budget": float(np.mean(all_cost <= b_token)),
        "pruned_frac": float(pruned.sum() / max(all_cost.sum(), 1)),
        "abort_rate": float(np.mean([r["aborted"] for r in rows])),
        "forced_rate": float(np.mean([r.get("forced_finish", False) for r in rows])),
        "branch_rate": (
            sum(r.get("n_branched", 0) for r in rows) / max(sum(r.get("n_expansions", 0) for r in rows), 1)
            if any("n_expansions" in r for r in rows) else float("nan")
        ),
        "mean_depth": float(np.mean([r.get("depth", 0) for r in rows])),
        "mean_calls": float(np.mean([r["n_llm_calls"] for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize search results.")
    parser.add_argument("patterns", nargs="+", help="Glob(s) of run_search.py JSONL files.")
    parser.add_argument("--baseline", action="append", default=[], help="NAME=path to rollout_runner JSONL.")
    parser.add_argument("--b-token", type=int, default=B_TOKEN)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    table = []
    for spec in args.baseline:
        name, path = spec.split("=", 1)
        table.append(((name, "-", "-", "-", 1, "-", "-", "-", "-", "-"), summarize_rows(load_baseline(name, path), args.b_token)))
    for key, rows in sorted(load_search_rows(args.patterns).items(), key=lambda kv: str(kv[0])):
        table.append((key, summarize_rows(rows, args.b_token)))

    header = [
        "method", "model", "scorer", "sampling", "K", "N", "W", "gate", "d_prune", "d_abort", "n_q", "seeds",
        "EM", "EM 95% CI", "mean C(q)", "C(q) 95% CI", "mean<=B", "q<=B", "pruned tok", "abort", "forced",
        "branch", "depth", "calls/q",
    ]
    print(f"B_token = {args.b_token}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    csv_rows = []
    for key, s in table:
        cells = [str(x) for x in key] + [
            str(s["n_questions"]),
            str(s["n_seeds"]),
            f"{100 * s['em']:.1f}",
            f"[{100 * s['em_ci'][0]:.1f}, {100 * s['em_ci'][1]:.1f}]",
            f"{s['mean_cq']:.0f}",
            f"[{s['cq_ci'][0]:.0f}, {s['cq_ci'][1]:.0f}]",
            "yes" if s["mean_within_budget"] else "no",
            f"{100 * s['frac_q_within_budget']:.0f}%",
            f"{100 * s['pruned_frac']:.0f}%",
            f"{100 * s['abort_rate']:.0f}%",
            f"{100 * s['forced_rate']:.0f}%",
            "-" if s["branch_rate"] != s["branch_rate"] else f"{100 * s['branch_rate']:.0f}%",
            f"{s['mean_depth']:.1f}",
            f"{s['mean_calls']:.1f}",
        ]
        print("| " + " | ".join(cells) + " |")
        csv_rows.append(cells)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(csv_rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
