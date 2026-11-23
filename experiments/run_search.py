"""Run value-guided search (or a baseline) over a HotpotQA split and log per-query results.

Example:
    python -m experiments.run_search --method beam --K 3 --scorer vtheta \
        --delta-prune 0.1 --delta-abort 0.2 --split "validation[:20]" --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Set

from tqdm import tqdm

from agent.search import beam_search, lats_search, mcts_search, phs_search
from agent.search.common import MockPolicy, OllamaPolicy, SearchConfig, SearchContext, make_scorer
from envs.hotpot_env import HotpotEnv

B_TOKEN = 2511  # mean C(q) of greedy GPT-4o on validation[:300] (Track A)

METHODS = {
    "greedy": beam_search,
    "beam": beam_search,
    "mcts": mcts_search,
    "phs": phs_search,
    "lats": lats_search,
}


def default_output(args: argparse.Namespace) -> Path:
    split_tag = args.split.replace("[", "_").replace("]", "").replace(":", "-")
    if args.method == "greedy":
        name = f"greedy_{args.policy}_{split_tag}_s{args.seed}"
    else:
        name = (
            f"{args.method}_{args.scorer}_{args.sampling}_K{args.K}_N{args.N}_W{args.beam_width or args.K}"
            f"_gate-{args.gate}{'' if args.gate == 'none' else f'-v{args.gate_value}-e{args.gate_entropy}'}"
            f"_dp{args.delta_prune}_da{args.delta_abort}_{args.policy}_{split_tag}_s{args.seed}"
        )
    return Path("results") / f"{name}.jsonl"


def load_done_ids(path: Path) -> Set[str]:
    done: Set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["task_id"])
    return done


def run_one(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    env = HotpotEnv.from_row(row)
    titles = list(row["context"]["title"])
    if args.policy == "mock":
        policy = MockPolicy(
            titles,
            answers=titles + [row["answer"], "yes", "no"],
            seed=zlib.crc32(str(row["id"]).encode()) + args.seed,
        )
    else:
        policy = OllamaPolicy(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_completion_tokens=args.max_completion_tokens,
            question=row["question"],
        )

    greedy = args.method == "greedy"
    lats = args.method == "lats"
    config = SearchConfig(
        k=1 if greedy else args.K,
        n_sims=args.N,
        delta_prune=0.0 if (greedy or lats) else args.delta_prune,
        delta_abort=0.0 if (greedy or lats) else args.delta_abort,
        max_steps=args.max_steps,
        max_query_tokens=args.max_query_tokens if args.max_query_tokens > 0 else None,
        temperature=args.temperature,
        c_uct=args.c_uct,
        seed=args.seed,
        sampling=args.sampling,
        force_finish=not args.no_force_finish,
        beam_width=args.beam_width,
        gate=args.gate,
        gate_value=args.gate_value,
        gate_entropy=args.gate_entropy,
    )
    scorer = make_scorer("none" if greedy or lats else args.scorer, seed=args.seed, max_steps=args.max_steps)
    ctx = SearchContext(
        question=row["question"],
        env=env,
        policy=policy,
        scorer=scorer,
        config=config,
        task_id=str(row["id"]),
        ground_truth=row["answer"],
    )

    start = time.time()
    result = METHODS[args.method](ctx)
    elapsed = time.time() - start

    summary = result.ledger.summary()
    out = {
        "task_id": str(row["id"]),
        "question": row["question"],
        "ground_truth": row["answer"],
        "level": row.get("level"),
        "type": row.get("type"),
        "method": args.method,
        "scorer": "lats_self_eval" if lats else ("none" if greedy else args.scorer),
        "policy": args.policy,
        "model": args.model,
        "config": asdict(config),
        "final_answer": result.final_answer,
        "is_correct": bool(result.trajectory.is_correct),
        "stop_reason": result.stop_reason,
        "aborted": result.aborted,
        "forced_finish": result.forced_finish,
        "depth": result.trajectory.length,
        "n_nodes": result.n_nodes,
        "n_backtracks": result.n_backtracks,
        "n_expansions": result.n_expansions,
        "n_branched": result.n_branched,
        **summary,
        "b_token": args.b_token,
        "within_budget": summary["C_q"] <= args.b_token,
        "elapsed_s": round(elapsed, 2),
        "trajectory": result.trajectory.to_dict(),
    }
    if not args.no_value_log:
        out["value_log"] = result.value_log
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Value-guided search on HotpotQA.")
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--scorer", choices=["vtheta", "random", "heuristic", "none"], default="vtheta")
    parser.add_argument("--K", type=int, default=3, help="Actions sampled per expansion (beam width for beam).")
    parser.add_argument("--N", type=int, default=3, help="MCTS/LATS simulations per decision step.")
    parser.add_argument("--delta-prune", type=float, default=0.0)
    parser.add_argument("--delta-abort", type=float, default=0.0)
    parser.add_argument("--split", default="validation[:20]")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N questions of the split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-query-tokens", type=int, default=3 * B_TOKEN, help="Per-query hard cap on C(q); 0 disables.")
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--sampling",
        choices=["diverse", "iid"],
        default="diverse",
        help="diverse: greedy first sample, later samples asked for a different action; iid: K independent samples.",
    )
    parser.add_argument("--beam-width", type=int, default=None, help="Beam nodes kept per level (default: K).")
    parser.add_argument(
        "--gate",
        choices=["none", "value", "entropy", "either"],
        default="none",
        help="Branch to K samples only at uncertain non-root nodes; otherwise take the greedy action alone.",
    )
    parser.add_argument("--gate-value", type=float, default=0.33, help="Branch if V(node) is below this.")
    parser.add_argument("--gate-entropy", type=float, default=0.08, help="Branch if greedy token entropy is above this.")
    parser.add_argument(
        "--no-force-finish",
        action="store_true",
        help="On budget exhaustion, abort (Y=0) instead of one forced finish() call.",
    )
    parser.add_argument("--c-uct", type=float, default=1.0)
    parser.add_argument("--b-token", type=int, default=B_TOKEN)
    parser.add_argument("--policy", choices=["ollama", "mock"], default="ollama")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default=os.getenv("OLLAMA_API_KEY", "ollama"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-value-log", action="store_true", help="Skip per-node logs (smaller files).")
    args = parser.parse_args()

    import datasets

    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    output = Path(args.output) if args.output else default_output(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(output)
    if done:
        print(f"Resuming: {len(done)} queries already in {output}")

    dataset = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    n = correct = tokens = 0
    pbar = tqdm(dataset, desc=f"{args.method}")
    for row in pbar:
        if str(row["id"]) in done:
            continue
        rec = run_one(row, args)
        with open(output, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        correct += int(rec["is_correct"])
        tokens += rec["C_q"]
        pbar.set_postfix({"EM": f"{100 * correct / n:.1f}%", "C(q)": f"{tokens / n:.0f}"})

    print(f"Wrote {n} new results to {output}")


if __name__ == "__main__":
    main()
