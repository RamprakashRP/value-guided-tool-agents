"""Value model training, evaluation, and calibration for Value-Guided Tool Agents.

Trains a StandardScaler + LogisticRegression (or MLP) pipeline on prefix
trajectories phi(tau_{1:t}) predicting final episode success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from value_model.features import FEATURE_NAMES, extract_prefix_features


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across n_bins equal-width bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == 0:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs > bin_lower) & (probs <= bin_upper)

        count = int(np.sum(in_bin))
        if count > 0:
            prop_in_bin = count / n
            accuracy_in_bin = float(np.mean(labels[in_bin]))
            avg_confidence_in_bin = float(np.mean(probs[in_bin]))
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)

    return float(ece)


def load_and_expand_dataset(
    file_path: str, max_steps: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load JSONL trajectories and expand into prefix feature examples.

    Args:
        file_path: Path to rollouts JSONL file.
        max_steps: Maximum interaction steps for depth normalization.

    Returns:
        X: Feature matrix of shape (num_prefixes, 16).
        y: Binary labels vector (1 for correct, 0 for incorrect).
        step_depths: Array of prefix depths (t = 1..T) for step-wise evaluation.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    step_depths: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            traj = json.loads(line_str)
            steps = traj.get("steps", [])
            if not steps:
                continue

            is_correct = int(bool(traj.get("is_correct", False)))

            # Expand trajectory into prefix examples tau_{1:t}
            for t in range(1, len(steps) + 1):
                prefix_steps = steps[:t]
                feat = extract_prefix_features(prefix_steps, max_steps=max_steps)
                X_list.append(feat)
                y_list.append(is_correct)
                step_depths.append(t)

    if not X_list:
        raise ValueError(f"No valid prefix examples found in {file_path}")

    return (
        np.array(X_list, dtype=np.float64),
        np.array(y_list, dtype=np.int64),
        np.array(step_depths, dtype=np.int64),
    )


def train_and_evaluate(args: argparse.Namespace) -> Pipeline:
    """Train value model pipeline and report validation metrics."""
    print(f"=== Loading Training Data from: {args.train_file} ===")
    X_train, y_train, train_depths = load_and_expand_dataset(args.train_file, max_steps=args.max_steps)
    print(f"Train prefixes: {len(y_train)} (Positive: {np.sum(y_train)}, Negative: {len(y_train) - np.sum(y_train)})")

    print(f"\n=== Loading Validation Data from: {args.val_file} ===")
    X_val, y_val, val_depths = load_and_expand_dataset(args.val_file, max_steps=args.max_steps)
    print(f"Val prefixes: {len(y_val)} (Positive: {np.sum(y_val)}, Negative: {len(y_val) - np.sum(y_val)})")

    # Build model pipeline
    if args.mlp:
        classifier = MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )
        model_name = "MLPClassifier (32 hidden)"
    else:
        classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        )
        model_name = "LogisticRegression (L2, C=1.0)"

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])

    print(f"\n=== Training Pipeline: {model_name} ===")
    pipeline.fit(X_train, y_train)

    # Validation predictions
    val_probs = pipeline.predict_proba(X_val)[:, 1]

    # Metrics computation
    brier = float(brier_score_loss(y_val, val_probs))
    ece = compute_ece(val_probs, y_val, n_bins=10)

    try:
        overall_auc = float(roc_auc_score(y_val, val_probs))
        auc_str = f"{overall_auc:.4f}"
    except ValueError:
        overall_auc = float("nan")
        auc_str = "N/A (single class in validation set)"

    print("\n" + "=" * 50)
    print("           VALIDATION EVALUATION METRICS           ")
    print("=" * 50)
    print(f"Overall Brier Score (lower is better): {brier:.4f}")
    print(f"Expected Calibration Error (ECE):       {ece:.4f}")
    print(f"Overall AUC-ROC:                        {auc_str}")
    print("-" * 50)

    # Step-by-step AUC-ROC
    print("Step-by-step AUC-ROC:")
    step_buckets = [("Step 1", val_depths == 1),
                    ("Step 2", val_depths == 2),
                    ("Step 3", val_depths == 3),
                    ("Step 4+", val_depths >= 4)]

    for step_label, mask in step_buckets:
        n_samples = int(np.sum(mask))
        if n_samples == 0:
            print(f"  {step_label:8s}: No samples")
            continue

        y_sub = y_val[mask]
        probs_sub = val_probs[mask]
        if len(np.unique(y_sub)) > 1:
            step_auc = float(roc_auc_score(y_sub, probs_sub))
            print(f"  {step_label:8s}: AUC-ROC = {step_auc:.4f}  (N={n_samples})")
        else:
            print(f"  {step_label:8s}: AUC-ROC = N/A (single class, N={n_samples})")

    # Top 5 most influential features
    if not args.mlp and hasattr(classifier, "coef_"):
        coefs = classifier.coef_[0]
        feature_ranking = sorted(
            zip(FEATURE_NAMES, coefs), key=lambda item: abs(item[1]), reverse=True
        )
        print("-" * 50)
        print("Top 5 Most Influential Features (by |weight|):")
        for rank, (name, weight) in enumerate(feature_ranking[:5], 1):
            direction = "(+ encourages success)" if weight > 0 else "(- indicates failure)"
            print(f"  {rank}. {name:30s} weight: {weight:+.4f} {direction}")

    print("=" * 50)

    # Save fitted pipeline to disk
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.feature_names_ = FEATURE_NAMES
    pipeline.max_steps_ = args.max_steps
    joblib.dump(pipeline, out_path)
    print(f"\nSaved trained value model pipeline to: {out_path}")

    return pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate value model on trajectory prefixes."
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train_rollouts.jsonl",
        help="Path to training rollouts JSONL (default: 'data/train_rollouts.jsonl')",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/val_rollouts.jsonl",
        help="Path to validation rollouts JSONL (default: 'data/val_rollouts.jsonl')",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="data/value_model.joblib",
        help="Path to save trained joblib model (default: 'data/value_model.joblib')",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum steps per episode for normalization (default: 8)",
    )
    parser.add_argument(
        "--mlp",
        action="store_true",
        default=False,
        help="Train an MLPClassifier instead of LogisticRegression",
    )
    args = parser.parse_args()

    train_and_evaluate(args)


if __name__ == "__main__":
    main()
