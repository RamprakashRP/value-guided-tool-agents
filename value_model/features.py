"""Prefix trajectory feature extraction for Value-Guided Tool Agents.

Extracts deterministic 1D float64 feature vectors phi(tau_{1:t})
from step sequences (supporting both dict records and StepRecord dataclasses).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set

import numpy as np

# Exact feature order for interpretability and model consistency
FEATURE_NAMES: List[str] = [
    # Step uncertainty (current step)
    "action_logprob",
    "token_entropy",
    "logit_margin",
    # Prefix uncertainty aggregates
    "mean_token_entropy",
    "min_action_logprob",
    "cumulative_action_logprob",
    # Error signals
    "current_is_error",
    "total_errors",
    "error_fraction",
    # Loop signals
    "repeat_action_count",
    "max_obs_jaccard_overlap",
    # Action type one-hot
    "is_search",
    "is_lookup",
    "is_finish",
    # Progress
    "normalized_depth",
    "normalized_total_tokens",
]


def _get_field(step: Any, field_name: str, default: Any = None) -> Any:
    """Access a field from either a dict or a dataclass instance."""
    if isinstance(step, dict):
        return step.get(field_name, default)
    return getattr(step, field_name, default)


def _get_info_dict(step: Any) -> Dict[str, Any]:
    """Retrieve the info dict from a step object or dict."""
    info = _get_field(step, "info", {})
    return info if isinstance(info, dict) else {}


def _serialize_action_input(action_input: Any) -> str:
    """Deterministically serialize action_input for loop detection."""
    try:
        return json.dumps(action_input, sort_keys=True)
    except Exception:
        return str(action_input)


def _extract_words(text: Optional[str]) -> Set[str]:
    """Extract lowercase word tokens from an observation string."""
    if not text:
        return set()
    return set(re.findall(r"\w+", str(text).lower()))


def _jaccard_overlap(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two word sets."""
    if not set1 or not set2:
        return 0.0
    union = set1 | set2
    if not union:
        return 0.0
    return float(len(set1 & set2) / len(union))


def extract_prefix_features(steps: List[Any], max_steps: int = 8) -> np.ndarray:
    """Extract a deterministic 1D float64 feature vector phi(tau_{1:t}) for a prefix of steps.

    Args:
        steps: Non-empty list of steps in the prefix (either dicts or StepRecord objects).
        max_steps: Maximum interaction steps for depth normalization (default: 8).

    Returns:
        1D numpy array of shape (16,) with dtype float64.
    """
    if not steps:
        raise ValueError("steps list cannot be empty when extracting prefix features.")

    current_step = steps[-1]
    curr_info = _get_info_dict(current_step)

    # 1. Step uncertainty (current step)
    step_action_logprob = float(curr_info.get("action_logprob", 0.0) or 0.0)
    step_token_entropy = float(curr_info.get("token_entropy", 0.0) or 0.0)
    step_logit_margin = float(curr_info.get("logit_margin", 0.0) or 0.0)

    # 2. Prefix uncertainty aggregates
    prefix_entropies: List[float] = []
    prefix_logprobs: List[float] = []
    prefix_errors: List[float] = []
    accumulated_tokens: float = 0.0

    for s in steps:
        s_info = _get_info_dict(s)
        prefix_entropies.append(float(s_info.get("token_entropy", 0.0) or 0.0))
        prefix_logprobs.append(float(s_info.get("action_logprob", 0.0) or 0.0))
        is_err = bool(s_info.get("is_error", False))
        prefix_errors.append(1.0 if is_err else 0.0)

        # Token accumulation fallback if total_tokens isn't directly on current step
        p_tok = float(s_info.get("prompt_tokens", 0) or 0)
        c_tok = float(s_info.get("completion_tokens", 0) or 0)
        accumulated_tokens += (p_tok + c_tok)

    mean_token_entropy = float(sum(prefix_entropies) / len(prefix_entropies))
    min_action_logprob = float(min(prefix_logprobs))
    cumulative_action_logprob = float(sum(prefix_logprobs))

    # 3. Error signals
    current_is_error = 1.0 if bool(curr_info.get("is_error", False)) else 0.0
    total_errors = float(sum(prefix_errors))
    error_fraction = float(total_errors / len(steps))

    # 4. Loop signals
    curr_action = str(_get_field(current_step, "action", "")).strip().lower()
    curr_input_serialized = _serialize_action_input(_get_field(current_step, "action_input", None))
    curr_key = (curr_action, curr_input_serialized)

    repeat_action_count = 0.0
    prior_obs_words: List[Set[str]] = []
    for s in steps[:-1]:
        s_action = str(_get_field(s, "action", "")).strip().lower()
        s_input = _serialize_action_input(_get_field(s, "action_input", None))
        if (s_action, s_input) == curr_key:
            repeat_action_count += 1.0
        prior_obs_words.append(_extract_words(_get_field(s, "observation", "")))

    curr_obs_words = _extract_words(_get_field(current_step, "observation", ""))
    if prior_obs_words:
        max_obs_jaccard_overlap = float(max(_jaccard_overlap(curr_obs_words, p_words) for p_words in prior_obs_words))
    else:
        max_obs_jaccard_overlap = 0.0

    # 5. Action type one-hot
    is_search = 1.0 if curr_action == "search" else 0.0
    is_lookup = 1.0 if curr_action == "lookup" else 0.0
    is_finish = 1.0 if curr_action == "finish" else 0.0

    # 6. Progress
    normalized_depth = float(len(steps) / max_steps)
    curr_total_tokens = float(curr_info.get("total_tokens", 0.0) or 0.0)
    if curr_total_tokens <= 0.0:
        curr_total_tokens = accumulated_tokens
    normalized_total_tokens = float(curr_total_tokens / 4096.0)

    feature_values = [
        step_action_logprob,
        step_token_entropy,
        step_logit_margin,
        mean_token_entropy,
        min_action_logprob,
        cumulative_action_logprob,
        current_is_error,
        total_errors,
        error_fraction,
        repeat_action_count,
        max_obs_jaccard_overlap,
        is_search,
        is_lookup,
        is_finish,
        normalized_depth,
        normalized_total_tokens,
    ]

    return np.array(feature_values, dtype=np.float64)


class FeatureExtractor:
    """Convenience class wrapper around extract_prefix_features."""

    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps

    def extract(self, trajectory: Any) -> np.ndarray:
        """Extract features from Trajectory object, list of steps, or prefix."""
        steps = getattr(trajectory, "steps", trajectory)
        if not steps:
            return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        return extract_prefix_features(steps, max_steps=self.max_steps)

    def __call__(self, trajectory: Any) -> np.ndarray:
        return self.extract(trajectory)
