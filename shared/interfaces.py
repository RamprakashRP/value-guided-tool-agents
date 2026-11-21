"""Interface contracts and scoring functions for Value-Guided Tool Agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol, runtime_checkable

import joblib

from value_model.features import extract_prefix_features

_CACHED_MODEL: Any = None
_CACHED_MODEL_PATH: Optional[str] = None


def score_prefix(
    trajectory: Any,
    max_steps: int = 8,
    model_path: str = "data/value_model.joblib",
) -> float:
    """Score a prefix trajectory using a trained value model.

    If model_path exists on disk, lazy-loads the saved model, extracts prefix
    features phi(tau_{1:t}), and returns predicted probability P(success | tau_{1:t}).
    If model_path does not exist yet or trajectory is empty, safely returns 0.5.

    Args:
        trajectory: List of steps (dicts or StepRecord instances), or Trajectory object.
        max_steps: Maximum step depth (default: 8).
        model_path: Path to serialized joblib model pipeline (default: 'data/value_model.joblib').

    Returns:
        Scalar probability in [0.0, 1.0].
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH

    # Support Trajectory dataclass, dict with 'steps', or list of steps
    if hasattr(trajectory, "steps"):
        steps = trajectory.steps
    elif isinstance(trajectory, dict) and "steps" in trajectory:
        steps = trajectory["steps"]
    elif isinstance(trajectory, list):
        steps = trajectory
    else:
        steps = []

    if not steps or len(steps) == 0:
        return 0.5

    p = Path(model_path)
    if not p.exists():
        return 0.5

    resolved_path = str(p.resolve())
    if _CACHED_MODEL is None or _CACHED_MODEL_PATH != resolved_path:
        try:
            loaded = joblib.load(resolved_path)
            if isinstance(loaded, dict) and "pipeline" in loaded:
                _CACHED_MODEL = loaded["pipeline"]
            else:
                _CACHED_MODEL = loaded
            _CACHED_MODEL_PATH = resolved_path
        except Exception:
            return 0.5

    try:
        feat = extract_prefix_features(steps, max_steps=max_steps)
        probs = _CACHED_MODEL.predict_proba([feat])
        return float(probs[0, 1])
    except Exception:
        return 0.5


@runtime_checkable
class ValueEstimator(Protocol):
    """Protocol for scoring prefix trajectories tau_{1:t}."""

    def score_prefix(self, trajectory: Any) -> float:
        """Estimate the expected success probability or value of the trajectory prefix."""
        ...


class BaseValueEstimator(ABC):
    """Abstract base class for trajectory value estimators."""

    @abstractmethod
    def score_prefix(self, trajectory: Any) -> float:
        """Score a trajectory prefix tau_{1:t} -> float."""
        raise NotImplementedError


# Standard callable signature type alias
ScorePrefixFn = Callable[..., float]
