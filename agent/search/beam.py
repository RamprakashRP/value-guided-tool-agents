"""Beam Search guided by value function [Parthiv's Track B]."""

from typing import Any, Dict, List
from shared.schema import Trajectory
from shared.interfaces import ValueEstimator


def beam_search(
    task: Dict[str, Any],
    value_model: ValueEstimator,
    beam_width: int = 3,
    max_depth: int = 8,
) -> Trajectory:
    """Run beam search expanding top candidates scored by score_prefix."""
    # TODO: Implement beam expansion and pruning with value_model.score_prefix
    raise NotImplementedError("Track B: Implement Beam Search")
