"""Monte Carlo Tree Search (MCTS) guided by value function [Parthiv's Track B]."""

from typing import Any, Dict
from shared.schema import Trajectory
from shared.interfaces import ValueEstimator


def mcts_search(
    task: Dict[str, Any],
    value_model: ValueEstimator,
    num_simulations: int = 20,
    max_depth: int = 8,
) -> Trajectory:
    """Run MCTS where rollout evaluation or node priors use score_prefix."""
    # TODO: Implement selection, expansion, simulation, and backpropagation
    raise NotImplementedError("Track B: Implement MCTS")
