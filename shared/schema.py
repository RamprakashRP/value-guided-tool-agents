"""Shared data structures and contracts for Value-Guided Tool Agents.

Defines StepRecord and Trajectory classes used across rollouts,
value models, and search algorithms.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    """Represents an individual interaction step within a trajectory."""

    step_idx: int
    thought: Optional[str] = None
    action: str = ""
    action_input: Any = None
    observation: str = ""
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StepRecord:
        return cls(**data)


@dataclass
class Trajectory:
    """Represents a full or partial agent rollout trajectory."""

    task_id: str
    question: str
    steps: List[StepRecord] = field(default_factory=list)
    final_answer: Optional[str] = None
    ground_truth: Optional[str] = None
    is_correct: Optional[bool] = None
    total_reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: StepRecord) -> None:
        """Append a step and accumulate reward."""
        self.steps.append(step)
        self.total_reward += step.reward

    def prefix(self, t: int) -> Trajectory:
        """Return a prefix trajectory tau_{1:t} truncated to step t (1-indexed)."""
        t = max(0, min(t, len(self.steps)))
        prefix_steps = [StepRecord.from_dict(s.to_dict()) for s in self.steps[:t]]
        prefix_reward = sum(s.reward for s in prefix_steps)
        return Trajectory(
            task_id=self.task_id,
            question=self.question,
            steps=prefix_steps,
            final_answer=self.final_answer if t == len(self.steps) else None,
            ground_truth=self.ground_truth,
            is_correct=self.is_correct if t == len(self.steps) else None,
            total_reward=prefix_reward,
            metadata=dict(self.metadata),
        )

    @property
    def length(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Trajectory:
        steps_data = data.pop("steps", [])
        steps = [StepRecord.from_dict(s) for s in steps_data]
        return cls(steps=steps, **data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> Trajectory:
        return cls.from_dict(json.loads(json_str))
