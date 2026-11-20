"""Shared path result and length; RRT* is implemented once in repair_search."""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PlanResult:
    planner: str
    path: list[list[float]] = field(default_factory=list)
    cost: float | None = None
    length_m: float | None = None
    # A*为节点扩展数，RRT*为样本提议数，日志明确保留单位。
    iterations: int = 0
    iteration_unit: str = ""
    first_solution_iteration: int | None = None
    elapsed_seconds: float = 0.0
    reason: str = "no_path_within_budget"


def path_length(path) -> float:
    points = np.asarray(path, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0

