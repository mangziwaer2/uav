"""仅使用可见估计地图的A*和标准RRT*；不把重新搜索称为增量修复。"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import time

import numpy as np

from .belief import MapSnapshot


PROFILES = {
    "balanced": (2.0, 0.4),
    "avoid_interference": (5.0, 0.5),
    "explore": (1.0, 0.15),
}
PLANNERS = ("astar", "rrt_star")


@dataclass(frozen=True)
class SamplingRegion:
    """模型给出的二维采样盒，m；不是已确认低干扰或安全的区域。"""

    low_m: tuple[float, float]
    high_m: tuple[float, float]
    weight: float = 1.0


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


class PlanningProblem:
    """路径代价=长度加干扰超参考值／信息不足软惩罚；几何是硬约束。"""

    def __init__(self, snapshot: MapSnapshot, interference_reference_dbm: float, profile: str):
        if profile not in PROFILES:
            raise ValueError("unknown planning profile")
        self.snapshot = snapshot
        self.geometry = snapshot.geometry
        self.reference = interference_reference_dbm
        self.interference_weight, self.uncertainty_weight = PROFILES[profile]

    def edge_cost(self, start, end) -> float:
        if not self.geometry.edge_free(start, end):
            return math.inf
        length = float(np.linalg.norm(np.asarray(end) - start))
        count = max(2, int(math.ceil(length / (self.geometry.config.resolution_m / 2))) + 1)
        points = np.linspace(start, end, count)
        nodes = np.rint(points / self.geometry.config.resolution_m).astype(int)
        interference = self.snapshot.estimate_dbm[nodes[:, 1], nodes[:, 0]]
        error = self.snapshot.uncertainty_db[nodes[:, 1], nodes[:, 0]]
        excess = np.clip((interference - self.reference) / 10, 0, 4)
        uncertainty = np.clip(error / 15, 0, 2)
        density = 1 + self.interference_weight * excess + self.uncertainty_weight * uncertainty
        return length * float(np.mean(density))

    def route_cost(self, path) -> float:
        if not path or not all(self.geometry.free(point) for point in path):
            return math.inf
        return sum(self.edge_cost(start, end) for start, end in zip(path, path[1:]))


def _astar(problem: PlanningProblem, start: np.ndarray, goal: np.ndarray, budget: int) -> PlanResult:
    result = PlanResult("astar", iteration_unit="node_expansions")
    geometry = problem.geometry
    if np.linalg.norm(start - goal) < 1e-9:
        result.path, result.reason = [start.tolist()], "goal_reached"
        result.first_solution_iteration = 0
        return result

    def connectors(point):
        center = geometry.node(point)
        values = {}
        for delta_column in range(-2, 3):
            for delta_row in range(-2, 3):
                node = (center[0] + delta_column, center[1] + delta_row)
                cost = problem.edge_cost(point, geometry.position(node))
                if math.isfinite(cost):
                    values[node] = cost
        return values

    starts, goals = connectors(start), connectors(goal)
    distances = dict(starts)
    parents = {node: None for node in starts}
    queue = [(cost + float(np.linalg.norm(geometry.position(node) - goal)), cost, node)
             for node, cost in starts.items()]
    heapq.heapify(queue)
    target = (-1, -1)
    while queue and result.iterations < budget:
        _, current_cost, current = heapq.heappop(queue)
        if current_cost > distances[current] + 1e-9:
            continue
        result.iterations += 1
        if current == target:
            reverse = []
            previous = parents[target]
            while previous is not None:
                reverse.append(geometry.position(previous).tolist())
                previous = parents[previous]
            result.path = [start.tolist(), *reversed(reverse), goal.tolist()]
            result.first_solution_iteration = result.iterations
            result.reason = "solved"
            return result
        neighbors = []
        if current in goals:
            neighbors.append((target, goals[current], 0.0))
        for delta_column, delta_row in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                        (1, 1), (1, -1), (-1, 1), (-1, -1)):
            node = (current[0] + delta_column, current[1] + delta_row)
            if not 0 <= node[0] < geometry.config.columns or not 0 <= node[1] < geometry.config.rows:
                continue
            cost = problem.edge_cost(geometry.position(current), geometry.position(node))
            if math.isfinite(cost):
                neighbors.append((node, cost, float(np.linalg.norm(geometry.position(node) - goal))))
        for node, edge, heuristic in neighbors:
            candidate = current_cost + edge
            if candidate < distances.get(node, math.inf) - 1e-9:
                distances[node], parents[node] = candidate, current
                heapq.heappush(queue, (candidate + heuristic, candidate, node))
    return result


def _rrt_star(problem: PlanningProblem, start: np.ndarray, goal: np.ndarray, budget: int,
              seed: int, regions: tuple[SamplingRegion, ...]) -> PlanResult:
    result = PlanResult("rrt_star", iteration_unit="sample_proposals")
    geometry = problem.geometry
    random = np.random.default_rng(seed)
    nodes, parents, costs, children = [start], [-1], [0.0], [set()]
    connectors = []
    step_m = 3 * geometry.config.resolution_m
    for attempt in range(1, budget + 1):
        result.iterations = attempt
        draw = random.random()
        if draw < 0.1:
            proposed = goal
        elif regions and draw >= 0.4:
            weights = np.array([region.weight for region in regions])
            region = regions[int(random.choice(len(regions), p=weights / weights.sum()))]
            proposed = random.uniform(region.low_m, region.high_m)
        else:
            proposed = random.uniform(np.zeros(2), geometry.high)
        positions = np.asarray(nodes)
        distances = np.linalg.norm(positions - proposed, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] < 1e-9:
            continue
        candidate = nodes[nearest] + (proposed - nodes[nearest]) * min(1, step_m / distances[nearest])
        if not geometry.edge_free(nodes[nearest], candidate):
            continue
        distances = np.linalg.norm(positions - candidate, axis=1)
        if distances.min() < 1e-7:
            continue
        radius = max(step_m, 6 * step_m * math.sqrt(math.log(len(nodes) + 2) / (len(nodes) + 1)))
        near = set(np.flatnonzero(distances <= radius).tolist()) | {nearest}
        edge_costs = {index: problem.edge_cost(nodes[index], candidate) for index in near}
        parent = min(near, key=lambda index: costs[index] + edge_costs[index])
        cost = costs[parent] + edge_costs[parent]
        if not math.isfinite(cost):
            continue
        current = len(nodes)
        nodes.append(candidate)
        parents.append(parent)
        costs.append(cost)
        children.append(set())
        children[parent].add(current)
        ancestors = set()
        ancestor = parent
        while ancestor >= 0:
            ancestors.add(ancestor)
            ancestor = parents[ancestor]
        for index in near - ancestors - {0}:
            new_cost = cost + edge_costs[index]
            if new_cost < costs[index] - 1e-9:
                children[parents[index]].remove(index)
                parents[index] = current
                children[current].add(index)
                delta = new_cost - costs[index]
                pending = [index]
                while pending:
                    descendant = pending.pop()
                    costs[descendant] += delta
                    pending.extend(children[descendant])
        if np.linalg.norm(candidate - goal) <= step_m:
            edge = problem.edge_cost(candidate, goal)
            if math.isfinite(edge):
                connectors.append((current, edge))
                if result.first_solution_iteration is None:
                    result.first_solution_iteration = attempt
    if connectors:
        current, _ = min(connectors, key=lambda item: costs[item[0]] + item[1])
        reverse = [goal.tolist()]
        while current >= 0:
            reverse.append(nodes[current].tolist())
            current = parents[current]
        result.path, result.reason = list(reversed(reverse)), "solved"
    return result


def plan(snapshot: MapSnapshot, start, goal, planner: str = "rrt_star", profile: str = "balanced",
         interference_reference_dbm: float = -80.0, budget: int = 1000, seed: int = 7,
         regions: tuple[SamplingRegion, ...] = ()) -> PlanResult:
    """统一入口，记录算法各自的预算单位，输出后独立复核几何与代价。"""
    if planner not in PLANNERS or type(budget) is not int or budget <= 0:
        raise ValueError("unsupported planner or invalid budget")
    if regions and planner != "rrt_star":
        raise ValueError("sampling regions only apply to rrt_star")
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    problem = PlanningProblem(snapshot, interference_reference_dbm, profile)
    if not problem.geometry.free(start) or not problem.geometry.free(goal):
        return PlanResult(planner, reason="invalid_endpoint")
    for region in regions:
        low, high = np.asarray(region.low_m), np.asarray(region.high_m)
        if (low.shape != (2,) or high.shape != (2,) or not np.all(np.isfinite([low, high]))
                or np.any(low < 0) or np.any(high > problem.geometry.high) or np.any(high <= low)
                or not math.isfinite(region.weight) or region.weight <= 0):
            raise ValueError("invalid sampling region")
    started = time.perf_counter()
    if np.linalg.norm(start - goal) < 1e-9:
        result = PlanResult(planner, path=[start.tolist()], reason="goal_reached",
                            iteration_unit="node_expansions" if planner == "astar" else "sample_proposals",
                            first_solution_iteration=0)
    else:
        result = (_astar(problem, start, goal, budget) if planner == "astar" else
                  _rrt_star(problem, start, goal, budget, seed, regions))
    if result.path:
        result.path = [point for index, point in enumerate(result.path)
                       if index == 0 or np.linalg.norm(np.asarray(point) - result.path[index - 1]) > 1e-8]
        cost = problem.route_cost(result.path)
        if not math.isfinite(cost):
            raise RuntimeError("planner returned a geometrically invalid route")
        result.cost, result.length_m = cost, path_length(result.path)
    result.elapsed_seconds = time.perf_counter() - started
    return result
