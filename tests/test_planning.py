import numpy as np
import pytest

from uav_em.belief import ElectromagneticBelief
from uav_em.config import ProjectConfig
from uav_em.environment import ElectromagneticWorld
from uav_em.planning import PlanningProblem, SamplingRegion, plan


@pytest.fixture
def snapshot():
    config = ProjectConfig()
    return ElectromagneticBelief(config, ElectromagneticWorld(config).geometry).snapshot(0)


@pytest.mark.parametrize("planner,budget", [("astar", 3000), ("rrt_star", 220)])
def test_planners_return_valid_routes_and_honest_counters(snapshot, planner, budget):
    result = plan(snapshot, [40, 300], [760, 300], planner, budget=budget, seed=7)
    assert result.path and result.reason == "solved"
    assert 0 < result.first_solution_iteration <= result.iterations <= budget
    np.testing.assert_allclose(result.path[0], [40, 300])
    np.testing.assert_allclose(result.path[-1], [760, 300])
    assert np.isfinite(PlanningProblem(snapshot, -80, "balanced").route_cost(result.path))
    assert result.length_m >= 720


@pytest.mark.parametrize("planner", ["astar", "rrt_star"])
def test_start_equals_goal(snapshot, planner):
    result = plan(snapshot, [40, 300], [40, 300], planner, budget=5)
    assert result.path == [[40, 300]]
    assert result.length_m == 0


def test_star_search_is_seed_reproducible(snapshot):
    first = plan(snapshot, [40, 300], [760, 300], "rrt_star", budget=120, seed=19)
    second = plan(snapshot, [40, 300], [760, 300], "rrt_star", budget=120, seed=19)
    assert first.path == second.path and first.cost == second.cost


def test_budget_failure_does_not_report_zero_path_length(snapshot):
    result = plan(snapshot, [40, 300], [760, 300], budget=1)
    assert not result.path and result.length_m is None and result.cost is None
    assert result.first_solution_iteration is None


def test_region_proposals_are_only_allowed_for_sampling_planner(snapshot):
    region = SamplingRegion((0, 400), (800, 600))
    with pytest.raises(ValueError):
        plan(snapshot, [40, 300], [760, 300], "astar", regions=(region,))
    result = plan(snapshot, [40, 300], [760, 300], "rrt_star", budget=220, regions=(region,))
    assert result.path


def test_invalid_region_and_unknown_algorithm_rejected(snapshot):
    with pytest.raises(ValueError):
        plan(snapshot, [40, 300], [760, 300], "dstar_lite")
    with pytest.raises(ValueError):
        plan(snapshot, [40, 300], [760, 300], "rrt_star", regions=(SamplingRegion((-1, 0), (100, 100)),))


def test_interference_estimate_changes_spatial_search(snapshot):
    before = plan(snapshot, [40, 300], [760, 300], "astar", profile="avoid_interference")
    snapshot.estimate_dbm[12:24, 18:26] = -30
    after = plan(snapshot, [40, 300], [760, 300], "astar", profile="avoid_interference")
    assert after.path != before.path
    assert all(not (360 <= point[0] <= 500 and 240 <= point[1] <= 460) for point in after.path)
