import numpy as np

from uav_em.belief import ElectromagneticBelief
from uav_em.config import ProjectConfig
from uav_em.context import build_context, probe_candidates, route_summary
from uav_em.environment import ElectromagneticWorld, Measurement
from uav_em.planning import PlanningProblem


def test_unknown_context_values_are_not_reported_as_low_interference():
    config = ProjectConfig()
    world = ElectromagneticWorld(config)
    snapshot = ElectromagneticBelief(config, world.geometry).snapshot(0)
    context = build_context(config.map.start_m, config.map.goal_m, [], snapshot,
                            config, 100, [], [], None)
    estimate = context["electromagnetic_estimate"]
    assert estimate["current_grid_estimate_dbm"] is None
    assert all(sample[2] is None for sample in estimate["coarse_samples"])
    assert all(item["supported_high_interference_fraction"] is None
               for item in estimate["corridor_summaries"])
    assert context["route"]["max_supported_estimate_dbm"] is None


def test_route_summary_excludes_unknown_background_from_exposure_fraction():
    config = ProjectConfig()
    world = ElectromagneticWorld(config)
    belief = ElectromagneticBelief(config, world.geometry)
    belief.update(Measurement(config.map.start_m, 0, -60))
    summary = route_summary([[40, 300], [200, 300]], belief.snapshot(0), config)
    assert 0 < summary["supported_fraction"] < 1
    assert summary["supported_above_reference_fraction"] == 1
    assert summary["max_supported_estimate_dbm"] == -60


def test_stronger_interference_increases_cost_and_never_changes_geometry():
    config = ProjectConfig()
    world = ElectromagneticWorld(config)
    snapshot = ElectromagneticBelief(config, world.geometry).snapshot(0)
    problem = PlanningProblem(snapshot, -80, "balanced")
    weak_cost = problem.edge_cost([40, 100], [200, 100])
    snapshot.estimate_dbm[:] = -60
    strong_cost = problem.edge_cost([40, 100], [200, 100])
    assert np.isfinite(strong_cost) and strong_cost > weak_cost
    assert world.geometry.edge_free([40, 100], [200, 100])


def test_probe_candidates_include_task_relation_without_unknown_safety_claims():
    config = ProjectConfig()
    world = ElectromagneticWorld(config)
    snapshot = ElectromagneticBelief(config, world.geometry).snapshot(0)
    candidates = probe_candidates(config.map.start_m, config.map.goal_m, snapshot, config, set())
    assert candidates
    for candidate in candidates:
        assert candidate["corridor_id"] in ("south", "middle", "north")
        assert isinstance(candidate["goal_distance_reduction_m"], float)
        assert candidate["approach_route"]["supported_fraction"] == 0
        assert candidate["approach_route"]["max_supported_estimate_dbm"] is None


def test_corridor_summary_reports_supported_peak_and_age_even_if_coarse_samples_miss_it():
    config = ProjectConfig()
    world = ElectromagneticWorld(config)
    belief = ElectromagneticBelief(config, world.geometry)
    belief.update(Measurement(config.map.start_m, 0, -60))
    context = build_context(config.map.start_m, config.map.goal_m, [], belief.snapshot(5),
                            config, 100, [], [], None)
    supported = [item for item in context["electromagnetic_estimate"]["corridor_summaries"]
                 if item["supported_fraction"] > 0]
    assert supported
    assert all(item["maximum_supported_estimate_dbm"] == -60 for item in supported)
    assert all(item["mean_supported_age_seconds"] == 5 for item in supported)
