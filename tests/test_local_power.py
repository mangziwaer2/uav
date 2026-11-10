from dataclasses import replace
import json

import numpy as np
import pytest

from uav_em.belief import ElectromagneticBelief
from uav_em.config import BeliefConfig, MapConfig, ProjectConfig, load_config
from uav_em.environment import Geometry, Measurement
from uav_em.local_estimation import local_gp_prediction
from uav_em.simulation import simulate


def test_gp_single_point_matches_conditional_gaussian_and_keeps_distant_cells_unknown():
    settings = BeliefConfig(estimation_method="local_gp", support_radius_m=80.)
    grid = np.array([[[0., 0.], [500., 0.]]])
    measured = Measurement((0., 0.), 0., -60.)
    mean, error, support, age, count = local_gp_prediction(grid, 0., [measured], settings, -100.)
    variance = settings.unknown_error_db ** 2
    noise = settings.measured_error_db ** 2
    assert mean[0, 0] == pytest.approx(-100 + variance / (variance + noise) * 40, abs=1e-7)
    assert error[0, 0] == pytest.approx(np.sqrt(variance - variance ** 2 / (variance + noise) + noise), abs=1e-7)
    assert mean[0, 1] == -100 and error[0, 1] == 15
    assert support.tolist() == [[True, False]] and age.tolist() == [[0., -1.]]
    assert count == 1


def test_gp_time_decay_raises_uncertainty_and_expiry_removes_support():
    settings = BeliefConfig(estimation_method="local_gp")
    grid = np.array([[[20., 20.]]])
    measurements = [Measurement((20., 20.), 0., -60.), Measurement((20., 20.), 100., -10.)]
    fresh = local_gp_prediction(grid, 0., measurements, settings, -100.)
    older = local_gp_prediction(grid, 20., measurements, settings, -100.)
    expired = local_gp_prediction(grid, 36., measurements, settings, -100.)
    assert fresh[0][0, 0] > older[0][0, 0] > -100.
    assert fresh[1][0, 0] < older[1][0, 0]
    assert fresh[-1] == older[-1] == 1  # future point is never used
    assert expired[-1] == 0 and not expired[2].any()
    assert expired[0][0, 0] == -100 and expired[3][0, 0] == -1


def test_gp_uses_multiple_points_without_inventing_a_bearing():
    config = replace(ProjectConfig(), map=MapConfig(obstacles_m=()),
                     belief=BeliefConfig(estimation_method="local_gp", support_radius_m=100.))
    geometry = Geometry(config.map)
    belief = ElectromagneticBelief(config, geometry)
    belief.update(Measurement((100., 100.), 0., -95.))
    first = belief.snapshot(0.)
    belief.update(Measurement((180., 100.), 0., -65.))
    second = belief.snapshot(0.)
    assert second.at((180, 100))[0] > second.at((100, 100))[0] + 20
    assert second.at((180, 100))[1] < first.at((180, 100))[1]
    assert not second.directional_observations
    assert second.estimation_details["training_point_count"] == 2
    assert second.source_hypothesis is None
    hidden_changed = replace(config, electromagnetic=replace(config.electromagnetic, fields=()))
    other = ElectromagneticBelief(hidden_changed, geometry)
    for item in belief.measurements.values():
        other.update(item)
    np.testing.assert_array_equal(other.snapshot(0.).estimate_dbm, second.estimate_dbm)


def test_gp_bounds_fit_and_falls_back_to_prior_with_no_measurements():
    settings = BeliefConfig(estimation_method="local_gp", gp_max_points=2)
    grid = np.array([[[0., 0.], [40., 0.]]])
    empty = local_gp_prediction(grid, 0., [], settings, -100.)
    assert not empty[2].any() and np.all(empty[0] == -100.)
    points = [Measurement((float(i), 0.), float(i), -90. + i) for i in range(6)]
    bounded = local_gp_prediction(grid, 6., points, settings, -100.)
    recent = local_gp_prediction(grid, 6., points[-2:], settings, -100.)
    assert bounded[-1] == 2
    np.testing.assert_allclose(bounded[0], recent[0])


def small_config():
    base = ProjectConfig()
    return replace(base, map=MapConfig(columns=15, rows=11, start_m=(20, 100), goal_m=(260, 100), obstacles_m=()),
                   sensor=replace(base.sensor, mode="power_only"),
                   electromagnetic=replace(base.electromagnetic, fields=(), measurement_noise_db=0.),
                   belief=replace(base.belief, estimation_method="local_gp", support_radius_m=80.))


def test_point_only_episode_starts_without_prior_and_visible_estimator_is_logged():
    result = simulate(small_config(), initial_planner="astar")
    assert result["metrics"]["success"]
    assert result["metrics"]["observations"] == result["metrics"]["flight_observations"]
    assert result["metrics"]["directional_scans"] == 0
    assert result["metrics"]["advisor_calls"] == 0
    context = result["decisions"][0]["context"]
    assert context["electromagnetic_estimate"]["estimator"]["method"] == "local_gp"
    assert not context["directional_sensing"]["available"]
    assert result["frames"][0]["observation_count"] == 1
    assert "uncertainty_db" in result["frames"][0]
    assert "age_seconds" in result["frames"][0]
    for forbidden in ('"fields":', '"eirp_dbm":', '"truth":', '"velocity_mps":'):
        assert forbidden not in json.dumps(context)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value", [("estimation_method", "oracle"), ("gp_length_scale_m", 0.),
                                         ("gp_time_scale_seconds", -1.), ("gp_max_points", 257),
                                         ("gp_max_points", True)])
def test_gp_configuration_rejects_invalid_parameters(field, value):
    config = small_config()
    with pytest.raises(ValueError):
        replace(config, belief=replace(config.belief, **{field: value})).validate()


def test_current_default_profile_is_explicit():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/local_power.json")
    assert config.sensor.mode == "power_only"
    assert config.belief.estimation_method == "local_gp"
