from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from uav_em.buildings import Building
from uav_em.config import ProjectConfig, load_config
from uav_em.environment import ElectromagneticWorld, Geometry
from uav_em.propagation import PropagationModel, propagation_received_dbm
from uav_em.simulation import simulate
from uav_em.source_model import parse_source_hypothesis, predict_power
from uav_em.visualization import config_from_result, render


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix="interference_25d_test_", dir=ROOT / "temp") as name:
        path = Path(name).resolve()
        assert path.parent == (ROOT / "temp").resolve()
        yield path


def building(height=40, footprint=None, **changes):
    value = {"id": "building", "height_m": height,
             "footprint_m": footprint or [[40, 40], [80, 40], [80, 80], [40, 80]]}
    value.update(changes)
    return value


def fixed_config(buildings=()):
    base = ProjectConfig()
    return replace(base, map=replace(base.map, columns=7, rows=7, resolution_m=20,
                                    start_m=(10, 10), goal_m=(110, 110), obstacles_m=(),
                                    geometry_mode="fixed_altitude", buildings=tuple(buildings)),
                   aircraft=replace(base.aircraft, altitude_m=30, vertical_clearance_m=5, clearance_m=2),
                   electromagnetic=replace(base.electromagnetic, fields=(), measurement_noise_db=0))


@pytest.mark.parametrize("height,blocked", [(12, False), (24.999, False), (25, True), (30, True), (60, True)])
def test_roof_clearance_is_inclusive_and_independent_of_horizontal_clearance(height, blocked):
    geometry = ElectromagneticWorld(fixed_config([building(height)])).geometry
    assert geometry.free([60, 60]) is not blocked
    assert geometry.edge_free([20, 60], [100, 60]) is not blocked
    assert geometry.free([37, 60])
    assert geometry.free([38, 60]) is not blocked


def test_planar_mode_and_legacy_rectangles_remain_explicit_exclusions():
    config = fixed_config([building(10)])
    planar = replace(config, map=replace(config.map, geometry_mode="planar"))
    assert not ElectromagneticWorld(planar).geometry.free([60, 60])
    rectangle = replace(config, map=replace(config.map, buildings=(), obstacles_m=((40, 40, 80, 80),)))
    assert not ElectromagneticWorld(rectangle).geometry.free([60, 60])
    with pytest.raises(ValueError, match="requires aircraft altitude"):
        Geometry(config.map)


def test_concave_notch_is_not_replaced_by_bounding_rectangle():
    footprint = [[40, 40], [80, 40], [80, 50], [50, 50], [50, 80], [40, 80]]
    geometry = ElectromagneticWorld(fixed_config([building(footprint=footprint)])).geometry
    assert geometry.free([65, 65])
    assert geometry.edge_free([65, 65], [100, 100])
    assert not geometry.edge_free([20, 65], [65, 65])
    assert not geometry.free([50, 65])


def test_courtyard_is_free_but_crossing_wall_is_blocked():
    item = building(holes_m=[[[50, 50], [70, 50], [70, 70], [50, 70]]])
    geometry = ElectromagneticWorld(fixed_config([item])).geometry
    assert geometry.free([60, 60]) and geometry.edge_free([55, 55], [65, 65])
    assert not geometry.edge_free([60, 60], [100, 60])
    assert not geometry.free([51, 60])


def test_horizontal_corner_clearance_uses_distance_not_square_bounding_box():
    geometry = ElectromagneticWorld(fixed_config([building()])).geometry
    assert geometry.free([38.5, 38.5])
    assert not geometry.free([39, 39])
    assert not geometry.edge_free([20, 38], [100, 38])


@pytest.mark.parametrize("changes", [
    {"height_m": 0}, {"height_m": True}, {"base_m": -1}, {"loss_db": float("nan")},
    {"footprint_m": [[0, 0], [10, 10], [0, 10], [10, 0]]},
    {"footprint_m": [[0, 0], [10, 0], [10, 0], [0, 10]]},
    {"holes_m": [[[60, 60], [90, 60], [90, 90], [60, 90]]]},
    {"holes_m": [[[50, 50], [70, 50], [70, 70], [50, 70]], [[55, 55], [60, 55], [60, 60], [55, 60]]]},
])
def test_invalid_building_geometry_and_physical_parameters_fail(changes):
    with pytest.raises(ValueError):
        fixed_config([building(**changes)]).validate()


def test_building_id_and_geometry_modes_are_validated():
    with pytest.raises(ValueError, match="unique"):
        fixed_config([building(), building()]).validate()
    config = fixed_config()
    with pytest.raises(ValueError, match="geometry_mode"):
        replace(config, map=replace(config.map, geometry_mode="unknown")).validate()


def test_prism_clips_ray_by_height_before_horizontal_intersection():
    item = Building.from_dict(building(10))
    assert not item.ray_intersection([20, 60], 30, [100, 60], 30)
    assert item.ray_intersection([20, 60], 30, [100, 60], 0)
    raised = Building.from_dict(building(10, base_m=40))
    assert not raised.ray_intersection([20, 60], 30, [100, 60], 0)


def test_prism_supports_vectorized_moving_centers_and_holes():
    item = Building.from_dict(building(holes_m=[[[50, 50], [70, 50], [70, 70], [50, 70]]]))
    result = item.ray_intersection([[55, 55], [20, 60], [20, 90]], 30,
                                   [[65, 65], [100, 60], [100, 90]], 20)
    np.testing.assert_array_equal(result, [False, True, False])


def test_lower_building_can_attenuate_source_without_blocking_flight():
    item = building(20, loss_db=11)
    config = fixed_config([item])
    world = ElectromagneticWorld(config)
    assert world.geometry.free([60, 60])
    model = PropagationModel.from_config(config)
    source = {"eirp_dbm": 0, "height_m": 0}
    blocked = propagation_received_dbm([20, 60], [100, 60], source, model)
    clear = propagation_received_dbm([20, 60], [100, 60], source, replace(model, buildings=()))
    assert clear - blocked == pytest.approx(11)
    encoded = json.dumps(asdict(model))
    assert '"fields"' not in encoded


def test_simulator_and_candidate_source_share_public_building_geometry():
    config = fixed_config([building(20)])
    source = {"center_m": [100, 60], "eirp_dbm": 0, "height_m": 0,
              "velocity_mps": [0, 0], "start_seconds": 0, "end_seconds": 100}
    config = replace(config, electromagnetic=replace(config.electromagnetic, fields=(source,)))
    hypothesis = parse_source_hypothesis({"reference_seconds": 0, "sources": [
        {name: source[name] for name in ("center_m", "eirp_dbm", "height_m", "velocity_mps")}]}, (120, 120))
    world = ElectromagneticWorld(config)
    predicted = predict_power(hypothesis, [20, 60], 0, -100, PropagationModel.from_config(config))
    assert world.truth([20, 60], 0) == pytest.approx(predicted)


def test_fixed_altitude_episode_context_replay_and_plot_are_consistent(tmp_path):
    config = load_config(ROOT / "configs/local_power.json")
    config = replace(config, simulation=replace(config.simulation, max_seconds=2, rrt_attempt_budget=100))
    episode = simulate(config)
    context = episode["decisions"][0]["context"]
    assert context["map"]["geometry_mode"] == "fixed_altitude"
    assert context["map"]["building_count"] == 3 and context["map"]["blocking_building_count"] == 2
    assert config_from_result(episode).map.buildings == config.map.buildings
    render(episode, tmp_path / "buildings.png")
    assert (tmp_path / "buildings.png").stat().st_size > 1000


def test_frozen_local_source_null_grid_metadata_is_accepted_without_mutating_episode(tmp_path):
    config = fixed_config()
    value = config.to_dict()
    value["electromagnetic"]["power_grid"] = None
    path = tmp_path / "frozen_local_source.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert load_config(path).to_dict() == json.loads(json.dumps(config.to_dict()))
    episode = {"schema_version": 2, "evaluation_only_config": value}
    assert config_from_result(episode).to_dict() == config.to_dict()
    assert "power_grid" in episode["evaluation_only_config"]["electromagnetic"]


@pytest.mark.parametrize("removed_grid", [{}, {"path": "old_grid.npz"}, [], False, 0])
def test_removed_nonnull_grid_is_explicitly_rejected_in_config_and_replay(tmp_path, removed_grid):
    value = fixed_config().to_dict()
    value["electromagnetic"]["power_grid"] = removed_grid
    path = tmp_path / "removed_grid.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="external power_grid support was removed"):
        load_config(path)
    with pytest.raises(ValueError, match="external power_grid support was removed"):
        config_from_result({"schema_version": 2, "evaluation_only_config": value})
