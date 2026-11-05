from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from uav_em.advisor import MockAdvisor
from uav_em.belief import ElectromagneticBelief
from uav_em.buildings import Building
from uav_em.config import ProjectConfig, load_config
from uav_em.context import build_context
from uav_em.environment import ElectromagneticWorld, Geometry, Measurement
from uav_em.map_import import import_buildings
from uav_em.power_grid import PowerGrid, load_power_grid
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
    assert '"fields"' not in encoded and '"power_grid"' not in encoded


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


def grid_config():
    config = fixed_config()
    grid = {"x_m": [0, 120], "y_m": [0, 120], "power_dbm": [[-100, -80], [-100, -80]],
            "altitude_m": 30, "carrier_frequency_hz": 2.4e9}
    return replace(config, sensor=replace(config.sensor, mode="power_only"),
                   electromagnetic=replace(config.electromagnetic, power_grid=grid))


def test_external_grid_interpolates_milliwatts_instead_of_dbm_and_never_extrapolates():
    config = grid_config()
    grid = PowerGrid(config.electromagnetic.power_grid)
    assert grid.sample([60, 60]) == pytest.approx(10 * np.log10((1e-10 + 1e-8) / 2))
    np.testing.assert_allclose(grid.sample([[0, 0], [120, 120]]), [-100, -80])
    with pytest.raises(ValueError, match="outside"):
        grid.sample([-1, 0])


def test_external_grid_has_no_source_positions_or_bearings_and_is_static():
    world = ElectromagneticWorld(grid_config())
    measurement = world.observe([10, 10], 0)
    assert measurement.directional_scan is None
    assert world.source_positions_for_evaluation(0) == []
    assert measurement.interference_dbm == world.truth([10, 10], 1000)


@pytest.mark.parametrize("changes", [
    {"altitude_m": 1}, {"carrier_frequency_hz": 900e6}, {"x_m": [0, 100]},
    {"x_m": [120, 0]}, {"power_dbm": [[float("nan"), -80], [-100, -80]]},
    {"power_dbm": [[True, False], [True, False]]}, {"power_dbm": [[-90, -80]]},
])
def test_external_grid_rejects_incompatible_or_missing_data(changes):
    config = grid_config()
    value = {**config.electromagnetic.power_grid, **changes}
    with pytest.raises(ValueError):
        replace(config, electromagnetic=replace(config.electromagnetic, power_grid=value)).validate()


def test_external_grid_rejects_bearing_fabrication_and_synthetic_source_mixing():
    config = grid_config()
    with pytest.raises(ValueError, match="power_only"):
        replace(config, sensor=replace(config.sensor, mode="power_bearing")).validate()
    with pytest.raises(ValueError, match="empty synthetic fields"):
        replace(config, electromagnetic=replace(config.electromagnetic, fields=ProjectConfig().electromagnetic.fields)).validate()


def test_grid_loading_embeds_data_for_replay_and_keeps_truth_out_of_context(tmp_path):
    config = load_config(ROOT / "configs/optional/demo_power_grid.json")
    assert "path" not in config.electromagnetic.power_grid
    short = replace(config, simulation=replace(config.simulation, max_seconds=2, rrt_attempt_budget=100))
    episode = simulate(short, MockAdvisor(), advisor_name="mock")
    replay = config_from_result(episode)
    assert replay.electromagnetic.power_grid == config.electromagnetic.power_grid
    contexts = json.dumps([item["context"] for item in episode["decisions"]])
    assert '"power_grid"' not in contexts and '"power_dbm"' not in contexts
    render(episode, tmp_path / "grid.png")
    assert (tmp_path / "grid.png").stat().st_size > 1000


def test_npz_grid_loading_does_not_enable_pickle(tmp_path):
    value = grid_config().electromagnetic.power_grid
    np.savez(tmp_path / "grid.npz", **value)
    assert load_power_grid(tmp_path / "grid.npz") == value
    np.savez(tmp_path / "wrong.npz", x_m=np.array([{}], dtype=object))
    with pytest.raises(ValueError):
        load_power_grid(tmp_path / "wrong.npz")


def test_fixed_altitude_episode_context_replay_and_plot_are_consistent(tmp_path):
    config = load_config(ROOT / "configs/demo_25d.json")
    config = replace(config, simulation=replace(config.simulation, max_seconds=2, rrt_attempt_budget=100))
    episode = simulate(config)
    context = episode["decisions"][0]["context"]
    assert context["map"]["geometry_mode"] == "fixed_altitude"
    assert context["map"]["building_count"] == 3 and context["map"]["blocking_building_count"] == 2
    assert config_from_result(episode).map.buildings == config.map.buildings
    render(episode, tmp_path / "buildings.png")
    assert (tmp_path / "buildings.png").stat().st_size > 1000


def test_geojson_import_preserves_polygon_height_and_explicit_source():
    template = load_config(ROOT / "configs/optional/import_25d_template.json")
    result = import_buildings(ROOT / "datasets/example_buildings_local.geojson", template, "local_m", "fixture", "fixture")
    assert result.map.geometry_mode == "fixed_altitude"
    assert len(result.map.buildings) == 2
    assert len(result.map.buildings[1]["footprint_m"]) == 6
    assert result.map.provenance["assumed_height_count"] == 0
    assert ElectromagneticWorld(result).geometry.free([200, 300])
    assert result.map.provenance["input_sha256"]


def feature_collection(coordinates, properties=None, kind="Polygon"):
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "id": "one",
            "properties": properties or {}, "geometry": {"type": kind, "coordinates": coordinates}}]}


def test_geojson_missing_height_needs_explicit_default_and_footprint_is_not_rectangularized(tmp_path):
    path = tmp_path / "buildings.geojson"
    path.write_text(json.dumps(feature_collection([[[40,40],[80,40],[80,80],[40,80],[40,40]]])), encoding="utf-8")
    config = fixed_config()
    with pytest.raises(ValueError, match="missing building height"):
        import_buildings(path, config, "local_m", "fixture", "fixture")
    imported = import_buildings(path, config, "local_m", "fixture", "fixture", default_height_m=12)
    assert imported.map.buildings[0]["height_source"] == "assumed_default"
    assert imported.map.provenance["assumed_height_count"] == 1


def test_geojson_lonlat_projection_is_explicit_and_local(tmp_path):
    origin = [114.0, 22.0]
    ring = [[114.0004,22.0004],[114.0007,22.0004],[114.0007,22.0007],[114.0004,22.0007],[114.0004,22.0004]]
    path = tmp_path / "geo.geojson"
    path.write_text(json.dumps(feature_collection([ring], {"height": "12 m"})), encoding="utf-8")
    with pytest.raises(ValueError, match="southwest origin"):
        import_buildings(path, fixed_config(), "lonlat", "fixture", "fixture")
    result = import_buildings(path, fixed_config(), "lonlat", "fixture", "fixture", origin)
    assert 40 < result.map.buildings[0]["footprint_m"][0][0] < 42
    assert result.map.provenance["projection"] == "local_equirectangular_sphere_5km_limit"


def test_geojson_multipolygon_holes_and_roof_vs_base_are_retained(tmp_path):
    outer = [[40,40],[80,40],[80,80],[40,80],[40,40]]
    hole = [[50,50],[70,50],[70,70],[50,70],[50,50]]
    other = [[85,40],[100,40],[100,80],[85,80],[85,40]]
    path = tmp_path / "multi.geojson"
    path.write_text(json.dumps(feature_collection([[outer,hole],[other]], {"height": 40, "min_height": 10}, "MultiPolygon")), encoding="utf-8")
    result = import_buildings(path, fixed_config(), "local_m", "fixture", "fixture")
    assert len(result.map.buildings) == 2 and result.map.buildings[0]["height_m"] == 30
    assert result.map.buildings[0]["base_m"] == 10 and len(result.map.buildings[0]["holes_m"]) == 1


def test_unobserved_grid_changes_do_not_change_belief_or_model_context():
    config = grid_config()
    changed = {**config.electromagnetic.power_grid, "power_dbm": [[-60, -60], [-60, -60]]}
    other = replace(config, electromagnetic=replace(config.electromagnetic, power_grid=changed))
    contexts = []
    for current in (config, other):
        world = ElectromagneticWorld(current)
        belief = ElectromagneticBelief(current, world.geometry)
        belief.update(Measurement((10, 10), 0, -95))
        contexts.append(build_context([10, 10], [110, 110], [], belief.snapshot(0), current, 100, [], [], None))
    assert contexts[0] == contexts[1]
