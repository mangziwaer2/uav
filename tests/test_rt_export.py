from dataclasses import replace
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from uav_em.config import ProjectConfig, load_config
from uav_em.environment import ElectromagneticWorld
from uav_em.rt_export import (combine_received_watts, load_rt_settings, rectangular_building,
                              validate_cell_centers, write_rt_bundle)
from uav_em.simulation import simulate


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix="interference_rt_test_", dir=ROOT / "temp") as name:
        path = Path(name).resolve()
        assert path.parent == (ROOT / "temp").resolve()
        yield path


def small_config():
    config = ProjectConfig()
    return replace(config, map=replace(config.map, columns=3, rows=3, resolution_m=10,
        start_m=(0, 0), goal_m=(20, 20), geometry_mode="fixed_altitude", obstacles_m=(), buildings=()),
        sensor=replace(config.sensor, mode="power_only"),
        electromagnetic=replace(config.electromagnetic, fields=(), measurement_noise_db=0),
        simulation=replace(config.simulation, max_seconds=1, rrt_attempt_budget=50))


def provenance():
    return {"engine_version": "TEST_FIXTURE_NOT_ENGINE_RESULT", "scene_name": "test",
            "license": "test fixture", "scene_sha256": "test-hash",
            "settings": {"transmitters": [{"name": "OFFLINE_ONLY_SOURCE", "position_m": [13, 17, 5]}]}}


def test_watts_are_converted_to_milliwatts_before_dbm_and_background_added_once():
    received = np.full((2, 2, 2), 1e-10)
    powers = combine_received_watts(received, -70)
    assert np.allclose(powers, -70 + 10 * np.log10(3))
    assert np.allclose(combine_received_watts(np.zeros((2, 2, 2)), -100), -100)


@pytest.mark.parametrize("received", [np.zeros((2, 2)), np.zeros((0, 2, 2)),
    np.full((1, 2, 2), -1), np.full((1, 2, 2), np.nan), np.full((1, 2, 2), np.inf)])
def test_invalid_engine_output_is_not_silently_replaced_by_background(received):
    with pytest.raises(ValueError):
        combine_received_watts(received, -100)


def test_rectangular_mesh_geometry_is_exact_and_origin_is_applied_once():
    vertices = np.array([[horizontal, vertical, altitude] for horizontal in (-10, 10)
                         for vertical in (-5, 5) for altitude in (0, 40)])
    building = rectangular_building("tower", vertices, [-100, -100])
    assert building["footprint_m"] == [[90, 95], [110, 95], [110, 105], [90, 105]]
    assert building["height_m"] == 40 and building["base_m"] == 0
    vertices[0, 0] += 1
    with pytest.raises(ValueError, match="rectangular"):
        rectangular_building("not_a_box", vertices, [-100, -100])


def test_cell_centers_reject_half_cell_offset_and_axis_flip():
    horizontal, vertical = np.meshgrid([0, 5, 10], [0, 5])
    centers = np.stack([horizontal - 100, vertical - 100, np.full_like(horizontal, 30)], axis=-1)
    validate_cell_centers(centers, [0, 5, 10], [0, 5], 30, [-100, -100])
    for wrong in (centers + 2.5, centers[::-1], centers.transpose(1, 0, 2)):
        with pytest.raises(ValueError, match="cell centers"):
            validate_cell_centers(wrong, [0, 5, 10], [0, 5], 30, [-100, -100])


def test_settings_are_separate_from_online_simulation_config():
    settings = load_rt_settings(ROOT / "configs/optional/sionna_rt_sources.json")
    config = load_config(ROOT / "configs/optional/sionna_rt_demo.json")
    assert len(settings["transmitters"]) == 2
    assert config.sensor.mode == "power_only" and not config.electromagnetic.fields


@pytest.mark.parametrize("field,value", [("samples_per_tx", 0), ("samples_per_tx", True),
    ("max_depth", -1), ("seed", -1), ("bandwidth_hz", 0), ("scene_name", "unknown"),
    ("origin_xy_m", [0]), ("transmitters", [])])
def test_invalid_settings_are_rejected_before_importing_engine(tmp_path, field, value):
    settings = load_rt_settings(ROOT / "configs/optional/sionna_rt_sources.json")
    settings[field] = value
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError):
        load_rt_settings(path)


def test_bundle_is_loadable_without_engine_and_does_not_leak_source_details(tmp_path):
    directory = tmp_path / "bundle"
    received = np.full((2, 3, 3), 1e-12)
    config_path = write_rt_bundle(directory, small_config(), received, [], provenance())
    config = load_config(config_path)
    world = ElectromagneticWorld(config)
    assert world.truth([0, 0], 0) == pytest.approx(10 * np.log10(2e-9 + 1e-10))
    measurement = world.observe([0, 0], 0)
    assert measurement.directional_scan is None
    assert measurement.interference_dbm == pytest.approx(world.truth([0, 0], 0))
    episode = simulate(config)
    for decision in episode["decisions"]:
        context = json.dumps(decision["context"])
        assert "OFFLINE_ONLY_SOURCE" not in context and "transmitters" not in context
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["transmitters"][0]["name"] == "OFFLINE_ONLY_SOURCE"
    assert manifest["power_grid_sha256"] == config.electromagnetic.power_grid["provenance"]["input_sha256"]
    with pytest.raises(FileExistsError):
        write_rt_bundle(directory, small_config(), received, [], provenance())


def test_bundle_validates_shape_before_creating_output(tmp_path):
    directory = tmp_path / "bad_bundle"
    with pytest.raises(ValueError, match="shape"):
        write_rt_bundle(directory, small_config(), np.ones((1, 2, 2)), [], provenance())
    assert not directory.exists()
