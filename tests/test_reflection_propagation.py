from dataclasses import asdict, replace

import numpy as np
import pytest

from uav_em.belief import ElectromagneticBelief
from uav_em.config import ProjectConfig, load_config
from uav_em.context import build_context
from uav_em.environment import ElectromagneticWorld, Measurement
from uav_em.propagation import PropagationModel, fresnel_power, propagation_arrivals, propagation_received_dbm
from uav_em.sensing import beam_power_dbm
from uav_em.source_model import parse_source_hypothesis, predict_power


def reflection_config(buildings=None, **changes):
    base = ProjectConfig()
    wall = {"id": "wall", "footprint_m": [[0, 20], [40, 20], [40, 22], [0, 22]],
            "height_m": 8, "loss_db": 12}
    source = {"center_m": [20, 10], "height_m": 2, "eirp_dbm": -10,
              "velocity_mps": [0.1, 0], "start_seconds": 0, "end_seconds": 100}
    return replace(base, map=replace(base.map, columns=61, rows=61, resolution_m=1,
                   start_m=(10, 10), goal_m=(50, 10), obstacles_m=(),
                   buildings=(wall,) if buildings is None else buildings),
                   aircraft=replace(base.aircraft, altitude_m=2, clearance_m=0, vertical_clearance_m=0),
                   electromagnetic=replace(base.electromagnetic, propagation_mode="single_reflection",
                       path_loss_exponent=2, conductivity_s_per_m=0, measurement_noise_db=0,
                       fields=(source,), **changes),
                   sensor=replace(base.sensor, noise_db=0, bearing_error_deg=0))


def paths(config, position=(10, 10), center=(20, 10)):
    return list(propagation_arrivals(position, center, config.electromagnetic.fields[0],
                                    PropagationModel.from_config(config)))


def test_fresnel_normal_incidence_and_passivity():
    assert fresnel_power(1, 2.4e9, 4, 0) == pytest.approx(1 / 9)
    np.testing.assert_allclose(fresnel_power(np.linspace(0.01, 1, 100), 2.4e9, 1, 0), 0, atol=1e-26)
    power = fresnel_power(np.linspace(0, 1, 100), 2.4e9, 4, 0.005)
    assert np.all((power >= 0) & (power <= 1))


def test_free_space_friis_and_inverse_square_without_reflective_material():
    config = reflection_config(buildings=(), relative_permittivity=1)
    model = PropagationModel.from_config(config)
    actual = propagation_received_dbm([[10, 10], [0, 10]], [20, 10], config.electromagnetic.fields[0], model)
    expected = -10 - 20 * np.log10(4 * np.pi * np.array([10, 20]) * 2.4e9 / 299792458)
    np.testing.assert_allclose(actual, expected)
    assert actual[0] - actual[1] == pytest.approx(20 * np.log10(2))


def test_wall_image_method_uses_total_length_and_reflection_arrival_direction():
    config = reflection_config()
    arrivals = paths(config)
    assert [arrival.kind for arrival in arrivals] == ["direct", "ground", "wall"]
    wall = arrivals[-1]
    distance = np.sqrt(10 ** 2 + 20 ** 2)
    reflection = fresnel_power(20 / distance, 2.4e9, 4, 0)
    expected = -10 - 20 * np.log10(4 * np.pi * distance * 2.4e9 / 299792458) + 10 * np.log10(reflection)
    assert wall.power_dbm == pytest.approx(expected)
    assert wall.bearing_deg == pytest.approx(np.degrees(np.arctan2(10, 5)))
    assert arrivals[0].bearing_deg == pytest.approx(0)


@pytest.mark.parametrize("wall", [
    {"id": "short", "footprint_m": [[0,20],[5,20],[5,22],[0,22]], "height_m": 8},
    {"id": "low", "footprint_m": [[0,20],[40,20],[40,22],[0,22]], "height_m": 1},
])
def test_reflection_point_must_be_on_finite_wall_and_within_height(wall):
    assert all(arrival.kind != "wall" for arrival in paths(reflection_config(buildings=(wall,))))


def test_reflected_legs_cannot_pass_through_another_building():
    config = reflection_config()
    blocker = {"id": "blocker", "footprint_m": [[12,13],[14,13],[14,18],[12,18]], "height_m": 8}
    blocked = replace(config, map=replace(config.map, buildings=(*config.map.buildings, blocker)))
    original_wall = paths(config)[-1].bearing_deg
    assert not any(arrival.kind == "wall" and abs(arrival.bearing_deg - original_wall) < 1e-6
                   for arrival in paths(blocked))


def test_reflection_does_not_cross_own_building_or_connect_opposite_sides():
    config = reflection_config()
    assert all(arrival.kind != "wall" for arrival in paths(config, position=(10, 30)))
    assert all(arrival.kind != "wall" for arrival in paths(config, center=(20, 21)))


def test_path_power_sums_match_truth_and_receiver_beams():
    config = reflection_config()
    world = ElectromagneticWorld(config)
    arrivals = paths(config)
    total = 10 ** (config.electromagnetic.background_dbm / 10)
    total += sum(10 ** (arrival.power_dbm / 10) for arrival in arrivals)
    observation = world.observe([10, 10], 0)
    assert observation.interference_dbm == pytest.approx(10 * np.log10(total))
    expected = beam_power_dbm([float(arrival.power_dbm) for arrival in arrivals],
                             [float(arrival.bearing_deg) for arrival in arrivals],
                             observation.directional_scan.azimuth_deg, config.electromagnetic.background_dbm,
                             config.sensor.beamwidth_deg, config.sensor.front_back_db)
    np.testing.assert_allclose(observation.directional_scan.power_dbm, expected)
    assert set(asdict(observation)) == {"position_m", "time_seconds", "interference_dbm", "directional_scan"}


def test_obstructed_direct_signal_can_leave_a_dominant_reflection_peak():
    config = reflection_config()
    blocker = {"id": "direct_blocker", "footprint_m": [[14,9],[16,9],[16,11],[14,11]],
               "height_m": 8, "loss_db": 40}
    config = replace(config, map=replace(config.map, buildings=(*config.map.buildings, blocker)))
    arrivals = paths(config)
    strongest = max(arrivals, key=lambda arrival: arrival.power_dbm)
    assert strongest.kind == "wall"
    assert strongest.bearing_deg == pytest.approx(np.degrees(np.arctan2(10, 5)))
    observation = ElectromagneticWorld(config).observe([10, 10], 0)
    peak = max(observation.directional_scan.peaks(), key=lambda item: item[1])
    assert peak[0] == pytest.approx(60)


def test_multiple_sources_and_birth_add_power_with_background_exactly_once():
    base = reflection_config()
    second = {**base.electromagnetic.fields[0], "center_m": [35, 15], "start_seconds": 5, "eirp_dbm": -8}
    both = replace(base, electromagnetic=replace(base.electromagnetic, fields=(*base.electromagnetic.fields, second)))
    only_second = replace(base, electromagnetic=replace(base.electromagnetic, fields=(second,)))
    first_world, second_world, both_world = map(ElectromagneticWorld, (base, only_second, both))
    points = [[10,10], [30,12], [50,50]]
    np.testing.assert_allclose(both_world.truth(points, 4), first_world.truth(points, 4))
    expected = (10 ** (np.asarray(first_world.truth(points, 8)) / 10)
                + 10 ** (np.asarray(second_world.truth(points, 8)) / 10)
                - 10 ** (base.electromagnetic.background_dbm / 10))
    np.testing.assert_allclose(both_world.truth(points, 8), 10 * np.log10(expected))


def test_legacy_replay_without_new_fields_retains_log_distance_semantics():
    from uav_em.visualization import config_from_result

    legacy = ProjectConfig().to_dict()
    for name in ("propagation_mode", "relative_permittivity", "conductivity_s_per_m"):
        legacy["electromagnetic"].pop(name)
    restored = config_from_result({"schema_version": 2, "evaluation_only_config": legacy})
    assert restored.electromagnetic.propagation_mode == "log_distance"
    assert restored.electromagnetic.path_loss_exponent == 2.2
    points = [[40, 300], [400, 350]]
    np.testing.assert_allclose(ElectromagneticWorld(restored).truth(points, 25),
                               ElectromagneticWorld(ProjectConfig()).truth(points, 25))


def test_vectorized_moving_source_matches_pointwise_truth_and_public_prediction():
    config = reflection_config()
    source = {name: config.electromagnetic.fields[0][name]
              for name in ("center_m", "eirp_dbm", "height_m", "velocity_mps")}
    hypothesis = parse_source_hypothesis({"reference_seconds": 0, "sources": [source]}, [60, 60])
    model, world = PropagationModel.from_config(config), ElectromagneticWorld(config)
    points, times = np.array([[10,10],[12,14],[35,25]]), np.array([0,3,9])
    actual = predict_power(hypothesis, points, times, config.electromagnetic.background_dbm, model)
    np.testing.assert_allclose(actual, [world.truth(point, now) for point, now in zip(points, times)])
    belief = ElectromagneticBelief(config, world.geometry)
    observation = world.observe([10, 10], 0)
    assert belief.source_tracker.bearing_check(hypothesis, observation)["mean_square_deg2"] == pytest.approx(0)


def test_unseen_source_scripts_do_not_enter_context_or_public_model():
    base = reflection_config()
    source = {**base.electromagnetic.fields[0], "center_m": [50, 50], "start_seconds": 12}
    changed = replace(base, electromagnetic=replace(base.electromagnetic, fields=(source,)))
    contexts = []
    for config in (base, changed):
        world = ElectromagneticWorld(config)
        belief = ElectromagneticBelief(config, world.geometry)
        belief.update(Measurement((10, 10), 0, -95))
        context = build_context([10,10], [50,10], [], belief.snapshot(0), config, 100, [], [], None)
        context["source_inference"] = belief.source_tracker.summary(0)
        contexts.append(context)
    assert contexts[0] == contexts[1]
    assert contexts[0]["source_inference"]["recommended_source_model"] == "single_reflection"


@pytest.mark.parametrize("changes", [
    {"propagation_mode": "full_wave"}, {"relative_permittivity": 0.5},
    {"relative_permittivity": True}, {"conductivity_s_per_m": -1},
    {"conductivity_s_per_m": float("nan")}, {"path_loss_exponent": 2.2},
])
def test_invalid_physical_parameters_are_rejected(changes):
    config = reflection_config()
    with pytest.raises(ValueError):
        replace(config, electromagnetic=replace(config.electromagnetic, **changes)).validate()


def test_primary_profile_uses_local_dynamic_sources_and_point_power():
    from pathlib import Path

    config = load_config(Path(__file__).resolve().parents[1] / "configs/local_power.json")
    assert config.electromagnetic.propagation_mode == "single_reflection"
    assert config.sensor.mode == "power_only"
    assert any(source["start_seconds"] > 0 for source in config.electromagnetic.fields)
    assert any(np.linalg.norm(source["velocity_mps"]) > 0 for source in config.electromagnetic.fields)
