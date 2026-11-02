from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from uav_em.belief import ElectromagneticBelief
from uav_em.config import ProjectConfig
from uav_em.context import build_context
from uav_em.decision_log import render_decision_log
from uav_em.environment import ElectromagneticWorld, Measurement
from uav_em.observation_monitor import ObservationMonitor
from uav_em.propagation import source_received_dbm
from uav_em.sensing import DirectionalReceiver
from uav_em.simulation import simulate
from uav_em.visualization import render


def emitter(**changes):
    return {"center_m": [300.0, 100.0], "eirp_dbm": -2.0, "height_m": 1.5,
            "start_seconds": 0.0, "end_seconds": 100.0, "velocity_mps": [0.0, 0.0], **changes}


@pytest.fixture
def config():
    base = ProjectConfig()
    return replace(base, map=replace(base.map, obstacles_m=()),
                   sensor=replace(base.sensor, noise_db=0, bearing_error_deg=0),
                   electromagnetic=replace(base.electromagnetic, fields=(emitter(),), measurement_noise_db=0))


def test_log_distance_loss_and_frequency_follow_analytic_formula(config):
    source = emitter(height_m=config.aircraft.altitude_m)
    first = source_received_dbm([200, 100], source["center_m"], source, config)
    second = source_received_dbm([100, 100], source["center_m"], source, config)
    assert first - second == pytest.approx(10 * config.electromagnetic.path_loss_exponent * np.log10(2))
    higher = replace(config, electromagnetic=replace(config.electromagnetic, carrier_frequency_hz=4.8e9))
    assert first - source_received_dbm([200, 100], source["center_m"], source, higher) == pytest.approx(20 * np.log10(2))
    reference = source_received_dbm([299, 100], source["center_m"], source, config)
    assert reference == pytest.approx(source["eirp_dbm"] - 20 * np.log10(4 * np.pi * 2.4e9 / 299792458))


def test_height_and_reference_distance_avoid_singular_near_field(config):
    source = emitter()
    first = source_received_dbm(source["center_m"], source["center_m"], source, config)
    higher = replace(config, aircraft=replace(config.aircraft, altitude_m=60))
    assert np.isfinite(first)
    assert source_received_dbm(source["center_m"], source["center_m"], source, higher) < first
    coplanar = emitter(height_m=config.aircraft.altitude_m)
    assert np.isfinite(source_received_dbm(coplanar["center_m"], coplanar["center_m"], coplanar, config))


def test_obstacle_loss_uses_3d_uninflated_prisms_not_flight_clearance(config):
    source = emitter()
    blocked = replace(config, map=replace(config.map, obstacles_m=((180, 90, 200, 110),)))
    first = source_received_dbm([100, 100], source["center_m"], source, config)
    second = source_received_dbm([100, 100], source["center_m"], source, blocked)
    assert first - second == pytest.approx(config.electromagnetic.obstacle_loss_db)
    inflated = replace(blocked, aircraft=replace(blocked.aircraft, clearance_m=80))
    assert source_received_dbm([100, 100], source["center_m"], source, inflated) == second
    above = replace(blocked, aircraft=replace(blocked.aircraft, altitude_m=100))
    clear_above = replace(above, map=config.map)
    assert source_received_dbm([100, 100], source["center_m"], source, above) == pytest.approx(
        source_received_dbm([100, 100], source["center_m"], source, clear_above))


def test_multiple_obstacles_accumulate_loss_and_vectorization_matches_scalars(config):
    config = replace(config, map=replace(config.map, obstacles_m=((180, 90, 190, 110), (220, 90, 230, 110))))
    source = emitter()
    points = np.array([[[100, 100], [100, 150]], [[100, 200], [250, 100]]])
    powers = source_received_dbm(points, source["center_m"], source, config)
    for row in range(2):
        for column in range(2):
            assert powers[row, column] == pytest.approx(source_received_dbm(
                points[row, column], source["center_m"], source, config))
    clear = replace(config, map=replace(config.map, obstacles_m=()))
    assert source_received_dbm([100, 100], source["center_m"], source, clear) - powers[0, 0] == pytest.approx(12)


def test_remote_emitter_is_detected_before_entering_strong_interference(config):
    world = ElectromagneticWorld(config)
    measurement = world.observe([100, 100], 0)
    assert measurement.interference_dbm < config.electromagnetic.interference_reference_dbm - 10
    assert measurement.directional_scan.peaks()[0][0] == 0
    assert measurement.directional_scan.power_dbm[0] > config.sensor.detection_threshold_dbm
    encoded = json.dumps(asdict(measurement))
    for forbidden in ("center_m", "eirp_dbm", "height_m", "velocity_mps", "start_seconds", "source_id", "range_m"):
        assert forbidden not in encoded


def test_detection_is_power_and_occlusion_dependent_not_a_fixed_radius(config):
    assert ElectromagneticWorld(config).observe([100, 100], 0).directional_scan.peaks()
    weak = replace(config, electromagnetic=replace(config.electromagnetic, fields=(emitter(eirp_dbm=-20),)))
    assert not ElectromagneticWorld(weak).observe([100, 100], 0).directional_scan.peaks()
    blocked = replace(config, map=replace(config.map, obstacles_m=((180, 90, 200, 110),)))
    assert not ElectromagneticWorld(blocked).observe([100, 100], 0).directional_scan.peaks()


def test_scan_sums_all_sources_before_detection_and_does_not_resolve_coincident_bearings(config):
    receiver = DirectionalReceiver(config)
    one = receiver.scan([-90], [0])
    two = receiver.scan([-90, -90], [0, 0])
    expected = 10 * np.log10(10 ** -10 + 2 * 10 ** -9)
    assert two.power_dbm[0] == pytest.approx(expected)
    assert len(two.peaks()) == 1
    assert two.power_dbm[0] > one.power_dbm[0]
    separated = receiver.scan([-90, -90], [0, 180])
    assert [angle for angle, _ in separated.peaks()] == [0, 180]


def test_bearing_wraparound_and_front_back_suppression(config):
    scan = DirectionalReceiver(config).scan([-80], [359])
    assert scan.peaks()[0][0] == 0
    assert scan.power_dbm[0] > scan.power_dbm[12] + 15


def test_directly_overhead_source_does_not_invent_an_azimuth(config):
    measurement = ElectromagneticWorld(config).observe([300, 100], 0)
    assert measurement.interference_dbm > config.electromagnetic.interference_reference_dbm
    assert not measurement.directional_scan.peaks()
    np.testing.assert_allclose(measurement.directional_scan.power_dbm,
                               measurement.directional_scan.power_dbm[0])


def test_flat_strong_sidelobes_are_not_reported_as_separate_sources(config):
    config = replace(config, sensor=replace(config.sensor, noise_db=0.2))
    scan = DirectionalReceiver(config).scan([-40], [0])
    assert [angle for angle, _ in scan.peaks()] == [0]


def test_sensor_noise_is_reproducible_and_independent_of_point_noise(config):
    config = replace(config, sensor=replace(config.sensor, noise_db=1, bearing_error_deg=3),
                     electromagnetic=replace(config.electromagnetic, measurement_noise_db=0.5))
    first, second = ElectromagneticWorld(config), ElectromagneticWorld(config)
    expected_random = np.random.default_rng(config.seed)
    scans = []
    for now in range(3):
        measurement = first.observe([100, 100], now)
        assert measurement == second.observe([100, 100], now)
        assert measurement.interference_dbm == pytest.approx(first.truth([100, 100], now) + expected_random.normal(0, 0.5))
        scans.append(measurement.directional_scan.power_dbm)
    assert scans[0] != scans[1]


def test_inactive_or_removed_emitter_has_no_directional_detection(config):
    config = replace(config, electromagnetic=replace(config.electromagnetic,
                     fields=(emitter(start_seconds=3, end_seconds=5),)))
    world = ElectromagneticWorld(config)
    assert not world.observe([100, 100], 2).directional_scan.peaks()
    assert world.observe([100, 100], 3).directional_scan.peaks()
    assert not world.observe([100, 100], 5).directional_scan.peaks()


def test_receiver_sensitivity_below_background_does_not_make_background_a_source(config):
    config = replace(config, sensor=replace(config.sensor, detection_threshold_dbm=-120),
                     electromagnetic=replace(config.electromagnetic, fields=()))
    measurement = ElectromagneticWorld(config).observe([100, 100], 0)
    assert not measurement.directional_scan.peaks()
    assert measurement.directional_scan.detection_threshold_dbm >= config.electromagnetic.background_dbm + 3


def test_out_of_order_measurements_at_different_nodes_cannot_reorder_scan_history(config):
    world = ElectromagneticWorld(config)
    belief = ElectromagneticBelief(config, world.geometry)
    belief.update(world.observe([100, 100], 5))
    with pytest.raises(ValueError, match="out-of-order"):
        belief.update(world.observe([140, 100], 4))
    assert belief.observation_count == 1


def test_moving_source_changes_observed_direction_without_returning_velocity(config):
    config = replace(config, electromagnetic=replace(config.electromagnetic,
                     fields=(emitter(velocity_mps=[0, 10]),)))
    world = ElectromagneticWorld(config)
    assert world.observe([200, 100], 0).directional_scan.peaks()[0][0] == 0
    assert world.observe([200, 100], 10).directional_scan.peaks()[0][0] == 45


def test_directional_observations_do_not_mark_remote_cells_as_measured(config):
    world = ElectromagneticWorld(config)
    belief = ElectromagneticBelief(config, world.geometry)
    measurement = world.observe([100, 100], 0)
    belief.update(measurement)
    snapshot = belief.snapshot(0)
    assert snapshot.directional_observations[-1]["peaks_deg_dbm"]
    assert not snapshot.at([300, 100])[2]
    plain = ElectromagneticBelief(config, world.geometry)
    plain.update(replace(measurement, directional_scan=None))
    np.testing.assert_array_equal(snapshot.supported, plain.snapshot(0).supported)
    np.testing.assert_array_equal(snapshot.estimate_dbm, plain.snapshot(0).estimate_dbm)
    assert not belief.snapshot(40).directional_observations
    assert not belief.snapshot(-1).directional_observations


def test_context_keeps_time_stamped_bearing_history_but_not_truth(config):
    world = ElectromagneticWorld(config)
    belief = ElectromagneticBelief(config, world.geometry)
    for now in range(10):
        belief.update(world.observe([100, 100 + now], now))
    context = build_context([100, 109], config.map.goal_m, [], belief.snapshot(9), config, 99, [], [], None)
    sensing = context["directional_sensing"]
    assert len(sensing["recent_scans"]) == 6
    assert sensing["recent_scans"][0]["time_seconds"] == 0
    assert sensing["recent_scans"][-1]["time_seconds"] == 9
    assert sensing["measured_range_available"] is False
    assert sensing["remote_field_cells_measured"] is False
    assert "eirp_dbm" not in json.dumps(context)


@pytest.mark.parametrize("changes", [
    {"azimuth_deg": (0, 90, 180)}, {"power_dbm": (float("nan"),) * 24},
    {"peak_indices": (24,)}, {"peak_indices": (True,)}, {"beamwidth_deg": 0},
    {"peak_indices": (12,)},
])
def test_invalid_scan_is_rejected_before_belief_mutation(config, changes):
    world = ElectromagneticWorld(config)
    belief = ElectromagneticBelief(config, world.geometry)
    measurement = world.observe([100, 100], 0)
    with pytest.raises(ValueError):
        belief.update(replace(measurement, directional_scan=replace(measurement.directional_scan, **changes)))
    assert belief.observation_count == 0
    assert not belief.directional_measurements


def test_sustained_rise_warns_even_when_each_increment_is_less_than_innovation_threshold(config):
    monitor = ObservationMonitor(config)
    initial = Measurement((100, 100), 0, -99)
    monitor.observe(initial)
    monitor.reviewed(initial)
    warnings = [monitor.observe(Measurement((100 + now, 100), now, power))
                for now, power in enumerate([-97, -94, -90], start=1)]
    assert all(abs(power) < config.simulation.innovation_trigger_db for power in (2, 3, 4))
    assert "power_rising" in warnings[-1]["reasons"]
    assert warnings[-1]["power_rise_db"] == 9
    monitor.reviewed(Measurement((103, 100), 3, -90))
    assert not monitor.observe(Measurement((104, 100), 4, -90))["reasons"]


def test_reference_crossing_does_not_need_a_large_innovation(config):
    monitor = ObservationMonitor(config)
    initial = Measurement((100, 100), 0, -81)
    monitor.observe(initial)
    monitor.reviewed(initial)
    assert "reference_crossed" in monitor.observe(Measurement((100, 100), 1, -79))["reasons"]


def test_full_episode_reviews_detectable_remote_source_before_strong_exposure(config):
    config = replace(config, map=replace(config.map, columns=31, rows=11, start_m=(20, 100), goal_m=(580, 100)),
                     simulation=replace(config.simulation, max_seconds=50, review_seconds=30),
                     electromagnetic=replace(config.electromagnetic, fields=(emitter(center_m=[400, 100]),)))
    result = simulate(config, initial_planner="astar")
    warnings = [item for item in result["decisions"] if item["trigger"] == "sensing_warning"]
    assert warnings
    first = warnings[0]
    assert first["time_seconds"] < 30
    assert "direction_detected" in first["context"]["sensing_warning"]["reasons"]
    assert first["context"]["latest_observation"]["interference_dbm"] < -80
    assert result["metrics"]["directional_scans"] == result["metrics"]["observations"]
    assert result["metrics"]["directional_detected_scans"] > 0
    assert "范围感知" in render_decision_log(result)
    times = [item["time_seconds"] for item in result["decisions"]]
    assert all(second - first >= 3 - 1e-6 for first, second in zip(times, times[1:]))
    json.dumps(result, allow_nan=False)


def test_replay_draws_logged_bearing_not_hidden_source_direction(config, monkeypatch):
    from pathlib import Path
    from matplotlib.figure import Figure
    from matplotlib.patches import Wedge

    config = replace(config, simulation=replace(config.simulation, max_seconds=0.1))
    result = simulate(config, initial_planner="astar")
    origin = result["frames"][-1]["position_m"]
    result["frames"][-1]["directional_observations"] = [
        {"time_seconds": 0, "position_m": origin, "peaks_deg_dbm": [[270, -90]], "beamwidth_deg": 45}]
    captured = []

    def capture(figure, path, **kwargs):
        for axis in figure.axes[:2]:
            wedges = [patch for patch in axis.patches if isinstance(patch, Wedge)]
            assert len(wedges) == 1
            np.testing.assert_allclose(wedges[0].center, origin)
            assert wedges[0].theta1 == 247.5
            assert wedges[0].theta2 == 292.5
            assert wedges[0].r == pytest.approx(0.22 * 600)
        assert not any(text.get_text().startswith("S") for text in figure.axes[1].texts)
        captured.append(path)

    monkeypatch.setattr(Figure, "savefig", capture)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    render(result, "unused.png")
    assert captured == [Path("unused.png")]


@pytest.mark.parametrize("section, changes", [
    ("sensor", {"azimuth_bins": 3}), ("sensor", {"azimuth_bins": 361}),
    ("sensor", {"azimuth_bins": True}), ("sensor", {"beamwidth_deg": 181}),
    ("sensor", {"bearing_error_deg": -1}), ("sensor", {"detection_threshold_dbm": float("nan")}),
    ("sensor", {"noise_db": float("nan")}), ("aircraft", {"altitude_m": 0}),
    ("electromagnetic", {"carrier_frequency_hz": 0}),
    ("electromagnetic", {"reference_distance_m": 0.01}),
    ("electromagnetic", {"path_loss_exponent": 1}),
    ("electromagnetic", {"obstacle_loss_db": -1}),
    ("electromagnetic", {"fields": (emitter(height_m=-1),)}),
    ("electromagnetic", {"fields": (emitter(sigma_m=10),)}),
    ("simulation", {"warning_window_seconds": 0}),
])
def test_new_physical_and_sensor_configuration_is_validated(config, section, changes):
    with pytest.raises(ValueError):
        replace(config, **{section: replace(getattr(config, section), **changes)}).validate()


def test_power_only_payload_preserves_point_measurements_without_inventing_bearings(config):
    config = replace(config, electromagnetic=replace(config.electromagnetic, measurement_noise_db=0.5))
    plain_config = replace(config, sensor=replace(config.sensor, mode="power_only"))
    directional, plain = ElectromagneticWorld(config), ElectromagneticWorld(plain_config)
    for now in range(3):
        with_bearing = directional.observe([100, 100], now)
        power_only = plain.observe([100, 100], now)
        assert power_only.interference_dbm == with_bearing.interference_dbm
        assert power_only.directional_scan is None
    plain_config = replace(plain_config, simulation=replace(plain_config.simulation, max_seconds=1))
    result = simulate(plain_config, initial_planner="astar")
    sensing = result["decisions"][0]["context"]["directional_sensing"]
    assert sensing["available"] is False
    assert sensing["observation_kind"] == "not_equipped"
    assert sensing["recent_scans"] == []
    assert result["metrics"]["directional_scans"] == 0
    assert result["metrics"]["observations"] > 0
    assert "未配备测向阵列" in render_decision_log(result)
    json.dumps(result, allow_nan=False)


def test_bearing_calibration_uses_elapsed_time_and_keeps_stationary_error_scale(config):
    config = replace(config, sensor=replace(config.sensor, bearing_error_deg=3,
                                           calibration_correlation_seconds=5))
    receiver = DirectionalReceiver(config)
    expected_random = np.random.default_rng(np.random.SeedSequence([config.seed, 1702]))
    expected = expected_random.normal(0, 3)
    receiver.scan([-80], [0], 0)
    assert receiver.calibration_error_deg == pytest.approx(expected)
    for now, elapsed in [(2, 2), (2, 0), (11, 9)]:
        if elapsed:
            correlation = np.exp(-elapsed / 5)
            expected = correlation * expected + expected_random.normal(0, 3 * np.sqrt(1 - correlation ** 2))
        scan = receiver.scan([-80], [0], now)
        assert receiver.calibration_error_deg == pytest.approx(expected)
        assert scan.calibration_correlation_seconds == 5
        assert scan.bearing_error_deg == 3
        assert "calibration_error_deg" not in asdict(scan)
    with pytest.raises(ValueError, match="out-of-order"):
        receiver.scan([-80], [0], 10)


def test_bearing_error_statistics_are_correlated_not_independent_or_unbounded(config):
    config = replace(config, sensor=replace(config.sensor, bearing_error_deg=3,
                                           calibration_correlation_seconds=5))
    receiver = DirectionalReceiver(config)
    errors = np.array([receiver._calibration_error(now) for now in range(5000)])
    assert np.std(errors) == pytest.approx(3, abs=0.25)
    assert np.corrcoef(errors[:-1], errors[1:])[0, 1] == pytest.approx(np.exp(-1 / 5), abs=0.04)


def test_zero_correlation_time_explicitly_selects_independent_calibration_errors(config):
    config = replace(config, sensor=replace(config.sensor, bearing_error_deg=3,
                                           calibration_correlation_seconds=0))
    receiver = DirectionalReceiver(config)
    expected_random = np.random.default_rng(np.random.SeedSequence([config.seed, 1702]))
    for now in [0, 0, 1]:
        receiver.scan([-80], [0], now)
        assert receiver.calibration_error_deg == pytest.approx(expected_random.normal(0, 3))


@pytest.mark.parametrize("peak_indices, expected", [((23, 1), (23,)), ((0, 3), (0, 3))])
def test_unresolved_noisy_angular_peaks_are_suppressed_across_zero_degrees(config, peak_indices, expected):
    class FixedNoise:
        def normal(self, mean, std, size):
            noise = np.zeros(size)
            noise[peak_indices[0]] = 12
            noise[peak_indices[1]] = 11
            return noise

    receiver = DirectionalReceiver(config)
    receiver.random = FixedNoise()
    scan = receiver.scan([], [], 0)
    assert scan.peak_indices == expected
    scan.validate()


def test_correlated_error_metadata_reaches_context_and_readable_log_without_realized_bias(config):
    config = replace(config, simulation=replace(config.simulation, max_seconds=1))
    result = simulate(config, initial_planner="astar")
    sensing = result["decisions"][0]["context"]["directional_sensing"]
    assert sensing["available"] is True
    assert sensing["calibration_correlation_seconds"] == 5
    assert sensing["recent_scans"][0]["minimum_peak_separation_deg"] == config.sensor.beamwidth_deg
    assert "reflections_and_array_IQ_not_simulated" in sensing["model_scope"]
    assert "calibration_error_deg" not in json.dumps(sensing)
    assert "相邻扫描的方位误差相关" in render_decision_log(result)


@pytest.mark.parametrize("changes", [
    {"mode": "radar"}, {"mode": True}, {"mode": None},
    {"calibration_correlation_seconds": -1}, {"calibration_correlation_seconds": float("nan")},
])
def test_sensor_capability_and_correlation_configuration_is_validated(config, changes):
    with pytest.raises(ValueError):
        replace(config, sensor=replace(config.sensor, **changes)).validate()


@pytest.mark.parametrize("correlation", [-1, float("nan")])
def test_invalid_correlation_metadata_is_rejected(config, correlation):
    scan = DirectionalReceiver(config).scan([-80], [0])
    with pytest.raises(ValueError, match="invalid directional"):
        replace(scan, calibration_correlation_seconds=correlation).validate()


def test_source_power_and_bearing_lengths_must_match(config):
    with pytest.raises(ValueError, match="equal lengths"):
        DirectionalReceiver(config).scan([-80, -90], [0])
