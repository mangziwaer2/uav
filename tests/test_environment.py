from dataclasses import replace

import numpy as np
import pytest

from uav_em.belief import ElectromagneticBelief
from uav_em.config import ProjectConfig
from uav_em.environment import ElectromagneticWorld, Measurement


def test_observation_is_single_point_and_snapshot_only_uses_measurements():
    config = ProjectConfig()
    world = ElectromagneticWorld(config)
    belief = ElectromagneticBelief(config, world.geometry)
    before = belief.snapshot(30)
    np.testing.assert_allclose(before.estimate_dbm, config.electromagnetic.background_dbm)
    assert not before.supported.any()
    measurement = world.observe(config.map.start_m, 30)
    belief.update(measurement)
    after = belief.snapshot(30)
    assert after.supported.any() and not after.supported.all()
    assert belief.observation_count == 1
    assert len(measurement.position_m) == 2


def test_unseen_truth_changes_do_not_change_belief():
    config = ProjectConfig()
    other = replace(config, electromagnetic=replace(config.electromagnetic, fields=()))
    first = ElectromagneticWorld(config)
    second = ElectromagneticWorld(other)
    first_belief = ElectromagneticBelief(config, first.geometry)
    second_belief = ElectromagneticBelief(other, second.geometry)
    measurement = Measurement((40, 300), 5, 28)
    first_belief.update(measurement)
    second_belief.update(measurement)
    np.testing.assert_array_equal(first_belief.snapshot(10).estimate_dbm,
                                  second_belief.snapshot(10).estimate_dbm)
    assert first.truth((410, 390), 10) != second.truth((410, 390), 10)


def test_stale_and_future_observations_do_not_support_current_map():
    config = ProjectConfig()
    world = ElectromagneticWorld(config)
    belief = ElectromagneticBelief(config, world.geometry)
    belief.update(Measurement((40, 300), 5, 20))
    assert not belief.snapshot(4).supported.any()
    assert belief.snapshot(5).supported.any()
    assert not belief.snapshot(50).supported.any()
    with pytest.raises(ValueError):
        belief.update(Measurement((40, 300), 4, 20))


def test_geometry_checks_entire_segment_not_only_endpoints():
    geometry = ElectromagneticWorld(ProjectConfig()).geometry
    assert geometry.free((200, 300)) and geometry.free((400, 300))
    assert not geometry.edge_free((200, 300), (400, 300))
    assert geometry.edge_free((200, 400), (400, 400))
    assert not geometry.free((-1, 0))


def test_interference_addition_uses_linear_power():
    config = ProjectConfig()
    field = config.electromagnetic.fields[0]
    one = ElectromagneticWorld(replace(config, electromagnetic=replace(config.electromagnetic, fields=(field,))))
    two = ElectromagneticWorld(replace(config, electromagnetic=replace(config.electromagnetic, fields=(field, field))))
    difference = two.truth(field["center_m"], 10) - one.truth(field["center_m"], 10)
    assert difference == pytest.approx(3.01, abs=0.02)


def test_dynamic_field_is_inactive_before_birth():
    config = ProjectConfig()
    field = config.electromagnetic.fields[1]
    world = ElectromagneticWorld(replace(config, electromagnetic=replace(config.electromagnetic, fields=(field,))))
    point = field["center_m"]
    assert world.truth(point, 24) > world.truth(point, 23) + 10


def test_background_has_no_station_distance_gradient():
    config = ProjectConfig()
    world = ElectromagneticWorld(replace(config, electromagnetic=replace(config.electromagnetic, fields=())))
    np.testing.assert_allclose(world.truth(world.geometry.grid(), 0), config.electromagnetic.background_dbm)


def test_integer_background_configuration_preserves_fractional_measurements():
    config = ProjectConfig()
    config = replace(config, electromagnetic=replace(config.electromagnetic, background_dbm=-100))
    world = ElectromagneticWorld(config)
    belief = ElectromagneticBelief(config, world.geometry)
    belief.update(Measurement((40, 300), 0, -72.35))
    assert belief.snapshot(0).at((40, 300))[0] == pytest.approx(-72.35)


def test_static_truth_does_not_change_when_vehicle_observes():
    config = ProjectConfig()
    field = config.electromagnetic.fields[0]
    world = ElectromagneticWorld(replace(config, electromagnetic=replace(config.electromagnetic, fields=(field,))))
    before = world.truth(world.geometry.grid(), 0)
    world.observe((40, 300), 10)
    np.testing.assert_array_equal(before, world.truth(world.geometry.grid(), 20))


def test_source_motion_and_disappearance_change_truth_only_as_scripted():
    config = ProjectConfig()
    field = {"center_m": [100, 100], "sigma_m": 20, "peak_dbm": -50,
             "start_seconds": 2, "end_seconds": 8, "velocity_mps": [20, 0]}
    world = ElectromagneticWorld(replace(config, electromagnetic=replace(config.electromagnetic, fields=(field,))))
    assert world.truth((100, 100), 2) > world.truth((100, 100), 6)
    assert world.truth((180, 100), 6) > world.truth((100, 100), 6)
    assert world.truth((180, 100), 8) == pytest.approx(config.electromagnetic.background_dbm)


def test_display_source_positions_follow_same_birth_motion_and_death_as_truth():
    config = ProjectConfig()
    source = {"center_m": [100, 100], "sigma_m": 20, "peak_dbm": -50,
              "start_seconds": 2, "end_seconds": 8, "velocity_mps": [20, 0]}
    world = ElectromagneticWorld(replace(config, electromagnetic=replace(config.electromagnetic, fields=(source,))))
    assert world.source_positions_for_evaluation(1) == []
    assert world.source_positions_for_evaluation(6) == [{"id": "S1", "position_m": [180, 100]}]
    assert world.source_positions_for_evaluation(8) == []


@pytest.mark.parametrize("change", [
    lambda config: replace(config, simulation=replace(config.simulation, step_seconds=0)),
    lambda config: replace(config, simulation=replace(config.simulation, max_advisor_calls=True)),
    lambda config: replace(config, aircraft=replace(config.aircraft, reserve_energy_wh=101)),
    lambda config: replace(config, belief=replace(config.belief, support_radius_m=-1)),
])
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        change(ProjectConfig()).validate()
