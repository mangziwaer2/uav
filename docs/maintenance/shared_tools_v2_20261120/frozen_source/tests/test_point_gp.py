"""Regression against pre-cleanup numerics, plus causal/sensing boundary checks."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from uav_em.config import BeliefConfig, load_config
from uav_em.environment import Geometry, Measurement, ElectromagneticWorld
from uav_em.point_gp import FrozenGP, PointGP
from uav_em.motion import advance_route


def geometry():
    c = load_config('configs/local_power.json')
    return c, Geometry(replace(c.map, buildings=(), obstacles_m=()), 0., c.aircraft.altitude_m)


@pytest.mark.parametrize('index', [0, 1, 2])
def test_gp_matches_pre_cleanup_empty_supported_unknown_and_expired_values(index):
    record = json.loads(Path('tests/fixtures/point_gp_reference.json').read_text())
    c, g = geometry()
    row = record['cases'][index]
    actual = FrozenGP(row['points'], g, row['now'], BeliefConfig(**record['settings']), -100.).predict(row['xy'], row['times'])
    np.testing.assert_allclose(actual[0], row['expected'][0], rtol=0., atol=1e-12)
    np.testing.assert_allclose(actual[1], row['expected'][1], rtol=0., atol=1e-12)
    np.testing.assert_array_equal(actual[2], row['expected'][2])


def test_freeze_cannot_use_future_history_or_query_before_its_version():
    c, g = geometry()
    estimator = PointGP(g, c.belief)
    estimator.observe(Measurement((40., 40.), 1., -85.))
    with pytest.raises(ValueError):
        estimator.freeze(0.)
    _, gp = estimator.freeze(1.)
    with pytest.raises(ValueError):
        gp.predict([[40., 40.]], [0.])


def test_gp_only_uses_supplied_points_and_does_not_have_world_capability():
    c, g = geometry()
    first, second = PointGP(g, c.belief), PointGP(g, c.belief)
    measurement = Measurement((40., 40.), 0., -85.)
    first.observe(measurement)
    second.observe(measurement)
    a, _ = first.freeze(0.)
    b, _ = second.freeze(0.)
    np.testing.assert_array_equal(a.estimate_dbm, b.estimate_dbm)
    assert not a.supported[-1, -1]
    assert a.estimation_details['calibrated'] is False
    assert set(measurement.__dataclass_fields__) == {'position_m', 'time_seconds', 'interference_dbm'}


def test_environment_rejects_retired_directional_mode():
    c, _ = geometry()
    with pytest.raises(ValueError, match='point power'):
        ElectromagneticWorld(replace(c, sensor=replace(c.sensor, mode='power_bearing')))


def test_motion_preserves_crossed_corner_and_remaining_route():
    route = [[0., 0.], [3., 0.], [3., 4.]]
    remaining, traversed = advance_route(route, 5.)
    assert traversed == [[0., 0.], [3., 0.], [3., 2.]]
    assert remaining == [[3., 2.], [3., 4.]]
    assert route == [[0., 0.], [3., 0.], [3., 4.]]
