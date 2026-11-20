"""Synthetic coverage regressions for the shared GP support convention."""
from dataclasses import replace

import numpy as np
import pytest

from uav_em.config import BeliefConfig, MapConfig
from uav_em.environment import Geometry
from uav_em.joint_prediction import JointSettings, local_prediction
from uav_em.point_gp import FrozenGP, support_reference


def fixture(obstacles=()):
    geometry = Geometry(MapConfig(columns=31, rows=31, resolution_m=20.,
                                  obstacles_m=obstacles, buildings=()))
    belief = replace(BeliefConfig(), support_radius_m=80.)
    history = [dict(position_m=[100., 100.], time_seconds=0., interference_dbm=-85.),
               dict(position_m=[140., 100.], time_seconds=30., interference_dbm=-75.)]
    return FrozenGP(history, geometry, 30., belief, -100.)


def test_expired_nearest_point_cannot_hide_a_fresh_supported_point():
    gp = fixture()
    query, times = [[100., 100.]], [36.]
    local = local_prediction(gp, query, times, JointSettings())
    supported, distance, age, indices = support_reference(gp.model, query, times)
    assert supported.tolist() == local['supported'].tolist() == [True]
    assert indices.tolist() == [1]
    assert distance.tolist() == [40.] and age.tolist() == [6.]
    # The blending distance and age must both refer to the same eligible point.
    sd = local['gp_std_db'][0]
    expected = np.exp(-.5**2)*np.exp(-6./35.)*np.exp(-6./20.)/(1.+(sd/8.)**2)
    assert local['gp_weight'][0] == pytest.approx(expected)
    assert local['gp_weight'][0] > 0.


def test_nearest_point_is_still_used_while_it_remains_fresh():
    gp = fixture()
    supported, distance, age, indices = support_reference(gp.model, [[100., 100.]], [34.])
    assert supported.tolist() == [True] and indices.tolist() == [0]
    assert distance.tolist() == [0.] and age.tolist() == [34.]


@pytest.mark.parametrize('position,time,supported', [
    ([220., 100.], 65., True),
    ([220.001, 100.], 65., False),
    ([220., 100.], 65.001, False),
])
def test_radius_and_age_boundaries_remain_hard(position, time, supported):
    local = local_prediction(fixture(), [position], [time], JointSettings())
    assert bool(local['supported'][0]) is supported
    if supported:
        assert local['gp_weight'][0] > 0.
    else:
        assert local['gp_weight'][0] == 0.
        assert local['gp_mean_dbm'][0] == -100.


def test_nearby_observations_do_not_support_a_blocked_receiver():
    gp = fixture(((115., 95., 125., 105.),))
    local = local_prediction(gp, [[120., 100.]], [36.], JointSettings())
    assert not local['supported'][0] and local['gp_weight'][0] == 0.
    _, distance, age, indices = support_reference(gp.model, [[120., 100.]], [36.])
    assert np.isinf(distance[0]) and np.isinf(age[0]) and indices[0] == -1


def test_empty_prefix_and_empty_queries_do_not_invent_support():
    gp = fixture()
    empty = FrozenGP([], gp.model.geometry, 30., gp.model.settings, -100.)
    local = local_prediction(empty, [[100., 100.]], [36.], JointSettings())
    assert local['supported'].tolist() == [False] and local['gp_weight'].tolist() == [0.]
    local = local_prediction(gp, np.empty((0, 2)), [], JointSettings())
    assert all(value.shape == (0,) for value in local.values())
