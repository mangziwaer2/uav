from dataclasses import replace
import json

import numpy as np
import pytest

from uav_em.environment import Geometry, ElectromagneticWorld
from uav_em.scenarios import ScenarioSettings, generate_scenario, public_geometry, free_graph, geometry_route


@pytest.mark.parametrize('seed', [0, 1, 7, 18])
def test_random_scenarios_reproducible_connected_and_physically_bounded(seed):
    settings = ScenarioSettings(columns=16, rows=12, building_count=2, horizon_seconds=20,
                                require_goal_reachable_in_time=False)
    config, report = generate_scenario(seed, settings)
    again, repeated = generate_scenario(seed, settings)
    assert config.to_dict() == again.to_dict()
    assert report == repeated
    assert report['connected_fraction'] >= settings.min_connected_fraction
    assert report['free_fraction'] >= settings.min_free_fraction
    assert report['building_count'] == settings.building_count
    geometry = public_geometry(config)
    route = geometry_route(geometry, free_graph(geometry), config.map.start_m, config.map.goal_m)
    assert route
    assert all(geometry.edge_free(first, second) for first, second in zip(route, route[1:]))
    ground = Geometry(config.map, 0, 1.5, 0)
    for source in config.electromagnetic.fields:
        endpoint = np.asarray(source['center_m']) + np.asarray(source['velocity_mps']) * (
            source['end_seconds'] - source['start_seconds'])
        assert ground.edge_free(source['center_m'], endpoint)
    assert config.electromagnetic.propagation_mode == 'single_reflection'
    assert np.isfinite(ElectromagneticWorld(config).truth(np.asarray(route), 10)).all()
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('changes', [{'columns': 7}, {'rows': 0}, {'max_sources': 4},
                                    {'horizon_seconds': float('nan')}, {'min_free_fraction': 1.1},
                                    {'min_building_count': -1}, {'min_building_count': 9},
                                    {'min_building_count': True}])
def test_invalid_random_settings_rejected(changes):
    with pytest.raises(ValueError):
        generate_scenario(7, replace(ScenarioSettings(), **changes))


def test_generation_rejection_is_bounded(monkeypatch):
    monkeypatch.setattr('uav_em.scenarios.free_graph', lambda geometry: {})
    with pytest.raises(ValueError, match='after 2 attempts'):
        generate_scenario(1, ScenarioSettings(building_count=0, max_attempts=2))


def test_random_building_count_stays_in_requested_range():
    settings = ScenarioSettings(columns=16, rows=12, building_count=3, min_building_count=1,
                                horizon_seconds=12, require_goal_reachable_in_time=False)
    counts = [generate_scenario(seed, settings)[1]['building_count'] for seed in range(5)]
    assert all(1 <= count <= 3 for count in counts)
    assert len(set(counts)) > 1


