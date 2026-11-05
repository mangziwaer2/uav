from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from uav_em.environment import Geometry, ElectromagneticWorld
from uav_em.map_learning.scenarios import ScenarioSettings, generate_scenario, public_geometry, free_graph, geometry_route
from uav_em.map_learning.data import (CollectionSettings, collect_episode, generate_dataset, history_window,
                                      temporal_target, CHANNELS)


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
    monkeypatch.setattr('uav_em.map_learning.scenarios.free_graph', lambda geometry: {})
    with pytest.raises(ValueError, match='after 2 attempts'):
        generate_scenario(1, ScenarioSettings(building_count=0, max_attempts=2))


def test_random_building_count_stays_in_requested_range():
    settings = ScenarioSettings(columns=16, rows=12, building_count=3, min_building_count=1,
                                horizon_seconds=12, require_goal_reachable_in_time=False)
    counts = [generate_scenario(seed, settings)[1]['building_count'] for seed in range(5)]
    assert all(1 <= count <= 3 for count in counts)
    assert len(set(counts)) > 1


@pytest.fixture
def small_scenario():
    return generate_scenario(10, ScenarioSettings(columns=12, rows=10, building_count=0,
                                                 max_sources=0, horizon_seconds=12,
                                                 require_goal_reachable_in_time=False))[0]


@pytest.mark.parametrize('policy', ['goal', 'coverage', 'revisit'])
def test_collect_only_observed_finite_maps_and_safe_routes(small_scenario, policy):
    settings = CollectionSettings(step_seconds=1, budget=5)
    arrays, meta = collect_episode(small_scenario, policy, settings, flight_seed=30)
    assert arrays['maps'].shape[1:] == (len(CHANNELS), 10, 12)
    assert np.isfinite(arrays['maps']).all()
    assert np.all(np.diff(arrays['times']) > 0)
    assert arrays['times'][-1] <= 12 + 1e-8
    assert np.all(np.diff(arrays['energy_wh']) <= 0)
    assert arrays['energy_wh'][-1] >= small_scenario.aircraft.reserve_energy_wh
    geometry = public_geometry(small_scenario)
    assert all(geometry.free(point) for point in arrays['positions'])
    unknown = arrays['maps'][:, 1] == 0
    assert np.all(arrays['maps'][:, 0][unknown] == 0)
    assert np.all(arrays['maps'][:, 3][unknown] == 0)
    assert not meta['is_llm_experiment']
    assert len(meta['observations']) == len(arrays['times'])
    world = ElectromagneticWorld(replace(small_scenario, seed=30))
    for observation in meta['observations']:
        measurement = world.observe(observation['position_m'], observation['time_seconds'])
        assert measurement.interference_dbm == observation['interference_dbm']


def test_future_hidden_source_does_not_change_collection(small_scenario):
    source = {'center_m': [100, 60], 'height_m': 1.5, 'eirp_dbm': 10,
              'velocity_mps': [0, 0], 'start_seconds': 20, 'end_seconds': 100}
    future = replace(small_scenario, electromagnetic=replace(small_scenario.electromagnetic, fields=(source,)))
    first, _ = collect_episode(small_scenario, 'coverage', CollectionSettings(budget=5), 40)
    second, _ = collect_episode(future, 'coverage', CollectionSettings(budget=5), 40)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_revisit_includes_same_position_at_distinct_times(small_scenario):
    config = replace(small_scenario, simulation=replace(small_scenario.simulation, max_seconds=45))
    arrays, _ = collect_episode(config, 'revisit', CollectionSettings(budget=5), 42)
    positions = arrays['positions']
    assert any(np.linalg.norm(point - positions[index + 1:], axis=1).min(initial=1000) < 1e-6
               for index, point in enumerate(positions[:-1]))
    assert np.max(arrays['maps'][:, 3]) > 0


def test_history_is_causal_and_pads_unavailable_frames():
    window = history_window(np.array([0, 2, 4, 6, 8, 10]), 2, frames=4, spacing_seconds=3)
    assert window['indices'] == [-1, -1, 0, 2]
    assert window['offset_seconds'] == [None, None, -4.0, 0.0]
    assert all(index <= 2 for index in window['indices'])
    duplicates = history_window([0, 10, 20], 2, frames=4, spacing_seconds=2)
    assert len(set(index for index in duplicates['indices'] if index >= 0)) == sum(duplicates['valid'])
    with pytest.raises(ValueError):
        history_window([0, 0], 1)


def test_temporal_targets_never_treat_unknown_as_power_change():
    older, newer = np.zeros((6, 2, 2)), np.zeros((6, 2, 2))
    older[1, 0, 0] = older[1, 1, 0] = 1
    newer[1, 0, 0] = newer[1, 0, 1] = 1
    newer[0] = 0.2
    target = temporal_target(older, newer, [0, 0, 2, 2])
    assert target['power_delta_db'] == [[18, None], [None, None]]
    assert target['newly_supported'] == [[0, 1], [0, 0]]
    assert target['lost_support'] == [[0, 0], [1, 0]]
    assert target['interpretation'] == 'belief_change_not_source_motion'


@pytest.fixture
def tiny_dataset(tmp_path):
    root = tmp_path / 'data'
    manifest = generate_dataset(root, scene_count=3, seed=90,
                                scenario=ScenarioSettings(columns=10, rows=8, building_count=0,
                                                          max_sources=1, horizon_seconds=6,
                                                          require_goal_reachable_in_time=False),
                                collection=CollectionSettings(step_seconds=2, budget=3, task_stride=1,
                                                              window_spacing_seconds=2, roi_size=4))
    return root, manifest


def test_dataset_split_provenance_and_temporal_tasks(tiny_dataset):
    root, manifest = tiny_dataset
    assert len(manifest['episodes']) == 9
    assert {item['split'] for item in manifest['scenarios']} == {'train', 'validation', 'test'}
    for scene in manifest['scenarios']:
        assert {item['split'] for item in manifest['episodes'] if item['scenario_id'] == scene['id']} == {scene['split']}
    tasks = [json.loads(line) for line in (root / 'temporal_tasks.jsonl').read_text(encoding='utf-8').splitlines()]
    assert tasks
    assert all(max(task['window']['indices']) == task['anchor'] for task in tasks)
    for episode in manifest['episodes']:
        public = json.loads((root / 'episodes' / f'{episode["id"]}.json').read_text(encoding='utf-8'))
        assert 'fields' not in public['public_propagation']
        assert 'source_count' not in public
        with np.load(root / episode['path'], allow_pickle=False) as arrays:
            assert set(arrays.files) == {'maps', 'times', 'positions', 'energy_wh', 'resolution_m'}
    with pytest.raises(FileExistsError):
        generate_dataset(root, scene_count=3)


@pytest.mark.parametrize('script', ['generate_map_dataset.py', 'audit_map_dataset.py',
                                    'split_map_dataset.py', 'preview_map_dataset.py',
                                    'train_map_autoencoder.py', 'inspect_map_encoding.py'])
def test_map_cli_help_without_root_directory_or_torch_import(script):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'tools' / script), '--help'], cwd=root / 'tools',
                            capture_output=True, text=True, encoding='utf-8', timeout=30,
                            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    assert result.returncode == 0, result.stderr
