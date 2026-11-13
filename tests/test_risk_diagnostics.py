from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.context import build_context
from uav_em.environment import ElectromagneticWorld
from uav_em.risk_diagnostics import received_before_decision, score_pool, select_decisions, summarize
from uav_em.risk_diagnostic_evaluation import evaluate_route


@pytest.fixture
def state():
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/local_power.json')
    cfg = replace(cfg, map=replace(cfg.map, buildings=(), obstacles_m=()),
                  electromagnetic=replace(cfg.electromagnetic, fields=()))
    world = ElectromagneticWorld(cfg)
    belief = ElectromagneticBelief(cfg, world.geometry)
    belief.update(world.observe(cfg.map.start_m, 0))
    snapshot = belief.snapshot(0)
    route = [list(cfg.map.start_m), list(cfg.map.goal_m)]
    context = build_context(route[0], route[-1], route, snapshot, cfg, 100, [], [], None)
    records = [{'id': 'incumbent', 'family': 'incumbent', 'path': route},
               {'id': 'detour', 'family': 'geometric', 'path': [route[0], [200, 150], route[-1]]}]
    return world, snapshot, context, records


def test_state_selection_uses_received_readings_and_merges_duplicate_labels():
    contexts = [{'time_seconds': t, 'latest_observation': {'interference_dbm': p}}
                for t, p in [(0, -90), (20, -85), (35, -75), (45, -77)]]
    selected, missing = select_decisions(contexts)
    assert selected == {1: ['after20'], 2: ['first_observed_reference', 'after30'], 3: ['after40']}
    assert missing == []
    assert select_decisions(contexts[:2])[1] == ['first_observed_reference', 'after30', 'after40']


def test_event_cutoff_excludes_future_measurement_even_at_same_timestamp():
    point = {'event': 'observation', 'time_seconds': 1, 'position_m': [2, 3], 'interference_dbm': -90}
    events = [point, {'event': 'decision', 'index': 0, 'time_seconds': 1}, {**point, 'interference_dbm': -10}]
    assert received_before_decision(events, 0) == [{k: point[k] for k in ('position_m', 'time_seconds', 'interference_dbm')}]
    with pytest.raises(ValueError, match='absent'):
        received_before_decision(events, 1)
    with pytest.raises(ValueError, match='power-only'):
        received_before_decision([{**point, 'directional_scan': {'bearing': 50}}] + events[1:], 0)


def test_fixed_lifetime_remaining_cap_excludes_detour_even_with_energy(state):
    _, snapshot, context, records = state
    direct = np.linalg.norm(np.array(records[0]['path'][-1]) - records[0]['path'][0])
    scored, selection, _ = score_pool(snapshot, context, records, 10000, direct + 1)
    assert scored[0]['feasible'] and not scored[1]['feasible']
    assert scored[1]['failure'] == 'mission_length_limit'
    assert selection['effective_limit_m'] == pytest.approx(direct + 1)
    assert selection['expanded_best_id'] == 'incumbent'


def test_cap_validation_and_endpoint_checks(state):
    _, snapshot, context, records = state
    with pytest.raises(ValueError):
        score_pool(snapshot, context, records, float('nan'), 1000)
    records[1]['path'][0] = [21, 20]
    scored, _, _ = score_pool(snapshot, context, records, 10000, 10000)
    assert scored[1]['failure'] == 'endpoint_mismatch'


def test_future_evaluation_cannot_change_current_selector(state):
    _, snapshot, context, records = state
    _, a, _ = score_pool(snapshot, context, records, 10000, 10000)
    records[0]['evaluation'] = {'future_joint_seconds': 1e9}
    records[1]['evaluation'] = {'future_joint_seconds': 0}
    _, b, _ = score_pool(snapshot, context, records, 10000, 10000)
    assert a == b


def test_opportunity_diagnosis_distinguishes_rejected_and_acceptable():
    def record(name, score, joint, excess):
        return {'id': name, 'family': 'incumbent' if name == 'incumbent' else 'geometric', 'feasible': True,
                'path': [[0, 0], [score, 0]], 'balance': {'score_seconds': score}, 'metrics': {'length_m': score},
                'evaluation': {'future_joint_seconds': joint, 'future_excess_db_seconds': excess}}
    records = [record('incumbent', 100, 100, 10), record('rejected_safe', 110, 70, 5),
               record('accepted_safe', 99, 80, 8), record('proxy_bad', 90, 120, 20)]
    d = summarize(records, {'original_best_id': 'incumbent', 'expanded_best_id': 'proxy_bad'})
    assert d['v3_acceptable_truth_opportunity_ids'] == ['accepted_safe']
    assert d['v3_one_percent_truth_opportunity_ids'] == ['accepted_safe']
    assert d['v3_rejected_truth_opportunity_ids'] == ['rejected_safe']
    assert d['expanded_pool_lowers_current_score'] and not d['expanded_selection_improves_true_joint']
    assert d['posthoc_true_joint_best_id'] == 'rejected_safe'
    records[0]['evaluation']['future_excess_db_seconds'] = 0
    assert summarize(records, {'original_best_id': 'incumbent', 'expanded_best_id': 'proxy_bad'})['truth_opportunity_ids'] == []


def test_fixed_route_full_and_exact_prefix_have_consistent_units(state):
    world, snapshot, _, records = state
    result = evaluate_route(world, snapshot, records[0]['path'])
    duration = np.linalg.norm(np.array(records[0]['path'][-1])-records[0]['path'][0])/world.config.aircraft.speed_mps
    assert result['future_flight_seconds'] == pytest.approx(duration)
    assert result['future_power_seconds'] == pytest.approx(.01*duration)
    assert result['future_joint_seconds'] == pytest.approx(1.01*duration)
    assert result['prefix10']['flight_seconds'] == pytest.approx(10)
    assert result['prefix10']['power_seconds'] == pytest.approx(.1)
    assert result['future_power_seconds'] == pytest.approx(result['future_supported_power_seconds'] + result['future_unknown_power_seconds'])
    assert result['future_excess_db_seconds'] == 0 and result['new_measurements'] == 0
    assert result['frozen_truth_joint_seconds'] == pytest.approx(result['future_joint_seconds'])


def test_prefix_does_not_fabricate_ten_seconds_after_arrival(state):
    world, snapshot, _, _ = state
    result = evaluate_route(world, snapshot, [[20, 20], [35, 20]])
    assert result['prefix10']['flight_seconds'] == pytest.approx(1)
    assert result['prefix10']['joint_seconds'] == pytest.approx(result['future_joint_seconds'])
