from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_em.balanced_objective import BalancedOptimizer
from uav_em.belief import ElectromagneticBelief
from uav_em.causal_feedback import prequential_evidence, validate_response, run_selection, numerical_choice
from uav_em.config import load_config
from uav_em.context import build_context
from uav_em.decision_objective import ObjectiveSettings
from uav_em.environment import ElectromagneticWorld, Measurement


@pytest.fixture
def sample():
    cfg = load_config(Path(__file__).resolve().parents[1]/'configs/local_power.json')
    cfg = replace(cfg, map=replace(cfg.map, buildings=(), obstacles_m=()),
                  electromagnetic=replace(cfg.electromagnetic, fields=()))
    world = ElectromagneticWorld(cfg)
    points = [{'position_m': [20+15*t, 20], 'time_seconds': float(t), 'interference_dbm': -90+float(t)}
              for t in range(6)]
    return cfg, world, points


def test_prediction_precedes_own_reading(sample):
    cfg, world, points = sample
    first = prequential_evidence(cfg, world.geometry, points, 5)
    modified = deepcopy(points)
    modified[-1]['interference_dbm'] += 30
    second = prequential_evidence(cfg, world.geometry, modified, 5)
    a, b = first['recent_checks'][-1], second['recent_checks'][-1]
    assert a['predicted_before_reading_dbm'] == b['predicted_before_reading_dbm']
    assert b['residual_db']-a['residual_db'] == pytest.approx(30)
    assert first['recent_checks'][0]['predicted_before_reading_dbm'] is None


def test_straight_constant_speed_design_cannot_separate_time_from_space(sample):
    cfg, world, points = sample
    result = prequential_evidence(cfg, world.geometry, points, 5)
    assert result['trajectory_near_collinear']
    assert result['centered_xy_time_design_rank_out_of3'] == 1
    assert not result['spatial_temporal_affine_effects_separable_on_this_design']
    assert not result['lateral_distribution_identified']


def test_future_and_unordered_measurements_are_rejected(sample):
    cfg, world, points = sample
    with pytest.raises(ValueError, match='future'):
        prequential_evidence(cfg, world.geometry, points, 4)
    with pytest.raises(ValueError, match='unordered'):
        prequential_evidence(cfg, world.geometry, points[::-1], 5)


@pytest.mark.parametrize('change', [{'request_id': 'stale'}, {'profile': 'run_python'},
                                  {'weight': 20}, {'reason': ''}, {'memory': 'x'*1201}])
def test_model_schema_rejects_extra_capabilities_and_stale_identity(change):
    value = dict(request_id='select_01', profile='stop', reason='sufficient evidence', memory='')
    value.update(change)
    with pytest.raises(ValueError):
        validate_response(value, 'select_01')


def test_numerical_stopping_rule_is_frozen_not_true_exposure_based():
    state = {'previous_numeric_feedback': []}
    assert numerical_choice('numeric_feedback', state) == 'local_pad3'
    state['previous_numeric_feedback'] = [{'current_score_relative_gain': .01, 'future_true_score': 1e9}]
    assert numerical_choice('numeric_feedback', state) == 'stop'
    state['previous_numeric_feedback'][0]['current_score_relative_gain'] = .0099
    assert numerical_choice('numeric_feedback', state) == 'global_focus'
    assert numerical_choice('fixed_two', state) == 'global_focus'


def _prepared(sample):
    cfg, world, points = sample
    belief = ElectromagneticBelief(cfg, world.geometry)
    for p in points:
        belief.update(Measurement(tuple(p['position_m']), p['time_seconds'], p['interference_dbm']))
    snapshot = belief.snapshot(5)
    route = [points[-1]['position_m'], list(cfg.map.goal_m)]
    context = build_context(route[0], route[-1], route, snapshot, cfg, 100, [], [], None)
    _, report = BalancedOptimizer(cfg.seed, limits=ObjectiveSettings(search_samples=3)).optimize(snapshot, context, route, 2000, 0, 0)
    case = {'seed': cfg.seed, 'context': context, 'max_length_m': report['mission_distance_cap_m'],
            'score_settings': report['power_time_settings']}
    return snapshot, case, report, prequential_evidence(cfg, world.geometry, points, 5)


def test_stop_preserves_route_and_charges_only_common_summary_and_final_verification(sample):
    snapshot, case, report, reliability = _prepared(sample)
    result = run_selection(snapshot, case, report, reliability, 'pool_only')
    assert not result['searches'] and result['model_calls'] == 0
    assert result['resources']['samples'] == report['resources']['samples']
    assert result['resources']['edges'] > report['resources']['edges']
    assert result['selected_balance'] == report['selected_balance']
    assert result['trace'][0]['state']['true_future_evaluation_included'] is False
    assert result['trace'][0]['state']['reliability']['new_measurements'] == 0
    assert all(r['feasible'] or 'evaluation' not in r for r in result['candidates'])


def test_policy_error_keeps_route_without_unlogged_recovery(sample):
    snapshot, case, report, reliability = _prepared(sample)
    class ErrorPolicy:
        last_call = {'wall_seconds': .01, 'status': 'error'}
        def decide(self, state):
            raise ValueError('invalid raw response')
    result = run_selection(snapshot, case, report, reliability, 'llm_feedback', ErrorPolicy())
    assert result['model_calls'] == result['model_errors'] == 1
    assert not result['searches'] and result['fallback_searches'] == 0
    assert result['selected_balance'] == report['selected_balance']
    assert result['trace'][0]['error'] == 'ValueError: invalid raw response'
