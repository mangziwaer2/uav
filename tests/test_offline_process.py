"""共同接受器、生成边界及跨初始池/有限过程的预算不变量。"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.context import build_context
from uav_em.decision_objective import ObjectiveSettings, select_candidate
from uav_em.environment import ElectromagneticWorld
from uav_em.offline_process import (
    OfflineProcessOptimizer, VERSION, _RiskSession, generation_contract,
    reference_program, task_evaluation, validate_program,
)
from uav_em.repair_process import canonical_path
from uav_em.repair_search import FixedObjective, WorkBudget
from uav_em.route_repair import RepairSession, RepairSettings, anchors


@pytest.fixture
def setup():
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/local_power.json')
    cfg = replace(cfg, map=replace(cfg.map, buildings=(), obstacles_m=()),
                  electromagnetic=replace(cfg.electromagnetic, fields=()))
    world = ElectromagneticWorld(cfg)
    belief = ElectromagneticBelief(cfg, world.geometry)
    belief.update(world.observe(cfg.map.start_m, 0))
    snapshot = belief.snapshot(0)
    route = [list(cfg.map.start_m), list(cfg.map.goal_m)]
    ctx = build_context(cfg.map.start_m, cfg.map.goal_m, route, snapshot, cfg, 100, [], [], None)
    return cfg, snapshot, ctx, route


@pytest.mark.parametrize('edit', [
    lambda p: p.update(python='import os'),
    lambda p: p['nodes'][0]['next'].update(default=0),
    lambda p: p['nodes'][0]['action'].update(metric='true_source'),
    lambda p: p['nodes'][0]['action'].update(allocation='unlimited'),
    lambda p: p['nodes'][0]['action'].update(operator='probe'),
    lambda p: p['nodes'][0]['action'].update(padding=True),
])
def test_generated_program_cannot_expand_information_actions_or_budget(edit):
    program = reference_program()
    edit(program)
    with pytest.raises(ValueError):
        validate_program(program)


def test_reference_and_submission_validation_use_same_version_and_schema():
    p = reference_program()
    assert validate_program(p) == p and p['version'] == VERSION
    assert generation_contract()['generation_budget']['max_submitted_artifacts'] == 4
    with pytest.raises(ValueError):
        OfflineProcessOptimizer('submitted_process')
    with pytest.raises(ValueError):
        OfflineProcessOptimizer('adaptive_process', p)


@pytest.mark.parametrize('method', ['pool_only', 'fixed_process', 'adaptive_process', 'submitted_process'])
def test_pool_search_process_and_verification_share_budget(setup, method):
    _, snapshot, ctx, route = setup
    settings = ObjectiveSettings(search_samples=3, sample_limit=60, edge_limit=900)
    program = reference_program() if method == 'submitted_process' else None
    plan, report = OfflineProcessOptimizer(method, program, settings).optimize(snapshot, ctx, route, 2000, 41, 0)
    assert plan.path[0] == route[0] and plan.path[-1] == route[-1]
    assert report['resources']['samples'] <= 60 and report['resources']['edges'] <= 900
    assert report['resources']['verification_edges'] >= len(plan.path) - 1
    assert report['resources']['samples'] >= report['common_pool_resources']['samples']
    assert len(report['process_trace']) <= 6
    assert not report['policy_errors'] and report['actual_advisor_calls'] == 0
    assert report['executed_distance_m'] + plan.length_m <= report['mission_distance_cap_m'] + 1e-7
    chosen = select_candidate(report['candidates'], 'prefix_v2', settings)
    assert canonical_path(chosen['path']) == canonical_path(plan.path)
    actual = FixedObjective(snapshot, -80, WorkBudget()).route_metrics(plan.path)
    assert actual == pytest.approx(report['best_metrics'])


def test_state_allowlist_drops_injected_hidden_context_and_preserves_unknown(setup):
    _, snapshot, ctx, route = setup
    ctx['true_sources'] = [{'x': 111, 'y': 222}]
    ctx['source_inference'] = {'bearing': 90}
    settings = ObjectiveSettings(search_samples=3, sample_limit=60, edge_limit=900)
    _, report = OfflineProcessOptimizer(settings=settings).optimize(snapshot, ctx, route, 2000, 41, 0)
    assert report['process_trace']
    for entry in report['process_trace']:
        state = entry['state']
        public = state['public_context']
        assert 'true_sources' not in public and 'source_inference' not in public
        assert state['task_objective'] == 'common_R10_window_then_length'
        em = public['electromagnetic_estimate']
        columns = em['sample_columns']
        for row in em['coarse_samples']:
            if not row[columns.index('supported')]:
                assert row[columns.index('estimate_dbm')] is None


def test_submitted_reference_reproduces_numeric_process_without_model_call(setup):
    _, snapshot, ctx, route = setup
    settings = ObjectiveSettings(search_samples=3, sample_limit=60, edge_limit=900)
    a, ra = OfflineProcessOptimizer('adaptive_process', settings=settings).optimize(snapshot, ctx, route, 2000, 41, 0)
    b, rb = OfflineProcessOptimizer('submitted_process', reference_program(), settings).optimize(snapshot, ctx, route, 2000, 41, 0)
    assert a.path == b.path and ra['best_metrics'] == rb['best_metrics']
    assert ra['resources'] == rb['resources']
    assert rb['model_calls_within_optimizer'] == 0


def test_common_selector_can_accept_larger_old_j_without_supported_risk_gate(setup, monkeypatch):
    _, snapshot, ctx, route = setup
    settings = ObjectiveSettings(search_samples=3, sample_limit=60, edge_limit=900)
    _, pool = OfflineProcessOptimizer('pool_only', settings=settings).optimize(snapshot, ctx, route, 2000, 41, 0)
    session = _RiskSession(snapshot, ctx, route, 2000, RepairSettings(sample_limit=60, edge_limit=900), 41, pool, settings)
    for record in session.records:
        if record['feasible']:
            record['prefix']['risk_db_seconds'] = 1000
    candidate = anchors([route[0], [400, 520], route[-1]], 80)
    metrics = FixedObjective(snapshot, -80, WorkBudget()).route_metrics(candidate)
    session.max_length = 2000
    # Isolate acceptance from stochastic search; candidate retains real geometry and metrics.
    def fake_apply(self, proposal):
        self.searches.append({'path': candidate})
        return {'accepted': False, 'reason': 'no_objective_improvement', 'candidate_metrics': metrics,
                'best_metrics': self.best_metrics, 'resources': self.work.summary()}
    monkeypatch.setattr(RepairSession, 'apply', fake_apply)
    before = session.best_metrics['objective']
    feedback = session.apply({'operator': 'global_rrt'})
    assert metrics['objective'] > before
    assert feedback['accepted'] and session.best == candidate


def test_fixed_lifetime_cap_does_not_reset_at_next_decision(setup):
    _, snapshot, ctx, route = setup
    settings = ObjectiveSettings(search_samples=3, sample_limit=60, edge_limit=900)
    optimizer = OfflineProcessOptimizer('pool_only', settings=settings)
    _, first = optimizer.optimize(snapshot, ctx, route, 2000, 41, 0)
    later = deepcopy(ctx)
    later['execution_feedback'] = {'interval': {'flown_m': 20}}
    snapshot.time_seconds = 1
    _, second = optimizer.optimize(snapshot, later, route, 2000, 41, 0)
    assert second['mission_distance_cap_m'] == first['mission_distance_cap_m']
    assert second['executed_distance_m'] == 20


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1, True])
def test_task_evaluation_rejects_invalid_metrics(bad):
    metrics = {'success': True, 'executed_length_m': 100, 'interference_excess_integral_db_seconds': 10}
    candidate = {**metrics, 'executed_length_m': bad}
    with pytest.raises(ValueError):
        task_evaluation([{'scene': 'a', 'baseline': metrics, 'candidate': candidate}])


def test_task_gate_cannot_reward_failure_or_hide_individual_scene_regression():
    def m(excess, length=100, success=True):
        return {'success': success, 'executed_length_m': length, 'interference_excess_integral_db_seconds': excess}
    good = [{'scene': 'a', 'baseline': m(100), 'candidate': m(80)}]
    assert task_evaluation(good)['passes_task_gate']
    assert not task_evaluation([{'scene': 'a', 'baseline': m(100), 'candidate': m(0, success=False)}])['passes_task_gate']
    rows = good + [{'scene': 'b', 'baseline': m(1), 'candidate': m(3)}]
    assert not task_evaluation(rows)['checks']['no_material_per_scene_regression']
    with pytest.raises(ValueError):
        task_evaluation(good + good)
