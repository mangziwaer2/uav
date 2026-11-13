from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import pytest

from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.context import build_context
from uav_em.decision_objective import ObjectiveSettings
from uav_em.environment import ElectromagneticWorld
from uav_em.offline_process import OfflineProcessOptimizer, reference_program
from uav_em.process_attribution import diagnostic_prefix, check_identical_executed_prefix, selection_comparison


@pytest.mark.parametrize('mode,boundary', [('local_prefix', 2), ('via_prefix', 4), ('global_prefix', 5), ('full_process', 6)])
def test_prefix_keeps_all_nodes_actions_and_quota_slots(mode, boundary):
    parent = reference_program()
    before = deepcopy(parent)
    child = diagnostic_prefix(parent, mode)
    assert parent == before and len(child['nodes']) == 6
    for old, new in zip(parent['nodes'], child['nodes']):
        assert old['action'] == new['action']
        for outcome, target in old['next'].items():
            assert new['next'][outcome] == ('stop' if type(target) is int and target >= boundary else target)
    if mode == 'full_process': assert child == parent


def test_prefix_does_not_accept_arbitrary_programs_or_new_mode():
    with pytest.raises(ValueError): diagnostic_prefix(reference_program(), 'best_fit')
    parent = reference_program(); parent['nodes'] = parent['nodes'][:1]; parent['nodes'][0]['next'] = {'default': 'stop'}
    with pytest.raises(ValueError): diagnostic_prefix(parent, 'local_prefix')


def test_actual_search_prefix_retains_proposals_feedback_and_work():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'configs/local_power.json')
    cfg = replace(cfg, map=replace(cfg.map, buildings=(), obstacles_m=()), electromagnetic=replace(cfg.electromagnetic, fields=()))
    world = ElectromagneticWorld(cfg); belief = ElectromagneticBelief(cfg, world.geometry)
    belief.update(world.observe(cfg.map.start_m, 0)); snapshot = belief.snapshot(0)
    route = [list(cfg.map.start_m), [400, 400], list(cfg.map.goal_m)]
    context = build_context(cfg.map.start_m, cfg.map.goal_m, route, snapshot, cfg, 100, [], [], None)
    settings = ObjectiveSettings(search_samples=3, sample_limit=80, edge_limit=1200)
    parent = reference_program(False)
    _, full = OfflineProcessOptimizer('submitted_process', parent, settings).optimize(snapshot, context, route, 2000, 19, 0)
    _, short = OfflineProcessOptimizer('submitted_process', diagnostic_prefix(parent, 'local_prefix'), settings).optimize(snapshot, context, route, 2000, 19, 0)
    assert check_identical_executed_prefix(full, short)['unused_budget_redistributed'] is False
    assert len(short['process_trace']) <= 2
    assert short['resources']['samples'] <= full['resources']['samples']
    bad = deepcopy(short); bad['common_pool_resources']['samples'] += 1
    with pytest.raises(ValueError): check_identical_executed_prefix(full, bad)
    if short['process_trace']:
        bad = deepcopy(short); bad['process_trace'][0]['work_delta']['edges'] += 1
        with pytest.raises(ValueError): check_identical_executed_prefix(full, bad)


def test_same_pool_ranking_is_blind_to_posthoc_truth_and_does_not_mutate():
    def record(name,j,risk,length):
        return {'id': name, 'feasible': True, 'metrics': {'objective': j, 'length_m': length},
                'prefix': {'risk_db_seconds': risk}, 'path': [[0, 0], [length, 0]]}
    records = [record('incumbent',100,20,100),record('process_candidate_1',150,1,110)]
    original = deepcopy(records); a = selection_comparison(records)
    assert a['selected_ids_differ'] and a['risk_choice']['id'] == 'process_candidate_1'
    for r in records: r['true_future_exposure'] = 999999
    assert selection_comparison(records) == a
    for r in records: r.pop('true_future_exposure')
    assert records == original
    assert a['not_a_search_objective_ablation']


def test_evaluation_only_prefix_matches_real_snapshot_counterfactual():
    from uav_em.decision_objective import cut_route
    from uav_em.diagnostic_evaluation import evaluate_fixed_route
    from uav_em.process_attribution_evaluation import evaluate_frozen_prefix
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'configs/local_power.json')
    world = ElectromagneticWorld(cfg)
    belief = ElectromagneticBelief(cfg, world.geometry)
    belief.update(world.observe(cfg.map.start_m, 4))
    snapshot = belief.snapshot(4)
    route = [list(cfg.map.start_m), [300, 300], list(cfg.map.goal_m)]
    expected = evaluate_fixed_route(world, snapshot, cut_route(route, cfg.aircraft.speed_mps * 10)[0])
    result = evaluate_frozen_prefix(world, route, 4)
    assert result == {key: expected[key] for key in result}
    assert result['flight_seconds'] == pytest.approx(10)
    with pytest.raises(ValueError): evaluate_frozen_prefix(world, [list(cfg.map.start_m)], 4)
