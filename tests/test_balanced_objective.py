from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_em.balanced_objective import (
    BalancedOptimizer, PowerTimeSettings, choose, normalized_expected_power,
    paired_seed, pareto_ids, power_time_score,
)
from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.context import build_context
from uav_em.decision_objective import ObjectiveSettings, prefix_risk
from uav_em.environment import ElectromagneticWorld
from uav_em.offline_process import reference_program
from uav_em.process_attribution import diagnostic_prefix, check_identical_executed_prefix
from uav_em.repair_search import FixedObjective, WorkBudget


@pytest.fixture
def setup():
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/local_power.json')
    cfg = replace(cfg, map=replace(cfg.map, buildings=(), obstacles_m=()),
                  electromagnetic=replace(cfg.electromagnetic, fields=()))
    world = ElectromagneticWorld(cfg)
    belief = ElectromagneticBelief(cfg, world.geometry)
    belief.update(world.observe(cfg.map.start_m, 0))
    snap = belief.snapshot(0)
    route = [list(cfg.map.start_m), list(cfg.map.goal_m)]
    ctx = build_context(cfg.map.start_m, cfg.map.goal_m, route, snap, cfg, 100, [], [], None)
    return cfg, snap, ctx, route


def test_linear_power_expectation_analytic_and_continuous_below_old_reference():
    assert normalized_expected_power(-80, 0, -80) == pytest.approx(1)
    assert normalized_expected_power(-90, 0, -80) == pytest.approx(.1)
    assert normalized_expected_power(-100, 15, -80) == pytest.approx(np.exp(-2*np.log(10)+.5*(1.5*np.log(10))**2))
    assert normalized_expected_power(-95, 0, -80) < normalized_expected_power(-90, 0, -80)
    assert normalized_expected_power(-80.0001, 0, -80) == pytest.approx(1, rel=1e-4)
    with pytest.raises(ValueError): normalized_expected_power(1000, 1000, -80)


@pytest.mark.parametrize('setting', [{'weight': -1}, {'weight': True}, {'weight': float('nan')}, {'prior_std_db': -1}, {'scale_dbm': float('inf')}])
def test_invalid_settings_rejected(setting):
    with pytest.raises(ValueError): PowerTimeSettings(**setting)


def test_unknown_grid_values_are_not_used_or_exposed(setup):
    cfg, snap, _, route = setup
    snap.supported[:] = False
    score = power_time_score(FixedObjective(snap, -80, WorkBudget()), route, cfg.aircraft.speed_mps)
    snap.estimate_dbm[:] = np.nan; snap.uncertainty_db[:] = np.nan
    again = power_time_score(FixedObjective(snap, -80, WorkBudget()), route, cfg.aircraft.speed_mps)
    assert score == again and score['supported_power_seconds'] == 0
    assert score['unknown_prior_power_seconds'] > 0 and not score['unknown_power_estimate_exposed']


def test_inserting_loop_cannot_hide_same_unknown_tail_beyond_horizon(setup):
    _, snap, _, _ = setup
    snap.supported[:] = False
    snap.supported[:4, :4] = True
    snap.estimate_dbm[:] = -100; snap.uncertainty_db[:] = 0
    problem = FixedObjective(snap, -80, WorkBudget())
    straight = [[20, 20], [300, 20]]
    loop = [[20, 20], [20, 60], [60, 60], [20, 20], [300, 20]]
    old_a, old_b = [prefix_risk(problem, r, 15)['risk_db_seconds'] for r in (straight, loop)]
    assert old_b < old_a
    a, b = [power_time_score(problem, r, 15) for r in (straight, loop)]
    assert b['unknown_seconds'] == pytest.approx(a['unknown_seconds'])
    assert b['score_seconds'] > a['score_seconds']


def test_score_invariant_to_collinear_waypoint_insertion_and_power_unit_change(setup):
    _, snap, _, _ = setup
    problem = FixedObjective(snap, -80, WorkBudget())
    route = [[20, 20], [220, 20]]
    a = power_time_score(problem, route, 15)
    b = power_time_score(problem, [[20,20], [70,20], [220,20]], 15)
    for key in ('score_seconds','remaining_flight_seconds','power_proxy_seconds','unknown_seconds'):
        assert a[key] == pytest.approx(b[key])
    rescaled = power_time_score(problem, route, 15, PowerTimeSettings(weight=10, scale_dbm=-70))
    assert rescaled['score_seconds'] == pytest.approx(a['score_seconds'])


def test_higher_power_always_costs_more_and_length_matters(setup):
    _, snap, _, _ = setup
    snap.supported[:] = True; snap.uncertainty_db[:] = 0
    problem = FixedObjective(snap, -80, WorkBudget())
    route = [[20,20], [220,20]]
    snap.estimate_dbm[:] = -95
    low = power_time_score(problem, route, 15)
    snap.estimate_dbm[:] = -85
    high = power_time_score(problem, route, 15)
    assert high['score_seconds'] > low['score_seconds']
    assert power_time_score(problem, [[20,20],[120,20]], 15)['score_seconds'] < high['score_seconds']


def test_pareto_and_selector_do_not_reward_dominated_or_infeasible_routes():
    def r(name, length, power, score, feasible=True):
        return {'id':name, 'metrics':{'length_m':length}, 'balance':{'power_proxy_seconds':power,'score_seconds':score}, 'feasible':feasible}
    records = [r('short',100,20,120), r('long_safe',150,1,151), r('dominated',160,21,181), r('invalid',1,0,1,False)]
    assert pareto_ids(records) == ['short','long_safe']
    assert choose(records)['id'] == 'short'
    assert choose([]) is None


def test_search_stream_ignores_other_calls_and_separates_stages():
    a = paired_seed(613,10,'node0','local_rrt')
    paired_seed(613,0,'node4','global_rrt')
    assert paired_seed(613,10,'node0','local_rrt') == a
    assert len({a,paired_seed(613,10,'node1','local_rrt'),paired_seed(613,10,'node0','via_leg1'),paired_seed(613,20,'node0','local_rrt'),paired_seed(613,10,'node0','local_rrt',1)}) == 5


def test_actual_common_pool_and_executed_prefix_reproduce_under_different_simulator_seeds(setup):
    _, snap, ctx, route = setup
    limits = ObjectiveSettings(search_samples=3, sample_limit=80, edge_limit=1200)
    program = reference_program(False)
    kwargs = dict(scene_seed=17,limits=limits)
    full = BalancedOptimizer(program=program, **kwargs)
    local = BalancedOptimizer(program=diagnostic_prefix(program,'local_prefix'), **kwargs)
    a, ra = full.optimize(snap, deepcopy(ctx), route, 2000, 999, 0)
    b, rb = local.optimize(snap, deepcopy(ctx), route, 2000, 12345, 0)
    check_identical_executed_prefix(ra,rb)
    assert ra['common_pool_seed'] == rb['common_pool_seed']
    n = len(rb['candidates'])-sum(r['id'].startswith('process_candidate_') for r in rb['candidates'])
    assert ra['candidates'][:n] == rb['candidates'][:n]
    for report in (ra,rb):
        assert report['resources']['samples'] <= 80 and report['resources']['edges'] <= 1200
        assert not report['policy_errors'] and report['actual_advisor_calls'] == 0
        for search in report['searches']:
            assert search['actual_seed'] == paired_seed(17,0,search['seed_stage'],search['seed_purpose'])
        selected = choose(report['candidates'])
        assert selected['balance'] == report['selected_balance']
        assert selected['metrics']['length_m'] <= report['mission_distance_cap_m']


def test_frozen_motion_integration_preserves_flight_units_and_start_progress(setup):
    from uav_em.balanced_evaluation import evaluate_motions
    cfg, _, _, _ = setup
    world = ElectromagneticWorld(cfg)
    a = list(cfg.map.start_m); b = [a[0]+150,a[1]]
    result = evaluate_motions(world,[{'end_time_seconds':10.5,'traversed_m':[a,b]}])
    assert result['flight_seconds_from_trace'] == pytest.approx(10)
    assert result['flight_power_integral_normalized_seconds'] == pytest.approx(.1)
    assert result['flight_received_power_integral_mw_seconds'] == pytest.approx(1e-9)
    assert result['reintegrated_excess_db_seconds'] == 0
    assert result['first50_goal_distance_progress_m'] == pytest.approx(50)
    assert result['first150_backward_projection_m'] == 0
