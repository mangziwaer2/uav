"""有限过程、公开状态、反馈分支及跨步骤/终检共享预算。"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.context import build_context
from uav_em.environment import ElectromagneticWorld
from uav_em.repair_search import FixedObjective,WorkBudget
from uav_em.route_repair import RepairSession,RepairSettings
from uav_em.repair_process import (RepairState,RepairFeedback,ProcessRepairOptimizer,baseline_process,validate_process,
    select_interval,allocate,canonical_path,feedback_outcome)

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def setup():
    cfg=load_config(ROOT/'configs/local_power.json')
    cfg=replace(cfg,map=replace(cfg.map,buildings=(),obstacles_m=()),
                electromagnetic=replace(cfg.electromagnetic,fields=(),measurement_noise_db=0))
    world=ElectromagneticWorld(cfg);belief=ElectromagneticBelief(cfg,world.geometry)
    belief.update(world.observe(cfg.map.start_m,0));snap=belief.snapshot(0)
    snap.estimate_dbm[:]=-100;snap.uncertainty_db[:]=0
    ctx=build_context(cfg.map.start_m,cfg.map.goal_m,[],snap,cfg,100,[],[],None)
    route=[list(cfg.map.start_m),[240,400],list(cfg.map.goal_m)]
    session=RepairSession(snap,ctx,route,2000,RepairSettings(),11)
    return cfg,snap,ctx,route,session.request(1,6)

def test_state_uses_allowlist_unknown_null_and_independent_copy(setup):
    *_,context=setup;context['evaluation_only_config']={'hidden_sources':[99]}
    context['source_inference']={'future':True}
    em=context['electromagnetic_estimate'];col=em['sample_columns'];row=em['coarse_samples'][0]
    row[col.index('supported')]=False;row[col.index('estimate_dbm')]=-30
    state=RepairState.from_context(context).to_dict();text=json.dumps(state)
    assert 'hidden_sources' not in text and 'source_inference' not in text
    assert state['public_context']['electromagnetic_estimate']['coarse_samples'][0][col.index('estimate_dbm')] is None
    state['public_context']['vehicle']['position_m'][0]=999
    assert context['vehicle']['position_m'][0]!=999

def test_future_measurement_is_rejected(setup):
    *_,context=setup
    context['measurement_evidence']['recent_points'][0]['time_seconds']=1
    with pytest.raises(ValueError,match='future'):RepairState.from_context(context)

def test_zero_supported_interference_explicitly_falls_back_to_unknown_front(setup):
    *_,context=setup
    interval,selection=select_interval(context,'supported_interference',1)
    assert selection['resolved_metric']=='unknown_front'
    assert selection['fallback']=='no_supported_interference_signal'
    assert not selection['supported_interference_discriminative']
    assert int(interval[0][1:])<int(interval[1][1:])

def test_selector_only_uses_current_remaining_anchor_ids(setup):
    *_,context=setup
    for metric in ('evidence','stale','detour','whole','unknown_front'):
        interval,_=select_interval(context,metric,3)
        assert set(interval)<={a['id'] for a in context['route_repair']['anchors']}

@pytest.mark.parametrize('change',[
    lambda p:p['nodes'][0]['next'].update(default=0),
    lambda p:p['nodes'][1]['next'].update(default=0),
    lambda p:p['nodes'][0]['action'].update(operator='exec'),
    lambda p:p['nodes'][0]['action'].update(metric='true_source'),
    lambda p:p['nodes'][0]['action'].update(padding=True),
    lambda p:p['nodes'][0]['next'].update(python='stop'),
    lambda p:p.update(code='import os'),
    lambda p:p['nodes'].append(deepcopy(p['nodes'][-1])),
])
def test_process_rejects_unbounded_control_or_unknown_operations(change):
    program=baseline_process();change(program)
    with pytest.raises(ValueError):validate_process(program)

def test_path_dedup_removes_collinear_anchors_but_preserves_backtracking():
    assert canonical_path([[0,0],[10,0],[20,0]])==canonical_path([[0,0],[20,0]])
    assert canonical_path([[0,0],[20,0],[10,0],[30,0]])!=canonical_path([[0,0],[30,0]])

def test_feedback_quota_uses_marginal_cost_and_reserves_for_later_steps(setup):
    *_,ctx=setup;r=ctx['route_repair']
    r['resources'].update(samples_left=1900,edges_left=5000)
    r['previous_feedback']=[{'resources':{'samples':100,'edges':2000}},
                            {'resources':{'samples':200,'edges':5000}}]
    few,trace=allocate(ctx,'feedback',4);one,_=allocate(ctx,'feedback',1)
    assert trace['estimated_edges_per_sample']==30 and 2<=few<one<=700
    r['resources']['edges_left']=16
    assert allocate(ctx,'feedback',1)[0]==0

@pytest.mark.parametrize('method',['library_numeric','fixed_process','adaptive_process'])
def test_process_shared_caps_final_verification_and_no_model_attribution(setup,method):
    _,snap,ctx,route,_=setup;before=snap.estimate_dbm.copy()
    opt=ProcessRepairOptimizer(method,settings=RepairSettings(sample_limit=60,edge_limit=900))
    result,report=opt.optimize(snap,ctx,route,2000,41,0)
    assert result.path[0]==route[0] and result.path[-1]==route[-1]
    assert report['best_metrics']['objective']<=report['initial_metrics']['objective']+1e-7
    assert report['resources']['samples']<=60 and report['resources']['edges']<=900
    assert report['resources']['verification_edges']==len(result.path)-1
    assert len(report['process_trace'])<=6 and not report['policy_errors']
    assert report['actual_advisor_calls']==report['accepted_model_proposals']==0
    assert not report['arbitrary_python_executed'] and report['new_observations']==0
    actual=FixedObjective(snap,-80,WorkBudget()).route_metrics(result.path)
    assert actual==pytest.approx(report['best_metrics'])
    np.testing.assert_array_equal(snap.estimate_dbm,before)

def test_identical_shortcuts_are_deduplicated_before_extra_edge_evaluation(setup):
    _,snap,ctx,_,_=setup;route=[[40,300],[760,300]]
    program=baseline_process(False);program['nodes']=program['nodes'][:2]
    program['nodes'][0]['next']={'default':1}
    program['nodes'][1]={'action':deepcopy(program['nodes'][0]['action']),'next':{'default':'stop'}}
    _,report=ProcessRepairOptimizer('fixed_process',program).optimize(snap,ctx,route,2000,41,0)
    assert [t['feedback']['outcome'] for t in report['process_trace']]==['duplicate','duplicate']
    assert all(t['feedback']['work_delta']['edges']==0 for t in report['process_trace'])
    assert report['resources']['samples']==0

def test_actual_geometry_failure_takes_forward_failure_branch(setup):
    cfg,snap,ctx,_,_=setup
    # Known obstacle blocks whole shortcut, while the incumbent goes above it.
    from uav_em.environment import Geometry
    cfg=replace(cfg,map=replace(cfg.map,obstacles_m=((320,250,400,350),)))
    snap.geometry=Geometry(cfg.map,0,cfg.aircraft.altitude_m,0)
    route=[[40,300],[280,400],[440,400],[760,300]]
    program=baseline_process(False);program['nodes'][0]['next']={'geometry_failure':2,'default':'stop'}
    program['nodes'][2]['action']=deepcopy(program['nodes'][0]['action'])
    program['nodes'][2]['next']={'default':'stop'}
    _,report=ProcessRepairOptimizer('fixed_process',program).optimize(snap,ctx,route,2000,41,0)
    assert [t['node'] for t in report['process_trace']]==[0,2]
    assert report['process_trace'][0]['feedback']['outcome']=='geometry_failure'
    assert report['final_geometry_verified']

def test_search_failure_and_no_gain_have_different_feedback():
    assert feedback_outcome({'accepted':False,'reason':'no_path_within_budget'})=='search_failure'
    assert feedback_outcome({'accepted':False,'reason':'no_objective_improvement'})=='no_gain'
    assert feedback_outcome({'accepted':False,'reason':'edge_budget_exhausted'})=='resource_stop'

def test_structured_feedback_is_causal_and_available_to_next_state(setup):
    _,snap,ctx,route,_=setup
    _,report=ProcessRepairOptimizer(settings=RepairSettings(sample_limit=60,edge_limit=900)).optimize(snap,ctx,route,2000,41,0)
    first=report['process_trace'][0]['feedback']
    assert report['process_trace'][1]['state']['previous_process_feedback'][0]==first
    bad=deepcopy(first);bad['true_future_exposure']=0
    with pytest.raises(ValueError,match='schema'):RepairFeedback.from_dict(bad)
    bad=deepcopy(first);bad['future_exposure_evaluation_included']=True
    with pytest.raises(ValueError,match='boundary'):RepairFeedback.from_dict(bad)

def test_initial_route_bootstrap_and_simulator_integration_use_same_ledger(setup):
    cfg,_,_,_,_=setup
    cfg=replace(cfg,map=replace(cfg.map,goal_m=(160,300)),simulation=replace(cfg.simulation,max_seconds=20))
    from uav_em.simulation import simulate
    episode=simulate(cfg,advisor_name='numeric_process_test',route_optimizer=ProcessRepairOptimizer())
    assert episode['metrics']['success'] and episode['metrics']['advisor_calls']==0
    reports=[d['repair_optimization'] for d in episode['decisions']]
    assert reports[0]['numeric_recovery'] and all(r['final_geometry_verified'] for r in reports)
    assert all(r['resources']['edges']<=10000 and r['resources']['samples']<=2100 for r in reports)

@pytest.mark.parametrize('success,length,excess,baseline_excess,expected',[
    (True,100,90,100,True),(False,50,10,100,False),(True,110,50,100,False),
    (True,100,0,0,False),(True,100,10,0,False),
])
def test_task_gate_does_not_reward_failure_tradeoff_or_zero_exposure_artifact(monkeypatch,success,length,excess,baseline_excess,expected):
    monkeypatch.syspath_prepend(str(ROOT/'tools'))
    from run_process_study import task_gate
    rows=[{'method':'library_numeric','metrics':{'success':True,'executed_length_m':100,
                                              'interference_excess_integral_db_seconds':baseline_excess}},
          {'method':'adaptive_process','metrics':{'success':success,'executed_length_m':length,
                                                  'interference_excess_integral_db_seconds':excess}}]
    assert task_gate(rows,'adaptive_process')['passes_development_task_gate']==expected
