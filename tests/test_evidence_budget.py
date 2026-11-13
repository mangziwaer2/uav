"""公共点测、机动额度与支持风险条件的行为不变量。"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from uav_em.evidence_budget import risk_evidence,choose_process,EvidenceBudgetOptimizer,METHODS
from uav_em.decision_objective import ObjectiveSettings
from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.environment import ElectromagneticWorld
from uav_em.context import build_context
from uav_em.planning import PlanResult


def context(power=-70,time=20):
    return {'time_seconds':time,'electromagnetic_estimate':{'interference_reference_dbm':-80},
        'latest_observation':{'time_seconds':time,'interference_dbm':power},
        'measurement_evidence':{'recent_points':[]}}


def record(name,length,known,unknown,j):
    return {'id':name,'feasible':True,'path':[[0,0],[length,0]],
        'metrics':{'length_m':length,'objective':j},
        'prefix':{'risk_db_seconds':known+10*unknown,'supported_expected_excess_db_seconds':known,'unknown_seconds':unknown}}


def test_evidence_uses_fresh_scalar_points_and_ignores_unrelated_hidden_fields():
    ctx=context(-90);ctx['measurement_evidence']['recent_points']=[{'time_seconds':11,'interference_dbm':-70}]
    a=risk_evidence(ctx);assert a['active'] and a['above_reference_point_count']==1
    ctx['evaluation_only_config']={'true_source_position':[500,400],'future_script':'switch'}
    ctx['source_inference']={'invented_bearing':99}
    assert risk_evidence(ctx)==a
    ctx['time_seconds']=22;ctx['latest_observation']['time_seconds']=22
    assert not risk_evidence(ctx)['active']


@pytest.mark.parametrize('timestamp,power',[(21,-70),(20,float('nan')),(20,float('inf'))])
def test_future_and_nonfinite_measurement_rejected(timestamp,power):
    ctx=context();ctx['latest_observation']={'time_seconds':timestamp,'interference_dbm':power}
    with pytest.raises(ValueError):risk_evidence(ctx)


def test_duplicate_summary_latest_not_counted_twice_and_reference_is_soft():
    ctx=context(-80);ctx['measurement_evidence']['recent_points']=[deepcopy(ctx['latest_observation'])]
    r=risk_evidence(ctx);assert r['point_count']==1 and not r['active']
    ctx['latest_observation']['interference_dbm']=-79.99
    assert risk_evidence(ctx)['active']
    assert not risk_evidence(ctx)['noise_significance_test']


def test_unknown_reduction_alone_cannot_buy_extra_detour():
    pool=[record('baseline',100,4,5,100),record('less_unknown',130,6,2,130)]
    before=deepcopy(pool)
    immediate,_=choose_process(pool,'immediate_v2',context(),0,200)
    chosen,state=choose_process(pool,'evidence_budget',context(),0,200)
    assert immediate['id']=='less_unknown' and chosen['id']=='baseline'
    check=next(c for c in state['candidate_checks'] if c['id']=='less_unknown')
    assert 'extra_length_without_supported_risk_proxy_gain' in check['reasons']
    assert pool==before


def test_no_fresh_strong_evidence_preserves_baseline_even_if_proxy_prefers_detour():
    pool=[record('baseline',100,10,5,100),record('detour',120,3,2,120)]
    for method in ('evidence_only','evidence_budget'):
        result,state=choose_process(pool,method,context(-90),0,200)
        assert result['id']=='baseline' and not state['evidence']['active']
    assert choose_process(pool,'evidence_only',context(-70),0,200)[0]['id']=='detour'


def test_half_margin_retains_future_slack_and_baseline_exception_is_explicit():
    pool=[record('baseline',100,20,5,100),record('near',120,8,2,120),record('far',160,1,1,160)]
    chosen,state=choose_process(pool,'evidence_budget',context(),50,250)
    assert state['remaining_maneuver_margin_m']==100 and state['local_length_cap_m']==150
    assert chosen['id']=='near'
    assert 'preserve_half_remaining_margin' in state['candidate_checks'][2]['reasons']
    # The old-J baseline can exceed the local allocation cap, and remains a legal retention.
    pool=[record('baseline',140,0,0,1),record('short',100,10,5,2)]
    chosen,state=choose_process(pool,'evidence_budget',context(),0,150)
    assert chosen['id']=='baseline' and state['baseline_retention_exception']


def test_allocation_uses_current_executed_distance_and_cannot_replenish_global_cap():
    pool=[record('baseline',100,10,5,100),record('detour',140,3,2,140)]
    _,a=choose_process(pool,'evidence_budget',context(),0,200)
    selected,b=choose_process(pool,'evidence_budget',context(),60,200)
    assert a['local_length_cap_m']==150 and b['local_length_cap_m']==120
    assert selected['id']=='baseline' and a['mission_distance_cap_m']==b['mission_distance_cap_m']==200


def test_same_state_all_methods_share_unchanged_candidates_and_charge_final_verification():
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/local_power.json')
    cfg=replace(cfg,map=replace(cfg.map,buildings=(),obstacles_m=()),electromagnetic=replace(cfg.electromagnetic,fields=()))
    world=ElectromagneticWorld(cfg);belief=ElectromagneticBelief(cfg,world.geometry)
    belief.update(world.observe(cfg.map.start_m,0));snapshot=belief.snapshot(0)
    ctx=build_context(cfg.map.start_m,cfg.map.goal_m,[],snapshot,cfg,100,[],[],None)
    ctx['latest_observation']={'time_seconds':0,'interference_dbm':-90}
    route=[list(cfg.map.start_m),list(cfg.map.goal_m)];settings=ObjectiveSettings(search_samples=3)
    reports=[]
    for method in METHODS:
        plan,report=EvidenceBudgetOptimizer(method,settings).optimize(snapshot,ctx,route,2000,19,0)
        assert report['resources']['verification_edges']>=2
        assert report['resources']['edges']<=settings.edge_limit
        assert report['executed_distance_m']+plan.length_m<=report['mission_distance_cap_m']+1e-6
        reports.append(report)
    assert all(r['candidates']==reports[0]['candidates'] for r in reports)


def test_resource_fallback_reuses_preverified_baseline_without_extra_search(monkeypatch):
    pool=[record('baseline',100,10,5,100),record('detour',120,3,2,120)]
    report={'candidates':pool,'executed_distance_m':0,'mission_distance_cap_m':200,
        'resources':{'edges':10000,'edge_limit':10000,'verification_edges':1,'edges_left':0,'total_edges_left':0}}
    optimizer=EvidenceBudgetOptimizer('evidence_only')
    baseline=PlanResult('rrt_star',path=pool[0]['path'])
    monkeypatch.setattr(optimizer.base,'optimize',lambda *args:(baseline,deepcopy(report)))
    result,trace=optimizer.optimize(None,context(),[],200,1,0)
    assert result.path==pool[0]['path']
    assert trace['selection_resource_fallback'] and trace['resources']['edges']==10000
    assert trace['selected_candidate_id']=='baseline'
