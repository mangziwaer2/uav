"""短段风险、未知边界、全程航程和共同候选的关键不变量。"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.context import build_context
from uav_em.environment import ElectromagneticWorld
from uav_em.repair_search import FixedObjective,WorkBudget
from uav_em.decision_objective import ObjectiveSettings,ObjectiveOptimizer,cut_route,prefix_risk,select_candidate

@pytest.fixture
def setup():
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/local_power.json')
    cfg=replace(cfg,map=replace(cfg.map,buildings=(),obstacles_m=()),electromagnetic=replace(cfg.electromagnetic,fields=()))
    world=ElectromagneticWorld(cfg);snap=ElectromagneticBelief(cfg,world.geometry).snapshot(0)
    ctx=build_context(cfg.map.start_m,cfg.map.goal_m,[],snap,cfg,100,[],[],None)
    return cfg,snap,ctx


def test_cut_preserves_bend_and_zero_length_and_goal():
    route=[[0,0],[30,0],[30,0],[30,40]]
    a,b=cut_route(route,50)
    assert a==[[0,0],[30.,0.],[30.,20.]] and b==[[30.,20.],[30,40]]
    assert cut_route(route,100)[1]==[[30,40]]
    assert cut_route(route,0)[0]==[[0,0]]


def test_unknown_risk_is_independent_of_hidden_numeric_prior(setup):
    _,snap,_=setup;problem=FixedObjective(snap,-80,WorkBudget())
    route=[[40,300],[760,300]]
    first=prefix_risk(problem,route,15)
    snap.estimate_dbm[:]=100;snap.uncertainty_db[:]=100
    assert prefix_risk(problem,route,15)==first
    assert first['unknown_seconds']==pytest.approx(10)
    assert first['risk_db_seconds']==pytest.approx(100)
    assert first['supported_expected_excess_db_seconds']==0


def test_short_horizon_ignores_distant_cells_and_keeps_units(setup):
    _,snap,_=setup;snap.supported[:]=True;snap.estimate_dbm[:]=-70;snap.uncertainty_db[:]=0
    problem=FixedObjective(snap,-80,WorkBudget());route=[[40,300],[760,300]]
    before=snap.estimate_dbm.copy();risk=prefix_risk(problem,route,15)
    assert risk['risk_db_seconds']==pytest.approx(100)
    np.testing.assert_array_equal(before,snap.estimate_dbm)
    snap.estimate_dbm[:,20:]=50
    assert prefix_risk(problem,route,15)==risk
    split=prefix_risk(problem,[[40,300],[113,300],[760,300]],15)
    for k in ('risk_db_seconds','supported_seconds','unknown_seconds','tail_length_m'):
        assert split[k]==pytest.approx(risk[k])
    assert prefix_risk(problem,[[40,300],[115,300]],15)['risk_db_seconds']==pytest.approx(50)


def test_risk_tolerance_uses_global_minimum_not_order_dependent_drift():
    def rec(name,risk,length):
        return dict(id=name,feasible=True,prefix={'risk_db_seconds':risk},metrics={'length_m':length,'objective':length})
    records=[rec('low',0,100),rec('middle',.8,90),rec('high',1.6,80)]
    assert select_candidate(records,'prefix_v2')['id']=='middle'
    assert select_candidate(records[::-1],'prefix_v2')['id']=='middle'
    records[1]['feasible']=False
    assert select_candidate(records,'prefix_v2')['id']=='low'


def test_mission_cap_is_fixed_and_actual_execution_is_deducted(setup):
    cfg,snap,ctx=setup
    opt=ObjectiveOptimizer('pool_j',ObjectiveSettings(search_samples=2))
    route=[list(cfg.map.start_m),list(cfg.map.goal_m)]
    p,r=opt.optimize(snap,ctx,route,2000,1,0)
    assert r['mission_distance_cap_m']==pytest.approx(900)
    snap.time_seconds=10;ctx=deepcopy(ctx);ctx['vehicle']['position_m']=[190,300]
    ctx['execution_feedback']={'interval':{'flown_m':150}}
    p2,r2=opt.optimize(snap,ctx,[[190,300],[760,300]],2000,2,0)
    assert r2['mission_distance_cap_m']==r['mission_distance_cap_m']
    assert r2['executed_distance_m']==150
    assert r2['executed_distance_m']+p2.length_m<=r2['mission_distance_cap_m']+1e-7
    assert r2['resources']['samples']<=2100 and r2['resources']['edges']<=10000
    with pytest.raises(ValueError,match='advancing'):opt.optimize(snap,ctx,p2.path,2000,2,0)


def test_same_snapshot_methods_receive_identical_candidate_pool(setup):
    cfg,snap,ctx=setup;route=[list(cfg.map.start_m),list(cfg.map.goal_m)]
    args=(snap,ctx,route,2000,17,0);settings=ObjectiveSettings(search_samples=5)
    _,a=ObjectiveOptimizer('pool_j',settings).optimize(*args)
    _,b=ObjectiveOptimizer('prefix_v2',settings).optimize(*args)
    assert a['candidates']==b['candidates']
    assert a['resources']==b['resources']
    assert a['mission_reference_length_m']==b['mission_reference_length_m']

@pytest.mark.parametrize('kwargs',[{'horizon_seconds':0},{'unknown_excess_db':float('nan')},
                                  {'risk_tolerance_db_seconds':-1},{'mission_length_ratio':.9},
                                  {'search_samples':2101}])
def test_invalid_objective_settings_rejected(kwargs):
    with pytest.raises(ValueError):ObjectiveSettings(**kwargs)
