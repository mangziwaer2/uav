from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from uav_em.aligned_search import PowerTimeSearchObjective, AlignedOptimizer, VERSION
from uav_em.balanced_objective import power_time_score, PowerTimeSettings, paired_seed
from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.context import build_context
from uav_em.environment import ElectromagneticWorld
from uav_em.decision_objective import ObjectiveSettings
from uav_em.repair_search import FixedObjective, WorkBudget, BudgetExhausted, repair_rrt_star

@pytest.fixture
def setup():
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/local_power.json')
    cfg=replace(cfg,map=replace(cfg.map,buildings=(),obstacles_m=()),electromagnetic=replace(cfg.electromagnetic,fields=()))
    world=ElectromagneticWorld(cfg);belief=ElectromagneticBelief(cfg,world.geometry)
    belief.update(world.observe(cfg.map.start_m,0));snap=belief.snapshot(0)
    route=[list(cfg.map.start_m),list(cfg.map.goal_m)]
    ctx=build_context(cfg.map.start_m,cfg.map.goal_m,route,snap,cfg,100,[],[],None)
    return cfg,snap,ctx,route

@pytest.mark.parametrize('weight',[0.,1.,4.])
def test_exact_mixed_support_edge_matches_acceptor_and_subdivision(setup,weight):
    _,s,_,_=setup
    rng=np.random.default_rng(42);s.supported[:]=rng.random(s.supported.shape)<.5
    s.estimate_dbm[:]=rng.uniform(-100,-70,s.supported.shape);s.uncertainty_db[:]=3
    settings=PowerTimeSettings(weight=weight);p=PowerTimeSearchObjective(s,15,WorkBudget(),settings)
    a,b=[20.,30.],[410.,210.];m=(np.array(a)+np.array(b))/2
    cost=p.edge_cost(a,b)
    expected=power_time_score(FixedObjective(s,-80,WorkBudget()),[a,b],15,settings)['score_seconds']
    assert cost==pytest.approx(expected)
    assert p.edge_cost(a,m)+p.edge_cost(m,b)==pytest.approx(cost)
    edges=p.budget.edges
    assert p.edge_cost(b,a)==cost and p.budget.edges==edges and p.budget.cache_hits==1

def test_unknown_nan_not_read(setup):
    _,s,_,_=setup;s.supported[:]=False
    p=PowerTimeSearchObjective(s,15,WorkBudget());a,b=[20,20],[300,20];expected=p.edge_cost(a,b)
    s.estimate_dbm[:]=np.nan;s.uncertainty_db[:]=np.nan
    q=PowerTimeSearchObjective(s,15,WorkBudget())
    assert q.edge_cost(a,b)==expected and np.isfinite(q.q).all()

def test_budget_collision_and_invalid_speed(setup,monkeypatch):
    _,s,_,_=setup
    p=PowerTimeSearchObjective(s,15,WorkBudget(edge_limit=1))
    monkeypatch.setattr(s.geometry,'edge_free',lambda a,b:False)
    assert np.isinf(p.edge_cost([20,20],[30,20]))
    assert np.isinf(p.edge_cost([30,20],[20,20])) and p.budget.edges==1
    with pytest.raises(BudgetExhausted):p.edge_cost([20,20],[40,20])
    with pytest.raises(ValueError):PowerTimeSearchObjective(s,0,WorkBudget())

def test_rrt_tree_cost_is_whole_path_C_not_only_final_rescore(setup):
    _,s,_,_=setup;s.supported[:]=True;s.estimate_dbm[:]=-100;s.uncertainty_db[:]=0
    # A synthetic fixture, no development truth or future source data.
    s.estimate_dbm[2:7,6:10]=-65
    p=PowerTimeSearchObjective(s,15,WorkBudget(500,20000))
    result,_=repair_rrt_star(p,[20,20],[250,20],400,42,{'focus_probability':0.,'regions':[]})
    assert result.path
    expected=power_time_score(p.integrator,result.path,15)['score_seconds']
    assert result.cost==pytest.approx(expected)

def test_matched_optimizer_preserves_pool_stream_and_constraints(setup):
    _,s,ctx,route=setup;limits=ObjectiveSettings(search_samples=3,sample_limit=80,edge_limit=1200)
    records=[]
    for objective in ('old_J','power_time_C'):
        opt=AlignedOptimizer(17,objective,limits=limits)
        plan,report=opt.optimize(s,deepcopy(ctx),route,2000,999,0)
        records.append(report)
        assert plan.path and report['final_geometry_verified'] and report['policy_errors']==0
        assert report['resources']['samples']<=80 and report['resources']['edges']<=1200
        assert plan.length_m<=report['mission_distance_cap_m']
        assert report['actual_advisor_calls']==0
        for search in report['searches']:
            assert search['actual_seed']==paired_seed(17,0,search['seed_stage'],search['seed_purpose'])
    a,b=records
    n=len(a['candidates'])-sum(c['id'].startswith('process_candidate_') for c in a['candidates'])
    assert a['candidates'][:n]==b['candidates'][:n]
    assert a['primitive_initial_resources']==b['primitive_initial_resources']
    assert a['trace'][-1]['proposal']==b['trace'][-1]['proposal']
    assert a['searches'][-1]['actual_seed']==b['searches'][-1]['actual_seed']
