"""诊断的真值等价、积分、未来隔离和候选/任务指标分离。"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.context import build_context
from uav_em.environment import ElectromagneticWorld
from uav_em.diagnostic_evaluation import (truth_at_times,route_samples,evaluate_fixed_route,diagnose_candidates)
from uav_em.repair_diagnostics import geometric_candidates,search_candidate,splice_interval
from uav_em.repair_search import FixedObjective,WorkBudget
from uav_em.route_repair import RepairSettings

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def setup():
    cfg=load_config(ROOT/'configs/local_power.json')
    cfg=replace(cfg,map=replace(cfg.map,buildings=(),obstacles_m=()),
                electromagnetic=replace(cfg.electromagnetic,measurement_noise_db=0,fields=({
                    'center_m':[400,300],'peak_dbm':-70,'sigma_m':1e12,
                    'start_seconds':0,'end_seconds':1000,'velocity_mps':[0,0]},)))
    world=ElectromagneticWorld(cfg);belief=ElectromagneticBelief(cfg,world.geometry)
    belief.update(world.observe(cfg.map.start_m,0));snap=belief.snapshot(0)
    ctx=build_context(cfg.map.start_m,cfg.map.goal_m,[],snap,cfg,100,[],[],None)
    return cfg,world,snap,ctx

def test_batched_truth_matches_scalar_original_with_motion_switches_and_reflections():
    cfg=load_config(ROOT/'configs/local_power.json');world=ElectromagneticWorld(cfg)
    times=np.array([0,19.999,20,21,44.999,45,50,120,121.])
    points=np.random.default_rng(11).uniform([0,0],[800,600],(len(times),2))
    expected=[world.truth(p,float(t)) for p,t in zip(points,times)]
    np.testing.assert_allclose(truth_at_times(world,points,times),expected,atol=1e-10)

def test_midpoint_integration_has_exact_flight_duration_and_constant_field_exposure(setup):
    cfg,world,snap,_=setup;route=[[40,300],[140,300],[140,400]]
    p,t,w=route_samples(route,cfg.aircraft.speed_mps,.25)
    duration=200/cfg.aircraft.speed_mps
    assert w.sum()==pytest.approx(duration)
    assert np.all(np.diff(t)>0) and np.max(w)<=.25
    e=evaluate_fixed_route(world,snap,route)
    assert e['future_exposure_seconds']==pytest.approx(duration)
    assert e['future_excess_db_seconds']==pytest.approx((world.truth([40,300],0)+80)*duration)
    assert e['departure_time_seconds']==.5 and e['new_observations']==0

def test_splice_preserves_all_outer_points_and_endpoints():
    route=[[0,0],[10,0],[20,0],[30,0],[40,0]]
    result=splice_interval(route,1,3,[[10,0],[15,10],[30,0]])
    assert result==[[0.,0.],[10.,0.],[15.,10.],[30.,0.],[40.,0.]]

def test_pool_is_deterministic_and_does_not_mutate_snapshot_context_or_route(setup):
    cfg,_,snap,ctx=setup;route=[list(cfg.map.start_m),[200,450],list(cfg.map.goal_m)]
    old_ctx=deepcopy(ctx);old_route=deepcopy(route);old_est=snap.estimate_dbm.copy()
    one,public,budget=geometric_candidates(snap,ctx,route,2000)
    two,_,_=geometric_candidates(snap,ctx,route,2000)
    assert one==two and ctx==old_ctx and route==old_route
    np.testing.assert_array_equal(snap.estimate_dbm,old_est)
    assert all(r['path'][0]==route[0] and r['path'][-1]==route[-1] for r in one)
    assert 'source_inference' not in public and budget['samples']==0

def test_rejected_search_candidate_is_retained_and_recomputed_metrics_match(setup):
    cfg,_,snap,ctx=setup;route=[list(cfg.map.start_m),list(cfg.map.goal_m)]
    snap.estimate_dbm[:]=-100;snap.uncertainty_db[:]=0
    record=search_candidate(snap,ctx,route,2000,'global_uniform',41,
                            RepairSettings(max_rounds=1,sample_limit=200,edge_limit=10000))
    assert record['path'] and record['feasible'] and not record['accepted']
    assert record['report']['actual_advisor_calls']==0 and record['report']['policy_errors']==0
    actual=FixedObjective(snap,-80,WorkBudget()).route_metrics(record['path'])
    assert actual==pytest.approx(record['metrics'])
    assert record['report']['resources']['samples']<=200
    assert record['report']['resources']['edges']<=10000

def test_diagnosis_does_not_confuse_lower_j_with_lower_true_exposure():
    def record(identifier,j,length,exposure,family='geometric'):
        return dict(id=identifier,family=family,metrics={'objective':j,'length_m':length},
                    evaluation={'future_excess_db_seconds':exposure},feasible=True)
    records=[record('incumbent',100,100,100,'incumbent'),record('short',95,95,120,'rrt_search'),
             record('detour',110,120,50),record('too_long',120,130,0)]
    d=diagnose_candidates(records)
    assert d['best_current_j_id']=='short' and d['posthoc_lowest_exposure_id']=='detour'
    assert d['material_improvements_rejected_by_current_j_ids']==['detour']
    assert d['acceptance_task_tradeoff_evidence'] and not d['finite_pool_search_coverage_gap']

@pytest.mark.parametrize('speed,step',[(0,.25),(10,0),(float('nan'),.25)])
def test_invalid_integration_arguments_are_rejected(speed,step):
    with pytest.raises(ValueError):route_samples([[0,0],[10,0]],speed,step)
