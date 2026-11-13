from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from uav_em.reconnection_search import EdgeSlice,reconnections,select_arm,run_competition,ReconnectionOptimizer,MODES,VERSION
from uav_em.balanced_objective import BalancedOptimizer,PowerTimeSettings,choose,paired_seed
from uav_em.repair_search import WorkBudget,BudgetExhausted
from uav_em.decision_objective import ObjectiveSettings
from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.environment import ElectromagneticWorld
from uav_em.context import build_context

@pytest.fixture
def setup():
 cfg=load_config(Path(__file__).resolve().parents[1]/'configs/local_power.json')
 cfg=replace(cfg,map=replace(cfg.map,buildings=(),obstacles_m=()),electromagnetic=replace(cfg.electromagnetic,fields=()))
 world=ElectromagneticWorld(cfg);belief=ElectromagneticBelief(cfg,world.geometry)
 belief.update(world.observe(cfg.map.start_m,0));s=belief.snapshot(0)
 route=[list(cfg.map.start_m),list(cfg.map.goal_m)]
 ctx=build_context(cfg.map.start_m,cfg.map.goal_m,route,s,cfg,100,[],[],None)
 return cfg,s,ctx,route

def test_slice_cannot_reset_shared_work_or_consume_next_batch():
 w=WorkBudget(10,100,edges=30,reserved_edges=10);local=EdgeSlice(w,2)
 local.take_sample();local.take_edge();local.take_edge();local.cache_hits+=1
 assert (w.samples,w.edges,w.cache_hits)==(1,32,1)
 with pytest.raises(BudgetExhausted,match='batch_edge'):local.take_edge()
 assert w.edge_limit==100 and w.reserved_edges==10
 other=EdgeSlice(w,99)
 w.edges=90
 with pytest.raises(BudgetExhausted,match='^edge_budget'):other.take_edge()

@pytest.mark.parametrize('v',[-1,1.5,True])
def test_invalid_slice_rejected(v):
 with pytest.raises(ValueError):EdgeSlice(WorkBudget(),v)

def test_forward_options_and_duplicate_terminal_are_removed():
 points=[{'id':f'a{i}','position_m':[float(i)*50,20.]} for i in range(7)]
 ctx={'route_repair':{'anchors':points,'incumbent_feasible':True,'segments':[
  {'index':i,'cost':{'length_m':50.,'interference_cost':100. if i==2 else 0.},'supported_fraction':1.} for i in range(6)]}}
 arms=reconnections(ctx)
 assert [(a['start_index'],a['end_index']) for a in arms]==[(1,4),(1,6)]
 assert all(a['end_index']>a['start_index']>=0 for a in arms)

def test_feedback_compares_full_route_cost_and_explores_first():
 arms=[dict(id=f'e{i}',end_index=i+1) for i in range(3)]
 h=[dict(arm_id='e0',candidate_C_seconds=100.,search_samples=20),dict(arm_id='e1',candidate_C_seconds=60.,search_samples=30),dict(arm_id='e2',candidate_C_seconds=None,search_samples=10)]
 assert select_arm('multi_feedback',arms,h,2)==arms[2]
 assert select_arm('multi_feedback',arms,h,3)==arms[1]
 assert select_arm('multi_equal',arms,h,3)==arms[0]
 assert select_arm('multi_feedback',arms,[dict(x,candidate_C_seconds=None) for x in h],3)==arms[2]

def test_shared_initial_pool_forward_splicing_seeds_and_batch_caps(setup):
 _,s,ctx,route=setup
 limits=ObjectiveSettings(search_samples=3,sample_limit=100,edge_limit=2500)
 _,pool=BalancedOptimizer(17,limits=limits).optimize(s,deepcopy(ctx),route,2000,999,0)
 results=[]
 for mode in MODES:
  r=run_competition(s,ctx,pool,2000,17,mode,limits=limits);results.append(r)
  assert r['resources']['samples']<=100 and r['resources']['edges']<=2500
  assert r['resources']['samples']-pool['resources']['samples']<=700
  assert r['model_calls']==r['policy_errors']==r['batch_errors']==0
  assert r['path'][0]==ctx['vehicle']['position_m'] and r['path'][-1]==ctx['vehicle']['goal_m']
  assert choose(r['candidates'])['balance']==r['selected_balance']
  base=r['frozen_base_route']
  for t,search,candidate in zip(r['competition_trace'],r['searches'],r['competition_candidates']):
   assert t['search_edges']<=t['edge_quota'] and t['search_samples']<=t['sample_quota']
   assert search['actual_seed']==paired_seed(17,0,VERSION,search['seed_purpose'])
   assert search['path']==[] or np.allclose(search['path'][-1],base[t['end_index']])
   assert t['best_C_seconds']<=t['previous_best_C_seconds']+1e-8
 assert all(r['initial_resources']==results[0]['initial_resources'] for r in results)
 assert len({r['searches'][0]['actual_seed'] for r in results})==1
 assert [h['arm_id'] for h in results[2]['competition_trace'][:3]]==[h['arm_id'] for h in results[3]['competition_trace'][:3]]

def test_flight_optimizer_preserves_paid_pool_and_reports(setup):
 _,s,ctx,route=setup;limits=ObjectiveSettings(search_samples=3,sample_limit=50,edge_limit=1200)
 plan,r=ReconnectionOptimizer(17,'multi_feedback',limits=limits).optimize(s,ctx,route,2000,77,0)
 assert plan.path and r['final_geometry_verified'] and plan.cost==r['selected_balance']['score_seconds']
 assert plan.length_m<=r['mission_distance_cap_m'] and r['actual_advisor_calls']==0
 assert r['primitive_initial_resources']==r['starting_resources_already_paid']
 assert r['numeric_compute_seconds']>=0 and r['searches'][0]['seed_stage']=='common_pool'
