"""Fixed objective, causal segment repair, search budgets and feedback acceptance."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import numpy as np
import pytest
from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.context import build_context
from uav_em.environment import ElectromagneticWorld
from uav_em.planning import path_length
from uav_em.repair_search import WorkBudget, BudgetExhausted, FixedObjective, sample_point, repair_rrt_star
from uav_em.route_repair import (RepairSettings, RepairSession, RouteRepairOptimizer, NumericalRepairPolicy,
                               anchors, parse_proposal, repair_response, UNIFORM)
from uav_em.simulation import simulate
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def setup():
    cfg=load_config(ROOT/'configs/local_power.json')
    cfg=replace(cfg,map=replace(cfg.map,columns=21,rows=16,start_m=(20,140),goal_m=(380,140),buildings=(),obstacles_m=()),
                electromagnetic=replace(cfg.electromagnetic,fields=(),measurement_noise_db=0),
                simulation=replace(cfg.simulation,max_seconds=35))
    world=ElectromagneticWorld(cfg);belief=ElectromagneticBelief(cfg,world.geometry);belief.update(world.observe(cfg.map.start_m,0))
    snap=belief.snapshot(0);snap.estimate_dbm[:]=-100.;snap.uncertainty_db[:]=0.;snap.supported[:]=True
    ctx=build_context(cfg.map.start_m,cfg.map.goal_m,[],snap,cfg,100,[],[],None)
    return cfg,snap,ctx


def test_exact_objective_is_invariant_to_collinear_split_and_direction(setup):
    _,snap,_=setup;snap.estimate_dbm[:,5:12]=-60;snap.uncertainty_db[:,8:16]=15
    p=FixedObjective(snap,-80,WorkBudget(20,100))
    a=[20,37];b=[380,263];m=(np.array(a)*.63+np.array(b)*.37).tolist()
    whole=p.route_metrics([a,b]);split=p.route_metrics([a,m,b]);reverse=p.route_metrics([b,a])
    for key in whole:
        assert whole[key]==pytest.approx(split[key],abs=1e-9)
        assert whole[key]==pytest.approx(reverse[key],abs=1e-9)
    assert whole['objective']==pytest.approx(whole['length_m']+whole['interference_cost']+whole['uncertainty_cost'])


def test_sampling_mixture_keeps_global_exploration_and_normalizes_regions():
    sampling={'focus_probability':.6,'regions':[{'low_m':[0,0],'high_m':[1,1],'weight':1.},
                                                {'low_m':[8,8],'high_m':[9,9],'weight':3.}]}
    rng=np.random.default_rng(41);counts={'goal':0,'focus':0,'global':0}
    for _ in range(10000):
        p,kind=sample_point(rng,[10,10],[10,10],sampling);counts[kind]+=1
        assert np.all(p>=0) and np.all(p<=10)
    assert .28<counts['global']/10000<.32
    assert .58<counts['focus']/10000<.62


def test_shortcut_replaces_only_selected_interval_and_rejects_energy_violation(setup):
    _,snap,ctx=setup
    route=[[20,140],[80,200],[160,200],[220,140],[380,140]]
    session=RepairSession(snap,ctx,route,1000,RepairSettings(anchor_spacing_m=1000),5)
    p={'route_version':0,'operator':'shortcut','start_id':'a1','end_id':'a3','reason':'shorten'}
    parsed=parse_proposal(p,session.request(1,3));feedback=session.apply(parsed)
    assert feedback['accepted']
    assert session.best==[[20,140],[80,200],[220,140],[380,140]]
    assert feedback['candidate_minus_previous_best']<0
    assert feedback['preserves_route_outside_interval']
    # Energy is checked even for a new global candidate; infeasible old routes do not become incumbents.
    limited=RepairSession(snap,ctx,route,100,RepairSettings(),5)
    assert limited.best is None
    f=limited.apply({'operator':'global_rrt','sample_budget':600,'sampling':deepcopy(UNIFORM)})
    assert not f['accepted'] and f['reason']=='energy_budget_exceeded'


def test_inferred_strong_region_shortcut_is_rejected_even_when_shorter(setup):
    _,snap,ctx=setup;snap.estimate_dbm[5:10,7:14]=-40
    route=[[20,140],[140,40],[280,40],[380,140]]
    session=RepairSession(snap,ctx,route,2000,RepairSettings(anchor_spacing_m=1000),4)
    old=deepcopy(session.best)
    p={'operator':'shortcut','start_id':'a0','end_id':'a3'}
    f=session.apply(p)
    assert f['candidate_metrics']['length_m']<f['previous_best_metrics']['length_m']
    assert not f['accepted'] and f['reason']=='no_objective_improvement'
    assert session.best==old


def test_rrt_uses_shared_budget_returns_valid_solution_and_matches_fixed_cost(setup):
    _,snap,_=setup;work=WorkBudget(500,10000);p=FixedObjective(snap,-80,work)
    result,stats=repair_rrt_star(p,[20,140],[380,140],500,19,UNIFORM)
    assert result.path and result.cost==pytest.approx(p.route_metrics(result.path)['objective'])
    assert result.iterations==work.samples==sum(stats.values())<=500
    assert work.edges<=10000
    assert result.path[0]==[20,140] and result.path[-1]==[380,140]
    before=work.samples
    other,_=repair_rrt_star(p,[20,140],[380,140],500,20,UNIFORM)
    assert work.samples==before # no extra allotment for a second call
    assert not other.path


def test_edge_budget_exhaustion_does_not_charge_beyond_cap(setup):
    _,snap,_=setup;work=WorkBudget(1000,17);p=FixedObjective(snap,-80,work)
    result,_=repair_rrt_star(p,[20,140],[380,140],1000,1,UNIFORM)
    assert work.edges==17 and work.samples<1000
    assert 'edge_budget_exhausted' in result.reason
    with pytest.raises(BudgetExhausted):p.edge_cost([0,0],[1,1])


def test_via_legs_and_final_verification_share_one_session_budget(setup):
    _,snap,ctx=setup;route=[[20,140],[20,260],[380,260],[380,140]]
    class Policy:
        def decide(self,c):
            r=c['route_repair'];return json.dumps(dict(route_version=r['route_version'],operator='via',
                start_id='a0',end_id=r['anchors'][-1]['id'],waypoint_m=[200,100],sample_budget=600,
                sampling=UNIFORM,reason='try shorter route'))
    settings=RepairSettings(max_rounds=2,sample_limit=700,edge_limit=20000)
    result,report=RouteRepairOptimizer(Policy(),settings).optimize(snap,ctx,route,2000,44,2)
    assert len(report['searches'])==2 and report['resources']['samples']==600
    assert report['actual_advisor_calls']==2 and report['advisor_failures']==1 # second excessive request rejected
    assert report['resources']['edges']<=20000 and report['resources']['verification_edges']==len(result.path)-1
    assert not report['numeric_recovery']
    assert report['trace'][0]['feedback']['accepted']
    assert report['best_metrics']['objective']<report['initial_metrics']['objective']


@pytest.mark.parametrize('edit',[
 lambda p:p.update(route_version=True),lambda p:p.update(route_version=1),lambda p:p.update(profile='explore'),
 lambda p:p.update(start_id='missing'),lambda p:p.update(sample_budget=True),
 lambda p:p.update(sampling={'focus_probability':.9,'regions':[]}),
 lambda p:p.update(waypoint_m=[float('nan'),0]),])
def test_invalid_proposals_do_not_modify_objective_or_resources(setup,edit):
    _,snap,ctx=setup;s=RepairSession(snap,ctx,[[20,140],[380,140]],1000,RepairSettings(),1)
    c=s.request(1,3);p=dict(route_version=0,operator='local_rrt',start_id='a0',end_id='a1',sample_budget=10,sampling=UNIFORM,reason='test')
    edit(p);before=s.work.summary()
    with pytest.raises((ValueError,TypeError)):parse_proposal(p,c)
    assert s.work.summary()==before


def test_frozen_snapshot_and_actual_numeric_feedback_are_given_to_next_round(setup):
    _,snap,ctx=setup;route=[[20,140],[80,240],[240,240],[380,140]];seen=[]
    class Policy:
        def decide(self,c):
            seen.append(deepcopy(c));r=c['route_repair']
            if len(seen)==1:return json.dumps(dict(route_version=r['route_version'],operator='shortcut',start_id='a0',end_id=r['anchors'][-1]['id'],reason='shorten'))
            return json.dumps(dict(route_version=r['route_version'],operator='keep',reason='accept feedback'))
    result,report=RouteRepairOptimizer(Policy()).optimize(snap,ctx,route,2000,7,3)
    assert len(seen)==2 and report['trace'][0]['feedback']['accepted']
    assert seen[1]['route_repair']['route_version']==1
    assert seen[1]['route_repair']['previous_feedback'][0]['candidate_minus_previous_best']<0
    assert seen[0]['measurement_evidence']==seen[1]['measurement_evidence']
    assert seen[0]['time_seconds']==seen[1]['time_seconds']==0
    assert all(not k in seen[0] for k in ('source_inference','evaluation_only_config','fields'))
    assert snap.time_seconds==0
    assert result.length_m==pytest.approx(360)


def test_invalid_model_response_is_not_repaired_and_recovery_has_remaining_budget(setup):
    _,snap,ctx=setup
    class Invalid:
        last_call={'backend':'synthetic_invalid_test'}
        def decide(self,c):return '{invalid'
    result,report=RouteRepairOptimizer(Invalid(),RepairSettings(max_rounds=1,sample_limit=500)).optimize(snap,ctx,[],1000,7,1)
    assert report['advisor_failures']==1 and report['numeric_recovery']
    assert report['trace'][0]['raw_response']=='{invalid'
    assert report['trace'][0]['proposal'] is None
    assert report['resources']['samples']<=500 and result.path


def test_closed_loop_repair_counts_calls_feedback_and_never_escapes_budget(setup):
    cfg,_,_=setup
    ep=simulate(cfg,initial_planner='rrt_star',advisor_name='numeric_repair',route_optimizer=RouteRepairOptimizer(
        settings=RepairSettings(sample_limit=600,edge_limit=5000)))
    assert ep['metrics']['success'] and ep['metrics']['advisor_calls']==0
    for d in ep['decisions']:
        r=d['repair_optimization']
        assert r['resources']['samples']<=600 and r['resources']['edges']<=5000
        assert r['final_geometry_verified']
        assert d['accepted_route_balanced_cost'] is None
    assert ep['metrics']['repair_sample_proposals']==ep['metrics']['rrt_sample_proposals']
    assert ep['metrics']['repair_sessions']==len(ep['decisions'])
    json.dumps(ep,allow_nan=False)


def test_call_limit_zero_makes_no_model_calls_and_explicit_numeric_recovery(setup):
    _,snap,ctx=setup
    class Never:
        last_call={'backend':'stale_previous_call'}
        def decide(self,c):raise AssertionError('should not consult model')
    opt=RouteRepairOptimizer(Never(),RepairSettings(sample_limit=500))
    result,report=opt.optimize(snap,ctx,[],1000,17,0)
    assert result.path and report['call_budget_exhausted'] and report['numeric_recovery']
    assert report['actual_advisor_calls']==report['advisor_failures']==0
    assert opt.last_call=={}
    assert report['trace'][0]['backend']=='numeric_recovery'


def test_keep_retains_incumbent_with_only_reserved_verification_work(setup):
    _,snap,ctx=setup;route=[[20,140],[380,140]]
    class Keep:
        def decide(self,c):return json.dumps(dict(route_version=c['route_repair']['route_version'],operator='keep',reason='keep feasible'))
    result,report=RouteRepairOptimizer(Keep(),RepairSettings(sample_limit=5,edge_limit=10,anchor_spacing_m=1000)).optimize(snap,ctx,route,1000,1,3)
    assert result.path==route
    assert report['resources']['edges']==2 # once initial cost, once final independent geometry check
    assert report['resources']['samples']==0 and report['accepted_proposals']==0
    assert not report['numeric_recovery']


def test_file_repair_policy_preserves_feedback_and_strictly_validates_proposals(setup,tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import time
    from uav_em.agent_navigation import save_json
    from uav_em.route_repair import FileRepairPolicy
    _,snap,ctx=setup;s=RepairSession(snap,ctx,[[20,140],[380,140]],1000,RepairSettings(),1)
    c=s.request(1,3)
    (tmp_path/'contract.md').write_text('Synthetic causal repair interface test')
    policy=FileRepairPolicy(tmp_path,'m',2)
    def respond():
        p=tmp_path/'requests/m_001.json'
        while not p.exists():time.sleep(.01)
        req=json.loads(p.read_text());assert req['visible_context']['route_repair']==c['route_repair']
        assert 'source_inference' not in req['visible_context']
        save_json(req['response_path'],{'request_id':req['request_id'],'decision':{
            'route_version':0,'operator':'keep','reason':'keep'},'strategy':{'mode':'baseline'},'memory':'measured only'})
    with ThreadPoolExecutor() as pool:
        f=pool.submit(respond);raw=policy.decide(c);f.result()
    assert parse_proposal(raw,c)['operator']=='keep'
    assert policy.last_call['status']=='accepted'
    assert policy.last_call['raw_agent_response']==(tmp_path/'responses/m_001.json').read_text()


def test_timeout_response_audit_preserves_failure_without_requiring_response_file(tmp_path):
    import importlib.util
    import sys
    sys.path.insert(0,str(ROOT/'tools'))
    spec=importlib.util.spec_from_file_location('repair_runner',ROOT/'tools/run_repair_study.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    public=tmp_path;'requests' # no response file is the actual timeout state
    (public/'requests').mkdir();ctx={'route_repair':{'round':1}}
    request=public/'requests/m_001.json';request.write_text(json.dumps({'visible_context':ctx}))
    runtime={'request_id':'m_001','request_sha256':runner.sha(request),'status':'error','error':'TimeoutError'}
    audit=runner.audit_exchange(public,runtime,ctx)
    assert audit['input_sha'] and audit['context_matches']
    assert audit['response_presence_consistent'] and audit['response_record_consistent']
    assert audit['model_runtime_status']=='error'
    assert audit['missing_response_is_failure_not_model_success']
