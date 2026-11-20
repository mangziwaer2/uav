from copy import deepcopy
import math

import numpy as np
import pytest

from uav_em.equivalent_field import kernel,normalized_expected_power,power_and_jacobian
from uav_em.event_inference import digest
from uav_em.scenario_planning import arrival_queries
from uav_em.task_contrast import RateProblem,audit,keep,make_request,numerical_choice,profile,validate


def fixture(peak=-83.):
    now=3.;models=[dict(id='rising',family='strength_change',weight=.8,std_db=4.,reference_time_seconds=0.,born_index=0,
        bases=[dict(center_m=[50.,0.],scale_m=20.,velocity_mps=[0.,0.],rate_db_per_second=.5)],peaks_dbm=[peak]),
        dict(id='static',family='static',weight=.2,std_db=4.,reference_time_seconds=0.,born_index=0,
        bases=[dict(center_m=[50.,0.],scale_m=20.,velocity_mps=[0.,0.],rate_db_per_second=0.)],peaks_dbm=[peak])]
    history=[]
    for t in range(4):
        xy=[t*5.,0.];mu=power_and_jacobian(kernel(models[0]['bases'],[xy],[t],0.),models[0]['peaks_dbm'])[0][0]
        history.append(dict(position_m=xy,time_seconds=float(t),interference_dbm=float(mu)))
    soft=dict(position_m=[[15.,0.],[15.,8.]],time_seconds=[now,now],weights=[.5,.5],prefix_end_seconds=now,independent_evidence_points=0)
    soft['mean_dbm']=power_and_jacobian(kernel(models[0]['bases'],soft['position_m'],soft['time_seconds'],0.),models[0]['peaks_dbm'])[0].tolist()
    packet=dict(time_seconds=now,history=history,state={},known_map={},fits=[dict(gp_soft_constraints=soft)],recent_source_predictions=[],recent_joint_predictions=[],consultation_reasons=[])
    candidates=[]
    for cid,path in [('a',[[15.,0.],[100.,0.]]),('b',[[15.,0.],[50.,40.],[100.,0.]]),('a_copy',[[15.,0.],[100.,0.]])]:
        q=arrival_queries(path,now,10.);g=np.full(len(q['time_seconds']),.3);ep=[]
        for m in models:
            mu=power_and_jacobian(kernel(m['bases'],q['position_m'],q['time_seconds'],0.),m['peaks_dbm'])[0]
            ep.append(g*1.3+(1-g)*normalized_expected_power(mu,m['std_db'],-80.))
        ep=np.asarray(ep);perq=ep@q['dt_seconds'];weights=np.asarray([m['weight'] for m in models]);T=q['duration_seconds']
        score=dict(score_seconds=T+float(weights@perq),remaining_flight_seconds=T,expected_normalized_power=ep.tolist(),
            gp_weight=g.tolist(),queries=q,scenario_scores=[dict(id=m['id'],weight=m['weight'],power_proxy_seconds=float(Q)) for m,Q in zip(models,perq)])
        candidates.append(dict(id=cid,path=path,feasible=True,score=score))
    selected=min(candidates,key=lambda c:(c['score']['score_seconds'],sum(math.dist(a,b) for a,b in zip(c['path'],c['path'][1:])),c['id']))
    record=dict(time_seconds=now,candidates=candidates,selected=selected,source_models=models)
    req=make_request(packet,record);plan=numerical_choice(req)
    return req,record,plan


def rebind(req):req['request_id']=digest({k:v for k,v in req.items() if k!='request_id'})


def test_cache_bins_conserve_integrals():
    req,record,_=fixture();a=audit(record)
    for p in a['pairs']:
        assert p['delta_c']==pytest.approx(p['delta_t']+sum(w['weighted_delta_q'] for w in p['worlds']))
        for w in p['worlds']:assert sum(w['delta_q_by_arrival_bin'])==pytest.approx(w['delta_q'])
    assert a['arrival_bins_absolute_seconds']==[[3.,8.],[8.,13.],[13.,23.],[23.,None]]
    # Identical path has identical immediate action, not an invented action split.
    assert any(p['next_15m_distance_m']==0 for p in a['pairs']) if a['route_a']=='a' else True
    record['candidates'][0]['score']['expected_normalized_power'][0][0]+=1
    with pytest.raises(ValueError,match='integral'):audit(record)


def test_rate_analytic_gradients_and_gp_cancellation():
    req,record,plan=fixture();p=RateProblem(req,record,plan)
    p.evaluate(p.seed);x=.32;h=1e-6
    _,loss,grad,q,dq=p.evaluate(x);lo=p.evaluate(x-h);hi=p.evaluate(x+h)
    assert grad==pytest.approx((hi[1]-lo[1])/(2*h),rel=1e-6,abs=1e-6)
    assert dq==pytest.approx((hi[3]-lo[3])/(2*h),rel=1e-6)
    # Full direct joint expectation agrees with the saved GP-cancelling difference.
    m=deepcopy(p.model);m['bases'][plan['basis_index']]['rate_db_per_second']=x
    totals=[]
    for s in p.routes:
        queries=s['queries'];mu=power_and_jacobian(kernel(m['bases'],queries['position_m'],queries['time_seconds'],m['reference_time_seconds']),m['peaks_dbm'])[0]
        g=np.asarray(s['gp_weight']);ep=g*1.3+(1-g)*normalized_expected_power(mu,m['std_db'],-80.)
        totals.append(float(ep@queries['dt_seconds']))
    assert p.world['delta_c']+q-p.baseline['qpart']==pytest.approx(p.pair['delta_t']+totals[1]-totals[0])


def test_profile_preserves_bank_and_keeps_only_compatible_witnesses():
    req,record,plan=fixture();before=digest([req,record]);r=profile(req,record,plan)
    assert digest([req,record])==before
    assert r['resources']['parameter_evaluations']<=40
    assert r['resources']['objective_calls']<=40
    assert r['resources']['gradient_calls']<=40
    assert r['resources']['kernel_terms']<=20000
    assert r['resources']['solver_calls']==2
    assert r['births']==r['route_changes']==r['weight_changes']==0
    assert not r['interval_is_global_bound']
    for e in r['endpoints']:
        assert e['feasible'] and -.5<=e['rate_db_per_second']<=.5
        assert e['real_mse_db2']<=r['baseline_real_mse_db2']+1.+1e-7
        assert e['gp_mse_db2']<=r['baseline_gp_mse_db2']+1.+1e-7
    assert any(abs(e['rate_db_per_second']-r['seed_rate_db_per_second'])>1e-5 for e in r['endpoints'])
    for key in ('world','frozen_weight'):
        lo,hi=r[key+'_delta_c_interval_seconds']
        assert r[key+'_ranking_crossed']==(lo < -1e-7 and hi>1e-7)


def test_budget_exhaustion_retains_seed_and_no_extra_eval():
    req,record,plan=fixture();req['operation']['max_parameter_evaluations']=1;rebind(req);plan['request_id']=req['request_id']
    r=profile(req,record,plan)
    assert r['resources']['parameter_evaluations']==1
    assert all(e['rate_db_per_second']==r['seed_rate_db_per_second'] for e in r['endpoints'])
    assert any(not s['success'] and 'budget exhausted' in s.get('error','') for s in r['solvers'])
    assert not r['world_ranking_crossed'] and not r['frozen_weight_ranking_crossed']


def test_keep_has_no_tools(monkeypatch):
    req,record,_=fixture()
    monkeypatch.setattr('uav_em.task_contrast.RateProblem',lambda *a:pytest.fail('keep queried tool'))
    assert profile(req,record,keep(req,'unneeded')) is None


@pytest.mark.parametrize('key,value',[('world_id','static'),('basis_index',True),('basis_index',4),('route_b','missing'),('request_id','wrong'),('action','spawn'),('reason','')])
def test_invalid_plan_rejected(key,value):
    req,_,plan=fixture();plan[key]=value
    with pytest.raises(ValueError):validate(req,plan)


def test_duplicate_json_and_mutation_rejected():
    req,_,plan=fixture()
    with pytest.raises(ValueError,match='duplicate'):validate(req,'{"action":"keep","action":"profile_rate"}')
    req['history'][0]['time_seconds']=999
    with pytest.raises(ValueError,match='changed'):validate(req,plan)


def test_noncausal_soft_points_rejected():
    req,record,_=fixture()
    packet=dict(time_seconds=3.,history=req['history'],fits=[dict(gp_soft_constraints=dict(req['gp_soft_constraints'],prefix_end_seconds=4.))],recent_source_predictions=[],recent_joint_predictions=[])
    with pytest.raises(ValueError,match='GP'):make_request(packet,record)


@pytest.mark.parametrize('corruption',['model','recent','history'])
def test_causal_public_validation(corruption):
    req,record,_=fixture();packet=deepcopy(req);packet['fits']=[dict(gp_soft_constraints=req['gp_soft_constraints'])]
    if corruption=='model':record['source_models'][0]['reference_time_seconds']=4.
    elif corruption=='history':packet['history'][1]['time_seconds']=0.
    else:packet['recent_joint_predictions']=[dict(issued=dict(time_seconds=4.),observation=dict(time_seconds=4.))]
    with pytest.raises(ValueError,match='noncausal'):make_request(packet,record)


def test_stale_cache_record_rejected():
    req,record,plan=fixture();record['candidates'][0]['score']['gp_weight'][0]=.9
    with pytest.raises(ValueError,match='binding'):profile(req,record,plan)


def test_successful_solver_cannot_promote_incompatible_trial(monkeypatch):
    from types import SimpleNamespace
    req,record,plan=fixture()
    def fake_solver(fun,x,**kwargs):
        fun(np.array([-.5]));return SimpleNamespace(success=True,status=0,message='synthetic success')
    monkeypatch.setattr('uav_em.task_contrast.minimize',fake_solver)
    r=profile(req,record,plan)
    assert any(not t['feasible'] for t in r['trace'])
    assert all(e['feasible'] for e in r['endpoints'])
    assert r['world_delta_c_interval_seconds'][0]==r['world_delta_c_interval_seconds'][1]


@pytest.mark.parametrize('peak,world_crossed,mixture_crossed',[(-78.,True,False),(-77.5,True,True),(-77.,False,True)])
def test_reached_crossings_distinguish_world_and_frozen_mixture(peak,world_crossed,mixture_crossed):
    req,record,plan=fixture(peak);r=profile(req,record,plan)
    assert r['world_ranking_crossed']==world_crossed
    assert r['frozen_weight_ranking_crossed']==mixture_crossed
