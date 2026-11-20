from copy import deepcopy
from dataclasses import replace
import json
import math
import numpy as np
import pytest
from uav_em.config import load_config
from uav_em.environment import Geometry
from uav_em.repair_search import WorkBudget
from uav_em.scenario_planning import (ScenarioForecast, arrival_queries, score_route,
    state_request, validate_search_plan, propose_and_accept, DispatchReceipt)


def setup():
    cfg=load_config('configs/local_power.json')
    g=Geometry(replace(cfg.map,buildings=(),obstacles_m=()),cfg.aircraft.clearance_m,cfg.aircraft.altitude_m,cfg.aircraft.vertical_clearance_m)
    history=[dict(position_m=[40.,40.],time_seconds=0.,interference_dbm=-80.)]
    model=dict(id='p',family='persistence',std_db=0.,born_index=0,weight=1.)
    f=ScenarioForecast(history,[model],0.)
    state=dict(position_m=[40.,40.],goal_m=[100.,40.],route_m=[[40.,40.],[100.,40.]],
        speed_mps=10.,spent_length_m=0.,maximum_length_m=100.,remaining_energy_wh=100.,
        flight_power_w=100.,hover_power_w=50.,reserve_energy_wh=1.,delay_seconds=.5)
    return g,f,state


def response(request, action='direct'):
    return dict(request_id=request['request_id'],action=action,waypoint_m=None,
        sample_quotas=[20] if action=='direct' else [],sampling='uniform',reason='synthetic only')


def test_constant_power_matches_units_and_full_flight_duration():
    _,f,s=setup();r=score_route(f,s['route_m'],10.)
    assert r['remaining_flight_seconds']==6.
    assert r['power_proxy_seconds']==pytest.approx(6.)
    assert r['score_seconds']==pytest.approx(12.)
    assert min(r['queries']['time_seconds'])>.5


def test_mixture_is_linear_power_not_dbm_or_optimistic_branch_selection():
    _,f,s=setup()
    low=dict(id='lo',family='static',std_db=0.,born_index=0,weight=.5,reference_time_seconds=0.,
             bases=[],peaks_dbm=[])
    high=deepcopy(low);high.update(id='hi',bases=[dict(center_m=[70.,40.],scale_m=1e6,velocity_mps=[0.,0.],rate_db_per_second=0.)],peaks_dbm=[-60.])
    f=ScenarioForecast(f.history,[low,high],0.)
    r=score_route(f,s['route_m'],10.)
    qs=[x['power_proxy_seconds'] for x in r['scenario_scores']]
    assert r['power_proxy_seconds']==pytest.approx(sum(qs)/2)
    assert r['power_proxy_seconds']>49.*6
    assert r['scenario_q_std_seconds']==pytest.approx(abs(qs[0]-qs[1])/2)
    assert f.models[0]['weight']==.5


def test_dynamic_field_uses_delay_and_complete_prefix_arrival():
    _,f,_=setup();m=dict(id='grow',family='strength_change',std_db=0.,born_index=0,weight=1.,reference_time_seconds=0.,
        bases=[dict(center_m=[70.,40.],scale_m=1e6,velocity_mps=[0.,0.],rate_db_per_second=1.)],peaks_dbm=[-80.])
    f=ScenarioForecast(f.history,[m],0.)
    r=score_route(f,[[40.,40.],[70.,40.],[100.,40.]],10.,delay=2.)
    q=r['queries'];expected=sum(dt*(.01+10**(t/10)) for t,dt in zip(q['time_seconds'],q['dt_seconds']))
    assert r['power_proxy_seconds']==pytest.approx(expected,rel=1e-8)
    assert score_route(f,[[40.,40.],[100.,40.]],10.,frozen_time=True)['score_seconds']<r['score_seconds']


def test_collinear_repartition_and_duplicates_do_not_change_integral():
    _,f,_=setup()
    a=score_route(f,[[40,40],[100,40]],10.)
    b=score_route(f,[[40,40],[40,40],[43,40],[78,40],[100,40]],10.)
    assert a==b
    q=arrival_queries([[40,40],[70,40],[40,40]],0.,10.)
    assert q['duration_seconds']==6.  # A reversal must not be simplified away.


def test_model_input_copy_and_causal_query_rejection():
    _,f,_=setup();models=deepcopy(f.models);history=deepcopy(f.history)
    ff=ScenarioForecast(history,models,0.);models[0]['weight']=0.;history[-1]['interference_dbm']=-30.
    assert ff.predict([[50,40]],[1.])['means'][0,0]==-80.
    with pytest.raises(ValueError):ff.predict([[50,40]],[-1.])


def test_unsupported_gp_values_use_declared_prior_and_missing_provider_rejected():
    _,f,s=setup()
    with pytest.raises(ValueError):score_route(f,s['route_m'],10.,method='gp')
    class GP:
        def predict(self,xy,times):return np.full(len(times),np.nan),np.full(len(times),np.nan),np.zeros(len(times),bool)
    f=ScenarioForecast(f.history,f.models,0.,GP(),gp_version='synthetic-no-support');r=score_route(f,s['route_m'],10.,method='gp')
    expected=6*math.exp(math.log(10)/10*(-20)+.5*(math.log(10)/10)**2*225)
    assert r['power_proxy_seconds']==pytest.approx(expected)
    assert r['supported_fraction']==0.


@pytest.mark.parametrize('bad',[True,0,-1,float('nan')])
def test_invalid_speed_rejected(bad):
    with pytest.raises(ValueError):arrival_queries([[0,0],[10,0]],0.,bad)


@pytest.mark.parametrize('change',[
    dict(sample_quotas=[True]),dict(sample_quotas=[701]),dict(waypoint_m=[60,50]),
    dict(action='probe'),dict(request_id='stale'),dict(unexpected=1),dict(reason='')])
def test_strict_plan_rejects_invalid_or_unbudgeted_fields(change):
    g,f,s=setup();r=state_request(s,f,g,WorkBudget(700,10000));p=response(r);p.update(change)
    with pytest.raises(ValueError):validate_search_plan(r,p)


def test_duplicate_json_keys_rejected():
    g,f,s=setup();r=state_request(s,f,g,WorkBudget(700,10000))
    raw=json.dumps(response(r));raw=raw[:-1]+',"action":"keep"}'
    with pytest.raises(ValueError):validate_search_plan(r,raw)


def test_live_state_changes_rejected_before_search_or_geometry_spend():
    g,f,s=setup();w=WorkBudget(700,10000);r=state_request(s,f,g,w);new=deepcopy(s)
    new['position_m']=[41.,40.]
    with pytest.raises(ValueError,match='state changed'):propose_and_accept(r,response(r),new,f,g,w)
    assert w.samples==w.edges==0


def test_flew_prefix_and_hover_energy_are_hard_constraints():
    g,f,s=setup();s['spent_length_m']=50.;w=WorkBudget(700,10000);r=state_request(s,f,g,w)
    with pytest.raises(ValueError,match='incumbent'):propose_and_accept(r,response(r),s,f,g,w)
    assert w.samples==0
    s['spent_length_m']=0.;s['remaining_energy_wh']=s['reserve_energy_wh']+6*100/3600
    r=state_request(s,f,g,w)
    with pytest.raises(ValueError,match='incumbent'):propose_and_accept(r,response(r),s,f,g,w)


def test_shared_search_budget_and_common_acceptor_keep_a_better_incumbent():
    g,f,s=setup();w=WorkBudget(24,1200);r=state_request(s,f,g,w);p=response(r);p['sample_quotas']=[24]
    result=propose_and_accept(r,p,s,f,g,w,seed=31)
    assert result['path']==s['route_m'] and not result['accepted']
    assert w.samples<=24 and w.edges<=1200 and w.reserved_edges==0
    assert w.verification_edges>=1
    assert len(result['searches'])==1


def test_dispatch_files_alone_never_count_calls_and_late_dispatch_cannot_retry():
    d=DispatchReceipt('r',100.);assert not d.snapshot()['actual_model_call']
    d.expire(161.);assert d.status=='undispatched_timeout'
    with pytest.raises(ValueError):d.dispatch('/isolated/test',162.)


def test_dispatch_reply_deadline_starts_at_dispatch_not_request_issue():
    g,f,s=setup();r=state_request(s,f,g,WorkBudget(700,10000));d=DispatchReceipt(r['request_id'],100.)
    d.dispatch('/synthetic/only',150.);d.receive(json.dumps(response(r,'keep')),329.)
    assert d.execute(r,r)['action']=='keep' and d.snapshot()['actual_model_call']
    with pytest.raises(ValueError):d.execute(r,r)


def test_late_and_stale_responses_never_execute():
    g,f,s=setup();r=state_request(s,f,g,WorkBudget(700,10000));d=DispatchReceipt(r['request_id'],100.)
    d.dispatch('/synthetic/only',110.);d.receive(json.dumps(response(r)),291.)
    with pytest.raises(ValueError):d.execute(r,r)
    assert d.status=='late_raw'
    d=DispatchReceipt(r['request_id'],100.);d.dispatch('/synthetic/only',110.)
    d.receive(json.dumps(response(r)),120.)
    rr=deepcopy(r);rr['request_id']='new'
    with pytest.raises(ValueError):d.execute(r,rr)
    assert d.status=='stale_rejected' and not d.executed


def test_freeze_mutation_and_private_state_fields_are_rejected():
    g,f,s=setup();f.models[0]['std_db']=2.
    with pytest.raises(ValueError,match='mutated'):f.predict([[50.,40.]],[1.])
    _,f,s=setup();s['true_sources']=[]
    with pytest.raises(ValueError,match='public state fields'):state_request(s,f,g,WorkBudget(700,10000))


def test_detour_can_win_on_full_cost_without_choosing_a_favorable_scenario(monkeypatch):
    from uav_em.planning import PlanResult
    g,f,s=setup()
    m=dict(id='hot',family='static',std_db=0.,born_index=0,weight=1.,reference_time_seconds=0.,
        bases=[dict(center_m=[70.,40.],scale_m=8.,velocity_mps=[0.,0.],rate_db_per_second=0.)],peaks_dbm=[-50.])
    f=ScenarioForecast(f.history,[m],0.)
    def direct(problem,a,b,quota,seed,sampling):
        for _ in range(quota):problem.budget.take_sample()
        return PlanResult('rrt_star',path=[list(a),list(b)],iterations=quota),{'goal':quota,'focus':0,'global':0}
    monkeypatch.setattr('uav_em.scenario_planning.repair_rrt_star',direct)
    w=WorkBudget(24,1200);r=state_request(s,f,g,w);p=response(r,'via')
    p.update(waypoint_m=[70.,65.],sample_quotas=[12,12])
    ans=propose_and_accept(r,p,s,f,g,w)
    assert ans['accepted'] and len(ans['path'])==3
    assert ans['candidate']['score']['remaining_flight_seconds']>ans['before']['remaining_flight_seconds']
    assert ans['candidate']['score']['score_seconds']<ans['before']['score_seconds']
    assert w.samples==24 and w.verification_edges==3
    assert f.models[0]['weight']==1.


def test_sampling_cap_and_no_preexisting_reserved_ledger_accepted():
    g,f,s=setup()
    with pytest.raises(ValueError,match='bounded shared ledger'):state_request(s,f,g,WorkBudget(701,10000))
    w=WorkBudget(700,10000);w.reserved_edges=1
    with pytest.raises(ValueError,match='bounded shared ledger'):state_request(s,f,g,w)


def test_response_timeout_has_actual_call_but_no_execution():
    d=DispatchReceipt('r',100.);d.dispatch('/synthetic/only',110.);d.expire(291.)
    assert d.snapshot()['actual_model_call'] and not d.executed and d.status=='response_timeout'
    with pytest.raises(ValueError):d.receive('{}',292.)

