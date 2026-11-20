from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from uav_em import task_proposals as task
from uav_em.event_inference import Budget, digest


@pytest.fixture
def packet():
    old=json.loads((task.M4/'inputs/01.json').read_text())
    ep=json.loads((task.shared.M3/'runs/geometry_amplitude/episode.json').read_text())
    event=next(e for e in ep['inferences'] if e['time_seconds']==25.)
    return task.make_packet(old,event)


def test_prior_uses_only_original_before_birth_and_rejects_continuation(packet):
    ep=json.loads((task.shared.M3/'runs/geometry_amplitude/episode.json').read_text())
    event=next(e for e in ep['inferences'] if e['time_seconds']==25.)
    assert packet['prior_models']==event['bank_before_birth']
    changed=deepcopy(event);changed['future_truth']='forbidden'
    assert task.make_packet(json.loads((task.M4/'inputs/01.json').read_text()),changed)==packet
    changed['bank_before_birth'][0]['family']='persistence'
    with pytest.raises(ValueError,match='source-only'):
        task.make_packet(json.loads((task.M4/'inputs/01.json').read_text()),changed)


@pytest.mark.parametrize('change',[
    dict(request_id='old'),dict(geometry_job=0),dict(action='keep',jobs=[]),
    dict(extra='forbidden'),dict(jobs=[dict(family='static',region='whole_map',max_bases=2)]),
    dict(jobs=[dict(family='static',region='task_disk',max_bases=True)]),
    dict(jobs=[dict(family='static',region='task_disk',max_bases=2)]*2),
])
def test_pilot_cannot_change_stage_budget_region_or_binding(packet,change):
    req=task.request_for(packet,'pilot');raw=task.numerical_choice(req);raw.update(change)
    with pytest.raises((ValueError,TypeError)):task.validate(req,raw)


def test_refine_binding_includes_feedback_and_geometry_is_one_integer(packet):
    req=task.request_for(packet,'refine',dict(job={'family':'static'},tail_rmse_db=5.))
    raw=task.numerical_choice(req)
    assert task.validate(req,json.dumps(raw))==raw
    for index in (True,-1,2,[0,1]):
        bad=dict(raw,geometry_job=index)
        with pytest.raises(ValueError):task.validate(req,bad)
    req['feedback']['tail_rmse_db']=1.
    with pytest.raises(ValueError,match='changed'):task.validate(req,raw)


def test_numeric_policy_uses_task_disagreement_and_known_tail_not_future(packet):
    for w in packet['world_actions']:w['allocation_priority']=0.
    families={m['id']:m['family'] for m in packet['models']}
    world=next(w for w in packet['world_actions'] if families[w['id']]=='translate')
    world['allocation_priority']=10.
    pilot=task.numerical_choice(task.request_for(packet,'pilot'))
    assert pilot['jobs'][0]['family']=='translate'
    for rmse,repeat in ((5.,True),(4.,False)):
        req=task.request_for(packet,'refine',dict(job=pilot['jobs'][0],tail_rmse_db=rmse))
        plan=task.numerical_choice(req)
        assert ('translate' in [j['family'] for j in plan['jobs']])==repeat
        assert plan['geometry_job']==0


def test_execution_keeps_pilot_tail_out_and_warm_start_only_full_prior(packet,monkeypatch):
    calls=[];geometry=[]
    def fit(bank,public,job,budget,identifier,soft):
        calls.append((deepcopy(bank),deepcopy(soft)))
        return dict(job=job,model={'bases':[{}]},marker=identifier)
    def refine(fit,public,mode):geometry.append(fit['marker']);return fit
    monkeypatch.setattr(task,'fit_job',fit);monkeypatch.setattr(task,'refine_fit',refine)
    budget=Budget();before=deepcopy(packet)
    pilot=task.numerical_choice(task.request_for(packet,'pilot'))
    task.execute_jobs(packet,'pilot',pilot,budget,1)
    plan=task.numerical_choice(task.request_for(packet,'refine',dict(job=pilot['jobs'][0],tail_rmse_db=5.)))
    task.execute_jobs(packet,'refine',plan,budget,1)
    assert len(calls)==3 and len(geometry)==1 and packet==before
    assert calls[0][0].history==packet['history'][:-3] and calls[0][0].models==[]
    assert calls[0][1]['prefix_end_seconds']==packet['history'][-4]['time_seconds']
    for bank,soft in calls[1:]:
        assert bank.history==packet['history'] and bank.models==packet['prior_models']
        assert soft['prefix_end_seconds']==25. and soft['independent_evidence_points']==0
    assert budget.reserved_jobs==3 and budget.reserved_calls==6
    with pytest.raises(ValueError,match='budget'):task.execute_jobs(packet,'pilot',pilot,budget,1)


def test_three_world_commit_replaces_birth_once_and_resets_evidence(packet):
    fits=deepcopy(packet['fits']);before=deepcopy(packet)
    models,committed=task.commit_models(packet,fits)
    assert committed and packet==before
    n=len(packet['prior_models'])
    assert len(models)==n+3 and sum(m['weight'] for m in models)==pytest.approx(1.)
    for old,new in zip(packet['prior_models'],models[:n]):
        assert new['weight']==pytest.approx(old['weight']*.75)
        assert {k:v for k,v in new.items() if k not in ('weight','log_weight')}=={k:v for k,v in old.items() if k not in ('weight','log_weight')}
    for model in models[n:]:
        assert model['weight']==pytest.approx(.25/3)
        assert model['reference_time_seconds']==25. and model['scored_points']==0
        assert model['born_index']==len(packet['history'])-1
    for failed in (fits[:2],[fits[0],fits[1],{'error':'solver failed'}]):
        assert task.commit_models(packet,failed)==(packet['models'],False)
    assert task.commit_models(packet,fits,keep=True)==(packet['models'],False)
