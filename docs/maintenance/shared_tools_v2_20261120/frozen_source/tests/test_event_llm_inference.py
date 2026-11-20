from copy import deepcopy
import math
import numpy as np
import pytest
from uav_em.event_inference import Budget, HypothesisBank
from uav_em.equivalent_field import kernel, power_and_jacobian
from uav_em.event_llm_inference import (development_gate, family_evidence, feedback_response,
    fixed_response, make_request, rebase_model, run_event, run_stream, score_result, validate_round)


def obs(t, y=None):
    return dict(position_m=[100.+5*t,100.],time_seconds=float(t),interference_dbm=-80.+.2*t if y is None else y)


def public():return dict(goal_m=[450.,100.],map=dict(bounds_m=[[0.,0.],[480.,360.]]))


def bank_at(t=8):
    bank=HypothesisBank(obs(0));rows=[]
    for k in range(1,t+1):
        p=obs(k);bank.issue(p['position_m'],p['time_seconds']);rows.append(bank.observe(p))
    return bank,rows


def test_reference_change_preserves_entire_dynamic_field():
    m=dict(id='x',family='translate',reference_time_seconds=2.,born_index=2,scored_points=0,std_db=4.,
        bases=[dict(center_m=[110.,100.],scale_m=80.,velocity_mps=[3.,0.],rate_db_per_second=.5)],peaks_dbm=[-75.])
    n=rebase_model(m,8.,8)
    xy=[[100.,50.],[180.,150.],[250.,100.]];times=[1.,8.,15.]
    a=power_and_jacobian(kernel(m['bases'],xy,times,2.),m['peaks_dbm'])[0]
    b=power_and_jacobian(kernel(n['bases'],xy,times,8.),n['peaks_dbm'])[0]
    assert np.allclose(a,b,atol=1e-12)
    assert m['bases'][0]['center_m']==[110.,100.] and n['born_index']==8


def test_pilot_excludes_tail_and_all_births_wait_for_future_evidence():
    bank,rows=bank_at()
    event=run_event(bank,public(),0,rows,'fixed')
    first=event['rounds'][0]
    assert max(p['time_seconds'] for p in first['fits'][0]['window'])==5.
    feedback=first['pilot_feedback']
    assert [p['time_seconds'] for p in feedback['tail_points']]==[6.,7.,8.]
    assert feedback['retrospective_only'] and not feedback['independent_future_evidence']
    assert all(m['born_index']==8 and m['scored_points']==0 and m['reference_time_seconds']==8. for m in event['committed_models'])
    assert len(bank.models)==5 and sum(math.exp(w) for w in bank.log_weights[-3:])==pytest.approx(.25)


@pytest.mark.parametrize('phase,count',[('pilot',1),('refine',2)])
def test_round_specific_job_count_is_enforced(phase,count):
    bank,rows=bank_at()
    request=make_request(bank,public(),0,phase,rows,None,Budget())
    reply=fixed_response(request)
    assert len(validate_round(bank,public(),request,reply)['jobs'])==count
    reply['jobs']*=3
    with pytest.raises(ValueError):validate_round(bank,public(),request,reply)


def test_rejected_or_timed_out_response_has_fixed_fallback_and_no_extra_round():
    bank,rows=bank_at();seen=[]
    def advisor(request):
        seen.append(request['phase'])
        return dict(raw='not json',timely=False,actual_llm_request=False,origin='test_only')
    event=run_event(bank,public(),0,rows,'llm',advisor)
    assert seen==['pilot','refine']
    assert all(not r['parser_accepted'] and r['fallback_error']=='response timeout' for r in event['rounds'])
    assert event['resources']['reserved_jobs']==3
    assert event['resources']['solver_calls']<=6 and event['resources']['residual_calls']<=240


def test_mutated_live_state_aborts_before_any_fit():
    bank,rows=bank_at();initial_models=deepcopy(bank.models)
    def advisor(request):
        p=obs(9);bank.issue(p['position_m'],9.);bank.observe(p)
        return dict(raw=fixed_response(request),timely=True,actual_llm_request=False)
    with pytest.raises(ValueError,match='advanced while waiting'):
        run_event(bank,public(),0,rows,'llm',advisor)
    assert len(bank.models)==len(initial_models)


def test_numeric_feedback_reuses_pilot_family_only_when_diagnostic_is_large():
    bank,rows=bank_at()
    request=make_request(bank,public(),0,'refine',rows,dict(job=dict(family='translate'),tail_rmse_db=5.),Budget())
    reply=feedback_response(request)
    assert [j['family'] for j in reply['jobs']]==['translate','static']
    request['feedback']['tail_rmse_db']=1.
    assert [j['family'] for j in feedback_response(request)['jobs']]==['static','strength_change']


def test_no_family_evidence_is_invented_before_birth():
    bank,rows=bank_at()
    evidence=family_evidence(bank,rows)
    assert all(v==dict(points=0,mean_nll=None) for v in evidence.values())


def test_stub_advisor_is_identical_to_fixed_without_becoming_real_model_evidence():
    stream=dict(public=public(),measurements=[obs(t) for t in range(25)])
    calls=[]
    def advisor(request):
        calls.append(request['request_id'])
        return dict(raw=fixed_response(request),timely=True,actual_llm_request=False,origin='test_only')
    fixed=run_stream(stream,'fixed');stub=run_stream(stream,'llm',advisor)
    assert len(calls)==4
    assert score_result(fixed)==score_result(stub)
    assert [e['time_seconds'] for e in stub['events']]==[8.,20.]
    assert all(not r['envelope']['actual_llm_request'] for e in stub['events'] for r in e['rounds'])


def test_future_measurement_changes_do_not_change_current_request_or_fit():
    bank,rows=bank_at();a=make_request(bank,public(),0,'pilot',rows,None,Budget())
    bank2,rows2=bank_at();b=make_request(bank2,public(),0,'pilot',rows2,None,Budget())
    assert a==b
    p=obs(9,-50.);bank2.issue(p['position_m'],9.);bank2.observe(p)
    assert a['history'][-1]['time_seconds']==8.
    assert a['request_id']==b['request_id']


def test_development_gate_requires_gain_and_persistence_control():
    def scores(nll,rmse):return dict(primary=dict(mean_nll=nll,rmse_db=rmse))
    row=dict(llm=scores(2.4,2.),feedback=scores(2.45,2.1),persistence=scores(2.4,2.))
    assert development_gate([row,row])['passed']
    row['llm']=scores(2.45,2.)
    assert not development_gate([row,row])['conditions']['mean_nll_gain']
    row['llm']=scores(2.3,2.);row['persistence']=scores(2.1,1.9)
    assert not development_gate([row,row])['conditions']['persistence_control']


