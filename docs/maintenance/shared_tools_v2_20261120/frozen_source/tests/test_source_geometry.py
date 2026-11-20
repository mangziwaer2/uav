from copy import deepcopy
import math

import numpy as np
import pytest

from uav_em.equivalent_field import kernel,power_and_jacobian
from uav_em.event_inference import HypothesisBank
from uav_em.event_llm_inference import run_event
from uav_em.source_geometry import GeometrySettings,pack,prediction_jacobian,refine_fit


def problem():
    public=dict(goal_m=[480.,280.],map=dict(bounds_m=[[0.,0.],[500.,500.]]))
    model=dict(id='seed',family='static',std_db=4.,born_index=11,scored_points=0,
        reference_time_seconds=11.,bases=[dict(center_m=[220.,240.],scale_m=80.,velocity_mps=[0.,0.],rate_db_per_second=0.)],
        peaks_dbm=[-72.],fit_mse=0.)
    truth=deepcopy(model);truth['bases'][0].update(center_m=[245.,225.],scale_m=70.)
    xy=[[x,y] for y in (180.,230.,280.) for x in (180.,220.,260.,300.)];times=list(range(12))
    labels=power_and_jacobian(kernel(truth['bases'],xy,times,11.),truth['peaks_dbm'])[0]
    history=[dict(position_m=point,time_seconds=float(t),interference_dbm=float(v)) for point,t,v in zip(xy,times,labels)]
    return public,dict(job=dict(family='static',region='task_disk',max_bases=1),model=model,
        window=history,bounds_dbm=[-140.,-60.],gp_soft_constraints=None,fit_objective=0.)


@pytest.mark.parametrize('mode',['amplitude','geometry'])
def test_analytic_derivative_matches_finite_difference_for_dynamic_multi_source(mode):
    _,fit=problem();m=fit['model'];m['bases'].append(deepcopy(m['bases'][0]));m['peaks_dbm'].append(-75.)
    m['bases'][0].update(velocity_mps=[3.,-2.],rate_db_per_second=.5)
    m['bases'][1].update(center_m=[300.,200.],scale_m=40.)
    p=pack(m,mode);xy=np.array([[170.,190.],[280.,260.],[310.,170.]])
    times=np.array([10.,11.,15.]);mean,jac=prediction_jacobian(m,p,xy,times,mode)
    for j in range(len(p)):
        delta=np.zeros(len(p));delta[j]=1e-5
        numerical=(prediction_jacobian(m,p+delta,xy,times,mode)[0]-prediction_jacobian(m,p-delta,xy,times,mode)[0])/2e-5
        assert np.allclose(jac[:,j],numerical,atol=1e-7,rtol=1e-5)


def test_off_grid_geometry_can_improve_fit_beyond_equal_budget_amplitude_refit():
    public,fit=problem();before=deepcopy(fit)
    amplitude=refine_fit(fit,public,'amplitude');geometry=refine_fit(fit,public,'geometry')
    assert fit==before
    a=amplitude['geometry_refinement'];g=geometry['geometry_refinement']
    assert g['accepted'] and g['candidate_objective']<a['candidate_objective']-1e-4
    assert geometry['model']['bases'][0]['center_m']==pytest.approx([245.,225.],abs=1e-3)
    assert geometry['model']['bases'][0]['scale_m']==pytest.approx(70.,abs=1e-3)
    assert geometry['model']['scored_points']==0
    assert a['resources']['solver_calls']==g['resources']['solver_calls']==1
    assert g['resources']['residual_calls']<=40 and g['resources']['jacobian_calls']<=40
    assert g['resources']['kernel_terms']<=5000


def test_budget_exhaustion_keeps_original_model_and_records_failure():
    public,fit=problem();result=refine_fit(fit,public,'geometry',GeometrySettings(max_kernel_terms=1))
    assert result['model']==fit['model']
    assert 'budget exhausted' in result['geometry_refinement']['error']
    assert result['geometry_refinement']['resources']['kernel_terms']==0


def test_boundary_and_nonconvergent_results_preserve_valid_seed():
    public,fit=problem();result=refine_fit(fit,public,'geometry',GeometrySettings(max_evaluations=1))
    assert result['model']==fit['model'] and not result['geometry_refinement']['accepted']
    public2=deepcopy(public);public2['map']['bounds_m'][1]=[200.,200.]
    result=refine_fit(fit,public2,'geometry')
    assert result['model']==fit['model'] and 'seed outside' in result['geometry_refinement']['error']


def test_geometry_cannot_use_soft_constraints_from_later_prefix_or_rewrite_evidence():
    public,fit=problem();fit['model']['scored_points']=1
    with pytest.raises(ValueError,match='precede birth'):refine_fit(fit,public,'geometry')
    fit['model']['scored_points']=0
    fit['gp_soft_constraints']=dict(position_m=[[220.,240.]],time_seconds=[12.],mean_dbm=[-70.],
        weights=[1.],strength=.25,prefix_end_seconds=12.,independent_evidence_points=0)
    with pytest.raises(ValueError,match='original fit prefix'):refine_fit(fit,public,'geometry')


def test_pilot_geometry_withholds_tail_and_event_uses_shared_explicit_caps():
    history=[dict(position_m=[100.+5*t,100.],time_seconds=float(t),interference_dbm=-85.+.2*t) for t in range(9)]
    bank=HypothesisBank(history[0]);rows=[]
    for p in history[1:]:bank.issue(p['position_m'],p['time_seconds']);rows.append(bank.observe(p))
    event=run_event(bank,dict(goal_m=[450.,100.],map=dict(bounds_m=[[0.,0.],[480.,360.]])),
        0,rows,'feedback',geometry_refinement='geometry')
    pilot=event['rounds'][0]['fits'][0]
    assert pilot['window'][-1]['time_seconds']==pilot['geometry_refinement']['prefix_end_seconds']==5.
    assert [p['time_seconds'] for p in event['rounds'][0]['pilot_feedback']['tail_points']]==[6.,7.,8.]
    assert all(m['born_index']==8 and m['scored_points']==0 for m in event['committed_models'])
    work=event['resources'];assert work['solver_calls']<=9 and work['residual_calls']<=360
    assert work['refinement']['calls']<=3 and work['refinement']['kernel_terms']<=15000


@pytest.mark.parametrize('kwargs',[dict(scale_ratio=1.),dict(max_evaluations=41),dict(center_radius_m=math.inf)])
def test_invalid_refinement_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):GeometrySettings(**kwargs)
