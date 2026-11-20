"""Synthetic candidate-retention properties; no simulator or historical states."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from uav_em import source_geometry as geometry
from uav_em.equivalent_field import kernel,power_and_jacobian


def problem(two_sources=False):
    public=dict(goal_m=[480.,280.],map=dict(bounds_m=[[0.,0.],[500.,500.]]))
    def base(center,scale):
        return dict(center_m=list(center),scale_m=scale,velocity_mps=[0.,0.],rate_db_per_second=0.)
    seed=dict(id='synthetic',family='static',std_db=4.,born_index=11,scored_points=0,
        reference_time_seconds=11.,bases=[base((200.,240.),80.)],peaks_dbm=[-72.],fit_mse=0.)
    if two_sources:
        seed['bases'].append(base((280.,210.),70.));seed['peaks_dbm'].append(-76.)
    target=deepcopy(seed)
    target['bases'][0].update(center_m=[239.,220.],scale_m=57.)
    if two_sources:target['bases'][1].update(center_m=[250.,245.],scale_m=98.)
    xy=[[x,y] for y in (180.,230.,280.) for x in (180.,220.,260.,300.)];times=list(range(12))
    labels=power_and_jacobian(kernel(target['bases'],xy,times,11.),target['peaks_dbm'])[0]
    history=[dict(position_m=p,time_seconds=float(t),interference_dbm=float(y)) for p,t,y in zip(xy,times,labels)]
    fit=dict(job=dict(family='static',region='task_disk',max_bases=len(seed['bases'])),model=seed,
        window=history,bounds_dbm=[-140.,-60.],gp_soft_constraints=None,fit_objective=0.)
    return public,fit,target


def test_actual_default_cap_keeps_best_legal_candidate_without_more_evaluations():
    public,fit,_=problem(two_sources=True);before=deepcopy(fit)
    result=geometry.constrained_refine_fit(fit,public);trace=result['geometry_refinement']
    legacy=geometry.constrained_refine_fit(fit,public,acceptance_policy='converged_v1')
    assert fit==before
    assert trace['accepted'] and not trace['solver_success']
    assert 'budget exhausted' in trace['stop_reason']
    assert trace['candidate_objective']<trace['previous_objective']/100
    legal=[r for r in trace['objective_trace'] if r['eligible']]
    assert trace['candidate_objective']==min(r['objective'] for r in legal)
    assert trace['resources']==legacy['geometry_refinement']['resources']
    assert trace['resources']['residual_calls']==40
    assert trace['resources']['jacobian_calls']<=40 and trace['resources']['kernel_terms']<=5000
    assert legacy['model']==fit['model'] and not legacy['geometry_refinement']['accepted']
    assert result['model']['scored_points']==0 and result['model']['born_index']==11


@pytest.mark.parametrize('method',['constrained','box'])
def test_legal_evaluated_improvement_survives_invalid_solver_final_without_extra_prediction(monkeypatch,method):
    public,fit,target=problem();initial=geometry.pack(fit['model'],'geometry');desired=geometry.pack(target,'geometry')
    units=np.array([4.,40.,40.,np.log(np.sqrt(2.))])
    seen=[];original=geometry.prediction_jacobian
    def predict(*args):
        seen.append(np.asarray(args[1]).copy())
        return original(*args)
    monkeypatch.setattr(geometry,'prediction_jacobian',predict)
    def solve(fun,x,jac,**kwargs):
        good=(desired-initial)/units if method=='constrained' else desired
        fun(good)
        illegal=np.full_like(x,np.nan)
        return SimpleNamespace(x=illegal,success=False,status=9,message='synthetic stopped',nfev=1,njev=0,optimality=float('nan'))
    monkeypatch.setattr(geometry,'minimize' if method=='constrained' else 'least_squares',solve)
    result=(geometry.constrained_refine_fit(fit,public) if method=='constrained'
        else geometry.refine_fit(fit,public,'geometry'))
    trace=result['geometry_refinement']
    assert trace['accepted'] and not trace['solver_success']
    assert trace['region_eligible'] and trace['bounds_eligible']
    assert result['model']['bases'][0]['center_m']==target['bases'][0]['center_m']
    assert trace['candidate_objective']<1e-12
    assert len(seen)==2  # seed plus one objective; never evaluate the unobserved final.
    assert trace['resources']['residual_calls']==1
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('method',['constrained','box'])
def test_solver_return_alone_cannot_create_an_unevaluated_candidate(monkeypatch,method):
    public,fit,target=problem();initial=geometry.pack(fit['model'],'geometry');desired=geometry.pack(target,'geometry')
    units=np.array([4.,40.,40.,np.log(np.sqrt(2.))])
    def solve(fun,x,jac,**kwargs):
        good=(desired-initial)/units if method=='constrained' else desired
        return SimpleNamespace(x=good,success=True,status=0,message='no objective called',nfev=0,njev=0,optimality=0.)
    monkeypatch.setattr(geometry,'minimize' if method=='constrained' else 'least_squares',solve)
    result=(geometry.constrained_refine_fit(fit,public) if method=='constrained'
        else geometry.refine_fit(fit,public,'geometry'))
    assert result['model']==fit['model'] and not result['geometry_refinement']['accepted']
    assert result['geometry_refinement']['resources']['residual_calls']==0


@pytest.mark.parametrize('method',['constrained','box'])
def test_illegal_lower_loss_never_displaces_legal_seed(monkeypatch,method):
    public,fit,target=problem();fit['job']['region']='left_half'
    fit['model']['bases'][0]['center_m']=[280.,300.]
    # The target is on the wrong side of the line from the final observation to goal.
    target['bases'][0]['center_m']=[300.,270.]
    xy=[p['position_m'] for p in fit['window']];times=[p['time_seconds'] for p in fit['window']]
    labels=power_and_jacobian(kernel(target['bases'],xy,times,11.),target['peaks_dbm'])[0]
    for row,y in zip(fit['window'],labels):row['interference_dbm']=float(y)
    initial=geometry.pack(fit['model'],'geometry');desired=geometry.pack(target,'geometry')
    units=np.array([4.,40.,40.,np.log(np.sqrt(2.))])
    def solve(fun,x,jac,**kwargs):
        point=(desired-initial)/units if method=='constrained' else desired
        fun(point)
        return SimpleNamespace(x=point,success=True,status=0,message='illegal point',nfev=1,njev=0,optimality=0.)
    monkeypatch.setattr(geometry,'minimize' if method=='constrained' else 'least_squares',solve)
    result=(geometry.constrained_refine_fit(fit,public) if method=='constrained'
        else geometry.refine_fit(fit,public,'geometry'))
    trace=result['geometry_refinement']
    assert not trace['accepted'] and result['model']==fit['model']
    assert trace['objective_trace'] and not trace['objective_trace'][0]['eligible']


def test_unknown_policy_is_rejected():
    public,fit,_=problem()
    with pytest.raises(ValueError,match='acceptance policy'):
        geometry.constrained_refine_fit(fit,public,acceptance_policy='trust_solver')


def test_nonfinite_gradient_stops_and_keeps_evaluated_legal_improvement(monkeypatch):
    public,fit,target=problem();initial=geometry.pack(fit['model'],'geometry');desired=geometry.pack(target,'geometry')
    units=np.array([4.,40.,40.,np.log(np.sqrt(2.))]);original=geometry.prediction_jacobian
    def predict(model,parameters,*args):
        mean,jac=original(model,parameters,*args)
        if np.allclose(parameters,desired,rtol=0.,atol=1e-12):jac[:]=np.nan
        return mean,jac
    monkeypatch.setattr(geometry,'prediction_jacobian',predict)
    def solve(fun,x,jac,**kwargs):
        point=(desired-initial)/units
        fun(point);jac(point)
        raise AssertionError('nonfinite gradient was passed back to solver')
    monkeypatch.setattr(geometry,'minimize',solve)
    result=geometry.constrained_refine_fit(fit,public);trace=result['geometry_refinement']
    assert trace['accepted'] and trace['stop_reason']=='nonfinite refinement gradient'
    assert trace['gradient_trace']==[None]
    assert trace['resources']['residual_calls']==trace['resources']['jacobian_calls']==1
    json.dumps(result,allow_nan=False)
