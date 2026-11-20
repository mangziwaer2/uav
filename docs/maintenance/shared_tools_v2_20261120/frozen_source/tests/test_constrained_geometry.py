from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from uav_em import source_geometry as geometry
from uav_em.equivalent_field import kernel,power_and_jacobian
from uav_em.event_inference import in_region


def problem(region='left_half',seed=(210.,220.),target=(230.,180.),goal=(400.,200.)):
    public=dict(goal_m=list(goal),map=dict(bounds_m=[[0.,0.],[500.,500.]]))
    model=dict(id='synthetic',family='static',std_db=4.,born_index=11,scored_points=0,reference_time_seconds=11.,
        bases=[dict(center_m=list(seed),scale_m=80.,velocity_mps=[0.,0.],rate_db_per_second=0.)],peaks_dbm=[-72.],fit_mse=0.)
    truth=deepcopy(model);truth['bases'][0]['center_m']=list(target)
    xy=[[x,y] for y in (160.,200.,240.) for x in (160.,200.,240.,280.)];xy[-1]=[200.,200.]
    times=list(range(12));labels=power_and_jacobian(kernel(truth['bases'],xy,times,11.),truth['peaks_dbm'])[0]
    history=[dict(position_m=p,time_seconds=float(t),interference_dbm=float(v)) for p,t,v in zip(xy,times,labels)]
    return public,dict(job=dict(family='static',region=region,max_bases=1),model=model,window=history,
        bounds_dbm=[-140.,-60.],gp_soft_constraints=None,fit_objective=0.)


@pytest.mark.parametrize('region',['task_disk','left_half','right_half'])
def test_region_jacobian_matches_finite_difference_and_is_per_source(region):
    p=np.array([-72.,-75.,230.,220.,270.,250.,4.,4.2]);origin=np.array([200.,200.]);goal=np.array([400.,270.])
    values,jac=geometry.region_values_jacobian(p,2,origin,goal,region)
    assert values.shape==(2 if region=='task_disk' else 4,)
    for j in range(len(p)):
        d=np.zeros(len(p));d[j]=1e-5
        numerical=(geometry.region_values_jacobian(p+d,2,origin,goal,region)[0]-geometry.region_values_jacobian(p-d,2,origin,goal,region)[0])/2e-5
        assert np.allclose(jac[:,j],numerical,atol=1e-8)


@pytest.mark.parametrize('region,seed,target',[
    ('left_half',(210.,220.),(230.,180.)),
    ('right_half',(210.,180.),(230.,220.)),
    ('task_disk',(350.,210.),(380.,200.)),
    ('left_half',(350.,210.),(360.,200.)),
])
def test_explicit_constraints_produce_improved_legal_fit_at_boundaries(region,seed,target):
    public,fit=problem(region,seed,target);before=deepcopy(fit)
    result=geometry.constrained_refine_fit(fit,public);trace=result['geometry_refinement']
    assert fit==before
    assert trace['accepted'],trace
    assert trace['solver_success'] and trace['region_eligible'] and trace['bounds_eligible']
    assert trace['candidate_objective']<trace['previous_objective']-1e-8
    assert in_region(result['model']['bases'][0]['center_m'],public,[200.,200.],region)
    assert trace['resources']['residual_calls']<=40 and trace['resources']['jacobian_calls']<=40
    assert trace['resources']['kernel_terms']<=5000


def test_goal_at_origin_does_not_create_a_false_halfplane():
    public,fit=problem(goal=(200.,200.),target=(230.,180.))
    result=geometry.constrained_refine_fit(fit,public)
    assert result['geometry_refinement']['accepted']
    values,_=geometry.region_values_jacobian(geometry.pack(fit['model'],'geometry'),1,[200.,200.],[200.,200.],'left_half')
    assert len(values)==1


def test_objective_gradient_includes_parameter_scaling_and_soft_point_weights(monkeypatch):
    public,fit=problem()
    fit['job']['max_bases']=2;fit['model']['family']='translate'
    fit['model']['bases'][0].update(velocity_mps=[3.,0.],rate_db_per_second=.5)
    second=deepcopy(fit['model']['bases'][0]);second.update(center_m=[260.,230.],velocity_mps=[0.,-3.],rate_db_per_second=-.5)
    fit['model']['bases'].append(second);fit['model']['peaks_dbm'].append(-75.)
    fit['gp_soft_constraints']=dict(position_m=[[200.,230.]],time_seconds=[11.],mean_dbm=[-73.],
        weights=[1.],strength=.25,prefix_end_seconds=11.,independent_evidence_points=0)
    def solve(fun,x,jac,**kwargs):
        actual=jac(x)
        for j in range(len(x)):
            d=np.zeros(len(x));d[j]=1e-6
            numerical=(fun(x+d)-fun(x-d))/2e-6
            assert actual[j]==pytest.approx(numerical,abs=1e-5,rel=1e-5)
        return SimpleNamespace(x=x,success=True,status=0,message='unchanged',nfev=8,njev=1)
    monkeypatch.setattr(geometry,'minimize',solve)
    result=geometry.constrained_refine_fit(fit,public,acceptance_policy='converged_v1')
    assert not result['geometry_refinement']['accepted']
    assert result['model']==fit['model']


@pytest.mark.parametrize('resource',['objective','gradient','kernel'])
def test_hard_caps_preserve_seed_and_count_actual_work(monkeypatch,resource):
    public,fit=problem();settings=geometry.GeometrySettings(max_kernel_terms=1 if resource=='kernel' else 5000)
    def exhaust(fun,x,jac,**kwargs):
        for _ in range(41):(fun if resource=='objective' else jac)(x)
        raise AssertionError('hard resource cap failed')
    monkeypatch.setattr(geometry,'minimize',exhaust)
    result=geometry.constrained_refine_fit(fit,public,settings);trace=result['geometry_refinement']
    assert result['model']==fit['model'] and not trace['accepted']
    assert 'budget exhausted' in trace['error']
    key={'objective':'residual_calls','gradient':'jacobian_calls','kernel':'kernel_terms'}[resource]
    assert trace['resources'][key]==(0 if resource=='kernel' else 40)


def test_solver_success_cannot_override_original_region_check(monkeypatch):
    public,fit=problem()
    def illegal(fun,x,jac,**kwargs):
        x=x.copy();x[2]=-1.
        return SimpleNamespace(x=x,success=True,status=0,message='synthetic invalid result',nfev=0,njev=0)
    monkeypatch.setattr(geometry,'minimize',illegal)
    result=geometry.constrained_refine_fit(fit,public);trace=result['geometry_refinement']
    assert trace['solver_success'] and not trace['solver_final_region_eligible']
    assert trace['region_eligible']  # The retained seed, separately from the invalid solver return.
    assert result['model']==fit['model'] and not trace['accepted']


def test_cannot_use_future_gp_prefix_or_inherit_birth_evidence():
    public,fit=problem();fit['model']['scored_points']=1
    with pytest.raises(ValueError,match='precede birth'):geometry.constrained_refine_fit(fit,public)
    fit['model']['scored_points']=0
    fit['gp_soft_constraints']=dict(position_m=[[200.,200.]],time_seconds=[12.],mean_dbm=[-70.],
        weights=[1.],strength=.25,prefix_end_seconds=12.,independent_evidence_points=0)
    with pytest.raises(ValueError,match='original fit prefix'):geometry.constrained_refine_fit(fit,public)
