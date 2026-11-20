from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from uav_em.config import load_config
from uav_em.environment import Geometry
from uav_em.equivalent_field import normalized_expected_power
from uav_em.event_inference import Budget, HypothesisBank, digest, fit_job
from uav_em.event_llm_inference import run_event
from uav_em.joint_prediction import JointSettings, ForwardWarningTrigger, soft_constraints
from uav_em.point_gp import FrozenGP, training_points
from uav_em.scenario_planning import ScenarioForecast, score_route


def setup():
    c=load_config('configs/local_power.json')
    c=replace(c,map=replace(c.map,buildings=(),obstacles_m=()))
    g=Geometry(c.map,0.,c.aircraft.altitude_m)
    history=[dict(position_m=[40.+10*i,100.],time_seconds=float(i),interference_dbm=-85.+i) for i in range(8)]
    bank=HypothesisBank(history[0])
    for p in history[1:]:bank.issue(p['position_m'],p['time_seconds']);bank.observe(p)
    gp=FrozenGP(training_points(history,g,7.,c.belief),g,7.,c.belief,-100.)
    forecast=ScenarioForecast(history,bank.snapshot(),7.,gp,gp_version=digest(history),joint_settings=JointSettings())
    return c,g,bank,forecast


def test_linear_expectation_is_convex_and_scoring_does_not_transform_variance_twice():
    c,g,bank,f=setup()
    xy=[[110.,100.],[700.,500.]];tt=[7.,7.]
    source=f.predict(xy,tt,'mixture');joint=f.predict(xy,tt,'joint');w=joint['gp_weight'][None,:]
    expected=w*normalized_expected_power(joint['gp_mean_dbm'][None,:],joint['gp_std_db'][None,:],-80.)+(1-w)*normalized_expected_power(source['means'],source['stds'],-80.)
    np.testing.assert_allclose(joint['expected_normalized_power'],expected,rtol=1e-14)
    assert w[0,0]>0 and w[0,1]==0
    np.testing.assert_array_equal(joint['expected_normalized_power'][:,1],normalized_expected_power(source['means'],source['stds'],-80.)[:,1])
    result=score_route(f,[[110.,100.],[170.,100.]],15.,method='joint')
    direct=f.predict(result['queries']['position_m'],result['query_time_seconds'],'joint')
    exact=float(direct['weights']@(direct['expected_normalized_power']@np.array(result['queries']['dt_seconds'])))
    assert result['power_proxy_seconds']==pytest.approx(exact)


def test_future_support_expires_and_interpolation_is_version_bound():
    c,g,bank,f=setup()
    d=f.predict([[110.,100.]]*3,[7.,12.,50.],'joint')
    assert d['gp_weight'][0]>d['gp_weight'][1]>d['gp_weight'][2]==0
    f.joint_settings=replace(f.joint_settings,distance_scale_m=90.)
    with pytest.raises(ValueError,match='mutated'):f.predict([[110.,100.]],[7.],'joint')


def test_duplicate_soft_points_do_not_increase_total_constraint_strength():
    c,g,bank,f=setup();settings=JointSettings()
    soft=soft_constraints(bank.history,g,c.belief,-100.,settings)
    duplicate=deepcopy(soft)
    for key in ('position_m','time_seconds','mean_dbm'):duplicate[key]*=2
    duplicate['weights']=[w/2 for w in soft['weights']]*2
    public=dict(goal_m=[760.,100.],map=dict(bounds_m=[[0.,0.],[800.,600.]]))
    job=dict(family='static',region='task_disk',max_bases=2)
    fits=[]
    for item in (soft,duplicate):
        budget=Budget();budget.reserve([job]);fits.append(fit_job(bank,public,job,budget,'new',soft=item))
    assert fits[0]['model']['bases']==fits[1]['model']['bases']
    np.testing.assert_allclose(fits[0]['model']['peaks_dbm'],fits[1]['model']['peaks_dbm'],atol=1e-8)
    assert fits[0]['fit_objective']==pytest.approx(fits[1]['fit_objective'],abs=1e-10)
    assert soft['independent_evidence_points']==0


def test_pilot_gp_cannot_see_the_held_out_tail():
    c,g,bank,f=setup()
    public=dict(goal_m=[760.,100.],map=dict(bounds_m=[[0.,0.],[800.,600.]]))
    event=run_event(bank,public,0,[],'feedback',soft_context=dict(geometry=g,belief=c.belief,background=-100.,settings=JointSettings()))
    for rnd in event['rounds']:
        soft=rnd['request']['gp_soft_constraints']
        expected=4. if rnd['request']['phase']=='pilot' else 7.
        assert soft['prefix_end_seconds']==expected
        assert all(t<=expected for t in soft['time_seconds'])
        assert soft['independent_evidence_points']==0
    assert all(m['scored_points']==0 for m in event['committed_models'])


def test_forward_warning_does_not_need_current_crossing_or_profitable_detour():
    class Forecast:
        now=10.;fingerprint='frozen'
        def predict(self,xy,times,method):
            return dict(weights=np.ones(1),means=np.full((1,len(times)),-77.),
                        supported=np.ones(len(times),bool),gp_weight=np.full(len(times),.6))
    trigger=ForwardWarningTrigger()
    row=trigger.observe(Forecast(),[[40.,100.],[400.,100.]],15.,4)
    assert row['requested'] and row['predicted_lead_seconds']>0
    assert not row['geometry_gain_precondition']
    Forecast.now=11.
    assert trigger.observe(Forecast(),[[55.,100.],[400.,100.]],15.,5)['blocker']=='risk_already_notified'


def test_loop_freezes_joint_prediction_before_the_next_measurement():
    from uav_em.scenario_loop import run
    c,g,bank,f=setup()
    c=replace(c,map=replace(c.map,start_m=[40.,40.],goal_m=[100.,40.]),electromagnetic=replace(c.electromagnetic,fields=()))
    ep=run(c,'mixture_numeric',timing='forward',joint_settings=JointSettings(),forecast_method='joint',soft_fit=True)
    assert ep['metrics']['success'] and ep['metrics']['actual_model_calls']==0
    assert len(ep['joint_prediction_records'])==len(ep['observations'])-1
    for index,row in enumerate(ep['joint_prediction_records'],1):
        assert row['issued']['history_sha256']==digest(ep['observations'][:index])
        assert row['issued']['forecast_at_seconds']<row['observation']['time_seconds']
        assert row['issued']['predicted_before_observation']
