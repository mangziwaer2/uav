"""Synthetic source-tool regressions; no world, flight or cached state inputs."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from uav_em import event_inference as tool
from uav_em.equivalent_field import kernel, power_and_jacobian


def base(center, scale=40., velocity=(0., 0.), rate=0.):
    return dict(center_m=list(center), scale_m=scale,
                velocity_mps=list(velocity), rate_db_per_second=rate)


def problem(capacity=2, exact_warm=False):
    public = dict(goal_m=[500., 300.], map=dict(bounds_m=[[0., 0.], [600., 600.]]))
    xy = [[x, y] for y in (220., 260., 300.) for x in (160., 200., 240., 280.)]
    xy[-1] = [300., 300.]
    times = np.arange(len(xy), dtype=float)
    source = base([220., 300.])
    values = power_and_jacobian(kernel([source], xy, times, times[-1]), [-65.])[0]
    history = [dict(position_m=p, time_seconds=float(t), interference_dbm=float(y))
               for p, t, y in zip(xy, times, values)]
    old = dict(id='old', family='static', std_db=4., born_index=0, scored_points=11,
               reference_time_seconds=0., bases=[source if exact_warm else base([380., 300.])],
               peaks_dbm=[-65.])
    if capacity == 2 and not exact_warm:
        old['bases'].append(base([300., 380.]))
        old['peaks_dbm'].append(-70.)
    bank = SimpleNamespace(history=history, models=[old], log_weights=np.zeros(1))
    job = dict(family='static', region='task_disk', max_bases=capacity)
    return bank, public, job


def fit(bank, public, job, soft=None):
    budget = tool.Budget()
    budget.reserve([job])
    result = tool.fit_job(bank, public, job, budget, 'new', soft=soft)
    return result, budget


@pytest.mark.parametrize('capacity,maximum_mse', [(1, 14.), (2, 8.)])
def test_full_wrong_warm_can_be_replaced_in_original_solver_budget(capacity, maximum_mse):
    bank, public, job = problem(capacity)
    before = deepcopy(bank.__dict__)
    result, budget = fit(bank, public, job)
    cold_bank = SimpleNamespace(history=bank.history, models=[], log_weights=[])
    cold, _ = fit(cold_bank, public, job)
    assert result['model']['fit_mse'] < maximum_mse
    assert result['fit_objective'] == pytest.approx(cold['fit_objective'], abs=1e-8)
    assert result['selected_branch'] == 'cold_dictionary'
    assert result['warm_candidate']['residual_calls'] == 1
    assert result['warm_start_bases'] == capacity
    assert budget.solver_calls == capacity == result['job_resources']['solver_calls']
    assert budget.residual_calls == 1+sum(row['residual_calls'] for row in result['trace'])
    assert budget.residual_calls <= budget.reserved_residual_calls == 40*capacity
    assert budget.dictionary_kernel_terms == len(result['dictionary'])*len(bank.history)
    assert result['model']['scored_points'] == 0 and result['model']['born_index'] == 11
    assert result['model']['reference_time_seconds'] == 11.
    assert bank.history == before['history'] and bank.models == before['models']
    np.testing.assert_array_equal(bank.log_weights, before['log_weights'])


def test_exact_warm_seed_is_retained_when_cold_dictionary_is_worse():
    bank, public, job = problem(1, exact_warm=True)
    result, budget = fit(bank, public, job)
    assert result['selected_branch'] == 'warm_seed'
    assert result['fit_objective'] < 1e-20
    assert result['model']['bases'] == bank.models[0]['bases']
    assert result['warm_candidate']['selected_for_return']
    assert budget.solver_calls == 1 and budget.residual_calls <= 40


@pytest.mark.parametrize('evaluate_cold', [False, True])
def test_solver_failure_keeps_best_already_evaluated_warm_or_cold(monkeypatch, evaluate_cold):
    bank, public, job = problem(1)

    def failed_solver(objective, initial, **kwargs):
        if evaluate_cold:
            objective(initial)
        raise ValueError('synthetic numerical failure')

    monkeypatch.setattr(tool, 'least_squares', failed_solver)
    result, budget = fit(bank, public, job)
    assert result['model']['bases']
    assert result['stop_reason'] == 'synthetic numerical failure'
    assert not result['trace'][0]['solver_success']
    assert result['selected_branch'] == ('cold_dictionary' if evaluate_cold else 'warm_seed')
    assert budget.solver_calls == 1 and budget.residual_calls == 1+int(evaluate_cold)
    assert result['model']['scored_points'] == 0


def test_warm_candidate_soft_constraints_are_charged_without_new_evidence():
    bank, public, job = problem(2)
    xy = [[280., 300.], [300., 280.]]
    soft = dict(position_m=xy, time_seconds=[11., 11.], mean_dbm=[-72., -76.],
                weights=[.25, .75], strength=.25, prefix_end_seconds=11.,
                independent_evidence_points=0)
    original = deepcopy(soft)
    result, budget = fit(bank, public, job, soft)
    model = result['model']
    predicted = power_and_jacobian(kernel(model['bases'], xy, [11., 11.], 11.), model['peaks_dbm'])[0]
    expected_shape = float(np.array(soft['weights'])@((predicted-soft['mean_dbm'])**2))
    assert result['gp_shape_mse'] == pytest.approx(expected_shape)
    assert result['fit_objective'] == pytest.approx(model['fit_mse']+.25*expected_shape)
    assert result['gp_dictionary_kernel_terms'] == len(result['dictionary'])*2
    assert budget.dictionary_kernel_terms == len(result['dictionary'])*14
    assert budget.residual_calls == 1+sum(row['residual_calls'] for row in result['trace']) <= 80
    assert result['window'] == bank.history and len(bank.history) == 12
    assert model['scored_points'] == 0 and not result['training_error_is_predictive_evidence']
    assert soft == original


def test_warm_positions_and_peaks_are_rebased_before_prefix_evaluation(monkeypatch):
    bank, public, job = problem(1)
    old = bank.models[0]
    old['family'] = job['family'] = 'translate'
    old['bases'][0] = base([300., 300.], velocity=(2., -1.), rate=.5)
    old['peaks_dbm'] = [-80.]
    monkeypatch.setattr(tool, 'operator_dictionary', lambda *_: [])
    result, budget = fit(bank, public, job)
    assert result['dictionary'][0]['center_m'] == [322., 289.]
    assert result['warm_candidate']['peaks_dbm'] == [-74.5]
    assert result['model']['reference_time_seconds'] == 11.
    assert budget.solver_calls == 1 and 1 < budget.residual_calls <= 40


def test_unreserved_fit_does_not_evaluate_or_publish_warm_candidate():
    bank, public, job = problem(1)
    budget = tool.Budget()
    with pytest.raises(ValueError, match='unreserved fitting budget'):
        tool.fit_job(bank, public, job, budget, 'new')
    assert budget.solver_calls == budget.residual_calls == 0


def test_off_dictionary_warm_geometry_remains_available_for_amplitude_refit(monkeypatch):
    bank,public,job=problem(1)
    old=bank.models[0];old['bases']=[base([225.,305.],scale=55.)];old['peaks_dbm']=[-85.]
    xy=[p['position_m'] for p in bank.history];times=[p['time_seconds'] for p in bank.history]
    labels=power_and_jacobian(kernel(old['bases'],xy,times,11.),[-65.])[0]
    for row,value in zip(bank.history,labels):row['interference_dbm']=float(value)
    # Only this off-dictionary shape is available. It must be optional and
    # amplitude-adjustable, even though its old amplitude is now wrong.
    monkeypatch.setattr(tool,'operator_dictionary',lambda *_: [])
    result,budget=fit(bank,public,job)
    assert result['model']['bases']==old['bases'] and result['model']['fit_mse']<1e-12
    assert result['model']['peaks_dbm']==pytest.approx([-65.],abs=1e-6)
    assert result['selected_branch']=='cold_dictionary' and budget.solver_calls==1
    assert budget.residual_calls<=40


def test_nonfinite_solver_optimality_does_not_prevent_strict_json_save(monkeypatch):
    bank,public,job=problem(1)
    def stopped(objective,initial,**kwargs):
        objective(initial)
        return SimpleNamespace(x=initial,success=False,status=0,message='synthetic',nfev=1,njev=0,optimality=float('inf'))
    monkeypatch.setattr(tool,'least_squares',stopped)
    result,_=fit(bank,public,job)
    assert result['model']['bases'] and result['trace'][0]['optimality'] is None
    json.dumps(result,allow_nan=False)
