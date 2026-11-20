from copy import deepcopy
import math
import numpy as np
import pytest
from uav_em.event_inference import (Budget, EventTrigger, HypothesisBank, digest,
    execute_plan, fit_job, inference_request, numerical_plan, operator_dictionary, validate_plan)


def observation(t, y=-80.):
    return dict(position_m=[100.+t, 100.], time_seconds=float(t), interference_dbm=y)


def public():
    return dict(goal_m=[400., 100.], map=dict(bounds_m=[[0., 0.], [480., 360.]]))


def advance(bank, t, y=-80.):
    p = observation(t, y)
    bank.issue(p['position_m'], p['time_seconds'])
    return bank.observe(p)


def model(identifier='field', now=1., index=1):
    return dict(id=identifier, family='translate', born_index=index, scored_points=0, std_db=4.,
                reference_time_seconds=now, peaks_dbm=[-70.],
                bases=[dict(center_m=[100., 100.], scale_m=40., velocity_mps=[3., 0.],
                            rate_db_per_second=0.)])


def test_issue_cannot_see_new_value_and_likelihood_matches_scalar_mixture():
    bank = HypothesisBank(observation(0))
    issued = bank.issue([101., 100.], 1.)
    assert issued['mean_dbm'] == -80.
    assert 'interference_dbm' not in issued
    row = bank.observe(observation(1, -76.))
    likelihood = [math.exp(-.5*(4./s)**2)/(s*math.sqrt(2*math.pi)) for s in (4., 12.)]
    assert row['nll'] == pytest.approx(-math.log(sum(likelihood)/2))
    assert np.exp(bank.log_weights) == pytest.approx(np.array(likelihood)/sum(likelihood))


def test_measurement_must_match_issued_position_time_exactly_once():
    bank = HypothesisBank(observation(0))
    with pytest.raises(ValueError): bank.observe(observation(1))
    bank.issue([101., 100.], 1.)
    with pytest.raises(ValueError): bank.issue([102., 100.], 2.)
    with pytest.raises(ValueError): bank.observe(observation(2))
    bank.observe(observation(1))
    with pytest.raises(ValueError): bank.observe(observation(1))
    with pytest.raises(ValueError): bank.issue([101., 100.], 1.)


def test_birth_preserves_old_mass_and_does_not_score_fitting_history():
    bank = HypothesisBank(observation(0))
    advance(bank, 1)
    old = np.exp(bank.log_weights)
    bank.birth([model()])
    assert np.exp(bank.log_weights) == pytest.approx(np.r_[.75*old, .25])
    assert bank.models[-1]['scored_points'] == 0
    row = advance(bank, 2)
    assert len(row['issued']['predictions']) == 3
    assert bank.models[-1]['scored_points'] == 1
    assert [m['scored_points'] for m in bank.models[:2]] == [2, 2]


def test_invalid_birth_is_atomic_and_cannot_enter_pending_prediction():
    bank = HypothesisBank(observation(0))
    advance(bank, 1)
    before = digest(bank.snapshot())
    with pytest.raises(ValueError): bank.birth([model('a'), model('a')])
    assert digest(bank.snapshot()) == before
    bank.issue([102., 100.], 2.)
    with pytest.raises(ValueError): bank.birth([model()])


def test_dynamic_hypothesis_advances_without_parameter_replacement():
    bank = HypothesisBank(observation(0))
    advance(bank, 1)
    bank.birth([model()])
    frozen = deepcopy(bank.models[-1])
    issued = bank.issue([127., 100.], 10.)
    expected = -80.+10.*math.log10(.01+10.)
    assert issued['predictions'][-1]['mean_dbm'] == pytest.approx(expected)
    assert bank.models[-1] == frozen


@pytest.mark.parametrize('mode,expected', [('periodic_equal', [8., 20.]), ('event_equal', [8., 16.])])
def test_event_minimum_interval_expiry_and_maximum_count(mode, expected):
    trigger = EventTrigger(mode)
    events = []
    for t in range(1, 40):
        if trigger.check(float(t), t+1, 9.):
            events.append(float(t)); trigger.commit(float(t))
    assert events == expected


def test_small_errors_do_not_trigger_before_expiry():
    trigger = EventTrigger('event_equal')
    events = []
    for t in range(1, 25):
        if trigger.check(float(t), t+1, .01):
            events.append(t);trigger.commit(float(t))
    assert events == [8, 20]
    with pytest.raises(ValueError):trigger.check(24., 25, .01)
    with pytest.raises(ValueError):trigger.commit(24.)


@pytest.mark.parametrize('change', ['stale', 'unknown', 'boolean', 'over_budget', 'nan'])
def test_plan_rejects_unbounded_or_untyped_input(change):
    bank = HypothesisBank(observation(0))
    request = inference_request(bank, public(), 'initial_history_ready', 0)
    response = numerical_plan(request)
    if change == 'stale': response['request_id'] = 'wrong'
    if change == 'unknown': response['source_power'] = -60.
    if change == 'boolean': response['jobs'][0]['max_bases'] = True
    if change == 'over_budget': response['jobs'] *= 2
    if change == 'nan': response['jobs'][0]['max_bases'] = float('nan')
    with pytest.raises(ValueError): validate_plan(request, response)


def test_budget_rejects_whole_plan_without_partial_reservation():
    b = Budget()
    b.reserve([dict(max_bases=2)]*3)
    old = deepcopy(b)
    with pytest.raises(ValueError):b.reserve([dict(max_bases=1)])
    assert b == old


def test_dictionary_centers_stay_in_declared_half_but_are_not_sensing_range():
    job = dict(family='translate', region='left_half', max_bases=2)
    bases = operator_dictionary(job, public(), [100., 100.])
    assert bases and all(b['center_m'][1] >= 100. for b in bases)
    assert any(b['center_m'][1] > 100. for b in bases)
    assert all(np.linalg.norm(np.asarray(b['center_m'])-[100., 100.]) <= 160.+1e-9 for b in bases)


def test_fitting_consumes_only_prefix_and_respects_numerical_bounds():
    bank = HypothesisBank(observation(0))
    for t in range(1, 10):advance(bank, t, -80.+.2*t)
    job = dict(family='static', region='task_disk', max_bases=2)
    budget = Budget();budget.reserve([job])
    fit = fit_job(bank, public(), job, budget, 'new')
    assert fit['model']['bases']
    assert fit['window'] == bank.history
    assert fit['model']['scored_points'] == 0
    assert fit['model']['fit_mse'] < 1.
    assert budget.solver_calls <= 2 and budget.residual_calls <= 80
    assert all(-140. <= p <= fit['bounds_dbm'][1] for p in fit['model']['peaks_dbm'])
    assert all(t['candidate_mse'] <= t['previous_mse']+1e-8 for t in fit['trace'] if t['accepted'])


def test_stale_live_state_rejected_before_fitting_and_invalid_plan_falls_back():
    bank = HypothesisBank(observation(0))
    request = inference_request(bank, public(), 'initial_history_ready', 0)
    advance(bank, 1)
    with pytest.raises(ValueError):execute_plan(bank, request, numerical_plan(request), public(), 'x')
    request = inference_request(bank, public(), 'initial_history_ready', 0)
    result = execute_plan(bank, request, {'request_id': 'bad'}, public(), 'x')
    assert not result['parser_accepted'] and result['fallback_error']
    assert not result['actual_llm_response']
    assert result['resources']['solver_calls'] <= 6


def test_same_prefix_gives_same_request_despite_different_unseen_future():
    a, b = HypothesisBank(observation(0)), HypothesisBank(observation(0))
    for t in range(1, 4):
        advance(a, t, -80.+t);advance(b, t, -80.+t)
    ra = inference_request(a, public(), 'initial_history_ready', 0)
    rb = inference_request(b, public(), 'initial_history_ready', 0)
    assert ra == rb
    advance(a, 4, -40.);advance(b, 4, -120.)
    assert ra['history'][-1]['time_seconds'] == 3.
    assert a.history != b.history

