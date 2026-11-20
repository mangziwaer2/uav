import copy
import pytest
from uav_em.decision_timing import TimingMonitor, TimingSettings, predicted_escalation
from uav_em.decision_timing import PowerRiseTrigger


def score():
    return dict(queries=dict(time_seconds=[10.,14.,18.]),
                scenario_mean_dbm=[[-80.,-73.,-72.],[-80.,-80.,-80.]],
                scenario_scores=[dict(id='moving',weight=.3),dict(id='constant',weight=.7)])


def test_shared_alternative_mass_not_sum_of_all_source_amplitudes():
    event=predicted_escalation(score(),9.,8.)
    assert event['event_time_seconds']==14. and event['model_ids']==['moving']
    assert not event['calibrated']
    assert predicted_escalation(score(),9.,8.,mass=.5) is None


def test_lookahead_never_uses_past_route_nodes_as_current_baseline():
    assert predicted_escalation(score(),15.,8.) is None
    assert predicted_escalation(score(),9.,8.,horizon=3.) is None


def test_future_forecast_and_misaligned_values_are_rejected():
    with pytest.raises(ValueError):predicted_escalation(score(),7.,8.)
    data=score();data['scenario_mean_dbm'][0].pop()
    with pytest.raises(ValueError):predicted_escalation(data,9.,8.)


def test_mutating_a_future_suffix_cannot_change_the_observation_prefix():
    prefix=[(0.,-90.,None),(1.,-90.,.1),(2.,-89.,.2)]
    a,b=TimingMonitor(),TimingMonitor()
    expected=[a.observe(*r) for r in prefix]
    actual=[b.observe(*r) for r in prefix]
    b.observe(3.,-20.,20.)
    assert expected==actual


def test_initialization_and_watchdog_do_not_wait_for_old_20_second_event():
    m=TimingMonitor()
    assert m.observe(0.,-90.)['requested']
    assert not m.observe(7.,-90.,.01)['requested']
    assert m.observe(8.,-90.,.01)['reasons']==['plan_expired']
    assert m.observe(16.,-90.,.01)['requested']


def test_three_preissued_mismatch_readings_trigger_without_expiry():
    m=TimingMonitor();m.observe(0.,-90.)
    assert not m.observe(1.,-90.,3.)['requested']
    assert not m.observe(2.,-90.,3.)['requested']
    row=m.observe(3.,-90.,3.)
    assert row['requested'] and row['reasons']==['predictive_mismatch']


def test_urgent_forecast_bypasses_cooldown_and_marks_latency_deficit():
    m=TimingMonitor();m.observe(0.,-90.)
    event=dict(forecast_at_seconds=0.,event_time_seconds=5.,forecast_key='v0')
    row=m.observe(1.,-90.,.1,event)
    assert row['requested'] and row['fast_numeric_review_needed']
    assert row['llm_deadline_slack_seconds']==-11.
    assert not m.observe(1.5,-90.,.1,event)['requested']


def test_future_event_is_scheduled_before_event_not_after_crossing():
    m=TimingMonitor(TimingSettings(maximum_age_seconds=100.))
    m.observe(0.,-90.)
    e=dict(forecast_at_seconds=0.,event_time_seconds=20.,forecast_key='v1')
    assert not m.observe(4.,-90.,.1,e)['requested']
    row=m.observe(5.,-90.,.1,e)
    assert row['requested'] and row['llm_deadline_slack_seconds']==0.


def test_urgent_does_not_bypass_total_budget_and_goal_needs_no_request():
    m=TimingMonitor(TimingSettings(max_requests=1));m.observe(0.,-90.)
    e=dict(forecast_at_seconds=0.,event_time_seconds=2.,forecast_key='v0')
    row=m.observe(1.,-80.,8.,e,route_feasible=False)
    assert not row['requested'] and row['blocker']=='request_budget_exhausted'
    assert row['fast_numeric_review_needed'] and not row['route_changed']
    row=m.observe(2.,-80.,8.,e,route_feasible=False,goal_reached=True)
    assert not row['reasons'] and not row['fast_numeric_review_needed']


def test_observed_change_does_not_claim_a_source_direction():
    m=TimingMonitor()
    for t in range(3):m.observe(float(t),-90.,.1)
    row=m.observe(3.,-83.,.1)
    assert row['requested'] and row['reasons']==['observed_power_change']


def test_strict_time_validation_does_not_consume_a_bad_sample():
    m=TimingMonitor();m.observe(0.,-90.)
    for now in [0.,-1.,float('nan')]:
        with pytest.raises(ValueError):m.observe(now,-90.)
    assert not m.observe(1.,-90.,.1)['requested']


def test_invalid_settings_and_weights_fail_instead_of_silently_normalizing():
    with pytest.raises(ValueError):TimingSettings(min_interval_seconds=10.,maximum_age_seconds=2.)
    with pytest.raises(ValueError):TimingSettings(max_requests=True)
    data=copy.deepcopy(score());data['scenario_scores'][0]['weight']=.8
    with pytest.raises(ValueError):predicted_escalation(data,9.,8.)


def test_simple_rise_has_no_initial_periodic_falling_or_plateau_consultation():
    m=PowerRiseTrigger()
    assert not m.observe(0.,-90.)['requested']
    assert not m.observe(8.,-90.)['requested']
    assert not m.observe(20.,-96.)['requested']
    row=m.observe(21.,-90.)
    assert row['requested'] and row['reasons']==['observed_power_rise']
    assert not row['source_direction_known']
    assert not m.observe(32.,-90.)['requested']


def test_simple_rise_catches_gradual_growth_without_waiting_for_fixed_event():
    m=PowerRiseTrigger()
    rows=[m.observe(float(t),-90.+2*t) for t in range(4)]
    assert [r['time_seconds'] for r in rows if r['requested']]==[3.]
    assert rows[-1]['rise_db']==6.


def test_simple_rise_cooldown_and_call_cap_are_explicit():
    m=PowerRiseTrigger(max_requests=2)
    m.observe(0.,-90.)
    assert m.observe(1.,-84.)['requested']
    assert m.observe(2.,-78.)['blocker']=='cooldown'
    assert m.observe(3.,-77.)['requested']
    assert m.observe(5.,-70.)['blocker']=='request_budget_exhausted'
    with pytest.raises(ValueError):m.observe(4.,-60.)


def test_simple_rise_rejects_invalid_parameters():
    with pytest.raises(ValueError):PowerRiseTrigger(rise_db=float('nan'))
    with pytest.raises(ValueError):PowerRiseTrigger(max_requests=True)

