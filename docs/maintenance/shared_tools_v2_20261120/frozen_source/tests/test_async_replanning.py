"""Synthetic host clocks/points only. No hidden field, real model or research flight."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from uav_em.async_replanning import AsyncReplanner, AsyncSettings
from uav_em.config import load_config
from uav_em.environment import Geometry
from uav_em.motion import advance_route
from uav_em.planning import path_length
from uav_em.scenario_planning import ScenarioForecast


def plan(request, action='keep', waypoint=None, quotas=None):
    return dict(request_id=request['request_id'], action=action, waypoint_m=waypoint,
                sample_quotas=quotas if quotas is not None else {'keep': [], 'direct': [20], 'via': [10, 10]}[action],
                sampling='uniform', reason='synthetic intent, not an actual model')


class ScriptedAdvisor:
    def __init__(self, delay=2., response=None, uncancellable=False):
        self.delay, self.response = delay, response or (lambda public: plan(public['request']))
        self.uncancellable = uncancellable
        self.current = None
        self.requests, self.cancelled = [], []

    def submit(self, public, now):
        assert self.current is None
        token = f'synthetic_{len(self.requests)}'
        self.requests.append(deepcopy(public))
        self.current = dict(token=token, ready_at=now+self.delay, public=deepcopy(public))
        return token

    def poll(self, now):
        if self.current is None or now < self.current['ready_at']:
            return []
        item, self.current = self.current, None
        return [dict(token=item['token'], raw=self.response(item['public']),
                     actual_model_call=False, origin='synthetic', error=None)]

    def cancel(self, token):
        self.cancelled.append(token)
        if self.uncancellable:
            return False
        self.current = None
        return True

    def busy(self):
        return self.current is not None


class SyntheticHost:
    """Moves along the returned route; observations are declared scalar fixtures."""
    def __init__(self, transport=None, settings=AsyncSettings(), gate=None, obstacles=()):
        cfg = load_config('configs/local_power.json')
        self.geometry = Geometry(replace(cfg.map, buildings=(), obstacles_m=obstacles), 0., 30.)
        self.state = dict(position_m=[20., 60.], goal_m=[380., 60.],
                          route_m=[[20., 60.], [380., 60.]], speed_mps=10.,
                          spent_length_m=0., maximum_length_m=450., remaining_energy_wh=100.,
                          flight_power_w=350., hover_power_w=300., reserve_energy_wh=15., delay_seconds=0.)
        self.history, self.results = [], []
        self.now = 0.
        self.scheduler = AsyncReplanner(self.geometry, transport=transport, settings=settings,
                                       **({} if gate is None else dict(gate=gate)))

    def tick(self, now, power=-85., innovation=None, hot=False):
        assert now >= self.now
        duration = now-self.now
        if duration:
            remaining, traversed = advance_route(self.state['route_m'], duration*self.state['speed_mps'])
            self.state['route_m'] = remaining
            self.state['position_m'] = deepcopy(remaining[0])
            self.state['spent_length_m'] += path_length(traversed)
            self.state['remaining_energy_wh'] -= duration*self.state['flight_power_w']/3600.
        self.now = now
        self.history.append(dict(position_m=deepcopy(self.state['position_m']), time_seconds=now,
                                 interference_dbm=power))
        models = [dict(id='persistence', family='persistence', born_index=0, std_db=4., weight=1.)]
        if hot:
            # A declared synthetic forecast, not an inferred or revealed simulator source.
            models = [dict(id='synthetic_field', family='static', born_index=0, std_db=0., weight=1.,
                           reference_time_seconds=0., peaks_dbm=[-60.],
                           bases=[dict(center_m=[150., 60.], scale_m=25., velocity_mps=[0., 0.],
                                       rate_db_per_second=0.)])]
        forecast = ScenarioForecast(self.history, models, now)
        result = self.scheduler.tick(self.state, forecast, innovation)
        self.state = deepcopy(result['state'])
        self.results.append(result)
        return result


def events(result, kind):
    return [e for e in result['events'] if e['kind'] == kind]


def test_no_consultation_still_reviews_and_flies_without_free_hover():
    advisor = ScriptedAdvisor()
    host = SyntheticHost(advisor, gate=lambda _: False)
    a, b = host.tick(0.), host.tick(8.)
    assert len(events(a, 'tool_transaction')) == len(events(b, 'tool_transaction')) == 1
    assert host.state['position_m'] == [100., 60.]
    assert host.state['spent_length_m'] == 80.
    assert host.state['remaining_energy_wh'] == pytest.approx(100.-8.*350./3600.)
    assert not advisor.requests and b['consultations'] == 0


def test_pending_response_does_not_block_moving_observations_or_numeric_alarm():
    advisor = ScriptedAdvisor(delay=10.)
    host = SyntheticHost(advisor)
    for t in range(4):
        host.tick(float(t))
    row = host.tick(4., power=-70.)
    assert row['pending'] and row['consultations'] == 1
    assert host.state['position_m'] == [60., 60.]
    assert 'observed_power_change' in events(row, 'observation_monitor')[0]['features']['reasons']
    assert events(row, 'tool_transaction')
    assert events(row, 'consultation_blocked')[0]['reason'] == 'request_in_flight'
    assert len(host.history) == 5
    assert advisor.requests[0]['request']['history'][-1]['time_seconds'] == 0.


def test_reply_is_rebased_and_scored_from_latest_position_and_forecast():
    advisor = ScriptedAdvisor(response=lambda p: plan(p['request'], 'via', [150., 110.]))
    host = SyntheticHost(advisor)
    host.tick(0.)
    old_id = advisor.requests[0]['request']['request_id']
    row = host.tick(2., hot=True)
    rebound = events(row, 'reply_rebased')[0]
    assert rebound['origin_request_id'] == old_id and rebound['rebased_request_id'] != old_id
    assert rebound['position_m'] == [40., 60.]
    transaction = [e for e in events(row, 'tool_transaction') if e['source'] == 'reply'][0]
    proposals = [c for c in transaction['candidates'] if c['id'].startswith('proposal')]
    assert proposals and all(c['path'][0] == [40., 60.] for c in proposals)
    assert all(min(c['score']['query_time_seconds']) >= 2. for c in proposals if c['feasible'])
    commit = events(row, 'route_committed')[0]
    assert commit['source'] == 'reply' and commit['selected_score'] < commit['previous_score']
    assert row['state']['route_m'][0] == [40., 60.]


def test_keep_does_not_restore_old_start_or_route():
    host = SyntheticHost(ScriptedAdvisor())
    host.tick(0.)
    row = host.tick(2.)
    assert events(row, 'reply_parsed') and not events(row, 'reply_rebased')
    assert row['state']['route_m'] == [[40., 60.], [380., 60.]]


@pytest.mark.parametrize('waypoint', [[30., 60.], [380., 60.]])
def test_passed_waypoint_or_goal_as_via_is_rejected(waypoint):
    host = SyntheticHost(ScriptedAdvisor(response=lambda p: plan(p['request'], 'via', waypoint)))
    host.tick(0.)
    row = host.tick(2.)
    assert events(row, 'reply_rejected')[0]['reason'] == 'waypoint_no_longer_forward'
    assert not events(row, 'reply_rebased')
    assert row['state']['route_m'] == [[40., 60.], [380., 60.]]


@pytest.mark.parametrize('raw', ['{bad', '{}', dict(request_id='wrong'), 'null', '[]'])
def test_illegal_reply_never_changes_route(raw):
    host = SyntheticHost(ScriptedAdvisor(response=lambda _: raw))
    host.tick(0.)
    row = host.tick(2.)
    assert events(row, 'reply_rejected') and not events(row, 'route_committed')
    assert row['state']['route_m'][0] == [40., 60.]


@pytest.mark.parametrize('delay, rejected', [(12., False), (12.1, True)])
def test_deadline_uses_physical_host_time_including_transport_delay(delay, rejected):
    host = SyntheticHost(ScriptedAdvisor(delay=delay), settings=AsyncSettings(max_requests=1))
    host.tick(0.)
    row = host.tick(delay)
    assert bool(events(row, 'reply_rejected')) is rejected
    assert bool(events(row, 'reply_parsed')) is not rejected


def test_uncancellable_worker_keeps_transport_slot_after_timeout_and_late_reply_is_ignored():
    advisor = ScriptedAdvisor(delay=20., uncancellable=True)
    host = SyntheticHost(advisor, gate=lambda _: True)
    host.tick(0.)
    row = host.tick(13.)
    assert not row['pending'] and len(advisor.requests) == 1
    assert events(row, 'request_closed')[0]['cancellation_confirmed'] is False
    assert events(row, 'consultation_blocked')[0]['reason'] == 'transport_still_busy'
    row = host.tick(20.)
    assert events(row, 'reply_ignored') and not events(row, 'reply_parsed')
    assert len(advisor.requests) == 2


def test_call_budget_exhaustion_never_stops_numeric_response():
    advisor = ScriptedAdvisor()
    host = SyntheticHost(advisor, settings=AsyncSettings(max_requests=0), gate=lambda _: True)
    for t in range(4):
        host.tick(float(t))
    row = host.tick(4., power=-70.)
    assert events(row, 'tool_transaction') and not row['stop']
    assert not advisor.requests
    assert events(row, 'consultation_blocked')[0]['reason'] == 'consultation_budget_exhausted'


def test_current_blocked_route_gets_numeric_rrt_repair_without_consultation():
    host = SyntheticHost(gate=lambda _: False, obstacles=((130., 40., 160., 80.),))
    row = host.tick(0.)
    assert not row['stop'] and events(row, 'route_committed')
    assert row['state']['route_m'][0] == [20., 60.]
    assert all(host.geometry.edge_free(a, b) for a, b in zip(row['state']['route_m'], row['state']['route_m'][1:]))
    assert 0 < row['resources']['samples'] <= 700


@pytest.mark.parametrize('field,value', [('remaining_energy_wh', 15.), ('maximum_length_m', 200.)])
def test_no_feasible_length_or_energy_path_stops_without_waiting(field, value):
    advisor = ScriptedAdvisor()
    host = SyntheticHost(advisor)
    host.state[field] = value
    row = host.tick(0.)
    assert row['stop'] == 'no_feasible_numeric_route'
    assert not advisor.requests and not events(row, 'route_committed')


def test_execution_geometry_check_cannot_get_free_edges_when_budget_exhausted():
    host = SyntheticHost(settings=AsyncSettings(max_edges=1, max_transactions=0))
    first = host.tick(0.)
    assert not first['stop'] and first['resources']['edges'] == 1
    last = host.tick(1.)
    assert last['stop'] == 'execution_check_budget_exhausted'
    assert last['resources']['edges'] == 1


def test_zero_sample_budget_is_preserved_and_cannot_repair_obstruction():
    host = SyntheticHost(settings=AsyncSettings(max_samples=0), obstacles=((130., 40., 160., 80.),))
    row = host.tick(0.)
    assert row['resources']['samples'] == 0
    assert row['stop'] == 'no_feasible_numeric_route'


def test_numeric_and_reply_transactions_share_one_task_budget():
    advisor = ScriptedAdvisor(response=lambda p: plan(p['request'], 'direct'))
    host = SyntheticHost(advisor, settings=AsyncSettings(max_transactions=2, max_samples=40))
    a = host.tick(0.)
    b = host.tick(2.)
    tx = events(a, 'tool_transaction')+events(b, 'tool_transaction')
    assert len(tx) == 2 and b['resources']['transactions'] == 2
    assert b['resources']['edges'] == sum(t['resources']['edges'] for t in tx)+2
    assert b['resources']['samples'] <= 40
    c = host.tick(8.)
    assert events(c, 'tool_blocked') and not c['stop']


def test_original_valid_quota_is_rejected_if_live_task_quota_was_spent():
    advisor = ScriptedAdvisor(response=lambda p: plan(p['request'], 'direct', quotas=[20]))
    host = SyntheticHost(advisor, settings=AsyncSettings(max_samples=40))
    host.tick(0.)
    # Represents paid work completed by other host activity in this same task.
    host.scheduler.budget.samples = 39
    row = host.tick(2.)
    tx = [e for e in events(row, 'tool_transaction') if e['source'] == 'reply'][0]
    assert tx['error'] == 'invalid/stale search plan' and not tx['changed']
    assert row['resources']['samples'] == 39


@pytest.mark.parametrize('gate', [lambda _: 'yes', lambda _: 1, lambda _: (_ for _ in ()).throw(RuntimeError('bad gate'))])
def test_bad_gate_is_logged_but_numeric_path_is_available(gate):
    host = SyntheticHost(ScriptedAdvisor(), gate=gate)
    row = host.tick(0.)
    assert events(row, 'gate_error') and events(row, 'tool_transaction') and not row['pending']


def test_callback_and_gate_mutations_do_not_change_snapshot_or_caller_state():
    def gate(features):
        features['task_budget']['samples_left'] = -10
        return True
    class Mutating(ScriptedAdvisor):
        def submit(self, public, now):
            token = super().submit(public, now)
            public['request']['history'][0]['interference_dbm'] = 1e9
            public['request']['state']['remaining_energy_wh'] = 1e9
            return token
    advisor = Mutating()
    host = SyntheticHost(advisor, gate=gate)
    old_state = deepcopy(host.state)
    host.tick(0.)
    assert host.scheduler.pending['request']['history'][0]['interference_dbm'] == -85.
    assert host.history[0]['interference_dbm'] == -85.
    assert host.scheduler.pending['request']['state']['remaining_energy_wh'] == old_state['remaining_energy_wh']


def test_public_request_has_no_simulator_source_or_future_observation():
    advisor = ScriptedAdvisor()
    host = SyntheticHost(advisor)
    host.tick(0.)
    public = advisor.requests[0]
    assert max(p['time_seconds'] for p in public['request']['history']) == 0.
    assert not any(k in json.dumps(public) for k in ('eirp_dbm', 'measurement_noise_db', 'fields', 'truth_dbm'))
    assert public['deadline_seconds'] == 12.


def test_goal_cancels_pending_without_extra_consultation_or_search():
    advisor = ScriptedAdvisor(delay=100.)
    host = SyntheticHost(advisor)
    host.state['goal_m'] = [40., 60.]
    host.state['route_m'][-1] = [40., 60.]
    host.tick(0.)
    result = host.tick(2.)
    assert result['stop'] == 'goal_reached' and not result['pending']
    assert not events(result, 'tool_transaction') and len(advisor.requests) == 1
    with pytest.raises(ValueError, match='finished'):
        host.tick(3.)


def test_changed_prefix_or_resource_reset_is_rejected_before_tool_work():
    host = SyntheticHost()
    host.tick(0.)
    paid = host.scheduler.budget.summary()
    host.history[0]['interference_dbm'] = -10.
    with pytest.raises(ValueError, match='causal prefix'):
        host.tick(1.)
    assert host.scheduler.budget.summary() == paid


def test_more_recent_observations_change_rejection_of_a_once_useful_detour():
    # At dispatch a via could have been useful; the latest flat forecast makes
    # it slower. The reply cannot bring an old, optimistic score into the acceptor.
    host = SyntheticHost(ScriptedAdvisor(response=lambda p: plan(p['request'], 'via', [150., 110.])))
    host.tick(0.)
    row = host.tick(2., power=-99.)
    tx = [e for e in events(row, 'tool_transaction') if e['source'] == 'reply'][0]
    assert any(c['id'] == 'proposal_raw' and c['feasible'] for c in tx['candidates'])
    assert not tx['changed'] and not events(row, 'route_committed')
    assert row['state']['route_m'] == [[40., 60.], [380., 60.]]


@pytest.mark.parametrize('phase', ['submit', 'poll'])
def test_transport_errors_do_not_interrupt_numeric_execution(phase):
    class Broken(ScriptedAdvisor):
        def submit(self, public, now):
            if phase == 'submit':
                raise RuntimeError('synthetic dispatch error')
            return super().submit(public, now)

        def poll(self, now):
            if phase == 'poll' and now > 0:
                raise RuntimeError('synthetic poll error')
            return super().poll(now)
    host = SyntheticHost(Broken())
    a, b = host.tick(0.), host.tick(8.)
    assert any(e['kind'] == 'transport_error' for e in a['events']+b['events'])
    assert b['state']['position_m'] == [100., 60.]
    assert events(b, 'tool_transaction') and not b['stop']


def test_duplicate_and_unknown_replies_are_ignored():
    class Duplicates(ScriptedAdvisor):
        def poll(self, now):
            replies = super().poll(now)
            return replies+deepcopy(replies)+[dict(token='unknown', raw={})]
    host = SyntheticHost(Duplicates())
    host.tick(0.)
    row = host.tick(2.)
    assert len(events(row, 'reply_parsed')) == 1 and len(events(row, 'reply_ignored')) == 2


def test_insufficient_search_edge_budget_is_charged_and_never_executes_unverified_route():
    host = SyntheticHost(settings=AsyncSettings(max_edges=3), obstacles=((130., 40., 160., 80.),))
    row = host.tick(0.)
    assert row['stop'] == 'no_feasible_numeric_route'
    assert row['resources']['edges'] <= 3
    assert not events(row, 'route_committed')


def test_goal_does_not_override_a_preexisting_resource_violation():
    host = SyntheticHost()
    host.state['position_m'] = deepcopy(host.state['goal_m'])
    host.state['route_m'] = [deepcopy(host.state['goal_m'])]
    host.state['remaining_energy_wh'] = 14.
    row = host.tick(0.)
    assert row['stop'] == 'mission_resource_violation'


def test_energy_cannot_reset_between_observations():
    host = SyntheticHost()
    host.tick(0.)
    host.state['remaining_energy_wh'] = 200.
    with pytest.raises(ValueError, match='cannot reset'):
        host.tick(1.)


def test_frozen_forecast_mutation_is_rejected_before_spending_budget():
    host = SyntheticHost()
    history = [dict(position_m=[20., 60.], time_seconds=0., interference_dbm=-85.)]
    forecast = ScenarioForecast(history, [dict(id='p', family='persistence', std_db=4., weight=1., born_index=0)], 0.)
    forecast.models[0]['weight'] = .5
    with pytest.raises(ValueError, match='mutated'):
        host.scheduler.tick(host.state, forecast)
    assert host.scheduler.budget.summary()['edges'] == 0


@pytest.mark.parametrize('setting,value', [('max_requests', True), ('max_samples', -1),
                                          ('request_ttl_seconds', 0.), ('change_db', float('nan'))])
def test_invalid_scheduler_settings_are_rejected(setting, value):
    with pytest.raises(ValueError):
        AsyncSettings(**{setting: value})
