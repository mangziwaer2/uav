"""Nonblocking public-state scheduler. The host owns sensing and physical time.

This interface charges numerical work but does not map solver wall time onto
flight time. It is not a real-time flight driver. See async_replanning_v1.md.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
import statistics
import time

import numpy as np

from .event_inference import digest
from .feasible_scenario import best, evaluate, execute, feedback_plan, pool, resource_feasible
from .repair_search import BudgetExhausted, WorkBudget
from .scenario_planning import state_request, validate_search_plan


@dataclass(frozen=True)
class AsyncSettings:
    minimum_interval_seconds: float = 2.
    review_age_seconds: float = 8.
    request_ttl_seconds: float = 12.
    change_db: float = 6.
    innovation_squared_mean: float = 2.25
    max_requests: int = 4
    max_transactions: int = 12
    max_samples: int = 2800
    max_edges: int = 40000

    def __post_init__(self):
        for name in ('minimum_interval_seconds', 'review_age_seconds', 'request_ttl_seconds',
                     'change_db', 'innovation_squared_mean'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('finite positive timing thresholds required')
        if self.review_age_seconds < self.minimum_interval_seconds:
            raise ValueError('review expiry precedes minimum interval')
        for name in ('max_requests', 'max_transactions', 'max_samples', 'max_edges'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError('nonnegative integer task budgets required')


class TaskBudget:
    """One mission ledger shared by execution checks and all planning transactions."""
    def __init__(self, settings):
        self.settings = settings
        self.samples = self.edges = self.transactions = 0

    def summary(self):
        s = self.settings
        return dict(samples=self.samples, edges=self.edges, transactions=self.transactions,
                    samples_left=s.max_samples-self.samples, edges_left=s.max_edges-self.edges,
                    transactions_left=s.max_transactions-self.transactions)

    def check_route(self, state, geometry):
        if not resource_feasible(state['route_m'], state):
            return False
        for a, b in zip(state['route_m'], state['route_m'][1:]):
            if self.edges >= self.settings.max_edges:
                raise BudgetExhausted('execution_check_budget_exhausted')
            self.edges += 1
            if not geometry.edge_free(a, b):
                return False
        return True

    def work(self):
        left = self.summary()
        if not left['transactions_left'] or not left['edges_left']:
            return None
        self.transactions += 1
        # The existing dataclass constructor requires positive limits. Explicit
        # zero sampling is preserved after construction, never replaced by 700.
        samples = min(700, left['samples_left'])
        work = WorkBudget(max(1, samples), min(10000, left['edges_left']))
        work.sample_limit = samples
        return work

    def charge(self, work):
        self.samples += work.samples
        self.edges += work.edges
        if self.samples > self.settings.max_samples or self.edges > self.settings.max_edges:
            raise RuntimeError('task resource accounting exceeded its hard limit')


def rule_gate(features):
    """Engineering reference only, not a learned consultation value model."""
    return bool(features['reasons'])


class AsyncReplanner:
    """Each tick accepts a new causal observation snapshot; no world capability.

    transport.submit(public, now) returns a unique token, poll(now) is nonblocking
    and returns token/raw records, cancel(token) best-effort, busy() reports actual
    outstanding work (including uncancellable work after expiry).
    """
    def __init__(self, geometry, transport=None, gate=rule_gate, settings=AsyncSettings(),
                 method='mixture', seed=0):
        if method not in ('mixture', 'gp', 'persistence'):
            raise ValueError('unknown prediction method')
        self.geometry, self.transport, self.gate = geometry, transport, gate
        self.settings, self.method, self.seed = settings, method, seed
        self.budget = TaskBudget(settings)
        self.pending = None
        self.requests = self.route_version = 0
        self.last_now = self.last_numeric = self.last_consult = None
        self.history = []
        self.powers, self.innovations, self.events = [], [], []
        self.previous_state = None
        self.finished = False
        self.used_tokens = set()

    def log(self, kind, now, **fields):
        row = dict(kind=kind, time_seconds=now, **deepcopy(fields))
        self.events.append(row)
        return row

    def _validate(self, state, forecast, innovation):
        now = forecast.now
        if self.finished:
            raise ValueError('scheduler is finished')
        if not math.isfinite(now) or now < 0 or self.last_now is not None and now <= self.last_now:
            raise ValueError('strictly increasing physical observation time required')
        if innovation is not None and (isinstance(innovation, bool) or not math.isfinite(innovation) or innovation < 0):
            raise ValueError('finite nonnegative pre-observation innovation required')
        # Reuse the existing typed public-state validation, without charging a transaction.
        work = WorkBudget(1, 1)
        state_request(state, forecast, self.geometry, work)
        if digest(dict(history=forecast.history, models=forecast.models, now=forecast.now,
                       gp_version=forecast.gp_version)) != forecast.fingerprint:
            raise ValueError('frozen forecast was mutated before the tick')
        if state['delay_seconds'] != 0.:
            raise ValueError('async host must not invent fixed consultation hover delay')
        if (not forecast.history or forecast.history[-1]['time_seconds'] != now
                or not np.array_equal(forecast.history[-1]['position_m'], state['position_m'])
                or not math.isfinite(forecast.history[-1]['interference_dbm'])):
            raise ValueError('latest observation must match current state and forecast time')
        if self.history and (len(forecast.history) != len(self.history)+1
                             or digest(forecast.history[:-1]) != digest(self.history)):
            raise ValueError('each tick must append exactly one observation to the causal prefix')
        route = np.asarray(state['route_m'], float)
        if (route.ndim != 2 or route.shape[1:] != (2,) or not len(route)
                or not np.all(np.isfinite(route))
                or not np.array_equal(route[0], state['position_m'])
                or not np.array_equal(route[-1], state['goal_m'])):
            raise ValueError('remaining route must run from current position to goal')
        if self.previous_state is not None:
            p = self.previous_state
            if (state['spent_length_m'] < p['spent_length_m'] or
                    state['remaining_energy_wh'] > p['remaining_energy_wh']+1e-10 or
                    state['goal_m'] != p['goal_m'] or
                    state['maximum_length_m'] != p['maximum_length_m']):
                raise ValueError('mission resources/goal cannot reset between ticks')
        return now

    def _request(self, state, forecast):
        left = self.budget.summary()
        work = WorkBudget(max(1, min(700, left['samples_left'])),
                          max(1, min(10000, left['edges_left'])))
        work.sample_limit = min(700, left['samples_left'])
        return state_request(state, forecast, self.geometry, work)

    def _transaction(self, state, forecast, feasible, proposal=None, origin=None):
        now = forecast.now
        work = self.budget.work()
        if work is None:
            self.log('tool_blocked', now, origin=origin, reason='task_budget_exhausted')
            return state, feasible
        started = time.perf_counter()
        before = deepcopy(forecast.counts)
        candidates, searches, error = [], [], None
        current_score = None
        try:
            incumbent = evaluate(state['route_m'], state, forecast, self.geometry, work, self.method, 'incumbent')
            candidates.append(incumbent)
            if incumbent['feasible']:
                current_score = incumbent['score']['score_seconds']
            if proposal is None:
                if incumbent['feasible']:
                    candidates.extend(pool(state, forecast, self.geometry, work, self.method)[1:])
                    selected = best(candidates)
                    request = state_request(state, forecast, self.geometry, work)
                    plan = feedback_plan(request, selected) if work.sample_limit-work.samples >= 2 else None
                else:
                    request = state_request(state, forecast, self.geometry, work)
                    quota = request['limits']['max_samples']
                    plan = dict(request_id=request['request_id'], action='direct', waypoint_m=None,
                                sample_quotas=[quota], sampling='uniform', reason='immediate numerical repair') if quota else None
            else:
                request = state_request(state, forecast, self.geometry, work)
                plan = dict(deepcopy(proposal), request_id=request['request_id'])
                # Both the original and current quota must allow the identical intent.
                validate_search_plan(request, plan)
                self.log('reply_rebased', now, origin_request_id=origin,
                         rebased_request_id=request['request_id'], route_version=self.route_version,
                         position_m=state['position_m'], forecast_sha256=forecast.fingerprint)
            if plan is not None:
                result = execute(request, plan, state, forecast, self.geometry, work,
                                 self.method, self.seed+self.budget.transactions*1000)
                candidates.extend(result['candidates'])
                searches = result['searches']
                error = result.get('failure')
        except (BudgetExhausted, ValueError) as exc:
            error = str(exc)
        finally:
            self.budget.charge(work)
        valid = [c for c in candidates if c['feasible']]
        selected = best(valid) if valid else None
        changed = bool(selected and (current_score is None or selected['score']['score_seconds'] < current_score-1e-9))
        if changed:
            state = dict(state, route_m=deepcopy(selected['path']))
            self.route_version += 1
            feasible = True
            self.log('route_committed', now, source='reply' if proposal is not None else 'numeric',
                     origin_request_id=origin, route_version=self.route_version,
                     route_m=state['route_m'], previous_score=current_score,
                     selected_score=selected['score']['score_seconds'], forecast_sha256=forecast.fingerprint)
        self.log('tool_transaction', now, source='reply' if proposal is not None else 'numeric',
                 candidates=candidates, searches=searches, changed=changed, error=error,
                 resources=work.summary(), task_resources=self.budget.summary(),
                 prediction_work={k: forecast.counts[k]-before[k] for k in before},
                 wall_seconds=time.perf_counter()-started, physical_compute_time_modelled=False)
        return state, feasible

    def _cancel(self, now, reason):
        if self.pending is None:
            return
        p = self.pending
        try:
            cancelled = self.transport.cancel(p['token'])
            error = None
        except Exception as exc:
            cancelled, error = False, repr(exc)
        self.log('request_closed', now, consultation_id=p['id'], reason=reason,
                 cancellation_confirmed=cancelled, error=error)
        self.pending = None

    def _poll(self, state, forecast, feasible):
        now = forecast.now
        if self.transport is None:
            return state, feasible
        try:
            replies = self.transport.poll(now)
            if not isinstance(replies, list):
                raise ValueError('transport poll must return a list without waiting')
        except Exception as exc:
            self.log('transport_error', now, phase='poll', error=repr(exc))
            replies = []
        for reply in replies:
            token = reply.get('token') if isinstance(reply, dict) else None
            if self.pending is None or token != self.pending['token']:
                self.log('reply_ignored', now, token=token, reason='unknown_duplicate_or_closed_request')
                continue
            p = self.pending
            self.pending = None
            self.log('reply_received', now, consultation_id=p['id'], envelope=reply,
                     physical_latency_seconds=now-p['issued_at'])
            if now > p['deadline']:
                self.log('reply_rejected', now, consultation_id=p['id'], reason='deadline_exceeded')
                continue
            try:
                if reply.get('error'):
                    raise ValueError('transport worker error: '+str(reply['error']))
                plan = validate_search_plan(p['request'], reply.get('raw'))
                if plan['action'] == 'via':
                    delta = np.asarray(state['goal_m'])-state['position_m']
                    via = np.asarray(plan['waypoint_m'])
                    if ((via-state['position_m'])@delta <= 1e-8 or
                            np.linalg.norm(via-state['goal_m']) < 1e-8):
                        raise ValueError('waypoint_no_longer_forward')
                self.log('reply_parsed', now, consultation_id=p['id'], plan=plan)
                if plan['action'] != 'keep':
                    state, feasible = self._transaction(state, forecast, feasible, plan, p['request']['request_id'])
            except (ValueError, TypeError) as exc:
                self.log('reply_rejected', now, consultation_id=p['id'], reason=str(exc))
        if self.pending is not None and now > self.pending['deadline']:
            self._cancel(now, 'deadline_exceeded')
        return state, feasible

    def _dispatch(self, state, forecast, features):
        now = forecast.now
        if self.transport is None:
            return
        try:
            consult = self.gate(deepcopy(features))
            if type(consult) is not bool:
                raise ValueError('consultation gate must return a strict bool')
        except Exception as exc:
            self.log('gate_error', now, error=repr(exc))
            return
        if not consult:
            return
        reason = None
        if self.pending is not None:
            reason = 'request_in_flight'
        elif self.requests >= self.settings.max_requests:
            reason = 'consultation_budget_exhausted'
        elif not self.budget.summary()['transactions_left'] or not self.budget.summary()['edges_left']:
            reason = 'tool_budget_exhausted'
        elif self.last_consult is not None and now-self.last_consult < self.settings.minimum_interval_seconds:
            reason = 'minimum_consultation_interval'
        if reason:
            self.log('consultation_blocked', now, reason=reason)
            return
        try:
            if self.transport.busy():
                self.log('consultation_blocked', now, reason='transport_still_busy')
                return
            request = self._request(state, forecast)
            public = dict(request=request, consultation_id=f'consult_{self.requests:03d}',
                          issued_at_seconds=now, deadline_seconds=now+self.settings.request_ttl_seconds,
                          route_version=self.route_version, features=deepcopy(features),
                          prior_feedback=deepcopy([e for e in self.events if e['kind'] in
                              ('reply_rejected', 'route_committed', 'request_closed')][-8:]))
            # A failed dispatch still spends its reserved consultation slot; no retry loop.
            self.requests += 1
            self.last_consult = now
            token = self.transport.submit(deepcopy(public), now)
            if not isinstance(token, str) or not token or token in self.used_tokens:
                raise ValueError('transport must return a unique nonempty token')
            self.used_tokens.add(token)
            self.pending = dict(token=token, id=public['consultation_id'], issued_at=now,
                                deadline=public['deadline_seconds'], request=deepcopy(request))
            self.log('request_dispatched', now, public=public, token=token,
                     actual_model_call=None, meaning='dispatch is not proof of an actual LLM call')
        except Exception as exc:
            self.log('transport_error', now, phase='submit_or_busy', error=repr(exc))

    def tick(self, state, forecast, innovation_squared=None):
        state = deepcopy(state)
        now = self._validate(state, forecast, innovation_squared)
        start = len(self.events)
        s = self.settings
        power = forecast.history[-1]['interference_dbm']
        change = abs(power-statistics.median(self.powers[-3:])) if len(self.powers) >= 3 else 0.
        self.powers.append(power)
        if innovation_squared is not None:
            self.innovations.append(innovation_squared)
        self.last_now = now
        self.history = deepcopy(forecast.history)
        goal = bool(np.linalg.norm(np.asarray(state['position_m'])-state['goal_m']) <= 1e-8)
        stop = None
        feasible = False
        if (state['spent_length_m'] > state['maximum_length_m']+1e-8 or
                state['remaining_energy_wh'] < state['reserve_energy_wh']):
            stop = 'mission_resource_violation'
        elif goal:
            stop = 'goal_reached'
        else:
            try:
                feasible = self.budget.check_route(state, self.geometry)
            except BudgetExhausted as exc:
                stop = str(exc)
        age = None if self.last_numeric is None else now-self.last_numeric
        recent = self.innovations[-3:]
        mismatch = len(recent) == 3 and statistics.mean(recent) >= s.innovation_squared_mean
        reasons = []
        if not stop:
            if age is None: reasons.append('initial_review')
            if not feasible: reasons.append('route_infeasible')
            if change >= s.change_db: reasons.append('observed_power_change')
            if mismatch: reasons.append('predictive_mismatch')
            if age is not None and age >= s.review_age_seconds: reasons.append('review_expired')
        features = dict(time_seconds=now, reasons=reasons, power_change_db=change,
                        recent_innovation_mean=statistics.mean(recent) if recent else None,
                        numeric_review_age_seconds=age,
                        consultation_age_seconds=None if self.last_consult is None else now-self.last_consult,
                        route_feasible=feasible, goal_reached=goal,
                        remaining_energy_wh=state['remaining_energy_wh'],
                        remaining_length_m=state['maximum_length_m']-state['spent_length_m'],
                        task_budget=self.budget.summary(), consultations_left=s.max_requests-self.requests,
                        request_pending=self.pending is not None)
        self.log('observation_monitor', now, features=features, history_sha256=digest(self.history),
                 forecast_sha256=forecast.fingerprint, position_m=state['position_m'])
        if not stop:
            if reasons and (not feasible or age is None or age >= s.minimum_interval_seconds):
                self.last_numeric = now
                state, feasible = self._transaction(state, forecast, feasible)
            if not feasible:
                stop = 'no_feasible_numeric_route'
        if stop:
            self._cancel(now, stop)
            self.finished = True
            self.log('mission_stopped', now, reason=stop)
        else:
            state, feasible = self._poll(state, forecast, feasible)
            features.update(route_feasible=feasible, task_budget=self.budget.summary(),
                            consultations_left=s.max_requests-self.requests,
                            request_pending=self.pending is not None, route_version=self.route_version)
            self._dispatch(state, forecast, features)
        self.previous_state = deepcopy(state)
        return dict(state=deepcopy(state), stop=stop, route_version=self.route_version,
                    pending=self.pending is not None, consultations=self.requests,
                    resources=self.budget.summary(), events=deepcopy(self.events[start:]))
