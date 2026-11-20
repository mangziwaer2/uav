"""Frozen alternative fields -> arrival-time route scores and bounded RRT proposals.

No world, field truth, sensor, fitting, LLM backend or flight enters this module.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import json
import math
import numpy as np
from .equivalent_field import normalized_expected_power
from .equivalent_field import kernel, power_and_jacobian
from .event_inference import digest
from .planning import path_length
from .repair_search import BudgetExhausted, WorkBudget, repair_rrt_star


def finite(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('finite numeric scalar required')
    if positive and value <= 0:
        raise ValueError('positive scalar required')
    return float(value)


def canonical_route(route):
    points = np.asarray(route, float)
    if points.ndim != 2 or points.shape[1:] != (2,) or not len(points) or not np.all(np.isfinite(points)):
        raise ValueError('nonempty finite Nx2 route required')
    result = []
    for p in points:
        if result and np.linalg.norm(p-result[-1]) <= 1e-10:
            continue
        while len(result) >= 2:
            u, v = result[-1]-result[-2], p-result[-1]
            cross = u[0]*v[1]-u[1]*v[0]
            if abs(cross) <= 1e-12*np.linalg.norm(u)*np.linalg.norm(v) and u@v > 0:
                result.pop()
            else:
                break
        result.append(p)
    return np.asarray(result)


def arrival_queries(route, now, speed, delay=0.5, spacing=10.):
    now, speed = finite(now), finite(speed, True)
    delay, spacing = finite(delay), finite(spacing, True)
    if now < 0 or delay < 0:
        raise ValueError('nonnegative time/delay required')
    route = canonical_route(route)
    xy, tt, dt, elapsed = [], [], [], 0.
    for a, b in zip(route, route[1:]):
        length = float(np.linalg.norm(b-a))
        count = max(1, math.ceil(length/spacing)); duration = length/speed
        for i in range(count):
            fraction = (i+.5)/count
            xy.append((a+fraction*(b-a)).tolist())
            tt.append(now+delay+elapsed+fraction*duration)
            dt.append(duration/count)
        elapsed += duration
    return dict(position_m=xy, time_seconds=tt, dt_seconds=dt, duration_seconds=elapsed)


class ScenarioForecast:
    """Immutable copy of a causal bank snapshot; quality is never route-dependent."""
    def __init__(self, history, models, now, gp=None, gp_version=None, joint_settings=None):
        self.history, self.models = deepcopy(history), deepcopy(models)
        self.now = finite(now)
        if (not self.history or self.history[-1]['time_seconds'] != self.now
                or any(p['time_seconds'] > self.now for p in self.history)
                or any(a['time_seconds'] >= b['time_seconds'] for a,b in zip(self.history,self.history[1:]))):
            raise ValueError('history must end at the current time')
        if not self.models or len({m['id'] for m in self.models}) != len(self.models):
            raise ValueError('nonempty distinct scenario IDs required')
        weights = [finite(m['weight']) for m in self.models]
        if any(w < 0 for w in weights) or not math.isclose(sum(weights), 1., abs_tol=1e-10):
            raise ValueError('scenario qualities must be normalized, no silent renormalization')
        for m in self.models:
            if (finite(m['std_db']) < 0 or type(m['born_index']) is not int
                    or not 0 <= m['born_index'] < len(history)
                    or m['family'] not in ('persistence','unknown_prior','static','translate','strength_change')):
                raise ValueError('invalid scenario uncertainty/birth')
            if m['family'] != 'persistence' and (m['reference_time_seconds'] > now or len(m['bases']) != len(m['peaks_dbm'])):
                raise ValueError('future/misaligned field model')
            if m['family'] == 'unknown_prior' and (m['bases'] or m['peaks_dbm']):
                raise ValueError('unknown placeholder cannot contain a spatial source')
        self.weights = np.asarray(weights)
        self.weights.flags.writeable = False
        self.gp = gp
        if gp is not None and (not isinstance(gp_version,str) or not gp_version):
            raise ValueError('a frozen GP version fingerprint is required')
        self.gp_version = gp_version
        self.joint_settings = joint_settings
        self.fingerprint = digest(self.identity())
        self.counts = dict(receiver_queries=0, scenario_receiver_terms=0, basis_receiver_terms=0, gp_receiver_queries=0)

    def identity(self):
        value = dict(history=self.history, models=self.models, now=self.now, gp_version=self.gp_version)
        if self.joint_settings is not None:
            value['joint_settings'] = asdict(self.joint_settings)
        return value

    def predict(self, xy, times, method='mixture'):
        if digest(self.identity()) != self.fingerprint:
            raise ValueError('frozen forecast was mutated')
        xy, times = np.asarray(xy, float), np.asarray(times, float)
        if xy.shape != (len(times), 2) or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(times)) or np.any(times < self.now):
            raise ValueError('finite current/future Nx2 queries required')
        self.counts['receiver_queries'] += len(times)
        if method == 'gp':
            if self.gp is None:
                raise ValueError('no GP provider; do not invent a GP reference')
            mu, sd, support = self.gp.predict(xy, times)
            mu, sd, support = np.asarray(mu), np.asarray(sd), np.asarray(support, bool)
            if mu.shape != times.shape or sd.shape != times.shape or support.shape != times.shape:
                raise ValueError('GP provider output shape mismatch')
            self.counts['gp_receiver_queries'] += len(times)
            means = np.where(support, mu, -100.)[None, :]
            stds = np.where(support, sd, 15.)[None, :]
            return dict(ids=['gp_with_declared_unknown_prior'], means=means, stds=stds,
                        weights=np.ones(1), supported=support)
        if method not in ('mixture', 'persistence', 'joint'):
            raise ValueError('unknown forecast method')
        models = self.models if method in ('mixture', 'joint') else [dict(id='power_persistence', family='persistence', std_db=4.)]
        means = []
        for m in models:
            if m['family'] == 'persistence':
                mu = np.full(len(times), self.history[-1]['interference_dbm'])
            else:
                mu = power_and_jacobian(kernel(m['bases'], xy, times, m['reference_time_seconds']), m['peaks_dbm'])[0]
                self.counts['basis_receiver_terms'] += len(m['bases'])*len(times)
            means.append(mu)
        self.counts['scenario_receiver_terms'] += len(models)*len(times)
        result = dict(ids=[m['id'] for m in models], means=np.asarray(means),
                    stds=np.asarray([np.full(len(times), m['std_db']) for m in models]),
                    weights=self.weights.copy() if method in ('mixture', 'joint') else np.ones(1), supported=None)
        if method == 'joint':
            from .joint_prediction import blend_prediction
            if self.gp is None or self.joint_settings is None:
                raise ValueError('joint forecast requires frozen GP and interpolation settings')
            self.counts['gp_receiver_queries'] += len(times)
            result = blend_prediction(result, self.gp, xy, times, self.joint_settings)
        return result


def score_route(forecast, route, speed, delay=.5, method='mixture', frozen_time=False):
    q = arrival_queries(route, forecast.now, speed, delay)
    if not q['position_m']:
        raise ValueError('route requires positive flight duration')
    times = np.full(len(q['time_seconds']), forecast.now) if frozen_time else np.asarray(q['time_seconds'])
    d = forecast.predict(q['position_m'], times, method)
    powers = (d['expected_normalized_power'] if 'expected_normalized_power' in d
              else normalized_expected_power(d['means'], d['stds'], -80.))
    per_q = powers @ np.asarray(q['dt_seconds'])
    mean_q = float(d['weights'] @ per_q)
    sd_q = math.sqrt(float(d['weights'] @ (per_q-mean_q)**2))
    result = dict(score_seconds=q['duration_seconds']+mean_q, remaining_flight_seconds=q['duration_seconds'],
        power_proxy_seconds=mean_q, scenario_q_std_seconds=sd_q,
        scenario_scores=[dict(id=i, weight=float(w), power_proxy_seconds=float(v),
                              score_seconds=q['duration_seconds']+float(v)) for i,w,v in zip(d['ids'],d['weights'],per_q)],
        queries=q, scenario_mean_dbm=d['means'].tolist(), scenario_std_db=d['stds'].tolist(),
        query_time_seconds=times.tolist(), method=method, arrival_field_predicted=not frozen_time,
        uncertainty_calibrated=False, hard_power_threshold=None, forecast_sha256=forecast.fingerprint)
    if d['supported'] is not None:
        result['supported_fraction'] = float(np.asarray(q['dt_seconds'])[d['supported']].sum()/q['duration_seconds'])
        result['unsupported_uses_declared_prior'] = method == 'gp'
    if method == 'joint':
        result['gp_weight'] = d['gp_weight'].tolist()
        result['expected_normalized_power'] = powers.tolist()
        result['unsupported_uses_source_or_unknown'] = True
    return result


def state_request(state, forecast, geometry, work):
    """State must contain only public/current information, supplied by the host."""
    required={'position_m','goal_m','route_m','speed_mps','spent_length_m','maximum_length_m',
              'remaining_energy_wh','flight_power_w','hover_power_w','reserve_energy_wh','delay_seconds'}
    if set(state) != required:
        raise ValueError('public state fields must match the planning contract')
    for k in required-{'position_m','goal_m','route_m'}:
        if finite(state[k]) < 0: raise ValueError('negative vehicle resource')
    finite(state['speed_mps'],True)
    if work.sample_limit>700 or work.edge_limit>10000 or work.reserved_edges:
        raise ValueError('planning requires a dedicated bounded shared ledger')
    for p in (state['position_m'],state['goal_m']):
        if len(p)!=2 or not all(math.isfinite(finite(v)) for v in p):raise ValueError('finite public endpoint required')
    body = dict(state=deepcopy(state), forecast_sha256=forecast.fingerprint,
                models=deepcopy(forecast.models), history=deepcopy(forecast.history),
                time_seconds=forecast.now,
                bounds_m=[[0.,0.], geometry.high.tolist()], resources=work.summary(),
                known_geometry=dict(obstacles_m=geometry.config.obstacles_m,clearance_m=geometry.clearance_m,
                    blocking_buildings=[dict(id=b.id,footprint_m=b.footprint_m) for b in geometry.blocking_buildings]),
                limits=dict(max_samples=min(700, work.sample_limit-work.samples),
                            actions=['keep','direct','via'], samplers=['uniform','corridor']))
    body['request_id'] = digest(body)
    return body


def validate_search_plan(request, raw):
    def fail(): raise ValueError('invalid/stale search plan')
    try:
        if isinstance(raw, str):
            def pairs(items):
                d={}
                for k,v in items:
                    if k in d: fail()
                    d[k]=v
                return d
            raw=json.loads(raw, object_pairs_hook=pairs)
        p=deepcopy(raw)
        if set(p) != {'request_id','action','waypoint_m','sample_quotas','sampling','reason'}: fail()
        if p['request_id'] != request['request_id'] or p['action'] not in ('keep','direct','via'): fail()
        if not isinstance(p['reason'],str) or not 1 <= len(p['reason']) <= 1000: fail()
        if p['sampling'] not in ('uniform','corridor') or not isinstance(p['sample_quotas'],list): fail()
        expected={'keep':0,'direct':1,'via':2}[p['action']]
        if len(p['sample_quotas']) != expected or any(type(n) is not int or n < 1 for n in p['sample_quotas']): fail()
        if sum(p['sample_quotas']) > request['limits']['max_samples']: fail()
        if p['action']=='via':
            point=p['waypoint_m']
            if not isinstance(point,list) or len(point)!=2: fail()
            point=np.asarray([finite(v) for v in point])
            if np.any(point < 0) or np.any(point > request['bounds_m'][1]): fail()
        elif p['waypoint_m'] is not None: fail()
        return p
    except (KeyError,TypeError,json.JSONDecodeError,OverflowError) as exc:
        raise ValueError('invalid/stale search plan') from exc


class FrozenSearchObjective:
    """Static spatial proposal proxy; future-arrival scoring is separate."""
    def __init__(self, geometry, forecast, speed, work, method):
        self.geometry,self.forecast,self.speed,self.budget,self.method=geometry,forecast,speed,work,method
        self.cache={}

    def edge_cost(self, start, end):
        key=tuple(sorted((tuple(map(float,start)),tuple(map(float,end)))))
        if key in self.cache:
            self.budget.cache_hits+=1;return self.cache[key]
        self.budget.take_edge()
        if not self.geometry.edge_free(start,end): value=math.inf
        elif np.linalg.norm(np.asarray(start)-end) < 1e-10: value=0.
        else: value=score_route(self.forecast,[start,end],self.speed,delay=0.,method=self.method,frozen_time=True)['score_seconds']
        self.cache[key]=value
        return value


def propose_and_accept(request, raw, live_state, forecast, geometry, work, seed=0, method='mixture'):
    if state_request(live_state,forecast,geometry,work)['request_id'] != request['request_id']:
        raise ValueError('state changed while waiting; reject before spending resources')
    p=validate_search_plan(request,raw)
    s=live_state; original=deepcopy(s['route_m']);position=s['position_m'];goal=s['goal_m'];speed=s['speed_mps']
    delay=s.get('delay_seconds',.5)
    def feasible(route):
        if (len(route)<2 or not np.allclose(route[0],position,atol=1e-8,rtol=0)
                or not np.allclose(route[-1],goal,atol=1e-8,rtol=0)):return False
        length=path_length(route)
        if (s['spent_length_m']+length>s['maximum_length_m']+1e-8
                or s['remaining_energy_wh']-(length/speed*s['flight_power_w']+delay*s['hover_power_w'])/3600<s['reserve_energy_wh']):return False
        for a,b in zip(route,route[1:]):
            work.take_edge();work.verification_edges+=1
            if not geometry.edge_free(a,b):return False
        return True
    if not feasible(original):raise ValueError('a feasible incumbent is required')
    base=score_route(forecast,original,speed,delay,method)
    result=dict(path=original,accepted=False,reason='keep',before=base,candidate=None,searches=[],plan=p)
    if p['action']=='keep':
        result['resources']=work.summary();return result
    endpoints=[position]+([p['waypoint_m']] if p['action']=='via' else [])+[goal]
    candidate=[];obj=FrozenSearchObjective(geometry,forecast,speed,work,method)
    try:
        # Reserve enough final geometry checks for a path at most two legs,
        # with at most one new tree node per sample and both end connectors.
        reserve=sum(p['sample_quotas'])+2*len(p['sample_quotas'])
        if work.edge_limit-work.edges < reserve:raise BudgetExhausted('no_candidate_verification_reserve')
        work.reserved_edges=reserve
        for leg,(a,b,quota) in enumerate(zip(endpoints,endpoints[1:],p['sample_quotas'])):
            if p['sampling']=='uniform':sampling=dict(focus_probability=0.,regions=[])
            else:
                lo=np.maximum(0.,np.minimum(a,b)-40.);hi=np.minimum(geometry.high,np.maximum(a,b)+40.)
                sampling=dict(focus_probability=.6,regions=[dict(low_m=lo.tolist(),high_m=hi.tolist(),weight=1.)])
            solved,stats=repair_rrt_star(obj,a,b,quota,seed+leg,sampling)
            result['searches'].append(dict(leg=leg,seed=seed+leg,sampling=sampling,sampling_counts=stats,**asdict(solved)))
            if not solved.path:
                result['reason']='search_failed';break
            candidate.extend(solved.path if not candidate else solved.path[1:])
        else:
            work.reserved_edges=0
            if feasible(candidate):
                score=score_route(forecast,candidate,speed,delay,method);result['candidate']=dict(path=candidate,score=score)
                if score['score_seconds'] < base['score_seconds']-1e-9:
                    result.update(path=candidate,accepted=True,reason='lower_full_arrival_cost')
                else:result['reason']='no_full_arrival_improvement'
            else:result['reason']='geometry_length_or_energy_rejected'
    except BudgetExhausted as exc:result['reason']=str(exc)
    finally:work.reserved_edges=0
    result['resources']=work.summary()
    return result


class DispatchReceipt:
    """Host lifecycle primitive. Creating a file alone never counts a model call."""
    def __init__(self, request_id, issued_monotonic):
        self.request_id=request_id;self.issued=finite(issued_monotonic);self.dispatched=None
        self.agent_id=None;self.raw=None;self.received=None;self.status='issued';self.executed=False

    def dispatch(self, agent_id, now):
        now=finite(now)
        if self.status!='issued' or not isinstance(agent_id,str) or not agent_id or not self.issued<=now<=self.issued+60:
            raise ValueError('dispatch missing, duplicate or late')
        self.dispatched=now;self.agent_id=agent_id;self.status='dispatched'

    def receive(self, raw, now):
        now=finite(now)
        if self.status!='dispatched' or not isinstance(raw,str) or now<self.dispatched:
            raise ValueError('raw requires an actual preceding dispatch')
        self.raw=raw;self.received=now
        self.status='raw_received' if now<=self.dispatched+180 else 'late_raw'

    def expire(self, now):
        now=finite(now)
        deadline=self.issued+60 if self.dispatched is None else self.dispatched+180
        if now<=deadline or self.status not in ('issued','dispatched'):raise ValueError('not waiting past deadline')
        self.status='undispatched_timeout' if self.dispatched is None else 'response_timeout'

    def execute(self, request, live_request):
        if self.status!='raw_received':raise ValueError('no timely raw response')
        if self.request_id!=request['request_id'] or live_request['request_id']!=self.request_id:
            self.status='stale_rejected';raise ValueError('stale request')
        try:plan=validate_search_plan(request,self.raw)
        except ValueError:
            self.status='parse_rejected';raise
        self.status='executed';self.executed=True;return plan

    def snapshot(self):
        return dict(request_id=self.request_id,issued_monotonic=self.issued,dispatched_monotonic=self.dispatched,
            agent_id=self.agent_id,raw=self.raw,received_monotonic=self.received,status=self.status,
            actual_model_call=self.dispatched is not None,executed=self.executed)
