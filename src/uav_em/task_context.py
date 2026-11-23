"""Public task snapshots shared by source inference and path organizations.

Only the declared finite geometric pool is covered. Building a snapshot spends
the caller's proposal/edge ledger; saved per-world scores need no extra queries.
There is no simulator, sensor, fitting, advisor or experiment I/O here.
"""
from copy import deepcopy
import json
import math

import numpy as np

from .event_inference import FAMILIES, digest
from .feasible_scenario import best, pool, search_waypoints
from .planning import path_length
from .scenario_planning import canonical_route, finite, score_route, state_request


VERSION = 'task-context-v1'
COMPOUND_VERSION = 'task-context-compound-v1'
METHODS = ('joint', 'mixture')
COUNTERS = ('receiver_queries', 'scenario_receiver_terms', 'basis_receiver_terms', 'gp_receiver_queries')
STATE_FIELDS = {'position_m', 'goal_m', 'route_m', 'speed_mps', 'spent_length_m',
    'maximum_length_m', 'remaining_energy_wh', 'flight_power_w', 'hover_power_w',
    'reserve_energy_wh', 'delay_seconds'}
RESOURCE_FIELDS = {'sample_limit', 'edge_limit', 'samples', 'edges', 'cache_hits',
    'reserved_edges', 'verification_edges', 'samples_left', 'edges_left', 'total_edges_left'}
SCORE_FIELDS = {'score_seconds', 'remaining_flight_seconds', 'power_proxy_seconds',
    'scenario_q_std_seconds', 'scenario_scores', 'queries', 'scenario_mean_dbm',
    'scenario_std_db', 'query_time_seconds', 'method', 'arrival_field_predicted',
    'uncertainty_calibrated', 'hard_power_threshold', 'forecast_sha256'}
JOINT_FIELDS = {'supported_fraction', 'unsupported_uses_declared_prior', 'gp_weight',
    'expected_normalized_power', 'unsupported_uses_source_or_unknown'}
LIMITATIONS = dict(
    coverage='only the declared finite pool; no global route or hypothesis identifiability guarantee',
    evidence='model weights are relative to this pool, not calibrated model adequacy or safety',
    extrapolation='future source dynamics and unsupported regions remain hypotheses, not measurements',
    gp='GP support and derived soft points are not new independent evidence or safe regions',
    action='whole-route predicted T+Q contrast; no first-15-m distance gate or multiplier',
    unknown='unknown_prior is a declared uncertain -100 dBm floor, never last-power continuation',
    fitting='fit_mse uses fitting history; only scored_points denotes post-birth prediction checks')
CONTEXT_FIELDS = {'version', 'task_snapshot_id', 'time_seconds', 'history_sha256',
    'models_sha256', 'forecast_sha256', 'state', 'state_sha256', 'bounds_m',
    'known_geometry', 'geometry_sha256', 'planner_method', 'candidates',
    'common_best_id', 'near_competitor', 'world_actions', 'family_priorities',
    'model_adequacy', 'common_pool', 'resources_before', 'resources_after',
    'receiver_work', 'limitations'}


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _keys(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError('task context fields must match the public allowlist')


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('task context SHA256 required')


def _number(value, *, nonnegative=False):
    result = finite(value)
    if nonnegative and result < 0:
        raise ValueError('nonnegative task context scalar required')
    return result


def _array(value, shape):
    if not isinstance(value, list):
        raise ValueError('task context numeric arrays must be lists')
    def numeric(v):
        if isinstance(v, list):
            for item in v: numeric(item)
        else: _number(v)
    numeric(value)
    result = np.asarray(value, float)
    if result.shape != tuple(shape):
        raise ValueError('task context array shape mismatch')
    return result


def _path(value):
    if not isinstance(value, list) or not value:
        raise ValueError('nonempty task context path required')
    return _array(value, (len(value), 2))


def _geometry_summary(geometry):
    return _json_copy(dict(
        bounds_m=[[0., 0.], geometry.high.tolist()],
        known_geometry=dict(obstacles_m=geometry.config.obstacles_m,
            clearance_m=geometry.clearance_m,
            blocking_buildings=[dict(id=b.id, footprint_m=b.footprint_m, holes_m=b.holes_m)
                                for b in geometry.blocking_buildings])))


def _check_forecast(forecast, method, allow_composite=False):
    if method not in METHODS:
        raise ValueError('task source context requires joint or mixture prediction')
    if digest(forecast.identity()) != forecast.fingerprint:
        raise ValueError('frozen forecast was mutated')
    if any(m['family'] not in (*FAMILIES, 'unknown_prior', *(('composite',) if allow_composite else ())) for m in forecast.models):
        raise ValueError('power persistence cannot enter the task source pool')


def _same_geometry(a, b):
    a, b = canonical_route(a), canonical_route(b)
    return a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-8)


def _world_score(candidate, identifier):
    return next(s['score_seconds'] for s in candidate['score']['scenario_scores'] if s['id'] == identifier)


def _task_contrasts(rows, models):
    feasible = [r for r in rows if r['feasible']]
    selected = best(feasible) if feasible else None
    distinct = ([r for r in feasible if not _same_geometry(r['path'], selected['path'])]
                if selected else [])
    competitor = (min(distinct, key=lambda r: (r['score']['score_seconds']-selected['score']['score_seconds'],
                                              path_length(r['path']), r['id']))
                  if distinct else None)
    near = None
    if competitor:
        near = dict(route_a=selected['id'], route_b=competitor['id'],
            delta_c_seconds=competitor['score']['score_seconds']-selected['score']['score_seconds'],
            per_world_delta_c_seconds={m['id']: _world_score(competitor, m['id'])-_world_score(selected, m['id'])
                                       for m in models})
    actions = []
    priorities = {family: 0. for family in FAMILIES}
    for m in models:
        individual = (min(feasible, key=lambda r: (_world_score(r, m['id']), path_length(r['path']), r['id']))
                      if feasible else None)
        regret = (max(0., _world_score(selected, m['id'])-_world_score(individual, m['id']))
                  if selected else 0.)
        delta = near['per_world_delta_c_seconds'][m['id']] if near else None
        weighted_regret = m['weight']*regret
        # Favor world-dependent decisions and close nominal competitors. This
        # is a transparent allocation heuristic, not a calibrated value of information.
        priority = weighted_regret + (m['weight']*abs(delta)/(1.+max(0., near['delta_c_seconds']))
                                      if near else 0.)
        actions.append(dict(id=m['id'], family=m['family'], weight=m['weight'],
            selected_id=individual['id'] if individual else None,
            common_best_world_score_seconds=_world_score(selected, m['id']) if selected else None,
            selected_world_score_seconds=_world_score(individual, m['id']) if individual else None,
            weighted_regret_seconds=weighted_regret, near_competitor_delta_c_seconds=delta,
            allocation_priority=priority))
        if m['family'] in priorities: priorities[m['family']] += priority
    return dict(common_best_id=selected['id'] if selected else None, near_competitor=near,
                world_actions=actions, family_priorities=priorities)


def _record_interrupted_waypoint(extra, state):
    """Recover the already charged final proposal omitted by the shared tool.

    search_waypoints exits while evaluating its last path on edge exhaustion.
    This adds its deterministic public proposal record, without retrying work.
    """
    if extra['proposals'] == len(extra['candidates']):
        return
    if (extra['stop_reason'] != 'edge_budget_exhausted'
            or extra['proposals'] != len(extra['candidates'])+1):
        raise ValueError('inconsistent interrupted finite waypoint search')
    index = extra['proposals']-1
    fraction = (.25, .5, .75, .1, .9)[index//10]
    offset = (20., -20., 40., -40., 80., -80., 120., -120., 160., -160.)[index % 10]
    position = np.asarray(state['position_m'], float)
    delta = np.asarray(state['goal_m'], float)-position
    forward = delta/np.linalg.norm(delta)
    waypoint = position+fraction*delta+offset*np.array([-forward[1], forward[0]])
    extra['candidates'].append(dict(id=f'lattice_{index+1:02d}',
        path=[position.tolist(), waypoint.tolist(), deepcopy(state['goal_m'])],
        feasible=False, reason='edge_budget_exhausted',
        proposal=dict(kind='deterministic_lateral_waypoint', fraction=fraction, offset_m=offset)))


def build_task_context(state, forecast, geometry, work, method='joint', *, allow_composite=False):
    """Pay for one finite shared pool and freeze its public causal task context."""
    _check_forecast(forecast, method, allow_composite)
    request = state_request(state, forecast, geometry, work)
    before = work.summary()
    counts_before = deepcopy(forecast.counts)
    rows = pool(state, forecast, geometry, work, method)
    extra = search_waypoints(state, forecast, geometry, work, method, max_proposals=50)
    _record_interrupted_waypoint(extra, state)
    rows.extend(extra['candidates'])
    models = [dict(id=m['id'], family=m['family'], weight=m['weight'], born_index=m['born_index'],
                   scored_points=m.get('scored_points', 0), fit_mse=m.get('fit_mse'))
              for m in forecast.models]
    public_geometry = _geometry_summary(geometry)
    value = _json_copy(dict(version=COMPOUND_VERSION if allow_composite else VERSION, time_seconds=forecast.now,
        history_sha256=digest(forecast.history), models_sha256=digest(forecast.models),
        forecast_sha256=forecast.fingerprint, state=request['state'], state_sha256=digest(request['state']),
        **public_geometry, geometry_sha256=digest(public_geometry), planner_method=method,
        candidates=rows, **_task_contrasts(rows, models), model_adequacy=models,
        common_pool=dict(base='incumbent_direct_and_legal_bearings',
            additional='deterministic_lateral_waypoints', max_proposals=50,
            proposals=extra['proposals'], stop_reason=extra['stop_reason']),
        resources_before=before, resources_after=work.summary(),
        receiver_work={key: forecast.counts[key]-counts_before[key] for key in COUNTERS},
        limitations=LIMITATIONS))
    value['task_snapshot_id'] = digest(value)
    validate_task_context(value, state=state, forecast=forecast, geometry=geometry)
    return value


def _validate_resources(value):
    _keys(value, RESOURCE_FIELDS)
    if any(type(v) is not int or v < 0 for v in value.values()):
        raise ValueError('nonnegative integer task resources required')
    if not 1 <= value['sample_limit'] <= 700 or not 1 <= value['edge_limit'] <= 10000:
        raise ValueError('task context exceeds shared planning resource caps')
    if (value['reserved_edges'] != 0 or value['samples'] > value['sample_limit']
            or value['edges'] > value['edge_limit'] or value['verification_edges'] > value['edges']
            or value['samples_left'] != value['sample_limit']-value['samples']
            or value['edges_left'] != value['edge_limit']-value['edges']
            or value['total_edges_left'] != value['edges_left']):
        raise ValueError('inconsistent task resource ledger')


def _validate_score(score, context):
    method = context['planner_method']
    _keys(score, SCORE_FIELDS | (JOINT_FIELDS if method == 'joint' else set()))
    if (score['method'] != method or score['forecast_sha256'] != context['forecast_sha256']
            or score['arrival_field_predicted'] is not True or score['uncertainty_calibrated'] is not False
            or score['hard_power_threshold'] is not None):
        raise ValueError('task score must use the frozen future field without a hard power line')
    for key in ('score_seconds', 'remaining_flight_seconds', 'power_proxy_seconds', 'scenario_q_std_seconds'):
        _number(score[key], nonnegative=True)
    q = score['queries']
    _keys(q, {'position_m', 'time_seconds', 'dt_seconds', 'duration_seconds'})
    n = len(q['position_m']); world_count = len(context['model_adequacy'])
    if not n: raise ValueError('task score requires future arrival samples')
    _array(q['position_m'], (n, 2))
    tt = _array(q['time_seconds'], (n,)); dt = _array(q['dt_seconds'], (n,))
    actual = _array(score['query_time_seconds'], (n,))
    _number(q['duration_seconds'], nonnegative=True)
    if (np.any(tt <= context['time_seconds']) or np.any(np.diff(tt) <= 0)
            or np.any(dt <= 0) or not np.array_equal(tt, actual)
            or not math.isclose(float(dt.sum()), q['duration_seconds'], rel_tol=1e-10, abs_tol=1e-8)
            or q['duration_seconds'] != score['remaining_flight_seconds']
            or not math.isclose(score['score_seconds'], score['remaining_flight_seconds']+score['power_proxy_seconds'], rel_tol=1e-10, abs_tol=1e-8)):
        raise ValueError('inconsistent future task arrival quadrature')
    _array(score['scenario_mean_dbm'], (world_count, n))
    sd = _array(score['scenario_std_db'], (world_count, n))
    if np.any(sd < 0): raise ValueError('negative task prediction uncertainty')
    per_world = score['scenario_scores']
    if not isinstance(per_world, list) or len(per_world) != world_count:
        raise ValueError('task score worlds mismatch')
    for item, model in zip(per_world, context['model_adequacy']):
        _keys(item, {'id', 'weight', 'power_proxy_seconds', 'score_seconds'})
        if item['id'] != model['id'] or item['weight'] != model['weight']:
            raise ValueError('task score world identity mismatch')
        _number(item['power_proxy_seconds'], nonnegative=True); _number(item['score_seconds'], nonnegative=True)
        if not math.isclose(item['score_seconds'], q['duration_seconds']+item['power_proxy_seconds'], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError('task world score arithmetic mismatch')
    if not math.isclose(sum(s['weight']*s['power_proxy_seconds'] for s in per_world), score['power_proxy_seconds'], rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError('task aggregate must use shared relative weights')
    if method == 'joint':
        g = _array(score['gp_weight'], (n,))
        powers = _array(score['expected_normalized_power'], (world_count, n))
        if (np.any(g < 0) or np.any(g > 1) or np.any(powers < 0)
                or not 0 <= _number(score['supported_fraction']) <= 1
                or score['unsupported_uses_declared_prior'] is not False
                or score['unsupported_uses_source_or_unknown'] is not True):
            raise ValueError('invalid task joint interpolation summary')


def validate_task_context(context, *, history=None, models=None, state=None, forecast=None, geometry=None):
    """Reject injected fields, content changes and optional stale live inputs.

    Validation is algebra over the saved packet and never spends query budget.
    When supplied, a live forecast is checked by recomputing its identity hash.
    """
    try:
        _keys(context, CONTEXT_FIELDS)
        for key in ('task_snapshot_id', 'history_sha256', 'models_sha256', 'forecast_sha256', 'state_sha256', 'geometry_sha256'):
            _sha(context[key])
        if context['version'] not in (VERSION, COMPOUND_VERSION) or context['planner_method'] not in METHODS:
            raise ValueError('unsupported task context version/method')
        if context['task_snapshot_id'] != digest({k: v for k, v in context.items() if k != 'task_snapshot_id'}):
            raise ValueError('task context content changed')
        _number(context['time_seconds'], nonnegative=True)
        _keys(context['state'], STATE_FIELDS)
        for key in ('position_m', 'goal_m'): _array(context['state'][key], (2,))
        _path(context['state']['route_m'])
        for key in STATE_FIELDS-{'position_m', 'goal_m', 'route_m'}:
            _number(context['state'][key], nonnegative=True)
        if context['state']['speed_mps'] <= 0 or digest(context['state']) != context['state_sha256']:
            raise ValueError('task state binding mismatch')
        bounds = _array(context['bounds_m'], (2, 2))
        known = context['known_geometry']
        _keys(known, {'obstacles_m', 'clearance_m', 'blocking_buildings'})
        _number(known['clearance_m'], nonnegative=True)
        if np.any(bounds[0] != 0) or np.any(bounds[1] <= 0): raise ValueError('invalid task bounds')
        if not isinstance(known['obstacles_m'], list) or not isinstance(known['blocking_buildings'], list):
            raise ValueError('task geometry must contain public lists')
        for obstacle in known['obstacles_m']: _array(obstacle, (4,))
        for building in known['blocking_buildings']:
            _keys(building, {'id', 'footprint_m', 'holes_m'})
            if not isinstance(building['id'], str) or not building['id'] or not isinstance(building['holes_m'], list):
                raise ValueError('invalid public task building')
            for ring in [building['footprint_m'], *building['holes_m']]:
                if len(ring) < 3: raise ValueError('complete public building rings required')
                _path(ring)
        geo_identity = dict(bounds_m=context['bounds_m'], known_geometry=known)
        if digest(geo_identity) != context['geometry_sha256']: raise ValueError('task geometry binding mismatch')
        adequacy = context['model_adequacy']
        if not isinstance(adequacy, list) or not adequacy: raise ValueError('task worlds required')
        for model in adequacy:
            _keys(model, {'id', 'family', 'weight', 'born_index', 'scored_points', 'fit_mse'})
            if (not isinstance(model['id'], str) or not model['id'] or model['family'] not in (
                    *FAMILIES, 'unknown_prior', *(('composite',) if context['version'] == COMPOUND_VERSION else ()))
                    or type(model['born_index']) is not int or model['born_index'] < 0
                    or type(model['scored_points']) is not int or model['scored_points'] < 0):
                raise ValueError('invalid task model evidence summary')
            _number(model['weight'], nonnegative=True)
            if model['fit_mse'] is not None: _number(model['fit_mse'], nonnegative=True)
        if (len({m['id'] for m in adequacy}) != len(adequacy)
                or not math.isclose(sum(m['weight'] for m in adequacy), 1., abs_tol=1e-10)):
            raise ValueError('task world weights/IDs mismatch')
        if not isinstance(context['candidates'], list) or not context['candidates']:
            raise ValueError('task pool requires at least the incumbent record')
        for candidate in context['candidates']:
            if not isinstance(candidate, dict): raise ValueError('task candidate must be a public object')
            expected = {'id', 'path', 'feasible', 'reason'} | ({'score'} if candidate.get('feasible') is True else set())
            if 'proposal' in candidate: expected.add('proposal')
            _keys(candidate, expected)
            if not isinstance(candidate['id'], str) or not candidate['id'] or type(candidate['feasible']) is not bool:
                raise ValueError('invalid task candidate identity')
            _path(candidate['path'])
            if candidate['feasible']:
                if candidate['reason'] is not None: raise ValueError('feasible task candidate cannot have failure reason')
                _validate_score(candidate['score'], context)
            elif candidate['reason'] not in ('length_energy_endpoints', 'geometry', 'edge_budget_exhausted', 'waypoint_outside_map'):
                raise ValueError('invalid task candidate failure reason')
            if 'proposal' in candidate:
                _keys(candidate['proposal'], {'kind', 'fraction', 'offset_m'})
                if (candidate['proposal']['kind'] != 'deterministic_lateral_waypoint'
                        or candidate['proposal']['fraction'] not in (.25, .5, .75, .1, .9)
                        or candidate['proposal']['offset_m'] not in (20., -20., 40., -40., 80., -80., 120., -120., 160., -160.)):
                    raise ValueError('task waypoint outside declared finite pool')
        if len({r['id'] for r in context['candidates']}) != len(context['candidates']):
            raise ValueError('duplicate task candidate IDs')
        _keys(context['family_priorities'], FAMILIES)
        for priority in context['family_priorities'].values(): _number(priority, nonnegative=True)
        if not isinstance(context['world_actions'], list): raise ValueError('task world actions must be a list')
        for action in context['world_actions']:
            _keys(action, {'id', 'family', 'weight', 'selected_id', 'common_best_world_score_seconds',
                'selected_world_score_seconds', 'weighted_regret_seconds', 'near_competitor_delta_c_seconds', 'allocation_priority'})
            for key in ('weight', 'weighted_regret_seconds', 'allocation_priority'): _number(action[key], nonnegative=True)
            for key in ('common_best_world_score_seconds', 'selected_world_score_seconds'):
                if action[key] is not None: _number(action[key], nonnegative=True)
            if action['near_competitor_delta_c_seconds'] is not None: _number(action['near_competitor_delta_c_seconds'])
        if context['near_competitor'] is not None:
            near = context['near_competitor']
            _keys(near, {'route_a', 'route_b', 'delta_c_seconds', 'per_world_delta_c_seconds'})
            _number(near['delta_c_seconds'])
            _keys(near['per_world_delta_c_seconds'], [m['id'] for m in adequacy])
            for value in near['per_world_delta_c_seconds'].values(): _number(value)
        for key, value in _task_contrasts(context['candidates'], adequacy).items():
            if context[key] != value: raise ValueError('task contrast summary mismatch')
        _keys(context['common_pool'], {'base', 'additional', 'max_proposals', 'proposals', 'stop_reason'})
        common = context['common_pool']
        if (common['base'] != 'incumbent_direct_and_legal_bearings'
                or common['additional'] != 'deterministic_lateral_waypoints' or common['max_proposals'] != 50
                or type(common['proposals']) is not int or not 0 <= common['proposals'] <= 50
                or common['stop_reason'] not in (None, 'proposal_limit', 'at_goal', 'sample_budget_exhausted', 'edge_budget_exhausted')):
            raise ValueError('invalid finite task pool summary')
        for key in ('resources_before', 'resources_after'): _validate_resources(context[key])
        before, after = context['resources_before'], context['resources_after']
        if (before['sample_limit'] != after['sample_limit'] or before['edge_limit'] != after['edge_limit']
                or after['samples']-before['samples'] != common['proposals']
                or any(after[k] < before[k] for k in ('edges', 'verification_edges', 'cache_hits'))):
            raise ValueError('task pool did not pay its shared work')
        _keys(context['receiver_work'], COUNTERS)
        if any(type(v) is not int or v < 0 for v in context['receiver_work'].values()):
            raise ValueError('invalid task receiver work')
        if context['limitations'] != LIMITATIONS: raise ValueError('task limitations cannot be altered or extended')
        for supplied, key in ((history, 'history_sha256'), (models, 'models_sha256'), (state, 'state_sha256')):
            if supplied is not None and digest(supplied) != context[key]: raise ValueError('task context live input changed')
        if geometry is not None and digest(_geometry_summary(geometry)) != context['geometry_sha256']:
            raise ValueError('task context live geometry changed')
        if forecast is not None:
            _check_forecast(forecast, context['planner_method'], context['version'] == COMPOUND_VERSION)
            if (forecast.fingerprint != context['forecast_sha256'] or forecast.now != context['time_seconds']
                    or digest(forecast.history) != context['history_sha256'] or digest(forecast.models) != context['models_sha256']):
                raise ValueError('task context live forecast changed')
        return context
    except (KeyError, TypeError, OverflowError, StopIteration, IndexError) as exc:
        raise ValueError('invalid public task context') from exc


def rescore_task_candidates(context, state, forecast, geometry, work, method='joint'):
    """Re-evaluate the same paths after a source revision, with no new proposals.

    Source models may change. History, physical time, vehicle state, known map
    and prediction method remain fixed. The bound snapshot certifies previous
    geometry/resource checks, so rescoring reuses those results and pays only
    new forecast receiver work. A depleted ledger cannot invalidate a retained
    certificate or silently discard a previously legal path.
    """
    validate_task_context(context, history=forecast.history, state=state, geometry=geometry)
    _check_forecast(forecast, method, context['version'] == COMPOUND_VERSION)
    if forecast.now != context['time_seconds'] or method != context['planner_method']:
        raise ValueError('task candidate rescore requires the same time and prediction method')
    # Keep the dedicated shared limits even when no samples/edges remain.
    state_request(state, forecast, geometry, work)
    if (work.sample_limit != context['resources_after']['sample_limit']
            or work.edge_limit != context['resources_after']['edge_limit']
            or any(work.summary()[k] < context['resources_after'][k] for k in ('samples', 'edges', 'verification_edges'))):
        raise ValueError('task candidate rescore cannot reset the shared work ledger')
    rows = []
    for old in context['candidates']:
        row = deepcopy(old)
        if old['feasible']:
            row['score'] = score_route(forecast, old['path'], state['speed_mps'], state['delay_seconds'], method)
        rows.append(row)
    return rows
