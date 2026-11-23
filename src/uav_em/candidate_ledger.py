"""Bounded candidate bookkeeping, with no fitting, prediction, sensing or truth.

All fits remain inspectable. Quality frontiers are split by the sign of the
conditional world's route contrast; a sign is never treated as the right one.
Stopping computation and selecting diagnostics are separate operations. Neither
operation gives a candidate a weight or commits it to an active source bank.
"""
from copy import deepcopy
import math

from .event_inference import digest
from . import feedback_revision as tool


def _finite(value, *, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or nonnegative and value < 0):
        raise ValueError('finite numeric diagnostic required')
    return float(value)


def _close(left, right):
    return math.isclose(_finite(left), _finite(right), rel_tol=1e-9, abs_tol=1e-8)


def _errors(value):
    if not isinstance(value, dict):
        raise ValueError('absolute fit errors required')
    for name in ('real_mse', 'real_rmse_db', 'tail_rmse_db'):
        _finite(value[name], nonnegative=True)
    if not _close(value['real_rmse_db']**2, value['real_mse']):
        raise ValueError('cached absolute real error changed')
    if value['gp_mse'] is not None:
        _finite(value['gp_mse'], nonnegative=True)
    if value.get('independent_future_evidence') is not False or value.get('retrospective_only') is not True:
        raise ValueError('known fit errors cannot be future evidence')


def _sign(value):
    value = _finite(value)
    return 'positive' if value > 0 else 'negative' if value < 0 else 'zero'


def mechanism_class(modes):
    """Coarse capability coverage; partial masks of a joint job share a class.

    For example [TR,TR] and [R,TR] both permit translation and strength change.
    They remain distinct jobs, but do not count as independent mechanism-class
    coverage. Pure strength and pure translation remain available countermodels.
    """
    tool.active_parameter_indices(modes)
    moving = any(mode in ('translate', 'translate_strength') for mode in modes)
    changing = any(mode in ('strength_change', 'translate_strength') for mode in modes)
    return 'joint' if moving and changing else 'translate' if moving else 'strength' if changing else 'static'


def _job_key(option):
    if not isinstance(option.get('target_id'), str) or not option['target_id']:
        raise ValueError('named candidate target required')
    mechanism_class(option['component_modes'])
    return option['target_id'], tuple(option['component_modes'])


def choose_option(options, coverage, *, continuations=(), remaining_jobs=1):
    """Shared deterministic scheduling from cached priorities and coverage only.

    The first job uses maximum shared priority. Later jobs first cover an unseen
    coarse mechanism class, then an unseen parent, before revisiting a covered
    parent/class with a nested mask. No legal mask is permanently banned. A
    higher-priority eligible unfinished continuation can still use the budget.
    This is an explicit heuristic, not an optimal experimental-design policy.
    """
    if type(remaining_jobs) is not int or remaining_jobs < 0:
        raise ValueError('nonnegative remaining job allowance required')
    if remaining_jobs == 0:
        return None
    choices = []
    keys = set()
    for option in options:
        key = _job_key(option)
        if key in keys:
            raise ValueError('duplicate public job option')
        keys.add(key)
        _finite(option['priority'], nonnegative=True)
        choices.append(option)
    seen = {_job_key(row) for row in coverage}
    if not seen.issubset(keys):
        raise ValueError('coverage outside common job options')
    classes = {mechanism_class(row['component_modes']) for row in coverage}
    parents = {row['target_id'] for row in coverage}
    def ranking(option):
        return (mechanism_class(option['component_modes']) in classes,
                option['target_id'] in parents, -option['priority'],
                option['target_id'], tuple(option['component_modes']))
    untried = sorted((x for x in choices if _job_key(x) not in seen), key=ranking)
    chosen = untried[0] if untried else None
    continuable = []
    for row in continuations:
        key = _job_key(row)
        if key not in seen:
            raise ValueError('continuation requires a covered original job')
        if type(row.get('numeric_continue_eligible')) is not bool:
            raise ValueError('explicit continuation eligibility required')
        _finite(row['priority'], nonnegative=True)
        if row['numeric_continue_eligible']:
            continuable.append(row)
    continuable.sort(key=lambda x: (-x['priority'], x['target_id'], tuple(x['component_modes'])))
    previous = continuable[0] if continuable else None
    if previous is not None and (chosen is None or previous['priority'] > chosen['priority']):
        return dict(target_id=previous['target_id'], component_modes=deepcopy(previous['component_modes']),
            start='previous', reason='continue higher-priority unfinished legal improvement; preserve all previous diagnostics')
    if chosen is None:
        return None
    return dict(target_id=chosen['target_id'], component_modes=deepcopy(chosen['component_modes']),
        start='original', reason='shared mechanism-class and parent coverage, then cached priority; no repeated original job')


def public_diagnostics(fit, anchor_cache):
    """Uncalibrated absolute fit/domain/extrapolation facts from existing arrays.

    Distances use the at-most-12-point fitted window, not an invented sensor
    support radius. No pass/fail applicability gate or future error is inferred.
    """
    if fit['history_sha256'] != anchor_cache['history_sha256']:
        raise ValueError('diagnostic history binding changed')
    tool._anchor(anchor_cache, fit['history_sha256'])
    _errors(fit['errors']); _errors(fit['seed_errors'])
    history = fit['window']
    if not history or history[-1]['time_seconds'] != anchor_cache['time_seconds']:
        raise ValueError('diagnostic window and task time differ')
    model = fit['model']; values = tool.pack(model).tolist()
    low, high = fit['domain']['lower'], fit['domain']['upper']
    if len(low) != len(values) or len(high) != len(values):
        raise ValueError('parameter domain shape changed')
    touches = []
    for index, value in enumerate(values):
        component, offset = divmod(index, 4)
        for side, bound in (('lower', low[index]), ('upper', high[index])):
            if _close(value, bound):
                touches.append(dict(component=component, parameter=tool.PARAMETER_NAMES[offset], side=side, value=value))
    for component, base in enumerate(model['bases']):
        speed = math.hypot(*base['velocity_mps'])
        if _close(speed, fit['domain']['velocity_norm_limit_mps']):
            touches.append(dict(component=component, parameter='velocity_norm_mps', side='upper', value=speed))
    lower = [min(p['position_m'][axis] for p in history) for axis in range(2)]
    upper = [max(p['position_m'][axis] for p in history) for axis in range(2)]
    scale = min(b['scale_m'] for b in model['bases'])
    span = history[-1]['time_seconds']-history[0]['time_seconds']
    routes = []
    for route in (anchor_cache['route_a'], anchor_cache['route_b']):
        queries = []
        for xy, time in zip(route['queries']['position_m'], route['queries']['time_seconds']):
            nearest = min(math.dist(xy, point['position_m']) for point in history)
            queries.append(dict(position_m=deepcopy(xy), time_seconds=time,
                horizon_seconds=time-history[-1]['time_seconds'],
                nearest_fit_point_distance_m=nearest,
                nearest_distance_in_smallest_scale=nearest/scale,
                inside_fit_window_bbox=all(lo <= x <= hi for lo, x, hi in zip(lower, xy, upper))))
        max_horizon = max(q['horizon_seconds'] for q in queries)
        routes.append(dict(id=route['id'], queries=queries, maximum_horizon_seconds=max_horizon,
            horizon_over_fit_time_span=max_horizon/span if span > 0 else None,
            maximum_nearest_fit_point_distance_m=max(q['nearest_fit_point_distance_m'] for q in queries),
            maximum_nearest_distance_in_smallest_scale=max(q['nearest_distance_in_smallest_scale'] for q in queries),
            outside_fit_window_bbox_count=sum(not q['inside_fit_window_bbox'] for q in queries)))
    return dict(absolute_errors=deepcopy(fit['errors']), parent_absolute_errors=deepcopy(fit['seed_errors']),
        gp_rmse_db=math.sqrt(fit['errors']['gp_mse']) if fit['errors']['gp_mse'] is not None else None,
        gp_real_mse_difference=(fit['errors']['gp_mse']-fit['errors']['real_mse']) if fit['errors']['gp_mse'] is not None else None,
        parameter_bound_touches=touches, routes=routes,
        fit_window_bounds_m=[lower, upper], fit_window_time_span_seconds=span,
        spatial_reference='raw points in the fitted window, not a calibrated support or safety domain',
        calibrated=False, applicable=None, unknown_is_safe=False,
        independent_future_evidence=False, new_prediction_queries=0)


class CandidateLedger:
    """Append-only immutable-context diagnostics with explicit selection intent."""
    def __init__(self, history, anchor_cache):
        if not history or anchor_cache['history_sha256'] != digest(history):
            raise ValueError('ledger requires one bound observed history')
        tool._anchor(anchor_cache, digest(history))
        if anchor_cache['time_seconds'] != history[-1]['time_seconds']:
            raise ValueError('ledger task time differs from observed history')
        self._history = deepcopy(history)
        self._anchor = deepcopy(anchor_cache)
        self._context_digest = digest(dict(history=history, anchor_cache=anchor_cache))
        self._entries = []
        self._parent_seeds = {}
        self._soft_digest = None
        self._stopped_reason = None
        self._selection_ids = []
        self._seal()

    def _state(self):
        return dict(history=self._history, anchor_cache=self._anchor, entries=self._entries,
            parent_seeds=self._parent_seeds,
            soft_digest=self._soft_digest, stopped_reason=self._stopped_reason,
            selection_ids=self._selection_ids)

    def _seal(self):
        self._state_digest = digest(self._state())

    def _check(self):
        if digest(self._state()) != self._state_digest:
            raise ValueError('candidate ledger content changed')
        if digest(dict(history=self._history, anchor_cache=self._anchor)) != self._context_digest:
            raise ValueError('candidate ledger context changed')

    def assert_bound(self, history, anchor_cache):
        self._check()
        if digest(dict(history=history, anchor_cache=anchor_cache)) != self._context_digest:
            raise ValueError('candidate ledger live context changed')

    def add(self, fit, feedback, *, origin=None):
        """Record one already-evaluated legal result; never grant credibility."""
        self._check()
        if self._stopped_reason is not None:
            raise ValueError('cannot add after stopping computation')
        # Validate JSON before accepting aliases or nonfinite metadata.
        digest(dict(fit=fit, feedback=feedback, origin=origin))
        model, seed = fit['model'], fit['seed_model']
        identifier = model['id']
        if not isinstance(identifier, str) or not identifier or any(e['candidate_id'] == identifier for e in self._entries):
            raise ValueError('unique named diagnostic candidate required')
        if (fit.get('legal') is not True or fit.get('accepted_nonincreasing') is not True
                or fit.get('route_cost_in_fit_objective') is not False
                or fit.get('independent_future_evidence') is not False
                or fit.get('fitting_history_is_future_evidence') is not False
                or fit['history_sha256'] != digest(self._history)
                or fit['window'] != self._history[-12:]):
            raise ValueError('fit policy or observed history binding changed')
        if (not {'weight', 'born_index', 'scored_points'}.issubset(model)
                or model.get('family') != 'composite' or model.get('weight') is not None
                or model.get('born_index') is not None or model.get('scored_points') != 0
                or type(model.get('scored_points')) is not int
                or 'log_weight' in model or model.get('diagnostic_only') is not True
                or model.get('inherited_scored_points') is not False
                or model.get('parent_id') != seed['id']
                or model.get('component_modes') != fit['component_modes']):
            raise ValueError('pool-external diagnostic metadata required')
        modes = fit['component_modes']
        if list(modes) not in tool.mechanism_options(seed):
            raise ValueError('candidate mode removed original dynamics')
        if seed['id'] in self._parent_seeds and digest(seed) != self._parent_seeds[seed['id']]:
            raise ValueError('original seed content changed across jobs')
        if (model['std_db'] != seed['std_db'] or model['reference_time_seconds'] != seed['reference_time_seconds']
                or [(b['center_m'], b['scale_m']) for b in model['bases']] !=
                   [(b['center_m'], b['scale_m']) for b in seed['bases']]):
            raise ValueError('fixed geometry, uncertainty or reference changed')
        parameters = tool.pack(model)
        data = tool._fit_data(seed, self._history, fit['gp_soft_constraints'])
        if (fit['domain']['lower'] != data['low'].tolist()
                or fit['domain']['upper'] != data['high'].tolist()
                or fit['domain']['original_seed_id'] != seed['id']
                or fit['domain']['velocity_norm_limit_mps'] != 3.
                or not tool._eligible(parameters, modes, data['low'], data['high'])):
            raise ValueError('candidate outside original parameter domain')
        _finite(fit['fit_objective'], nonnegative=True); _finite(fit['seed_objective'], nonnegative=True)
        if fit['fit_objective'] > fit['seed_objective']+1e-8:
            raise ValueError('nonincreasing fit objective required')
        _errors(fit['errors']); _errors(fit['seed_errors'])
        soft = fit['gp_soft_constraints']
        strength = soft['strength'] if soft is not None and len(soft['position_m']) else 0.
        for errors, objective in ((fit['errors'], fit['fit_objective']), (fit['seed_errors'], fit['seed_objective'])):
            if ((errors['gp_mse'] is None) != (not strength)
                    or errors['tail_points'] != fit['window'][-3:]):
                raise ValueError('GP presence or known-tail context changed')
            if not _close(objective, errors['real_mse']+strength*(errors['gp_mse'] or 0.)):
                raise ValueError('cached real/GP objective algebra changed')
        if (not _close(model['fit_mse'], fit['errors']['real_mse'])
                or not _close(model['fit_objective'], fit['fit_objective'])):
            raise ValueError('candidate absolute fit metadata changed')
        soft_digest = digest(fit['gp_soft_constraints'])
        if self._soft_digest is not None and soft_digest != self._soft_digest:
            raise ValueError('candidate quality comparisons require the same GP soft constraints')
        anchor = self._anchor
        if (feedback['candidate_id'] != identifier or feedback['seed_id'] != seed['id']
                or feedback['route_a'] != anchor['route_a']['id'] or feedback['route_b'] != anchor['route_b']['id']
                or feedback.get('conditional_substitution_only') is not True
                or feedback.get('candidate_weight') is not None
                or feedback.get('independent_future_evidence') is not False
                or feedback.get('route_cost_in_fit_objective') is not False
                or seed['id'] not in anchor['worlds']
                or feedback['errors'] != fit['errors'] or feedback['seed_errors'] != fit['seed_errors']
                or not _close(feedback['fit_objective'], fit['fit_objective'])
                or not _close(feedback['seed_objective'], fit['seed_objective'])
                or not _close(feedback['baseline_delta_c'], anchor['baseline_delta_c'])
                or not _close(feedback['frozen_weight'], anchor['worlds'][seed['id']]['weight'])):
            raise ValueError('task feedback candidate, parent or frozen-context binding changed')
        change = feedback['candidate_delta_c']-feedback['seed_delta_c']
        duration_change = anchor['route_b']['queries']['duration_seconds']-anchor['route_a']['queries']['duration_seconds']
        weight = _finite(feedback['frozen_weight'], nonnegative=True)
        if weight > 1:
            raise ValueError('frozen source weight outside probability domain')
        if (not _close(change, feedback['delta_c_change'])
                or not _close(feedback['seed_delta_c'], duration_change+anchor['worlds'][seed['id']]['seed_delta_q'])
                or not _close(sum(feedback['delta_c_change_by_arrival_bin']), change)
                or not _close(feedback['frozen_weight_substitution_delta_c'],
                    anchor['baseline_delta_c']+feedback['frozen_weight']*change)):
            raise ValueError('cached task feedback algebra changed')
        expected_bins = [[0., 5.], [5., 10.], [10., 20.], [20., None]]
        if (feedback.get('arrival_bins_relative_seconds') != expected_bins
                or any(len(feedback[key]) != 4 for key in ('delta_c_change_by_arrival_bin',
                    'seed_delta_q_by_arrival_bin', 'candidate_delta_q_by_arrival_bin'))):
            raise ValueError('task feedback time bins changed')
        if (not _close(sum(feedback['seed_delta_q_by_arrival_bin'])+duration_change, feedback['seed_delta_c'])
                or not _close(sum(feedback['candidate_delta_q_by_arrival_bin'])+duration_change, feedback['candidate_delta_c'])
                or any(not _close(c-a, change) for a, c, change in zip(feedback['seed_delta_q_by_arrival_bin'],
                    feedback['candidate_delta_q_by_arrival_bin'], feedback['delta_c_change_by_arrival_bin']))):
            raise ValueError('cached per-bin task contrasts changed')
        entry = dict(candidate_id=identifier, parent_id=seed['id'], fit=deepcopy(fit), feedback=deepcopy(feedback),
            origin=deepcopy(origin), diagnostics=public_diagnostics(fit, anchor),
            candidate_delta_c_sign=_sign(feedback['candidate_delta_c']),
            conditional_delta_c_sign=_sign(feedback['frozen_weight_substitution_delta_c']))
        self._entries.append(entry)
        self._parent_seeds[seed['id']] = digest(seed)
        self._soft_digest = soft_digest
        self._seal()
        return identifier

    def stop(self, reason='keep'):
        """Stop further computation while preserving every computed diagnostic."""
        self._check()
        if not isinstance(reason, str) or not 1 <= len(reason) <= 1000:
            raise ValueError('bounded stopping reason required')
        if self._stopped_reason is not None and self._stopped_reason != reason:
            raise ValueError('stopping reason already frozen')
        self._stopped_reason = reason
        self._seal()

    def select(self, candidate_ids):
        """Record explicit selection intent and return diagnostic copies only."""
        self._check()
        if (not isinstance(candidate_ids, list) or any(not isinstance(x, str) for x in candidate_ids)
                or len(set(candidate_ids)) != len(candidate_ids)):
            raise ValueError('distinct candidate selection IDs required')
        entries = {e['candidate_id']: e for e in self._entries}
        if any(x not in entries for x in candidate_ids):
            raise ValueError('selection outside computed legal diagnostics')
        self._selection_ids = deepcopy(candidate_ids)
        self._seal()
        return [deepcopy(entries[x]['fit']['model']) for x in candidate_ids]

    def snapshot(self):
        self._check()
        groups = {}
        frontier_ids = set()
        for sign in ('negative', 'zero', 'positive'):
            rows = [e for e in self._entries if e['candidate_delta_c_sign'] == sign]
            nondominated = []
            def quality(row):
                errors = row['fit']['errors']
                return (errors['real_mse'],) if errors['gp_mse'] is None else (errors['real_mse'], errors['gp_mse'])
            for row in rows:
                q = quality(row)
                if not any(all(a <= b for a, b in zip(quality(other), q))
                    and any(a < b for a, b in zip(quality(other), q)) for other in rows):
                    nondominated.append(row['candidate_id'])
            extremes = []
            for field in ('candidate_delta_c', 'frozen_weight_substitution_delta_c'):
                if rows:
                    for reverse in (False, True):
                        order = sorted(rows, key=lambda e: ((-1 if reverse else 1)*e['feedback'][field], e['candidate_id']))
                        if order[0]['candidate_id'] not in extremes:
                            extremes.append(order[0]['candidate_id'])
            protected = set(nondominated+extremes)
            groups[sign] = dict(quality_frontier_ids=nondominated, task_extreme_ids=extremes,
                frontier_ids=[e['candidate_id'] for e in rows if e['candidate_id'] in protected])
            frontier_ids.update(protected)
        result = dict(history_sha256=digest(self._history), anchor_sha256=digest(self._anchor),
            entries=deepcopy(self._entries), groups=groups,
            frontier_ids=[e['candidate_id'] for e in self._entries if e['candidate_id'] in frontier_ids],
            stopped_reason=self._stopped_reason, selection_ids=deepcopy(self._selection_ids),
            active_pool_committed=False, candidate_weights_assigned=False,
            frontier_policy='per-sign real/GP Pareto quality plus contrast extremes; all legal entries retained',
            task_sign_is_correctness=False, calibrated=False)
        result['ledger_id'] = digest(result)
        return result
