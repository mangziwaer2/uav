"""Bounded joint amplitude/dynamics diagnostics; no births, sensing or truth.

Every component uses p/vx/vy/r in that order. Geometry, reference time and
uncertainty remain fixed to the original seed, including on continuation.
"""
from copy import deepcopy
from itertools import product
import math

import numpy as np
from scipy.optimize import minimize

from .equivalent_field import kernel, normalized_expected_power, power_and_jacobian
from .event_inference import digest


MODES = ('static', 'translate', 'strength_change', 'translate_strength')
PARAMETER_NAMES = ('peak_dbm', 'vx_mps', 'vy_mps', 'rate_db_per_second')
PARAMETER_UNITS = np.array([4., 3., 3., .5])
MAX_EVALUATIONS = 40
MAX_KERNEL_TERMS = 5000
BINS = (0., 5., 10., 20., math.inf)


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError('finite numeric scalar required')
    return float(value)


def _array(value, shape):
    result = np.asarray(value, float)
    if result.size == 0 and math.prod(shape) == 0:
        result = result.reshape(shape)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError('finite array of the declared shape required')
    def validate(item):
        if isinstance(item, (list, tuple, np.ndarray)):
            for x in item: validate(x)
        else: _number(item)
    validate(value)
    return result


def _seed(seed):
    if not isinstance(seed, dict) or not isinstance(seed.get('id'), str) or not seed['id']:
        raise ValueError('named original seed required')
    bases = seed.get('bases')
    if not isinstance(bases, list) or not 1 <= len(bases) <= 2:
        raise ValueError('one or two original bases required')
    _array(seed.get('peaks_dbm'), (len(bases),))
    if _number(seed.get('std_db')) < 0 or _number(seed.get('reference_time_seconds')) < 0:
        raise ValueError('nonnegative fixed uncertainty/reference required')
    for b in bases:
        _array(b['center_m'], (2,)); velocity = _array(b['velocity_mps'], (2,))
        if (_number(b['scale_m']) <= 0 or np.linalg.norm(velocity) > 3.+1e-9
                or abs(_number(b['rate_db_per_second'])) > .5+1e-12):
            raise ValueError('original seed outside dynamic domain')
    if np.any(np.asarray(seed['peaks_dbm']) < -140.):
        raise ValueError('original amplitude outside domain')
    return len(bases)


def pack(model):
    """Full interleaved p/vx/vy/r vector, independent of a mechanism mask."""
    count = _seed(model)
    return np.array([[model['peaks_dbm'][i], *b['velocity_mps'], b['rate_db_per_second']]
                     for i, b in enumerate(model['bases'])], float).reshape(4*count)


def unpack(model, parameters):
    count = len(model['bases']); values = _array(parameters, (4*count,)).reshape(count, 4)
    current = deepcopy(model); current['peaks_dbm'] = values[:, 0].tolist()
    for b, values_b in zip(current['bases'], values):
        b['velocity_mps'] = values_b[1:3].tolist(); b['rate_db_per_second'] = float(values_b[3])
    return current


def prediction_jacobian(model, parameters, xy, times):
    """dBm predictions and full analytic Jacobian for fixed centers/scales."""
    current = unpack(model, parameters); count = len(current['bases'])
    times = _array(times, (len(times),)); xy = _array(xy, (len(times), 2))
    matrix = kernel(current['bases'], xy, times, current['reference_time_seconds'])
    mean, amplitude = power_and_jacobian(matrix, current['peaks_dbm'])
    dt = times-current['reference_time_seconds']
    centers = np.asarray([b['center_m'] for b in current['bases']])
    velocity = np.asarray([b['velocity_mps'] for b in current['bases']])
    scales = np.asarray([b['scale_m'] for b in current['bases']])
    delta = xy[:, None, :]-centers[None, :, :]-dt[:, None, None]*velocity[None, :, :]
    jac = np.zeros((len(times), count, 4)); jac[:, :, 0] = amplitude
    jac[:, :, 1:3] = (10./math.log(10.)*amplitude[:, :, None]*dt[:, None, None]
                      *delta/scales[None, :, None]**2)
    jac[:, :, 3] = amplitude*dt[:, None]
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(jac)):
        raise FloatingPointError('nonfinite joint dynamics prediction')
    return mean, jac.reshape(len(times), 4*count)


def mechanism_options(seed_model):
    """All masks containing every actually nonzero original dynamic parameter."""
    _seed(seed_model); choices = []
    for b in seed_model['bases']:
        moving = any(v != 0. for v in b['velocity_mps'])
        changing = b['rate_db_per_second'] != 0.
        choices.append([m for m in MODES if (not moving or m in ('translate', 'translate_strength'))
                        and (not changing or m in ('strength_change', 'translate_strength'))])
    return [list(modes) for modes in product(*choices)]


def active_parameter_indices(modes):
    if not isinstance(modes, (list, tuple)) or not 1 <= len(modes) <= 2 or any(m not in MODES for m in modes):
        raise ValueError('valid component modes required')
    indices = []
    for i, mode in enumerate(modes):
        indices.append(4*i)
        if mode in ('translate', 'translate_strength'): indices.extend((4*i+1, 4*i+2))
        if mode in ('strength_change', 'translate_strength'): indices.append(4*i+3)
    return indices


def _fit_data(seed, history, soft):
    count = _seed(seed)
    if not isinstance(history, list) or not history: raise ValueError('observed history required')
    for p in history:
        if set(p) != {'position_m', 'time_seconds', 'interference_dbm'}: raise ValueError('point history fields required')
        _array(p['position_m'], (2,)); _number(p['time_seconds']); _number(p['interference_dbm'])
    if (history[0]['time_seconds'] < 0 or any(a['time_seconds'] >= b['time_seconds'] for a, b in zip(history, history[1:]))
            or seed['reference_time_seconds'] > history[-1]['time_seconds']):
        raise ValueError('causal chronological prefix required')
    window = deepcopy(history[-12:]); real_count = len(window)
    xy = [p['position_m'] for p in window]; tt = [p['time_seconds'] for p in window]
    labels = [p['interference_dbm'] for p in window]; weights = [1./math.sqrt(real_count)]*real_count
    if soft is not None:
        n = len(soft['position_m'])
        if (soft['prefix_end_seconds'] != history[-1]['time_seconds'] or soft['independent_evidence_points'] != 0):
            raise ValueError('GP soft points must share the observed prefix and are not independent evidence')
        _array(soft['position_m'], (n, 2)); times = _array(soft['time_seconds'], (n,))
        means = _array(soft['mean_dbm'], (n,)); sw = _array(soft['weights'], (n,))
        strength = _number(soft['strength'])
        if (np.any(times < 0) or np.any(times > history[-1]['time_seconds']) or np.any(sw < 0)
                or strength <= 0 or (n and not math.isclose(float(sw.sum()), 1., abs_tol=1e-10))):
            raise ValueError('normalized causal GP soft constraints required')
        xy += deepcopy(soft['position_m']); tt += times.tolist(); labels += means.tolist()
        weights += np.sqrt(strength*sw).tolist()
    low = np.tile([-140., -3., -3., -.5], count)
    upper_amplitude = max(max(p['interference_dbm'] for p in window)+6., max(seed['peaks_dbm']))
    high = np.tile([upper_amplitude, 3., 3., .5], count)
    return dict(window=window, real_count=real_count, xy=np.asarray(xy), times=np.asarray(tt),
                labels=np.asarray(labels), weights=np.asarray(weights), low=low, high=high,
                upper_amplitude=upper_amplitude, history_sha256=digest(history))


def _eligible(parameters, modes, low, high):
    p = np.asarray(parameters); n = len(modes)
    if (p.shape != (4*n,) or not np.all(np.isfinite(p)) or np.any(p < low-1e-10) or np.any(p > high+1e-10)):
        return False
    for values, mode in zip(p.reshape(n, 4), modes):
        if np.linalg.norm(values[1:3]) > 3.+1e-9: return False
        if mode not in ('translate', 'translate_strength') and np.any(values[1:3] != 0.): return False
        if mode not in ('strength_change', 'translate_strength') and values[3] != 0.: return False
    return True


def _errors(prediction, data, soft):
    residual = prediction-data['labels']; real = residual[:data['real_count']]
    gp = residual[data['real_count']:]
    return dict(real_mse=float(np.mean(real**2)), real_rmse_db=float(np.sqrt(np.mean(real**2))),
        gp_mse=float(np.asarray(soft['weights'])@(gp**2)) if len(gp) else None,
        tail_rmse_db=float(np.sqrt(np.mean(real[-3:]**2))),
        tail_points=deepcopy(data['window'][-3:]), retrospective_only=True, independent_future_evidence=False)


def _diagnostic_model(seed, parameters, modes, identifier, objective, errors):
    model = unpack(seed, parameters)
    model.pop('log_weight', None)
    model.update(id=identifier, family='composite', component_modes=list(modes), parent_id=seed['id'],
        scored_points=0, weight=None, fit_mse=errors['real_mse'], fit_objective=objective,
        born_index=None, original_born_index=seed.get('born_index'),
        diagnostic_only=True, inherited_scored_points=False)
    return model


def fit_mechanism(seed_model, modes, history, soft, identifier, start_model=None):
    """One hard-budget joint solve; continuation always uses the original domain."""
    if not isinstance(identifier, str) or not identifier: raise ValueError('new diagnostic identifier required')
    data = _fit_data(seed_model, history, soft); count = len(seed_model['bases'])
    if list(modes) not in mechanism_options(seed_model): raise ValueError('mode must contain original dynamics')
    modes = list(modes); seed_parameters = pack(seed_model)
    low, high = data['low'], data['high']
    if not _eligible(seed_parameters, modes, low, high): raise ValueError('original seed outside the fixed domain')
    start = seed_parameters.copy()
    if start_model is not None:
        if (start_model.get('parent_id') != seed_model['id'] or start_model.get('component_modes') != modes
                or start_model.get('family') != 'composite' or start_model.get('weight') is not None
                or start_model.get('scored_points') != 0
                or start_model['reference_time_seconds'] != seed_model['reference_time_seconds']
                or start_model['std_db'] != seed_model['std_db']
                or [(b['center_m'], b['scale_m']) for b in start_model['bases']] !=
                   [(b['center_m'], b['scale_m']) for b in seed_model['bases']]):
            raise ValueError('continuation must preserve the original seed geometry, reference and mode')
        start = pack(start_model)
        if not _eligible(start, modes, low, high): raise ValueError('continuation outside original domain')
    resources = dict(solver_calls=0, parameter_evaluations=0, objective_calls=0, gradient_calls=0,
                     kernel_terms=0, cache_hits=0, constraint_calls=0, constraint_jacobian_calls=0)
    trace = []; cache = {}; best = None
    def evaluate(parameters):
        nonlocal best
        parameters = np.asarray(parameters, float); key = parameters.tobytes()
        if key in cache:
            resources['cache_hits'] += 1; return cache[key]
        terms = count*len(data['times'])
        if resources['parameter_evaluations'] >= MAX_EVALUATIONS: raise ValueError('parameter evaluation budget exhausted')
        if resources['kernel_terms']+terms > MAX_KERNEL_TERMS: raise ValueError('fit kernel budget exhausted')
        resources['parameter_evaluations'] += 1; resources['kernel_terms'] += terms
        prediction, jac = prediction_jacobian(seed_model, parameters, data['xy'], data['times'])
        residual = (prediction-data['labels'])*data['weights']; loss = float(residual@residual)
        legal = _eligible(parameters, modes, low, high)
        value = dict(parameters=parameters.copy(), prediction=prediction, jacobian=jac, objective=loss, eligible=legal)
        cache[key] = value
        trace.append(dict(parameters=parameters.tolist(), objective=loss, eligible=legal))
        if legal and (best is None or loss < best['objective']): best = value
        return value
    baseline = evaluate(seed_parameters)
    if start_model is not None and evaluate(start)['objective'] > baseline['objective']+1e-8:
        raise ValueError('continuation cannot worsen the original fit objective')
    free = np.asarray(active_parameter_indices(modes), int); units = np.tile(PARAMETER_UNITS, count)[free]
    def parameters(z):
        p = start.copy(); p[free] += units*np.asarray(z); return p
    def objective(z):
        if resources['objective_calls'] >= MAX_EVALUATIONS: raise ValueError('objective budget exhausted')
        resources['objective_calls'] += 1; return evaluate(parameters(z))['objective']
    def gradient(z):
        if resources['gradient_calls'] >= MAX_EVALUATIONS: raise ValueError('gradient budget exhausted')
        resources['gradient_calls'] += 1; value = evaluate(parameters(z))
        return 2*(value['jacobian'].T@((value['prediction']-data['labels'])*data['weights']**2))[free]*units
    moving = [i for i, m in enumerate(modes) if m in ('translate', 'translate_strength')]
    def constraint(z):
        resources['constraint_calls'] += 1; p = parameters(z).reshape(count, 4)
        return np.array([9.-p[i, 1:3]@p[i, 1:3] for i in moving])
    def constraint_jacobian(z):
        resources['constraint_jacobian_calls'] += 1; p = parameters(z).reshape(count, 4)
        jac = np.zeros((len(moving), 4*count))
        for row, i in enumerate(moving): jac[row, 4*i+1:4*i+3] = -2*p[i, 1:3]
        return jac[:, free]*units
    report = dict(solver_success=False, solver_status=None, solver_message=None)
    try:
        resources['solver_calls'] = 1
        solved = minimize(objective, np.zeros(len(free)), jac=gradient, method='SLSQP',
            bounds=list(zip((low[free]-start[free])/units, (high[free]-start[free])/units)),
            constraints=[dict(type='ineq', fun=constraint, jac=constraint_jacobian)] if moving else [],
            options=dict(ftol=1e-8, maxiter=40))
        report.update(solver_success=bool(solved.success), solver_status=int(solved.status),
            solver_message=str(solved.message), solver_final_parameters=parameters(solved.x).tolist(),
            solver_final_eligible=_eligible(parameters(solved.x), modes, low, high))
    except (ValueError, FloatingPointError, RuntimeError) as exc:
        report['error'] = str(exc)
    selected = best if best is not None and best['objective'] <= baseline['objective']+1e-8 else baseline
    errors = _errors(selected['prediction'], data, soft); seed_errors = _errors(baseline['prediction'], data, soft)
    improved = selected['objective'] < baseline['objective']-1e-8
    stop = report.get('error') or ('solver_converged' if report['solver_success'] else 'solver_nonconverged')
    return dict(version='joint_mechanism_fit_v1', model=_diagnostic_model(seed_model, selected['parameters'], modes,
            identifier, selected['objective'], errors), seed_model=deepcopy(seed_model), component_modes=modes,
        window=data['window'], history_sha256=data['history_sha256'], gp_soft_constraints=deepcopy(soft),
        fit_objective=selected['objective'], seed_objective=baseline['objective'], improved=improved, legal=True,
        errors=errors, seed_errors=seed_errors, weighted_fit_jacobian=(selected['jacobian']*data['weights'][:, None]).tolist(),
        resources=resources, trace=trace, stop_reason=stop, solver=report,
        domain=dict(lower=low.tolist(), upper=high.tolist(), parameter_units=np.tile(PARAMETER_UNITS, count).tolist(),
            velocity_norm_limit_mps=3., original_seed_id=seed_model['id']),
        fitting_history_is_future_evidence=False, route_cost_in_fit_objective=False,
        accepted_nonincreasing=True, independent_future_evidence=False)


def _anchor(anchor, history_sha256=None):
    if history_sha256 is not None and anchor.get('history_sha256') != history_sha256:
        raise ValueError('task feedback history binding changed')
    now = _number(anchor['time_seconds']); _number(anchor['baseline_delta_c'])
    routes = [anchor['route_a'], anchor['route_b']]
    for route in routes:
        if not isinstance(route['id'], str) or not route['id']: raise ValueError('named anchor routes required')
        q = route['queries']; n = len(q['time_seconds'])
        _array(q['position_m'], (n, 2)); tt = _array(q['time_seconds'], (n,)); dt = _array(q['dt_seconds'], (n,))
        g = _array(route['gp_weight'], (n,)); fixed = _array(route['gp_fixed_power'], (n,))
        if (not n or np.any(tt <= now) or np.any(np.diff(tt) <= 0) or np.any(dt <= 0)
                or np.any(g < 0) or np.any(g > 1) or np.any(fixed < -1e-9)
                or not math.isclose(float(dt.sum()), _number(q['duration_seconds']), abs_tol=1e-7)):
            raise ValueError('valid frozen future route quadrature required')
    if routes[0]['id'] == routes[1]['id']: raise ValueError('two distinct anchor routes required')
    return routes


def _route_results(model, parameters, anchor, resources, cache):
    routes = _anchor(anchor); xy = sum((r['queries']['position_m'] for r in routes), [])
    times = sum((r['queries']['time_seconds'] for r in routes), []); key = np.asarray(parameters).tobytes()
    if key in cache:
        resources['cache_hits'] += 1; mean, jac = cache[key]
    else:
        terms = len(model['bases'])*len(times)
        if resources['kernel_terms']+terms > MAX_KERNEL_TERMS: raise ValueError('task feedback kernel budget exhausted')
        resources['kernel_terms'] += terms; resources['parameter_evaluations'] += 1
        mean, jac = prediction_jacobian(model, parameters, xy, times); cache[key] = (mean, jac)
    expected = normalized_expected_power(mean, model['std_db'], -80.); results = []; offset = 0
    for route in routes:
        q = route['queries']; n = len(q['time_seconds']); region = slice(offset, offset+n)
        dt = np.asarray(q['dt_seconds']); g = np.asarray(route['gp_weight']); fixed = np.asarray(route['gp_fixed_power'])
        powers = fixed+(1-g)*expected[region]; now = anchor['time_seconds']
        horizon = np.asarray(q['time_seconds'])-now
        bins = [float(powers[(horizon >= lo)&(horizon < hi)]@dt[(horizon >= lo)&(horizon < hi)]) for lo, hi in zip(BINS, BINS[1:])]
        gradient = (dt*(1-g)*expected[region]*(math.log(10.)/10.))@jac[region]
        power = float(powers@dt)
        results.append(dict(id=route['id'], power_proxy_seconds=power,
            score_seconds=q['duration_seconds']+power, delta_q_bins=bins, gradient=gradient))
        offset += n
    return results


def task_feedback(fit, anchor_cache):
    """Conditional A/B algebra and gradients, with no candidate weight inheritance."""
    _anchor(anchor_cache, fit['history_sha256'])
    if fit['window'][-1]['time_seconds'] != anchor_cache['time_seconds']: raise ValueError('task feedback time changed')
    seed = fit['seed_model']; model = fit['model']; seed_id = seed['id']
    if seed_id not in anchor_cache['worlds']: raise ValueError('seed world absent from shared task cache')
    world = anchor_cache['worlds'][seed_id]; weight = _number(world['weight'])
    if not 0 <= weight <= 1: raise ValueError('frozen seed weight required')
    resources = dict(kernel_terms=0, parameter_evaluations=0, cache_hits=0); cache = {}
    before = _route_results(seed, pack(seed), anchor_cache, resources, cache)
    after = _route_results(seed, pack(model), anchor_cache, resources, cache)
    seed_dq = before[1]['power_proxy_seconds']-before[0]['power_proxy_seconds']
    if not math.isclose(seed_dq, _number(world['seed_delta_q']), rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError('cached original seed task contrast changed')
    seed_dc = before[1]['score_seconds']-before[0]['score_seconds']
    candidate_dc = after[1]['score_seconds']-after[0]['score_seconds']; change = candidate_dc-seed_dc
    def contrast_bins(rows): return (np.asarray(rows[1]['delta_q_bins'])-rows[0]['delta_q_bins']).tolist()
    seed_bins, candidate_bins = contrast_bins(before), contrast_bins(after)
    gradient = (after[1]['gradient']-after[0]['gradient']).tolist()
    return dict(version='joint_task_feedback_v1', seed_id=seed_id, candidate_id=model['id'],
        route_a=anchor_cache['route_a']['id'], route_b=anchor_cache['route_b']['id'],
        seed_delta_c=seed_dc, candidate_delta_c=candidate_dc, delta_c_change=change,
        baseline_delta_c=anchor_cache['baseline_delta_c'],
        frozen_weight_substitution_delta_c=anchor_cache['baseline_delta_c']+weight*change,
        frozen_weight=weight, candidate_weight=None, conditional_substitution_only=True,
        seed_delta_q_by_arrival_bin=seed_bins, candidate_delta_q_by_arrival_bin=candidate_bins,
        delta_c_change_by_arrival_bin=(np.asarray(candidate_bins)-seed_bins).tolist(),
        arrival_bins_relative_seconds=[[lo, None if math.isinf(hi) else hi] for lo, hi in zip(BINS, BINS[1:])],
        binning='frozen future quadrature midpoints; bins contain power differences, not flight time',
        task_gradient=gradient, seed_task_gradient=(before[1]['gradient']-before[0]['gradient']).tolist(),
        weighted_fit_jacobian=deepcopy(fit['weighted_fit_jacobian']),
        errors=deepcopy(fit['errors']), seed_errors=deepcopy(fit['seed_errors']),
        fit_objective=fit['fit_objective'], seed_objective=fit['seed_objective'],
        parameter_units=deepcopy(fit['domain']['parameter_units']),
        resources=resources, source_reference_time_seconds=seed['reference_time_seconds'],
        gp_time_offsets_seconds=[t-seed['reference_time_seconds'] for t in (fit['gp_soft_constraints'] or {}).get('time_seconds', [])],
        independent_future_evidence=False, route_cost_in_fit_objective=False)


def estimate_seed_analysis_terms(seed_model, history, soft, anchor_cache):
    data = _fit_data(seed_model, history, soft); routes = _anchor(anchor_cache, data['history_sha256'])
    return len(seed_model['bases'])*(len(data['times'])+sum(len(r['queries']['time_seconds']) for r in routes))


def analyze_seed(seed_model, history, soft, anchor_cache):
    """Pay once for full seed fit/task Jacobians; mode masks reuse this table."""
    data = _fit_data(seed_model, history, soft); count = len(seed_model['bases'])
    _anchor(anchor_cache, data['history_sha256'])
    terms = count*len(data['times'])
    if terms > MAX_KERNEL_TERMS: raise ValueError('seed fit kernel budget exhausted')
    parameters = pack(seed_model)
    prediction, jac = prediction_jacobian(seed_model, parameters, data['xy'], data['times'])
    residual = (prediction-data['labels'])*data['weights']; loss = float(residual@residual)
    errors = _errors(prediction, data, soft)
    fit = dict(model=deepcopy(seed_model), seed_model=deepcopy(seed_model), window=data['window'],
        history_sha256=data['history_sha256'], gp_soft_constraints=deepcopy(soft), fit_objective=loss, seed_objective=loss,
        weighted_fit_jacobian=(jac*data['weights'][:, None]).tolist(), errors=errors, seed_errors=errors,
        domain=dict(parameter_units=np.tile(PARAMETER_UNITS, count).tolist()))
    feedback = task_feedback(fit, anchor_cache)
    feedback['resources']['kernel_terms'] += terms
    feedback['resources']['fit_parameter_evaluations'] = 1
    feedback['resources']['solver_calls'] = 0
    feedback['mechanism_options'] = mechanism_options(seed_model)
    feedback['history_sha256'] = data['history_sha256']
    return feedback
