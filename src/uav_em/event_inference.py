"""Causal, persistent equivalent-field hypotheses and bounded inference plans.

No world, oracle, path planner, neural model or LLM backend enters this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
from scipy.optimize import least_squares

from .probability import gaussian_logpdf, logsumexp, normalized
from .equivalent_field import kernel, power_and_jacobian


FAMILIES = ('static', 'translate', 'strength_change')
REGIONS = ('task_disk', 'left_half', 'right_half')
COMPOUND_MODES = ('static', 'translate', 'strength_change', 'translate_strength')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def point(value):
    if set(value) != {'position_m', 'time_seconds', 'interference_dbm'}:
        raise ValueError('only point measurements accepted')
    xy = np.asarray(value['position_m'], float)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError('finite position required')
    t, y = float(value['time_seconds']), float(value['interference_dbm'])
    if not np.all(np.isfinite([t, y])) or t < 0:
        raise ValueError('finite measurement required')
    return dict(position_m=xy.tolist(), time_seconds=t, interference_dbm=y)


def validate_compound_model(model, *, active=False):
    """Validate a compound field without expanding the legacy tool families.

    A mode is a permitted parameter group: its permitted velocity/rate may be
    zero. Parameters omitted by a mode must be exactly zero. Active validation
    accepts already scored bank snapshots; birth additionally enforces fresh
    evidence and strips no submitted evidence on the caller's behalf.
    """
    def scalar(value):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                or not math.isfinite(value)):
            raise ValueError('finite compound scalar required')
        return float(value)

    def pair(value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError('compound two-dimensional vector required')
        return np.asarray([scalar(v) for v in value])

    if (not isinstance(model, dict) or model.get('family') != 'composite'
            or not isinstance(model.get('id'), str) or not 1 <= len(model['id']) <= 200):
        raise ValueError('named compound field required')
    bases, modes, peaks = model.get('bases'), model.get('component_modes'), model.get('peaks_dbm')
    if (not isinstance(bases, list) or not 1 <= len(bases) <= 2
            or not isinstance(modes, list) or len(modes) != len(bases)
            or any(mode not in COMPOUND_MODES for mode in modes)
            or not isinstance(peaks, list) or len(peaks) != len(bases)):
        raise ValueError('one or two bases with explicit compound modes required')
    if (scalar(model.get('std_db')) < 0.
            or scalar(model.get('reference_time_seconds')) < 0):
        raise ValueError('compound nonnegative fixed uncertainty/reference required')
    for base, mode, peak in zip(bases, modes, peaks):
        if (not isinstance(base, dict)
                or set(base) != {'center_m', 'scale_m', 'velocity_mps', 'rate_db_per_second'}):
            raise ValueError('exact compound basis fields required')
        pair(base['center_m']); velocity = pair(base['velocity_mps'])
        rate = scalar(base['rate_db_per_second']); scalar(peak)
        if scalar(base['scale_m']) <= 0 or np.linalg.norm(velocity) > 3.+1e-9 or abs(rate) > .5+1e-12:
            raise ValueError('compound basis outside physical parameter domain')
        if mode not in ('translate', 'translate_strength') and np.any(velocity != 0.):
            raise ValueError('compound mode forbids velocity')
        if mode not in ('strength_change', 'translate_strength') and rate != 0.:
            raise ValueError('compound mode forbids strength change')
    if active:
        if (type(model.get('born_index')) is not int or model['born_index'] < 0
                or type(model.get('scored_points')) is not int or model['scored_points'] < 0
                or model.get('diagnostic_only') is not False):
            raise ValueError('active compound version metadata required')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            matrix = kernel(bases, [base['center_m'] for base in bases],
                            [model['reference_time_seconds']]*len(bases), model['reference_time_seconds'])
            means, jacobian = power_and_jacobian(matrix, peaks)
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(jacobian)):
            raise ValueError('nonfinite compound field at its reference time')
    except FloatingPointError as exc:
        raise ValueError('compound field overflows at its reference time') from exc
    return deepcopy(model)


def prepare_compound_version(model, now, index, identifier=None):
    """Rebase a diagnostic as a fresh version, never carrying its old evidence.

    Its field predictions and the parent's declared uncertainty are unchanged.
    The declared uncertainty is not a calibrated guarantee.
    No parent posterior, birth index, historical score or diagnostic weight is
    imported into the new active version.
    """
    validate_compound_model(model)
    if (isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now)
            or now < model['reference_time_seconds'] or type(index) is not int or index < 0):
        raise ValueError('causal compound birth time/index required')
    if identifier is not None and (not isinstance(identifier, str) or not identifier):
        raise ValueError('new compound identifier required')
    result = {key: deepcopy(model[key]) for key in ('id', 'family', 'std_db', 'reference_time_seconds',
        'bases', 'peaks_dbm', 'component_modes')}
    for key in ('parent_id', 'lineage_id'):
        if key in model:
            if not isinstance(model[key], str) or not model[key]:
                raise ValueError('compound parent/lineage identifier required')
            result[key] = model[key]
    result['diagnostic_origin_id'] = model['id']
    result['id'] = identifier or model['id']
    dt = float(now)-model['reference_time_seconds']
    for i, base in enumerate(result['bases']):
        base['center_m'] = (np.asarray(base['center_m'])+dt*np.asarray(base['velocity_mps'])).tolist()
        result['peaks_dbm'][i] += dt*base['rate_db_per_second']
    result.update(reference_time_seconds=float(now), born_index=index, scored_points=0,
                  diagnostic_only=False)
    validate_compound_model(result, active=True)
    return result


class HypothesisBank:
    def __init__(self, initial):
        self.history = [point(initial)]
        self.models = [dict(id=name, family='persistence', std_db=sd, born_index=0,
                            scored_points=0, bases=[], peaks_dbm=[])
                       for name, sd in (('persistence', 4.), ('broad_residual', 12.))]
        self.log_weights = np.full(2, -math.log(2.))
        self.pending = None

    def snapshot(self):
        return [dict(deepcopy(m), log_weight=float(w), weight=float(math.exp(w)))
                for m, w in zip(self.models, self.log_weights)]

    def issue(self, position, now):
        if self.pending is not None:
            raise ValueError('consume pending measurement first')
        xy = np.asarray(position, float)
        if (xy.shape != (2,) or not np.all(np.isfinite(xy)) or not math.isfinite(now)
                or now <= self.history[-1]['time_seconds']):
            raise ValueError('strictly future finite query required')
        predictions = []
        for m, log_weight in zip(self.models, self.log_weights):
            mean = (self.history[-1]['interference_dbm'] if m['family'] == 'persistence'
                    else float(power_and_jacobian(kernel(m['bases'], [xy], [now],
                                   m['reference_time_seconds']), m['peaks_dbm'])[0][0]))
            predictions.append(dict(id=m['id'], mean_dbm=mean, std_db=m['std_db'],
                                    log_weight=float(log_weight)))
        weights = np.exp(self.log_weights)
        means = np.array([p['mean_dbm'] for p in predictions])
        mean = float(weights @ means)
        variance = float(weights @ np.array([p['std_db']**2 + (p['mean_dbm']-mean)**2
                                             for p in predictions]))
        self.pending = dict(index=len(self.history), position_m=xy.tolist(), time_seconds=float(now),
                            history_sha256=digest(self.history), predictions=predictions,
                            mean_dbm=mean, std_db=math.sqrt(variance))
        return deepcopy(self.pending)

    def observe(self, measurement):
        measurement = point(measurement)
        if (self.pending is None or measurement['time_seconds'] != self.pending['time_seconds']
                or measurement['position_m'] != self.pending['position_m']):
            raise ValueError('measurement must match pending query exactly once')
        issued = self.pending
        y = measurement['interference_dbm']
        likelihood = np.array([gaussian_logpdf(y, p['mean_dbm'], p['std_db'])
                               for p in issued['predictions']])
        nll = -logsumexp(self.log_weights+likelihood)
        self.log_weights = normalized(self.log_weights+likelihood)
        for m in self.models:
            m['scored_points'] += 1
        self.history.append(measurement)
        self.pending = None
        return dict(issued=deepcopy(issued), observation=measurement, nll=float(nll),
                    squared_error=float((y-issued['mean_dbm'])**2),
                    absolute_error=float(abs(y-issued['mean_dbm'])),
                    innovation_squared=float(((y-issued['mean_dbm'])/issued['std_db'])**2),
                    posterior=self.snapshot())

    def birth(self, candidates):
        if self.pending is not None:
            raise ValueError('cannot change bank after issuing a prediction')
        if not candidates:
            return
        ids = {m['id'] for m in self.models}
        for model in candidates:
            if (model['id'] in ids or model['born_index'] != len(self.history)-1
                    or model['reference_time_seconds'] != self.history[-1]['time_seconds']
                    or not model['bases'] or len(model['bases']) != len(model['peaks_dbm'])
                    or model['scored_points'] != 0 or model['std_db'] != 4.):
                raise ValueError('valid new, unscored hypothesis required')
            if (model['family'] not in FAMILIES
                    or not np.all(np.isfinite(model['peaks_dbm']))
                    or not np.all(np.isfinite(kernel(model['bases'], [[0., 0.]],
                        [model['reference_time_seconds']], model['reference_time_seconds'])))):
                raise ValueError('invalid field hypothesis')
            ids.add(model['id'])
        self.log_weights = np.r_[self.log_weights+math.log(.75),
                                 np.full(len(candidates), math.log(.25/len(candidates)))]
        self.models.extend(deepcopy(candidates))


class EventTrigger:
    def __init__(self, mode):
        if mode not in ('periodic_equal', 'event_equal'):
            raise ValueError('unknown event policy')
        self.mode, self.last_time, self.events = mode, None, 0
        self.innovations = []
        self.checked_time, self.checked_count, self.pending_reason = None, None, None

    def check(self, now, count, innovation_squared):
        if (not math.isfinite(innovation_squared) or innovation_squared < 0
                or not math.isfinite(now) or type(count) is not int
                or (self.checked_time is not None and now <= self.checked_time)
                or (self.checked_count is not None and count != self.checked_count+1)):
            raise ValueError('finite squared innovation required')
        self.checked_time, self.checked_count, self.pending_reason = float(now), count, None
        self.innovations.append(float(innovation_squared))
        if self.events >= 2 or count < 9 or now < 8.:
            return None
        if self.last_time is None:
            self.pending_reason = 'initial_history_ready'
            return self.pending_reason
        elapsed = now-self.last_time
        if elapsed < 8.:
            return None
        if (self.mode == 'event_equal' and len(self.innovations) >= 3
                and np.mean(self.innovations[-3:]) >= 2.25):
            self.pending_reason = 'predictive_mismatch'
        elif elapsed >= 12.:
            self.pending_reason = 'plan_expired'
        return self.pending_reason

    def commit(self, now):
        if (self.pending_reason is None or now != self.checked_time or self.events >= 2
                or (self.last_time is not None and now-self.last_time < 8.)):
            raise ValueError('event budget or minimum interval')
        self.last_time = float(now)
        self.events += 1
        self.pending_reason = None


@dataclass
class Budget:
    reserved_jobs: int = 0
    reserved_calls: int = 0
    reserved_residual_calls: int = 0
    solver_calls: int = 0
    residual_calls: int = 0
    jacobian_calls: int = 0
    dictionary_kernel_terms: int = 0

    def reserve(self, jobs):
        calls = sum(j['max_bases'] for j in jobs)
        if (self.reserved_jobs+len(jobs) > 3 or self.reserved_calls+calls > 6
                or self.reserved_residual_calls+40*calls > 240):
            raise ValueError('atomic inference budget exceeded')
        self.reserved_jobs += len(jobs)
        self.reserved_calls += calls
        self.reserved_residual_calls += 40*calls


def inference_request(bank, public, reason, event_number):
    if bank.pending is not None:
        raise ValueError('request requires fully assimilated prefix')
    now = bank.history[-1]
    context = dict(stage='inference', event_number=event_number, trigger_reason=reason,
        position_m=now['position_m'], time_seconds=now['time_seconds'],
        history=deepcopy(bank.history), history_sha256=digest(bank.history),
        public=deepcopy(public), hypotheses=bank.snapshot(),
        tools=dict(families=list(FAMILIES), regions=list(REGIONS), max_bases=[1, 2]),
        budget=dict(jobs=3, solver_calls=6, residual_calls=240),
        evidence_meaning='relative evidence, not truth probability; fitting history is not validation')
    return dict(context, request_id=digest(context))


def numerical_plan(request):
    return dict(request_id=request['request_id'], stage='inference',
                jobs=[dict(family=f, region='task_disk', max_bases=2) for f in FAMILIES],
                reason='fixed equal allocation; numerical plan, not an LLM response')


def validate_plan(request, response):
    if isinstance(response, str):
        response = json.loads(response, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if not isinstance(response, dict) or set(response) != {'request_id', 'stage', 'jobs', 'reason'}:
        raise ValueError('exact inference plan fields required')
    if response['request_id'] != request['request_id'] or response['stage'] != 'inference':
        raise ValueError('stale or wrong-stage plan')
    if (not isinstance(response['reason'], str) or not 1 <= len(response['reason']) <= 1000
            or not isinstance(response['jobs'], list) or not 1 <= len(response['jobs']) <= 3):
        raise ValueError('bounded reason and jobs required')
    for job in response['jobs']:
        if not isinstance(job, dict) or set(job) != {'family', 'region', 'max_bases'}:
            raise ValueError('exact operator fields required')
        if (job['family'] not in FAMILIES or job['region'] not in REGIONS
                or type(job['max_bases']) is not int or job['max_bases'] not in (1, 2)):
            raise ValueError('unknown operator, region or capacity')
    trial = Budget()
    trial.reserve(response['jobs'])
    return deepcopy(response)


def in_region(center, public, position, region):
    c, origin = np.asarray(center), np.asarray(position)
    bounds = np.asarray(public['map']['bounds_m'])
    delta = c-origin
    if np.any(c < bounds[0]) or np.any(c > bounds[1]) or np.linalg.norm(delta) > 160.+1e-9:
        return False
    goal = np.asarray(public['goal_m'])-origin
    side = float(goal[0]*delta[1]-goal[1]*delta[0])
    return region == 'task_disk' or (side >= -1e-9 if region == 'left_half' else side <= 1e-9)


def operator_dictionary(job, public, position):
    velocities = ((0., 0.),) if job['family'] != 'translate' else ((3., 0.), (-3., 0.), (0., 3.), (0., -3.))
    rates = (-.5, .5) if job['family'] == 'strength_change' else (0.,)
    centers = [np.asarray(position)+[x, y] for y in (-160., -80., 0., 80., 160.)
               for x in (-160., -80., 0., 80., 160.)
               if in_region(np.asarray(position)+[x, y], public, position, job['region'])]
    return [dict(center_m=c.tolist(), scale_m=s, velocity_mps=list(v), rate_db_per_second=r)
            for c in centers for s in (40., 80., 160.) for v in velocities for r in rates]


def fit_job(bank, public, job, budget, identifier, *, soft=None):
    history = bank.history[-12:]
    now, position = history[-1]['time_seconds'], history[-1]['position_m']
    xy, times = [p['position_m'] for p in history], [p['time_seconds'] for p in history]
    labels = np.array([p['interference_dbm'] for p in history])
    upper = float(labels.max()+6.)
    if upper <= -140.:
        raise ValueError('observations outside amplitude bound')
    bases = operator_dictionary(job, public, position)
    warm, warm_peaks = [], []
    old = sorted([(float(w), m) for w, m in zip(bank.log_weights, bank.models)
                  if m['family'] == job['family'] and m['bases']], key=lambda p: -p[0])
    if old:
        model = old[0][1]
        dt = now-model['reference_time_seconds']
        for b, peak in zip(model['bases'], model['peaks_dbm']):
            b = deepcopy(b)
            b['center_m'] = (np.asarray(b['center_m'])+dt*np.asarray(b['velocity_mps'])).tolist()
            if in_region(b['center_m'], public, position, job['region']):
                warm.append(b)
                warm_peaks.append(float(np.clip(peak+dt*b['rate_db_per_second'], -140.+1e-8, upper-1e-8)))
            if len(warm) == job['max_bases']:
                break
    bases = warm+bases
    point_count = len(labels)
    has_soft = soft is not None and bool(soft['position_m'])
    scales = np.full(point_count, 1./math.sqrt(point_count))
    if has_soft:
        if soft['prefix_end_seconds'] != now or soft['independent_evidence_points'] != 0:
            raise ValueError('GP constraints must share exactly this fit prefix')
        weights=np.asarray(soft['weights'],float)
        if np.any(weights<0) or not math.isclose(float(weights.sum()),1.,abs_tol=1e-10):
            raise ValueError('normalized nonnegative soft weights required')
        xy = xy+soft['position_m']
        times = times+soft['time_seconds']
        labels = np.r_[labels, soft['mean_dbm']]
        scales = np.r_[scales, np.sqrt(soft['strength']*np.asarray(soft['weights']))]
    matrix = kernel(bases, xy, times, now)
    budget.dictionary_kernel_terms += int(matrix.size)
    chosen, peaks, traces = [], [], []
    prediction = np.full(len(labels), -100.)
    loss = float(np.sum(((prediction-labels)*scales)**2))
    started_calls, started_residuals = budget.solver_calls, budget.residual_calls
    job_call_limit = min(job['max_bases'], budget.reserved_calls-started_calls)
    job_residual_limit = min(40*job['max_bases'], budget.reserved_residual_calls-started_residuals)
    if job_call_limit <= 0 or job_residual_limit <= 0:
        raise ValueError('unreserved fitting budget')
    selected_branch, selected_trace, stop_reason = 'background', None, None
    warm_candidate = None
    if warm:
        # The old version is an optional, freshly evaluated candidate, not a
        # compulsory set of slots. Charge its prefix objective to the same job.
        budget.residual_calls += 1
        warm_prediction = power_and_jacobian(matrix[:, :len(warm)], warm_peaks)[0]
        warm_loss = float(np.sum(((warm_prediction-labels)*scales)**2))
        eligible = bool(np.all(np.isfinite(warm_peaks)) and np.all(np.isfinite(warm_prediction))
                        and np.all(np.asarray(warm_peaks) >= -140.) and np.all(np.asarray(warm_peaks) <= upper))
        retained = bool(eligible and warm_loss <= loss+1e-8)
        warm_candidate = dict(columns=list(range(len(warm))), peaks_dbm=list(warm_peaks),
            candidate_mse=warm_loss, eligible=eligible, accepted=retained,
            residual_calls=1, solver_calls=0, inherited_scored_points=0)
        if retained:
            chosen, peaks, prediction, loss = list(range(len(warm))), list(warm_peaks), warm_prediction, warm_loss
            selected_branch = 'warm_seed'

    # Cold greedy growth has its own incumbent: a better warm seed must not
    # prevent the cold branch from building a useful two-basis alternative.
    cold_columns, cold_peaks = [], []
    cold_prediction = np.full(len(labels), -100.)
    cold_loss = float(np.sum(((cold_prediction-labels)*scales)**2))
    for step in range(job['max_bases']):
        remaining_residuals = job_residual_limit-(budget.residual_calls-started_residuals)
        if budget.solver_calls-started_calls >= job_call_limit or remaining_residuals <= 0:
            stop_reason = 'job_fitting_budget_exhausted'
            break
        if len(cold_columns) >= job['max_bases'] or not len(bases):
            break
        derivatives = matrix/10.**((cold_prediction+80.)/10.)[:, None]
        norms = np.linalg.norm(derivatives*scales[:, None], axis=0)
        gradient = (derivatives*scales[:, None]).T@((labels-cold_prediction)*scales)
        scores = np.divide(gradient, norms, out=np.full(len(bases), -np.inf), where=norms > 1e-12)
        # Old off-grid geometry competes on equal terms; it can be selected for
        # amplitude refitting, but never occupies a slot by obligation.
        scores[cold_columns] = -np.inf
        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]) or scores[index] <= 0:
            stop_reason = 'no_positive_dictionary_direction'
            break
        residual = 10.**((labels+80.)/10.)-10.**((cold_prediction+80.)/10.)
        a = matrix[:, index]
        amplitude = max(1e-6, float((a*scales)@(residual*scales))/max(float((a*scales)@(a*scales)), 1e-30))
        initial = np.r_[cold_peaks, np.clip(-80.+10.*math.log10(amplitude), -140.+1e-8, upper-1e-8)]
        columns = cold_columns+[index]
        selected = matrix[:, columns]
        budget.solver_calls += 1
        residual_calls = jacobian_calls = 0
        evaluation_limit = min(40, remaining_residuals)
        best_evaluated = None

        def objective(p):
            nonlocal residual_calls, best_evaluated
            if (residual_calls >= evaluation_limit or budget.residual_calls >= budget.reserved_residual_calls
                    or budget.residual_calls-started_residuals >= job_residual_limit):
                raise ValueError('residual budget exhausted')
            residual_calls += 1
            budget.residual_calls += 1
            evaluated_prediction = power_and_jacobian(selected, p)[0]
            residual = (evaluated_prediction-labels)*scales
            evaluated_loss = float(residual@residual)
            if (np.all(np.isfinite(p)) and np.all(np.isfinite(evaluated_prediction))
                    and np.all(p >= -140.) and np.all(p <= upper)
                    and (best_evaluated is None or evaluated_loss < best_evaluated['loss'])):
                best_evaluated = dict(peaks=np.array(p, copy=True), prediction=evaluated_prediction.copy(), loss=evaluated_loss)
            return residual

        def jacobian(p):
            nonlocal jacobian_calls
            if jacobian_calls >= 40:
                raise ValueError('Jacobian budget exhausted')
            jacobian_calls += 1
            budget.jacobian_calls += 1
            return power_and_jacobian(selected, p)[1]*scales[:, None]

        report = dict(branch='cold_dictionary', columns=columns, initial_peaks_dbm=initial.tolist(),
            previous_mse=cold_loss, solver_success=False, solver_status=None, solver_message=None,
            nfev=0, njev=0, optimality=None)
        try:
            solved = least_squares(objective, initial, jac=jacobian, bounds=(-140., upper),
                                   max_nfev=evaluation_limit, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            report.update(solver_success=bool(solved.success), solver_status=int(solved.status),
                solver_message=str(solved.message), nfev=int(solved.nfev), njev=int(solved.njev),
                optimality=float(solved.optimality) if math.isfinite(solved.optimality) else None)
        except (ValueError, FloatingPointError) as exc:
            report.update(error=str(exc), nfev=residual_calls, njev=jacobian_calls)
        accepted = bool(best_evaluated is not None and best_evaluated['loss'] <= cold_loss+1e-8)
        report.update(peaks_dbm=best_evaluated['peaks'].tolist() if best_evaluated is not None else None,
            candidate_mse=best_evaluated['loss'] if best_evaluated is not None else None,
            accepted=accepted, residual_calls=residual_calls, jacobian_calls=jacobian_calls,
            selection_reason='best_evaluated_legal' if accepted else 'no_legal_improvement')
        traces.append(report)
        if not accepted:
            stop_reason = report.get('error', 'no_legal_improvement')
            break
        cold_columns, cold_peaks = columns, best_evaluated['peaks'].tolist()
        cold_prediction, cold_loss = best_evaluated['prediction'], best_evaluated['loss']
        if cold_loss < loss-1e-8 or (not chosen and cold_loss <= loss+1e-8):
            chosen, peaks, prediction, loss = cold_columns, cold_peaks, cold_prediction, cold_loss
            selected_branch, selected_trace = 'cold_dictionary', len(traces)-1
        if report.get('error'):
            stop_reason = report['error']
            break
    for i, row in enumerate(traces):
        row['selected_for_return'] = i == selected_trace
    if warm_candidate is not None:
        warm_candidate['selected_for_return'] = selected_branch == 'warm_seed'
    model = dict(id=identifier, family=job['family'], std_db=4., born_index=len(bank.history)-1,
                 scored_points=0, reference_time_seconds=now, bases=[bases[i] for i in chosen],
                 peaks_dbm=peaks, fit_mse=float(np.mean((prediction[:point_count]-labels[:point_count])**2)))
    return dict(job=deepcopy(job), dictionary=bases, window=deepcopy(history), model=model,
                bounds_dbm=[-140., upper], trace=traces,
                warm_start_bases=len(warm), warm_candidate=warm_candidate,
                selection_policy='warm_candidate_open_dictionary_v2', selected_branch=selected_branch,
                stop_reason=stop_reason,
                job_resources=dict(solver_calls=budget.solver_calls-started_calls,
                    residual_calls=budget.residual_calls-started_residuals,
                    solver_call_limit=job_call_limit, residual_call_limit=job_residual_limit),
                training_error_is_predictive_evidence=False,
                fit_objective=loss, gp_soft_constraints=deepcopy(soft),
                gp_dictionary_kernel_terms=len(bases)*(len(labels)-point_count),
                gp_shape_mse=float(np.sum(np.asarray(soft['weights'])*(prediction[point_count:]-labels[point_count:])**2))
                    if soft is not None and soft['position_m'] else None)



def execute_plan(bank, request, response, public, identifier):
    # Recompute the request against live state: an otherwise valid old response is stale.
    live = inference_request(bank, public, request['trigger_reason'], request['event_number'])
    if live['request_id'] != request['request_id']:
        raise ValueError('request no longer matches live bank state')
    error = None
    try:
        accepted = validate_plan(request, response)
    except (ValueError, TypeError, KeyError) as exc:
        error = str(exc)
        accepted = numerical_plan(request)
    budget = Budget()
    budget.reserve(accepted['jobs'])
    results, candidates = [], []
    for i, job in enumerate(accepted['jobs']):
        try:
            result = fit_job(bank, public, job, budget, f'{identifier}_job{i}')
            results.append(result)
            if result['model']['bases']:
                candidates.append(result['model'])
        except (ValueError, FloatingPointError) as exc:
            results.append(dict(job=job, error=str(exc)))
    before = bank.snapshot()
    bank.birth(candidates)
    return dict(raw_response=deepcopy(response), accepted_plan=accepted,
                parser_accepted=error is None, fallback_error=error,
                backend='numerical_fixed', actual_llm_response=False,
                resources=asdict(budget), fits=results, bank_before_birth=before,
                bank_after_birth=bank.snapshot(), new_hypotheses=len(candidates))
