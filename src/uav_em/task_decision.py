"""Bounded source decisions bound to one public task snapshot.

This component has no simulator, sensor, truth, flight or LLM backend. An
external advisor may organize exactly the same tools as the numerical policy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

from .event_inference import Budget, FAMILIES, digest, fit_job, inference_request, validate_plan
from .event_llm_inference import family_evidence, pilot_feedback, ranked_families, rebase_model, state_digest
from .source_geometry import GeometrySettings, constrained_refine_fit
from .source_revision import SourceOnlyBank
from .task_context import validate_task_context


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError('nonfinite JSON')


def validate_response(request, raw):
    """Validate only finite, exact-schema tool organization, never parameters."""
    if digest({k: v for k, v in request.items() if k != 'request_id'}) != request['request_id']:
        raise ValueError('request content changed')
    if isinstance(raw, str):
        raw = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    if not isinstance(raw, dict) or set(raw) != {'request_id', 'action', 'jobs', 'geometry_job', 'reason'}:
        raise ValueError('exact task decision fields required')
    # Also reject non-JSON Python objects and nonfinite dict inputs.
    try:
        digest(raw)
    except (TypeError, OverflowError) as exc:
        raise ValueError('finite JSON decision required') from exc
    if raw['request_id'] != request['request_id']:
        raise ValueError('stale task decision')
    if not isinstance(raw['reason'], str) or not 1 <= len(raw['reason']) <= 1000:
        raise ValueError('bounded reason required')
    if raw['action'] == 'keep':
        if raw['jobs'] != [] or raw['geometry_job'] is not None:
            raise ValueError('keep requires no jobs or geometry')
        return deepcopy(raw)
    if raw['action'] != 'propose':
        raise ValueError('unknown task decision action')
    projected = dict(request_id=raw['request_id'], stage='inference', jobs=raw['jobs'], reason=raw['reason'])
    validate_plan(request, projected)
    if len(raw['jobs']) != request['required_jobs']:
        raise ValueError('wrong number of jobs in round')
    geometry = raw['geometry_job']
    if geometry is not None and (request['phase'] != 'refine' or type(geometry) is not int or geometry not in (0, 1)):
        raise ValueError('geometry requires one refine job index')
    return deepcopy(raw)


def numeric_response(request):
    """Organize the shared tools using finite task disagreement and old evidence."""
    evidence_order = ranked_families(request['family_evidence'])
    priorities = request['task_context']['family_priorities']
    order = sorted(FAMILIES, key=lambda f: (-priorities[f], evidence_order.index(f)))
    if request['phase'] == 'pilot':
        families = order[:1]
    else:
        feedback = request['feedback']
        pilot = feedback['job']['family']
        if feedback['tail_rmse_db'] is not None and feedback['tail_rmse_db'] > 4.:
            families = [pilot, next(f for f in order if f != pilot)]
        else:
            families = [f for f in order if f != pilot]
    return dict(request_id=request['request_id'], action='propose',
        jobs=[dict(family=f, region='task_disk', max_bases=2) for f in families],
        geometry_job=0 if request['phase'] == 'refine' else None,
        reason='numerical task disagreement and birth-after evidence; known-tail feedback is retrospective')


def _check_context(bank, public, context):
    if not isinstance(bank, SourceOnlyBank) or any(m['family'] == 'persistence' for m in bank.models):
        raise ValueError('task decisions require a source-only bank without persistence')
    if (not isinstance(public, dict) or set(public) != {'goal_m', 'map'}
            or not isinstance(public['map'], dict) or set(public['map']) != {'bounds_m'}):
        raise ValueError('exact public goal and bounds fields required')
    if bank.pending is not None:
        raise ValueError('task decisions require fully assimilated history')
    validate_task_context(context, history=bank.history, models=bank.snapshot())
    if (context['bounds_m'] != public['map']['bounds_m']
            or context['state']['goal_m'] != public['goal_m']
            or context['state']['position_m'] != bank.history[-1]['position_m']):
        raise ValueError('task context does not match public bounds, goal or receiver')


def _resources(budget, refinement):
    result = asdict(budget)
    for key in ('solver_calls', 'residual_calls', 'jacobian_calls', 'reserved_calls', 'reserved_residual_calls'):
        result['base_' + key] = result[key]
    for key in ('solver_calls', 'residual_calls', 'jacobian_calls'):
        result[key] += refinement[key]
    result['reserved_calls'] += refinement['calls']
    result['reserved_residual_calls'] += 40 * refinement['calls']
    result['geometry_reserved_jacobian_calls'] = 40 * refinement['calls']
    result['refinement'] = deepcopy(refinement)
    return result


def _observed_residuals(bank, records):
    sources = {m['id']: m['family'] for m in bank.models if m['family'] in FAMILIES}
    result = []
    for row in records[-6:]:
        observation, issued = row['observation'], row['issued']
        index = issued['index']
        if (type(index) is not int or not 1 <= index < len(bank.history)
                or observation != bank.history[index]
                or issued['position_m'] != observation['position_m']
                or issued['time_seconds'] != observation['time_seconds']
                or issued['history_sha256'] != digest(bank.history[:index])):
            raise ValueError('residual diagnostics require matching preissued observed predictions')
        result.append(dict(position_m=deepcopy(observation['position_m']),
            observed_time_seconds=observation['time_seconds'],
            preissued_time_seconds=issued['time_seconds'],
            observed_dbm=observation['interference_dbm'],
            predictions=[dict(id=p['id'], family=sources[p['id']], mean_dbm=p['mean_dbm'],
                std_db=p['std_db'], residual_db=observation['interference_dbm']-p['mean_dbm'])
                for p in issued['predictions'] if p['id'] in sources],
            preissued=True, calibrated=False, fitting_residual=False))
    return result


def _make_request(bank, public, event, phase, evidence, observed_residuals, feedback, budget, refinement,
                  task_context, soft, trigger_reason):
    request = deepcopy(inference_request(bank, public, trigger_reason or (
        'common_initial' if event == 0 else 'common_revision'), event))
    request.pop('request_id')
    request.update(schema='task_source_event_v1', phase=phase,
        required_jobs=1 if phase == 'pilot' else 2,
        live_state_sha256=state_digest(bank, public), family_evidence=deepcopy(evidence),
        latest_observation_residuals=deepcopy(observed_residuals),
        task_snapshot_id=task_context['task_snapshot_id'], task_context=deepcopy(task_context),
        pilot_training_cutoff_seconds=bank.history[-4]['time_seconds'],
        pilot_training_anchor_m=deepcopy(bank.history[-4]['position_m']),
        feedback=deepcopy(feedback), spent_resources=_resources(budget, refinement),
        geometry_tool=dict(name='constrained_refine_fit', acceptance_policy='best_feasible_v2',
            settings=asdict(GeometrySettings()), maximum_calls=1, available_in='refine'),
        decision_contract=dict(actions=['propose', 'keep'],
            keep='retain the active pool; do not commit any event candidate',
            new_version_scored_points=0, fitting_and_known_tail_are_new_evidence=False))
    if soft is not None:
        request['gp_soft_constraints'] = deepcopy(soft)
    request['request_id'] = digest(request)
    return request


def run_task_event(bank, public, event, records, method, advisor=None, emit=None, *,
                   task_context, soft_context=None, trigger_reason=None):
    """Fit at most three candidates, then commit once unless either round keeps.

    A pilot candidate is diagnostic until a refine proposal completes. Keeping
    in either round retains the entire active source pool without a new birth.
    """
    if method not in ('feedback', 'llm'):
        raise ValueError('task decision method must be feedback or llm')
    if method == 'llm' and advisor is None:
        raise ValueError('LLM task decisions require an explicit advisor')
    if len(bank.history) < 4:
        raise ValueError('pilot requires at least four known observations')
    _check_context(bank, public, task_context)
    # Evidence is the observed causal suffix, never a future or retrospective
    # fitted prediction. Its immutable identity also covers advisor callbacks.
    if records and [r['observation'] for r in records] != bank.history[-len(records):]:
        raise ValueError('decision evidence must match the observed history suffix')
    original_context = digest(task_context)
    original_state = state_digest(bank, public)
    original_records = digest(records)

    def check_live():
        if (bank.pending is not None or original_state != state_digest(bank, public)
                or original_records != digest(records)):
            raise ValueError('live state advanced or changed while waiting; reject before fitting')
        if digest(task_context) != original_context:
            raise ValueError('task context changed while waiting; reject before fitting')
        _check_context(bank, public, task_context)

    budget, rounds, candidates = Budget(), [], []
    refinement = dict(calls=0, solver_calls=0, residual_calls=0, jacobian_calls=0,
                      kernel_terms=0, accounting='actual_reported', exceptional_calls=0)
    evidence, feedback, kept = family_evidence(bank, records), None, False
    observed_residuals = _observed_residuals(bank, records)
    before_birth = bank.snapshot()
    for phase in ('pilot', 'refine'):
        check_live()
        source_history = bank.history[:-3] if phase == 'pilot' else bank.history
        if soft_context is not None:
            from .joint_prediction import soft_constraints
            soft = soft_constraints(source_history, **soft_context)
        else:
            soft = None
        request = _make_request(bank, public, event, phase, evidence, observed_residuals, feedback, budget,
                                refinement, task_context, soft, trigger_reason)
        if method == 'llm':
            try:
                envelope = deepcopy(advisor(deepcopy(request)))
            except TimeoutError as exc:
                envelope = dict(raw=None, timely=False, actual_llm_request=None,
                    invocation_accounting='unknown_after_advisor_exception', advisor_error=str(exc))
            if not isinstance(envelope, dict):
                envelope = dict(raw=None, timely=None, actual_llm_request=None,
                    original_envelope=deepcopy(envelope),
                    invocation_accounting='invalid_envelope_invocation_unknown')
            if envelope.get('actual_llm_request') is None:
                envelope.setdefault('actual_llm_request', None)
                envelope.setdefault('invocation_accounting', 'not_reported_by_advisor')
        else:
            envelope = dict(raw=numeric_response(request), timely=True,
                            actual_llm_request=False, wall_seconds=0., origin='numerical_policy')
        check_live()
        error = None
        try:
            if not isinstance(envelope, dict) or type(envelope.get('timely')) is not bool:
                raise ValueError('advisor envelope requires explicit timeliness')
            if not envelope['timely']:
                raise ValueError('response timeout')
            plan = validate_response(request, envelope['raw'])
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            plan = numeric_response(request)
        fits = []
        if plan['action'] == 'keep':
            kept = True
        else:
            budget.reserve(plan['jobs'])
            for j, job in enumerate(plan['jobs']):
                check_live()
                source_bank = (SimpleNamespace(history=bank.history[:-3], models=[], log_weights=[])
                               if phase == 'pilot' else bank)
                resources_before = _resources(budget, refinement)
                try:
                    fit = fit_job(source_bank, public, job, budget,
                                  f'event{event}_{phase}_job{j}', soft=soft)
                except (ValueError, FloatingPointError) as exc:
                    fits.append(dict(job=deepcopy(job), error=str(exc),
                        resources_before=resources_before, resources_after=_resources(budget, refinement)))
                    continue
                if phase == 'refine' and plan['geometry_job'] == j:
                    if fit['model']['bases']:
                        if refinement['calls']:
                            raise ValueError('geometry call budget exceeded')
                        refinement['calls'] += 1
                        try:
                            fit = constrained_refine_fit(fit, public, acceptance_policy='best_feasible_v2')
                        except (ValueError, FloatingPointError, RuntimeError) as exc:
                            # Preserve the legal basic fit. A tool exception has
                            # no trustworthy actual-work report, so charge its
                            # full finite allowance and label it explicitly.
                            fit = deepcopy(fit)
                            fit['geometry_refinement'] = dict(accepted=False, error=str(exc),
                                stop_reason='exception_preserved_basic_fit',
                                acceptance_policy='best_feasible_v2',
                                resources=dict(solver_calls=1, residual_calls=40,
                                               jacobian_calls=40, kernel_terms=5000),
                                resource_accounting='conservative_allowance_after_exception')
                            refinement['exceptional_calls'] += 1
                            refinement['accounting'] = 'includes_conservative_exception_allowance'
                        for key in ('solver_calls', 'residual_calls', 'jacobian_calls', 'kernel_terms'):
                            refinement[key] += fit['geometry_refinement']['resources'][key]
                    else:
                        fit['geometry_refinement'] = dict(accepted=False,
                            stop_reason='no_source_basis_no_call', resources=dict(
                                solver_calls=0, residual_calls=0, jacobian_calls=0, kernel_terms=0))
                fits.append(fit)
                if fit['model']['bases']:
                    candidates.append(rebase_model(fit['model'], bank.history[-1]['time_seconds'], len(bank.history)-1))
        if phase == 'pilot' and fits:
            feedback = pilot_feedback(fits[0], bank)
            feedback.update(retrospective_only=True, independent_future_evidence=False)
        result = dict(request=request, envelope=deepcopy(envelope), accepted_plan=plan,
            parser_accepted=error is None, fallback_error=error, fits=fits,
            resources_after=_resources(budget, refinement),
            pilot_feedback=deepcopy(feedback) if phase == 'pilot' else None)
        rounds.append(result)
        if emit:
            emit(phase, deepcopy(result))
        if kept:
            break
    check_live()
    committed = [] if kept else candidates
    if committed:
        bank.birth(committed)
    return dict(index=event, time_seconds=bank.history[-1]['time_seconds'], rounds=rounds,
        task_snapshot_id=task_context['task_snapshot_id'], bank_before_birth=before_birth,
        bank_after_birth=bank.snapshot(), committed_models=committed,
        pool_committed=bool(committed), kept_active_pool=kept,
        diagnostic_candidates=deepcopy(candidates), resources=_resources(budget, refinement),
        new_hypotheses=len(committed), independent_future_evidence=False)
