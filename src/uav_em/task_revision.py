"""Explicit compound task events; synthetic integration, no experiment runner.

The existing dictionary event bootstraps unknowns. Subsequent events organize
two bounded fits, retain their diagnostics, and submit fresh versions once.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math

import numpy as np

from . import feedback_revision as tool
from .candidate_ledger import CandidateLedger, choose_option, public_diagnostics
from .equivalent_field import kernel, normalized_expected_power, power_and_jacobian
from .event_inference import digest, prepare_compound_version
from .event_llm_inference import state_digest
from .feedback_organization import COUNTERS, _priority
from .source_revision import CompoundSourceBank
from .task_context import COMPOUND_VERSION, validate_task_context


VERSION = 'compound-task-event-v1'


@dataclass(frozen=True)
class TaskRevisionSettings:
    maximum_jobs: int = 2
    maximum_worlds: int = 8
    common_kernel_limit: int = 40000
    birth_mass: float = .25

    def __post_init__(self):
        for name, cap in [('maximum_jobs', 2), ('maximum_worlds', 8), ('common_kernel_limit', 40000)]:
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError('explicit bounded compound settings required')
        if isinstance(self.birth_mass, bool) or not math.isfinite(self.birth_mass) or not 0 < self.birth_mass < 1:
            raise ValueError('explicit development cohort prior required')


def _bind(value):
    result = deepcopy(value)
    result['request_id'] = digest(result)
    return result


def _anchor(context, models):
    near = context['near_competitor']
    if near is None:
        return None
    routes = [next(c for c in context['candidates'] if c['id'] == near[key]) for key in ('route_a', 'route_b')]
    anchor = dict(time_seconds=context['time_seconds'], history_sha256=context['history_sha256'],
        baseline_delta_c=near['delta_c_seconds'],
        worlds={m['id']: dict(weight=m['weight'], seed_delta_q=(
            routes[1]['score']['scenario_scores'][index]['power_proxy_seconds']
            -routes[0]['score']['scenario_scores'][index]['power_proxy_seconds'])) for index, m in enumerate(models)})
    for name, route in zip(('route_a', 'route_b'), routes):
        anchor[name] = dict(id=route['id'], queries=deepcopy(route['score']['queries']),
            gp_weight=deepcopy(route['score']['gp_weight']), gp_fixed_power=None)
    return anchor


def build_common(context, models, history, soft, settings):
    """Reconstruct cached GP constants and pay for complete original analyses.

    Only cached A/B arrivals are used. A complete world's reconstruction and
    fit/task derivatives must fit the shared budget before any of it is spent.
    """
    validate_task_context(context, history=history, models=models)
    if context['version'] != COMPOUND_VERSION or context['planner_method'] != 'joint':
        raise ValueError('explicit compound joint task context required')
    anchor = _anchor(context, models)
    resources = dict(kernel_terms=0, cache_reconstruction_kernel_terms=0, analysis_kernel_terms=0,
        worlds=0, new_gp_queries=0, new_route_queries=0)
    common = dict(anchor_cache=anchor, options=[], excluded_targets=[], analyses={}, resources=resources)
    if anchor is None:
        common['stop_reason'] = 'no_distinct_common_competitor'
        return common
    count = sum(len(anchor[key]['queries']['time_seconds']) for key in ('route_a', 'route_b'))
    for seed in sorted(models, key=lambda m: (-m['weight'], m['id'])):
        reason = None
        try:
            modes = tool.mechanism_options(seed)
            basis_count = len(seed['bases'])
            reconstruction = basis_count*count
            estimate = basis_count*(len(history[-12:])+len((soft or {}).get('time_seconds', []))+count)
            if seed['weight'] <= 0:
                reason = 'zero_weight'
            elif resources['worlds'] >= settings.maximum_worlds:
                reason = 'common_world_limit'
            elif 2*reconstruction > tool.MAX_KERNEL_TERMS:
                reason = 'complete_feedback_kernel_limit'
            elif resources['kernel_terms']+reconstruction+estimate > settings.common_kernel_limit:
                reason = 'common_kernel_limit'
        except (ValueError, KeyError, TypeError) as exc:
            reason = 'ineligible_seed: '+str(exc)
        if reason is not None:
            common['excluded_targets'].append(dict(target_id=seed['id'], reason=reason))
            continue
        index = next(i for i, m in enumerate(models) if m['id'] == seed['id'])
        for key in ('route_a', 'route_b'):
            route = anchor[key]; queries = route['queries']
            score = next(c['score'] for c in context['candidates'] if c['id'] == route['id'])
            mean = power_and_jacobian(kernel(seed['bases'], queries['position_m'], queries['time_seconds'],
                seed['reference_time_seconds']), seed['peaks_dbm'])[0]
            source_power = normalized_expected_power(mean, seed['std_db'], -80.)
            fixed = np.asarray(score['expected_normalized_power'][index])-(1-np.asarray(route['gp_weight']))*source_power
            if not np.all(np.isfinite(fixed)) or np.any(fixed < -1e-8):
                raise ValueError('invalid cached joint power reconstruction')
            fixed = np.maximum(0., fixed)  # only floating-point cancellation at zero
            if route['gp_fixed_power'] is None:
                route['gp_fixed_power'] = fixed.tolist()
            elif not np.allclose(fixed, route['gp_fixed_power'], rtol=1e-7, atol=1e-8):
                raise ValueError('cached GP constants differ between source worlds')
        resources['cache_reconstruction_kernel_terms'] += reconstruction
        resources['kernel_terms'] += reconstruction
        analysis = tool.analyze_seed(seed, history, soft, anchor)
        actual = analysis['resources']['kernel_terms']
        if actual != estimate:
            raise ValueError('complete-world kernel allowance changed')
        resources['analysis_kernel_terms'] += actual
        resources['kernel_terms'] += actual
        resources['worlds'] += 1
        common['analyses'][seed['id']] = analysis
        for mask in modes:
            ambiguity, priority = _priority(analysis, mask, seed['weight'], anchor['baseline_delta_c'])
            common['options'].append(dict(target_id=seed['id'], component_modes=mask,
                local_ambiguity_seconds=ambiguity, priority=priority))
    common['options'].sort(key=lambda o: (-o['priority'], o['target_id'], tuple(o['component_modes'])))
    common['stop_reason'] = None if common['options'] else 'no_eligible_complete_world'
    common['common_id'] = digest(common)
    return common


def numeric_response(request):
    chosen = choose_option(request['common']['options'], request['coverage'],
        continuations=[request['continuation']] if request['continuation'] is not None else [],
        remaining_jobs=request['remaining_jobs'])
    return dict(request_id=request['request_id'], action='fit' if chosen else 'stop',
        target_id=chosen['target_id'] if chosen else None,
        component_modes=chosen['component_modes'] if chosen else [], start=chosen['start'] if chosen else None,
        submit_ids=['$frontier'] if chosen else None,
        reason=chosen['reason'] if chosen else 'stop computation and preserve explicit previous selection')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def validate_response(request, raw):
    if request['request_id'] != digest({k: v for k, v in request.items() if k != 'request_id'}):
        raise ValueError('compound request changed')
    if isinstance(raw, str):
        raw = json.loads(raw, object_pairs_hook=_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if not isinstance(raw, dict) or set(raw) != {'request_id', 'action', 'target_id', 'component_modes', 'start', 'submit_ids', 'reason'}:
        raise ValueError('exact compound reply fields required')
    digest(raw)
    if raw['request_id'] != request['request_id'] or not isinstance(raw['reason'], str) or not 1 <= len(raw['reason']) <= 1000:
        raise ValueError('bound compound reply and bounded reason required')
    if raw['action'] == 'stop':
        if raw['target_id'] is not None or raw['component_modes'] != [] or raw['start'] is not None:
            raise ValueError('stop requires no fitting job')
    elif raw['action'] == 'fit':
        if not any(o['target_id'] == raw['target_id'] and o['component_modes'] == raw['component_modes'] for o in request['common']['options']):
            raise ValueError('job outside shared options')
        seen = any(o['target_id'] == raw['target_id'] and o['component_modes'] == raw['component_modes'] for o in request['coverage'])
        if raw['start'] == 'original':
            if seen:
                raise ValueError('original job already evaluated')
        elif raw['start'] == 'previous':
            previous = request['continuation']
            if previous is None or previous['target_id'] != raw['target_id'] or previous['component_modes'] != raw['component_modes']:
                raise ValueError('continuation requires the previous legal candidate')
        else:
            raise ValueError('invalid fitting start')
    else:
        raise ValueError('invalid compound action')
    selected = raw['submit_ids']
    if selected is not None:
        if not isinstance(selected, list) or len(selected) > request['maximum_submit_models'] or any(not isinstance(x, str) for x in selected):
            raise ValueError('bounded explicit candidate selection required')
        if len(set(selected)) != len(selected):
            raise ValueError('duplicate selection')
        known = {entry['candidate_id'] for entry in request['ledger']['entries']}
        if selected == ['$frontier']:
            pass
        elif any(x not in known and not (x == '$this' and raw['action'] == 'fit') for x in selected):
            raise ValueError('selection outside legal ledger')
    return deepcopy(raw)


def _add(total, part):
    for key in COUNTERS:
        value = part.get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError('nonnegative actual fitting/feedback accounting required')
        total[key] += value


def _unique_versions(bank, candidates):
    """Deduplicate only exact physical clones, without discarding other modes."""
    signatures = {bank._field_signature(m) for m in bank.models if m['bases']}
    unique, skipped = [], []
    for model in candidates:
        signature = bank._field_signature(model)
        if signature in signatures:
            skipped.append(dict(id=model['id'], reason='same_physical_field_no_additional_birth_mass'))
        else:
            unique.append(model)
            signatures.add(signature)
    return unique, skipped


def run_revision_event(bank, public, event, records, method, advisor=None, emit=None, *,
                       task_context, settings, soft_context=None, trigger_reason=None):
    """Two computation rounds, an explicit selection, and one atomic birth.

    Selection survives later malformed replies/tool failures. All fit history
    and task feedback remains diagnostic; only subsequent readings score births.
    """
    if (not isinstance(settings, TaskRevisionSettings) or not isinstance(bank, CompoundSourceBank)
            or bank.birth_mass != settings.birth_mass):
        raise ValueError('compound bank and explicit prior settings required')
    if method not in ('feedback', 'llm') or method == 'llm' and advisor is None:
        raise ValueError('explicit shared organization/advisor required')
    if bank.pending is not None or len(bank.history) < 4:
        raise ValueError('fully observed compound history required')
    validate_task_context(task_context, history=bank.history, models=bank.snapshot())
    if task_context['version'] != COMPOUND_VERSION or task_context['planner_method'] != 'joint':
        raise ValueError('explicit compound joint task context required')
    policy = (bank.birth_mass, bank.max_models, bank.max_birth_models)

    def same_policy():
        return (bank.birth_mass, bank.max_models, bank.max_birth_models) == policy
    if not any(m['bases'] for m in bank.models):
        from .task_decision import run_task_event
        from .source_revision import SourceOnlyBank
        # One paid legacy bootstrap, retaining every diagnostic fit. The old
        # bank's provisional birth is isolated from the active new bank; exact
        # duplicate fitted fields cannot reject the other legal bootstrap fits.
        temporary = SourceOnlyBank(bank.history[0])
        temporary.history = deepcopy(bank.history)
        temporary.models, temporary.log_weights = deepcopy(bank.models), bank.log_weights.copy()
        original = state_digest(bank, public)
        def checked_advisor(request):
            try:
                return advisor(request)
            finally:
                if state_digest(bank, public) != original or bank.pending is not None or not same_policy():
                    raise ValueError('active bootstrap state changed while waiting')
        result = run_task_event(temporary, public, event, records, method,
            checked_advisor if advisor is not None else None, emit,
            task_context=task_context, soft_context=soft_context, trigger_reason=trigger_reason)
        if state_digest(bank, public) != original or bank.pending is not None or not same_policy():
            raise ValueError('active bootstrap state changed while fitting')
        selected, skipped = _unique_versions(bank, result['committed_models'])
        if selected:
            bank.birth(selected)
        result['bootstrap_explicit_keep'] = result['kept_active_pool']
        result.update(committed_models=selected, new_hypotheses=len(selected),
            bank_after_birth=bank.snapshot(), pool_committed=bool(selected),
            kept_active_pool=not bool(selected), skipped_clones=skipped)
        result['resources']['soft_gp_queries'] = sum(
            (rnd['request'].get('gp_soft_constraints') or {}).get('gp_queries', 0) for rnd in result['rounds'])
        result.update(event_schema=VERSION, revision_phase='dictionary_bootstrap_only', compound_settings=asdict(settings))
        return result
    if (set(public) != {'goal_m', 'map'} or set(public['map']) != {'bounds_m'}
            or public['goal_m'] != task_context['state']['goal_m']
            or public['map']['bounds_m'] != task_context['bounds_m']
            or bank.history[-1]['position_m'] != task_context['state']['position_m']):
        raise ValueError('public task binding mismatch')
    if records and [r['observation'] for r in records] != bank.history[-len(records):]:
        raise ValueError('causal records must match the observed suffix')
    state_id, context_id, record_id = state_digest(bank, public), digest(task_context), digest(records)

    def check_live():
        if (bank.pending is not None or not same_policy() or state_digest(bank, public) != state_id
                or digest(task_context) != context_id or digest(records) != record_id):
            raise ValueError('compound task live inputs changed while waiting')
        validate_task_context(task_context, history=bank.history, models=bank.snapshot())

    if soft_context is not None:
        from .joint_prediction import soft_constraints
        soft = soft_constraints(bank.history, **soft_context)
    else:
        soft = None
    models = bank.snapshot()
    common = build_common(task_context, models, bank.history, soft, settings)
    before = bank.snapshot()
    fitting = {k: 0 for k in COUNTERS}; feedback_work = {k: 0 for k in COUNTERS}
    rounds, coverage, pending = [], [], []
    continuation = feedback = previous_fit = None
    ledger = CandidateLedger(bank.history, common['anchor_cache']) if common['options'] else None
    for index in range(settings.maximum_jobs if ledger is not None else 0):
        check_live()
        request = _bind(dict(schema=VERSION, phase='probe' if index == 0 else 'revise',
            task_context=deepcopy(task_context), models=deepcopy(models), history=deepcopy(bank.history),
            gp_soft_constraints=deepcopy(soft), common=deepcopy(common),
            ledger=ledger.snapshot(), pending_submit_ids=deepcopy(pending), coverage=deepcopy(coverage),
            feedback=deepcopy(feedback), continuation=deepcopy(continuation),
            remaining_jobs=settings.maximum_jobs-index, maximum_submit_models=bank.max_birth_models,
            resources=dict(fitting=deepcopy(fitting), feedback=deepcopy(feedback_work), common=deepcopy(common['resources'])),
            submission_contract='null preserves choice; [] keeps pool; IDs/$this replace atomically; $frontier is a recommendation, not credibility',
            applicability_calibrated=False, birth_mass=settings.birth_mass))
        if method == 'llm':
            try:
                envelope = deepcopy(advisor(deepcopy(request)))
            except TimeoutError as exc:
                envelope = dict(raw=None, timely=False, actual_llm_request=None, advisor_error=str(exc))
            if not isinstance(envelope, dict):
                envelope = dict(raw=None, timely=None, actual_llm_request=None)
        else:
            envelope = dict(raw=numeric_response(request), timely=True, actual_llm_request=False)
        check_live()
        error = selection_error = tool_error = None
        fit = task_feedback = None
        try:
            if envelope.get('timely') is not True:
                raise ValueError('reply unavailable or late')
            plan = validate_response(request, envelope.get('raw'))
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            error = str(exc)
            plan = dict(request_id=request['request_id'], action='stop', target_id=None,
                component_modes=[], start=None, submit_ids=None, reason='preserve earlier legal selection after invalid reply')
        if plan['action'] == 'fit':
            seed = next(m for m in models if m['id'] == plan['target_id'])
            attempted = dict(target_id=seed['id'], component_modes=deepcopy(plan['component_modes']),
                candidate_id=None, start=plan['start'])
            coverage.append(attempted)
            try:
                fit = tool.fit_mechanism(seed, plan['component_modes'], bank.history, soft,
                    f'compound_event{event}_job{index}',
                    start_model=previous_fit['model'] if plan['start'] == 'previous' else None)
                _add(fitting, fit['resources'])
                task_feedback = tool.task_feedback(fit, common['anchor_cache'])
                _add(feedback_work, task_feedback['resources'])
                candidate_id = ledger.add(fit, task_feedback, origin=method)
                attempted['candidate_id'] = candidate_id
                ambiguity, priority = _priority(task_feedback, plan['component_modes'], seed['weight'], common['anchor_cache']['baseline_delta_c'])
                continuation = dict(target_id=seed['id'], component_modes=deepcopy(plan['component_modes']),
                    priority=priority, local_ambiguity_seconds=ambiguity,
                    numeric_continue_eligible=bool(fit['improved'] and not fit['solver'].get('solver_success', False)))
                previous_fit = fit
                feedback = dict(task=deepcopy(task_feedback), applicability=public_diagnostics(fit, common['anchor_cache']),
                    stop_reason=fit['stop_reason'], independent_future_evidence=False)
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                # Unknown exceptional work is charged conservatively, preserving
                # earlier diagnostics and explicit selections; never retry.
                tool_error = str(exc)
                if fit is None:
                    _add(fitting, dict(solver_calls=1, parameter_evaluations=40, objective_calls=40,
                        gradient_calls=40, kernel_terms=5000))
                elif task_feedback is None:
                    _add(feedback_work, dict(kernel_terms=5000))
                continuation = None
        if plan['submit_ids'] is not None:
            try:
                selected = plan['submit_ids']
                if selected == ['$frontier']:
                    selected = ledger.snapshot()['frontier_ids']
                else:
                    selected = [fit['model']['id'] if x == '$this' and fit is not None and tool_error is None else x for x in selected]
                ledger.select(selected)
                pending = deepcopy(selected)
            except (ValueError, KeyError, TypeError) as exc:
                selection_error = str(exc)
        row = dict(request=request, envelope=deepcopy(envelope), accepted_plan=plan,
            parser_accepted=error is None, fallback_error=error, tool_error=tool_error,
            selection_error=selection_error, fit=deepcopy(fit), task_feedback=deepcopy(task_feedback),
            pending_submit_ids=deepcopy(pending), ledger=ledger.snapshot(),
            resource_accounting='conservative_exception_allowance' if tool_error else 'actual_reported',
            resources_after=dict(fitting=deepcopy(fitting), feedback=deepcopy(feedback_work), common=deepcopy(common['resources'])))
        rounds.append(row)
        if emit is not None:
            emit(request['phase'], deepcopy(row))
        check_live()
        if plan['action'] == 'stop':
            break
    check_live()
    committed, commit_error, skipped = [], None, []
    if ledger is not None:
        ledger.stop('event computation complete; selection is explicit')
        try:
            selected = ledger.select(pending)
            candidates = [prepare_compound_version(m, bank.history[-1]['time_seconds'], len(bank.history)-1) for m in selected]
            candidates, skipped = _unique_versions(bank, candidates)
            if candidates:
                bank.birth(candidates)
                committed = candidates
        except ValueError as exc:
            commit_error = str(exc)
    return dict(event_schema=VERSION, revision_phase='compound_revision', index=event,
        time_seconds=bank.history[-1]['time_seconds'], rounds=rounds,
        task_snapshot_id=task_context['task_snapshot_id'], bank_before_birth=before, bank_after_birth=bank.snapshot(),
        committed_models=committed, pool_committed=bool(committed), kept_active_pool=not bool(committed),
        commit_error=commit_error, pending_submit_ids=pending, skipped_clones=skipped,
        candidate_ledger=ledger.snapshot() if ledger is not None else None,
        common=common, new_hypotheses=len(committed), independent_future_evidence=False,
        resources=dict(solver_calls=fitting['solver_calls'], residual_calls=fitting['objective_calls'],
            fitting=fitting, feedback=feedback_work, common=common['resources'], soft_gp_queries=(soft or {}).get('gp_queries', 0)),
        compound_settings=asdict(settings))
