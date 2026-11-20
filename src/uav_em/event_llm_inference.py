"""Two-round inference organization, with an explicit external advisor seam."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
from types import SimpleNamespace

import numpy as np

from .probability import gaussian_logpdf, logsumexp
from .equivalent_field import kernel, power_and_jacobian
from .event_inference import (Budget, FAMILIES, HypothesisBank, digest, fit_job,
                              inference_request, validate_plan)

METHODS = ('persistence', 'fixed', 'feedback', 'llm')


def family_evidence(bank, records):
    families = {m['id']: m['family'] for m in bank.models}
    result = {}
    for family in FAMILIES:
        scores = []
        for row in records[-6:]:
            pred = [p for p in row['issued']['predictions'] if families.get(p['id']) == family]
            if pred:
                ll = [gaussian_logpdf(row['observation']['interference_dbm'],
                                     p['mean_dbm'], p['std_db']) for p in pred]
                scores.append(float(-logsumexp(ll)+math.log(len(ll))))
        result[family] = dict(points=len(scores), mean_nll=float(np.mean(scores)) if scores else None)
    return result


def ranked_families(evidence):
    return sorted(FAMILIES, key=lambda f: (evidence[f]['mean_nll'] if evidence[f]['points'] >= 3
                                          else math.inf, FAMILIES.index(f)))


def state_digest(bank, public):
    return digest(dict(history=bank.history, hypotheses=bank.snapshot(), public=public))


def make_request(bank, public, event, phase, records, feedback, budget, trigger_reason=None):
    if phase not in ('pilot', 'refine'):
        raise ValueError('unknown round')
    context = inference_request(bank, public, trigger_reason or ('common_initial' if event == 0 else 'common_expiry'), event)
    context.pop('request_id')
    context.update(phase=phase, required_jobs=1 if phase == 'pilot' else 2,
        live_state_sha256=state_digest(bank, public), family_evidence=family_evidence(bank, records),
        pilot_training_cutoff_seconds=bank.history[-4]['time_seconds'],
        pilot_training_anchor_m=bank.history[-4]['position_m'],
        feedback=deepcopy(feedback), spent_resources=asdict(budget))
    return dict(context, request_id=digest(context))


def fixed_response(request):
    families = ['static'] if request['phase'] == 'pilot' else ['translate', 'strength_change']
    return dict(request_id=request['request_id'], stage='inference',
                jobs=[dict(family=f, region='task_disk', max_bases=2) for f in families],
                reason='fixed stage allocation; numerical reference')


def feedback_response(request):
    order = ranked_families(request['family_evidence'])
    if request['phase'] == 'pilot':
        families = order[:1]
    else:
        pilot = request['feedback']['job']['family']
        if request['feedback']['tail_rmse_db'] is not None and request['feedback']['tail_rmse_db'] > 4.:
            families = [pilot, next(f for f in order if f != pilot)]
        else:
            families = [f for f in order if f != pilot]
    return dict(request_id=request['request_id'], stage='inference',
                jobs=[dict(family=f, region='task_disk', max_bases=2) for f in families],
                reason='rank birth-after prediction evidence; inspect known-tail residual; numerical reference')


def rebase_model(model, now, index):
    result = deepcopy(model)
    dt = now-model['reference_time_seconds']
    for i, b in enumerate(result['bases']):
        b['center_m'] = (np.asarray(b['center_m'])+dt*np.asarray(b['velocity_mps'])).tolist()
        result['peaks_dbm'][i] += dt*b['rate_db_per_second']
    result.update(reference_time_seconds=now, born_index=index, scored_points=0)
    return result


def pilot_feedback(fit, bank):
    if 'error' in fit:
        return dict(job=fit['job'], error=fit['error'], tail_rmse_db=None,
                    tail_points=[], retrospective_only=True, training_mse=None)
    model = fit['model']
    tail = bank.history[-3:]
    predicted = power_and_jacobian(kernel(model['bases'], [p['position_m'] for p in tail],
        [p['time_seconds'] for p in tail], model['reference_time_seconds']), model['peaks_dbm'])[0]
    diagnostics = [dict(position_m=p['position_m'], time_seconds=p['time_seconds'],
                        observed_dbm=p['interference_dbm'], predicted_dbm=float(y),
                        residual_db=float(p['interference_dbm']-y)) for p, y in zip(tail, predicted)]
    return dict(job=fit['job'], training_mse=model['fit_mse'], fitted_model=deepcopy(model),
        tail_rmse_db=float(np.sqrt(np.mean([p['residual_db']**2 for p in diagnostics]))),
        tail_points=diagnostics, retrospective_only=True, independent_future_evidence=False,
        diagnostic_kernel_terms=len(model['bases'])*len(tail),
        solver_reports=[dict(status=t['solver_status'], success=t['solver_success'], accepted=t['accepted'])
                        for t in fit['trace']])


def validate_round(bank, public, request, raw):
    if request['live_state_sha256'] != state_digest(bank, public):
        raise ValueError('request no longer matches live state')
    if digest({k: v for k, v in request.items() if k != 'request_id'}) != request['request_id']:
        raise ValueError('request content hash mismatch')
    response = validate_plan(request, raw)
    if len(response['jobs']) != request['required_jobs']:
        raise ValueError('wrong number of jobs in round')
    return response


def run_event(bank, public, event, records, method, advisor=None, emit=None, *, trigger_reason=None,
              soft_context=None):
    if method not in METHODS[1:]:
        raise ValueError('only updating methods have events')
    budget, rounds, all_models, feedback = Budget(), [], [], None
    for phase in ('pilot', 'refine'):
        request = make_request(bank, public, event, phase, records, feedback, budget, trigger_reason)
        if soft_context is not None:
            from .joint_prediction import soft_constraints
            source_history = bank.history[:-3] if phase == 'pilot' else bank.history
            soft = soft_constraints(source_history, **soft_context)
            request['gp_soft_constraints'] = deepcopy(soft)
            request['request_id'] = digest({k:v for k,v in request.items() if k != 'request_id'})
        else:
            soft = None
        if method == 'llm':
            if advisor is None:raise ValueError('LLM method requires an explicit advisor')
            envelope = advisor(deepcopy(request))
        else:
            response = fixed_response(request) if method == 'fixed' else feedback_response(request)
            envelope = dict(raw=response, timely=True, actual_llm_request=False, wall_seconds=0.)
        if request['live_state_sha256'] != state_digest(bank, public):
            raise ValueError('live state advanced while waiting; reject before any fitting')
        error, plan = None, None
        try:
            if not envelope['timely']:raise ValueError('response timeout')
            plan = validate_round(bank, public, request, envelope['raw'])
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            plan = fixed_response(request)
        budget.reserve(plan['jobs'])
        fits = []
        for j, job in enumerate(plan['jobs']):
            source_bank = (SimpleNamespace(history=bank.history[:-3], models=[], log_weights=[])
                           if phase == 'pilot' else bank)
            before = asdict(budget)
            try:
                fit = fit_job(source_bank, public, job, budget, f'event{event}_{phase}_job{j}', soft=soft)
                fits.append(fit)
                if fit['model']['bases']:
                    all_models.append(rebase_model(fit['model'], bank.history[-1]['time_seconds'], len(bank.history)-1))
            except (ValueError, FloatingPointError) as exc:
                fits.append(dict(job=job, error=str(exc), resources_before=before, resources_after=asdict(budget)))
        if phase == 'pilot':feedback = pilot_feedback(fits[0], bank)
        result = dict(request=request, envelope=deepcopy(envelope), accepted_plan=plan,
            parser_accepted=error is None, fallback_error=error, fits=fits,
            resources_after=asdict(budget), pilot_feedback=deepcopy(feedback) if phase == 'pilot' else None)
        rounds.append(result)
        if emit:emit(phase, result)
    before = bank.snapshot()
    bank.birth(all_models)
    return dict(index=event, time_seconds=bank.history[-1]['time_seconds'], rounds=rounds,
                bank_before_birth=before, bank_after_birth=bank.snapshot(),
                committed_models=all_models, resources=asdict(budget), new_hypotheses=len(all_models))


def run_stream(stream, method, advisor=None, emit=None):
    if method not in METHODS:raise ValueError('unknown method')
    bank = HypothesisBank(stream['measurements'][0])
    records, events = [], []
    for measurement in stream['measurements'][1:]:
        bank.issue(measurement['position_m'], measurement['time_seconds'])
        row = bank.observe(measurement)
        records.append(row)
        if method != 'persistence' and measurement['time_seconds'] in (8., 20.):
            callback = (lambda phase, r: emit(len(events), phase, r)) if emit else None
            event = run_event(bank, stream['public'], len(events), records, method, advisor, callback)
            row['event_index'] = len(events)
            events.append(event)
    return dict(method=method, records=records, events=events, final_bank=bank.snapshot())


def score_result(result):
    rows = result['records']
    primary = [r for r in rows if r['observation']['time_seconds'] > 8.]
    def metrics(items):
        return dict(points=len(items), mean_nll=float(np.mean([r['nll'] for r in items])),
                    rmse_db=float(np.sqrt(np.mean([r['squared_error'] for r in items]))),
                    mae_db=float(np.mean([r['absolute_error'] for r in items])))
    return dict(all=metrics(rows), primary=metrics(primary))


def development_gate(city_scores):
    llm = [s['llm']['primary'] for s in city_scores]
    feedback = [s['feedback']['primary'] for s in city_scores]
    persistence = [s['persistence']['primary'] for s in city_scores]
    conditions = dict(
        mean_nll_gain=float(np.mean([x['mean_nll'] for x in llm])) <= float(np.mean([x['mean_nll'] for x in feedback]))-.02,
        each_city_nll=all(x['mean_nll'] <= y['mean_nll']+.05 for x, y in zip(llm, feedback)),
        mean_rmse=float(np.mean([x['rmse_db'] for x in llm])) <= float(np.mean([x['rmse_db'] for x in feedback])),
        persistence_control=float(np.mean([x['mean_nll'] for x in llm])) <= float(np.mean([x['mean_nll'] for x in persistence]))+.02)
    return dict(conditions=conditions, passed=all(conditions.values()))
