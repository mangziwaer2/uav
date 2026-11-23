"""Exact shared numerical references over a prepared corridor DAG.

All costs and observation likelihoods come from the bound public preparation.
No simulator, sensor, source fitting or language-model call is made here.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import math

from .strategy_graph import (BudgetExhausted, PolicyError, evaluate_strategy,
                             validate_prepared)


VERSION = 'corridor-policy-v1'


@dataclass(frozen=True)
class _Solution:
    decision: dict
    q: tuple
    flight: tuple
    length: tuple
    worst_length: float


def _serial(decision):
    return json.dumps(decision, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _score(solution, weights):
    return math.fsum(w * (q + t) for w, q, t in zip(weights, solution.q, solution.flight))


def _key(solution, weights):
    return _score(solution, weights), solution.worst_length, _serial(solution.decision)


def _terminal(edge, size):
    return _Solution(dict(edge_id=edge['id'], threshold_dbm=None, below=None, above=None),
        tuple(edge['q_by_world']), (edge['duration_seconds'],) * size,
        (edge['length_m'],) * size, edge['length_m'])


def _fixed(edge, threshold, child):
    return _Solution(dict(edge_id=edge['id'], threshold_dbm=threshold,
        below=deepcopy(child.decision), above=deepcopy(child.decision)),
        tuple(math.fsum((x, y)) for x, y in zip(edge['q_by_world'], child.q)),
        tuple(edge['duration_seconds'] + x for x in child.flight),
        tuple(edge['length_m'] + x for x in child.length),
        edge['length_m'] + child.worst_length)


def _conditional(edge, cut, below, above):
    low, high = cut['below_by_world'], cut['above_by_world']
    def combine(immediate, first, second):
        return tuple(math.fsum((v, p * a, r * b))
                     for v, p, a, r, b in zip(immediate, low, first, high, second))
    size = len(low)
    return _Solution(dict(edge_id=edge['id'], threshold_dbm=cut['threshold_dbm'],
        below=deepcopy(below.decision), above=deepcopy(above.decision)),
        combine(edge['q_by_world'], below.q, above.q),
        combine((edge['duration_seconds'],) * size, below.flight, above.flight),
        combine((edge['length_m'],) * size, below.length, above.length),
        edge['length_m'] + max(below.worst_length, above.worst_length))


def _search(prepared, budget, *, conditional):
    p = validate_prepared(prepared)
    before = deepcopy(budget.summary())
    root_id = p['root_id']
    weights = tuple(float(w['weight']) for w in p['task']['worlds'])
    size = len(weights)
    memo, reserved_states, baseline_memo = {}, set(), {}
    stats = dict(cache_hits=0, baseline_cache_hits=0, zero_probability_fallbacks=0)
    best = baseline_record = root_best_internal = None
    evaluations, failures = [], []
    attempted_verifications = set()

    def visit(node_id, belief):
        key = node_id, tuple(belief)
        if key not in reserved_states:
            budget.reserve(search_states=1)
            reserved_states.add(key)
        return key

    def verify_root(solution):
        nonlocal best, baseline_record, root_best_internal
        if solution is None:
            return
        comparison = _key(solution, weights)
        if root_best_internal is not None and comparison >= root_best_internal:
            return
        signature = _serial(solution.decision)
        if signature in attempted_verifications:
            return
        attempted_verifications.add(signature)
        policy = dict(version=VERSION, task_id=p['task_id'], root=deepcopy(solution.decision))
        try:
            record = evaluate_strategy(p, policy, budget)
        except PolicyError as exc:
            failures.append(dict(policy=policy, error=str(exc),
                error_type=type(exc).__name__, work=deepcopy(getattr(exc, 'work', {}))))
            return
        score = record['expected_score_seconds']
        length = record['worst_length_m']
        if (type(score) not in (int, float) or not math.isfinite(score) or score < 0
                or type(length) not in (int, float) or not math.isfinite(length) or length < 0):
            raise ValueError('shared evaluator returned invalid comparison metadata')
        evaluations.append(deepcopy(record))
        if baseline_record is None:
            baseline_record = deepcopy(record)
        root_best_internal = comparison
        record_key = score, length, _serial(record['policy']['root'])
        if best is None or record_key < (best['expected_score_seconds'], best['worst_length_m'],
                                       _serial(best['policy']['root'])):
            best = deepcopy(record)

    def first_fixed(node_id):
        """Pay for a complete legal fallback before exploring contingent trees."""
        visit(node_id, weights)
        if node_id in baseline_memo:
            stats['baseline_cache_hits'] += 1
            return baseline_memo[node_id]
        node = p['nodes'][node_id]
        for edge_id in sorted(node['terminal_edges']):
            budget.reserve(action_evaluations=1)
            value = _terminal(p['edges'][edge_id], size)
            baseline_memo[node_id] = value
            return value
        for edge_id in sorted(node['step_edges']):
            edge = p['edges'][edge_id]
            cuts = p['nodes'][edge['to_id']]['cuts']
            if not cuts:
                continue
            budget.reserve(action_evaluations=1)
            child = first_fixed(edge['to_id'])
            if child is not None:
                value = _fixed(edge, cuts[0]['threshold_dbm'], child)
                baseline_memo[node_id] = value
                return value
        baseline_memo[node_id] = None
        return None

    def posterior(belief, probabilities):
        mass = tuple(w * likelihood for w, likelihood in zip(belief, probabilities))
        total = math.fsum(mass)
        if total == 0:
            stats['zero_probability_fallbacks'] += 1
            # A probability-zero branch still receives a complete legal policy.
            # Reuse the incoming prior, without inventing new evidence.
            return tuple(belief)
        return tuple(value / total for value in mass)

    def solve(node_id, belief):
        key = visit(node_id, belief)
        if key in memo:
            stats['cache_hits'] += 1
            return memo[key]
        node = p['nodes'][node_id]
        local = None
        def consider(value):
            nonlocal local
            if value is not None and (local is None or _key(value, belief) < _key(local, belief)):
                local = value
                if node_id == root_id and tuple(belief) == weights:
                    verify_root(value)
        for edge_id in sorted(node['terminal_edges']):
            budget.reserve(action_evaluations=1)
            consider(_terminal(p['edges'][edge_id], size))
        for edge_id in sorted(node['step_edges']):
            edge = p['edges'][edge_id]
            cuts = p['nodes'][edge['to_id']]['cuts']
            if not cuts:
                continue
            # All thresholds induce the same fixed control if both leaves agree.
            for cut in cuts if conditional else cuts[:1]:
                budget.reserve(action_evaluations=1)
                if conditional:
                    budget.reserve(belief_updates=2)
                    below_belief = posterior(belief, cut['below_by_world'])
                    above_belief = posterior(belief, cut['above_by_world'])
                    below = solve(edge['to_id'], below_belief)
                    above = solve(edge['to_id'], above_belief)
                    if below is not None and above is not None:
                        consider(_conditional(edge, cut, below, above))
                else:
                    child = solve(edge['to_id'], belief)
                    if child is not None:
                        consider(_fixed(edge, cut['threshold_dbm'], child))
        memo[key] = local
        return local

    exhausted = None
    search_complete = False
    try:
        verify_root(first_fixed(root_id))
        solution = solve(root_id, weights)
        verify_root(solution)
        search_complete = True
    except BudgetExhausted as exc:
        exhausted = dict(error=str(exc), error_type=type(exc).__name__,
                         work=deepcopy(getattr(exc, 'work', {})))
    after = deepcopy(budget.summary())
    work = {key:after.get(key, 0) - before.get(key, 0) for key in set(before) | set(after)}
    if any(value < 0 for value in work.values()):
        raise ValueError('shared work ledger decreased during search')
    complete = search_complete and p['preparation_complete'] and not failures
    if exhausted is not None:
        reason = 'budget_exhausted'
    elif failures:
        reason = 'verification_failed'
    elif not p['preparation_complete']:
        reason = 'preparation_incomplete'
    elif best is None:
        reason = 'no_feasible_strategy'
    else:
        reason = 'domain_complete'
    stats.update(reserved_states=len(reserved_states), completed_states=len(memo),
                 baseline_completed_nodes=len(baseline_memo))
    return dict(best=best, baseline=baseline_record, evaluations=evaluations, failures=failures,
        complete=complete, search_complete=search_complete, stopped_reason=reason,
        stopped_details=exhausted, work=work, budget_before=before, budget_after=after,
        memo_stats=stats, preparation_work=deepcopy(p['work']),
        mode='conditional' if conditional else 'open_loop',
        scope=dict(exact_belief_memo=True, quantized_belief=False,
            optimality='declared prepared DAG only, and only when complete is true',
            zero_probability_fallback='incoming-prior recursion; all branches remain legal',
            baseline='first complete fixed-control path, paid before exact expansion',
            cache_hits_charged=False, new_sensor_measurements=0, language_model_calls=0))


def search_dynamic(prepared, budget):
    """Find an exact finite-belief conditional strategy, subject to paid work."""
    return _search(prepared, budget, conditional=True)


def search_open_loop(prepared, budget):
    """Find the best complete fixed-control route in the same prepared DAG."""
    return _search(prepared, budget, conditional=False)
