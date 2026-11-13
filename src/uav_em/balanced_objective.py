"""连续功率—航时共同接受器及独立搜索流；只读取因果快照。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

import numpy as np

from .decision_objective import ObjectiveOptimizer, ObjectiveSettings, prefix_risk, select_candidate
from .offline_process import validate_program
from .planning import path_length
from .repair_process import RepairState, _candidate, _proposal, canonical_path, feedback_outcome
from .repair_search import BudgetExhausted, FixedObjective
from .route_repair import RepairSession, RepairSettings, anchors, parse_proposal

VERSION = 'full_route_power_time_v3'


@dataclass(frozen=True)
class PowerTimeSettings:
    weight: float = 1.
    scale_dbm: float = -80.
    prior_mean_dbm: float = -100.
    prior_std_db: float = 15.

    def __post_init__(self):
        if any(isinstance(v, bool) or not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('finite numeric settings required')
        if self.weight < 0 or self.prior_std_db < 0:
            raise ValueError('nonnegative weight/std required')
        normalized_expected_power(self.prior_mean_dbm, self.prior_std_db, self.scale_dbm)


def normalized_expected_power(mean, std, scale_dbm):
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std < 0) or not math.isfinite(scale_dbm):
        raise ValueError('finite mean/scale and nonnegative std required')
    a = math.log(10) / 10
    with np.errstate(over='raise', invalid='raise'):
        try:
            return np.exp(a * (mean - scale_dbm) + .5 * a * a * std * std)
        except FloatingPointError as exc:
            raise ValueError('power expectation overflow; no clipping applied') from exc


def power_time_score(problem, route, speed, settings=PowerTimeSettings()):
    """全剩余路径分段精确积分；未知只用声明常数，不读取其数值栅格。"""
    if not math.isfinite(speed) or speed <= 0 or not route:
        raise ValueError('positive speed and nonempty route required')
    known_q = known_t = unknown_t = 0.
    nodes_count = 0
    s = problem.snapshot
    for a, b in zip(route, route[1:]):
        lengths, nodes = problem.quadrature(a, b)
        duration = lengths / speed
        rows, cols = nodes[:, 1], nodes[:, 0]
        mask = s.supported[rows, cols]
        mean = s.estimate_dbm[rows[mask], cols[mask]]
        std = s.uncertainty_db[rows[mask], cols[mask]]
        known_q += float(duration[mask] @ normalized_expected_power(mean, std, settings.scale_dbm))
        known_t += float(duration[mask].sum())
        unknown_t += float(duration[~mask].sum())
        nodes_count += len(nodes)
    prior_q = float(normalized_expected_power(settings.prior_mean_dbm, settings.prior_std_db, settings.scale_dbm))
    unknown_q = prior_q * unknown_t
    total_t = known_t + unknown_t
    return {'score_seconds': total_t + settings.weight * (known_q + unknown_q),
            'remaining_flight_seconds': total_t, 'supported_power_seconds': known_q,
            'unknown_prior_power_seconds': unknown_q, 'power_proxy_seconds': known_q + unknown_q,
            'supported_seconds': known_t, 'unknown_seconds': unknown_t,
            'unknown_prior_relative_power': prior_q, 'field_nodes': nodes_count,
            'whole_remaining_route': True, 'arrival_field_predicted': False,
            'unknown_power_estimate_exposed': False, 'unknown_cost_is_declared_prior_not_bound': True,
            'uncertainty_calibrated': False, 'hard_power_threshold': None}


def choose(records, rule='balanced'):
    if rule == 'prefix':
        return select_candidate(records, 'prefix_v2')
    if rule != 'balanced':
        raise ValueError('unknown selection rule')
    valid = [r for r in records if r['feasible']]
    if not valid:
        return None
    return min(valid, key=lambda r: (r['balance']['score_seconds'], r['metrics']['length_m']))


def pareto_ids(records):
    valid = [r for r in records if r['feasible']]
    def pair(r): return r['metrics']['length_m'], r['balance']['power_proxy_seconds']
    return [r['id'] for r in valid if not any(
        pair(q)[0] <= pair(r)[0] and pair(q)[1] <= pair(r)[1] and pair(q) != pair(r) for q in valid)]


def paired_seed(scene_seed, time_seconds, stage, purpose, repeat=0):
    if type(scene_seed) is not int or type(repeat) is not int or repeat < 0 or not math.isfinite(time_seconds) or time_seconds < 0:
        raise ValueError('valid scene seed, nonnegative time/repeat required')
    key = ['paired_repair_stream_v1', scene_seed, int(round(time_seconds * 1_000_000)), repeat, stage, purpose]
    return int.from_bytes(hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).digest()[:8], 'big')


class _BalancedSession(RepairSession):
    def __init__(self, snapshot, context, route, maximum, limits, scene_seed, report, score_settings, rule):
        super().__init__(snapshot, context, [], maximum,
                         RepairSettings(sample_limit=limits.sample_limit, edge_limit=limits.edge_limit), scene_seed)
        self.scene_seed, self.score_settings, self.rule = scene_seed, score_settings, rule
        self.speed = float(context['vehicle']['speed_mps'])
        for key in ('samples', 'edges', 'cache_hits', 'verification_edges'):
            setattr(self.work, key, report['resources'][key])
        self.records = deepcopy(report['candidates'])
        self.best = deepcopy(route)
        self.best_metrics = deepcopy(choose(self.records, rule)['metrics'])
        self.initial_metrics = deepcopy(self.best_metrics)
        self.work.reserved_edges = max(len(r['path']) - 1 for r in self.records if r['feasible'])
        self.node_id = 0
        self.score_nodes = 0

    def search(self, a, b, quota, sampling, purpose):
        # The inherited routine adds len(searches); compensate so each stream has
        # its declared node/leg seed independently of any other search calls.
        actual = paired_seed(self.scene_seed, self.snapshot.time_seconds, f'node{self.node_id}', purpose)
        old = self.seed
        self.seed = actual - len(self.searches)
        try:
            path = super().search(a, b, quota, sampling, purpose)
        finally:
            self.seed = old
        self.searches[-1].update(actual_seed=actual, seed_stage=f'node{self.node_id}', seed_purpose=purpose)
        return path

    def request(self, round_id, max_rounds):
        ctx = super().request(round_id, max_rounds)
        ctx['route_repair']['fixed_objective']['role'] = 'old_J_search_diagnostic_not_acceptance'
        ctx['route_repair']['selection_rule'] = self.rule
        return ctx

    def apply(self, proposal):
        before, prior = deepcopy(self.best), deepcopy(self.best_metrics)
        version, accepted, reserve = self.version, self.accepted_proposals, self.work.reserved_edges
        nsearch = len(self.searches)
        feedback = super().apply(proposal)
        candidate = _candidate(proposal, before, self.searches[nsearch:], self.settings.anchor_spacing_m)
        self.best, self.best_metrics = before, prior
        self.version, self.accepted_proposals, self.work.reserved_edges = version, accepted, reserve
        feedback.update(accepted=False, best_metrics=prior, route_version_after=version)
        if proposal['operator'] == 'keep' or feedback['reason'] not in ('improved', 'no_objective_improvement'):
            return feedback
        candidate = anchors(candidate, self.settings.anchor_spacing_m)
        duplicate = next((r['id'] for r in self.records if r['feasible'] and canonical_path(r['path']) == canonical_path(candidate)), None)
        feedback['duplicate_of'] = duplicate
        if duplicate:
            feedback['reason'] = 'duplicate_candidate'
            return feedback
        risk = prefix_risk(self.problem, candidate, self.speed)
        balance = power_time_score(self.problem, candidate, self.speed, self.score_settings)
        self.score_nodes += risk['field_nodes'] + balance['field_nodes']
        record = {'id': f'process_candidate_{len(self.records)}', 'path': candidate, 'feasible': True,
                  'failure': None, 'metrics': feedback['candidate_metrics'], 'prefix': risk, 'balance': balance}
        self.records.append(record)
        chosen = choose(self.records, self.rule)
        changed = canonical_path(chosen['path']) != canonical_path(before)
        if changed:
            self.best, self.best_metrics = deepcopy(chosen['path']), deepcopy(chosen['metrics'])
            self.version += 1
            self.accepted_proposals += 1
        self.work.reserved_edges = max(reserve, len(candidate) - 1)
        feedback.update(accepted=changed, reason='improved' if changed else 'no_objective_improvement',
                        candidate_prefix=risk, candidate_balance=balance, best_metrics=deepcopy(self.best_metrics),
                        route_version_after=self.version, selected_candidate_id=chosen['id'],
                        improvement_meaning='declared_common_proxy_not_true_exposure')
        return feedback


class BalancedOptimizer:
    policy = None
    last_call = {}

    def __init__(self, scene_seed, score_settings=PowerTimeSettings(), program=None, rule='balanced', limits=ObjectiveSettings()):
        if rule not in ('balanced', 'prefix'):
            raise ValueError('unknown rule')
        self.scene_seed, self.score_settings, self.rule, self.limits = scene_seed, score_settings, rule, limits
        self.program = validate_program(program) if program is not None else None
        self.base = ObjectiveOptimizer('prefix_v2', limits)

    def optimize(self, snapshot, context, route, max_length_m, seed, call_limit):
        started = time.perf_counter()
        common_seed = paired_seed(self.scene_seed, snapshot.time_seconds, 'common_pool', 'rrt')
        result, report = self.base.optimize(snapshot, context, route, max_length_m, common_seed, 0)
        report.update(contract_version=VERSION, common_pool_resources=deepcopy(report['resources']),
                      selection_rule=self.rule, power_time_settings=asdict(self.score_settings),
                      common_pool_seed=common_seed, simulator_cumulative_seed_ignored=True,
                      process_program=deepcopy(self.program), process_trace=[], score_field_nodes=0,
                      actual_advisor_calls=0, model_calls_within_optimizer=0, active_probe_enabled=False,
                      submitted_artifact_is_model_call=False)
        if not result.path:
            return result, report
        problem = FixedObjective(snapshot, context['electromagnetic_estimate']['interference_reference_dbm'],
                                 self._dummy_budget())
        speed = context['vehicle']['speed_mps']
        for record in report['candidates']:
            if record['feasible']:
                record['balance'] = power_time_score(problem, record['path'], speed, self.score_settings)
                report['score_field_nodes'] += record['balance']['field_nodes']
        chosen = choose(report['candidates'], self.rule)
        effective_limit = min(max_length_m, report['mission_distance_cap_m'] - report['executed_distance_m'])
        session = _BalancedSession(snapshot, context, chosen['path'], effective_limit, self.limits,
                                   self.scene_seed, report, self.score_settings, self.rule)
        node_id = 0
        for step in range(1, 7) if self.program is not None else ():
            if session.work.summary()['edges_left'] < len(session.best) - 1:
                break
            session.node_id = node_id
            ctx = session.request(step, 6)
            state = RepairState.from_context(ctx).to_dict()
            state.update(contract_version=VERSION, task_objective=self.rule,
                         current_balance=power_time_score(session.problem, session.best, speed, self.score_settings),
                         selection_settings=asdict(self.score_settings))
            session.score_nodes += state['current_balance']['field_nodes']
            before = session.work.summary()
            node = self.program['nodes'][node_id]
            slots = sum(n['action']['operator'] not in ('keep', 'shortcut') for n in self.program['nodes'][node_id:])
            proposal, construction, error = None, {}, None
            try:
                proposal, construction = _proposal(RepairState(state), node['action'], slots)
                if proposal is None:
                    feedback = {'accepted': False, 'reason': 'insufficient_shared_resources',
                                'resources': session.work.summary(), 'best_metrics': deepcopy(session.best_metrics)}
                else:
                    proposal = parse_proposal(proposal, ctx)
                    feedback = session.apply(proposal)
            except (ValueError, TypeError, BudgetExhausted) as exc:
                error = f'{type(exc).__name__}: {exc}'
                feedback = {'accepted': False, 'reason': 'policy_error', 'error': error,
                            'resources': session.work.summary(), 'best_metrics': deepcopy(session.best_metrics)}
            outcome = feedback_outcome(feedback, feedback.get('duplicate_of'))
            target = node['next'].get(outcome, node['next']['default'])
            report['process_trace'].append({'node': node_id, 'state': state, 'proposal': proposal,
                'construction': construction, 'feedback': feedback, 'outcome': outcome, 'next': target, 'error': error,
                'work_delta': {k: session.work.summary()[k] - before[k] for k in ('samples', 'edges', 'cache_hits')}})
            session.trace.append({'feedback': feedback})
            if target == 'stop' or outcome in ('stopped', 'policy_error', 'resource_stop'):
                break
            node_id = target
        chosen = choose(session.records, self.rule)
        assert canonical_path(chosen['path']) == canonical_path(session.best)
        session.work.reserved_edges = 0
        for a, b in zip(session.best, session.best[1:]):
            session.work.take_edge()
            session.work.verification_edges += 1
            if not snapshot.geometry.edge_free(a, b):
                raise RuntimeError('balanced final geometry invalid')
        elapsed = time.perf_counter() - started
        result.path, result.length_m, result.cost = session.best, path_length(session.best), chosen['balance']['score_seconds'] if self.rule == 'balanced' else chosen['prefix']['risk_db_seconds']
        result.elapsed_seconds, result.iterations = elapsed, session.work.samples
        report.update(resources=session.work.summary(), candidates=session.records,
                      selected_candidate_id=chosen['id'], selected_prefix=chosen['prefix'],
                      selected_balance=chosen['balance'], best_metrics=deepcopy(chosen['metrics']),
                      pareto_candidate_ids=pareto_ids(session.records),
                      searches=report['searches'] + session.searches,
                      accepted_proposals=int(not route or canonical_path(route) != canonical_path(session.best)),
                      process_accepted_proposals=session.accepted_proposals,
                      policy_errors=sum(t['error'] is not None for t in report['process_trace']),
                      score_field_nodes=report['score_field_nodes'] + session.score_nodes,
                      numeric_compute_seconds=elapsed, final_geometry_verified=True)
        report['searches'][0].update(actual_seed=common_seed, seed_stage='common_pool', seed_purpose='rrt')
        return result, report

    @staticmethod
    def _dummy_budget():
        from .repair_search import WorkBudget
        return WorkBudget()
