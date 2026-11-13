"""离线有限过程合同：共同池、共同R10接受器、共享搜索预算。

本模块执行已经提交的JSON过程，不调用模型，不执行Python代码。
旧J仍是RRT*的搜索代价；R10是本版本所有方法的共同选路代理。
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

from .decision_objective import ObjectiveOptimizer, ObjectiveSettings, prefix_risk, select_candidate
from .planning import path_length
from .repair_process import (
    CONTRACT_VERSION as OLD_VERSION, RepairState, _candidate, _proposal,
    baseline_process, canonical_path, feedback_outcome, validate_process,
)
from .repair_search import BudgetExhausted, FixedObjective, WorkBudget
from .route_repair import RepairSession, RepairSettings, anchors, parse_proposal

VERSION = 'offline_finite_process_v1'
METHODS = ('pool_only', 'fixed_process', 'adaptive_process', 'submitted_process')


def validate_program(program):
    """仅复用已验证的有限操作语法；版本独立，拒绝额外字段和自由代码。"""
    if not isinstance(program, dict) or program.get('version') != VERSION:
        raise ValueError('offline process version required')
    old = deepcopy(program)
    old['version'] = OLD_VERSION
    validated = validate_process(old)
    validated['version'] = VERSION
    return validated


def reference_program(adaptive=True):
    program = baseline_process(adaptive)
    program['version'] = VERSION
    return validate_program(program)


def generation_contract():
    """可公开的生成规范，不含开发场景配置/真值或历史事后参考路线。"""
    return {
        'version': VERSION,
        'artifact': 'JSON finite repair process composed of existing operators, not new primitive code',
        'max_nodes': 6,
        'example': reference_program(),
        'operators': ['shortcut', 'local_rrt', 'via', 'global_rrt', 'keep'],
        'selectors': ['whole', 'supported_interference', 'unknown_front', 'stale', 'detour', 'evidence'],
        'action_fields': {'operator': 'operators', 'metric': 'selectors', 'padding': [0, 1, 2, 3],
                          'side': ['both', 'left', 'right', 'tube'],
                          'scale': ['near', 'local', 'wide'], 'allocation': ['fixed', 'feedback']},
        'outcomes': ['improved', 'no_gain', 'geometry_failure', 'search_failure', 'resource_stop',
                     'duplicate', 'stopped', 'policy_error'],
        'transitions': 'next contains default and optional outcomes; only later node indices or stop',
        'information': 'received scalar point measurements, causal GP summaries/support/age, known geometry, '
                       'remaining route, remaining energy/distance, numeric feedback; no bearing/source/future truth',
        'hard_constraints': ['geometry', 'current_position_and_goal', 'energy_distance_limit',
                             'flown_plus_remaining_le_fixed_1.25D0', 'shared_2100_samples_10000_edges'],
        'common_selection': 'global minimum R10 plus 1 dB s window, then shortest remaining route; stable ties',
        'risk_definition': '10s current GP expected positive excess above -80 dBm on supported cells '
                           '+ 10 dB times unknown seconds; not arrival-time prediction or safety guarantee',
        'search_objective': 'unchanged old current-GP J inside RRT*; reported separately',
        'soft_diagnostics': ['old_J', 'supported_expected_excess_db_seconds', 'unknown_seconds',
                             'risk_db_seconds', 'support_age', 'point_measurement_trend', 'distance_margin_m'],
        'not_allowed': ['extra_measurements', 'source_inference', 'changed_GP_or_objective',
                        'changed_hard_constraints_or_budget', 'arbitrary_Python', 'loops', 'file_access'],
        'attribution': 'common-pool cost charged before any process; selected-route verification charged last',
        'generation_budget': {'initial_drafts': 2, 'revision_drafts': 2, 'max_submitted_artifacts': 4,
                              'max_evaluations_per_artifact': 1, 'development_scenes_per_artifact': 4,
                              'max_generated_closed_loop_flights': 16, 'reference_closed_loop_flights': 12,
                              'feedback': 'aggregate task metrics and errors only; no hidden sources or truth maps',
                              'invalid_artifact_counts_against_budget': True},
        'development_seeds': [611, 613, 617, 619],
        'validation_test_policy': 'do not generate/read until a separate frozen validation protocol',
        'costs_to_record': ['actual_requests', 'raw_responses', 'parse_acceptance', 'errors', 'revisions',
                            'samples', 'edge_evaluations', 'field_nodes', 'model_tokens_if_available', 'wall_seconds'],
    }


class _RiskSession(RepairSession):
    """借用原数值构造，但不继承旧J改善准入条件。"""

    def __init__(self, snapshot, context, route, max_length_m, settings, seed, pool_report, selection_settings):
        super().__init__(snapshot, context, [], max_length_m, settings, seed)
        self.selection_settings = selection_settings
        self.speed = float(context['vehicle']['speed_mps'])
        resources = pool_report['resources']
        for key in ('samples', 'edges', 'cache_hits', 'verification_edges'):
            setattr(self.work, key, resources[key])
        self.problem = FixedObjective(self.snapshot, self.problem.reference, self.work)
        self.records = deepcopy(pool_report['candidates'])
        self.best = deepcopy(route)
        self.best_metrics = deepcopy(pool_report['best_metrics'])
        self.initial_metrics = deepcopy(self.best_metrics)
        # Keep every candidate in the common risk window retainable even after
        # later searches. The selector considers the full pool, not just the last pair.
        self.work.reserved_edges = max(len(r['path']) - 1 for r in self.records if r['feasible'])
        self.flown = pool_report['executed_distance_m']
        self.mission_cap = pool_report['mission_distance_cap_m']
        self.extra_field_nodes = 0

    def request(self, round_id, max_rounds):
        context = super().request(round_id, max_rounds)
        repair = context['route_repair']
        repair['fixed_objective']['role'] = 'old_J_search_and_diagnostic_not_acceptance'
        repair['selection_rule'] = 'common_global_R10_window_then_length'
        repair['executed_distance_m'] = self.flown
        repair['mission_distance_cap_m'] = self.mission_cap
        return context

    def apply(self, proposal):
        # super.apply constructs, checks and scores a complete candidate using the
        # shared ledger. Its temporary J decision is explicitly undone below.
        before = deepcopy(self.best)
        prior = deepcopy(self.best_metrics)
        version, accepted, reserve = self.version, self.accepted_proposals, self.work.reserved_edges
        searches = len(self.searches)
        feedback = super().apply(proposal)
        candidate = _candidate(proposal, before, self.searches[searches:], self.settings.anchor_spacing_m)
        self.best, self.best_metrics = before, prior
        self.version, self.accepted_proposals, self.work.reserved_edges = version, accepted, reserve
        feedback.update(accepted=False, best_metrics=prior, route_version_after=version)
        feedback['candidate_prefix'] = None
        if proposal['operator'] == 'keep':
            return feedback
        if feedback['reason'] not in ('improved', 'no_objective_improvement'):
            return feedback
        candidate = anchors(candidate, self.settings.anchor_spacing_m)
        metrics = feedback['candidate_metrics']
        if not self.endpoints(candidate):
            feedback['reason'] = 'route_endpoints_invalid'
            return feedback
        if metrics['length_m'] > self.max_length + 1e-8:
            feedback['reason'] = 'energy_budget_exceeded'
            return feedback
        duplicate = next((r['id'] for r in self.records
                          if r['feasible'] and canonical_path(r['path']) == canonical_path(candidate)), None)
        feedback['duplicate_of'] = duplicate
        if duplicate is not None:
            feedback['reason'] = 'duplicate_candidate'
            return feedback
        risk = prefix_risk(self.problem, candidate, self.speed, self.selection_settings)
        self.extra_field_nodes += risk['field_nodes']
        record = {'id': f'process_candidate_{len(self.records)}', 'path': candidate,
                  'feasible': True, 'failure': None, 'metrics': metrics, 'prefix': risk}
        self.records.append(record)
        chosen = select_candidate(self.records, 'prefix_v2', self.selection_settings)
        changed = canonical_path(chosen['path']) != canonical_path(before)
        if changed:
            self.best = deepcopy(chosen['path'])
            self.best_metrics = deepcopy(chosen['metrics'])
            self.version += 1
            self.accepted_proposals += 1
        self.work.reserved_edges = max(reserve, len(candidate) - 1)
        feedback.update(accepted=changed, reason='improved' if changed else 'no_objective_improvement',
                        candidate_prefix=risk, best_metrics=deepcopy(self.best_metrics),
                        route_version_after=self.version, selected_candidate_id=chosen['id'],
                        improvement_meaning='common_R10_window_and_length_not_J_or_true_exposure')
        return feedback


class OfflineProcessOptimizer:
    """所有方法共用初始池/接受器；submitted_process本身不代表实际LLM调用。"""
    policy = None
    last_call = {}

    def __init__(self, method='adaptive_process', program=None, settings=ObjectiveSettings()):
        if method not in METHODS:
            raise ValueError('unknown offline process method')
        if method == 'submitted_process' and program is None:
            raise ValueError('submitted process artifact required')
        if method != 'submitted_process' and program is not None:
            raise ValueError('reference methods cannot receive a replacement program')
        self.method, self.settings = method, settings
        self.program = validate_program(program) if program is not None else reference_program(method != 'fixed_process')
        self.base = ObjectiveOptimizer('prefix_v2', settings)

    def optimize(self, snapshot, context, route, max_length_m, seed, call_limit):
        started = time.perf_counter()
        result, report = self.base.optimize(snapshot, context, route, max_length_m, seed, call_limit)
        report.update(contract_version=VERSION, method=self.method,
                      common_pool_resources=deepcopy(report['resources']), process_program=deepcopy(self.program),
                      process_trace=[], model_calls_within_optimizer=0, submitted_artifact_is_model_call=False,
                      active_probe_enabled=False, hard_constraints_and_selection_shared=True)
        if not result.path or self.method == 'pool_only':
            return result, report
        available = report['resources']['total_edges_left']
        reserve = max(len(r['path']) - 1 for r in report['candidates'] if r['feasible'])
        if available < reserve:
            report['process_stop_reason'] = 'preserve_already_verified_pool_route_no_extra_budget'
            return result, report
        effective_limit = min(max_length_m, report['mission_distance_cap_m'] - report['executed_distance_m'])
        session = _RiskSession(snapshot, context, result.path, effective_limit,
                               RepairSettings(sample_limit=self.settings.sample_limit, edge_limit=self.settings.edge_limit),
                               seed, report, self.settings)
        node_id = 0
        for step in range(1, 7):
            if session.work.summary()['edges_left'] < len(session.best) - 1:
                report['process_stop_reason'] = 'resource_stop_before_state_summary'
                break
            ctx = session.request(step, 6)
            state = RepairState.from_context(ctx).to_dict()
            state.update(contract_version=VERSION, task_objective='common_R10_window_then_length',
                         mission={'executed_distance_m': session.flown, 'fixed_distance_cap_m': session.mission_cap,
                                  'remaining_distance_cap_m': session.mission_cap - session.flown},
                         current_prefix=prefix_risk(session.problem, session.best, session.speed, self.settings))
            session.extra_field_nodes += state['current_prefix']['field_nodes']
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
            record = {'step': step, 'node': node_id, 'state': state, 'proposal': proposal,
                      'construction': construction, 'feedback': feedback, 'outcome': outcome, 'next': target,
                      'error': error, 'work_delta': {k: session.work.summary()[k] - before[k]
                                                   for k in ('samples', 'edges', 'cache_hits')}}
            report['process_trace'].append(record)
            session.trace.append({'feedback': feedback})
            if target == 'stop' or outcome in ('stopped', 'policy_error', 'resource_stop'):
                break
            node_id = target
        session.work.reserved_edges = 0
        for a, b in zip(session.best, session.best[1:]):
            session.work.take_edge()
            session.work.verification_edges += 1
            if not snapshot.geometry.edge_free(a, b):
                raise RuntimeError('offline process final geometry invalid')
        chosen = select_candidate(session.records, 'prefix_v2', self.settings)
        if canonical_path(chosen['path']) != canonical_path(session.best):
            raise RuntimeError('selected route differs from common selector')
        result.path, result.length_m, result.cost = session.best, path_length(session.best), session.best_metrics['objective']
        result.elapsed_seconds = time.perf_counter() - started
        result.iterations = session.work.samples
        report.update(resources=session.work.summary(), candidates=session.records,
                      selected_candidate_id=chosen['id'], selected_prefix=chosen['prefix'],
                      best_metrics=deepcopy(session.best_metrics), searches=report['searches'] + session.searches,
                      accepted_proposals=int(not route or canonical_path(route) != canonical_path(session.best)),
                      process_accepted_proposals=session.accepted_proposals,
                      policy_errors=sum(t['error'] is not None for t in report['process_trace']),
                      risk_field_nodes=report['risk_field_nodes'] + session.extra_field_nodes,
                      numeric_compute_seconds=result.elapsed_seconds, final_geometry_verified=True,
                      process_stop_reason=report.get('process_stop_reason') or report['process_trace'][-1]['outcome'])
        return result, report


def task_evaluation(rows):
    """完整闭环的任务级门槛；快照求解不得调用此函数冒充飞行收益。"""
    if not rows:
        raise ValueError('nonempty paired closed-loop results required')
    seen = set()
    for row in rows:
        if set(row) != {'scene', 'baseline', 'candidate'} or row['scene'] in seen:
            raise ValueError('one complete pair per scene required')
        seen.add(row['scene'])
        for key in ('baseline', 'candidate'):
            m = row[key]
            if type(m['success']) is not bool:
                raise ValueError('boolean mission success required')
            for field in ('executed_length_m', 'interference_excess_integral_db_seconds'):
                if isinstance(m[field], bool) or not math.isfinite(m[field]) or m[field] < 0:
                    raise ValueError('finite nonnegative task metrics required')
    total = lambda who, field: sum(row[who][field] for row in rows)
    excess = 'interference_excess_integral_db_seconds'
    length = 'executed_length_m'
    base_excess = total('baseline', excess)
    checks = {
        'all_arrive': all(row[k]['success'] for row in rows for k in ('baseline', 'candidate')),
        'total_excess_reduction_at_least_10_percent': base_excess > 0 and total('candidate', excess) <= .9 * base_excess,
        'mean_length_at_most_1_25_baseline': total('candidate', length) <= 1.25 * total('baseline', length),
        'no_material_per_scene_regression': all(row['candidate'][excess] <= row['baseline'][excess] +
                                                max(1., .1 * row['baseline'][excess]) for row in rows),
        'zero_excess_scene_no_unnecessary_detour': all(row['candidate'][length] <= 1.05 * row['baseline'][length]
                                                      for row in rows if row['baseline'][excess] == 0),
    }
    return {'checks': checks, 'passes_task_gate': all(checks.values()),
            'note': 'resource/hard-constraint audit required separately; development gate is not significance or safety'}
