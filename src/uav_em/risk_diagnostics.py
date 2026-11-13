"""风险诊断的因果状态与共同评分；不读取world、真值或未来。"""
from copy import deepcopy
import math

import numpy as np

from .balanced_objective import PowerTimeSettings, choose, power_time_score
from .decision_objective import prefix_risk
from .repair_process import canonical_path
from .repair_search import FixedObjective, WorkBudget


def select_decisions(contexts, reference_dbm=-80):
    """只用当时公开时刻/读数取样；不按将来暴露挑状态。"""
    conditions = {
        'after20': lambda c: c['time_seconds'] >= 20,
        'first_observed_reference': lambda c: c['latest_observation']['interference_dbm'] >= reference_dbm,
        'after30': lambda c: c['time_seconds'] >= 30,
        'after40': lambda c: c['time_seconds'] >= 40,
    }
    selected, missing = {}, []
    for label, condition in conditions.items():
        index = next((i for i, c in enumerate(contexts) if condition(c)), None)
        if index is None:
            missing.append(label)
        else:
            selected.setdefault(index, []).append(label)
    return selected, missing


def received_before_decision(events, decision_index):
    """按事件顺序截断；同时间戳但在决策之后的测量也不能读取。"""
    result = []
    for event in events:
        if event['event'] == 'decision' and event['index'] == decision_index:
            return result
        if event['event'] == 'observation':
            if event.get('directional_scan') is not None:
                raise ValueError('power-only observations required')
            result.append({k: deepcopy(event[k]) for k in
                           ('position_m', 'time_seconds', 'interference_dbm')})
    raise ValueError('decision event absent; cannot infer a time cutoff')


def score_pool(snapshot, context, records, energy_limit_m, mission_remaining_m,
               settings=PowerTimeSettings()):
    """扩大诊断池独立记账；原J仅诊断，v3为共同接受分。"""
    if any(not math.isfinite(x) or x < 0 for x in (energy_limit_m, mission_remaining_m)):
        raise ValueError('finite nonnegative remaining caps required')
    work = WorkBudget(2100, 1_000_000)
    problem = FixedObjective(snapshot, context['electromagnetic_estimate']['interference_reference_dbm'], work)
    speed = context['vehicle']['speed_mps']
    start, goal = context['vehicle']['position_m'], context['vehicle']['goal_m']
    output, seen, field_nodes = [], {}, 0
    for source in records:
        record = deepcopy(source)
        path = record['path']
        metrics, failure = None, None
        if not path:
            failure = 'no_search_path'
        elif not (np.allclose(path[0], start, rtol=0, atol=1e-8) and
                  np.allclose(path[-1], goal, rtol=0, atol=1e-8)):
            failure = 'endpoint_mismatch'
        else:
            metrics = problem.route_metrics(path)
            if metrics is None:
                failure = 'geometry_invalid'
            elif metrics['length_m'] > energy_limit_m + 1e-8:
                failure = 'energy_limit'
            elif metrics['length_m'] > mission_remaining_m + 1e-8:
                failure = 'mission_length_limit'
        record.update(metrics=metrics, feasible=failure is None, failure=failure,
                      balance=None, prefix=None, duplicate_of=None)
        if path:
            fingerprint = canonical_path(path)
            record['duplicate_of'] = seen.get(fingerprint)
            seen.setdefault(fingerprint, record['id'])
        if metrics is not None:
            record['balance'] = power_time_score(problem, path, speed, settings)
            record['prefix'] = prefix_risk(problem, path, speed)
            field_nodes += record['balance']['field_nodes'] + record['prefix']['field_nodes']
        output.append(record)
    selected = choose(output)
    if selected is None:
        raise ValueError('incumbent must remain feasible')
    source = choose([r for r in output if r['family'] in ('incumbent', 'original_pool')])
    return output, {'expanded_best_id': selected['id'], 'original_best_id': source['id'],
                    'energy_limit_m': energy_limit_m, 'mission_remaining_m': mission_remaining_m,
                    'effective_limit_m': min(energy_limit_m, mission_remaining_m)}, {
                        'pool_rescoring': work.summary(), 'field_nodes': field_nodes}


def summarize(records, selection):
    """只在冻结候选经事后评价后使用；不把橙色参考回传控制器。"""
    valid = [r for r in records if r['feasible']]
    by = {r['id']: r for r in valid}
    base = by['incumbent']
    base_c = base['balance']['score_seconds']
    base_e = base['evaluation']['future_excess_db_seconds']
    base_j = base['evaluation']['future_joint_seconds']
    opportunities = [r for r in valid if base_e > 1e-8 and
                     r['evaluation']['future_excess_db_seconds'] <= .9 * base_e and
                     r['evaluation']['future_joint_seconds'] < base_j - 1e-7]
    accepted = [r for r in opportunities if r['balance']['score_seconds'] < base_c - 1e-7]
    material = [r for r in accepted if r['balance']['score_seconds'] <= .99 * base_c]
    best_true = min(valid, key=lambda r: (r['evaluation']['future_joint_seconds'], r['metrics']['length_m']))
    original, expanded = by[selection['original_best_id']], by[selection['expanded_best_id']]
    return {
        'feasible_count': len(valid), 'candidate_count': len(records),
        'unique_feasible_count': len({canonical_path(r['path']) for r in valid}),
        'original_best_id': original['id'], 'expanded_best_id': expanded['id'],
        'posthoc_true_joint_best_id': best_true['id'],
        'truth_opportunity_ids': [r['id'] for r in opportunities],
        'v3_acceptable_truth_opportunity_ids': [r['id'] for r in accepted],
        'v3_one_percent_truth_opportunity_ids': [r['id'] for r in material],
        'v3_rejected_truth_opportunity_ids': [r['id'] for r in opportunities if r not in accepted],
        'expanded_pool_lowers_current_score': expanded['balance']['score_seconds'] < original['balance']['score_seconds'] - 1e-7,
        'expanded_selection_improves_true_joint': expanded['evaluation']['future_joint_seconds'] < original['evaluation']['future_joint_seconds'] - 1e-7,
        'expanded_selection_is_truth_opportunity': expanded['id'] in {r['id'] for r in opportunities},
        'interpretation': 'finite-pool fixed-route open-loop diagnosis; posthoc reference unavailable online',
    }
