"""已保存结构试验的只读诊断；不生成候选、不接入飞行控制器。"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from .belief import ElectromagneticBelief
from .environment import Measurement
from .planning import path_length
from .repair_process import canonical_path
from .risk_diagnostics import received_before_decision
from .structural_policy import ExpressionPolicy


def causal_snapshot(config, geometry, episode, decision_index):
    decision = episode['decisions'][decision_index]
    points = received_before_decision(episode['events'], decision_index)
    belief = ElectromagneticBelief(config, geometry)
    for p in points:
        if p['time_seconds'] > decision['time_seconds']:
            raise ValueError('future observation in decision prefix')
        belief.update(Measurement(tuple(p['position_m']), p['time_seconds'], p['interference_dbm']))
    snapshot = belief.snapshot(decision['time_seconds'])
    if snapshot.observation_count != decision['context']['electromagnetic_estimate']['observation_count']:
        raise ValueError('observation count does not reconstruct')
    return snapshot, points


def common_candidates(report):
    return [r for r in report['candidates'] if not r['id'].startswith(
        ('generated_candidate_', 'reconnection_candidate_'))]


def pool_signature(records):
    """只比较实际候选/代价/失败，不把耗时等记录字段当因果状态。"""
    return [(r['id'], r['feasible'], canonical_path(r['path']), r.get('failure'),
             r.get('balance'), r.get('metrics')) for r in records]


def compare_decisions(first, first_points, second, second_points):
    a, b = first['repair_optimization'], second['repair_optimization']
    state = dict(
        same_time=first['time_seconds'] == second['time_seconds'],
        same_vehicle=first['context']['vehicle'] == second['context']['vehicle'],
        same_observation_prefix=first_points == second_points,
        same_public_geometry=first['context']['map'] == second['context']['map'],
        same_common_pool=pool_signature(common_candidates(a)) == pool_signature(common_candidates(b)),
        same_common_pool_seed=a['common_pool_seed'] == b['common_pool_seed'],
        same_common_pool_resources=a['starting_resources_already_paid'] == b['starting_resources_already_paid'],
        same_frozen_base=canonical_path(a['frozen_base_route']) == canonical_path(b['frozen_base_route']),
        same_score_settings=a['power_time_settings'] == b['power_time_settings'])
    state['strict_same_state'] = all(state.values())
    state['same_selected_route'] = canonical_path(a['path']) == canonical_path(b['path'])
    return state


def score_delta(before, after):
    keys = ('remaining_flight_seconds', 'supported_power_seconds',
            'unknown_prior_power_seconds', 'supported_seconds', 'unknown_seconds', 'score_seconds')
    result = {k: after[k] - before[k] for k in keys}
    expected = result['remaining_flight_seconds'] + result['supported_power_seconds'] + result['unknown_prior_power_seconds']
    if not math.isclose(expected, result['score_seconds'], abs_tol=1e-7):
        raise ValueError('lambda1 score decomposition does not add up')
    return result


def entry_roundoff_check(report, tolerance_m=1e-9):
    """只重算原程序的已记录特征，不生成新策略或运行搜索。"""
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError('finite nonnegative tolerance required')
    if 'generated_program' not in report:
        return None
    policy = ExpressionPolicy(report['generated_program'])
    build = report['build_trace']
    original = [policy.value('entry_score', r) for r in build['segments']]
    if original != build['entry_ranks']:
        raise ValueError('recorded entry expression does not reproduce')
    modified = deepcopy(build['segments'])
    for row in modified:
        if abs(row['detour']) <= tolerance_m:
            row['detour'] = 0.
    ranks = [policy.value('entry_score', r) for r in modified]
    index = max(range(len(ranks)), key=lambda k: ranks[k])
    offset = policy.value('entry_offset', dict(index=index, count=len(ranks)))
    return dict(recorded_segment=build['selected_segment'], recorded_start=max(0, build['selected_segment'] + int(build['entry_offset'])),
                zeroed_roundoff_segment=index, zeroed_roundoff_start=max(0, index + int(offset)),
                changed=index != build['selected_segment'], tolerance_m=tolerance_m,
                new_searches=0, original_ranks=original, zeroed_roundoff_ranks=ranks)


def clip_motion(motion, speed_mps, start_seconds, end_seconds):
    """按真实执行折线和仿真时间裁剪；悬停不伪装为飞行。"""
    if not math.isfinite(speed_mps) or speed_mps <= 0 or not 0 <= start_seconds <= end_seconds:
        raise ValueError('positive speed and ordered nonnegative interval required')
    from .decision_objective import cut_route
    path = motion['traversed_m']
    duration = path_length(path) / speed_mps
    start = motion['end_time_seconds'] - duration
    lo, hi = max(start_seconds, start), min(end_seconds, motion['end_time_seconds'])
    if hi - lo <= 1e-10:
        return None
    _, tail = cut_route(path, max(0., lo - start) * speed_mps)
    front, _ = cut_route(tail, (hi - lo) * speed_mps)
    return dict(end_time_seconds=hi, traversed_m=front)


def position_on_motions(motions, speed_mps, time_seconds):
    """仅从已执行轨迹恢复位置，间隙保持上次端点。"""
    if not motions or not math.isfinite(time_seconds) or time_seconds < 0:
        raise ValueError('nonempty motions and nonnegative finite time required')
    last = np.asarray(motions[0]['traversed_m'][0], float)
    for m in motions:
        route = m['traversed_m']
        beginning = m['end_time_seconds'] - path_length(route) / speed_mps
        if time_seconds < beginning:
            return last.tolist()
        if time_seconds <= m['end_time_seconds']:
            distance = max(0., time_seconds - beginning) * speed_mps
            for a, b in zip(route, route[1:]):
                a, b = np.asarray(a, float), np.asarray(b, float)
                length = float(np.linalg.norm(b-a))
                if length > 0 and distance <= length:
                    return (a+(b-a)*distance/length).tolist()
                distance -= length
            return list(route[-1])
        last = np.asarray(route[-1], float)
    return last.tolist()
