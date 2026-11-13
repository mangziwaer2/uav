"""执行冻结后的连续真值评价；不得导入在线决策路径。"""
import numpy as np

from .decision_objective import cut_route
from .diagnostic_evaluation import route_samples, truth_at_times
from .planning import path_length


def evaluate_motions(world, motions, scale_dbm=-80):
    cfg = world.config
    speed = cfg.aircraft.speed_mps
    q = excess = exposure = duration = 0.
    paths = []
    for motion in motions:
        path = motion['traversed_m']
        points, offsets, weights = route_samples(path, speed, .25)
        if not len(points): continue
        start = motion['end_time_seconds'] - float(weights.sum())
        power = truth_at_times(world, points, start + offsets)
        q += float(weights @ np.power(10., (power - scale_dbm) / 10))
        excess += float(weights @ np.maximum(power - cfg.electromagnetic.interference_reference_dbm, 0))
        exposure += float(weights @ (power > cfg.electromagnetic.interference_reference_dbm))
        duration += float(weights.sum())
        if paths:
            if not np.allclose(paths[-1], path[0], rtol=0, atol=1e-7): raise ValueError('motion trace is discontinuous')
            paths.extend(path[1:])
        else: paths.extend(path)
    if not paths: raise ValueError('nonempty actual flight required')
    result = {'flight_power_integral_normalized_seconds': q,
              'flight_received_power_integral_mw_seconds': q * 10**(scale_dbm/10),
              'flight_seconds_from_trace': duration,
              'reintegrated_excess_db_seconds': excess, 'reintegrated_exposure_seconds': exposure,
              'integration_max_step_seconds': .25, 'physical_absorption_or_failure_model': False}
    start, goal = np.asarray(cfg.map.start_m), np.asarray(cfg.map.goal_m)
    direction = (goal-start)/np.linalg.norm(goal-start)
    for distance in (50,150):
        front,_ = cut_route(paths, distance)
        arcs = path_length(front)
        endpoint = np.asarray(front[-1])
        result[f'first{distance}_flown_m'] = arcs
        result[f'first{distance}_goal_distance_progress_m'] = float(np.linalg.norm(goal-start)-np.linalg.norm(goal-endpoint))
        result[f'first{distance}_backward_projection_m'] = sum(max(0.,-float((np.asarray(b)-a)@direction)) for a,b in zip(front,front[1:]))
    return result
