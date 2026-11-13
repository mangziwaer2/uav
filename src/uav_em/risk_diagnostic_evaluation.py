"""仅事后比较冻结风险候选；在线模块不得调用。"""
import numpy as np

from .decision_objective import cut_route
from .diagnostic_evaluation import route_samples, truth_at_times


def _integrals(power, weights, reference_dbm, scale_dbm):
    duration = float(weights.sum())
    q = float(weights @ np.power(10., (power - scale_dbm) / 10))
    return {'flight_seconds': duration, 'power_seconds': q, 'joint_seconds': duration + q,
            'excess_db_seconds': float(weights @ np.maximum(power - reference_dbm, 0)),
            'exposure_seconds': float(weights @ (power > reference_dbm))}


def evaluate_route(world, snapshot, route, scale_dbm=-80, delay_seconds=.5):
    cfg = world.config
    speed, reference = cfg.aircraft.speed_mps, cfg.electromagnetic.interference_reference_dbm
    points, offsets, weights = route_samples(route, speed, .25)
    if not len(points):
        raise ValueError('nonempty flight required')
    now = truth_at_times(world, points, np.full(len(points), snapshot.time_seconds))
    future = truth_at_times(world, points, snapshot.time_seconds + delay_seconds + offsets)
    result = {'future_' + k: v for k, v in _integrals(future, weights, reference, scale_dbm).items()}
    result.update({'frozen_truth_' + k: v for k, v in _integrals(now, weights, reference, scale_dbm).items()})
    cols, rows = np.rint(points / snapshot.geometry.config.resolution_m).astype(int).T
    mask = snapshot.supported[rows, cols]
    mean = snapshot.estimate_dbm[rows[mask], cols[mask]]
    known_time = float(weights[mask].sum())
    result.update(supported_seconds=known_time, unknown_seconds=float(weights[~mask].sum()),
                  future_supported_power_seconds=float(weights[mask] @ np.power(10., (future[mask]-scale_dbm)/10)),
                  future_unknown_power_seconds=float(weights[~mask] @ np.power(10., (future[~mask]-scale_dbm)/10)),
                  supported_mean_mae_now_db=float(weights[mask] @ np.abs(mean-now[mask]) / known_time) if known_time else None,
                  supported_mean_mae_at_arrival_db=float(weights[mask] @ np.abs(mean-future[mask]) / known_time) if known_time else None)
    prefix, _ = cut_route(route, speed * 10)
    p, t, w = route_samples(prefix, speed, .25)
    prefix_power = truth_at_times(world, p, snapshot.time_seconds + delay_seconds + t)
    result['prefix10'] = _integrals(prefix_power, w, reference, scale_dbm)
    result.update(evaluation_kind='fixed_route_open_loop_not_online_policy', new_measurements=0,
                  new_replanning=0, maximum_step_seconds=.25, delay_seconds=delay_seconds,
                  hover_exposure_included=False, scale_dbm=scale_dbm, fixed_weight=1.,
                  threshold_is_not_safety_guarantee=True)
    return result
