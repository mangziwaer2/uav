"""仅事后评价冻结路线前段，不构造或更新估计器。"""
import numpy as np

from .decision_objective import cut_route
from .diagnostic_evaluation import route_samples, truth_at_times


def evaluate_frozen_prefix(world, route, decision_time_seconds):
    cfg = world.config
    front, _ = cut_route(route, cfg.aircraft.speed_mps * 10)
    points, offsets, weights = route_samples(front, cfg.aircraft.speed_mps, .25)
    if not len(points):
        raise ValueError('nonempty flight route required')
    power = truth_at_times(world, points, decision_time_seconds + .5 + offsets)
    reference = cfg.electromagnetic.interference_reference_dbm
    return {'future_excess_db_seconds': float(weights @ np.maximum(power - reference, 0)),
            'future_exposure_seconds': float(weights @ (power > reference)),
            'flight_seconds': float(weights.sum()),
            'length_m': float(weights.sum() * cfg.aircraft.speed_mps)}
