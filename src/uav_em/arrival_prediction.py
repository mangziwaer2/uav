"""固定已收证据的候选功率预测；不导入真值、搜索或控制器。"""
from __future__ import annotations

import math
import numpy as np

from .balanced_objective import PowerTimeSettings, normalized_expected_power
from .environment import Measurement
from .local_estimation import local_gp_prediction
from .repair_search import FixedObjective, WorkBudget

METHODS = ('grid_now', 'grid_arrival', 'point_now', 'point_arrival')


def received_measurements(points, geometry, now):
    """每访问网格只保留最近读数；严格拒绝未来/乱序输入。"""
    if not math.isfinite(now) or now < 0:
        raise ValueError('finite nonnegative decision time required')
    cached = {}; last = -1.
    for point in points:
        t, xy, value = point['time_seconds'], point['position_m'], point['interference_dbm']
        if not np.all(np.isfinite([*xy, t, value])) or not last <= t <= now:
            raise ValueError('future, unordered or nonfinite received point')
        if not geometry.free(xy):
            raise ValueError('measurement outside free geometry')
        cached[geometry.node(xy)] = Measurement(tuple(xy), t, value)
        last = t
    return tuple(cached.values())


def forecast_marginals(mean, std, supported, delays, settings, prior_mean):
    """未来指数核闭式条件边缘；训练集/支持不因未来查询而更新。"""
    mean, std, mask, dt = np.broadcast_arrays(np.asarray(mean, float), np.asarray(std, float),
                                              np.asarray(supported, bool), np.asarray(delays, float))
    if np.any(~np.isfinite(dt)) or np.any(dt < 0):
        raise ValueError('nonnegative finite query delays required')
    if not math.isfinite(prior_mean) or np.any(~np.isfinite(mean[mask])) or np.any(~np.isfinite(std[mask])):
        raise ValueError('finite supported posterior/prior required')
    if np.any(std[mask]**2 < settings.measured_error_db**2 - 1e-7):
        raise ValueError('supported std must include observation noise')
    prior_variance = settings.unknown_error_db**2
    latent = np.maximum(0., std[mask]**2-settings.measured_error_db**2)
    if np.any(latent > prior_variance+1e-6):
        raise ValueError('posterior latent variance exceeds prior')
    out_mean = np.full(mean.shape, prior_mean, float)
    out_std = np.full(mean.shape, settings.unknown_error_db, float)
    decay = np.exp(-dt[mask]/settings.gp_time_scale_seconds)
    out_mean[mask] = prior_mean+decay*(mean[mask]-prior_mean)
    out_std[mask] = np.sqrt(prior_variance-decay**2*(prior_variance-latent)+settings.measured_error_db**2)
    return out_mean, out_std


def query_predictions(snapshot, points, query_xy, delays, settings, prior_mean):
    xy = np.asarray(query_xy, float)
    dt = np.asarray(delays, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or dt.shape != (len(xy),) or not np.all(np.isfinite(xy)):
        raise ValueError('finite Nx2 queries and N delays required')
    if np.any(xy < 0) or np.any(xy > snapshot.geometry.high):
        raise ValueError('query outside map')
    measurements = received_measurements(points, snapshot.geometry, snapshot.time_seconds)
    grid_nodes = np.rint(xy/snapshot.geometry.config.resolution_m).astype(int)
    cols, rows = grid_nodes.T
    output = {}
    for mode in ('grid', 'point'):
        if mode == 'grid':
            mu = snapshot.estimate_dbm[rows, cols]; sigma = snapshot.uncertainty_db[rows, cols]
            supported = snapshot.supported[rows, cols]; age = snapshot.age_seconds[rows, cols]
        else:
            mu, sigma, supported, age, _ = local_gp_prediction(xy, snapshot.time_seconds, measurements, settings, prior_mean)
        mu, sigma = np.where(supported, mu, prior_mean), np.where(supported, sigma, settings.unknown_error_db)
        future_mu, future_std = forecast_marginals(mu, sigma, supported, dt, settings, prior_mean)
        for timing, m, sd in (('now', mu, sigma), ('arrival', future_mu, future_std)):
            output[mode+'_'+timing] = dict(mean_dbm=m.tolist(), std_db=sd.tolist(), supported=supported.tolist(),
                evidence_age_at_decision_seconds=age.tolist(), query_delay_seconds=dt.tolist(),
                evidence_age_at_query_seconds=np.where(supported, age+dt, -1).tolist())
    return output


def route_queries(snapshot, route, speed, delay=.5):
    if not math.isfinite(speed) or speed <= 0 or not math.isfinite(delay) or delay < 0:
        raise ValueError('positive speed and nonnegative delay required')
    if len(route) < 2:
        raise ValueError('nonempty flight required')
    problem = FixedObjective(snapshot, -80, WorkBudget())
    positions = []; dt = []; weights = []; elapsed = 0.
    for first, second in zip(route, route[1:]):
        a, b = np.asarray(first, float), np.asarray(second, float)
        lengths, _ = problem.quadrature(a, b)
        length = float(lengths.sum())
        if length <= 1e-12: continue
        midpoint = np.cumsum(lengths)-lengths/2
        positions.extend((a+(midpoint/length)[:, None]*(b-a)).tolist())
        dt.extend((delay+elapsed+midpoint/speed).tolist())
        weights.extend((lengths/speed).tolist()); elapsed += length/speed
    if not weights: raise ValueError('nonzero flight required')
    return np.asarray(positions), np.asarray(dt), np.asarray(weights)


def predict_route(snapshot, received, route, speed, belief_settings, prior_mean, score_settings=PowerTimeSettings()):
    xy, delays, weights = route_queries(snapshot, route, speed)
    prediction = query_predictions(snapshot, received, xy, delays, belief_settings, prior_mean)
    result = {}
    for method, values in prediction.items():
        mu, sd, support = np.array(values['mean_dbm']), np.array(values['std_db']), np.array(values['supported'])
        power = normalized_expected_power(mu, sd, score_settings.scale_dbm)
        t, q = float(weights.sum()), float(weights@power)
        result[method] = dict(score_seconds=t+score_settings.weight*q, flight_seconds=t, power_seconds=q,
            supported_seconds=float(weights@support), unknown_seconds=float(weights@~support),
            field_nodes=len(weights), arrival_field_predicted=method.endswith('arrival'),
            uncertainty_calibrated=False, query_coordinates='nearest_grid' if method.startswith('grid') else 'continuous_midpoint',
            supported_evidence_age_at_query_max_seconds=max((age for age, mask in zip(values['evidence_age_at_query_seconds'], support) if mask), default=None))
    return result
