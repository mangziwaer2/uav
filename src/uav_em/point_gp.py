"""Causal local constant-mean space-time GP; support is not a sensor or safety guarantee."""
from dataclasses import dataclass,asdict
import math
import numpy as np

def space_time_kernel(first_xy, first_t, second_xy, second_t, settings):
    """Matérn-3/2空间核乘指数时间核，协方差单位dB²。"""
    distance = np.linalg.norm(np.asarray(first_xy)[:, None, :] - np.asarray(second_xy)[None, :, :], axis=-1)
    radius = np.sqrt(3.) * distance / settings.gp_length_scale_m
    elapsed = np.abs(np.asarray(first_t)[:, None] - np.asarray(second_t)[None, :])
    return settings.unknown_error_db ** 2 * (1. + radius) * np.exp(
        -radius - elapsed / settings.gp_time_scale_seconds)


def validate_points(points, geometry, now):
    if not math.isfinite(now) or now < 0:
        raise ValueError('nonnegative finite decision time required')
    last = -1.
    result = []
    for p in points:
        xy, t, y = p['position_m'], p['time_seconds'], p['interference_dbm']
        if len(xy) != 2 or not np.all(np.isfinite([*xy, t, y])) or not 0 <= t <= now or t < last:
            raise ValueError('future, unordered or nonfinite point')
        if not geometry.free(xy):
            raise ValueError('point outside free geometry')
        result.append(dict(position_m=list(xy), time_seconds=float(t), interference_dbm=float(y)))
        last = t
    return result


def training_points(points, geometry, now, settings):
    """按因果时间窗选点，每网格保留最新读数，再限制总点数。"""
    points = validate_points(points, geometry, now)
    cached = {}
    for p in points:
        if now-p['time_seconds'] <= settings.max_age_seconds:
            cached[geometry.node(p['position_m'])] = p
    return sorted(cached.values(), key=lambda p:p['time_seconds'])[-settings.gp_max_points:]

def covariance(x,t,z,u,settings,geometry,mode):
    return space_time_kernel(x,t,z,u,settings)

class LocalPredictor:
    def __init__(self, points, geometry, now, settings, mode, background=-100.):
        if mode != 'local_level_gp': raise ValueError('only local_level_gp is active')
        self.geometry, self.now, self.settings, self.mode, self.background = geometry, now, settings, mode, background
        self.points = training_points(points, geometry, now, settings)
        self.xy = np.array([p['position_m'] for p in self.points],float).reshape(-1,2)
        self.times = np.array([p['time_seconds'] for p in self.points])
        self.mean_level = background
        if not self.points: return
        values = np.array([p['interference_dbm'] for p in self.points])
        K = covariance(self.xy, self.times, self.xy, self.times, settings, geometry, mode)
        K.flat[::len(K)+1] += settings.measured_error_db**2+1e-8
        self.inverse = np.linalg.solve(K, np.eye(len(K)))
        self.inverse_ones = self.inverse@np.ones(len(K))
        self.level_precision = float(self.inverse_ones.sum())
        self.mean_level = float(self.inverse_ones@values/self.level_precision)
        self.alpha = self.inverse@(values-self.mean_level)


class FrozenGP:
    """Local constant-mean GP at arbitrary future receiver times; fixed training prefix."""
    def __init__(self, points, geometry, now, settings, background):
        self.model = LocalPredictor(points, geometry, now, settings, 'local_level_gp', background)
        self.now = now

    def predict(self, xy, times):
        m = self.model
        xy = np.asarray(xy, float)
        times = np.asarray(times, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or times.shape != (len(xy),):
            raise ValueError('Nx2 receivers and N query times required')
        if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(times)) or np.any(times < self.now):
            raise ValueError('finite current/future queries required')
        if np.any(xy < 0) or np.any(xy > m.geometry.high):
            raise ValueError('receiver outside map')
        mean = np.full(len(xy), m.background)
        std = np.full(len(xy), m.settings.unknown_error_db)
        supported = np.zeros(len(xy), bool)
        if m.points:
            for start in range(0, len(xy), 2048):
                q = xy[start:start + 2048]
                tt = times[start:start + 2048]
                cross = covariance(q, tt, m.xy, m.times, m.settings, m.geometry, m.mode)
                mu = m.mean_level + cross @ m.alpha
                variance = m.settings.unknown_error_db ** 2 - np.sum((cross @ m.inverse) * cross, axis=1)
                variance += (1 - cross @ m.inverse_ones) ** 2 / m.level_precision
                sigma = np.sqrt(np.maximum(0, variance) + m.settings.measured_error_db ** 2)
                distance = np.linalg.norm(q[:, None, :] - m.xy[None, :, :], axis=-1)
                nearest = np.argmin(distance, axis=1)
                support = ((distance[np.arange(len(q)), nearest] <= m.settings.support_radius_m)
                           & (tt - m.times[nearest] <= m.settings.max_age_seconds)
                           & np.array([m.geometry.free(p) for p in q]))
                sl = slice(start, start + len(q))
                mean[sl] = np.where(support, mu, mean[sl])
                std[sl] = np.where(support, sigma, std[sl])
                supported[sl] = support
        return mean, std, supported


@dataclass
class EstimateSnapshot:
    time_seconds: float
    estimate_dbm: np.ndarray
    uncertainty_db: np.ndarray
    supported: np.ndarray
    geometry: object
    recent_point_measurements: tuple
    estimation_details: dict

class PointGP:
    """Estimator seam: observed points in, snapshot/arbitrary-time query out."""
    def __init__(self, geometry, settings, background=-100.):
        self.geometry, self.settings, self.background = geometry, settings, background
        self.points = []

    def observe(self, measurement):
        point = {k: asdict(measurement)[k] for k in ('position_m', 'time_seconds', 'interference_dbm')}
        if self.points and point['time_seconds'] <= self.points[-1]['time_seconds']:
            raise ValueError('each observation must be new and ordered')
        self.points.append(point)

    def freeze(self, now):
        gp = FrozenGP(self.points, self.geometry, now, self.settings, self.background)
        xy = self.geometry.grid().reshape(-1, 2)
        mu, sd, support = gp.predict(xy, np.full(len(xy), now))
        shape = self.geometry.grid().shape[:-1]
        snapshot = EstimateSnapshot(now, mu.reshape(shape), sd.reshape(shape), support.reshape(shape),
            self.geometry, tuple(self.points[-16:]), {})
        snapshot.estimation_details = dict(method='local_level_gp', calibrated=False,
            support_radius_m=self.settings.support_radius_m,
            maximum_age_seconds=self.settings.max_age_seconds)
        return snapshot, gp

