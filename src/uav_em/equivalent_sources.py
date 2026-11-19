"""Causal additive power bases; an equivalent field, never emitter localization."""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import least_squares

from .extrapolation import LocalPredictor, covariance, validate_points

SCALES = (40., 100., 240.)
VELOCITIES = ((0., 0.), (3., 0.), (-3., 0.), (0., 3.), (0., -3.))
METHODS = ('eq_static', 'eq_dynamic', 'eq_uniform', 'dual_region', 'online_gp', 'persistence')
SIGMA = math.sqrt(17.)


def normalize(logs):
    values = np.asarray(logs, float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError('finite nonempty log weights required')
    values = values - values.max()
    return values - math.log(float(np.exp(values).sum()))


def field(scenario, xy, times, birth, background):
    """Sum nonnegative contributions in mW; times may vary per receiver."""
    xy = np.asarray(xy, float)
    times = np.broadcast_to(np.asarray(times, float), (len(xy),))
    if xy.shape != (len(xy), 2) or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(times)):
        raise ValueError('finite receiver coordinates and times required')
    power = np.full(len(xy), 10 ** (background / 10))
    for source in scenario['sources']:
        if source['scale_m'] <= 0 or not np.all(np.isfinite([
                *source['center_m'], *source['velocity_mps'], source['scale_m'], source['peak_dbm']])):
            raise ValueError('finite source and positive scale required')
        center = np.asarray(source['center_m']) + (times - birth)[:, None] * source['velocity_mps']
        d2 = np.sum((xy - center) ** 2, axis=1)
        power += 10 ** (source['peak_dbm'] / 10) * np.exp(-d2 / (2 * source['scale_m'] ** 2))
    return 10 * np.log10(power)


def fit(points, geometry, now, background):
    points = validate_points(points, geometry, now)[-16:]
    if not points:
        raise ValueError('received measurements required')
    started = time.perf_counter()
    birth = points[-1]['time_seconds']
    xy = np.asarray([p['position_m'] for p in points])
    times = np.asarray([p['time_seconds'] for p in points])
    y = np.asarray([p['interference_dbm'] for p in points])
    target = 10 ** ((y + 80) / 10)
    bg = 10 ** ((background + 80) / 10)
    centers = np.array([(x, z) for z in np.linspace(0, geometry.high[1], 7)
                        for x in np.linspace(0, geometry.high[0], 9)])
    empty_loss = float(np.sum((background - y) ** 2) / (2 * SIGMA ** 2))
    scenarios = [dict(id='background', sources=[], loss=empty_loss, static=True)]
    resources = dict(scenario_slots=15, least_squares_calls=0, nfev=0,
                     unsuccessful_optimizations=0, correlation_scalar_terms=0, basis_receiver_terms=0)
    for scale in SCALES:
        for velocity in VELOCITIES:
            shifted = centers[None, :, :] + (times - birth)[:, None, None] * np.asarray(velocity)
            A = np.exp(-np.sum((xy[:, None, :] - shifted) ** 2, axis=2) / (2 * scale ** 2))
            resources['basis_receiver_terms'] += int(A.size)
            chosen, peaks = [], []
            prediction = np.full(len(y), bg)
            best_sources, best_loss = [], empty_loss
            for _ in range(4):
                # Relative-power residual approximates a dBm residual during basis selection.
                weighted = A / target[:, None]
                residual = (target - prediction) / target
                norms = np.linalg.norm(weighted, axis=0)
                scores = np.divide(weighted.T @ residual, norms,
                                   out=np.full(len(centers), -np.inf), where=norms > 1e-14)
                scores[chosen] = -np.inf
                resources['correlation_scalar_terms'] += int(A.size)
                index = int(np.argmax(scores))
                if not np.isfinite(scores[index]) or scores[index] <= 0:
                    break
                numerator = float(A[:, index] @ (target - prediction))
                amplitude = max(1e-7, numerator / max(1e-30, float(A[:, index] @ A[:, index])))
                chosen.append(index)
                peaks.append(float(np.clip(-80 + 10 * np.log10(amplitude), -150, -30)))
                B = A[:, chosen]

                def objective(db):
                    relative = bg + B @ (10 ** ((db + 80) / 10))
                    return (-80 + 10 * np.log10(relative) - y) / SIGMA

                solved = least_squares(objective, peaks, bounds=(-150., -30.), max_nfev=60)
                resources['least_squares_calls'] += 1
                resources['nfev'] += int(solved.nfev)
                resources['unsuccessful_optimizations'] += int(not solved.success)
                peaks = solved.x.tolist()
                prediction = bg + B @ (10 ** ((solved.x + 80) / 10))
                loss = float(.5 * np.sum(objective(solved.x) ** 2)
                             + .5 * 3 * len(chosen) * math.log(len(y)))
                if loss < best_loss:
                    best_loss = loss
                    best_sources = [dict(center_m=centers[i].tolist(), scale_m=scale,
                                         velocity_mps=list(velocity), peak_dbm=float(peak))
                                    for i, peak in zip(chosen, peaks)]
            # All empty fits represent the same background scene; do not multiply its prior.
            if best_sources:
                scenarios.append(dict(id=f's{scale:g}_v{velocity[0]:g}_{velocity[1]:g}',
                                      sources=best_sources, loss=best_loss, static=velocity == (0., 0.)))
    resources['elapsed_seconds'] = time.perf_counter() - started
    return dict(birth_seconds=birth, valid_until_seconds=birth + 10, background_dbm=float(background),
                fit_received_count=len(points), scenarios=scenarios,
                log_weights=normalize([-s['loss'] for s in scenarios]).tolist(), resources=resources)


def source_distribution(state, xy, times, method='eq_dynamic'):
    if method not in ('eq_static', 'eq_dynamic', 'eq_uniform'):
        raise ValueError('unknown source mixture')
    scenarios = state['scenarios']
    ids = [i for i, s in enumerate(scenarios) if method != 'eq_static' or s['static']]
    logs = np.zeros(len(ids)) if method == 'eq_uniform' else np.asarray(state['log_weights'])[ids]
    weights = np.exp(normalize(logs))
    means = np.array([field(scenarios[i], xy, times, state['birth_seconds'], state['background_dbm']) for i in ids])
    return dict(means=means, stds=np.full_like(means, SIGMA), weights=weights)


def update(state, xy, now, observed):
    if not state['birth_seconds'] < now <= state['valid_until_seconds']:
        raise ValueError('only post-birth readings inside epoch may update')
    if now <= state.get('last_updated_seconds', state['birth_seconds']):
        raise ValueError('duplicate or unordered observation')
    if not math.isfinite(observed):
        raise ValueError('finite observed power required')
    distribution = source_distribution(state, [xy], [now])
    residual = (observed - distribution['means'][:, 0]) / SIGMA
    state['log_weights'] = normalize(np.asarray(state['log_weights']) - .5 * residual ** 2).tolist()
    state['last_updated_seconds'] = float(now)


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


def predictions(state, gp, xy, times, last_power):
    """Return KxN Gaussian marginals; no independence multiplication across estimators."""
    xy, times = np.asarray(xy, float), np.asarray(times, float)
    mu, sd, support = gp.predict(xy, times)
    valid = (times >= state['birth_seconds']) & (times <= state['valid_until_seconds'])
    local = dict(means=mu[None, :], stds=sd[None, :], weights=np.ones(1))
    persistence = dict(means=np.full((1, len(xy)), last_power), stds=np.full((1, len(xy)), SIGMA), weights=np.ones(1))
    result = dict(online_gp=local, persistence=persistence)
    for method in ('eq_static', 'eq_dynamic', 'eq_uniform'):
        d = source_distribution(state, xy, times, method)
        d['means'][:, ~valid] = mu[~valid]
        d['stds'][:, ~valid] = sd[~valid]
        result[method] = d
    source = result['eq_dynamic']
    means = np.vstack((source['means'], persistence['means']))
    stds = np.vstack((source['stds'], persistence['stds']))
    weights = np.r_[.8 * source['weights'], .2]
    use_local = support | ~valid
    means[:, use_local] = mu[use_local]
    stds[:, use_local] = sd[use_local]
    result['dual_region'] = dict(means=means, stds=stds, weights=weights)
    return result, dict(gp_supported=support, source_valid=valid)


def expected_relative_power(d, scale=-80.):
    a = math.log(10) / 10
    return np.asarray(d['weights']) @ np.exp(a * (np.asarray(d['means']) - scale)
                                            + .5 * a * a * np.asarray(d['stds']) ** 2)


def serial_distribution(d):
    return {k: np.asarray(v).tolist() for k, v in d.items()}
