"""Task-focused source summaries; no measurements, fitting or truth queries.

Detailed configurations are display/decision hypotheses, not identified emitters.
Moment residuals preserve candidate mass and dBm prediction moments, not modes.
"""
from copy import deepcopy
from itertools import permutations
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def point_segment_distance(point, a, b):
    point, a, b = (np.asarray(v, float) for v in (point, a, b))
    d = b - a
    fraction = np.clip(float((point - a) @ d) / float(d @ d), 0., 1.) if d @ d else 0.
    return float(np.linalg.norm(point - a - fraction * d))


def segment_distance(a, b, c, d):
    a, b, c, d = (np.asarray(v, float) for v in (a, b, c, d))
    cross = lambda u, v: float(u[0] * v[1] - u[1] * v[0])
    denominator = cross(b - a, d - c)
    if abs(denominator) > 1e-12:
        t, u = cross(c - a, d - c) / denominator, cross(c - a, b - a) / denominator
        if 0 <= t <= 1 and 0 <= u <= 1:
            return 0.
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def corridor(position, goal, speed, horizon=10., radius=80.):
    position, goal = np.asarray(position, float), np.asarray(goal, float)
    if position.shape != (2,) or goal.shape != (2,) or not np.all(np.isfinite([position, goal])) or min(speed, horizon, radius) <= 0:
        raise ValueError('finite positions and positive corridor scales required')
    delta = goal - position
    distance = float(np.linalg.norm(delta))
    end = position + delta * min(1., speed * horizon / distance) if distance else position.copy()
    return dict(start_m=position.tolist(), end_m=end.tolist(), radius_m=float(radius), horizon_seconds=float(horizon))


def current_source(source, birth, now):
    dt = now - birth
    return (np.asarray(source['anchor_m'], float) + dt * np.asarray(source['velocity_mps'], float),
            float(source['eirp_dbm'] + dt * source['power_slope_db_s']))


def region_mask(components, birth, now, area):
    if not math.isfinite(now) or not birth <= now <= birth + 10:
        raise ValueError('source summary outside its epoch horizon')
    horizon = min(area['horizon_seconds'], birth + 10 - now)
    a, b = area['start_m'], area['end_m']
    mask = []
    for component in components:
        inside = False
        for source in component['sources']:
            p, _ = current_source(source, birth, now)
            q = p + horizon * np.asarray(source['velocity_mps'])
            if segment_distance(a, b, p, q) <= area['radius_m']:
                inside = True
                break
        mask.append(inside)
    return np.asarray(mask, bool)


def _arrays(items, now):
    xy, power, velocity = [], [], []
    for item in items:
        values = [current_source(s, item['birth'], now) for s in item['sources']]
        xy.append([v[0] for v in values]);power.append([v[1] for v in values])
        velocity.append([s['velocity_mps'] for s in item['sources']])
    return np.array(xy), np.array(power), np.array(velocity)


def associate(previous, components, indices, birth, now):
    """One-to-one, permutation-invariant config association at one common time."""
    answer = {}
    for count in (1, 2):
        old = [p for p in previous if len(p['sources']) == count]
        ids = [i for i in indices if len(components[i]['sources']) == count]
        if not old or not ids:
            continue
        new = [dict(sources=components[i]['sources'], birth=birth) for i in ids]
        ox, op, ov = _arrays(old, now);nx, npower, nv = _arrays(new, now)
        best = np.full((len(old), len(ids)), 1e6)
        for order in permutations(range(count)):
            order = list(order)
            distance = np.linalg.norm(ox[:, None] - nx[None, :, order, :], axis=-1)
            power = np.abs(op[:, None] - npower[None, :, order])
            velocity = np.linalg.norm(ov[:, None] - nv[None, :, order, :], axis=-1)
            valid = np.all((distance <= 80.) & (power <= 10.), axis=-1)
            cost = np.mean(distance / 80. + power / 10. + velocity / 6., axis=-1)
            best = np.minimum(best, np.where(valid, cost, 1e6))
        rows, columns = linear_sum_assignment(best)
        for r, c in zip(rows, columns):
            if best[r, c] < 1e6:
                answer[ids[c]] = dict(track_id=old[r]['track_id'], cost=float(best[r, c]))
    return answer


class DetailTracker:
    """Association/hysteresis changes only detailed selection, never evidence."""
    def __init__(self, prefix, stable=False, limit=31):
        if type(limit) is not int or not 1 <= limit <= 31:
            raise ValueError('detail limit must leave one residual slot')
        self.prefix, self.stable, self.limit = prefix, stable, limit
        self.previous, self.counter, self.last_time = [], 0, -math.inf

    def select(self, components, logs, indices, birth, now):
        logs = np.asarray(logs, float)
        if logs.shape != (len(components),) or not np.all(np.isfinite(logs)) or now < self.last_time:
            raise ValueError('ordered finite candidate evidence required')
        ranked = sorted(set(indices), key=lambda i: (-logs[i], i))
        kept = []
        if self.stable and ranked:
            linked = associate(self.previous, components, ranked, birth, now)
            cutoff = logs[ranked[min(self.limit, len(ranked)) - 1]] - math.log(2.)
            kept = [i for i in ranked if i in linked and logs[i] >= cutoff][:self.limit]
        chosen = sorted((kept + [i for i in ranked if i not in kept])[:self.limit])
        links = associate(self.previous, components, chosen, birth, now)
        old_count = len(self.previous)
        changed = 1. - 2. * len(links) / (old_count + len(chosen)) if old_count + len(chosen) else 0.
        records = []
        for i in chosen:
            if i in links:
                identity = links[i]['track_id']
            else:
                identity = f'{self.prefix}_{self.counter:05d}';self.counter += 1
            records.append(dict(index=i, track_id=identity, sources=deepcopy(components[i]['sources']),
                                birth=birth, linked=i in links, association_cost=links.get(i, {}).get('cost')))
        result = dict(selected_indices=chosen, tracks=deepcopy(records), previous_count=old_count,
                      matched_count=len(links), churn=changed, has_previous=self.last_time > -math.inf)
        self.previous, self.last_time = records, now
        return result


def compress(means, weights, selected, sigma_db, gp=None):
    """Keep explicit components and moment-match every unselected component."""
    mu, w = np.asarray(means, float), np.asarray(weights, float)
    if mu.ndim != 1 or mu.shape != w.shape or not len(mu) or not np.all(np.isfinite([mu, w])) or np.any(w < 0) or sigma_db <= 0 or not np.isclose(w.sum(), 1., atol=1e-9):
        raise ValueError('finite normalized mixture required')
    if len(set(selected)) != len(selected) or any(i < 0 or i >= len(mu) for i in selected) or len(selected) > 31:
        raise ValueError('at most 31 unique valid explicit components')
    ids = np.asarray(selected, int)
    mask = np.ones(len(mu), bool);mask[ids] = False
    residual = float(w[mask].sum())
    mm, ss, ww = mu[ids].tolist(), [float(sigma_db)] * len(ids), w[ids].tolist()
    residual_summary = None
    if residual > 0:
        mean = float(w[mask] @ mu[mask] / residual)
        variance = float(w[mask] @ ((mu[mask] - mean) ** 2 + sigma_db ** 2) / residual)
        if gp is not None:
            mean, variance = float(gp['mean_dbm']), float(gp['std_db']) ** 2
        residual_summary = dict(mass=residual, mean_dbm=mean, std_db=math.sqrt(max(0., variance)),
                                component_count=int(mask.sum()), representation='gp' if gp is not None else 'moment_gaussian')
        mm.append(mean);ss.append(residual_summary['std_db']);ww.append(residual)
    total = sum(ww)
    return dict(distribution=dict(means=mm, stds=ss, weights=[v / total for v in ww]),
                detailed_mass=float(w[ids].sum()), residual=residual_summary,
                component_count_before=len(mu), gaussian_count_after=len(mm),
                residual_mass=residual)
