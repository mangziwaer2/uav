"""Finite source-configuration search and causal flat hypothesis pools.

Only public propagation/geometry and received scalar measurements are accepted.
The fitted weights are approximate candidate evidence, not world probabilities.
"""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from .branch_belief import gaussian_logpdf, normalized
from .spatial_hypotheses import (
    MODES, SIGMA_DB, SpatialHypothesisBank, total_dbm, unit_gains,
    valid_motion, validate_points,
)

OPERATORS = ('move', 'power', 'mode', 'add', 'delete')


def configuration_key(sources):
    """Permutation-invariant parameter identity; metadata does not add mass."""
    rows = [tuple(round(float(v), 6) for v in (
        *s['anchor_m'], s['height_m'], *s['velocity_mps'],
        s['power_slope_db_s'], s['eirp_dbm'])) for s in sources]
    return json.dumps(sorted(rows), separators=(',', ':'))


def objective(component, count):
    k = 3 * component['source_count']
    return count * component['fit_rmse_db'] ** 2 / (2 * SIGMA_DB ** 2) + .5 * k * math.log(count)


def initial_logs(components, count):
    counts = {k: sum(c['source_count'] == k for c in components) for k in (0, 1, 2)}
    families = sum(v > 0 for v in counts.values())
    return normalized([-objective(c, count) - math.log(families) - math.log(counts[c['source_count']]) for c in components])


def signature_queries(context, geometry, now):
    points = [[float(x), float(y)] for y in np.linspace(20, context.height_m - 20, 4)
              for x in np.linspace(20, context.width_m - 20, 5) if geometry.free([x, y])]
    if not points:
        raise ValueError('public signature grid has no free point')
    return [dict(position_m=p, time_seconds=now + d, delay_seconds=d)
            for d in (0., 5., 10.) for p in points]


def diverse_indices(components, logs, signatures, limit=32):
    """Evidence-qualified farthest signatures; family quotas are declared policy."""
    z = np.asarray(signatures, float)
    if z.ndim != 2 or z.shape[1] != len(components) or len(logs) != len(components) or not np.all(np.isfinite(z)):
        raise ValueError('finite matching signatures required')
    if type(limit) is not int or limit < 1 or limit > 32:
        raise ValueError('pool limit must be in 1..32')
    chosen = []
    for family, quota in ((0, 1), (1, 16), (2, 15)):
        ids = sorted((i for i, c in enumerate(components) if c['source_count'] == family), key=lambda i: (-logs[i], i))
        if not ids:
            continue
        selected = ids[:1]
        eligible = [i for i in ids[1:] if logs[i] >= logs[ids[0]] - 3.]
        while eligible and len(selected) < quota:
            best = max(eligible, key=lambda i: (min(float(np.mean((z[:, i] - z[:, j]) ** 2)) for j in selected), logs[i], -i))
            selected.append(best)
            eligible.remove(best)
        selected.extend(i for i in ids if i not in selected and len(selected) < quota)
        chosen.extend(selected)
    if len(chosen) < min(limit, len(components)):
        chosen.extend(i for i in sorted(range(len(components)), key=lambda i: (-logs[i], i))
                      if i not in chosen and len(chosen) < limit)
    # A smaller limit is used only by tests; production uses all 32 slots.
    return sorted(sorted(chosen, key=lambda i: (-logs[i], i))[:limit])


def random_source(context, rng):
    mode = list(MODES)[int(rng.integers(len(MODES)))]
    vx, vy, slope = MODES[mode]
    return dict(anchor_m=[float(rng.uniform(0, context.width_m)), float(rng.uniform(0, context.height_m))],
                height_m=float(rng.choice([2., 20.])), velocity_mps=[vx, vy],
                power_slope_db_s=slope, eirp_dbm=float(rng.uniform(0, 50)), mode=mode)


def mutate(sources, operator, context, rng):
    if operator not in OPERATORS:
        raise ValueError('unknown operator')
    ss = deepcopy(sources)
    if operator == 'add' or not ss:
        if len(ss) < 2:
            ss.append(random_source(context, rng))
            return ss
        operator = 'move'
    index = int(rng.integers(len(ss)))
    s = ss[index]
    if operator == 'delete':
        del ss[index]
    elif operator == 'move':
        s['anchor_m'] = np.clip(np.asarray(s['anchor_m']) + rng.normal(0, 20, 2), [0, 0], [context.width_m, context.height_m]).tolist()
    elif operator == 'power':
        s['eirp_dbm'] = float(np.clip(s['eirp_dbm'] + rng.normal(0, 3), 0, 50))
    elif operator == 'mode':
        mode = str(rng.choice([m for m in MODES if m != s['mode']]))
        vx, vy, slope = MODES[mode]
        s.update(mode=mode, velocity_mps=[vx, vy], power_slope_db_s=slope)
    return ss


def fit_batch(context, configurations, points, now):
    """Cache geometry gains once; bounded coordinate amplitude optimization."""
    validate_points(points, now)
    if not configurations or any(len(ss) > 2 for ss in configurations):
        raise ValueError('one or more configurations of zero to two sources required')
    window = points[-16:]
    xy = [p['position_m'] for p in window]
    times = [p['time_seconds'] for p in window]
    y = np.array([p['interference_dbm'] for p in window])[:, None]
    n = len(configurations)
    sources = [s for ss in configurations for s in ss]
    flat = unit_gains(context, sources, xy, times, now) if sources else np.empty((len(window), 0))
    gains = np.full((len(window), n, 2), -1000.)
    initial = np.zeros((n, 2))
    present = np.zeros((n, 2), bool)
    offset = 0
    for j, ss in enumerate(configurations):
        for slot, s in enumerate(ss):
            gains[:, j, slot] = flat[:, offset]
            initial[j, slot] = s['eirp_dbm']
            present[j, slot] = True
            offset += 1
    powers = initial.copy()
    scalar_terms = 0
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(4):
        for slot in (0, 1):
            ids = np.flatnonzero(present[:, slot])
            if not len(ids):
                continue
            other = 1 - slot
            background = total_dbm(context.background_dbm, gains[:, ids, other], powers[ids, other])
            lo = np.maximum(0., initial[ids, slot] - 10)
            hi = np.minimum(50., initial[ids, slot] + 10)
            def loss(amplitude):
                return np.mean((total_dbm(background, gains[:, ids, slot], amplitude) - y) ** 2, axis=0)
            for _ in range(14):
                a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
                left = loss(a) <= loss(b)
                hi, lo = np.where(left, b, hi), np.where(left, lo, a)
                scalar_terms += 2 * len(window) * len(ids)
            powers[ids, slot] = (hi + lo) / 2
    predicted = total_dbm(total_dbm(context.background_dbm, gains[:, :, 0], powers[:, 0]), gains[:, :, 1], powers[:, 1])
    errors = np.sqrt(np.mean((predicted - y) ** 2, axis=0))
    components = []
    for j, ss in enumerate(configurations):
        fitted = deepcopy(ss)
        for slot, s in enumerate(fitted):
            s['eirp_dbm'] = float(powers[j, slot])
        key = configuration_key(fitted)
        components.append(dict(id='opt_' + hashlib.sha256(key.encode()).hexdigest()[:20],
                               group_id='optimized', source_count=len(fitted), sources=fitted,
                               fit_rmse_db=float(errors[j]), fitted_amplitudes=len(fitted)))
    return components, dict(propagation_source_receiver_pairs=len(window) * len(sources), amplitude_loss_scalar_terms=scalar_terms,
                            proposed_slots=n, fitted_source_count=len(sources))


def extend_archive(context, base_state, points, geometry, method, seed):
    if method not in ('random', 'evolved'):
        raise ValueError('random or evolved search only')
    now = base_state['birth_time_seconds']
    validate_points(points, now)
    queries = signature_queries(context, geometry, now)
    xy, times = [q['position_m'] for q in queries], [q['time_seconds'] for q in queries]
    archive = deepcopy(base_state['components'])
    keys = {configuration_key(c['sources']) for c in archive}
    z = SpatialHypothesisBank(context, dict(base_state, components=archive)).predict_components(xy, times)
    rng = np.random.default_rng(seed)
    records = []
    counters = dict(proposed_slots=0, propagation_source_receiver_pairs=0, amplitude_loss_scalar_terms=0,
                    signature_source_receiver_pairs=len(queries) * sum(c['source_count'] for c in archive),
                    invalid_geometry_fallbacks=0, duplicate_configurations=0)
    start = points[-16:][0]['time_seconds']
    for generation in range(4):
        logs = initial_logs(archive, min(16, len(points)))
        pool = diverse_indices(archive, logs, z)
        proposals, generation_records = [], []
        for slot in range(64):
            operator = OPERATORS[slot % len(OPERATORS)] if method == 'evolved' else 'random'
            parent = archive[int(rng.choice(pool))] if method == 'evolved' else None
            accepted = False
            for attempt in range(32):
                ss = mutate(parent['sources'], operator, context, rng) if parent else [random_source(context, rng) for _ in range(int(rng.integers(1, 3)))]
                if all(valid_motion(context, s, start, now) for s in ss):
                    accepted = True
                    break
            if not accepted:
                ss = deepcopy(parent['sources']) if parent else []
                counters['invalid_geometry_fallbacks'] += 1
            proposals.append(ss)
            generation_records.append(dict(generation=generation, slot=slot, operator=operator,
                                           parent_id=parent['id'] if parent else None, geometric_attempts=attempt + 1,
                                           geometry_fallback=not accepted))
        fitted, cost = fit_batch(context, proposals, points, now)
        for k, v in cost.items():
            if k in counters:
                counters[k] += v
        new = []
        for c, record in zip(fitted, generation_records):
            key = configuration_key(c['sources'])
            duplicate = key in keys
            record.update(candidate_id=c['id'], duplicate=duplicate, fitted_sources=c['sources'], fit_rmse_db=c['fit_rmse_db'])
            if duplicate:
                counters['duplicate_configurations'] += 1
            else:
                keys.add(key)
                archive.append(c)
                new.append(c)
            records.append(record)
        if new:
            state = dict(base_state, components=new, log_weights=[-math.log(len(new))] * len(new))
            z = np.column_stack((z, SpatialHypothesisBank(context, state).predict_components(xy, times)))
            counters['signature_source_receiver_pairs'] += len(queries) * sum(c['source_count'] for c in new)
    state = dict(deepcopy(base_state), components=archive,
                 log_weights=initial_logs(archive, min(16, len(points))).tolist(),
                 approximate_search_evidence=True)
    return state, z, queries, records, counters


class HypothesisPool:
    """Epoch replacement and one-issue/one-observe evidence; no physical tree."""
    def __init__(self, context, state, signatures, full=False):
        self.context = context
        self.state = deepcopy(state)
        self.components = self.state['components']
        self.logs = np.array(state['log_weights'], float)
        normalized(self.logs)
        self.signatures = np.asarray(signatures, float).copy()
        self.full = full
        self.pending = None
        self.last_event = -1
        self.scored_count = 0

    def issue(self, event_id, xy, time_seconds, gp):
        if self.pending is not None or type(event_id) is not int or event_id <= self.last_event:
            raise ValueError('ordered issue requires previous observation consumed')
        means = SpatialHypothesisBank(self.context, self.state).predict_components([xy], [time_seconds])[0]
        ids = list(range(len(self.components))) if self.full else diverse_indices(self.components, self.logs, self.signatures)
        weights = np.exp(self.logs[ids]).tolist()
        mass = min(1., sum(weights))
        mu, sd = means[ids].tolist(), [SIGMA_DB] * len(ids)
        if not self.full and mass < 1:
            mu.append(float(gp['mean_dbm']))
            sd.append(float(gp['std_db']))
            weights.append(1 - mass)
        weights = (np.asarray(weights) / sum(weights)).tolist()
        distribution = dict(means=mu, stds=sd, weights=weights)
        self.pending = dict(event_id=event_id, source_means=means.tolist(), prior_log_weights=self.logs.tolist(),
                            selected_indices=ids, retained_mass=mass, distribution=distribution,
                            source_count_mass={str(k): float(np.exp(self.logs[[i for i, c in enumerate(self.components) if c['source_count'] == k]]).sum()) for k in (0, 1, 2)},
                            epoch_birth=self.state['birth_time_seconds'], scored_count_before=self.scored_count)
        return deepcopy(self.pending)

    def observe(self, event_id, value):
        if self.pending is None or event_id != self.pending['event_id'] or not math.isfinite(value):
            raise ValueError('finite matching issued observation required exactly once')
        before = self.pending
        self.logs = normalized(self.logs + np.array([gaussian_logpdf(value, m, SIGMA_DB) for m in before['source_means']]))
        self.last_event = event_id
        self.scored_count += 1
        self.pending = None
        return dict(event_id=event_id, posterior_log_weights=self.logs.tolist(), scored_count=self.scored_count)
