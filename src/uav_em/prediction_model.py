"""CPU residual predictors from public point histories; no simulator capabilities."""
from __future__ import annotations

import numpy as np

from .buildings import Building, point_segment_distance, ring_edges


LAGS = (0., 1., 2., 4., 8., 12., 20., 32.)
SEEDS = (1101, 1103, 1107)


def feature_names():
    names = ['last_level', 'history_span', 'history_count', 'horizon', 'query_forward',
        'query_side', 'query_distance', 'direct_geometry_free', 'recent_change',
        'recent_std', 'recent_range', 'goal_progress', 'current_wall_distance',
        'query_wall_distance', 'crossed_buildings', 'nearest_forward', 'nearest_side',
        'nearest_power_delta', 'nearest_age', 'nearest_distance', 'weighted_power_delta', 'plane_projection']
    return names + [f'lag_{lag:g}_{name}' for lag in LAGS
                    for name in ('forward', 'side', 'power_delta', 'age', 'valid')]


def features(value):
    """N queries -> Nx62. Extra fields/IDs/labels are never accessed."""
    history, now = value['history'], float(value['time_seconds'])
    times = np.array([p['time_seconds'] for p in history], float)
    xy = np.array([p['position_m'] for p in history], float)
    power = np.array([p['interference_dbm'] for p in history], float)
    position = np.asarray(value['position_m'], float)
    if (not len(history) or not np.all(np.isfinite(times)) or np.any(times > now)
            or np.any(np.diff(times) <= 0) or times[-1] != now
            or not np.allclose(xy[-1], position)):
        raise ValueError('current ordered causal point history required')
    if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(power)):
        raise ValueError('finite point history required')
    delta = np.asarray(value['goal_m'], float)-position
    forward = delta/np.linalg.norm(delta) if np.linalg.norm(delta) > 1e-8 else np.array([1., 0.])
    basis = np.array([forward, [-forward[1], forward[0]]]).T
    relative = (xy-position)@basis
    age = now-times
    last = power[-1]
    recent = power[age <= 8.+1e-8]
    lagged = []
    for lag in LAGS:
        indices = np.flatnonzero(times <= now-lag+1e-9)
        if not len(indices):
            lagged.extend([0.]*5)
        else:
            k = indices[-1]
            lagged.extend([*np.clip(relative[k]/120., -4., 4.), (power[k]-last)/10., age[k]/35., 1.])
    a = np.column_stack([relative/60., -age/20.])
    w = np.exp(-age/20.)
    plane = np.linalg.solve(a.T@(w[:, None]*a)+np.eye(3), a.T@(w*(power-last)))
    buildings = [Building.from_dict(b) for b in value['map']['buildings']]
    def wall_distance(point):
        return min([120.] + [float(point_segment_distance(point, x, y))
                            for b in buildings for x, y in ring_edges(b.footprint_m)])/120.
    current_wall = wall_distance(position)
    geom_cache, result = {}, []
    for q in value['queries']:
        qp = np.asarray(q['position_m'], float)
        horizon = float(q['time_seconds'])-now
        if not np.all(np.isfinite(qp)) or not np.isfinite(horizon) or horizon < 0:
            raise ValueError('finite future/current query required')
        qr = (qp-position)@basis
        distance = float(np.linalg.norm(qr))
        distances = np.linalg.norm(xy-qp, axis=1)
        nearest = int(np.argmin(distances))
        key = tuple(qp)
        if key not in geom_cache:
            geom_cache[key] = (wall_distance(qp), sum(b.segment_blocked(position, qp, 0.) for b in buildings)/5.)
        wd, crossed = geom_cache[key]
        local_w = np.exp(-distances/60.-age/20.)
        weighted = np.dot(local_w, power-last)/local_w.sum()
        projected = np.clip(np.dot([qr[0]/60., qr[1]/60., horizon/20.], plane), -15., 15.)
        row = [(last+80.)/20., min(now-times[0], 80.)/35., np.log1p(len(history))/np.log(65.),
            horizon/12., qr[0]/120., qr[1]/120., distance/120., float(q['direct_geometry_free']),
            (recent[-1]-recent[0])/10., np.std(recent)/10., np.ptp(recent)/10.,
            (np.linalg.norm(delta)-np.linalg.norm(np.asarray(value['goal_m'])-qp))/120.,
            current_wall, wd, crossed, *((xy[nearest]-qp)@basis/120.),
            (power[nearest]-last)/10., (q['time_seconds']-times[nearest])/35.,
            distances[nearest]/120., weighted/10., projected/10., *lagged]
        result.append(row)
    x = np.asarray(result, float)
    if x.shape != (len(value['queries']), len(feature_names())) or not np.all(np.isfinite(x)):
        raise ValueError('invalid feature matrix')
    return x


def group_weights(groups, windows):
    """Equal cities, equal states within each city, equal queries within each state."""
    groups, windows = np.asarray(groups), np.asarray(windows)
    weights = np.zeros(len(groups))
    cities = np.unique(groups)
    for city in cities:
        ids = np.unique(windows[groups == city])
        for sid in ids:
            mask = (groups == city)&(windows == sid)
            weights[mask] = 1./(len(cities)*len(ids)*mask.sum())
    return weights


def standardize_fit(x, w):
    mean = w@x
    scale = np.sqrt(w@((x-mean)**2))
    scale = np.maximum(scale, .05)
    return mean, scale


def ridge_fit(x, y, w, penalty=.01):
    a = np.column_stack([np.ones(len(x)), x])
    regularizer = np.eye(a.shape[1])*penalty
    regularizer[0, 0] = 0.
    return np.linalg.solve(a.T@(w[:, None]*a)+regularizer, a.T@(w*y))


def initialize(dimension, seed):
    rng = np.random.default_rng(seed)
    result = {}
    for k, (a, b) in enumerate(zip((dimension, 32, 16), (32, 16, 1))):
        result[f'w{k}'] = rng.normal(0., np.sqrt(2./(a+b)), (a, b))
        result[f'b{k}'] = np.zeros(b)
    return result


def forward(x, parameters):
    h1 = np.tanh(x@parameters['w0']+parameters['b0'])
    h2 = np.tanh(h1@parameters['w1']+parameters['b1'])
    return (h2@parameters['w2']+parameters['b2']).ravel(), (h1, h2)


def loss_gradient(x, y, w, parameters, penalty=.001):
    pred, (h1, h2) = forward(x, parameters)
    error = pred-y
    loss = float(w@(error*error)+.5*penalty*sum(np.sum(parameters[f'w{k}']**2) for k in range(3)))
    d = (2*w*error)[:, None]
    grad = dict(w2=h2.T@d+penalty*parameters['w2'], b2=d.sum(axis=0))
    d = (d@parameters['w2'].T)*(1-h2*h2)
    grad.update(w1=h1.T@d+penalty*parameters['w1'], b1=d.sum(axis=0))
    d = (d@parameters['w1'].T)*(1-h1*h1)
    grad.update(w0=x.T@d+penalty*parameters['w0'], b0=d.sum(axis=0))
    return loss, grad


def neural_fit(x, y, w, seed, epochs=200):
    parameters = initialize(x.shape[1], seed)
    first = {k: np.zeros_like(v) for k, v in parameters.items()}
    second = {k: np.zeros_like(v) for k, v in parameters.items()}
    trace = []
    for epoch in range(1, epochs+1):
        loss, grad = loss_gradient(x, y, w, parameters)
        if not np.isfinite(loss):
            raise ValueError('nonfinite training loss')
        trace.append(dict(epoch=epoch, objective_before_update=loss))
        for k in parameters:
            first[k] = .9*first[k]+.1*grad[k]
            second[k] = .999*second[k]+.001*grad[k]**2
            parameters[k] -= .003*(first[k]/(1-.9**epoch))/(np.sqrt(second[k]/(1-.999**epoch))+1e-8)
    return parameters, trace


def predict(x, latest, package):
    z = (x-package['feature_mean'])/package['feature_scale']
    ridge = np.column_stack([np.ones(len(z)), z])@package['ridge']
    result = dict(ridge=latest+.5*np.clip(ridge*10., -15., 15.))
    members = []
    for seed in SEEDS:
        p = {k: package[f'{seed}_{k}'] for k in ('w0', 'b0', 'w1', 'b1', 'w2', 'b2')}
        raw, _ = forward(z, p)
        members.append(latest+.5*np.clip(raw*10., -15., 15.))
    result['mlp_ensemble'] = np.mean(members, axis=0)
    result['mlp_member_predictions'] = np.array(members).T
    result['seed_disagreement_db'] = np.std(members, axis=0)
    return result
