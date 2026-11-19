"""Constructive oracle capacity check, independent of point-history inference."""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import least_squares

from .prediction_data import smooth
from .scenarios import free_graph, geometry_route

BUDGETS = (1, 2, 4, 8)


def task_points(geometry, position, now, heldout=False):
    offset = np.arange(-140., 141., 40.) if heldout else np.arange(-160., 161., 40.)
    horizons = (3., 9.) if heldout else (0., 6., 12.)
    xy = [np.asarray(position)+[x, y] for y in offset for x in offset
          if x*x+y*y <= 160.**2 and geometry.free(np.asarray(position)+[x, y])]
    return [dict(position_m=p.tolist(), time_seconds=now+h, horizon_seconds=h)
            for h in horizons for p in xy]


def dictionary(high, position):
    centers = [np.asarray(position)+[x, y] for y in np.arange(-160., 161., 40.)
               for x in np.arange(-160., 161., 40.)
               if np.all(np.asarray(position)+[x, y] >= 0) and np.all(np.asarray(position)+[x, y] <= high)]
    return [dict(center_m=c.tolist(), scale_m=s, velocity_mps=list(v), rate_db_per_second=r)
            for c in centers for s in (25., 60., 140.)
            for v in ((0., 0.), (3., 0.), (-3., 0.), (0., 3.), (0., -3.)) for r in (-.75, 0., .75)]


def kernel(bases, xy, times, now):
    xy, times = np.asarray(xy, float), np.asarray(times, float)
    if xy.shape != (len(times), 2) or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(times)):
        raise ValueError('finite Nx2 and N queries required')
    if not bases: return np.empty((len(times), 0))
    centers = np.array([b['center_m'] for b in bases])
    speed = np.array([b['velocity_mps'] for b in bases])
    scales = np.array([b['scale_m'] for b in bases])
    rate = np.array([b['rate_db_per_second'] for b in bases])
    if np.any(scales <= 0) or not all(np.all(np.isfinite(a)) for a in (centers, speed, scales, rate)):
        raise ValueError('finite bases and positive scales required')
    dt = times-now
    delta = xy[:, None, :]-centers[None, :, :]-dt[:, None, None]*speed[None, :, :]
    return np.exp(-np.sum(delta*delta, axis=-1)/(2*scales**2)+np.log(10.)/10.*dt[:, None]*rate)


def power_and_jacobian(matrix, peaks):
    contribution = matrix*10.**((np.asarray(peaks)+80.)/10.)
    total = .01+contribution.sum(axis=1)  # -100 dBm relative to -80 dBm
    return -80.+10.*np.log10(total), contribution/total[:, None]


def fit(bases, xy, times, now, labels, max_sources=8):
    labels = np.asarray(labels, float)
    if not len(labels) or not np.all(np.isfinite(labels)) or max_sources < 1:
        raise ValueError('finite nonempty oracle labels and positive budget required')
    a = kernel(bases, xy, times, now)
    if len(a) != len(labels): raise ValueError('target shape')
    upper = float(labels.max()+6.)
    if upper <= -140.: raise ValueError('invalid peak bound')
    chosen, peaks, trace, prefixes = [], [], [], {}
    prediction = np.full(len(labels), -100.)
    loss = float(np.mean((prediction-labels)**2))
    nfev = 0
    for step in range(1, max_sources+1):
        derivatives = a/10.**((prediction+80.)/10.)[:, None]
        norms = np.linalg.norm(derivatives, axis=0)
        scores = np.divide(derivatives.T@(labels-prediction), norms,
                           out=np.full(len(bases), -np.inf), where=norms > 1e-12)
        scores[chosen] = -np.inf
        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]) or scores[index] <= 0: break
        target_relative = 10.**((labels+80.)/10.)-10.**((prediction+80.)/10.)
        amplitude = max(1e-6, float(a[:, index]@target_relative)/max(float(a[:, index]@a[:, index]), 1e-30))
        initial = np.r_[peaks, np.clip(-80.+10.*np.log10(amplitude), -140.+1e-8, upper-1e-8)]
        columns = chosen+[index]
        matrix = a[:, columns]
        solved = least_squares(lambda p: (power_and_jacobian(matrix, p)[0]-labels)/np.sqrt(len(labels)),
            initial, jac=lambda p: power_and_jacobian(matrix, p)[1]/np.sqrt(len(labels)),
            bounds=(-140., upper), max_nfev=100, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        nfev += int(solved.nfev)
        pred, _ = power_and_jacobian(matrix, solved.x)
        candidate_loss = float(np.mean((pred-labels)**2))
        accepted = bool(np.all(np.isfinite(solved.x)) and np.all(solved.x >= -140.)
            and np.all(solved.x <= upper) and candidate_loss <= loss+1e-8)
        trace.append(dict(step=step, proposed_index=index, chosen_indices=columns,
            initial_peaks_dbm=initial.tolist(), result_peaks_dbm=solved.x.tolist(),
            previous_mse=loss, candidate_mse=candidate_loss, accepted=accepted,
            solver_success=bool(solved.success), solver_status=int(solved.status),
            optimality=float(solved.optimality), nfev=int(solved.nfev), njev=int(solved.njev)))
        if not accepted: break
        chosen, peaks, prediction, loss = columns, solved.x.tolist(), pred, candidate_loss
        prefixes[step] = dict(indices=chosen[:], peaks_dbm=peaks[:], fit_mse=loss)
    outputs = {}
    for count in BUDGETS:
        candidates = [k for k in prefixes if k <= count]
        outputs[str(count)] = prefixes[max(candidates)] if candidates else dict(indices=[], peaks_dbm=[], fit_mse=float(np.mean((-100.-labels)**2)))
    return dict(prefixes=outputs, trace=trace, peak_bounds_dbm=[-140., upper],
        resources=dict(least_squares_calls=len(trace), objective_nfev=nfev, dictionary_columns=len(bases)),
        constant_dbm=float(labels.mean()), global_optimality_certified=False)


def predict(bases, state, xy, times, now):
    selected = [bases[i] for i in state['indices']]
    return power_and_jacobian(kernel(selected, xy, times, now), state['peaks_dbm'])[0]


def route_queries(route, now, speed):
    xy, times, weights, elapsed = [], [], [], 0.
    for start, end in zip(route, route[1:]):
        start, end = np.asarray(start), np.asarray(end)
        length = float(np.linalg.norm(end-start))
        duration = length/speed
        for i in range(max(1, int(math.ceil(length/10.)))):
            count = max(1, int(math.ceil(length/10.)))
            fraction = (i+.5)/count
            xy.append((start+fraction*(end-start)).tolist())
            times.append(now+elapsed+fraction*duration)
            weights.append(duration/count)
        elapsed += duration
    return dict(position_m=xy, time_seconds=times, dt_seconds=weights, duration_seconds=elapsed)


def candidates(geometry, position, goal, now, speed):
    graph = free_graph(geometry)
    forward = np.asarray(goal)-position
    forward = forward/np.linalg.norm(forward)
    side = np.array([-forward[1], forward[0]])
    result = []
    for angle in (-90., -60., -30., 0., 30., 60., 90.):
        theta = math.radians(angle)
        endpoint = np.asarray(position)+120.*(math.cos(theta)*forward+math.sin(theta)*side)
        if not geometry.free(endpoint): continue
        path = smooth(geometry, geometry_route(geometry, graph, position, endpoint))
        if not path or any(np.linalg.norm(np.asarray(p)-position) > 160.+1e-8 for p in path): continue
        query = route_queries(path, now, speed)
        if query['duration_seconds'] > 12.+1e-8: continue
        result.append(dict(id=f'angle_{angle:g}', angle_degrees=angle, route_m=path, queries=query))
    return result


def cost(power_dbm, query):
    power = np.asarray(power_dbm)
    weights = np.asarray(query['dt_seconds'])
    q = float(weights@10.**((power+80.)/10.))
    return dict(q_seconds=q, c_seconds=q+query['duration_seconds'])
