"""Sparse nonnegative compression of frozen predictions, fitted to task integrals."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import nnls

SCALES = (40., 100., 240.)
VELOCITIES = ((0., 0.), (3., 0.), (-3., 0.), (0., 3.), (0., -3.))
POWER_FACTOR = math.exp(.5 * (math.log(10) / 10) ** 2 * 17)
METHODS = ('full_reference', 'spatial_compression', 'task_compression')


def dictionary(high):
    high = np.asarray(high, float)
    if high.shape != (2,) or not np.all(np.isfinite(high)) or np.any(high <= 0):
        raise ValueError('positive finite city dimensions required')
    return [dict(center_m=[float(x), float(y)], scale_m=scale, velocity_mps=list(v))
            for y in np.linspace(0, high[1], 7) for x in np.linspace(0, high[0], 9)
            for scale in SCALES for v in VELOCITIES]


def kernels(bases, xy, times, reference_time):
    xy, times = np.asarray(xy, float), np.asarray(times, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or times.shape != (len(xy),):
        raise ValueError('Nx2 receivers and N times required')
    if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(times)) or not math.isfinite(reference_time):
        raise ValueError('finite query required')
    centers = np.array([b['center_m'] for b in bases], float).reshape(-1, 2)
    speed = np.array([b['velocity_mps'] for b in bases], float).reshape(-1, 2)
    scales = np.array([b['scale_m'] for b in bases], float)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(speed)) or np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError('finite bases and positive spatial scales required')
    out = np.empty((len(xy), len(bases)))
    for start in range(0, len(xy), 512):
        q, t = xy[start:start + 512], times[start:start + 512]
        delta = q[:, None, :] - centers[None, :, :] - (t - reference_time)[:, None, None] * speed[None, :, :]
        out[start:start + len(q)] = np.exp(-np.sum(delta ** 2, axis=-1) / (2 * scales ** 2))
    return out


def split_routes(count):
    if type(count) is not int or count < 1:
        raise ValueError('nonempty candidate list required')
    even = list(range(0, count, 2))
    train = [even[i] for i in np.unique(np.linspace(0, len(even) - 1, min(6, len(even))).astype(int))]
    return dict(fit=train, heldout=list(range(1, count, 2)),
                other=[i for i in even if i not in train])


def compress(matrix, target, max_bases=4):
    """One independent sparse NNLS fit per world; no observations or truth accepted."""
    A, Y = np.asarray(matrix, float), np.asarray(target, float)
    if A.ndim != 2 or Y.ndim != 2 or A.shape[0] != Y.shape[0] or A.shape[1] == 0:
        raise ValueError('matching nonnegative design and target matrices required')
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(Y)) or np.any(A < 0) or np.any(Y < 0):
        raise ValueError('finite nonnegative compression inputs required')
    if type(max_bases) is not int or not 1 <= max_bases <= A.shape[1]:
        raise ValueError('positive integer basis budget required')
    count = Y.shape[1]
    coefficients = np.zeros((A.shape[1], count))
    residual = Y.copy()
    norms = np.linalg.norm(A, axis=0)
    usable = norms > 1e-12
    selected = [[] for _ in range(count)]
    trajectories = [[] for _ in range(count)]
    finished = np.zeros(count, bool)
    calls = 0
    correlation_terms = 0
    for _ in range(max_bases):
        corr = A.T @ residual
        correlation_terms += int(A.shape[0] * A.shape[1] * count)
        corr = np.divide(corr, norms[:, None], out=np.full_like(corr, -np.inf), where=usable[:, None])
        for h in range(count):
            if finished[h]:continue
            if np.linalg.norm(residual[:, h]) <= 1e-12 * max(1., np.linalg.norm(Y[:, h])):
                finished[h] = True;continue
            corr[selected[h], h] = -np.inf
            j = int(np.argmax(corr[:, h]))
            if not np.isfinite(corr[j, h]) or corr[j, h] <= 0:
                finished[h] = True;continue
            selected[h].append(j)
            values, error = nnls(A[:, selected[h]], Y[:, h], maxiter=100)
            calls += 1
            coefficients[:, h] = 0
            coefficients[selected[h], h] = values
            residual[:, h] = Y[:, h] - A[:, selected[h]] @ values
            trajectories[h].append(dict(added_index=j, indices=selected[h].copy(), amplitudes=values.tolist(),
                                         residual_l2=float(error)))
        if np.all(finished):break
    return dict(coefficients=coefficients, trajectories=trajectories,
                resources=dict(nnls_calls=calls, correlation_scalar_terms=correlation_terms,
                               rows=int(A.shape[0]), worlds=int(count), dictionary_size=int(A.shape[1]),
                               maximum_bases=max_bases))


def cost_metrics(reference, compressed, weights, split):
    ref, cmp, w = np.asarray(reference, float), np.asarray(compressed, float), np.asarray(weights, float)
    if ref.shape != cmp.shape or ref.ndim != 2 or w.shape != (ref.shape[1],):
        raise ValueError('RxH costs and H weights required')
    if np.any(w < 0) or not np.isclose(w.sum(), 1) or not np.all(np.isfinite(ref)) or not np.all(np.isfinite(cmp)):
        raise ValueError('valid costs and world mass required')
    ref_mean, cmp_mean = ref @ w, cmp @ w
    ref_var = ((ref - ref_mean[:, None]) ** 2) @ w
    cmp_var = ((cmp - cmp_mean[:, None]) ** 2) @ w
    chosen, original = int(np.argmin(cmp_mean)), int(np.argmin(ref_mean))
    epsilon = float(np.max(np.abs(cmp_mean - ref_mean)))
    regret = float(ref_mean[chosen] - ref_mean[original])
    result = dict(epsilon_seconds=epsilon, reference_regret_seconds=regret,
                  bound_2epsilon_seconds=2 * epsilon, bound_holds=bool(regret <= 2 * epsilon + 1e-8),
                  selected_index=chosen, reference_selected_index=original,
                  between_scenario_variance_mae_seconds2=float(np.mean(np.abs(cmp_var - ref_var))))
    for label, ids in dict(all=list(range(len(ref))), **split).items():
        if not ids:
            result[label] = dict(count=0);continue
        anchor = min(ids, key=lambda i: ref_mean[i])
        pairs = [i for i in ids if i != anchor and abs(ref_mean[i] - ref_mean[anchor]) > 1e-8]
        signs = [bool(np.sign(cmp_mean[i] - cmp_mean[anchor]) == np.sign(ref_mean[i] - ref_mean[anchor])) for i in pairs]
        separated = [i for i in pairs if abs(ref_mean[i] - ref_mean[anchor]) > 2 * epsilon]
        separated_correct = [bool(np.sign(cmp_mean[i] - cmp_mean[anchor]) == np.sign(ref_mean[i] - ref_mean[anchor])) for i in separated]
        result[label] = dict(count=len(ids), world_weighted_mae_seconds=float(np.mean(np.abs(cmp[ids] - ref[ids]) @ w)),
                            posterior_mean_max_error_seconds=float(np.max(np.abs(cmp_mean[ids] - ref_mean[ids]))),
                            non_tied_pairs=len(pairs), sign_agreement=float(np.mean(signs)) if signs else None,
                            separated_pairs=len(separated), separated_order_holds=bool(all(separated_correct)))
    return result
