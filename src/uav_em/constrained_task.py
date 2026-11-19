"""Bounded sparse proxies of frozen model predictions; no observations or truth."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize

TOLERANCE = 1e-7
POWER_FACTOR = math.exp(.5 * (math.log(10) / 10) ** 2 * 17)


def build_blocks(route_matrix, route_targets, outer_duration, probe_matrix, probe_targets):
    """Accept only fit corridors and public anchors, never held-out targets."""
    A, Q, dt, B, Z = (np.asarray(v, float) for v in
                       (route_matrix, route_targets, outer_duration, probe_matrix, probe_targets))
    if A.ndim != 2 or Q.ndim != 2 or B.ndim != 2 or Z.ndim != 2:
        raise ValueError('two-dimensional matrices required')
    if A.shape[0] != Q.shape[0] or B.shape[0] != Z.shape[0] or A.shape[1] != B.shape[1] or Q.shape[1] != Z.shape[1] or dt.shape != (len(A),):
        raise ValueError('matching fit/probe dimensions required')
    if not A.shape[1] or not Q.shape[1]:raise ValueError('nonempty dictionary and world set required')
    if any(not np.all(np.isfinite(v)) or np.any(v < 0) for v in (A, Q, dt, B, Z)):
        raise ValueError('finite nonnegative inputs required')
    positive = dt > 0
    if np.any(Q[~positive] > 0) or np.any(A[~positive] > 0):
        raise ValueError('zero outer duration must have zero source integral')
    U = np.max(Z, axis=0) if len(Z) else np.zeros(Q.shape[1])
    if np.any(positive):U = np.maximum(U, np.max(Q[positive] / (POWER_FACTOR * dt[positive, None]), axis=0))
    Znorm = np.divide(Z, U[None, :], out=np.zeros_like(Z), where=U[None, :] > 0)
    Dspace = B / math.sqrt(len(B)) if len(B) else B.copy()
    yspace = Znorm / math.sqrt(len(B)) if len(B) else Znorm
    tau = float(dt[positive].mean()) if np.any(positive) else None
    if tau is not None:
        scale = POWER_FACTOR * tau * math.sqrt(len(A))
        Dtask = A / scale
        ytask = np.divide(Q, U[None, :], out=np.zeros_like(Q), where=U[None, :] > 0) / scale
        combined, target = np.vstack((Dtask, Dspace)), np.vstack((ytask, yspace))
    else:
        combined, target = Dspace.copy(), yspace.copy()
    return dict(envelope=U, mean_positive_outer_seconds=tau, space_matrix=Dspace, space_target=yspace,
                task_matrix=combined, task_target=target, task_supported=tau is not None and len(B) > 0,
                space_supported=len(B) > 0)


def kkt_witness(matrix, target, b):
    """First-order witness on the currently selected convex simplex problem."""
    matrix, target, b = np.asarray(matrix), np.asarray(target), np.asarray(b)
    finite = bool(np.all(np.isfinite(b)))
    if not finite:return dict(passed=False, finite=False)
    mass = float(b.sum());gradient = 2 * matrix.T @ (matrix @ b - target)
    active = b > 1e-12
    multiplier = max(0., -float(gradient[active].mean())) if mass >= 1 - TOLERANCE and np.any(active) else 0.
    reduced = gradient + multiplier
    primal = max(0., -float(b.min(initial=0)), mass - 1)
    stationarity = float(np.max(np.abs(reduced[active]), initial=0))
    inactive = max(0., -float(np.min(reduced[~active], initial=0)))
    complementarity = abs(multiplier * (1 - mass))
    return dict(passed=bool(max(primal, stationarity, inactive, complementarity) <= TOLERANCE), finite=True,
                mass=mass, multiplier=multiplier, primal_error=primal,
                active_stationarity=stationarity, inactive_violation=inactive, complementarity=complementarity)


def solve_sparse(matrix, target, envelope, supported=True, max_bases=4):
    """At most four raw-correlation additions and bounded corrective solves/world."""
    D, Y, U = np.asarray(matrix, float), np.asarray(target, float), np.asarray(envelope, float)
    if D.ndim != 2 or Y.ndim != 2 or D.shape[0] != Y.shape[0] or U.shape != (Y.shape[1],) or D.shape[1] < 1:
        raise ValueError('matching matrix/target/envelope required')
    if any(not np.all(np.isfinite(v)) or np.any(v < 0) for v in (D, Y, U)):
        raise ValueError('finite nonnegative matrix, target and envelope required')
    if type(max_bases) is not int or not 1 <= max_bases <= min(4, D.shape[1]):
        raise ValueError('one to four dictionary terms required')
    count = Y.shape[1];b = np.zeros((D.shape[1], count));residual = Y.copy()
    selected = [[] for _ in range(count)];traces = [[] for _ in range(count)]
    statuses = ['abstain' if not supported else ('empty' if u == 0 else 'running') for u in U]
    calls = iterations = correlation_terms = 0
    for _ in range(max_bases):
        if not any(s == 'running' for s in statuses):break
        corr = D.T @ residual;correlation_terms += D.shape[0] * D.shape[1] * count
        for h in range(count):
            if statuses[h] != 'running':continue
            corr[selected[h], h] = -np.inf
            j = int(np.argmax(corr[:, h]))
            if not np.isfinite(corr[j, h]) or corr[j, h] <= 0:
                statuses[h] = 'success';continue
            indices = selected[h] + [j];M = D[:, indices];y = Y[:, h]
            x0 = np.r_[b[selected[h], h], 0.]
            gram, rhs = M.T @ M, M.T @ y
            result = minimize(lambda x: float(x @ gram @ x - 2 * rhs @ x + y @ y), x0,
                jac=lambda x: 2 * (gram @ x - rhs), method='SLSQP', bounds=[(0., None)] * len(indices),
                constraints=[dict(type='ineq', fun=lambda x: 1 - x.sum(), jac=lambda x: -np.ones_like(x))],
                options=dict(maxiter=200, ftol=1e-10))
            calls += 1;iterations += int(result.nit)
            witness = kkt_witness(M, y, result.x)
            finite = bool(np.all(np.isfinite(result.x)))
            loss = float(np.sum((M @ result.x - y) ** 2)) if finite else None
            old_loss = float(residual[:, h] @ residual[:, h])
            record = dict(added_index=j, indices=indices, initial=x0.tolist(),
                normalized_amplitudes=result.x.tolist() if finite else None,
                optimizer_success=bool(result.success), optimizer_status=int(result.status),
                message=str(result.message), iterations=int(result.nit), witness=witness,
                loss=loss, previous_loss=old_loss, accepted=False)
            traces[h].append(record)
            if not result.success or not witness['passed']:
                statuses[h] = 'failed';continue
            if loss >= old_loss:
                statuses[h] = 'success';continue
            selected[h] = indices;b[:, h] = 0;b[indices, h] = result.x
            residual[:, h] = y - M @ result.x;record['accepted'] = True
    statuses = ['success' if s == 'running' else s for s in statuses]
    return dict(normalized_coefficients=b, coefficients=b * U[None, :], statuses=statuses, trajectories=traces,
                resources=dict(solver_calls=calls, optimizer_iterations=iterations,
                               correlation_scalar_terms=correlation_terms, maximum_bases=max_bases))
