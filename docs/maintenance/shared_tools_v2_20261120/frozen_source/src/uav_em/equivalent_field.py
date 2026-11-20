"""Nonnegative equivalent field bases and linear-power objective; not emitter localization."""
import math
import numpy as np

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


def normalized_expected_power(mean, std, scale_dbm):
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std < 0) or not math.isfinite(scale_dbm):
        raise ValueError('finite mean/scale and nonnegative std required')
    a = math.log(10) / 10
    with np.errstate(over='raise', invalid='raise'):
        try:
            return np.exp(a * (mean - scale_dbm) + .5 * a * a * std * std)
        except FloatingPointError as exc:
            raise ValueError('power expectation overflow; no clipping applied') from exc

