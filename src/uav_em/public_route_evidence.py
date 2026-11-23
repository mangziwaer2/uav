"""Public-only route pricing for point measurements augmented by an optional GP."""
import numpy as np


def _rows(values, width):
    data = np.asarray(values, dtype=float)
    if data.shape == (0,):
        return data.reshape(0, width)
    if data.ndim != 2 or data.shape[1] != width or not np.isfinite(data).all():
        raise ValueError('invalid evidence array')
    return data


def route_evidence(request, positions, times, durations):
    """Use exactly one source per quadrature node; never treat a GP as a new reading."""
    issued = float(request['time_s'])
    raw = _rows(request['points_xy_t_dbm'], 4)
    grid = _rows(request.get('gp', []), 5)
    xy = _rows(positions, 2)
    clocks, dt = np.asarray(times, float), np.asarray(durations, float)
    if (not np.isfinite(issued) or not len(xy) or clocks.shape != (len(xy),) or dt.shape != clocks.shape
            or not np.isfinite(clocks).all() or not np.isfinite(dt).all() or np.any(dt <= 0)
            or np.any(clocks < issued) or np.any(np.diff(clocks) < 0)
            or np.any(raw[:, 2] > issued) or np.any(np.diff(raw[:, 2]) <= 0)
            or np.any(grid[:, 2] < issued) or np.any(grid[:, 4] < 0)):
        raise ValueError('invalid public prefix or route quadrature')
    power = np.ones(len(xy)); source = np.full(len(xy), 'unknown', dtype='U16')
    if len(raw):
        distance = np.linalg.norm(xy[:, None] - raw[None, :, :2], axis=2)
        age = clocks[:, None] - raw[None, :, 2]
        exact = (distance == 0) & (age == 0)
        local = (distance <= 80) & (age <= 20)
        weights = np.where(local, np.exp(-(distance/40)**2-age/10), 0.)
        known = local.any(axis=1)
        linear = 10**((raw[:, 3]+80)/10)
        power[known] = (weights[known] @ linear)/weights[known].sum(axis=1)
        source[known] = 'local_points'
    if len(grid):
        distance = np.linalg.norm(xy[:, None]-grid[None, :, :2], axis=2)
        delta = np.abs(clocks[:, None]-grid[None, :, 2])
        metric = np.where((distance <= 45) & (delta <= 6), (distance/45)**2+(delta/6)**2, np.inf)
        supported = np.isfinite(metric.min(axis=1))
        power[supported] = 10**((grid[metric[supported].argmin(axis=1), 3]+80)/10)
        source[supported] = 'gp_grid'
    if len(raw):
        direct = exact.any(axis=1)
        source[direct] = 'measurement'
        power[direct] = linear[exact[direct].argmax(axis=1)]
    seconds = {name:float(dt[source == name].sum()) for name in ('measurement','gp_grid','local_points','unknown')}
    exposure = float(dt @ power)
    return dict(time_s=float(dt.sum()), exposure_proxy_s=exposure, total_proxy_s=float(dt.sum()+exposure),
                seconds_by_source=seconds, source_by_node=source.tolist())
