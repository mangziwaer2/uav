"""Public-only finite-domain comparison of fixed-review route prices."""

import math

import numpy as np
from scipy.special import log_ndtr, ndtri

from .passive_map import _common_time, _coverage, quadrature


POLICIES = ("shortest", "mean", "risk")
TAIL_MASS = 0.05
UNKNOWN_PRICE = 1.0
_LOG_POWER_SCALE = math.log(10) / 10
_UPPER_QUANTILE = -ndtri(TAIL_MASS)


def upper_tail_power(mean_dbm, std_db):
    """Pointwise upper-tail mean of normalized linear power under the dBm GP."""
    log_std = _LOG_POWER_SCALE * np.asarray(std_db)
    log_tail = log_ndtr(log_std - _UPPER_QUANTILE) - math.log(TAIL_MASS)
    return np.exp(_LOG_POWER_SCALE * (np.asarray(mean_dbm) + 80) + 0.5 * log_std ** 2 + log_tail)


def compare_prices(routes, model, now, speed):
    """Rank the same legal routes, using a single batch of causal GP queries."""
    samples = [quadrature(route, now, speed) for route in routes]
    if not samples:
        raise ValueError("no legal route to score")
    points, clocks, weights = (np.concatenate(column) for column in zip(*samples))
    mean, std, source = model.predict(points, clocks)
    supported = source < 2
    mean_price = np.full(len(points), UNKNOWN_PRICE)
    risk_price = np.full(len(points), UNKNOWN_PRICE)
    mean_price[supported] = 10 ** ((mean[supported] + 80) / 10)
    risk_price[supported] = upper_tail_power(mean[supported], std[supported])

    stops = np.cumsum([len(sample[1]) for sample in samples])
    chunks = [slice(start, stop) for start, stop in zip([0, *stops[:-1]], stops)]
    rows = []
    for route, chunk in zip(routes, chunks):
        dt = weights[chunk]
        duration = float(dt.sum())
        costs = dict(shortest=duration, mean=float(dt @ (1 + mean_price[chunk])),
                     risk=float(dt @ (1 + risk_price[chunk])))
        rows.append(dict(route=route, time_s=duration, costs=costs, coverage=_coverage(dt, source[chunk])))
    selected = {policy: min(range(len(rows)), key=lambda idx: (rows[idx]["costs"][policy], rows[idx]["time_s"], idx))
                for policy in POLICIES}

    split = {}
    for policy in ("shortest", "risk"):
        first, second = selected["mean"], selected[policy]
        common_until = now + _common_time(routes[first], routes[second], speed)
        coverage = {}
        for arm, idx in (("mean", first), (policy, second)):
            chunk = chunks[idx]
            start = clocks[chunk] - weights[chunk] / 2
            end = clocks[chunk] + weights[chunk] / 2
            duration = np.maximum(0, end - np.maximum(start, common_until))
            coverage[arm] = _coverage(duration, source[chunk])
        split[policy] = dict(common_until_s=common_until, coverage=coverage)
    return dict(candidates=rows, selected=selected, split_vs_mean=split)
