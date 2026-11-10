"""只从已获得点测预测附近平均干扰；不接收world、发射源或未来脚本。"""

from __future__ import annotations

import numpy as np


def active_gp_measurements(measurements, time_seconds, settings):
    """与GP完全相同的因果窗口和点数上限；不补造未访问位置。"""
    active = [m for m in measurements if 0 <= time_seconds - m.time_seconds <= settings.max_age_seconds]
    return sorted(active, key=lambda m: m.time_seconds)[-settings.gp_max_points:]


def local_support_evidence(grid, time_seconds, measurements, settings):
    """给出最近训练点距离m和局部训练点数；它们不是探测范围或有效独立样本数。"""
    active = active_gp_measurements(measurements, time_seconds, settings)
    shape = grid.shape[:-1]
    if not active:
        return np.full(shape, -1.), np.zeros(shape, dtype=int)
    xy = np.array([m.position_m for m in active])
    distance = np.linalg.norm(grid[..., None, :] - xy, axis=-1)
    return distance.min(axis=-1), (distance <= settings.support_radius_m).sum(axis=-1)


def space_time_kernel(first_xy, first_t, second_xy, second_t, settings):
    """Matérn-3/2空间核乘指数时间核，协方差单位dB²。"""
    distance = np.linalg.norm(np.asarray(first_xy)[:, None, :] - np.asarray(second_xy)[None, :, :], axis=-1)
    radius = np.sqrt(3.) * distance / settings.gp_length_scale_m
    elapsed = np.abs(np.asarray(first_t)[:, None] - np.asarray(second_t)[None, :])
    return settings.unknown_error_db ** 2 * (1. + radius) * np.exp(
        -radius - elapsed / settings.gp_time_scale_seconds)


def local_gp_prediction(grid, time_seconds, measurements, settings, prior_dbm,
                        prior_at_measurements=None):
    """对最新有限点测做GP回归；远处/过期区域保留先验、未知误差和age=-1。

    uncertainty_db是所选核/超参数下的预测标准差，含测量噪声下限，尚非校准置信度。
    support_radius_m只是向规划器公开估计的局部范围，不是一次测得圆盘。
    prior_at_measurements可来自已接受的模型假设；默认使用公开背景，绝不查询真值。
    """
    shape = grid.shape[:-1]
    mean = np.array(np.broadcast_to(prior_dbm, shape), dtype=float, copy=True)
    uncertainty = np.full(shape, settings.unknown_error_db, dtype=float)
    supported = np.zeros(shape, dtype=bool)
    age = np.full(shape, -1., dtype=float)
    active = active_gp_measurements(measurements, time_seconds, settings)
    if not active:
        return mean, uncertainty, supported, age, 0
    xy = np.array([m.position_m for m in active], dtype=float)
    times = np.array([m.time_seconds for m in active])
    values = np.array([m.interference_dbm for m in active])
    if prior_at_measurements is None:
        # 此分支要求常数背景先验；残差模型由调用者提供各点自己的先验。
        if np.ndim(prior_dbm) != 0:
            raise ValueError("nonconstant GP prior requires prior_at_measurements")
        values -= prior_dbm
    else:
        values -= np.array([prior_at_measurements(m) for m in active])
    query = grid.reshape(-1, 2)
    distances = np.linalg.norm(query[:, None, :] - xy[None, :, :], axis=-1)
    nearest = np.argmin(distances, axis=1)
    usable = distances[np.arange(len(query)), nearest] <= settings.support_radius_m
    covariance = space_time_kernel(xy, times, xy, times, settings)
    covariance.flat[::len(xy) + 1] += settings.measured_error_db ** 2 + 1e-8
    cross = space_time_kernel(query, np.full(len(query), time_seconds), xy, times, settings)
    # 一次求解同时计算均值权重和预测方差，不显式求逆。
    solved = np.linalg.solve(covariance, np.column_stack((values, cross.T)))
    posterior_mean = cross @ solved[:, 0]
    latent_variance = np.maximum(0., settings.unknown_error_db ** 2 - np.sum(cross * solved[:, 1:].T, axis=1))
    mean.flat[np.flatnonzero(usable)] += posterior_mean[usable]
    uncertainty.flat[np.flatnonzero(usable)] = np.sqrt(latent_variance[usable] + settings.measured_error_db ** 2)
    supported.flat[:] = usable
    age.flat[np.flatnonzero(usable)] = time_seconds - times[nearest[usable]]
    return mean, uncertainty, supported, age, len(active)
