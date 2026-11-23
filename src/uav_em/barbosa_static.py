"""Static scalar-field subset of Barbosa et al. (arXiv:2109.11287)."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import ndtri


def _matrix(rows, width):
    values = np.asarray(rows, dtype=float)
    if values.shape == (0,):
        values = values.reshape(0, width)
    if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError(f"expected finite rows with {width} columns")
    return values


class FrozenSpatialGP:
    """Fit a stationary squared-exponential GP to an observed (x, y, value) prefix."""

    def __init__(self, observations, *, mean, length_scale, signal_std, noise_variance):
        if (not np.isfinite([mean, length_scale, signal_std, noise_variance]).all()
                or min(length_scale, signal_std, noise_variance) <= 0):
            raise ValueError("finite prior and positive kernel scales and observation variance required")
        samples = _matrix(observations, 3)
        self._positions = samples[:, :2].copy()
        self._mean = float(mean)
        self._length_scale = float(length_scale)
        self._variance = float(signal_std) ** 2
        if len(samples):
            gram = self._kernel(self._positions, self._positions)
            gram.flat[::len(gram) + 1] += noise_variance
            self._factor = cho_factor(gram, lower=True)
            self._weights = cho_solve(self._factor, samples[:, 2] - self._mean)

    def _kernel(self, first, second):
        squared_distance = cdist(first, second, metric="sqeuclidean")
        return self._variance * np.exp(-0.5 * squared_distance / self._length_scale ** 2)

    def predict(self, positions):
        """Return latent mean and standard deviation, not noisy-reading uncertainty."""
        points = _matrix(positions, 2)
        mean = np.full(len(points), self._mean)
        variance = np.full(len(points), self._variance)
        if len(self._positions) and len(points):
            cross = self._kernel(self._positions, points)
            mean += cross.T @ self._weights
            whitened = solve_triangular(self._factor[0], cross, lower=True)
            variance -= np.sum(whitened ** 2, axis=0)
        return mean, np.sqrt(np.maximum(variance, 0.0))


def gaussian_upper_cvar(mean, std, tail_mass):
    """Expected value conditional on falling in the upper tail of mass tail_mass."""
    if not 0 < tail_mass < 1:
        raise ValueError("tail mass must be in (0, 1)")
    cutoff = -ndtri(tail_mass)
    multiplier = np.exp(-0.5 * cutoff ** 2 - 0.5 * np.log(2 * np.pi) - np.log(tail_mass))
    return np.asarray(mean) + np.asarray(std) * multiplier


def _route_samples(route, max_spacing):
    points = _matrix(route, 2)
    if not len(points):
        raise ValueError("route must contain at least its current position")
    samples = [points[:1]]
    step_lengths = []
    for start, end in zip(points, points[1:]):
        length = float(np.linalg.norm(end - start))
        if length:
            count = math.ceil(length / max_spacing)
            samples.append(np.linspace(start, end, count + 1)[1:])
            step_lengths.extend([length / count] * count)
    return np.concatenate(samples), np.asarray(step_lengths)


@dataclass(frozen=True)
class RiskIncrease:
    position: tuple[float, float]
    current_cvar: float
    committed_cvar: float


@dataclass(frozen=True)
class StaticRiskRule:
    tail_mass: float
    reference_tail_mass: float
    threshold: float
    penalty: float
    max_spacing: float

    def __post_init__(self):
        if (not 0 < self.reference_tail_mass < self.tail_mass < 1
                or not np.isfinite([self.threshold, self.penalty, self.max_spacing]).all()
                or self.penalty < 0 or self.max_spacing <= 0):
            raise ValueError("invalid risk thresholds, tail masses or integration spacing")

    def route_cost(self, route, model):
        """Trapezoidal evaluation of the risk-weighted arc length (Eq. 11)."""
        points, step_lengths = _route_samples(route, self.max_spacing)
        if not len(step_lengths):
            return 0.0
        risk = gaussian_upper_cvar(*model.predict(points), self.tail_mass)
        with np.errstate(over="raise", invalid="raise"):
            weight = np.exp(self.penalty * np.maximum(risk - self.threshold, 0.0))
            return float((0.5 * weight[:-1] + 0.5 * weight[1:]) @ step_lengths)

    def risk_increase(self, unflown_route, *, current, committed):
        """Find a sampled witness of Eq. 12 against the posterior at route commitment."""
        points, step_lengths = _route_samples(unflown_route, self.max_spacing)
        if not len(step_lengths):
            return None
        new_risk = gaussian_upper_cvar(*current.predict(points), self.tail_mass)
        old_risk = gaussian_upper_cvar(*committed.predict(points), self.reference_tail_mass)
        crossings = np.flatnonzero(new_risk > old_risk)
        if not len(crossings):
            return None
        first = crossings[0]
        return RiskIncrease(tuple(map(float, points[first])), float(new_risk[first]), float(old_risk[first]))

    def choose_route(self, feasible_routes, model):
        """Exact choice on a caller-supplied finite feasible set; not BIT*."""
        costs = tuple(self.route_cost(route, model) for route in feasible_routes)
        if not costs:
            raise ValueError("at least one feasible route required")
        return min(range(len(costs)), key=costs.__getitem__), costs
