"""Causal GP interpolation, bounded soft constraints and route warnings.

All inputs are already observed public data. No truth/sensor/LLM access.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np

from .equivalent_field import normalized_expected_power
from .point_gp import FrozenGP, training_points


@dataclass(frozen=True)
class JointSettings:
    distance_scale_m: float = 80.
    age_scale_seconds: float = 35.
    horizon_scale_seconds: float = 20.
    error_scale_db: float = 8.
    soft_strength: float = .25
    soft_max_std_db: float = 8.

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('positive finite joint settings required')


def local_prediction(gp, xy, times, settings):
    xy, times = np.asarray(xy, float), np.asarray(times, float)
    mu, sd, supported = gp.predict(xy, times)
    weight = np.zeros(len(times))
    if gp.model.points and len(times):
        points = gp.model.points
        distances = np.linalg.norm(xy[:, None]-np.asarray([p['position_m'] for p in points])[None], axis=2)
        nearest = np.argmin(distances, axis=1)
        distance = distances[np.arange(len(times)), nearest]
        age = times-np.asarray([p['time_seconds'] for p in points])[nearest]
        weight = (np.exp(-(distance/settings.distance_scale_m)**2)
                  * np.exp(-age/settings.age_scale_seconds)
                  * np.exp(-(times-gp.now)/settings.horizon_scale_seconds)
                  / (1.+(sd/settings.error_scale_db)**2))
        weight = np.where(supported, weight, 0.)
    return dict(gp_mean_dbm=mu, gp_std_db=sd, supported=supported, gp_weight=weight)


def blend_prediction(source, gp, xy, times, settings):
    local = local_prediction(gp, xy, times, settings)
    g = local['gp_weight'][None, :]
    gp_mu, gp_sd = local['gp_mean_dbm'][None, :], local['gp_std_db'][None, :]
    source_mu, source_sd = source['means'], source['stds']
    means = g*gp_mu+(1-g)*source_mu
    variance = g*(gp_sd**2+(gp_mu-means)**2)+(1-g)*(source_sd**2+(source_mu-means)**2)
    expected = (g*normalized_expected_power(gp_mu, gp_sd, -80.)
                +(1-g)*normalized_expected_power(source_mu, source_sd, -80.))
    return dict(source, means=means, stds=np.sqrt(variance), **local,
                source_mean_dbm=source_mu, source_std_db=source_sd,
                expected_normalized_power=expected,
                predictive_moments='convex representation mixture; expected linear power saved separately')


def soft_constraints(history, geometry, belief, background, settings):
    """Build from this fit's prefix; pilot callers must withhold their tail."""
    now = history[-1]['time_seconds']
    points = training_points(history, geometry, now, belief)
    gp = FrozenGP(points, geometry, now, belief, background)
    xy = []
    for p in history[-5:]:
        for offset in (0., -20., 20.):
            q = np.asarray(p['position_m'])+[0., offset]
            if geometry.free(q) and q.tolist() not in xy:
                xy.append(q.tolist())
    if not xy:
        return None
    times = np.full(len(xy), now)
    local = local_prediction(gp, xy, times, settings)
    eligible = local['supported'] & (local['gp_std_db'] <= settings.soft_max_std_db)
    weights = local['gp_weight'][eligible]
    if not len(weights) or weights.sum() <= 0:
        return dict(position_m=[], time_seconds=[], mean_dbm=[], weights=[],
                    strength=settings.soft_strength, gp_queries=len(xy), prefix_end_seconds=now,
                    original_measurements=len(history), independent_evidence_points=0)
    return dict(position_m=np.asarray(xy)[eligible].tolist(), time_seconds=times[eligible].tolist(),
                mean_dbm=local['gp_mean_dbm'][eligible].tolist(), weights=(weights/weights.sum()).tolist(),
                strength=settings.soft_strength, gp_queries=len(xy), prefix_end_seconds=now,
                original_measurements=len(history), independent_evidence_points=0)


class ForwardWarningTrigger:
    """Supported forward mean crossing requests advice, independently of gain."""
    def __init__(self, threshold_dbm=-78., release_dbm=-81., horizon_seconds=10.,
                 repeat_seconds=8., max_requests=4):
        if (not all(math.isfinite(v) for v in (threshold_dbm, release_dbm, horizon_seconds, repeat_seconds))
                or release_dbm >= threshold_dbm or min(horizon_seconds, repeat_seconds) <= 0
                or type(max_requests) is not int or max_requests < 1):
            raise ValueError('invalid forward warning settings')
        self.threshold_dbm, self.release_dbm = threshold_dbm, release_dbm
        self.horizon_seconds, self.repeat_seconds, self.max_requests = horizon_seconds, repeat_seconds, max_requests
        self.last_time = self.last_request = None
        self.request_count, self.armed = 0, True

    def observe(self, forecast, route, speed, history_count):
        from .scenario_planning import arrival_queries
        now = forecast.now
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('strictly chronological monitoring required')
        self.last_time = now
        q = arrival_queries(route, now, speed, delay=.5, spacing=15.)
        indices = [i for i, t in enumerate(q['time_seconds']) if now < t <= now+self.horizon_seconds]
        xy = [q['position_m'][i] for i in indices]
        times = [q['time_seconds'][i] for i in indices]
        predicted = forecast.predict(np.asarray(xy).reshape(-1, 2), times, 'joint')
        means = predicted['weights'] @ predicted['means']
        supported = predicted['supported']
        supported_means = means[supported]
        if not len(supported_means) or float(max(supported_means)) < self.release_dbm:
            self.armed = True
        high = np.flatnonzero(supported & (means >= self.threshold_dbm))
        detected = bool(len(high))
        due_again = self.last_request is not None and now-self.last_request >= self.repeat_seconds
        blocker = None
        if detected:
            if history_count < 4:
                blocker = 'insufficient_history'
            elif self.request_count >= self.max_requests:
                blocker = 'request_budget_exhausted'
            elif not self.armed and not due_again:
                blocker = 'risk_already_notified'
            elif self.last_request is not None and now-self.last_request < self.repeat_seconds:
                blocker = 'minimum_interval'
        requested = detected and blocker is None
        if requested:
            self.last_request, self.armed = now, False
            self.request_count += 1
        return dict(time_seconds=now, requested=requested, blocker=blocker,
            reasons=['supported_forward_mean_crossing'] if detected else [],
            request_count=self.request_count, threshold_dbm=self.threshold_dbm,
            forecast_sha256=forecast.fingerprint, predicted_lead_seconds=times[int(high[0])]-now if detected else None,
            queries=dict(position_m=xy, time_seconds=times, mean_dbm=means.tolist(),
                         supported=supported.tolist(), gp_weight=predicted['gp_weight'].tolist()),
            geometry_gain_precondition=False, calibrated=False)
