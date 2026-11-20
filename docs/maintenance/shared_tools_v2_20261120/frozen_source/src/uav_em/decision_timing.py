"""Causal advisory request timing; no inference, sensor, LLM or planner calls.

The defaults are frozen prototype settings, not calibrated safety thresholds.
This module does not switch a route or wait for an LLM response.
"""
from dataclasses import dataclass
import math
import statistics


class PowerRiseTrigger:
    """Minimal per-measurement trigger; no expiry, initial call or learned gate.

    Compare with the lowest power since the last request, so gradual increases
    are visible. Six dB is a prototype change setting, not a safe power limit.
    """
    def __init__(self, rise_db=6., cooldown_seconds=2., max_requests=4):
        if not all(math.isfinite(x) and x > 0 for x in (rise_db, cooldown_seconds)):
            raise ValueError('positive finite rise and cooldown required')
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError('positive integer request cap required')
        self.rise_db, self.cooldown_seconds, self.max_requests = rise_db, cooldown_seconds, max_requests
        self.last_time = self.last_power = self.baseline = self.last_request = None
        self.request_count = 0

    def observe(self, now, power_dbm):
        if not math.isfinite(now) or now < 0 or not math.isfinite(power_dbm):
            raise ValueError('finite causal measurement required')
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('strictly increasing measurement time required')
        baseline = power_dbm if self.baseline is None else min(self.baseline, power_dbm)
        rise = power_dbm-baseline
        rising = self.last_power is not None and power_dbm > self.last_power
        detected = rising and rise >= self.rise_db
        blocker = None
        if detected:
            if self.request_count >= self.max_requests:
                blocker = 'request_budget_exhausted'
            elif self.last_request is not None and now-self.last_request < self.cooldown_seconds:
                blocker = 'cooldown'
        requested = detected and blocker is None
        if requested:
            self.last_request = now
            self.request_count += 1
        self.baseline = power_dbm if requested else baseline
        self.last_time, self.last_power = now, power_dbm
        return dict(time_seconds=now, power_dbm=power_dbm, baseline_dbm=baseline,
                    rise_db=rise, requested=requested, blocker=blocker,
                    reasons=['observed_power_rise'] if detected else [],
                    request_count=self.request_count, source_direction_known=False)


@dataclass(frozen=True)
class TimingSettings:
    min_interval_seconds: float = 2.
    maximum_age_seconds: float = 8.
    max_requests: int = 4
    innovation_window: int = 3
    innovation_squared_mean: float = 2.25
    change_db: float = 6.
    llm_lead_seconds: float = 15.

    def __post_init__(self):
        values = (self.min_interval_seconds, self.maximum_age_seconds,
                  self.innovation_squared_mean, self.change_db, self.llm_lead_seconds)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError('finite positive timing settings required')
        if self.maximum_age_seconds < self.min_interval_seconds:
            raise ValueError('expiry cannot precede minimum interval')
        if type(self.max_requests) is not int or self.max_requests < 1:
            raise ValueError('positive integer request cap required')
        if type(self.innovation_window) is not int or self.innovation_window < 1:
            raise ValueError('positive integer window required')


def predicted_escalation(score, now, forecast_at, horizon=20., rise_db=6., mass=.25):
    """Inspect previously saved route query values, not a new forecast query.

    Return the first future query where sufficient alternative-model mass has
    predicted a rise relative to its own first remaining query. A rise is not
    a device failure event. Mixture weights need not be calibrated probabilities.
    """
    if not all(math.isfinite(x) for x in (now, forecast_at, horizon, rise_db, mass)):
        raise ValueError('finite forecast inputs required')
    if forecast_at > now or horizon <= 0 or rise_db <= 0 or not 0 < mass <= 1:
        raise ValueError('causal forecast and positive range required')
    times = score['queries']['time_seconds']
    means = score['scenario_mean_dbm']
    scenarios = score['scenario_scores']
    if len(means) != len(scenarios) or not times:
        raise ValueError('nonempty aligned route forecast required')
    if not all(math.isfinite(t) for t in times) or any(a >= b for a,b in zip(times,times[1:])):
        raise ValueError('strictly increasing finite query times required')
    weights = [s['weight'] for s in scenarios]
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights),1.,abs_tol=1e-9):
        raise ValueError('normalized finite weights required')
    if any(len(row) != len(times) or not all(math.isfinite(y) for y in row) for row in means):
        raise ValueError('finite aligned model means required')
    indices = [i for i,t in enumerate(times) if now <= t <= now+horizon]
    if not indices:
        return None
    first = indices[0]
    crossed = set()
    for i in indices[1:]:
        crossed.update(j for j,row in enumerate(means) if row[i]-row[first] >= rise_db)
        weight = math.fsum(weights[j] for j in crossed)
        if weight+1e-12 >= mass:
            return dict(forecast_at_seconds=forecast_at, event_time_seconds=times[i],
                        forecast_key=str(forecast_at), model_mass=weight,
                        model_ids=[scenarios[j]['id'] for j in sorted(crossed)],
                        event='conditional_route_power_rise', calibrated=False)
    return None


class TimingMonitor:
    def __init__(self, settings=TimingSettings()):
        self.settings = settings
        self.last_observation = None
        self.last_request = None
        self.request_count = 0
        self.innovations = []
        self.powers = []
        self.notified_forecasts = set()

    def observe(self, now, power_dbm, innovation_squared=None, forecast_event=None,
                route_feasible=True, goal_reached=False):
        s = self.settings
        if not math.isfinite(now) or now < 0 or not math.isfinite(power_dbm):
            raise ValueError('finite observation required')
        if self.last_observation is not None and now <= self.last_observation:
            raise ValueError('observations must be strictly chronological')
        if innovation_squared is not None and (not math.isfinite(innovation_squared) or innovation_squared < 0):
            raise ValueError('nonnegative finite innovation required')
        if type(route_feasible) is not bool or type(goal_reached) is not bool:
            raise ValueError('explicit mission flags required')
        deadline_slack = None
        if forecast_event is not None:
            origin, event = forecast_event['forecast_at_seconds'], forecast_event['event_time_seconds']
            if not all(math.isfinite(x) for x in (origin,event)) or not 0 <= origin <= now or event < origin:
                raise ValueError('prediction cannot originate in the future')
            if not isinstance(forecast_event['forecast_key'],str):
                raise ValueError('forecast revision key required')
            deadline_slack = event-now-s.llm_lead_seconds
        self.last_observation = now
        previous = self.powers[-3:]
        change = abs(power_dbm-statistics.median(previous)) if len(previous)==3 else 0.
        self.powers.append(power_dbm)
        if innovation_squared is not None:
            self.innovations.append(innovation_squared)
        recent = self.innovations[-s.innovation_window:]
        mismatch = len(recent)==s.innovation_window and statistics.mean(recent)>=s.innovation_squared_mean
        reasons = []
        if not goal_reached:
            if self.last_request is None:reasons.append('initial_plan')
            if not route_feasible:reasons.append('route_infeasible')
            if mismatch:reasons.append('predictive_mismatch')
            if change>=s.change_db:reasons.append('observed_power_change')
            if forecast_event is not None and deadline_slack <= 0 and forecast_event['forecast_key'] not in self.notified_forecasts:
                reasons.append('forecast_action_deadline')
            if self.last_request is not None and now-self.last_request>=s.maximum_age_seconds:
                reasons.append('plan_expired')
        urgent = any(r in reasons for r in ('route_infeasible','forecast_action_deadline'))
        blocker = None
        if reasons:
            if self.request_count>=s.max_requests:blocker='request_budget_exhausted'
            elif self.last_request is not None and now-self.last_request<s.min_interval_seconds and not urgent:
                blocker='minimum_interval'
        requested = bool(reasons) and blocker is None
        if requested:
            self.last_request = now
            self.request_count += 1
            self.innovations.clear()
            if 'forecast_action_deadline' in reasons:
                self.notified_forecasts.add(forecast_event['forecast_key'])
        return dict(time_seconds=now, reasons=reasons, requested=requested, blocker=blocker,
                    urgent=urgent, request_count=self.request_count,
                    recent_innovation_mean=statistics.mean(recent) if recent else None,
                    change_from_previous_median_db=change,
                    llm_deadline_slack_seconds=deadline_slack,
                    fast_numeric_review_needed=not goal_reached and (
                        not route_feasible or ('forecast_action_deadline' in reasons and deadline_slack < 0)),
                    goal_reached=goal_reached, route_changed=False)
