"""Source-only alternatives and causal mismatch-triggered bounded revision.

No simulator, truth, flight or LLM access. Legacy bank behavior is unchanged.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .event_inference import HypothesisBank, digest
from .scenario_planning import arrival_queries


class SourceOnlyBank(HypothesisBank):
    """Relative evidence conditional on a source dictionary, not model adequacy."""
    def __init__(self, initial, background=-100., unknown_std=15.):
        if not math.isfinite(background) or not math.isfinite(unknown_std) or unknown_std <= 0:
            raise ValueError('finite background and positive unknown uncertainty required')
        # The inherited causal issue/observe protocol evaluates an empty kernel as
        # the declared -100dBm floor. Never use a last measurement as a global mean.
        if background != -100.:
            raise ValueError('equivalent-field background must be -100dBm')
        super().__init__(initial)
        self.models = [dict(id='uninitialized_unknown', family='unknown_prior', std_db=unknown_std,
            born_index=0, scored_points=0, reference_time_seconds=initial['time_seconds'], bases=[], peaks_dbm=[])]
        self.log_weights = np.zeros(1)

    def birth(self, candidates):
        initial = all(m['family'] == 'unknown_prior' for m in self.models)
        # Parent validation is atomic, including the pending-prediction guard.
        super().birth(candidates)
        if candidates and initial:
            self.models = self.models[-len(candidates):]
            self.log_weights = np.full(len(candidates), -math.log(len(candidates)))


@dataclass(frozen=True)
class RevisionSettings:
    enable_mismatch: bool = False
    initialize_seconds: float = 8.
    initialize_points: int = 9
    mismatch_db: float = 8.
    minimum_seconds: float = 2.
    risk_repeat_seconds: float = 8.
    max_requests: int = 4

    def __post_init__(self):
        if type(self.enable_mismatch) is not bool:
            raise ValueError('explicit mismatch policy required')
        for key in ('initialize_seconds', 'mismatch_db', 'minimum_seconds', 'risk_repeat_seconds'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError('positive finite revision settings required')
        if type(self.initialize_points) is not int or self.initialize_points < 4:
            raise ValueError('pilot needs at least four observations')
        if type(self.max_requests) is not int or not 1 <= self.max_requests <= 4:
            raise ValueError('bounded event budget required')


class SourceRevisionTrigger:
    def __init__(self, settings):
        self.settings = settings
        self.last_time = self.last_request = None
        self.request_count, self.armed = 0, True

    def observe(self, forecast, route, speed, history_count, latest_record=None):
        now, settings = forecast.now, self.settings
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('strictly chronological source monitoring required')
        self.last_time = now
        q = arrival_queries(route, now, speed, delay=.5, spacing=15.)
        indices = [i for i, t in enumerate(q['time_seconds']) if now < t <= now+10.]
        xy = [q['position_m'][i] for i in indices]
        times = [q['time_seconds'][i] for i in indices]
        d = forecast.predict(np.asarray(xy).reshape(-1, 2), times, 'joint')
        means, support = d['weights'] @ d['means'], d['supported']
        if not np.any(support) or max(means[support]) < -81.:
            self.armed = True
        high = np.flatnonzero(support & (means >= -78.))
        has_source = any(m['bases'] for m in forecast.models)
        min_error = None
        if latest_record is not None:
            issued = latest_record['issued']
            if issued['time_seconds'] != now or latest_record['observation']['time_seconds'] != now:
                raise ValueError('mismatch must evaluate the latest pre-issued prediction')
            if (issued['history_sha256'] != digest(forecast.history[:-1]) or
                    latest_record['observation'] != forecast.history[-1] or
                    issued['position_m'] != latest_record['observation']['position_m']):
                raise ValueError('mismatch must match the causal public prefix and receiver')
            y = latest_record['observation']['interference_dbm']
            predictions = [p for p in issued['predictions'] if p['id'] != 'uninitialized_unknown']
            if predictions:
                min_error = min(abs(y-p['mean_dbm']) for p in predictions)
        mismatch = bool(has_source and min_error is not None and min_error >= settings.mismatch_db)
        reasons = []
        if not has_source and now >= settings.initialize_seconds and history_count >= settings.initialize_points:
            reasons.append('initial_source_fit')
        if mismatch and settings.enable_mismatch:
            reasons.append('all_source_predictions_mismatch')
        elapsed = math.inf if self.last_request is None else now-self.last_request
        if len(high) and (self.armed or elapsed >= settings.risk_repeat_seconds):
            reasons.append('supported_forward_mean_crossing')
        blocker = None
        if reasons:
            if history_count < settings.initialize_points or now < settings.initialize_seconds:
                blocker = 'insufficient_initial_history'
            elif self.request_count >= settings.max_requests:
                blocker = 'request_budget_exhausted'
            elif elapsed < settings.minimum_seconds:
                blocker = 'minimum_interval'
        requested = bool(reasons) and blocker is None
        if requested:
            self.request_count += 1
            self.last_request, self.armed = now, False
        return dict(time_seconds=now, requested=requested, blocker=blocker, reasons=reasons,
            request_count=self.request_count, threshold_dbm=-78., forecast_sha256=forecast.fingerprint,
            predicted_lead_seconds=times[int(high[0])]-now if len(high) else None,
            queries=dict(position_m=xy, time_seconds=times, mean_dbm=means.tolist(),
                         supported=support.tolist(), gp_weight=d['gp_weight'].tolist()),
            has_source=has_source, all_source_mismatch=mismatch,
            min_abs_source_error_db=min_error, risk_detected=bool(len(high)),
            revision_settings=asdict(settings), calibrated=False,
            weights_meaning='relative conditional on current source-model set; not absolute adequacy')
