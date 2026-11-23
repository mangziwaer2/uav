"""Source-only alternatives and causal mismatch-triggered bounded revision.

No simulator, truth, flight or LLM access. Legacy bank behavior is unchanged.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np

from .event_inference import (FAMILIES, HypothesisBank, digest,
                              prepare_compound_version, validate_compound_model)
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


class CompoundSourceBank(SourceOnlyBank):
    """Explicit compound-version bank with a declared, non-evidence birth mass.

    The shared cohort prior is split equally across lineage groups, then across
    unique fields within a group. More sibling candidates cannot increase that
    lineage's *birth* mass. It is a development prior, not absolute trust or a
    transfer of the parent's posterior. Exact physical clones are rejected even
    when their IDs, parameter masks, basis order or reference times differ.

    Unlike the legacy bank, capacity overflow is an atomic error and pending
    predictions bind the full bank state. No active model is silently retired.
    """
    def __init__(self, initial, background=-100., unknown_std=15., *, birth_mass=.25,
                 max_models=64, max_birth_models=6):
        super().__init__(initial, background, unknown_std)
        if (isinstance(birth_mass, bool) or not isinstance(birth_mass, (int, float))
                or not math.isfinite(birth_mass) or not 0 < birth_mass < 1):
            raise ValueError('explicit finite birth mass between zero and one required')
        if (type(max_models) is not int or max_models < 1
                or type(max_birth_models) is not int or not 1 <= max_birth_models <= max_models):
            raise ValueError('bounded compound pool and birth capacities required')
        self.birth_mass = float(birth_mass)
        self.max_models, self.max_birth_models = max_models, max_birth_models
        self.birth_events = []
        self._pending_state_sha256 = self._pending_receipt_sha256 = None

    @staticmethod
    def _field_signature(model):
        # Canonical t=0 centers and amplitudes preserve the exact dynamic field
        # across rebasing. Rounding only absorbs binary rebase round-off; it is
        # deliberately not a broad "similar enough" scientific rejection rule.
        reference = model['reference_time_seconds']
        kernels = {}
        for base, peak in zip(model['bases'], model['peaks_dbm']):
            key = tuple(round(float(v), 10) for v in (
                *(np.asarray(base['center_m'])-reference*np.asarray(base['velocity_mps'])),
                base['scale_m'], *base['velocity_mps'], base['rate_db_per_second']))
            kernels.setdefault(key, []).append(peak-reference*base['rate_db_per_second'])
        # Splitting one identical kernel into two amplitudes is also the same
        # world. Add their power stably, rather than counting the split as new
        # evidence or a new prior group.
        bases = []
        for key, peaks in kernels.items():
            maximum = max(peaks)
            combined = maximum+10.*math.log10(sum(10.**((peak-maximum)/10.) for peak in peaks))
            bases.append((*key, round(float(combined), 10)))
        return tuple(sorted(bases)), float(model['std_db'])

    def _state_sha256(self):
        return digest(dict(history=self.history, hypotheses=self.snapshot()))

    def _validate_fresh(self, model):
        if (not isinstance(model, dict) or model.get('family') not in (*FAMILIES, 'composite')
                or 'weight' in model or 'log_weight' in model
                or type(model.get('born_index')) is not int
                or model['born_index'] != len(self.history)-1
                or model.get('reference_time_seconds') != self.history[-1]['time_seconds']
                or type(model.get('scored_points')) is not int or model['scored_points'] != 0):
            raise ValueError('fresh unweighted compound-bank version required')
        if (isinstance(model.get('std_db'), bool) or not isinstance(model.get('std_db'), (int, float))
                or not math.isfinite(model['std_db']) or model['std_db'] <= 0):
            raise ValueError('positive finite compound-bank likelihood uncertainty required')
        if model['family'] == 'composite':
            validate_compound_model(model, active=True)
        else:
            # Legacy dictionary generation is unchanged. This explicit new
            # bank validates its accepted fields against their declared mode.
            validate_compound_model(dict(model, family='composite', diagnostic_only=False,
                component_modes=[model['family']]*len(model.get('bases', []))), active=True)
        try:
            digest(model)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('finite JSON compound version required') from exc

    def birth(self, candidates):
        if self.pending is not None:
            raise ValueError('cannot change bank after issuing a prediction')
        if not isinstance(candidates, list):
            raise ValueError('explicit compound candidate list required')
        if not candidates:
            return
        if len(candidates) > self.max_birth_models:
            raise ValueError('compound birth capacity exceeded; active pool retained')
        initial = all(model['family'] == 'unknown_prior' for model in self.models)
        old_models = [] if initial else self.models
        if len(old_models)+len(candidates) > self.max_models:
            raise ValueError('compound pool capacity exceeded; active pool retained')
        ids = {model['id'] for model in self.models}
        signatures = {self._field_signature(model) for model in old_models}
        groups, stored = {}, []
        old_by_id = {model['id']: model for model in old_models}
        for candidate in candidates:
            self._validate_fresh(candidate)
            if candidate['id'] in ids:
                raise ValueError('fresh distinct version identifier required')
            signature = self._field_signature(candidate)
            if signature in signatures:
                raise ValueError('duplicate physical compound field; no clone birth mass')
            parent = old_by_id.get(candidate.get('parent_id'))
            parent_lineage = (parent.get('lineage_id', parent['id']) if parent is not None else None)
            lineage = candidate.get('lineage_id', parent_lineage or candidate.get('parent_id', candidate['id']))
            if not isinstance(lineage, str) or not lineage:
                raise ValueError('named compound lineage required')
            if parent_lineage is not None and lineage != parent_lineage:
                raise ValueError('compound child cannot split its existing parent lineage')
            version = deepcopy(candidate)
            # Persist the resolved lineage, including for legacy bootstrap
            # seeds. Subsequent fit deepcopy/rebase operations must not turn
            # two sibling revisions into independent prior groups.
            version['lineage_id'] = lineage
            stored.append(version)
            groups.setdefault(lineage, []).append(candidate['id'])
            ids.add(candidate['id']); signatures.add(signature)
        mass = 1. if initial else self.birth_mass
        weights = {identifier: mass/len(groups)/len(identifiers)
                   for identifiers in groups.values() for identifier in identifiers}
        old_weights = np.empty(0) if initial else self.log_weights+math.log1p(-mass)
        self.models = deepcopy(old_models)+stored
        self.log_weights = np.r_[old_weights, [math.log(weights[m['id']]) for m in candidates]]
        self.birth_events.append(dict(time_seconds=self.history[-1]['time_seconds'],
            history_sha256=digest(self.history), new_ids=[m['id'] for m in candidates],
            groups=deepcopy(groups), new_birth_mass=mass, new_weights=weights,
            policy='fixed_cohort_mass_equal_lineages_equal_unique_fields',
            old_scores_inherited=False, calibrated=False))

    def issue(self, position, now):
        issued = super().issue(position, now)
        self._pending_state_sha256 = self._state_sha256()
        self._pending_receipt_sha256 = digest(self.pending)
        return issued

    def observe(self, measurement):
        if (self.pending is not None and (self._state_sha256() != self._pending_state_sha256
                or digest(self.pending) != self._pending_receipt_sha256)):
            raise ValueError('compound bank or pending prediction changed after issue')
        result = super().observe(measurement)
        self._pending_state_sha256 = self._pending_receipt_sha256 = None
        return result


class SourceRenewalBank(SourceOnlyBank):
    """Suspend the failed active cohort only after a valid replacement exists.

    This is a development eligibility rule, not calibrated model rejection.
    It uses the existing all-source 8dB failure definition, never fit residuals.
    """
    def __init__(self, initial, background=-100., unknown_std=15., *, mismatch_db=8.):
        if not math.isfinite(mismatch_db) or mismatch_db <= 0:
            raise ValueError('positive finite mismatch threshold required')
        super().__init__(initial, background, unknown_std)
        self.mismatch_db = float(mismatch_db)
        self.latest_record = None
        self.replacement_events = []

    def observe(self, measurement):
        result = super().observe(measurement)
        self.latest_record = deepcopy(result)
        return result

    def birth(self, candidates):
        retired_ids={m['id'] for e in self.replacement_events for m in e['retired']}
        if any(m['id'] in retired_ids for m in candidates):
            raise ValueError('retired source identifiers cannot be reused as fresh evidence')
        before = self.snapshot()
        record = self.latest_record
        source_ids = {m['id'] for m in self.models if m['bases']}
        failures = []
        if record is not None and source_ids:
            issued = record['issued']
            if (record['observation'] != self.history[-1]
                    or issued['history_sha256'] != digest(self.history[:-1])):
                raise ValueError('renewal requires the latest causal prediction')
            y = record['observation']['interference_dbm']
            failures = [dict(id=p['id'], absolute_error_db=abs(y-p['mean_dbm']))
                        for p in issued['predictions'] if p['id'] in source_ids]
        replace = bool(source_ids) and {r['id'] for r in failures} == source_ids and all(
            r['absolute_error_db'] >= self.mismatch_db for r in failures)
        # Parent validates the entire batch and pending-observation guard first.
        # An empty/invalid fit cannot discard the existing active cohort.
        super().birth(candidates)
        if replace and candidates:
            self.models = self.models[-len(candidates):]
            self.log_weights = np.full(len(candidates), -math.log(len(candidates)))
            self.replacement_events.append(dict(time_seconds=self.history[-1]['time_seconds'],
                history_sha256=digest(self.history), threshold_db=self.mismatch_db,
                reason='all_preissued_source_predictions_mismatch_and_new_fit_committed',
                retired=before, failures=failures, failure_record=deepcopy(record),
                active_after=self.snapshot(), old_scores_inherited=False,
                meaning='suspended from active pool; not globally disproved or calibrated rejection'))


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
