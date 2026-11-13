"""因果可靠性反馈及有限在线搜索选择；不导入真值评价。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .agent_navigation import append_json, save_json, strict_json
from .balanced_objective import _BalancedSession, PowerTimeSettings, choose, paired_seed, power_time_score
from .belief import ElectromagneticBelief
from .decision_objective import ObjectiveSettings
from .environment import Measurement
from .repair_diagnostics import SEARCH_PROFILES, search_proposal
from .repair_process import RepairState, canonical_path, feedback_outcome
from .route_repair import RepairSession, anchors, parse_proposal

VERSION = 'causal_repair_feedback_v1'
MODES = ('pool_only', 'fixed_two', 'numeric_feedback', 'llm_feedback')
PROFILES = ('stop',) + SEARCH_PROFILES


def prequential_evidence(config, geometry, points, now):
    """每条读数先预测、后同化；只接受时序有效的已收点测。"""
    belief = ElectromagneticBelief(config, geometry)
    records = []
    last = -1.
    for point in points:
        t = point['time_seconds']
        if not last <= t <= now or not np.isfinite(t):
            raise ValueError('future or unordered measurement')
        snapshot = belief.snapshot(t)
        mean, std, supported = snapshot.at(point['position_m'])
        value = point['interference_dbm']
        if not np.isfinite(value):
            raise ValueError('finite reading required')
        records.append({'position_m': deepcopy(point['position_m']), 'time_seconds': t,
                        'observed_dbm': value, 'predicted_before_reading_dbm': mean if supported else None,
                        'predictive_std_db': std if supported else None, 'supported_before_reading': supported,
                        'residual_db': value-mean if supported else None})
        belief.update(Measurement(tuple(point['position_m']), t, value))
        last = t
    recent = [r for r in records if 0 <= now-r['time_seconds'] <= config.belief.max_age_seconds]
    checks = recent[-8:]
    errors = np.array([r['residual_db'] for r in checks if r['supported_before_reading']], float)
    positions = np.asarray([r['position_m'] for r in recent], float).reshape(-1, 2)
    times = np.asarray([r['time_seconds'] for r in recent], float)
    if len(positions):
        xy = positions-positions.mean(axis=0)
        singular = np.linalg.svd(xy, compute_uv=False)
        ratio = float(singular[1]/singular[0]) if len(singular)>1 and singular[0]>1e-12 else 0.
        design = np.column_stack((xy/config.belief.gp_length_scale_m,
                                  (times-times.mean())/config.belief.gp_time_scale_seconds))
        rank = int(np.linalg.matrix_rank(design, tol=1e-8))
    else:
        ratio, rank = 0., 0
    return {'origin': 'received_point_measurements_predict_before_update', 'observation_count': len(points),
            'recent_checks': checks, 'supported_check_count': len(errors),
            'supported_mean_residual_db': float(errors.mean()) if len(errors) else None,
            'supported_rmse_db': float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            'trajectory_window_seconds': config.belief.max_age_seconds,
            'trajectory_second_first_singular_ratio': ratio,
            'centered_xy_time_design_rank_out_of3': rank,
            'lateral_distribution_identified': False,
            'trajectory_near_collinear': ratio <= .001,
            'spatial_temporal_affine_effects_separable_on_this_design': rank == 3,
            'uncertainty_calibrated': False, 'new_measurements': 0,
            'limits': 'Predictive checks are correlated historical samples, not confidence calibration or source/side inference.'}


def validate_response(raw, request_id):
    value = strict_json(raw) if isinstance(raw, str) else deepcopy(raw)
    if not isinstance(value, dict) or set(value) != {'request_id', 'profile', 'reason', 'memory'} or value['request_id'] != request_id:
        raise ValueError('response identity or fields invalid')
    if value['profile'] not in PROFILES:
        raise ValueError('unknown profile')
    if not isinstance(value['reason'], str) or not 1 <= len(value['reason'].strip()) <= 600:
        raise ValueError('nonempty short reason required')
    if not isinstance(value['memory'], str) or len(value['memory']) > 1200:
        raise ValueError('short same-task memory required')
    return value


class FileFeedbackPolicy:
    """真实外部会话文件桥；不通过旧visible_context删除新合同字段。"""
    def __init__(self, public_root, timeout_seconds=600):
        self.root = Path(public_root)
        self.contract = (self.root/'contract.md').read_text()
        self.timeout = timeout_seconds
        self.calls, self.memory, self.last_call = 0, '', {}

    def decide(self, state):
        self.calls += 1
        request_id = f'select_{self.calls:02d}'
        request_path = self.root/'requests'/f'{request_id}.json'
        response_path = self.root/'responses'/f'{request_id}.json'
        if request_path.exists() or response_path.exists():
            raise FileExistsError('refuse to overwrite request/response')
        request = {'request_id': request_id, 'response_path': str(response_path.resolve()),
                   'previous_memory': self.memory, 'state': state}
        save_json(request_path, request)
        self.last_call = {'request_id': request_id, 'status': 'waiting',
            'request_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
            'contract_sha256': hashlib.sha256(self.contract.encode()).hexdigest(),
            'model_input_text': self.contract+'\nACTUAL REQUEST:\n'+request_path.read_text(),
            'raw_response': None, 'parsed_response': None, 'tokens': None,
            'filesystem_isolation': 'procedural_allowlist_not_os_sandbox',
            'full_orchestrator_chat_not_exported': True}
        started = time.monotonic()
        try:
            while not response_path.exists():
                if time.monotonic()-started > self.timeout:
                    raise TimeoutError('independent decision agent response timed out')
                time.sleep(.2)
            raw = response_path.read_text()
            self.last_call.update(raw_response=raw, response_sha256=hashlib.sha256(raw.encode()).hexdigest())
            value = validate_response(raw, request_id)
            self.last_call.update(status='accepted', parsed_response=value)
            self.memory = value['memory']
            return value
        except Exception as exc:
            self.last_call.update(status='error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            self.last_call['wall_seconds'] = time.monotonic()-started
            append_json(self.root/'exchange.jsonl', self.last_call)


class FeedbackSession(_BalancedSession):
    def search(self, a, b, quota, sampling, purpose):
        key = self.active_profile+'::'+canonical_path(self.best)+'::'+purpose
        actual = paired_seed(self.scene_seed, self.snapshot.time_seconds, VERSION, key)
        prior = self.seed
        self.seed = actual-len(self.searches)
        try:
            path = RepairSession.search(self, a, b, quota, sampling, purpose)
        finally:
            self.seed = prior
        self.searches[-1].update(actual_seed=actual, seed_stage=VERSION, seed_purpose=key)
        return path


def public_state(session, reliability, feedback, turn):
    ctx = session.request(turn, 2)
    state = RepairState.from_context(ctx).to_dict()
    current = power_time_score(session.problem, session.best, session.speed, session.score_settings)
    session.score_nodes += current['field_nodes']
    best = choose(session.records)
    top = sorted((r for r in session.records if r['feasible']),
                 key=lambda r: (r['balance']['score_seconds'], r['metrics']['length_m']))[:5]
    summary = [{'id': r['id'], 'length_m': r['metrics']['length_m'], 'balance': deepcopy(r['balance']),
                'score_delta_seconds': r['balance']['score_seconds']-best['balance']['score_seconds']} for r in top]
    state.update(contract_version=VERSION, task_objective='fixed_task_time_plus_power; current_GP_C_is_only_a_proxy',
                 current_balance=current, selection_settings=deepcopy(session.score_settings.__dict__),
                 reliability=deepcopy(reliability), previous_numeric_feedback=deepcopy(feedback),
                 common_pool_summary=summary, available_profiles=list(PROFILES),
                 choices_left=3-turn, per_search_sample_quota=min(700, session.work.summary()['samples_left']),
                 objective_and_GP_modification_allowed=False, true_future_evaluation_included=False)
    return state, ctx


def numerical_choice(mode, state):
    history = state['previous_numeric_feedback']
    if mode == 'pool_only':
        return 'stop'
    if not history:
        return 'local_pad3'
    if mode == 'numeric_feedback' and history[-1]['current_score_relative_gain'] >= .01:
        return 'stop'
    return 'global_focus'


def run_selection(snapshot, case, pool_report, reliability, mode, policy=None):
    if mode not in MODES or (mode == 'llm_feedback') != (policy is not None):
        raise ValueError('mode/policy mismatch')
    started = time.perf_counter()
    settings = PowerTimeSettings(**case['score_settings'])
    selected = choose(pool_report['candidates'])
    route = anchors(selected['path'], 80.)
    session = FeedbackSession(snapshot, case['context'], route, case['max_length_m'], ObjectiveSettings(),
                              case['seed'], pool_report, settings, 'balanced')
    traces, feedbacks, used = [], [], set()
    model_calls = model_errors = 0
    model_wall = 0.
    for turn in (1, 2):
        state, ctx = public_state(session, reliability, feedbacks, turn)
        before = session.work.summary()
        old_score = state['current_balance']['score_seconds']
        profile, reason, proposal, error, call = None, None, None, None, None
        try:
            if policy is None:
                profile, reason = numerical_choice(mode, state), '预声明数值选择'
            else:
                model_calls += 1
                try:
                    value = policy.decide(deepcopy(state))
                finally:
                    call = deepcopy(policy.last_call)
                    model_wall += call.get('wall_seconds', 0.)
                profile, reason = value['profile'], value['reason']
            if profile not in PROFILES:
                raise ValueError('invalid selected profile')
            if profile == 'stop':
                fb = {'accepted': False, 'reason': 'kept_incumbent', 'operator': 'keep'}
            elif profile in used:
                fb = {'accepted': False, 'reason': 'duplicate_operator', 'operator': profile}
            elif before['samples_left'] < (2 if profile.startswith('via') else 1) or before['edges_left'] <= len(session.best)-1:
                fb = {'accepted': False, 'reason': 'insufficient_shared_resources', 'operator': profile}
            else:
                used.add(profile)
                session.active_profile = profile
                proposal = parse_proposal(search_proposal(ctx, profile, min(700, before['samples_left'])), ctx)
                proposal['reason'] = reason
                fb = session.apply(proposal)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            model_errors += int(policy is not None)
            fb = {'accepted': False, 'reason': 'policy_error', 'operator': profile, 'error': error}
        current = choose(session.records)
        score = current['balance']['score_seconds']
        public_feedback = {key: deepcopy(fb[key]) for key in
            ('operator', 'accepted', 'reason', 'candidate_metrics', 'candidate_balance', 'duplicate_of') if key in fb}
        public_feedback.update(profile=profile, current_best_balance=deepcopy(current['balance']),
            previous_score_seconds=old_score, current_score_delta_seconds=score-old_score,
            current_score_relative_gain=(old_score-score)/old_score,
            resources=session.work.summary(), no_new_measurements=True, true_future_included=False,
            work_delta={k: session.work.summary()[k]-before[k] for k in ('samples', 'edges', 'cache_hits')})
        feedbacks.append(public_feedback)
        traces.append({'turn': turn, 'state': state, 'selected_profile': profile, 'reason': reason,
                       'parsed_proposal': proposal, 'feedback': public_feedback, 'error': error, 'model_exchange': call})
        session.trace.append({'feedback': fb})
        if profile == 'stop' or error or fb['reason'] == 'insufficient_shared_resources':
            break
    session.work.reserved_edges = 0
    for a, b in zip(session.best, session.best[1:]):
        session.work.take_edge()
        session.work.verification_edges += 1
        if not snapshot.geometry.edge_free(a, b):
            raise ValueError('final route geometry invalid')
    best = choose(session.records)
    assert canonical_path(best['path']) == canonical_path(session.best)
    elapsed = time.perf_counter()-started
    return {'contract_version': VERSION, 'mode': mode, 'path': session.best,
            'candidates': session.records, 'trace': traces, 'searches': session.searches,
            'selected_id': best['id'], 'selected_balance': best['balance'],
            'starting_resources_already_paid': deepcopy(pool_report['resources']),
            'resources': session.work.summary(), 'score_field_nodes': session.score_nodes,
            'model_calls': model_calls, 'model_errors': model_errors,
            'numeric_errors': sum(t['error'] is not None for t in traces) if policy is None else 0,
            'model_wall_seconds': model_wall, 'numeric_wall_seconds': max(0., elapsed-model_wall),
            'wall_seconds': elapsed, 'fallback_searches': 0, 'new_measurements': 0,
            'common_pool_replayed': False, 'common_anchor_spacing_m': 80.}
