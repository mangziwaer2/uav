"""CPU experiment bridge: causal observations to an external decision agent via JSON files.

No API credentials, GPU model, generated-code execution, hidden-world tools or truth
feedback are available through this interface. The external agent shares the filesystem;
its documented read restriction is procedural, not an OS security sandbox.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from .advisor import parse_decision


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + x)))


VISIBLE_FIELDS = frozenset((
    'time_seconds', 'vehicle', 'map', 'electromagnetic_estimate', 'measurement_evidence',
    'information_sources', 'route', 'directional_sensing', 'available_planners', 'profiles',
    'probe_candidates', 'active_probe', 'recent_feedback', 'constraints', 'preferred_planner',
    'waypoint_replanning', 'decision_trigger', 'latest_observation', 'sensing_warning',
    'source_inference', 'source_refinement', 'stage_plan', 'execution_feedback',
    'remaining_budgets', 'available_actions', 'route_repair',
))


def visible_context(context):
    """Keep the public builder schema and all GP evidence; reject extra top-level data.

    The builder never accepts a world/truth object. This allowlist is an additional
    interface check, not a sanitizer that can make arbitrary caller data trustworthy.
    """
    selected = {key: deepcopy(value) for key, value in context.items() if key in VISIBLE_FIELDS}
    if 'source_inference' in selected:
        source = selected['source_inference']
        selected['source_inference'] = {key: value for key, value in source.items() if key in (
            'status', 'revision', 'background_prior_dbm', 'hypothesis', 'measurements',
            'measurement_columns', 'measurement_count', 'window_seconds', 'new_point_checks',
            'public_propagation_model', 'limits', 'current_fit', 'recommended_source_model',
            'source_fields', 'meaning', 'bearing_measurements', 'bearing_measurement_columns',
        )}
    return selected


def response_policy(value, request_id, context):
    if (not isinstance(value, dict) or set(value) != {'request_id', 'decision', 'strategy', 'memory'}
            or value['request_id'] != request_id):
        raise ValueError('response schema or request identity mismatch')
    decision = value['decision']
    if not isinstance(decision, dict) or set(decision) - {
            'action', 'planner', 'profile', 'reason', 'target_id', 'waypoint_m'}:
        raise ValueError('unsupported decision fields')
    if decision.get('action') not in context['available_actions']:
        raise ValueError('action unavailable at this observation')
    raw = json.dumps(decision, ensure_ascii=False, allow_nan=False)
    parse_decision(raw, context['map']['bounds_m'][1],
                   {item['id'] for item in context.get('probe_candidates', [])}, allow_waypoint=True)
    strategy = value['strategy']
    if strategy != {'mode': 'baseline'}:
        raise ValueError('only baseline strategy is supported; use a waypoint for route proposals')
    memory = value['memory']
    if not isinstance(memory, str) or len(memory) > 1200:
        raise ValueError('memory must be a string of at most 1200 characters')
    return raw, strategy, memory


class FileDecisionAdvisor:
    """A request blocks simulation until its matching external agent response arrives.

    The 0.5s simulation decision delay remains independent of actual wall time;
    both are logged. Invalid responses are not repaired into model successes.
    """
    def __init__(self, public_root, mission_id, timeout_seconds=600.0, validator=None):
        self.validator = validator or response_policy
        self.public_root = Path(public_root).resolve()
        self.mission_id = mission_id
        self.timeout_seconds = timeout_seconds
        self.calls = 0
        self.memory = ''
        self.strategy = {'mode': 'baseline'}
        self.last_call = {}
        self.contract = (self.public_root / 'contract.md').read_text(encoding='utf-8')

    def decide(self, context):
        self.calls += 1
        identifier = f'{self.mission_id}_{self.calls:03d}'
        request_path = self.public_root / 'requests' / f'{identifier}.json'
        response_path = self.public_root / 'responses' / f'{identifier}.json'
        if request_path.exists() or response_path.exists():
            raise FileExistsError('refusing to overwrite or replay an old decision request')
        request = {'schema_version': 1, 'request_id': identifier, 'mission_id': self.mission_id,
                   'decision_number': self.calls, 'backend': 'codex_subagent',
                   'response_path': str(response_path), 'previous_memory': self.memory,
                   'previous_strategy': deepcopy(self.strategy), 'visible_context': visible_context(context)}
        request_text = json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        self.last_call = {'backend': 'codex_subagent', 'status': 'waiting', 'request_id': identifier,
                          'request_sha256': hashlib.sha256(request_text.encode()).hexdigest(),
                          'contract_sha256': hashlib.sha256(self.contract.encode()).hexdigest(),
                          'model_input_text': self.contract + '\nACTUAL REQUEST:\n' + request_text,
                          'full_orchestrator_chat_not_exported': True,
                          'generated_explanation': True,
                          'response_is_action_codebook_mapping': False,
                          'filesystem_isolation': 'procedural_allowlist_not_os_sandbox'}
        started = time.monotonic()
        save_json(request_path, request)
        try:
            while not response_path.is_file():
                if time.monotonic() - started > self.timeout_seconds:
                    raise TimeoutError('decision agent did not return a matching response')
                time.sleep(.2)
            response_text = response_path.read_text(encoding='utf-8')
            # Preserve rejected responses too; parsing acceptance and execution are
            # separate facts and neither should replace the original agent text.
            self.last_call.update(raw_agent_response=response_text,
                                  response_sha256=hashlib.sha256(response_text.encode()).hexdigest())
            value = strict_json(response_text)
            raw, strategy, memory = self.validator(value, identifier, context)
            self.last_call.update(status='accepted', raw_agent_response=response_text,
                                  strategy=strategy, memory=memory,
                                  strategy_applies_on_next_search=True,
                                  response_sha256=hashlib.sha256(response_text.encode()).hexdigest())
            self.strategy, self.memory = strategy, memory
            return raw
        except Exception as exc:
            previous_strategy = deepcopy(self.strategy)
            self.strategy = {'mode': 'baseline'}
            self.last_call.update(status='error', error=repr(exc),
                                  strategy_before_error=previous_strategy,
                                  invalid_response_reset_strategy_to_baseline=True)
            raise
        finally:
            self.last_call['wall_seconds'] = time.monotonic() - started
            append_json(self.public_root / 'exchange.jsonl', self.last_call)
