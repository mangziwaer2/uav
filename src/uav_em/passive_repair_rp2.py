"""Freeze RP2 and synchronously relay public tool calls to fresh built-in subagents."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
from secrets import randbits
import sys
from time import monotonic
from uuid import UUID

from .environment import ElectromagneticWorld
from .equivalent_prediction_ep1 import environment_record
from .passive_repair_alert import RepairAlertSettings
from .passive_repair_checkpoint import _json, _write
from .passive_repair_flight import fly_repair
from .passive_repair_pair import publish_tool_pair
from .passive_repair_rp1 import FAMILIES, _record, _scenario
from .passive_repair_tools import INSTRUCTIONS
from .passive_route_exchange import ReceivedReply, _sha256


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/passive_repair_RP2_20261126'
CONTRACT = ROOT / 'docs/passive_repair_RP2_contract.md'
ALERT = RepairAlertSettings(early_dbm=-85., early_z=1., worsening_db=3., clear_samples=2)
TIMEOUT_S, TOTAL_TIMEOUT_S = 180., 1200.
EFFORT = 'medium'
EXPECTED_RUNTIME_MODEL = 'gpt-6.1-sol'
ISOLATION = (
    'You are an isolated decision maker for one civilian UAV simulation tool call. '
    'The JSON below is the complete scientific input, including all observations and feedback you may use. '
    'Use no tools, files, repository, network, other sessions, or commentary. '
    'The repository maintenance instructions do not authorize reading any files for this decision. '
    'Only decide the next tool call from this packet. Do not invent observations or treat unknown risk as zero. '
    'Return exactly one raw JSON line as requested by the packet, without markdown.\n\n'
)
PACKET_KEYS = {'request_id', 'checkpoint_id', 'state', 'routes', 'geometry_attempts_left',
               'gp_query_points_left', 'comparisons', 'last_feedback'}
STATE_KEYS = {'t_s', 'xy_m', 'goal_xy_m', 'flown_m', 'cap_m', 'hard_left_m', 'shortest_known_m',
              'energy_wh', 'reserve_wh', 'flight_power_w', 'remaining_time_s', 'peak_dbm', 'window_s',
              'points', 'geometry', 'alert'}
GEOMETRY_KEYS = {'max_xy_m', 'buildings', 'obstacles_m', 'speed_mps', 'mode', 'altitude_m',
                 'clearance_m', 'vertical_clearance_m'}
ALERT_KEYS = {'review', 'reason', 'measured_dbm', 'peak_dbm', 'lookahead_alert', 'residual_alert', 'residual_z'}


def public_message(wire):
    body = json.loads(wire)
    if set(body) != {'instructions', 'input'} or not isinstance(body['instructions'], str):
        raise ValueError('unexpected scientific input envelope')
    packet = body['input']
    if set(packet) != PACKET_KEYS or set(packet['state']) != STATE_KEYS:
        raise ValueError('unexpected public checkpoint fields')
    state = packet['state']
    if set(state['geometry']) != GEOMETRY_KEYS or set(state['alert']) != ALERT_KEYS:
        raise ValueError('unexpected geometry or alert fields')
    points = state['points']
    if (not points or any(len(row) != 5 or not 0 <= row[0] <= state['t_s'] for row in points)
            or any(a[0] >= b[0] for a, b in zip(points, points[1:]))
            or points[-1][:3] != [state['t_s'], *state['xy_m']]):
        raise ValueError('only the causal current-position point prefix is allowed')
    return ISOLATION + wire.decode('utf-8')


def _source_files():
    paths = [*Path(__file__).parent.glob('*.py'), *Path(__file__).parent.glob('*.js'),
             *Path(__file__).parent.glob('*.html'), ROOT / 'requirements.txt',
             ROOT / 'tests/test_passive_repair_rp2.py', ROOT / 'tests/test_passive_repair_replay_progress.py']
    return sorted(paths)


def freeze(out=OUTPUT, *, label='RP2', contract=CONTRACT):
    seeds = [(randbits(64) | (1 << 63), randbits(64) | (1 << 63)) for _ in FAMILIES]
    manifest = dict(experiment=label, utc=datetime.now(timezone.utc).isoformat(),
                    contract_sha256=_sha256(contract.read_bytes()), environment=environment_record(),
                    isolation_sha256=_sha256(ISOLATION.encode()), tool_instructions_sha256=_sha256(INSTRUCTIONS.encode()),
                    source={str(path.relative_to(ROOT)): _sha256(path.read_bytes()) for path in _source_files()},
                    arms=['numerical', 'llm'], alert=asdict(ALERT), route_progress=True,
                    transport=dict(kind='built_in_subagent', fork_context=False, requested_model='inherit',
                                   expected_runtime_model=EXPECTED_RUNTIME_MODEL, reasoning_effort=EFFORT,
                                   per_call_seconds=TIMEOUT_S, per_scene_seconds=TOTAL_TIMEOUT_S,
                                   batch_seconds=3 * TOTAL_TIMEOUT_S, usage='from_raw_session_when_present', price=None),
                    scenes={f'{label}-{index:02d}': _record(family, *pair)
                            for index, (family, pair) in enumerate(zip(FAMILIES, seeds, strict=True), 1)})
    out.mkdir()
    _write(out / 'contract.md', contract.read_bytes())
    _write(out / 'isolation.txt', ISOLATION.encode())
    _write(out / 'tool_instructions.txt', INSTRUCTIONS.encode())
    for path in _source_files():
        destination = out / 'implementation' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write(destination, path.read_bytes())
    raw = _json(manifest)
    _write(out / 'manifest.json', raw)
    return _sha256(raw)


def verify(out, fingerprint, *, contract=CONTRACT):
    raw = (out / 'manifest.json').read_bytes()
    if _sha256(raw) != fingerprint:
        raise ValueError('RP2 manifest binding mismatch')
    manifest = json.loads(raw)
    if (manifest['contract_sha256'] != _sha256(contract.read_bytes())
            or (out / 'contract.md').read_bytes() != contract.read_bytes()
            or manifest['environment'] != environment_record()
            or manifest['isolation_sha256'] != _sha256(ISOLATION.encode())
            or (out / 'isolation.txt').read_bytes() != ISOLATION.encode()
            or manifest['tool_instructions_sha256'] != _sha256(INSTRUCTIONS.encode())
            or (out / 'tool_instructions.txt').read_bytes() != INSTRUCTIONS.encode()
            or manifest['alert'] != asdict(ALERT)):
        raise ValueError('RP2 contract, environment, prompt or alert changed')
    for name, expected in manifest['source'].items():
        if any(_sha256(path.read_bytes()) != expected for path in (ROOT / name, out / 'implementation' / name)):
            raise ValueError('RP2 frozen source changed: ' + name)
    return manifest


def inspect_rollout(raw, *, session_id, expected_message, raw_reply):
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    metadata = [event['payload'] for event in events if event.get('type') == 'session_meta']
    contexts = [event['payload'] for event in events if event.get('type') == 'turn_context']
    if (len(metadata) != 1 or metadata[0].get('id', metadata[0].get('session_id')) != session_id
            or len(contexts) != 1 or contexts[0].get('model') != EXPECTED_RUNTIME_MODEL
            or contexts[0].get('effort') != EFFORT):
        raise ValueError('subagent identity or reasoning setting mismatch')
    users, finals = [], []
    for event in events:
        if event.get('type') != 'response_item':
            continue
        item = event['payload']
        kind = item.get('type', '')
        if kind not in {'message', 'reasoning', 'additional_tools'}:
            raise ValueError('subagent peripheral tool activity or unknown item: ' + kind)
        if kind != 'message':
            continue
        message = ''.join(part.get('text', '') for part in item.get('content', []) if isinstance(part, dict))
        if item.get('role') == 'user':
            users.append(message)
        if item.get('role') == 'assistant' and item.get('phase') in {'final', 'final_answer'}:
            finals.append(message)
        if item.get('role') == 'assistant' and item.get('phase') == 'commentary':
            raise ValueError('subagent sent commentary instead of isolated final output')
    if (not users or users[-1] != expected_message or len(users) > 2
            or len(users) == 2 and not users[0].startswith('# AGENTS.md instructions for ')
            or finals != [raw_reply]):
        raise ValueError('subagent input history or final reply mismatch')
    usage = [event['payload'].get('info') for event in events if event.get('type') == 'event_msg'
             and event.get('payload', {}).get('type') == 'token_count']
    return dict(runtime_model_id=contexts[0]['model'], reasoning_effort=contexts[0]['effort'],
                upstream_identity_verified=False, peripheral_tool_calls=0, user_message_count=len(users),
                maintenance_context_injected=len(users) == 2, usage=usage[-1] if usage else None, price=None)


class SubagentRelay:
    def __init__(self, out, input_stream=None, output_stream=None):
        self.out = out
        out.mkdir()
        self.input = sys.stdin if input_stream is None else input_stream
        self.output = sys.stdout if output_stream is None else output_stream
        self.wait_seconds = 0.
        self.calls = 0
        self.integrity_error = None

    def __call__(self, wire, fingerprint):
        message = public_message(wire)
        if _sha256(wire) != fingerprint:
            raise OSError('public relay request hash mismatch')
        remaining = min(TIMEOUT_S, TOTAL_TIMEOUT_S - self.wait_seconds)
        if remaining <= 0:
            return None
        self.calls += 1
        destination = self.out / f'{self.calls:04d}'
        destination.mkdir()
        _write(destination / 'request.json', wire)
        _write(destination / 'message.txt', message.encode())
        request_record = dict(request_sha256=fingerprint, message_sha256=_sha256(message.encode()),
                              fork_context=False, reasoning_effort=EFFORT, requested_model='inherit',
                              started_utc=datetime.now(timezone.utc).isoformat(), timeout_s=remaining)
        _write(destination / 'request_record.json', _json(request_record))
        print(json.dumps(dict(event='subagent_request', directory=str(destination.relative_to(ROOT)),
                              request_sha256=fingerprint, timeout_s=remaining)), file=self.output, flush=True)
        started = monotonic()
        status = dict(status='missing_reply', request_sha256=fingerprint)
        reply = None
        try:
            ready, _, _ = select.select([self.input], [], [], remaining)
            if ready:
                line = self.input.readline()
                if line:
                    envelope = json.loads(line)
                    _write(destination / 'orchestrator_receipt.json', _json(envelope))
                    session_id = str(UUID(envelope['session_id']))
                    raw_reply = envelope['raw_reply']
                    if (envelope['received_sha256'] != fingerprint or not isinstance(raw_reply, str)
                            or envelope['spawn_receipt']['agent_id'] != session_id
                            or envelope['wait_receipt']['status'][session_id] != dict(completed=raw_reply)
                            or envelope['fork_context'] is not False or envelope['reasoning_effort'] != EFFORT):
                        raise ValueError('orchestrator receipt binding mismatch')
                    session_root = Path(os.environ.get('CODEX_HOME', '/root/autodl-tmp/.codex')) / 'sessions'
                    paths = list(session_root.rglob('*' + session_id + '*.jsonl'))
                    if len(paths) != 1:
                        raise ValueError('subagent original rollout is not uniquely available')
                    raw = paths[0].read_bytes()
                    _write(destination / 'session.jsonl', raw)
                    _write(destination / 'reply.txt', raw_reply.encode())
                    verified = inspect_rollout(raw, session_id=session_id, expected_message=message, raw_reply=raw_reply)
                    status.update(verified, status='delivered', session_id=session_id,
                                  raw_reply_sha256=_sha256(raw_reply.encode()), rollout_sha256=_sha256(raw))
                    reply = ReceivedReply(fingerprint, session_id, raw_reply.encode())
        except (ValueError, KeyError, TypeError, OSError, UnicodeError) as error:
            self.integrity_error = str(error)
            status.update(status='integrity_error', reason=str(error))
        self.wait_seconds += monotonic() - started
        status['cumulative_wait_seconds'] = self.wait_seconds
        _write(destination / 'receipt.json', _json(status))
        return reply


def run(out, fingerprint, *, contract=CONTRACT):
    manifest = verify(out, fingerprint, contract=contract)
    _write(out / 'started.json', _json(dict(manifest_sha256=fingerprint, utc=datetime.now(timezone.utc).isoformat())))
    closed = []
    batch_stop = 'completed'
    for scene_id, record in manifest['scenes'].items():
        verify(out, fingerprint, contract=contract)
        scene = _scenario(record['family'], record['layout_seed'], record['noise_seed'])
        if _sha256(_json(scene.to_dict())) != record['scenario_sha256']:
            raise ValueError('frozen RP2 scenario changed')
        destination = out / scene_id
        destination.mkdir()
        _write(destination / 'started.json', _json(dict(scene_id=scene_id, manifest_sha256=fingerprint)))
        logs, wall = {}, {}
        relay = None
        for arm in ['numerical', 'llm']:
            world = ElectromagneticWorld(scene)
            adapter = {}
            if arm == 'llm':
                relay = SubagentRelay(destination / 'rounds')
                adapter = dict(request_prefix='flight', transport=relay)
            started = monotonic()
            log = fly_repair(world.geometry, scene.aircraft, scene.belief, world.observe, arm=arm,
                             separation_m=20, min_post_s=.5, tool_geometry_limit=record['resources']['geometry'],
                             tool_query_limit=record['resources']['query_points'], tool_alert_settings=ALERT,
                             tool_route_progress=True, **adapter)
            wall[arm] = monotonic() - started
            logs[arm] = json.loads(_json(log))
            _write(destination / f'{arm}.json', _json(logs[arm]))
            print(json.dumps(dict(event='arm_closed', scene=scene_id, arm=arm, stop=log['stop'],
                                  t=log['frames'][-1]['t'])), flush=True)
        _write(destination / 'wall_clock.json', _json(wall))
        report = publish_tool_pair(destination / 'flight', world, logs, synthetic=False)
        result = dict(scene_id=scene_id, paired_delta_tq=report['paired_delta_tq'],
                      model_calls=relay.calls, model_wait_seconds=relay.wait_seconds,
                      integrity_error=relay.integrity_error,
                      arms={key: value['metrics'] for key, value in report['arms'].items()})
        _write(destination / 'result.json', _json(result))
        closed.append(result)
        print(json.dumps(dict(event='pair_published', scene=scene_id,
                              replay=str((destination / 'flight/replay.html').relative_to(ROOT)),
                              delta=report['paired_delta_tq'])), flush=True)
        if relay.integrity_error:
            batch_stop = 'integrity_stop'
            break
    closure = dict(stop=batch_stop, scenes=closed, manifest_sha256=fingerprint,
                   model_calls=sum(row['model_calls'] for row in closed),
                   model_wait_seconds=sum(row['model_wait_seconds'] for row in closed))
    _write(out / 'closed.json', _json(closure))
    print(json.dumps(dict(event='batch_closed', stop=batch_stop, pairs=len(closed))), flush=True)
    return closure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('freeze')
    execute = commands.add_parser('run')
    execute.add_argument('digest')
    args = parser.parse_args()
    if args.command == 'freeze':
        print(freeze(args.out), flush=True)
    else:
        if sys.stdin.isatty():
            import termios

            settings = termios.tcgetattr(sys.stdin)
            settings[3] &= ~(termios.ECHO | termios.ICANON)
            settings[6][termios.VMIN], settings[6][termios.VTIME] = 1, 0
            termios.tcsetattr(sys.stdin, termios.TCSANOW, settings)
        run(args.out, args.digest)


if __name__ == '__main__':
    main()
