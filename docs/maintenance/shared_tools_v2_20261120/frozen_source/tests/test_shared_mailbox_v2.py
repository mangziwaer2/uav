"""Atomic publication and bounded receipt retries, with no real model service."""
import json

import pytest

import uav_em.event_demo as module


def fixture(tmp_path):
    for name in ('requests', 'responses'):
        (tmp_path/name).mkdir()
    return dict(request_id='bound', phase='pilot', time_seconds=3.)


def test_save_publishes_complete_json_once_and_never_overwrites(tmp_path, monkeypatch):
    target = tmp_path/'receipt.json'
    link = module.os.link
    observations = []
    def inspect(source, destination):
        observations.append((target.exists(), json.loads(source.read_text())))
        link(source, destination)
    monkeypatch.setattr(module.os, 'link', inspect)
    module.save(target, dict(agent='synthetic', payload=list(range(100))))
    assert observations == [(False, dict(agent='synthetic', payload=list(range(100))))]
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        module.save(target, dict(agent='replacement'))
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_invalid_json_value_does_not_publish_partial_file(tmp_path):
    target = tmp_path/'receipt.json'
    with pytest.raises(ValueError):
        module.save(target, dict(value=float('nan')))
    assert not target.exists() and list(tmp_path.iterdir()) == []


def test_partial_receipt_is_retried_until_a_bound_complete_reply(tmp_path):
    request = fixture(tmp_path)
    receipt = tmp_path/'responses/00.started.json'
    receipt.write_text('{"agent_id":')
    clock = [100.]
    sleeps = [0]
    def sleep(_):
        clock[0] += .15
        sleeps[0] += 1
        if sleeps[0] == 2:
            receipt.write_text(json.dumps(dict(agent_id='synthetic', request_id='bound',
                                               started_monotonic=100.15)))
            (tmp_path/'responses/00.raw.txt').write_text('{"action":"keep"}')
            (tmp_path/'responses/00.ready').touch()
    env = module.FileAdvisor(tmp_path, 'synthetic', clock=lambda: clock[0], sleeper=sleep).ask('inference', request)
    assert env['started'] and env['actual_llm_request'] and env['timely']
    assert env['raw'] == '{"action":"keep"}' and env['error'] is None
    assert env['receipt_json_retries'] == 2
    assert env['wall_seconds'] == pytest.approx(.3)


def test_permanently_incomplete_receipt_stops_at_dispatch_deadline(tmp_path):
    request = fixture(tmp_path)
    (tmp_path/'responses/00.started.json').write_text('{"agent_id":')
    clock = [0.]
    def sleep(_):
        clock[0] += 15.
    env = module.FileAdvisor(tmp_path, 'synthetic', clock=lambda: clock[0], sleeper=sleep).ask('inference', request)
    assert env['error'] == 'undispatched timeout'
    assert not env['started'] and not env['actual_llm_request'] and not env['timely']
    assert env['receipt_json_retries'] == 5
    assert env['wall_seconds'] == 75.


@pytest.mark.parametrize('change', [dict(agent_id='forged'), dict(request_id='other'),
                                 dict(started_monotonic=-1.), dict(started_monotonic=True),
                                 dict(started_monotonic=1.)])
def test_complete_but_invalid_receipt_is_rejected_without_retry(tmp_path, change):
    request = fixture(tmp_path)
    receipt = dict(agent_id='synthetic', request_id='bound', started_monotonic=0.)
    receipt.update(change)
    module.save(tmp_path/'responses/00.started.json', receipt)
    env = module.FileAdvisor(tmp_path, 'synthetic', clock=lambda: 0.,
                             sleeper=lambda _: pytest.fail('invalid binding retried')).ask('inference', request)
    assert env['error'] == 'invalid start receipt binding/clock'
    assert not env['started'] and env['receipt_json_retries'] == 0


def test_reply_after_response_deadline_is_not_consumed(tmp_path):
    request = fixture(tmp_path)
    clock = [0.]
    module.save(tmp_path/'responses/00.started.json',
                dict(agent_id='synthetic', request_id='bound', started_monotonic=0.))
    def sleep(_):
        clock[0] = 181.
        (tmp_path/'responses/00.raw.txt').write_text('{"late":true}')
        (tmp_path/'responses/00.ready').touch()
    env = module.FileAdvisor(tmp_path, 'synthetic', clock=lambda: clock[0], sleeper=sleep).ask('inference', request)
    assert env['error'] == 'response timeout'
    assert env['started'] and env['actual_llm_request'] and not env['timely']
    assert env['raw'] is None
