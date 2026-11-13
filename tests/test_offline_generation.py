"""生成请求身份、非法提交与执行产物边界；不重复搜索器实现测试。"""
import json
from pathlib import Path
import pytest
from uav_em.offline_process import reference_program


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'tools'))
    import run_offline_generation
    return run_offline_generation


def response():
    return {'request_id': 'draft_01', 'program': reference_program(), 'rationale': 'Use bounded feedback branches.'}


@pytest.mark.parametrize('change', [
    lambda r: r.update(request_id='draft_02'),
    lambda r: r.update(code='import os'),
    lambda r: r['program']['nodes'][0]['next'].update(default=0),
    lambda r: r['program']['nodes'][0]['action'].update(operator='probe'),
    lambda r: r.update(rationale=''),
])
def test_response_identity_and_schema_cannot_expand_execution(bridge, change):
    r = response(); change(r)
    with pytest.raises(ValueError): bridge.parse_response(json.dumps(r), 'draft_01')


def test_raw_duplicate_keys_rejected_and_valid_program_not_rewritten(bridge):
    raw = json.dumps(response())
    assert bridge.parse_response(raw, 'draft_01') == response()['program']
    with pytest.raises(ValueError): bridge.parse_response('{"request_id":"draft_01",' + raw[1:], 'draft_01')


def test_invalid_submission_occupies_slot_and_preserves_exact_raw(bridge, monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, 'check', lambda out: {})
    public = tmp_path / 'public/draft_01'; public.mkdir(parents=True)
    (tmp_path / 'submissions').mkdir()
    (public / 'request_manifest.json').write_text('{}')
    raw = '{"invalid":true}\n'
    (public / 'response.json').write_text(raw)
    status = bridge.submit(tmp_path, 'draft_01', 'actual-agent-id')
    assert not status['accepted'] and status['actual_offline_design_sessions'] == 1
    assert (tmp_path / 'submissions/draft_01/raw_response.json').read_text() == raw
    assert not (tmp_path / 'submissions/draft_01/program.json').exists()
    with pytest.raises(FileExistsError): bridge.submit(tmp_path, 'draft_01', 'another-id')


def test_unknown_submission_cannot_create_fifth_slot(bridge, monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, 'check', lambda out: {})
    with pytest.raises(ValueError): bridge.submit(tmp_path, 'draft_05', 'actual-agent-id')
    assert not (tmp_path / 'submissions').exists()


def test_tampered_request_is_detected_before_parse(bridge, monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, 'check', lambda out: {})
    public = tmp_path / 'public/draft_01'; public.mkdir(parents=True)
    (tmp_path / 'submissions').mkdir()
    (public / 'request.json').write_text('changed')
    (public / 'request_manifest.json').write_text('{"request.json":"incorrect-hash"}')
    with pytest.raises(AssertionError): bridge.submit(tmp_path, 'draft_01', 'actual-agent-id')
