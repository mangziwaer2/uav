import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('torch')

from uav_em.map_learning.data import file_hash, json_write
from uav_em.map_learning.sensor_evidence import (SensorAlignmentExamples, build_sensor_evidence,
                                                compact_evidence, visible_observation)


def observation(time, scan=True):
    position = [float(time), 0.0]
    return {'time_seconds': time, 'position_m': position, 'interference_dbm': -90.123456789,
            'hidden_source': [900, 900],
            'directional': {'time_seconds': time, 'position_m': position,
                            'peaks_deg_dbm': [[359.0, -85.0]] if time == 2 else [],
                            'beamwidth_deg': 45., 'bearing_error_std_deg': 3.,
                            'calibration_correlation_seconds': 5., 'minimum_peak_separation_deg': 45.,
                            'detection_threshold_dbm': -94., 'source_id': 'do_not_copy'} if scan else None}


@pytest.fixture
def evidence_data(tmp_path):
    source, alignment = tmp_path / 'source', tmp_path / 'alignment'
    (source / 'episodes').mkdir(parents=True)
    (alignment / 'tokens').mkdir(parents=True)
    source_entries, token_entries = [], []
    for split in ('train', 'validation', 'test'):
        identity = f'{split}_scene_goal_000'
        relative = f'episodes/{identity}.npz'
        public = {'split': split, 'scenario_id': split, 'observations': [observation(time) for time in (0, 2, 4)]}
        json_write((source / relative).with_suffix('.json'), public)
        source_entries.append({'id': identity, 'path': relative, 'split': split, 'scenario_id': split,
                               'metadata_sha256': file_hash((source / relative).with_suffix('.json'))})
        if split == 'test':
            continue
        token_path = f'tokens/{identity}.npz'
        np.savez(alignment / token_path, tokens=np.zeros((3, 4, 16), dtype=np.float32), times_seconds=[0., 2., 4.])
        token_entries.append({'episode_id': identity, 'scenario_id': split, 'split': split,
                              'path': token_path, 'sha256': file_hash(alignment / token_path), 'frames': 3})
        inputs = {'token_path': token_path, 'frame_indices': [-1, 0, 1], 'valid': [False, True, True],
                  'offset_seconds': [None, -2., 0.], 'positions_m': [None, [0., 0.], [2., 0.]],
                  'roi_grid_xyxy': [0, 0, 1, 1], 'map_grid': {'rows': 2, 'columns': 2}}
        row = {'id': identity + '_1', 'episode_id': identity, 'scenario_id': split, 'split': split,
               'kind': 'current_region', 'inputs': inputs, 'answer': {'sentinel_answer': 123}}
        (alignment / f'{split}.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    json_write(source / 'manifest.json', {'episodes': source_entries})
    json_write(alignment / 'manifest.json',
               {'schema_version': 1, 'stage': 'alignment_data_only_no_llm_training',
                'source_manifest_sha256': file_hash(source / 'manifest.json'), 'checkpoint_sha256': 'frozen',
                'encoder_settings': {'hidden_dim': 16, 'latent_tokens': 4},
                'episodes': token_entries, 'counts': {'train': 1, 'validation': 1},
                'qa_sha256': {split: file_hash(alignment / f'{split}.jsonl') for split in ('train', 'validation')}})
    return source, alignment


def test_sensor_build_is_separate_and_never_reads_test_or_truth(evidence_data, tmp_path, monkeypatch):
    source, alignment = evidence_data
    original_hash = file_hash(alignment / 'manifest.json')
    original_open = Path.open

    def guard(path, *args, **kwargs):
        assert 'evaluation_only' not in path.parts
        assert not path.name.startswith('test_scene')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guard)
    output = tmp_path / 'evidence'
    report = build_sensor_evidence(alignment, source, output)
    assert report['counts'] == {'observations': 6, 'scans': 6, 'detected_scans': 2}
    assert file_hash(alignment / 'manifest.json') == original_hash
    data = SensorAlignmentExamples(alignment, output, 'train')
    evidence = data.evidence(0)
    assert [record['time_seconds'] for record in evidence['observations']] == [0., 2.]
    assert evidence['observations'][-1]['directional']['peaks_deg_dbm'] == [[359., -85.]]
    inputs = data.model_input(0)
    assert set(inputs) == {'tokens', 'valid', 'offset_seconds', 'question'}
    assert 'hidden_source' not in inputs['question']
    assert 'source_id' not in inputs['question']
    assert 'sentinel_answer' not in inputs['question']
    assert data.answer(0) == {'sentinel_answer': 123}
    assert json.loads((output / 'quality_report.json').read_text())['passed']
    with pytest.raises(FileExistsError):
        build_sensor_evidence(alignment, source, output)


def test_sensor_window_ignores_future_and_preserves_missing_vs_empty(evidence_data, tmp_path):
    source, alignment = evidence_data
    output = tmp_path / 'evidence'
    build_sensor_evidence(alignment, source, output)
    data = SensorAlignmentExamples(alignment, output, 'train', history_seconds=1)
    evidence = data.evidence(0)
    assert evidence['older_observations_excluded'] == 1
    assert len(evidence['observations']) == 1
    before = data.question(0)
    data.cache[data.rows[0]['episode_id']][2]['directional']['peaks_deg_dbm'] = [[42., 999.]]
    assert data.question(0) == before
    evidence['observations'] = [visible_observation(observation(0, False)), visible_observation(observation(2))]
    compact = compact_evidence(evidence)
    assert compact['observations'][0][-2:] == [None, None]
    assert compact['observations'][1][-1] == [[359., -85.]]
    assert compact['observations'][0][3] == -90.123456789


def test_sensor_reader_rejects_wrong_versions_hashes_and_frame_alignment(evidence_data, tmp_path):
    source, alignment = evidence_data
    output = tmp_path / 'evidence'
    build_sensor_evidence(alignment, source, output)
    data = SensorAlignmentExamples(alignment, output, 'train')
    data.rows[0]['inputs']['positions_m'][-1] = [999., 0.]
    with pytest.raises(ValueError, match='map history'):
        data.evidence(0)
    data = SensorAlignmentExamples(alignment, output, 'train')
    entry = next(iter(data.entries.values()))
    with (output / entry['path']).open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='hash'):
        data.evidence(0)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['alignment_manifest_sha256'] = 'wrong'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='version mismatch'):
        SensorAlignmentExamples(alignment, output, 'train')


def test_sensor_tool_help_is_lazy():
    script = Path(__file__).resolve().parents[1] / 'tools/build_map_sensor_evidence.py'
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=script.parent, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert b'--source' in result.stdout
