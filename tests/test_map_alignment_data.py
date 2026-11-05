import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from uav_em.map_learning.alignment_data import (AlignmentExamples, audit_alignment_dataset,
                                               build_alignment_dataset, current_region_target)
from uav_em.map_learning.data import CollectionSettings, file_hash, generate_dataset
from uav_em.map_learning.model import EncoderSettings
from uav_em.map_learning.scenarios import ScenarioSettings
from uav_em.map_learning.training import train_autoencoder


@pytest.fixture
def alignment_source(tmp_path):
    root = tmp_path / 'source'
    manifest = generate_dataset(root, 3, 30,
                                ScenarioSettings(columns=10, rows=8, building_count=0, max_sources=0,
                                                 horizon_seconds=8, require_goal_reachable_in_time=False),
                                CollectionSettings(step_seconds=2, budget=3, task_stride=1, window_spacing_seconds=2))
    train_autoencoder(root, tmp_path / 'encoder', EncoderSettings(hidden_dim=16, heads=2, layers=1, latent_tokens=4),
                      epochs=1, batch_size=4, device='cpu')
    return root, manifest, tmp_path / 'encoder/best.pt'


def test_current_roi_target_uses_physical_units_and_null_unknowns():
    maps = np.zeros((6, 3, 4), dtype=np.float32)
    maps[1, 1, 1] = 1
    maps[0, 1, 1] = 0.5
    maps[3, 1, 1] = 0.2
    answer = current_region_target(maps, [0, 0, 2, 2])
    assert answer['power_dbm'][1][1] == -75
    assert answer['age_seconds'][1][1] == 7
    assert answer['power_dbm'][0][0] is None
    assert answer['age_seconds'][0][0] is None


def test_alignment_build_preserves_splits_and_reads_no_test_or_truth(alignment_source, tmp_path, monkeypatch):
    root, manifest, checkpoint = alignment_source
    original_open, original_load = Path.open, np.load
    test_files = {(root / entry['path']).resolve() for entry in manifest['episodes'] if entry['split'] == 'test'}

    def visible_open(path, *args, **kwargs):
        assert 'evaluation_only' not in path.parts
        assert path.resolve() not in test_files
        return original_open(path, *args, **kwargs)

    def visible_load(path, *args, **kwargs):
        assert Path(path).resolve() not in test_files
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', visible_open)
    monkeypatch.setattr(np, 'load', visible_load)
    original_hash = file_hash(root / 'manifest.json')
    output = tmp_path / 'alignment'
    result = build_alignment_dataset(root, checkpoint, output)
    for split in ('train', 'validation'):
        assert result['counts'][split] == 2 * sum(entry['temporal_tasks'] for entry in manifest['episodes'] if entry['split'] == split)
    assert result['test_exported'] is False
    assert not (output / 'test.jsonl').exists()
    data = AlignmentExamples(output, 'validation', file_hash(checkpoint))
    example = data.model_input(0)
    assert set(example) == {'tokens', 'valid', 'offset_seconds', 'question'}
    assert example['tokens'].shape == (4, 4, 16)
    assert torch.all(example['tokens'][~example['valid']] == 0)
    assert 'answer' not in example['question']
    assert 'target' not in example['question']
    assert data.answer(0)['interpretation'] == 'onboard_belief_not_truth'
    for entry in result['episodes']:
        with np.load(output / entry['path'], allow_pickle=False) as archive:
            assert set(archive.files) == {'tokens', 'times_seconds'}
    assert file_hash(root / 'manifest.json') == original_hash
    with pytest.raises(FileExistsError):
        build_alignment_dataset(root, checkpoint, output)
    with pytest.raises(ValueError, match='checkpoint mismatch'):
        AlignmentExamples(output, 'train', 'wrong')
    audit = audit_alignment_dataset(output, root, output / 'quality_report.json')
    assert audit['passed'] is True
    assert audit['questions_checked'] == result['counts']
    assert audit['scenes_checked'] == {'train': 1, 'validation': 1}


def test_alignment_reader_rejects_future_insertion_and_wrong_split(alignment_source, tmp_path):
    root, _, checkpoint = alignment_source
    output = tmp_path / 'alignment'
    build_alignment_dataset(root, checkpoint, output)
    data = AlignmentExamples(output, 'validation')
    row = data.rows[0]
    inputs = row['inputs']
    inputs['frame_indices'][0] = inputs['frame_indices'][-1] + 1
    inputs['valid'][0] = True
    inputs['offset_seconds'][0] = 2
    with pytest.raises(ValueError, match='noncausal'):
        data.model_input(0)
    row['split'] = 'train'
    with pytest.raises(ValueError, match='split'):
        data.model_input(0)


def test_alignment_integrity_rejects_edited_questions_and_tokens(alignment_source, tmp_path):
    root, _, checkpoint = alignment_source
    output = tmp_path / 'alignment'
    build_alignment_dataset(root, checkpoint, output)
    data = AlignmentExamples(output, 'validation')
    token_file = output / data.rows[0]['inputs']['token_path']
    with token_file.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        data.model_input(0)
    with (output / 'validation.jsonl').open('a', encoding='utf-8') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='question data hash'):
        AlignmentExamples(output, 'validation')


def test_audit_recomputes_answers_even_if_question_file_hash_is_updated(alignment_source, tmp_path):
    root, _, checkpoint = alignment_source
    output = tmp_path / 'alignment'
    build_alignment_dataset(root, checkpoint, output)
    question_path = output / 'validation.jsonl'
    rows = [json.loads(line) for line in question_path.read_text(encoding='utf-8').splitlines()]
    rows[0]['answer']['power_dbm'][0][0] = 999
    question_path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['qa_sha256']['validation'] = file_hash(question_path)
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='answer differs'):
        audit_alignment_dataset(output, root, output / 'quality_report.json')
    assert not (output / 'quality_report.json').exists()
