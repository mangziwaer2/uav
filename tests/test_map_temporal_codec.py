from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from uav_em.map_learning.data import CollectionSettings, file_hash, generate_dataset, temporal_tasks
from uav_em.map_learning.model import EncoderSettings, MapAutoencoder, PATCH_ARCHITECTURE
from uav_em.map_learning.scenarios import ScenarioSettings
from uav_em.map_learning.temporal import audit_temporal_codec, decode_frames, pair_sums, summarize, validate_task
from uav_em.map_learning.training import train_autoencoder


def example_pair():
    older = np.zeros((6, 3, 4), dtype=np.float32)
    newer = older.copy()
    older[1, 1, :3] = 1
    newer[1, 1, 1:] = 1
    older[0] = older[1] * 0.4
    newer[0] = newer[1] * 0.5
    older[3] = older[1] * 0.1
    newer[3] = newer[1] * 0.3
    decoded = [value.copy() for value in (older, newer)]
    for value in decoded:
        value[1] = np.where(value[1] > 0.5, 8.0, -8.0)
    return older, newer, *decoded


def test_identity_codec_preserves_support_changes_power_direction_and_age():
    result = summarize(pair_sums(*example_pair()))
    assert result['new_support']['recall'] == 1
    assert result['lost_support']['iou'] == 1
    assert result['power_change']['precision'] == 1
    assert result['common_support_retention'] == 1
    assert result['changed_direction_recall'] == 1
    assert result['power_delta_mae_db'] == 0
    assert result['age_delta_mae_seconds'] == 0
    assert result['changed_cells'] == 2


def test_dropped_support_is_not_hidden_by_regression_mask():
    older, newer, before, after = example_pair()
    after[1] = -8
    after[0] = before[0]
    result = summarize(pair_sums(older, newer, before, after))
    assert result['common_cells'] == 2
    assert result['common_support_retention'] == 0
    assert result['changed_direction_recall'] == 0
    assert result['new_support']['recall'] == 0
    assert result['power_delta_mae_db'] > 8


def test_unknown_placeholders_cannot_create_reference_power_changes():
    older, newer, before, after = example_pair()
    older[0, 0] = 100
    after[0, 0] = -100
    result = summarize(pair_sums(older, newer, before, after))
    assert result['changed_cells'] == 2
    assert result['power_delta_mae_db'] == 0
    empty = np.zeros((6, 2, 2), dtype=np.float32)
    result = summarize(pair_sums(empty, empty, empty, empty))
    assert result['new_support']['iou'] is None
    assert result['power_delta_mae_db'] is None
    assert result['changed_direction_recall'] is None


@pytest.mark.parametrize('threshold', [0, -1, float('inf'), float('nan')])
def test_bad_change_threshold_rejected(threshold):
    with pytest.raises(ValueError, match='change_db'):
        pair_sums(*example_pair(), change_db=threshold)


@pytest.mark.parametrize('corruption', ['future', 'labels', 'roi'])
def test_modified_causal_task_or_label_rejected(corruption):
    older, newer, _, _ = example_pair()
    arrays = {'maps': np.stack((older, newer)), 'times': np.array([0.0, 2.0]),
              'positions': np.zeros((2, 2)), 'resolution_m': np.array(1)}
    collection = CollectionSettings(task_stride=1, window_spacing_seconds=2, roi_size=3)
    task = next(temporal_tasks(arrays, collection, 0))
    validate_task(task, arrays['maps'], arrays['times'], collection)
    if corruption == 'future':
        task['window']['indices'][-1] = 2
    elif corruption == 'labels':
        task['target']['interpretation'] = 'source_motion'
    else:
        task['roi_grid_xyxy'][2] = 100
    with pytest.raises(ValueError):
        validate_task(task, arrays['maps'], arrays['times'], collection)


def test_frame_reconstruction_cannot_depend_on_future_frames():
    torch.manual_seed(3)
    model = MapAutoencoder(EncoderSettings(hidden_dim=16, heads=2, layers=1, latent_tokens=4,
                                           architecture=PATCH_ARCHITECTURE)).eval()
    maps = np.random.default_rng(3).random((3, 6, 5, 7), dtype=np.float32)
    baseline = decode_frames(model, maps[:2], torch.device('cpu'), 2)
    maps[-1] = 20
    with_future = decode_frames(model, maps, torch.device('cpu'), 3)
    np.testing.assert_allclose(baseline, with_future[:2], atol=2e-6)


@pytest.fixture
def temporal_dataset(tmp_path):
    root = tmp_path / 'dataset'
    manifest = generate_dataset(root, 3, 30,
                                ScenarioSettings(columns=10, rows=8, building_count=0, max_sources=0,
                                                 horizon_seconds=6, require_goal_reachable_in_time=False),
                                CollectionSettings(step_seconds=2, budget=3, task_stride=1, window_spacing_seconds=2))
    train_autoencoder(root, tmp_path / 'train', EncoderSettings(hidden_dim=16, heads=2, layers=1, latent_tokens=4),
                      epochs=1, batch_size=4, frame_stride=1, device='cpu')
    return root, manifest, tmp_path / 'train/best.pt'


def test_temporal_audit_reads_no_hidden_truth_and_preserves_input(temporal_dataset, tmp_path, monkeypatch):
    root, manifest, checkpoint = temporal_dataset
    original_read = Path.read_text
    original_load = np.load

    def visible_read(path, *args, **kwargs):
        assert 'evaluation_only' not in path.parts
        return original_read(path, *args, **kwargs)

    def visible_load(path, *args, **kwargs):
        assert 'evaluation_only' not in Path(path).parts
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', visible_read)
    monkeypatch.setattr(np, 'load', visible_load)
    before = file_hash(root / 'manifest.json')
    report = audit_temporal_codec(root, checkpoint, tmp_path / 'audit')
    expected = sum(item['temporal_tasks'] for item in manifest['episodes'] if item['split'] == 'validation')
    assert report['overall']['tasks'] == expected
    assert report['split'] == 'validation'
    assert report['uses_evaluation_only'] is False
    assert set(report['by_policy']) == {'goal', 'coverage', 'revisit'}
    assert file_hash(root / 'manifest.json') == before
    assert (tmp_path / 'audit/task_metrics.csv').is_file()
    assert (tmp_path / 'audit/README.md').is_file()
    assert list((tmp_path / 'audit').glob('*/comparison.png'))
    with pytest.raises(FileExistsError):
        audit_temporal_codec(root, checkpoint, tmp_path / 'audit')


def test_task_hash_corruption_rejected_before_output(temporal_dataset, tmp_path):
    root, _, checkpoint = temporal_dataset
    with (root / 'temporal_tasks.jsonl').open('a', encoding='utf-8') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='task hash mismatch'):
        audit_temporal_codec(root, checkpoint, tmp_path / 'audit')
    assert not (tmp_path / 'audit').exists()


def test_cli_help_without_root_working_directory():
    entry = Path(__file__).resolve().parents[1] / 'tools/audit_map_temporal_codec.py'
    result = subprocess.run([sys.executable, str(entry), '--help'], cwd=entry.parent,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert b'--change-db' in result.stdout
