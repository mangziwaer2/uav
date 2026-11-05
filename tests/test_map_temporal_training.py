from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from uav_em.map_learning.data import CollectionSettings, file_hash, generate_dataset
from uav_em.map_learning.model import EncoderSettings, PATCH_ARCHITECTURE
from uav_em.map_learning.scenarios import ScenarioSettings
from uav_em.map_learning.temporal_training import (MapPairs, TemporalSettings, spatial_guard,
                                                   temporal_loss, train_temporal_autoencoder)
from uav_em.map_learning.training import load_autoencoder, train_autoencoder


def loss_example():
    target = torch.zeros(1, 2, 6, 4, 4)
    target[:, :, 1, 1:3, 1:3] = 1
    target[:, :, 0] = target[:, :, 1] * 0.5
    target[:, 1, 0, 1, 1] += 0.1
    target[:, 1, 3] = target[:, 1, 1] * 0.2
    predicted = target.clone()
    predicted[:, :, 1] = torch.where(target[:, :, 1] > 0.5, 10, -10)
    return target, predicted, torch.ones(1, 4, 4, dtype=torch.bool)


def test_temporal_loss_preserves_real_change_instead_of_forcing_smoothness():
    target, predicted, region = loss_example()
    correct, good = temporal_loss(predicted, target, region, TemporalSettings())
    predicted[:, 1, 0] = predicted[:, 0, 0]
    wrong, bad = temporal_loss(predicted, target, region, TemporalSettings())
    assert good['power_delta'] == 0
    assert good['age_delta'] == 0
    assert bad['power_delta'] > good['power_delta']
    assert wrong > correct


def test_temporal_loss_ignores_unknown_and_outside_roi_delta_values():
    target, predicted, region = loss_example()
    reference = temporal_loss(predicted, target, region, TemporalSettings())[1]
    predicted[:, 1, 0, 0] = 100
    predicted[:, 1, 3, 0] = 100
    result = temporal_loss(predicted, target, region, TemporalSettings())[1]
    assert result['power_delta'] == reference['power_delta']
    assert result['age_delta'] == reference['age_delta']
    region[:, 1, 1] = False
    predicted[:, 1, 0, 1, 1] = -100
    result = temporal_loss(predicted, target, region, TemporalSettings())[1]
    assert result['power_delta'] == 0


def test_temporal_loss_is_finite_for_empty_support_and_has_no_unknown_power_gradient():
    target = torch.zeros(2, 2, 6, 4, 4)
    predicted = torch.zeros_like(target, requires_grad=True)
    loss, parts = temporal_loss(predicted, target, torch.ones(2, 4, 4, dtype=torch.bool), TemporalSettings())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    assert torch.all(predicted.grad[:, :, 0] == 0)


@pytest.mark.parametrize('changes', [{'epochs': 0}, {'patience': 0}, {'seed': -1}, {'augment': 'yes'},
                                     {'change_db': float('nan')}, {'power_weight': -1}])
def test_temporal_settings_reject_invalid_values(changes):
    with pytest.raises(ValueError):
        replace(TemporalSettings(), **changes).validate()


@pytest.fixture
def paired_dataset(tmp_path):
    root = tmp_path / 'data'
    manifest = generate_dataset(root, 3, 30,
                                ScenarioSettings(columns=10, rows=8, building_count=0, max_sources=0,
                                                 horizon_seconds=8, require_goal_reachable_in_time=False),
                                CollectionSettings(step_seconds=2, budget=3, task_stride=1, window_spacing_seconds=2))
    return root, manifest


def test_pairs_are_synchronized_train_only_augmented_and_weights_are_balanced(paired_dataset, monkeypatch):
    root, _ = paired_dataset
    original = MapPairs(root, 'train')
    augmented = MapPairs(root, 'train', augment=True)
    target, region = original[0]
    monkeypatch.setattr(torch, 'rand', lambda *args: torch.tensor(0.0))
    flipped, flipped_region = augmented[0]
    assert torch.equal(flipped, target.flip((-2, -1)))
    assert torch.equal(flipped_region, region.flip((-2, -1)))
    again, _ = original[0]
    assert torch.equal(again, target)
    for stratum in set(original.strata):
        assert sum(weight for name, weight in zip(original.strata, original.weights) if name == stratum) == pytest.approx(1)
    with pytest.raises(ValueError, match='training-only'):
        MapPairs(root, 'validation', augment=True)
    with pytest.raises(ValueError):
        MapPairs(root, 'test')


def test_pair_loader_rejects_data_hash_mismatch(paired_dataset):
    root, _ = paired_dataset
    dataset = MapPairs(root, 'train')
    entry = dataset.entries[dataset.tasks[0]['episode_id']]
    with (root / entry['path']).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='episode hash'):
        dataset[0]


def test_guard_rejects_spatial_regression():
    baseline = {'support_iou': 0.98, 'supported_power_mae_db': 2.3, 'high_power_mae_db': 2.2}
    assert spatial_guard(baseline, baseline)
    assert not spatial_guard({**baseline, 'support_iou': 0.9}, baseline)
    assert not spatial_guard({**baseline, 'high_power_mae_db': 3.0}, baseline)


def test_temporal_training_uses_no_test_or_truth_and_keeps_baseline_candidate(paired_dataset, tmp_path, monkeypatch):
    root, manifest = paired_dataset
    settings = EncoderSettings(hidden_dim=16, heads=2, layers=1, latent_tokens=4, architecture=PATCH_ARCHITECTURE)
    train_autoencoder(root, tmp_path / 'parent', settings, epochs=1, batch_size=4, device='cpu')
    original_load, original_open = np.load, Path.open
    test_files = {(root / item['path']).resolve() for item in manifest['episodes'] if item['split'] == 'test'}

    def no_test(path, *args, **kwargs):
        assert Path(path).resolve() not in test_files
        return original_load(path, *args, **kwargs)

    def no_truth(path, *args, **kwargs):
        assert 'evaluation_only' not in path.parts
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', no_test)
    monkeypatch.setattr(Path, 'open', no_truth)
    monkeypatch.setattr('uav_em.map_learning.temporal_training.spatial_guard', lambda *args: False)
    original_hash = file_hash(root / 'manifest.json')
    output = tmp_path / 'temporal'
    result = train_temporal_autoencoder(root, tmp_path / 'parent/best.pt', output,
                                       TemporalSettings(epochs=2, patience=1, batch_size=4), 'cpu')
    assert result['best_epoch'] == 0
    assert result['epochs_completed'] == 1
    assert result['test_evaluated'] is False
    model, saved = load_autoencoder(output / 'best.pt')
    assert saved['parent_epoch'] == 1
    assert saved['training_stage'] == 'temporal_regularized_spatial_autoencoder'
    parent, _ = load_autoencoder(tmp_path / 'parent/best.pt')
    assert all(torch.equal(value, parent.state_dict()[key]) for key, value in model.state_dict().items())
    config = json.loads((output / 'training_config.json').read_text(encoding='utf-8'))
    assert config['uses_evaluation_only'] is False
    assert config['parent_checkpoint_sha256'] == file_hash(tmp_path / 'parent/best.pt')
    assert file_hash(root / 'manifest.json') == original_hash
    with pytest.raises(FileExistsError):
        train_temporal_autoencoder(root, tmp_path / 'parent/best.pt', output, device='cpu')


def test_temporal_cli_help_is_lazy_and_not_cwd_dependent():
    entry = Path(__file__).resolve().parents[1] / 'tools/train_map_temporal_autoencoder.py'
    result = subprocess.run([sys.executable, str(entry), '--help'], cwd=entry.parent,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert b'--checkpoint' in result.stdout
