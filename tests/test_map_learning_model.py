import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from uav_em.map_learning.data import generate_dataset, CollectionSettings
from uav_em.map_learning.scenarios import ScenarioSettings
from uav_em.map_learning.model import PATCH_ARCHITECTURE, EncoderSettings, MapAutoencoder, reconstruction_loss, shape_binary
from uav_em.map_learning.training import MapFrames, train_autoencoder, load_autoencoder, export_window


@pytest.fixture(autouse=True, scope='module')
def limit_test_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('architecture', ['spatial_anchor_local_decode_v1', PATCH_ARCHITECTURE])
def test_encoder_bottleneck_shapes_variable_size_and_gradients(architecture):
    model = MapAutoencoder(EncoderSettings(hidden_dim=16, heads=2, layers=1, latent_tokens=4, architecture=architecture))
    maps = torch.rand(2, 6, 11, 13)
    maps[:, 1] = (maps[:, 1] > 0.8).float()
    maps[:, 4] = (maps[:, 4] > 0.7).float()
    tokens = model.encode(maps)
    assert tokens.shape == (2, 4, 16)
    predicted = model.decode(tokens, 11, 13)
    assert predicted.shape == maps.shape
    loss, _ = reconstruction_loss(predicted, maps)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.queries.grad.abs().sum() > 0
    assert model.patch.weight.grad.abs().sum() > 0
    assert model(torch.rand(1, 6, 8, 10)).shape == (1, 6, 8, 10)


@pytest.mark.parametrize('objective', ['balanced_v1', 'boundary_v2'])
def test_unknown_power_is_not_a_regression_label(objective):
    predicted = torch.zeros(1, 6, 4, 4, requires_grad=True)
    target = torch.zeros_like(predicted)
    loss, _ = reconstruction_loss(predicted, target, objective)
    loss.backward()
    assert torch.all(predicted.grad[:, 0] == 0)
    assert torch.all(predicted.grad[:, 3] == 0)
    assert torch.isfinite(loss)


def test_autoencoder_learns_two_distinct_supported_regions():
    torch.manual_seed(10)
    model = MapAutoencoder(EncoderSettings(hidden_dim=32, heads=2, layers=1, latent_tokens=4, patch_size=2))
    maps = torch.zeros(2, 6, 8, 8)
    maps[0, 1, 1:7, :3] = 1
    maps[1, 1, 1:7, 5:] = 1
    maps[:, 0] = maps[:, 1] * 0.4
    maps[:, 2] = 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    initial = float(reconstruction_loss(model(maps), maps)[0].item())
    for _ in range(100):
        optimizer.zero_grad()
        loss, _ = reconstruction_loss(model(maps), maps)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predicted = model(maps)
        final = float(reconstruction_loss(predicted, maps)[0].item())
        intersection = ((predicted[:, 1] > 0) & (maps[:, 1] > 0)).sum()
        union = ((predicted[:, 1] > 0) | (maps[:, 1] > 0)).sum()
    assert final < initial * 0.6
    assert float(intersection / union) > 0.5


@pytest.mark.parametrize('changes', [{'hidden_dim': 15}, {'latent_tokens': 0}, {'patch_size': -1}, {'architecture': 'unknown'}])
def test_bad_encoder_settings_rejected(changes):
    with pytest.raises(ValueError):
        MapAutoencoder(EncoderSettings(**changes))


def test_boundary_loss_penalizes_false_support_and_is_finite_without_positives():
    target = torch.zeros(1, 8, 8)
    target[:, 2:6, 2:6] = 1
    correct = torch.where(target > 0.5, 6.0, -6.0)
    broad = correct.clone()
    broad[:, 1:7, 1:7] = 6
    assert shape_binary(correct, target) < shape_binary(broad, target)
    empty = torch.full_like(target, -4, requires_grad=True)
    loss = shape_binary(empty, torch.zeros_like(target))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(empty.grad).all()


def test_training_checkpoint_and_causal_window_export(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset'
    manifest = generate_dataset(dataset, 3, 30,
                                ScenarioSettings(columns=10, rows=8, building_count=0, max_sources=0, horizon_seconds=6,
                                                 require_goal_reachable_in_time=False),
                                CollectionSettings(step_seconds=2, budget=3, task_stride=1, window_spacing_seconds=2))
    original_read = type(dataset).read_text

    def no_oracle_read(path, *args, **kwargs):
        assert 'evaluation_only' not in path.parts
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(type(dataset), 'read_text', no_oracle_read)
    settings = EncoderSettings(hidden_dim=16, heads=2, layers=1, latent_tokens=4)
    output = tmp_path / 'training'
    metrics = train_autoencoder(dataset, output, settings, epochs=1, batch_size=3, frame_stride=1, device='cpu')
    assert metrics['best_epoch'] == 1
    assert np.isfinite(metrics['test']['loss'])
    model, checkpoint = load_autoencoder(output / 'best.pt')
    assert checkpoint['channels'] == manifest['channels']
    example = MapFrames(dataset, 'test')[0].unsqueeze(0)
    assert model.encode(example).shape == (1, 4, 16)
    episode = dataset / manifest['episodes'][0]['path']
    destination = export_window(output / 'best.pt', episode, tmp_path / 'inspect', anchor=1,
                                frames=4, spacing_seconds=2)
    metadata = json.loads((destination / 'window.json').read_text(encoding='utf-8'))
    assert metadata['window']['valid'] == [False, False, True, True]
    assert all(item['time_seconds'] <= metadata['anchor_time_seconds'] for item in metadata['recent_observations'])
    assert metadata['token_shape'] == [4, 4, 16]
    assert (destination / 'reconstruction.png').is_file()
    with np.load(destination / 'window_tokens.npz', allow_pickle=False) as window:
        assert np.all(window['tokens'][:2] == 0)
        assert np.all(window['reconstructed_window'][:2] == 0)
        assert np.array_equal(window['reconstructed_window'][-1], window['reconstruction'])
        assert window['reconstructed_window'].shape[:2] == (4, 6)
        assert window['tokens'].shape == (4, 4, 16)
    assert metadata['window_schema_version'] == 2
    assert metadata['temporal_encoding'] == 'independent_spatial_frames_with_causal_timestamps'
    with pytest.raises(FileExistsError):
        train_autoencoder(dataset, output, settings, epochs=1, device='cpu')
    value = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    value['episodes'][0]['split'] = 'incorrect'
    (dataset / 'manifest.json').write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='leakage'):
        MapFrames(dataset, 'train')


def test_formal_training_rejects_missing_partial_and_stale_audits(tmp_path):
    from uav_em.map_learning.quality import audit_dataset

    dataset = tmp_path / 'dataset'
    generate_dataset(dataset, 3, 30,
                     ScenarioSettings(columns=10, rows=8, building_count=0, max_sources=0, horizon_seconds=6,
                                      require_goal_reachable_in_time=False),
                     CollectionSettings(budget=3, task_stride=1, window_spacing_seconds=2))
    output = tmp_path / 'training'
    with pytest.raises(FileNotFoundError):
        train_autoencoder(dataset, output, epochs=1, require_quality=True, device='cpu')
    report = audit_dataset(dataset, dataset / 'quality_report.json')
    assert report['passed']
    with pytest.raises(ValueError, match='full coverage audit'):
        train_autoencoder(dataset, output, epochs=1, require_quality=True, device='cpu')
    report['coverage_required'] = True
    report['manifest_sha256'] = 'stale'
    (dataset / 'quality_report.json').write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='manifest changed'):
        train_autoencoder(dataset, output, epochs=1, require_quality=True, device='cpu')
    assert not output.exists()


def test_training_loader_rechecks_episode_hash_after_audit(tmp_path):
    dataset = tmp_path / 'dataset'
    manifest = generate_dataset(dataset, 3, 30,
                                ScenarioSettings(columns=10, rows=8, building_count=0, max_sources=0,
                                                 horizon_seconds=6, require_goal_reachable_in_time=False),
                                CollectionSettings(budget=3))
    frames = MapFrames(dataset, 'train')
    relative = next(entry['path'] for entry in manifest['episodes'] if entry['split'] == 'train')
    with (dataset / relative).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        frames[0]


def test_patch_checkpoint_early_stopping_and_training_only_augmentation(tmp_path, monkeypatch):
    from uav_em.map_learning import training

    dataset = tmp_path / 'data'
    generate_dataset(dataset, 3, 30,
                     ScenarioSettings(columns=10, rows=8, building_count=0, max_sources=0,
                                      horizon_seconds=6, require_goal_reachable_in_time=False),
                     CollectionSettings(budget=3))
    original = MapFrames(dataset, 'train')[0]
    monkeypatch.setattr(torch, 'rand', lambda *args, **kwargs: torch.tensor(0.0))
    augmented = MapFrames(dataset, 'train', augment=True)
    assert torch.equal(augmented[0], original.flip((1, 2)))
    assert torch.equal(MapFrames(dataset, 'train')[0], original)
    with pytest.raises(ValueError, match='training-only'):
        MapFrames(dataset, 'validation', augment=True)
    real_evaluate = training.evaluate
    evaluations = []

    def constant_validation(model, loader, device, objective):
        metrics = real_evaluate(model, loader, device, objective)
        evaluations.append(loader.dataset)
        metrics['loss'] = 1.0
        return metrics

    monkeypatch.setattr(training, 'evaluate', constant_validation)
    settings = EncoderSettings(hidden_dim=16, heads=2, layers=1, latent_tokens=4, architecture=PATCH_ARCHITECTURE)
    output = tmp_path / 'encoder'
    metrics = train_autoencoder(dataset, output, settings, epochs=5, patience=1, augment=True,
                                batch_size=8, device='cpu')
    assert metrics['best_epoch'] == 1 and metrics['epochs_completed'] == 2
    assert metrics['objective'] == 'boundary_v2'
    assert len(evaluations) == 3
    assert [item.split for item in evaluations] == ['validation', 'validation', 'test']
    assert all(not item.augment for item in evaluations)
    model, checkpoint = load_autoencoder(output / 'best.pt')
    assert model.settings.architecture == PATCH_ARCHITECTURE
    assert checkpoint['architecture'] == PATCH_ARCHITECTURE
    checkpoint['architecture'] = 'spatial_anchor_local_decode_v1'
    torch.save(checkpoint, output / 'mismatched.pt')
    with pytest.raises(ValueError, match='architecture'):
        load_autoencoder(output / 'mismatched.pt')
