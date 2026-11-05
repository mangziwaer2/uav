"""独立地图重建训练；不训练LLM，不启用RL，不改变在线simulate。"""

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .data import CHANNELS, NORMALIZATION, json_write, history_window, file_hash
from .model import ARCHITECTURES, PATCH_ARCHITECTURE, EncoderSettings, MapAutoencoder, reconstruction_loss, metric_sums


class MapFrames(Dataset):
    def __init__(self, root, split, stride=1, augment=False):
        if type(augment) is not bool or augment and split != 'train':
            raise ValueError('spatial augmentation is boolean and training-only')
        self.augment = augment
        self.split = split
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if (self.manifest.get('schema_version') != 1 or self.manifest.get('channels') != list(CHANNELS)
                or self.manifest.get('normalization') != NORMALIZATION
                or self.manifest.get('split_unit') != 'scenario'):
            raise ValueError('unsupported map dataset schema')
        if type(stride) is not int or stride < 1:
            raise ValueError('frame stride must be a positive integer')
        scene_splits = {item['id']: item['split'] for item in self.manifest['scenarios']}
        if len(scene_splits) != len(self.manifest['scenarios']):
            raise ValueError('duplicate scenario ids')
        for item in self.manifest['episodes']:
            if scene_splits.get(item['scenario_id']) != item['split']:
                raise ValueError('scenario split leakage')
        episodes = [item for item in self.manifest['episodes'] if item['split'] == split]
        self.hashes = {item['path']: item.get('sha256') for item in episodes}
        self.index = [(item['path'], frame) for item in episodes for frame in range(0, item['frames'], stride)]
        if not self.index:
            raise ValueError(f'empty split: {split}')
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        relative, frame = self.index[index]
        if relative not in self.cache:
            path = (self.root / relative).resolve()
            if self.root.resolve() not in path.parents or path.parent.name != 'episodes':
                raise ValueError('episode path escapes dataset episodes directory')
            if self.hashes[relative] and file_hash(path) != self.hashes[relative]:
                raise ValueError('episode hash mismatch after dataset audit')
            with np.load(path, allow_pickle=False) as arrays:
                self.cache[relative] = arrays['maps']
        self.cache.move_to_end(relative)
        maps = torch.from_numpy(self.cache[relative][frame].copy())
        if self.augment:
            for axis in (1, 2):
                if torch.rand(()) < 0.5:
                    maps = maps.flip((axis,))
        return maps


def resolve_device(device):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)


def evaluate(model, loader, device, objective='balanced_v1'):
    model.eval()
    total, count, sums, components = 0.0, 0, {}, {}
    with torch.no_grad():
        for maps in loader:
            maps = maps.to(device)
            predicted = model(maps)
            loss, parts = reconstruction_loss(predicted, maps, objective)
            total += float(loss.item()) * len(maps)
            count += len(maps)
            for key, value in parts.items():
                components[key] = components.get(key, 0) + float(value.item()) * len(maps)
            for key, value in metric_sums(predicted, maps).items():
                sums[key] = sums.get(key, 0) + value
    true_positive = sums['support_intersection']
    return {'loss': total / count, 'loss_components': {key: value / count for key, value in components.items()},
            'supported_power_mae_db': sums['power_error_db'] / sums['power_count'] if sums['power_count'] else None,
            'high_power_mae_db': sums['high_error_db'] / sums['high_count'] if sums['high_count'] else None,
            'supported_age_mae_seconds': sums['age_error_seconds'] / sums['power_count'] if sums['power_count'] else None,
            'support_iou': sums['support_intersection'] / sums['support_union'] if sums['support_union'] else 1.0,
            'support_precision': true_positive / (true_positive + sums['support_false_positive'])
            if true_positive + sums['support_false_positive'] else None,
            'support_recall': true_positive / sums['power_count'] if sums['power_count'] else None,
            'unknown_false_positive_rate': sums['support_false_positive'] / sums['unknown_count']
            if sums['unknown_count'] else None,
            'blocked_iou': sums['blocked_intersection'] / sums['blocked_union'] if sums['blocked_union'] else 1.0,
            'uncertainty_mae_db': sums['uncertainty_error_db'] / sums['cell_count'],
            'building_height_mae_m': sums['height_error_m'] / sums['cell_count'],
            'supported_cells': sums['power_count'], 'high_cells': sums['high_count'], 'frames': count}


def load_autoencoder(checkpoint, device='cpu'):
    device = resolve_device(device)
    value = torch.load(checkpoint, map_location=device, weights_only=True)
    if (value.get('schema_version') != 1 or value.get('architecture') not in ARCHITECTURES
            or value.get('channels') != list(CHANNELS) or value.get('normalization') != NORMALIZATION):
        raise ValueError('unsupported map checkpoint')
    settings = EncoderSettings(**value['encoder_settings'])
    if settings.architecture != value['architecture']:
        raise ValueError('checkpoint architecture and encoder settings mismatch')
    model = MapAutoencoder(settings).to(device)
    model.load_state_dict(value['state_dict'])
    model.eval()
    return model, value


def train_autoencoder(dataset, output, encoder=EncoderSettings(), epochs=20, batch_size=16,
                      learning_rate=0.0003, frame_stride=2, seed=7, device='auto', require_quality=False,
                      augment=False, patience=None, objective=None):
    encoder.validate()
    if any(type(value) is not int or value < 1 for value in (epochs, batch_size, frame_stride)):
        raise ValueError('epochs, batch_size and frame_stride must be positive integers')
    if not math.isfinite(learning_rate) or learning_rate <= 0 or type(seed) is not int or seed < 0:
        raise ValueError('invalid learning rate or seed')
    if patience is not None and (type(patience) is not int or patience < 1):
        raise ValueError('patience must be a positive integer or null')
    if objective is None:
        objective = 'boundary_v2' if encoder.architecture == PATCH_ARCHITECTURE else 'balanced_v1'
    if objective not in ('balanced_v1', 'boundary_v2'):
        raise ValueError('unsupported reconstruction objective')
    quality_hash = None
    if require_quality:
        quality_path = Path(dataset) / 'quality_report.json'
        quality = json.loads(quality_path.read_text(encoding='utf-8'))
        if (not quality.get('passed') or not quality.get('coverage_required')
                or quality.get('manifest_sha256') != file_hash(Path(dataset) / 'manifest.json')):
            raise ValueError('dataset has not passed full coverage audit or manifest changed')
        quality_hash = file_hash(quality_path)
    train = MapFrames(dataset, 'train', frame_stride, augment)
    validation = MapFrames(dataset, 'validation', frame_stride)
    test = MapFrames(dataset, 'test', frame_stride)
    device = resolve_device(device)
    torch.manual_seed(seed)
    model = MapAutoencoder(encoder).to(device)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    manifest_hash = hashlib.sha256((Path(dataset) / 'manifest.json').read_bytes()).hexdigest()
    json_write(root / 'training_config.json', {'dataset': str(Path(dataset).resolve()), 'manifest_sha256': manifest_hash,
               'encoder': asdict(encoder), 'epochs': epochs, 'batch_size': batch_size,
               'learning_rate': learning_rate, 'frame_stride': frame_stride, 'seed': seed, 'device': str(device),
               'objective': objective, 'augment': augment, 'patience': patience,
               'architecture': encoder.architecture,
               'quality_report_sha256': quality_hash,
               'torch_version': str(torch.__version__), 'stage': 'spatial_reconstruction_only'})
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best, best_epoch = math.inf, 0
    with (root / 'history.jsonl').open('x', encoding='utf-8') as history:
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            for maps in train_loader:
                maps = maps.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = reconstruction_loss(model(maps), maps, objective)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite reconstruction loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.item()) * len(maps)
            metrics = evaluate(model, validation_loader, device, objective)
            if not math.isfinite(metrics['loss']):
                raise RuntimeError('nonfinite validation loss')
            record = {'epoch': epoch, 'train_loss': total / len(train), 'validation': metrics}
            history.write(json.dumps(record, allow_nan=False) + '\n')
            history.flush()
            if metrics['loss'] < best:
                best, best_epoch = metrics['loss'], epoch
                torch.save({'schema_version': 1, 'architecture': encoder.architecture,
                            'channels': list(CHANNELS), 'normalization': NORMALIZATION,
                            'encoder_settings': asdict(encoder), 'state_dict': model.state_dict(),
                            'manifest_sha256': manifest_hash, 'epoch': epoch, 'objective': objective}, root / 'best.pt')
            print(f'[训练 {epoch}/{epochs}] loss={record["train_loss"]:.5f} '
                  f'val={metrics["loss"]:.5f} supported_MAE={metrics["supported_power_mae_db"]} '
                  f'support_IoU={metrics["support_iou"]:.3f} precision={metrics["support_precision"]}', flush=True)
            if patience is not None and epoch - best_epoch >= patience:
                print(f'[验证早停] 连续{patience}轮未改善；保留第{best_epoch}轮，不读取test调参。', flush=True)
                break
    model, _ = load_autoencoder(root / 'best.pt', device)
    result = {'best_epoch': best_epoch, 'epochs_completed': epoch, 'objective': objective, 'validation_loss': best,
              'test': evaluate(model, DataLoader(test, batch_size=batch_size), device, objective),
              'note': 'reconstruction metrics, not LLM or navigation effectiveness'}
    json_write(root / 'metrics.json', result)
    return result


def export_window(checkpoint, episode, output, anchor=-1, frames=4, spacing_seconds=5.0, device='cpu'):
    """导出连续向量而非浮点文本；不加载evaluation_only，不包含锚点之后的测量。"""
    device = resolve_device(device)
    model, value = load_autoencoder(checkpoint, device)
    episode = Path(episode)
    with np.load(episode, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    anchor = len(arrays['times']) - 1 if anchor == -1 else anchor
    window = history_window(arrays['times'], anchor, frames, spacing_seconds)
    indices = [index for index in window['indices'] if index >= 0]
    with torch.no_grad():
        maps = torch.from_numpy(arrays['maps'][indices]).to(device)
        tokens = model.encode(maps)
        decoded = model.decode(tokens, *maps.shape[-2:]).cpu().numpy()
        reconstruction = decoded[-1]
    packed = np.zeros((frames, model.settings.latent_tokens, model.settings.hidden_dim), dtype=np.float32)
    reconstructed_window = np.zeros((frames, len(CHANNELS), *maps.shape[-2:]), dtype=np.float32)
    positions = np.zeros((frames, 2), dtype=np.float32)
    cursor = 0
    for slot, index in enumerate(window['indices']):
        if index >= 0:
            packed[slot] = tokens[cursor].cpu().numpy()
            reconstructed_window[slot] = decoded[cursor]
            positions[slot] = arrays['positions'][index]
            cursor += 1
    metadata = json.loads(episode.with_suffix('.json').read_text(encoding='utf-8'))
    timestamp = float(arrays['times'][anchor])
    observations = [item for item in metadata['observations'] if item['time_seconds'] <= timestamp]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    with (root / 'window_tokens.npz').open('xb') as stream:
        np.savez_compressed(stream, tokens=packed, positions_m=positions, valid=np.asarray(window['valid']),
                            offset_seconds=np.array([offset if offset is not None else 0 for offset in window['offset_seconds']]),
                            reconstruction=reconstruction, reconstructed_window=reconstructed_window,
                            onboard_target=arrays['maps'][anchor])
    json_write(root / 'window.json', {'window_schema_version': 2,
               'temporal_encoding': 'independent_spatial_frames_with_causal_timestamps',
               'anchor_time_seconds': timestamp, 'window': window,
               'token_shape': list(packed.shape), 'channels': list(CHANNELS), 'normalization': NORMALIZATION,
               'checkpoint_epoch': value['epoch'], 'public_map': metadata['map'],
               'architecture': value['architecture'], 'encoder_settings': value['encoder_settings'],
               'objective': value.get('objective', 'balanced_v1'), 'checkpoint_sha256': file_hash(checkpoint),
               'training_manifest_sha256': value['manifest_sha256'],
               'aircraft': metadata['aircraft'], 'public_propagation': metadata['public_propagation'],
               'vehicle': {'position_m': arrays['positions'][anchor].tolist(),
                           'energy_wh': float(arrays['energy_wh'][anchor])},
               'recent_observations': [item for item in observations if timestamp - item['time_seconds'] <= 35],
               'note': 'not yet projected to LLM embeddings; padding validity must be respected'})
    plot_reconstruction(arrays['maps'][anchor], reconstruction, root / 'reconstruction.png')
    return root


def plot_reconstruction(target, predicted, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot
    figure, axes = pyplot.subplots(1, 4, figsize=(16, 4))
    visible = target[1] > 0.5
    images = [np.where(visible, target[0] * 90 - 120, np.nan),
              np.where(predicted[1] > 0, predicted[0] * 90 - 120, np.nan),
              1 / (1 + np.exp(-np.clip(predicted[1], -80, 80))),
              np.where(visible, np.abs(predicted[0] - target[0]) * 90, np.nan)]
    titles = ('Onboard estimate (not truth)', 'Reconstruction', 'Support sigmoid score', 'Supported error (dB)')
    for axis, image, title in zip(axes, images, titles):
        low, high = (0, 1) if 'score' in title else (0, 30) if 'error' in title else (-120, -30)
        artist = axis.imshow(image, origin='lower', vmin=low, vmax=high, cmap='viridis')
        axis.set_title(title)
        figure.colorbar(artist, ax=axis, shrink=0.7)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    pyplot.close(figure)
