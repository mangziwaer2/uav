"""以原始机载时序题微调空间编解码器；不训练LLM或使用隐藏源。"""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import CollectionSettings, file_hash, json_write
from .model import masked_mean, reconstruction_loss
from .quality import checked_path
from .temporal import validate_task
from .training import MapFrames, evaluate, load_autoencoder, resolve_device


@dataclass(frozen=True)
class TemporalSettings:
    epochs: int = 12
    batch_size: int = 16
    learning_rate: float = 0.0001
    seed: int = 7
    patience: int = 4
    change_db: float = 3.0
    power_weight: float = 0.5
    age_weight: float = 0.1
    transition_weight: float = 0.1
    augment: bool = True

    def validate(self):
        for name in ('epochs', 'batch_size', 'patience'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.seed) is not int or self.seed < 0 or type(self.augment) is not bool:
            raise ValueError('invalid seed or augment')
        for name in ('learning_rate', 'change_db', 'power_weight', 'age_weight', 'transition_weight'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')


class MapPairs(Dataset):
    def __init__(self, root, split, change_db=3.0, augment=False):
        if split not in ('train', 'validation') or type(augment) is not bool or augment and split != 'train':
            raise ValueError('pairs support train/validation and training-only augmentation')
        if not math.isfinite(change_db) or change_db <= 0:
            raise ValueError('change_db must be positive and finite')
        self.root, self.split, self.augment = Path(root), split, augment
        self.manifest = MapFrames(root, split).manifest
        self.collection = CollectionSettings(**self.manifest['collection_settings'])
        self.collection.validate()
        self.entries = {entry['id']: entry for entry in self.manifest['episodes']}
        if len(self.entries) != len(self.manifest['episodes']):
            raise ValueError('duplicate episode ids')
        path = self.root / 'temporal_tasks.jsonl'
        if file_hash(path) != self.manifest['temporal_tasks_sha256']:
            raise ValueError('temporal task hash mismatch')
        self.tasks, self.strata, self.cache, self.checked = [], [], {}, set()
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                task = json.loads(line)
                entry = self.entries.get(task['episode_id'])
                if entry is None or entry['split'] != task['split']:
                    raise ValueError('task split or episode mismatch')
                if entry['split'] != split:
                    continue
                values = task['target']
                changed = any(value is not None and abs(value) >= change_db
                              for row in values['power_delta_db'] for value in row)
                transition = any(value for key in ('newly_supported', 'lost_support')
                                 for row in values[key] for value in row)
                self.tasks.append(task)
                self.strata.append(f'power_{int(changed)}_support_{int(transition)}')
        counts = Counter(task['episode_id'] for task in self.tasks)
        if not self.tasks or any(counts[entry['id']] != entry['temporal_tasks'] for entry in self.entries.values()
                                 if entry['split'] == split):
            raise ValueError('empty or inconsistent temporal task counts')
        self.stratum_counts = dict(Counter(self.strata))
        self.weights = [1 / self.stratum_counts[stratum] for stratum in self.strata]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        task = self.tasks[index]
        entry = self.entries[task['episode_id']]
        if entry['id'] not in self.cache:
            path = checked_path(self.root, entry['path'])
            if path.parent != (self.root / 'episodes').resolve() or path.suffix != '.npz':
                raise ValueError('expected episodes npz file')
            if file_hash(path) != entry['sha256']:
                raise ValueError('episode hash mismatch')
            with np.load(path, allow_pickle=False) as arrays:
                self.cache[entry['id']] = (arrays['maps'], arrays['times'])
        maps, times = self.cache[entry['id']]
        if index not in self.checked:
            validate_task(task, maps, times, self.collection)
            self.checked.add(index)
        pair = torch.from_numpy(maps[[task['older_index'], task['anchor']]].copy())
        region = torch.zeros(maps.shape[-2:], dtype=torch.bool)
        left, top, right, bottom = task['roi_grid_xyxy']
        region[top:bottom, left:right] = True
        if self.augment:
            for axis in (-2, -1):
                if torch.rand(()) < 0.5:
                    pair, region = pair.flip((axis,)), region.flip((axis,))
        return pair, region


def temporal_loss(predicted, target, region, settings):
    if predicted.shape != target.shape or target.ndim != 5 or target.shape[1:3] != (2, 6):
        raise ValueError('temporal loss expects aligned [batch,2,6,rows,columns] pairs')
    if region.shape != target[:, 0, 1].shape or region.dtype != torch.bool:
        raise ValueError('ROI mask must be boolean and match pair dimensions')
    before, after = target[:, 0, 1] > 0.5, target[:, 1, 1] > 0.5
    common = before & after & region
    delta = (target[:, 1, 0] - target[:, 0, 0]) * 90
    decoded_delta = (predicted[:, 1, 0] - predicted[:, 0, 0]) * 90
    change = common & (delta.abs() >= settings.change_db)
    stable = common & ~change
    error = functional.smooth_l1_loss(decoded_delta / 10, delta / 10, reduction='none')
    groups = change.any().to(error.dtype) + stable.any().to(error.dtype)
    power = (masked_mean(error, change) + masked_mean(error, stable)) / groups.clamp_min(1)
    age_delta = target[:, 1, 3] - target[:, 0, 3]
    predicted_age_delta = predicted[:, 1, 3] - predicted[:, 0, 3]
    age = masked_mean(functional.smooth_l1_loss(predicted_age_delta, age_delta, reduction='none'), common)
    first, last = predicted[:, 0, 1].sigmoid(), predicted[:, 1, 1].sigmoid()
    new_probability, lost_probability = (1 - first) * last, first * (1 - last)
    transitions = []
    for probability, labels in ((new_probability, ~before & after), (lost_probability, before & ~after)):
        binary = functional.binary_cross_entropy(probability.clamp(1e-6, 1 - 1e-6), labels.float(), reduction='none')
        transitions.append(masked_mean(binary, region))
    transition = sum(transitions) / len(transitions)
    base, _ = reconstruction_loss(predicted.flatten(0, 1), target.flatten(0, 1), 'boundary_v2')
    loss = base + settings.power_weight * power + settings.age_weight * age + settings.transition_weight * transition
    return loss, {'spatial': base, 'power_delta': power, 'age_delta': age, 'support_transition': transition}


def evaluate_pairs(model, loader, device, settings):
    model.eval()
    total, count, components = 0.0, 0, Counter()
    with torch.inference_mode():
        for pairs, region in loader:
            pairs, region = pairs.to(device), region.to(device)
            predicted = model(pairs.flatten(0, 1)).reshape_as(pairs)
            loss, parts = temporal_loss(predicted, pairs, region, settings)
            total += loss.item() * len(pairs)
            count += len(pairs)
            for name, value in parts.items():
                components[name] += value.item() * len(pairs)
    return {'loss': total / count, 'components': {name: value / count for name, value in components.items()},
            'tasks': count}


def spatial_guard(candidate, baseline):
    return (candidate['support_iou'] >= baseline['support_iou'] - 0.01
            and candidate['supported_power_mae_db'] <= baseline['supported_power_mae_db'] + 0.25
            and (baseline['high_power_mae_db'] is None or
                 candidate['high_power_mae_db'] is not None and
                 candidate['high_power_mae_db'] <= baseline['high_power_mae_db'] + 0.25))


def train_temporal_autoencoder(dataset, checkpoint, output, settings=TemporalSettings(), device='auto', require_quality=False):
    settings.validate()
    root, device = Path(dataset), resolve_device(device)
    manifest_hash = file_hash(root / 'manifest.json')
    quality_hash = None
    if require_quality:
        report_path = root / 'quality_report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if not report.get('passed') or not report.get('coverage_required') or report['manifest_sha256'] != manifest_hash:
            raise ValueError('need matching complete coverage audit')
        quality_hash = file_hash(report_path)
    torch.manual_seed(settings.seed)
    model, parent = load_autoencoder(checkpoint, device)
    if parent['manifest_sha256'] != manifest_hash or parent['objective'] != 'boundary_v2':
        raise ValueError('warm start requires matching manifest and boundary_v2 checkpoint')
    train = MapPairs(root, 'train', settings.change_db, settings.augment)
    validation = MapPairs(root, 'validation', settings.change_db)
    train_generator = torch.Generator().manual_seed(settings.seed)
    sampler = WeightedRandomSampler(train.weights, len(train), replacement=True, generator=train_generator)
    train_loader = DataLoader(train, batch_size=settings.batch_size, sampler=sampler)
    validation_loader = DataLoader(validation, batch_size=settings.batch_size)
    frame_loader = DataLoader(MapFrames(root, 'validation', stride=2), batch_size=settings.batch_size * 2)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parent
    configuration = {'stage': 'temporal_regularized_spatial_autoencoder', 'dataset': str(root.resolve()),
                     'manifest_sha256': manifest_hash, 'parent_checkpoint_sha256': file_hash(checkpoint),
                     'parent_checkpoint': str(Path(checkpoint).resolve()), 'parent_epoch': parent['epoch'],
                     'settings': asdict(settings), 'encoder_settings': parent['encoder_settings'],
                     'quality_report_sha256': quality_hash, 'device': str(device), 'torch_version': str(torch.__version__),
                     'source_sha256': {path.name: file_hash(path) for path in sorted(source_root.glob('*.py'))},
                     'train_stratum_counts': train.stratum_counts, 'validation_stratum_counts': validation.stratum_counts,
                     'sampling': 'train only: equal mass per nonempty power-change/support-transition stratum',
                     'selection': 'natural validation paired loss plus spatial guards; baseline epoch 0 retained',
                     'guard_support_iou_drop': 0.01, 'guard_power_mae_increase_db': 0.25,
                     'guard_high_power_mae_increase_db': 0.25, 'validation_frame_stride': 2,
                     'test_evaluated': False, 'uses_evaluation_only': False}
    json_write(destination / 'training_config.json', configuration)
    baseline = {'temporal': evaluate_pairs(model, validation_loader, device, settings),
                'spatial': evaluate(model, frame_loader, device, 'boundary_v2')}
    json_write(destination / 'baseline_validation.json', baseline)
    best, best_epoch, best_metrics = baseline['temporal']['loss'], 0, baseline
    saved = {**parent, 'epoch': 0, 'training_stage': configuration['stage'],
             'parent_checkpoint_sha256': configuration['parent_checkpoint_sha256'],
             'parent_epoch': parent['epoch'], 'temporal_settings': asdict(settings), 'test_evaluated': False}
    torch.save({**saved, 'state_dict': model.state_dict()}, destination / 'best.pt')
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    stale, completed = 0, 0
    print(f'[时序基线] validation_loss={best:.6f}; train={len(train)} validation={len(validation)}', flush=True)
    with (destination / 'history.jsonl').open('x', encoding='utf-8') as stream:
        for epoch in range(1, settings.epochs + 1):
            model.train()
            total, count = 0.0, 0
            for pairs, region in train_loader:
                pairs, region = pairs.to(device), region.to(device)
                optimizer.zero_grad(set_to_none=True)
                predicted = model(pairs.flatten(0, 1)).reshape_as(pairs)
                loss, _ = temporal_loss(predicted, pairs, region, settings)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite temporal training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(pairs)
                count += len(pairs)
            metrics = {'temporal': evaluate_pairs(model, validation_loader, device, settings),
                       'spatial': evaluate(model, frame_loader, device, 'boundary_v2')}
            guard = spatial_guard(metrics['spatial'], baseline['spatial'])
            improved = guard and metrics['temporal']['loss'] < best
            if improved:
                best, best_epoch, best_metrics = metrics['temporal']['loss'], epoch, metrics
                torch.save({**saved, 'epoch': epoch, 'state_dict': model.state_dict()}, destination / 'best.pt')
            stale = 0 if improved else stale + 1
            record = {'epoch': epoch, 'train_loss': total / count, 'validation': metrics,
                      'spatial_guard_passed': guard, 'selected': improved}
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            completed = epoch
            print(f'[时序训练] {epoch}/{settings.epochs} train={total / count:.6f} '
                  f'validation={metrics["temporal"]["loss"]:.6f} spatial_guard={guard} best={best_epoch}', flush=True)
            if stale >= settings.patience:
                break
    result = {'best_epoch': best_epoch, 'epochs_completed': completed, 'baseline_validation': baseline,
              'selected_validation': best_metrics, 'test_evaluated': False,
              'note': 'temporal reconstruction candidate, not source inference or LLM decision effectiveness'}
    json_write(destination / 'metrics.json', result)
    return result
