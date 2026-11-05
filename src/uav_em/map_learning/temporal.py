"""同一区域时序重建核查；只读取因果机载地图，不推断真实源运动。"""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from .data import (CHANNELS, NORMALIZATION, CollectionSettings, file_hash, history_window,
                   json_write, temporal_target)
from .quality import checked_path
from .training import load_autoencoder, resolve_device


def pair_sums(older, newer, predicted_older, predicted_newer, change_db=3.0):
    if not math.isfinite(change_db) or change_db <= 0:
        raise ValueError('change_db must be positive and finite')
    values = (older, newer, predicted_older, predicted_newer)
    if (older.ndim != 3 or older.shape[0] != len(CHANNELS)
            or any(value.shape != older.shape or not np.isfinite(value).all() for value in values)):
        raise ValueError('temporal pairs require finite aligned six-channel maps')
    before, after = older[1] > 0.5, newer[1] > 0.5
    decoded_before, decoded_after = predicted_older[1] > 0, predicted_newer[1] > 0
    common, decoded_common = before & after, decoded_before & decoded_after
    delta = (newer[0] - older[0]) * 90
    decoded_delta = (predicted_newer[0] - predicted_older[0]) * 90
    age_delta = (newer[3] - older[3]) * 35
    decoded_age_delta = (predicted_newer[3] - predicted_older[3]) * 35
    changed = common & (np.abs(delta) >= change_db)
    correct_direction = ((delta > 0) == (decoded_delta > 0)) & (np.abs(decoded_delta) >= change_db)
    sums = {'tasks': 1, 'common_cells': int(common.sum()), 'changed_cells': int(changed.sum()),
            'common_retained': int((common & decoded_common).sum()),
            'delta_error_db': float(np.abs(delta - decoded_delta)[common].sum()),
            'changed_delta_error_db': float(np.abs(delta - decoded_delta)[changed].sum()),
            'age_delta_error_seconds': float(np.abs(age_delta - decoded_age_delta)[common].sum()),
            'changed_direction_correct': int((changed & correct_direction & decoded_common).sum()),
            'current_cells': int(after.sum()),
            'current_age_error_seconds': float(np.abs(newer[3] - predicted_newer[3])[after].sum() * 35)}
    events = {'new_support': (~before & after, ~decoded_before & decoded_after),
              'lost_support': (before & ~after, decoded_before & ~decoded_after),
              'power_change': (changed, decoded_common & (np.abs(decoded_delta) >= change_db))}
    for name, (target, predicted) in events.items():
        sums.update({f'{name}_tp': int((target & predicted).sum()),
                     f'{name}_fp': int((~target & predicted).sum()),
                     f'{name}_fn': int((target & ~predicted).sum())})
    return sums


def summarize(sums):
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    result = {key: int(sums.get(key, 0)) for key in ('tasks', 'common_cells', 'changed_cells', 'current_cells')}
    for name in ('new_support', 'lost_support', 'power_change'):
        positive, false, missed = (int(sums.get(f'{name}_{key}', 0)) for key in ('tp', 'fp', 'fn'))
        result[name] = {'true_positive': positive, 'false_positive': false, 'false_negative': missed,
                        'precision': ratio(positive, positive + false),
                        'recall': ratio(positive, positive + missed),
                        'iou': ratio(positive, positive + false + missed)}
    result.update({'common_support_retention': ratio(sums.get('common_retained', 0), result['common_cells']),
                   'power_delta_mae_db': ratio(sums.get('delta_error_db', 0), result['common_cells']),
                   'changed_power_delta_mae_db': ratio(sums.get('changed_delta_error_db', 0), result['changed_cells']),
                   'changed_direction_recall': ratio(sums.get('changed_direction_correct', 0), result['changed_cells']),
                   'age_delta_mae_seconds': ratio(sums.get('age_delta_error_seconds', 0), result['common_cells']),
                   'current_age_mae_seconds': ratio(sums.get('current_age_error_seconds', 0), result['current_cells'])})
    return result


def validate_task(task, maps, times, collection):
    anchor, older = task['anchor'], task['older_index']
    if type(older) is not int:
        raise ValueError('older_index must be an integer')
    window = history_window(times, anchor, collection.window_frames, collection.window_spacing_seconds)
    valid = [index for index in window['indices'] if index >= 0]
    if len(valid) < 2 or older != valid[0] or task['window'] != window:
        raise ValueError('noncausal or inconsistent temporal window')
    box = task['roi_grid_xyxy']
    height, width = maps.shape[-2:]
    if (len(box) != 4 or any(type(value) is not int for value in box)
            or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
        raise ValueError('invalid temporal ROI')
    if task['task'] != 'temporal_region' or task['target'] != temporal_target(maps[older], maps[anchor], box):
        raise ValueError('temporal labels do not match recorded onboard frames')


def decode_frames(model, maps, device, batch_size):
    decoded = []
    with torch.inference_mode():
        for begin in range(0, len(maps), batch_size):
            batch = torch.from_numpy(maps[begin:begin + batch_size]).to(device)
            tokens = model.encode(batch)
            decoded.append(model.decode(tokens, *maps.shape[-2:]).cpu().numpy())
    return np.concatenate(decoded)


def audit_temporal_codec(dataset, checkpoint, output, split='validation', device='cpu', batch_size=32, change_db=3.0):
    """固定权重重建已有任务B；重叠ROI按任务计数，不当作独立样本置信区间。"""
    if split not in ('validation', 'test') or type(batch_size) is not int or batch_size < 1:
        raise ValueError('need validation/test split and positive integer batch size')
    if not math.isfinite(change_db) or change_db <= 0:
        raise ValueError('change_db must be positive and finite')
    root = Path(dataset)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != 1 or manifest.get('channels') != list(CHANNELS)
            or manifest.get('normalization') != NORMALIZATION or manifest.get('split_unit') != 'scenario'
            or manifest.get('quantity') != 'onboard_belief_not_truth'):
        raise ValueError('unsupported onboard map dataset')
    scenes = {item['id']: item['split'] for item in manifest['scenarios']}
    entries = {item['id']: item for item in manifest['episodes']}
    if len(scenes) != len(manifest['scenarios']) or len(entries) != len(manifest['episodes']):
        raise ValueError('duplicate scene or episode ids')
    if any(scenes.get(item['scenario_id']) != item['split'] for item in entries.values()):
        raise ValueError('scenario split leakage')
    collection = CollectionSettings(**manifest['collection_settings'])
    collection.validate()
    task_path = root / 'temporal_tasks.jsonl'
    if file_hash(task_path) != manifest['temporal_tasks_sha256']:
        raise ValueError('temporal task hash mismatch')
    tasks = defaultdict(list)
    with task_path.open(encoding='utf-8') as stream:
        for line in stream:
            task = json.loads(line)
            entry = entries.get(task['episode_id'])
            if entry is None or entry['split'] != task['split']:
                raise ValueError('task split or episode mismatch')
            if task['split'] == split:
                tasks[entry['id']].append(task)
    selected = [item for item in entries.values() if item['split'] == split]
    if not selected or not tasks:
        raise ValueError('empty temporal split')
    device = resolve_device(device)
    model, saved = load_autoencoder(checkpoint, device)
    manifest_hash = file_hash(root / 'manifest.json')
    if saved['manifest_sha256'] != manifest_hash:
        raise ValueError('checkpoint and dataset manifest mismatch')
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    totals, by_policy, candidates = defaultdict(float), {}, {}
    frames_checked = 0
    with (destination / 'task_metrics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = None
        for entry in selected:
            episode_tasks = tasks[entry['id']]
            if len(episode_tasks) != entry['temporal_tasks']:
                raise ValueError('temporal task count mismatch')
            path = checked_path(root, entry['path'])
            if path.parent != (root / 'episodes').resolve() or path.suffix != '.npz':
                raise ValueError('expected direct episodes npz file')
            if file_hash(path) != entry['sha256']:
                raise ValueError('episode hash mismatch')
            with np.load(path, allow_pickle=False) as archive:
                maps, times = archive['maps'], archive['times']
            if (maps.ndim != 4 or maps.shape[:2] != (entry['frames'], len(CHANNELS))
                    or times.shape != (len(maps),) or not np.isfinite(maps).all()):
                raise ValueError('invalid episode maps or times')
            for task in episode_tasks:
                validate_task(task, maps, times, collection)
            decoded = decode_frames(model, maps, device, batch_size)
            frames_checked += len(maps)
            policy_totals = by_policy.setdefault(entry['policy'], defaultdict(float))
            for task in episode_tasks:
                left, top, right, bottom = task['roi_grid_xyxy']
                indices = [task['older_index'], task['anchor']]
                target = maps[indices, :, top:bottom, left:right]
                predicted = decoded[indices, :, top:bottom, left:right]
                sums = pair_sums(*target, *predicted, change_db)
                for key, value in sums.items():
                    totals[key] += value
                    policy_totals[key] += value
                row = {'episode_id': entry['id'], 'policy': entry['policy'], 'anchor': task['anchor'],
                       'older_index': task['older_index'], 'older_time_seconds': float(times[indices[0]]),
                       'anchor_time_seconds': float(times[indices[1]]), 'roi_grid_xyxy': json.dumps(task['roi_grid_xyxy']),
                       **sums}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                scores = {'new_support': sums['new_support_tp'] + sums['new_support_fn'],
                          'lost_support': sums['lost_support_tp'] + sums['lost_support_fn'],
                          'power_change': sums['changed_cells'],
                          'stable_region': sums['common_cells'] if not sums['changed_cells'] else 0}
                for kind, score in scores.items():
                    if score > candidates.get(kind, {}).get('score', 0):
                        candidates[kind] = {'score': score, 'task': task, 'row': row,
                                            'target': target.copy(), 'predicted': predicted.copy()}
            print(f'[时序核查] {entry["id"]}: {len(episode_tasks)} tasks', flush=True)
    report = {'schema_version': 1, 'dataset': str(root.resolve()), 'split': split,
              'manifest_sha256': manifest_hash, 'temporal_tasks_sha256': file_hash(task_path),
              'checkpoint_sha256': file_hash(checkpoint), 'architecture': saved['architecture'],
              'encoder_settings': saved['encoder_settings'], 'checkpoint_epoch': saved['epoch'],
              'change_threshold_db': change_db, 'episodes': len(selected), 'frames_decoded': frames_checked,
              'overall': summarize(totals), 'by_policy': {key: summarize(value) for key, value in by_policy.items()},
              'raw_sums': dict(totals), 'uses_evaluation_only': False,
              'counting': 'overlapping task ROI-cell occurrences; not independent trials',
              'regression_mask': 'recorded common support, not decoder-selected easy cells',
              'interpretation': 'belief reconstruction only; not source motion, LLM or navigation effectiveness'}
    json_write(destination / 'report.json', report)
    for kind, candidate in candidates.items():
        export_temporal_example(destination / kind, candidate)
    write_audit_readme(destination, report)
    return report


def export_temporal_example(destination, candidate):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot
    from matplotlib.colors import BoundaryNorm, ListedColormap

    destination.mkdir(parents=True, exist_ok=False)
    target, predicted = candidate['target'], candidate['predicted']
    decoded_labels = predicted.copy()
    decoded_labels[:, 1] = (decoded_labels[:, 1] > 0).astype(np.float32)
    height, width = target.shape[-2:]
    box = [0, 0, width, height]
    json_write(destination / 'sample.json', {
        'selection': 'largest recorded event count for illustration, not worst/best reconstruction',
        'task': candidate['task'], 'metrics': candidate['row'],
        'decoded_target': temporal_target(*decoded_labels, box),
        'decoded_target_note': 'derived approximate labels, not LLM output or new observations'})
    with (destination / 'roi_reconstruction.npz').open('xb') as stream:
        np.savez_compressed(stream, onboard=target, reconstructed=predicted)
    figure, axes = pyplot.subplots(2, 5, figsize=(17, 7), squeeze=False)
    titles = ('Older power (dBm)', 'Current power (dBm)', 'Power change (dB)',
              'Support: lost / unchanged / new', 'Current age (s)')
    event_cmap = ListedColormap(('#3876bc', '#eeeeee', '#e78b31'))
    event_norm = BoundaryNorm((-1.5, -0.5, 0.5, 1.5), event_cmap.N)
    for row, values in enumerate((target, decoded_labels)):
        before, after = values[:, 1] > 0.5
        common = before & after
        events = (~before & after).astype(int) - (before & ~after).astype(int)
        images = (np.where(before, values[0, 0] * 90 - 120, np.nan),
                  np.where(after, values[1, 0] * 90 - 120, np.nan),
                  np.where(common, (values[1, 0] - values[0, 0]) * 90, np.nan),
                  events, np.where(after, values[1, 3] * 35, np.nan))
        for column, image in enumerate(images):
            options = ({'vmin': -120, 'vmax': -30, 'cmap': 'turbo'} if column < 2 else
                       {'vmin': -30, 'vmax': 30, 'cmap': 'coolwarm'} if column == 2 else
                       {'norm': event_norm, 'cmap': event_cmap} if column == 3 else
                       {'vmin': 0, 'vmax': 35, 'cmap': 'viridis'})
            artist = axes[row, column].imshow(image, origin='lower', **options)
            axes[row, column].set_title(titles[column])
            axes[row, column].set_xlabel('ROI column')
            axes[row, column].set_ylabel(('Onboard' if row == 0 else 'Decoded') + ' ROI row')
            axes[row, column].set_facecolor('#dddddd')
            figure.colorbar(artist, ax=axes[row, column], shrink=0.7)
    record = candidate['row']
    figure.suptitle(f'{record["episode_id"]} | same ROI | '
                   f't={record["older_time_seconds"]:g} -> {record["anchor_time_seconds"]:g}s')
    figure.tight_layout()
    figure.savefig(destination / 'comparison.png', dpi=130)
    pyplot.close(figure)


def write_audit_readme(destination, report):
    overall = report['overall']
    text = f'''# 同一区域时序重建核查

分组：{report['split']}；{report['episodes']}条轨迹、{overall['tasks']}道原始任务B。
固定checkpoint第{report['checkpoint_epoch']}轮；仅解码连续地图向量，不训练LLM或时序模块。

先打开new_support、lost_support、power_change目录中的comparison.png（只有对应事件存在才生成）。
stable_region展示原图无达到阈值的变化、且共支持格最多的任务，便于观察编码是否引入假变化。
上行为原始机载估计，下行为编码后重建；五列依次为较早功率、当前功率、功率差、支持变化、当前信息年龄。
功率差只在各行自身前后均有支持的格子显示，灰色为无支持，不是零功率。
事件列蓝色=支持失效、橙色=新增支持、浅灰=未变化；这不是源运动真值。
样本按原始任务的事件格数最大选取，不根据重建好坏挑图，不代表平均水平。
sample.json含原始标签、近似重建标签及指标，roi_reconstruction.npz保留两时刻数值。

整体功率差MAE：{overall['power_delta_mae_db']}dB。
强变化格（绝对变化至少{report['change_threshold_db']}dB）功率差MAE：{overall['changed_power_delta_mae_db']}dB。
方向召回率：{overall['changed_direction_recall']}；前后共支持保留率：{overall['common_support_retention']}。
完整事件精度／召回／IoU及goal／coverage／revisit分组见report.json，逐题原始计数见task_metrics.csv。

功率差与年龄差回归以原图共支持格为分母，不能只计算解码器自己保留的容易格。
变化方向召回要求两个时刻均被解码为有支持、变化方向正确且达到同一阈值；漏支持计失败。
没有分母时指标为null，不用1.0冒充成功。阈值只是诊断尺度，不是设备或安全阈值。
任务ROI和时间窗可能重叠，格数是任务格出现次数，不是独立样本；不据此给置信区间。

只读取manifest、已校验哈希的episodes及temporal_tasks；核对场景划分、因果窗口和原始标签。
不读取evaluation_only，也不加载未来测量给较早帧编码；各帧独立encode/decode，保持同一世界坐标。
此工具不是重新生成数据的完整物理审核，不证明源可辨识、时序推理、LLM收益或安全最优。
'''
    with (destination / 'README.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
