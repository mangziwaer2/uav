"""冻结地图编码器的区域问答准备；答案与机载因果输入分离，不训练LLM。"""

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .data import CHANNELS, NORMALIZATION, CollectionSettings, file_hash, json_write
from .quality import checked_path
from .temporal import validate_task
from .training import MapFrames, load_autoencoder, resolve_device


MAP_MARKER = '<|onboard_map_embeddings|>'


def current_region_target(frame, box):
    left, top, right, bottom = box
    region = frame[:, top:bottom, left:right]
    support = region[1] > 0.5

    def physical(channel, scale, shift=0, masked=False):
        return [[round(float(value * scale + shift), 3) if not masked or valid else None
                 for value, valid in zip(row, mask)] for row, mask in zip(region[channel], support)]
    return {'supported': support.astype(int).tolist(), 'power_dbm': physical(0, 90, -120, True),
            'uncertainty_db': physical(2, 30), 'age_seconds': physical(3, 35, masked=True),
            'blocked': (region[4] > 0.5).astype(int).tolist(), 'building_height_m': physical(5, 80),
            'interpretation': 'onboard_belief_not_truth'}


def input_record(entry, arrays, task, altitude_m):
    indices = task['window']['indices']
    return {'token_path': f'tokens/{entry["id"]}.npz', 'frame_indices': indices,
            'valid': task['window']['valid'], 'offset_seconds': task['window']['offset_seconds'],
            'positions_m': [arrays['positions'][index].tolist() if index >= 0 else None for index in indices],
            'roi_grid_xyxy': task['roi_grid_xyxy'],
            'map_grid': {'rows': int(arrays['maps'].shape[-2]), 'columns': int(arrays['maps'].shape[-1]),
                         'resolution_m': float(arrays['resolution_m']),
                         'flight_altitude_m': altitude_m}}


def question_text(kind, record):
    context = {key: record[key] for key in ('offset_seconds', 'positions_m', 'roi_grid_xyxy', 'map_grid')}
    instruction = ('还原当前帧指定ROI的六通道机载地图，功率和年龄未知处输出null。'
                   if kind == 'current_region' else
                   '比较历史窗口最早有效帧与当前帧的同一ROI，输出支持增失、共支持功率差及当前年龄。')
    return ('输入是按时间从旧到新排列的机载地图连续向量，不是完整环境真值。\n'
            + MAP_MARKER + '\n公开上下文：' + json.dumps(context, ensure_ascii=False, separators=(',', ':'))
            + '\n任务：' + instruction + '只输出JSON；不能把估计变化称为真实源移动。')


def build_alignment_dataset(dataset, checkpoint, output, device='cpu', batch_size=32):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    root, device = Path(dataset), resolve_device(device)
    manifest = MapFrames(root, 'train').manifest
    if manifest.get('quantity') != 'onboard_belief_not_truth':
        raise ValueError('alignment input must be onboard belief')
    collection = CollectionSettings(**manifest['collection_settings'])
    collection.validate()
    task_path = root / 'temporal_tasks.jsonl'
    if file_hash(task_path) != manifest['temporal_tasks_sha256']:
        raise ValueError('temporal task hash mismatch')
    entries = {entry['id']: entry for entry in manifest['episodes']}
    if len(entries) != len(manifest['episodes']):
        raise ValueError('duplicate episode ids')
    tasks = defaultdict(list)
    with task_path.open(encoding='utf-8') as stream:
        for line in stream:
            task = json.loads(line)
            entry = entries.get(task['episode_id'])
            if entry is None or task['split'] != entry['split']:
                raise ValueError('task split mismatch')
            if entry['split'] in ('train', 'validation'):
                tasks[entry['id']].append(task)
    model, saved = load_autoencoder(checkpoint, device)
    manifest_hash = file_hash(root / 'manifest.json')
    if saved['manifest_sha256'] != manifest_hash:
        raise ValueError('checkpoint manifest mismatch')
    model.requires_grad_(False)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'tokens').mkdir()
    (destination / 'samples').mkdir()
    cache_entries, counts, frame_counts, examples = [], Counter(), Counter(), set()
    with (destination / 'train.jsonl').open('x', encoding='utf-8') as train_stream, \
            (destination / 'validation.jsonl').open('x', encoding='utf-8') as validation_stream:
        for entry in entries.values():
            if entry['split'] not in ('train', 'validation'):
                continue
            if len(tasks[entry['id']]) != entry['temporal_tasks']:
                raise ValueError('temporal task count mismatch')
            path = checked_path(root, entry['path'])
            if path.parent != (root / 'episodes').resolve() or path.suffix != '.npz':
                raise ValueError('expected direct episodes npz file')
            if file_hash(path) != entry['sha256'] or file_hash(path.with_suffix('.json')) != entry['metadata_sha256']:
                raise ValueError('episode data or public metadata hash mismatch')
            public = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in ('maps', 'times', 'positions', 'resolution_m')}
            for task in tasks[entry['id']]:
                validate_task(task, arrays['maps'], arrays['times'], collection)
            encoded = []
            with torch.inference_mode():
                for begin in range(0, len(arrays['maps']), batch_size):
                    batch = torch.from_numpy(arrays['maps'][begin:begin + batch_size]).to(device)
                    encoded.append(model.encode(batch).cpu().numpy())
            tokens = np.concatenate(encoded)
            relative = f'tokens/{entry["id"]}.npz'
            with (destination / relative).open('xb') as stream:
                np.savez_compressed(stream, tokens=tokens, times_seconds=arrays['times'])
            cache_entries.append({'episode_id': entry['id'], 'scenario_id': entry['scenario_id'], 'split': entry['split'],
                                  'path': relative, 'sha256': file_hash(destination / relative),
                                  'source_episode_sha256': entry['sha256'], 'frames': len(tokens)})
            frame_counts[entry['split']] += len(tokens)
            for task in tasks[entry['id']]:
                inputs = input_record(entry, arrays, task, public['aircraft']['altitude_m'])
                for kind in ('current_region', 'temporal_region'):
                    answer = (current_region_target(arrays['maps'][task['anchor']], task['roi_grid_xyxy'])
                              if kind == 'current_region' else task['target'])
                    row = {'id': f'{entry["id"]}_{task["anchor"]}_{kind}', 'episode_id': entry['id'],
                           'scenario_id': entry['scenario_id'], 'split': entry['split'], 'kind': kind,
                           'inputs': inputs, 'question': question_text(kind, inputs), 'answer': answer}
                    stream = train_stream if entry['split'] == 'train' else validation_stream
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
                    counts[entry['split']] += 1
                    if entry['split'] == 'validation' and kind not in examples:
                        json_write(destination / 'samples' / f'{kind}.json', row)
                        examples.add(kind)
            print(f'[对齐数据] {entry["id"]}: {len(tasks[entry["id"]]) * 2} questions', flush=True)
    result = {'schema_version': 1, 'stage': 'alignment_data_only_no_llm_training',
              'source_dataset': str(root.resolve()), 'source_manifest_sha256': manifest_hash,
              'source_tasks_sha256': file_hash(task_path), 'checkpoint_sha256': file_hash(checkpoint),
              'architecture': saved['architecture'], 'encoder_settings': saved['encoder_settings'],
              'checkpoint_epoch': saved['epoch'], 'quantity': 'onboard_belief_not_truth',
              'channels': list(CHANNELS), 'normalization': NORMALIZATION,
              'counts': dict(counts), 'frame_counts': dict(frame_counts), 'episodes': cache_entries,
              'qa_sha256': {split: file_hash(destination / f'{split}.jsonl') for split in ('train', 'validation')},
              'test_exported': False, 'uses_evaluation_only': False,
              'codec_selection': 'explicit checkpoint, not automatic acceptance or a safety guarantee',
              'roi_selection': 'specified original task ROI, not learned LLM region selection'}
    json_write(destination / 'manifest.json', result)
    write_alignment_card(destination, result)
    return result


def write_alignment_card(destination, report):
    text = f'''# UAV机载地图编码对齐数据 v1

本目录只准备区域问答和冻结地图tokens，没有训练LLM或投影器。
训练问答{report['counts'].get('train', 0)}条，验证问答{report['counts'].get('validation', 0)}条；不导出test。
每道原始任务派生当前ROI六通道还原和同ROI时序比较两类问答，不标注隐藏源位置或移动。

先看samples/current_region.json与samples/temporal_region.json。inputs和question是模型可见部分，answer只作监督，禁止拼入问题。
tokens/按episode缓存各帧独立编码，只含tokens与时间，不含原图、重建、标准答案或隐藏源。
取样器按frame_indices取因果窗口；缓存含后续帧不意味着模型能看到，禁止把整条轨迹直接输入当前决策。
缺失历史位为-1／valid=false，必须屏蔽；实际offset_seconds不能用等间隔假设替代。
当前ROI固定来自原始题，不能称为LLM已会自主选区域；六通道答案是机载估计，不是完整环境真值。
字段物理单位为dBm、dB、秒、米；未知功率／年龄是null，不能解释为安全。

manifest记录编码器及源数据哈希；不同编码器缓存不能混用。train/validation按原场景组保持，不能把两类同源题拆到不同组。
本轮显式固定原v2只为接口基线，未消除功率变化误报；不把数据生成成功当作时序保真或LLM理解成功。
原始机载数据不改动，所有答案仅来自当时原图；evaluation_only和test轨迹没有读取。
'''
    with (destination / 'DATASET_CARD.md').open('x', encoding='utf-8') as stream:
        stream.write(text)


class AlignmentExamples:
    """读取监督行，模型输入通过model_input独立取出，不能包含标准答案。"""

    def __init__(self, root, split, checkpoint_sha256=None):
        if split not in ('train', 'validation'):
            raise ValueError('alignment split must be train or validation')
        self.root, self.split = Path(root), split
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('schema_version') != 1 or self.manifest.get('stage') != 'alignment_data_only_no_llm_training':
            raise ValueError('unsupported alignment schema')
        if checkpoint_sha256 is not None and self.manifest['checkpoint_sha256'] != checkpoint_sha256:
            raise ValueError('encoder checkpoint mismatch')
        path = self.root / f'{split}.jsonl'
        if file_hash(path) != self.manifest['qa_sha256'][split]:
            raise ValueError('question data hash mismatch')
        with path.open(encoding='utf-8') as stream:
            self.rows = [json.loads(line) for line in stream]
        self.entries = {entry['path']: entry for entry in self.manifest['episodes']}
        self.cache = {}
        if len(self.rows) != self.manifest['counts'][split]:
            raise ValueError('question count mismatch')

    def __len__(self):
        return len(self.rows)

    def model_input(self, index):
        row = self.rows[index]
        record = row['inputs']
        entry = self.entries.get(record['token_path'])
        if (entry is None or entry['split'] != self.split or row['split'] != self.split
                or entry['episode_id'] != row['episode_id'] or entry['scenario_id'] != row['scenario_id']):
            raise ValueError('alignment split or episode mismatch')
        relative = entry['path']
        if relative not in self.cache:
            path = checked_path(self.root, relative)
            if path.parent != (self.root / 'tokens').resolve() or file_hash(path) != entry['sha256']:
                raise ValueError('token path or hash mismatch')
            with np.load(path, allow_pickle=False) as archive:
                self.cache[relative] = (archive['tokens'], archive['times_seconds'])
        encoded, times = self.cache[relative]
        indices, valid, offsets = record['frame_indices'], record['valid'], record['offset_seconds']
        if not len(indices) or len(indices) != len(valid) or len(indices) != len(offsets):
            raise ValueError('invalid window metadata')
        if (not valid[-1] or type(indices[-1]) is not int or not 0 <= indices[-1] < len(times)):
            raise ValueError('invalid current frame')
        anchor, previous = indices[-1], -1
        packed = np.zeros((len(indices), *encoded.shape[1:]), dtype=np.float32)
        for slot, (frame, available, offset) in enumerate(zip(indices, valid, offsets)):
            if type(available) is not bool or type(frame) is not int:
                raise ValueError('invalid frame index or validity')
            if not available:
                if frame != -1 or offset is not None:
                    raise ValueError('invalid missing history slot')
                continue
            if not previous < frame <= anchor or offset is None or not np.isclose(offset, times[frame] - times[anchor], atol=1e-6):
                raise ValueError('noncausal or inconsistent frame times')
            packed[slot] = encoded[frame]
            previous = frame
        if not np.isfinite(packed).all():
            raise ValueError('nonfinite map tokens')
        return {'tokens': torch.from_numpy(packed), 'valid': torch.tensor(valid, dtype=torch.bool),
                'offset_seconds': torch.tensor([offset if offset is not None else 0 for offset in offsets], dtype=torch.float32),
                'question': question_text(row['kind'], record)}

    def answer(self, index):
        return self.rows[index]['answer']


def audit_alignment_dataset(dataset, source, output):
    root, source = Path(dataset), Path(source)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if file_hash(source / 'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('source manifest mismatch')
    original = MapFrames(source, 'train').manifest
    if file_hash(source / 'temporal_tasks.jsonl') != manifest['source_tasks_sha256']:
        raise ValueError('source task hash mismatch')
    collection = CollectionSettings(**original['collection_settings'])
    source_entries = {entry['id']: entry for entry in original['episodes']}
    raw_tasks = {}
    with (source / 'temporal_tasks.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            task = json.loads(line)
            if task['split'] in ('train', 'validation'):
                key = (task['episode_id'], task['anchor'])
                if key in raw_tasks:
                    raise ValueError('duplicate source task key')
                raw_tasks[key] = task
    counts, scenes, seen = Counter(), defaultdict(set), set()
    for split in ('train', 'validation'):
        data = AlignmentExamples(root, split, manifest['checkpoint_sha256'])
        previous, arrays = None, None
        for index, row in enumerate(data.rows):
            data.model_input(index)
            entry = source_entries.get(row['episode_id'])
            if entry is None or entry['split'] != split or entry['scenario_id'] != row['scenario_id']:
                raise ValueError('source scenario or split mismatch')
            scenes[split].add(entry['scenario_id'])
            if previous != entry['id']:
                path = checked_path(source, entry['path'])
                if path.parent != (source / 'episodes').resolve() or file_hash(path) != entry['sha256']:
                    raise ValueError('source episode hash mismatch')
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in ('maps', 'times', 'positions', 'resolution_m')}
                public_path = path.with_suffix('.json')
                if file_hash(public_path) != entry['metadata_sha256']:
                    raise ValueError('source public metadata hash mismatch')
                altitude = json.loads(public_path.read_text(encoding='utf-8'))['aircraft']['altitude_m']
                previous = entry['id']
            anchor = row['inputs']['frame_indices'][-1]
            task = raw_tasks.get((entry['id'], anchor))
            if task is None:
                raise ValueError('unknown original task')
            validate_task(task, arrays['maps'], arrays['times'], collection)
            if row['inputs'] != input_record(entry, arrays, task, altitude):
                raise ValueError('input differs from original causal window')
            if row['kind'] not in ('current_region', 'temporal_region'):
                raise ValueError('unknown alignment task kind')
            expected = (current_region_target(arrays['maps'][anchor], task['roi_grid_xyxy'])
                        if row['kind'] == 'current_region' else task['target'])
            if row['answer'] != expected or row['question'] != question_text(row['kind'], row['inputs']):
                raise ValueError('question or answer differs from original onboard supervision')
            identity = (entry['id'], anchor, row['kind'])
            if identity in seen:
                raise ValueError('duplicate alignment example')
            seen.add(identity)
            counts[split] += 1
        expected_count = 2 * sum(entry['temporal_tasks'] for entry in source_entries.values() if entry['split'] == split)
        if counts[split] != expected_count:
            raise ValueError('incomplete source question coverage')
        del data
    if scenes['train'] & scenes['validation']:
        raise ValueError('scene leakage')
    report = {'passed': True, 'alignment_manifest_sha256': file_hash(root / 'manifest.json'),
              'source_manifest_sha256': manifest['source_manifest_sha256'], 'checkpoint_sha256': manifest['checkpoint_sha256'],
              'questions_checked': dict(counts), 'scenes_checked': {key: len(value) for key, value in scenes.items()},
              'answers_recomputed_from_onboard_frames': True, 'test_episodes_read': False, 'uses_evaluation_only': False}
    json_write(output, report)
    return report
