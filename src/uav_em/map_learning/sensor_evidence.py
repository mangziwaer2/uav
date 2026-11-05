"""独立机载点测／测向证据；不改地图编码，不提供源位置或未来观测。"""

from collections import Counter, OrderedDict
import json
import math
from pathlib import Path

import numpy as np

from .alignment_data import AlignmentExamples
from .data import file_hash, json_write
from .quality import checked_path


SCAN_FIELDS = ('time_seconds', 'position_m', 'peaks_deg_dbm', 'beamwidth_deg',
               'bearing_error_std_deg', 'calibration_correlation_seconds',
               'minimum_peak_separation_deg', 'detection_threshold_dbm')


def visible_observation(record):
    time = float(record['time_seconds'])
    position = [float(value) for value in record['position_m']]
    power = float(record['interference_dbm'])
    if len(position) != 2 or time < 0 or not all(math.isfinite(value) for value in [time, power, *position]):
        raise ValueError('invalid visible observation')
    scan = record.get('directional')
    if scan is not None:
        scan = {field: scan[field] for field in SCAN_FIELDS}
        if scan['time_seconds'] != time or not np.allclose(scan['position_m'], position, atol=1e-5, rtol=0):
            raise ValueError('directional time or position mismatch')
        if (not all(math.isfinite(scan[field]) for field in SCAN_FIELDS[3:])
                or not 0 < scan['beamwidth_deg'] <= 180 or not 0 <= scan['bearing_error_std_deg'] <= 45
                or scan['calibration_correlation_seconds'] < 0 or scan['minimum_peak_separation_deg'] <= 0
                or not math.isfinite(scan['detection_threshold_dbm'])):
            raise ValueError('invalid directional uncertainty metadata')
        for peak in scan['peaks_deg_dbm']:
            if (len(peak) != 2 or not all(math.isfinite(value) for value in peak)
                    or not 0 <= peak[0] < 360 or peak[1] < scan['detection_threshold_dbm']):
                raise ValueError('invalid angular peak')
    return {'time_seconds': time, 'position_m': position, 'interference_dbm': power, 'directional': scan}


def build_sensor_evidence(alignment_dataset, source, output):
    alignment, source, destination = Path(alignment_dataset), Path(source), Path(output)
    parent = json.loads((alignment / 'manifest.json').read_text(encoding='utf-8'))
    if file_hash(source / 'manifest.json') != parent['source_manifest_sha256']:
        raise ValueError('source manifest mismatch')
    original = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    originals = {entry['id']: entry for entry in original['episodes']}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'observations').mkdir()
    entries, counts = [], Counter()
    scenes = {'train': set(), 'validation': set()}
    for entry in parent['episodes']:
        source_entry = originals[entry['episode_id']]
        if (entry['split'] not in scenes or source_entry['split'] != entry['split']
                or source_entry['scenario_id'] != entry['scenario_id']):
            raise ValueError('sensor scenario split mismatch')
        path = checked_path(source, source_entry['path']).with_suffix('.json')
        if path.parent != (source / 'episodes').resolve() or file_hash(path) != source_entry['metadata_sha256']:
            raise ValueError('public observation metadata hash mismatch')
        public = json.loads(path.read_text(encoding='utf-8'))
        if public['split'] != entry['split'] or public['scenario_id'] != entry['scenario_id']:
            raise ValueError('public metadata split mismatch')
        records = [visible_observation(record) for record in public['observations']]
        token_path = checked_path(alignment, entry['path'])
        if file_hash(token_path) != entry['sha256']:
            raise ValueError('map token hash mismatch')
        with np.load(token_path, allow_pickle=False) as archive:
            times = archive['times_seconds']
        if len(records) != entry['frames'] or not np.array_equal(times, [record['time_seconds'] for record in records]):
            raise ValueError('sensor and map frame times differ')
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError('nonchronological sensor records')
        relative = f'observations/{entry["episode_id"]}.json'
        json_write(destination / relative, records)
        entries.append({'episode_id': entry['episode_id'], 'scenario_id': entry['scenario_id'],
                        'split': entry['split'], 'path': relative, 'sha256': file_hash(destination / relative),
                        'frames': len(records), 'source_metadata_sha256': source_entry['metadata_sha256']})
        scenes[entry['split']].add(entry['scenario_id'])
        counts['observations'] += len(records)
        counts['scans'] += sum(record['directional'] is not None for record in records)
        counts['detected_scans'] += sum(bool((record['directional'] or {}).get('peaks_deg_dbm')) for record in records)
    if scenes['train'] & scenes['validation']:
        raise ValueError('sensor scene leakage')
    manifest = {'schema_version': 1, 'quantity': 'visible_sensor_evidence_not_source_hypotheses',
                'alignment_manifest_sha256': file_hash(alignment / 'manifest.json'),
                'source_manifest_sha256': parent['source_manifest_sha256'],
                'checkpoint_sha256': parent['checkpoint_sha256'], 'episodes': entries,
                'counts': dict(counts), 'test_exported': False, 'uses_evaluation_only': False}
    json_write(destination / 'manifest.json', manifest)
    (destination / 'samples').mkdir()
    checked = {}
    for split in ('train', 'validation'):
        data = SensorAlignmentExamples(alignment, destination, split)
        sample_index = 0
        for index in range(len(data)):
            evidence = data.evidence(index)
            if not sample_index and evidence['detected_scans']:
                sample_index = index
        checked[split] = len(data)
        json_write(destination / 'samples' / f'{split}.json',
                   {'sample_id': data.rows[sample_index]['id'], 'sensor_evidence': data.evidence(sample_index),
                    'question': data.model_input(sample_index)['question']})
    json_write(destination / 'quality_report.json',
               {'passed': True, 'windows_checked': checked, 'source_public_metadata_hashes_verified': True,
                'map_times_verified': True, 'test_episodes_read': False, 'uses_evaluation_only': False,
                'manifest_sha256': file_hash(destination / 'manifest.json')})
    (destination / 'DATASET_CARD.md').write_text(
        '# UAV机载独立测向证据 v1\n\n'
        '配合interference_map_alignment_v1使用，不替代六通道地图，不需要重训地图编码器。\n'
        '只有train／validation的公开点测与测向；不包含源位置、源标识、反射点或未来脚本。\n'
        '每episode缓存完整观测，读取器只取当前及此前60秒；未采样／无扫描与扫描无角峰分别保留。\n'
        '方向以东为0度、逆时针、世界坐标；接收功率单位dBm。波束宽度不是置信区间。\n'
        '角峰可能来自反射，峰数不是源数，未检测不等于安全；不预计算源坐标。\n'
        '原始数据及地图token缓存保持不变，manifest绑定两者哈希。先看samples中的输入。\n', encoding='utf-8')
    return manifest


class SensorAlignmentExamples:
    """相同地图问答加独立的因果观测文本，不提供答案或源先验。"""

    def __init__(self, alignment_dataset, evidence_dataset, split, history_seconds=60.0):
        if not math.isfinite(history_seconds) or history_seconds <= 0:
            raise ValueError('sensor history_seconds must be positive and finite')
        self.maps = AlignmentExamples(alignment_dataset, split)
        self.rows, self.manifest = self.maps.rows, self.maps.manifest
        self.root, self.history_seconds = Path(evidence_dataset), history_seconds
        evidence = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if (evidence.get('schema_version') != 1
                or evidence.get('quantity') != 'visible_sensor_evidence_not_source_hypotheses'
                or evidence['alignment_manifest_sha256'] != file_hash(Path(alignment_dataset) / 'manifest.json')
                or evidence['checkpoint_sha256'] != self.manifest['checkpoint_sha256']):
            raise ValueError('sensor evidence and map alignment version mismatch')
        self.entries = {entry['episode_id']: entry for entry in evidence['episodes']}
        if len(self.entries) != len(evidence['episodes']):
            raise ValueError('duplicate sensor episode')
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.maps)

    def evidence(self, index):
        row = self.rows[index]
        entry = self.entries.get(row['episode_id'])
        if (entry is None or entry['split'] != self.maps.split or row['split'] != self.maps.split
                or entry['scenario_id'] != row['scenario_id']):
            raise ValueError('sensor split or episode mismatch')
        identity = row['episode_id']
        if identity not in self.cache:
            path = checked_path(self.root, entry['path'])
            if path.parent != (self.root / 'observations').resolve() or file_hash(path) != entry['sha256']:
                raise ValueError('sensor observation hash mismatch')
            records = json.loads(path.read_text(encoding='utf-8'))
            if len(records) != entry['frames']:
                raise ValueError('sensor frame count mismatch')
            self.cache[identity] = records
        self.cache.move_to_end(identity)
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        records = self.cache[identity]
        anchor = row['inputs']['frame_indices'][-1]
        if type(anchor) is not int or not 0 <= anchor < len(records):
            raise ValueError('invalid sensor anchor')
        current = records[anchor]['time_seconds']
        start = max(0, current - self.history_seconds)
        visible = []
        previous = -1.0
        older = 0
        for record in records[:anchor + 1]:
            record = visible_observation(record)
            time = record['time_seconds']
            if not previous < time <= current:
                raise ValueError('noncausal sensor time')
            previous = time
            if time < start:
                older += 1
                continue
            visible.append(record)
        for frame, valid, offset, position in zip(row['inputs']['frame_indices'], row['inputs']['valid'],
                                                 row['inputs']['offset_seconds'], row['inputs']['positions_m']):
            if valid and (not 0 <= frame <= anchor or not np.isclose(records[frame]['time_seconds'] - current, offset)
                          or not np.allclose(records[frame]['position_m'], position, atol=1e-4, rtol=0)):
                raise ValueError('sensor and map history mismatch')
        return {'history_seconds': self.history_seconds, 'start_seconds': start, 'current_seconds': current,
                'older_observations_excluded': older, 'selection': 'all_observations_in_causal_time_window',
                'azimuth_convention': 'east=0_deg; counterclockwise; world_frame',
                'detected_scans': sum(bool((record['directional'] or {}).get('peaks_deg_dbm')) for record in visible),
                'observations': visible,
                'meaning': 'received_at_vehicle; peaks_not_sources; reflections_possible; no_range; '
                           'no_detection_not_safe; beamwidth_not_confidence; nearby_errors_correlated'}

    def question(self, index):
        row = self.rows[index]
        from .alignment_data import question_text
        return (question_text(row['kind'], row['inputs']) + '\n独立点测与测向证据：'
                + json.dumps(compact_evidence(self.evidence(index)), ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False))

    def model_input(self, index):
        result = self.maps.model_input(index)
        result['question'] = self.question(index)
        return result

    def answer(self, index):
        return self.maps.answer(index)


def compact_evidence(evidence):
    settings, observations = [], []
    for record in evidence['observations']:
        scan = record['directional']
        settings_index, peaks = None, None
        if scan is not None:
            metadata = {field: scan[field] for field in SCAN_FIELDS[3:]}
            if metadata not in settings:
                settings.append(metadata)
            settings_index, peaks = settings.index(metadata), scan['peaks_deg_dbm']
        observations.append([record['time_seconds'], *record['position_m'], record['interference_dbm'],
                             settings_index, peaks])
    result = {key: value for key, value in evidence.items() if key not in ('observations', 'detected_scans')}
    result.update({'columns': ['time_seconds', 'x_m', 'y_m', 'power_dbm', 'scan_settings_index', 'peaks_deg_dbm'],
                   'scan_settings': settings, 'observations': observations,
                   'missing_scan': 'null settings and peaks = no scan record; [] peaks = scan with no detection'})
    return result
