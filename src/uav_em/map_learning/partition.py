"""离线按源数分层，整场景划分；另写新目录，不改采集数据或原始证据。"""

from collections import defaultdict
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from .data import file_hash, json_write
from .quality import checked_path


def stratified_splits(source_counts, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('split seed must be a nonnegative integer')
    groups = defaultdict(list)
    for scene_id, count in sorted(source_counts.items()):
        groups[count].append(scene_id)
    holdout = max(1, int(len(source_counts) * 0.15))
    if not groups or any(len(group) < 3 for group in groups.values()) or holdout < len(groups):
        raise ValueError('not enough scenes for each source-count stratum in all three splits')
    ideal = {count: len(group) * holdout / len(source_counts) for count, group in groups.items()}
    allocated = {count: max(1, int(ideal[count])) for count in groups}
    if sum(allocated.values()) > holdout:
        raise ValueError('not enough holdout scenes for stratification')
    while sum(allocated.values()) < holdout:
        eligible = [count for count in groups if 2 * (allocated[count] + 1) < len(groups[count])]
        if not eligible:
            raise ValueError('stratification would empty a training stratum')
        count = max(eligible, key=lambda value: (ideal[value] - allocated[value], -value))
        allocated[count] += 1
    random = np.random.default_rng(seed)
    result = {}
    for count, group in sorted(groups.items()):
        shuffled = random.permutation(group).tolist()
        size = allocated[count]
        for index, scene_id in enumerate(shuffled):
            result[scene_id] = 'test' if index < size else 'validation' if index < 2 * size else 'train'
    return result


def repartition_dataset(dataset, output, seed=7):
    root, destination = Path(dataset), Path(output)
    original = root / 'manifest.json'
    manifest = copy.deepcopy(json.loads(original.read_text(encoding='utf-8')))
    if manifest.get('schema_version') != 1 or manifest.get('split_unit') != 'scenario':
        raise ValueError('unsupported dataset protocol')
    counts = {}
    for scene in manifest['scenarios']:
        report = json.loads(checked_path(root, f'evaluation_only/{scene["id"]}_report.json').read_text(encoding='utf-8'))
        counts[scene['id']] = report['source_count']
    if set(counts.values()) != set(range(manifest['scenario_settings']['max_sources'] + 1)):
        raise ValueError('dataset is missing a configured source-count class')
    splits = stratified_splits(counts, seed)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'episodes').mkdir()
    (destination / 'evaluation_only').mkdir()
    for scene in manifest['scenarios']:
        scene['split'] = splits[scene['id']]
        for suffix in ('.json', '_report.json'):
            relative = f'evaluation_only/{scene["id"]}{suffix}'
            shutil.copy2(checked_path(root, relative), checked_path(destination, relative))
    episode_splits = {}
    for entry in manifest['episodes']:
        path = checked_path(root, entry['path'])
        public_path = path.with_suffix('.json')
        if file_hash(path) != entry['sha256'] or file_hash(public_path) != entry['metadata_sha256']:
            raise ValueError('source episode hash mismatch')
        public = json.loads(public_path.read_text(encoding='utf-8'))
        split = splits[entry['scenario_id']]
        entry['split'] = public['split'] = split
        episode_splits[entry['id']] = split
        target = checked_path(destination, entry['path'])
        shutil.copy2(path, target)
        json_write(target.with_suffix('.json'), public)
        entry['metadata_sha256'] = file_hash(target.with_suffix('.json'))
    if file_hash(root / 'temporal_tasks.jsonl') != manifest['temporal_tasks_sha256']:
        raise ValueError('source temporal tasks hash mismatch')
    with (root / 'temporal_tasks.jsonl').open(encoding='utf-8') as source:
        with (destination / 'temporal_tasks.jsonl').open('x', encoding='utf-8') as target:
            for line in source:
                task = json.loads(line)
                task['split'] = episode_splits[task['episode_id']]
                target.write(json.dumps(task, ensure_ascii=False, allow_nan=False) + '\n')
    manifest['temporal_tasks_sha256'] = file_hash(destination / 'temporal_tasks.jsonl')
    manifest['partition_provenance'] = {'method': 'source_count_stratified_scenario', 'seed': seed,
                                      'input_manifest_sha256': file_hash(original),
                                      'partition_source_sha256': file_hash(Path(__file__)),
                                      'uses_evaluation_only_for_split_not_actor': True}
    json_write(destination / 'manifest.json', manifest)
    return manifest
