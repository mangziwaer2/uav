import json

import pytest

from uav_em.map_learning.data import generate_dataset, CollectionSettings, file_hash
from uav_em.map_learning.scenarios import ScenarioSettings
from uav_em.map_learning.quality import audit_dataset
from uav_em.map_learning.partition import stratified_splits, repartition_dataset


def build_dataset(path, workers=1):
    return generate_dataset(path, 3, 810,
                            ScenarioSettings(columns=10, rows=8, building_count=0, max_sources=0,
                                             horizon_seconds=8, require_goal_reachable_in_time=False),
                            CollectionSettings(budget=3, task_stride=1, window_spacing_seconds=2), workers)


def test_complete_audit_replays_all_frames_and_targets(tmp_path):
    root = tmp_path / 'dataset'
    manifest = build_dataset(root)
    report = audit_dataset(root, root / 'quality_report.json')
    assert report['passed'], report['errors']
    assert report['episodes_checked'] == 9
    assert sum(value['replayed_frames'] for value in report['split_statistics'].values()) == sum(
        episode['frames'] for episode in manifest['episodes'])
    strict = audit_dataset(root, root / 'strict.json', require_coverage=True)
    assert not strict['passed']
    assert any('missing sources_' in error for error in strict['errors'])


def test_modified_measurements_rejected_even_if_metadata_hash_updated(tmp_path):
    root = tmp_path / 'dataset'
    manifest = build_dataset(root)
    entry = manifest['episodes'][0]
    path = (root / entry['path']).with_suffix('.json')
    public = json.loads(path.read_text(encoding='utf-8'))
    public['observations'][0]['interference_dbm'] += 30
    path.write_text(json.dumps(public), encoding='utf-8')
    entry['metadata_sha256'] = file_hash(path)
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    report = audit_dataset(root, root / 'audit.json')
    assert not report['passed']
    assert any('differs from causal measurement replay' in error for error in report['errors'])


def test_changed_tasks_and_files_cannot_pass_hash_check(tmp_path):
    root = tmp_path / 'dataset'
    manifest = build_dataset(root)
    with (root / 'temporal_tasks.jsonl').open('a', encoding='utf-8') as stream:
        stream.write('{}\n')
    path = root / manifest['episodes'][0]['path']
    with path.open('ab') as stream:
        stream.write(b'changed')
    report = audit_dataset(root, root / 'audit.json')
    assert not report['passed']
    assert any('hash mismatch' in error for error in report['errors'])


def test_parallel_and_serial_generation_have_same_data_and_split(tmp_path):
    serial = build_dataset(tmp_path / 'serial')
    parallel = build_dataset(tmp_path / 'parallel', workers=2)
    assert serial['scenarios'] == parallel['scenarios']
    assert serial['episodes'] == parallel['episodes']
    assert serial['temporal_tasks_sha256'] == parallel['temporal_tasks_sha256']


def test_stratification_is_deterministic_and_keeps_all_source_counts():
    counts = {f'scene_{index}': index % 4 for index in range(120)}
    splits = stratified_splits(counts, 7)
    assert splits == stratified_splits(counts, 7)
    assert [sum(value == split for value in splits.values()) for split in ('train', 'validation', 'test')] == [84, 18, 18]
    for split in ('train', 'validation', 'test'):
        assert {counts[scene] for scene in splits if splits[scene] == split} == {0, 1, 2, 3}
    with pytest.raises(ValueError, match='not enough scenes'):
        stratified_splits({'scene_0': 0, 'scene_1': 1, 'scene_2': 1}, 7)


def test_repartition_preserves_arrays_original_files_and_causal_labels(tmp_path):
    root = tmp_path / 'original'
    original = build_dataset(root)
    before = file_hash(root / 'manifest.json')
    destination = tmp_path / 'stratified'
    manifest = repartition_dataset(root, destination)
    assert file_hash(root / 'manifest.json') == before
    assert manifest['partition_provenance']['input_manifest_sha256'] == before
    assert [entry['sha256'] for entry in manifest['episodes']] == [entry['sha256'] for entry in original['episodes']]
    report = audit_dataset(destination, destination / 'quality_report.json')
    assert report['passed'], report['errors']
    with pytest.raises(FileExistsError):
        repartition_dataset(root, destination)
