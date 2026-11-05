import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from uav_em.map_learning.data import CollectionSettings, generate_dataset
from uav_em.map_learning.preview import preview_episode
from uav_em.map_learning.scenarios import ScenarioSettings


@pytest.fixture
def preview_dataset(tmp_path):
    root = tmp_path / 'data'
    manifest = generate_dataset(root, 3, 30,
                                ScenarioSettings(columns=10, rows=8, building_count=0, max_sources=0,
                                                 horizon_seconds=6, require_goal_reachable_in_time=False),
                                CollectionSettings(budget=3, task_stride=1, window_spacing_seconds=2))
    return root, manifest['episodes'][0]


def test_preview_is_causal_and_never_reads_truth(preview_dataset, tmp_path, monkeypatch):
    root, entry = preview_dataset
    original_read = Path.read_text

    def visible_only(path, *args, **kwargs):
        assert 'evaluation_only' not in path.parts
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', visible_only)
    output = preview_episode(root, entry['id'], tmp_path / 'preview', animate=False)
    sample = json.loads((output / 'sample.json').read_text(encoding='utf-8'))
    task = json.loads((output / 'temporal_task.json').read_text(encoding='utf-8'))
    assert sample['uses_evaluation_only'] is False
    assert sample['is_model_reconstruction'] is False
    assert all(index <= task['anchor'] for index in task['window']['indices'])
    assert any(index >= 0 for index in task['window']['indices'])
    for filename in ('channels.png', 'window.png', 'temporal_task.png'):
        with Image.open(output / filename) as image:
            assert image.width > 100 and image.height > 100
    assert not (output / 'replay.gif').exists()
    with (output / 'frame_grid.csv').open(encoding='utf-8-sig', newline='') as stream:
        cells = list(csv.DictReader(stream))
    assert len(cells) == 80
    assert all(cell['power_dbm'] == '' and cell['age_seconds'] == '' for cell in cells if cell['supported'] == '0')
    assert (output / 'observations.csv').is_file()
    with pytest.raises(FileExistsError):
        preview_episode(root, entry['id'], output, animate=False)


def test_preview_gif_keeps_simulation_time_and_final_pause(preview_dataset, tmp_path):
    root, entry = preview_dataset
    output = preview_episode(root, entry['id'], tmp_path / 'preview', stride=2)
    sample = json.loads((output / 'sample.json').read_text(encoding='utf-8'))
    durations = []
    with Image.open(output / 'replay.gif') as animation:
        for frame in range(animation.n_frames):
            animation.seek(frame)
            durations.append(animation.info['duration'])
    assert len(durations) >= 2
    assert abs(sum(durations) - sample['times_seconds'][-1] * 1000 - 2000) <= 20
    assert durations[-1] == 2000


def test_preview_checks_parameters_and_episode_hash(preview_dataset, tmp_path):
    root, entry = preview_dataset
    with pytest.raises(ValueError, match='positive'):
        preview_episode(root, entry['id'], tmp_path / 'bad', stride=0)
    with pytest.raises(ValueError, match='unknown episode'):
        preview_episode(root, 'missing', tmp_path / 'bad')
    with (root / entry['path']).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        preview_episode(root, entry['id'], tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
