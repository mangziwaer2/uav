from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_em.config import ProjectConfig
from uav_em.simulation import simulate
from uav_em.environment import ElectromagneticWorld
from uav_em.visualization import (animation_timeline, config_from_result, decision_caption, interference_region_masks, render,
                                  strong_measurement_positions)


@pytest.fixture
def result():
    config = ProjectConfig()
    config = replace(config,
                     map=replace(config.map, columns=8, rows=6, resolution_m=20,
                                 start_m=(20, 40), goal_m=(120, 40), obstacles_m=()),
                     simulation=replace(config.simulation, max_seconds=20),
                     electromagnetic=replace(config.electromagnetic, fields=(), measurement_noise_db=0))
    return simulate(config)


def test_summary_renders_synchronized_truth_and_estimate_without_files(result, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    rendered = []

    def capture(figure, path, **kwargs):
        first, second = figure.axes[:2]
        assert first.get_xlim() == second.get_xlim()
        assert first.get_ylim() == second.get_ylim()
        assert first.images[0].get_array().shape == second.images[0].get_array().shape
        assert first.images[0].get_clim() == second.images[0].get_clim()
        assert "interference truth" in first.get_title()
        assert all(line.get_marker() != "^" for axis in (first, second) for line in axis.lines)
        assert all(text.get_position()[1] > 0.8 for text in figure.texts)
        assert "Last decision:" not in figure.texts[-1].get_text()
        assert figure.texts[-1].get_text()
        rendered.append(path)

    monkeypatch.setattr(Figure, "savefig", capture)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    render(result, "unused.png")
    assert rendered == [Path("unused.png")]


def test_animation_updates_all_selected_frames(result, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import uav_em.visualization

    updates = []

    def capture(figure, update, indexes, durations, output):
        assert len(indexes) == len(durations)
        assert all(duration >= 20 for duration in durations)
        for index in indexes:
            update(index)
            updates.append(index)

    monkeypatch.setattr(uav_em.visualization, "save_gif", capture)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    render(result, "unused.gif", animate=True, frame_stride=3)
    assert updates[0] == 0 and updates[-1] == len(result["frames"]) - 1


def test_timeline_preserves_decisions_and_simulation_time_despite_stride():
    result = {"frames": [{"time_seconds": float(index)} for index in range(8)],
              "decisions": [{"time_seconds": 0}, {"time_seconds": 3}]}
    indexes, durations = animation_timeline(result, 4, 1, 2)
    assert indexes == [0, 3, 4, 7]
    assert durations == [5000, 3000, 3000, 2000]
    _, faster = animation_timeline(result, 4, 2, 0)
    assert faster == [1500, 500, 1500, 2000]


def test_gif_saves_per_frame_durations(monkeypatch):
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageSequence
    from uav_em.visualization import save_gif

    figure = plt.figure(figsize=(1, 1), dpi=30)
    buffer = BytesIO()
    original_save = Image.Image.save

    def capture(image, output, **kwargs):
        original_save(image, buffer, format="GIF", **kwargs)

    monkeypatch.setattr(Image.Image, "save", capture)

    def update(index):
        figure.set_facecolor(("red", "blue")[index])

    try:
        save_gif(figure, update, [0, 1], [1500, 3200], Path("unused.gif"))
        buffer.seek(0)
        with Image.open(buffer) as image:
            assert image.n_frames == 2
            assert [frame.info["duration"] for frame in ImageSequence.Iterator(image)] == [1500, 3200]
    finally:
        buffer.close()
        plt.close(figure)


@pytest.mark.parametrize("stride,speed,pause", [
    (0, 1, 2), (True, 1, 2), (1, 0, 2), (1, float("nan"), 2),
    (1, True, 2), (1, 1, -1), (1, 1, float("inf")),
])
def test_timeline_rejects_invalid_playback_parameters(stride, speed, pause):
    with pytest.raises(ValueError):
        animation_timeline({"frames": [{"time_seconds": 0}]}, stride, speed, pause)


def test_caption_uses_actual_fallback_and_past_measurements_not_future_decisions():
    result = {"decisions": [
        {"time_seconds": 0, "accepted_action": "replan", "backend": "local", "trigger": "initial",
         "error": "invalid JSON", "decision": {"reason": "fallback"},
         "stage_plan_after": {"status": "suspended"}},
        {"time_seconds": 10, "accepted_action": "probe", "backend": "local", "trigger": "scheduled",
         "decision": {"reason": "FUTURE_SENTINEL"}},
    ], "events": [
        {"time_seconds": 0, "event": "observation", "interference_dbm": -95},
        {"time_seconds": 10, "event": "observation", "interference_dbm": -30},
    ]}
    caption = "\n".join(decision_caption(result, 1))
    assert "模型响应失败" in caption and "阶段计划已暂停" in caption
    assert "-95.0 dBm" in caption
    assert "FUTURE_SENTINEL" not in caption and "-30.0 dBm" not in caption


def test_replay_restores_exact_config_and_rejects_invalid_stride(result):
    assert config_from_result(result).to_dict() == result["evaluation_only_config"]
    with pytest.raises(ValueError):
        render(result, "unused.png", frame_stride=0)


def test_old_quality_log_is_not_reinterpreted_as_interference(result):
    with pytest.raises(ValueError, match="schema v2"):
        config_from_result({**result, "schema_version": 1})


def test_region_overlap_does_not_turn_unknown_or_low_estimate_into_detection():
    truth = [[-60, -60, -95], [-60, -95, -60]]
    estimate = [[-60, -95, -60], [-60, -60, -60]]
    support = [[True, True, True], [False, False, True]]
    masks = interference_region_masks(truth, estimate, support, -80)
    np.testing.assert_array_equal(masks["truth"], [[True, True, False], [True, False, True]])
    np.testing.assert_array_equal(masks["inferred"], [[True, False, True], [False, False, True]])
    np.testing.assert_array_equal(masks["overlap"], [[True, False, False], [False, False, True]])


def test_region_mask_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="matching 2d"):
        interference_region_masks([[0]], [[0, 0]], [[True]], -80)


def test_strong_samples_use_latest_valid_measured_values_not_truth(result):
    config = config_from_result(result)
    geometry = ElectromagneticWorld(config).geometry
    events = [
        {"event": "observation", "position_m": [20, 40], "time_seconds": 0, "interference_dbm": -50},
        {"event": "observation", "position_m": [20, 40], "time_seconds": 3, "interference_dbm": -95},
        {"event": "observation", "position_m": [40, 40], "time_seconds": 3, "interference_dbm": -60},
        {"event": "observation", "position_m": [60, 40], "time_seconds": 8, "interference_dbm": -50}]
    assert strong_measurement_positions({"events": events}, geometry, 4, 5, -80) == [[40, 40]]
    assert strong_measurement_positions({"events": events}, geometry, 20, 5, -80) == []


def test_dynamic_source_and_hatch_render_only_in_authorized_panels(result, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import uav_em.visualization

    source = {"center_m": [60, 40], "sigma_m": 20, "peak_dbm": -50,
              "start_seconds": 2, "end_seconds": 6, "velocity_mps": [5, 0]}
    result["evaluation_only_config"]["electromagnetic"]["fields"] = [source]
    for frame in result["frames"]:
        frame["estimate_dbm"] = np.full(np.shape(frame["supported"]), -60).tolist()
    original = [np.array(frame["estimate_dbm"]) for frame in result["frames"]]
    visited = []

    def capture(figure, update, indexes, durations, output):
        for index in indexes:
            update(index)
            now = result["frames"][index]["time_seconds"]
            first, second = figure.axes[:2]
            labels = [text.get_text() for text in first.texts]
            assert ("S1" in labels) == (2 <= now < 6)
            assert not second.texts
            np.testing.assert_array_equal(second.images[0].get_array(), original[index])
            assert second.collections
            visited.append(index)

    monkeypatch.setattr(uav_em.visualization, "save_gif", capture)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    render(result, "unused.gif", animate=True, frame_stride=1)
    assert len(visited) == len(result["frames"])
