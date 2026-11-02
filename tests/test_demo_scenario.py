from pathlib import Path

import numpy as np

from uav_em.config import ProjectConfig, load_config
from uav_em.environment import ElectromagneticWorld


def test_default_and_file_use_the_same_localized_interference_fields():
    config = load_config(Path(__file__).resolve().parent.parent / "configs/demo.json")
    assert list(config.electromagnetic.fields) == list(ProjectConfig().electromagnetic.fields)
    assert [field["eirp_dbm"] for field in config.electromagnetic.fields] == [-2, -5]
    assert config.sensor == ProjectConfig().sensor


def test_initial_demo_does_not_force_start_or_goal_into_strong_interference():
    config = ProjectConfig()
    world = ElectromagneticWorld(config)
    for now in np.arange(0, config.simulation.max_seconds + 1, 3):
        endpoints = world.truth([config.map.start_m, config.map.goal_m], now)
        assert np.all(endpoints < config.electromagnetic.interference_reference_dbm - 10)
        coverage = np.mean(world.truth(world.geometry.grid(), now)
                           >= config.electromagnetic.interference_reference_dbm)
        assert coverage < 0.15


def test_demo_has_a_sampled_low_interference_bypass_through_known_geometry():
    """仅离线用真值检查场景可绕；此折线不会作为策略输入或规划初解。"""
    config = ProjectConfig()
    world = ElectromagneticWorld(config)
    bypass = [config.map.start_m, [200, 180], [480, 180], [480, 60],
              [620, 60], [760, 180], config.map.goal_m]
    points = []
    for start, end in zip(bypass, bypass[1:]):
        assert world.geometry.edge_free(start, end)
        count = int(np.ceil(np.linalg.norm(np.asarray(end) - start) / 5)) + 1
        points.extend(np.linspace(start, end, count))
    for now in np.arange(0, config.simulation.max_seconds + 1, 3):
        assert np.all(world.truth(points, now) < config.electromagnetic.interference_reference_dbm - 5)
