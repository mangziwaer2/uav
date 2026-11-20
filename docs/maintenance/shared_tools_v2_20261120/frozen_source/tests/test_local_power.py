from dataclasses import replace
import pytest
from uav_em.config import BeliefConfig, MapConfig, ProjectConfig, load_config


def small_config():
    base = ProjectConfig()
    return replace(base, map=MapConfig(columns=15, rows=11, start_m=(20, 100), goal_m=(260, 100), obstacles_m=()),
                   sensor=replace(base.sensor, mode="power_only"),
                   electromagnetic=replace(base.electromagnetic, fields=(), measurement_noise_db=0.),
                   belief=replace(base.belief, estimation_method="local_gp", support_radius_m=80.))


@pytest.mark.parametrize("field,value", [("estimation_method", "oracle"), ("gp_length_scale_m", 0.),
                                         ("gp_time_scale_seconds", -1.), ("gp_max_points", 257),
                                         ("gp_max_points", True)])
def test_gp_configuration_rejects_invalid_parameters(field, value):
    config = small_config()
    with pytest.raises(ValueError):
        replace(config, belief=replace(config.belief, **{field: value})).validate()


def test_current_default_profile_is_explicit():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/local_power.json")
    assert config.sensor.mode == "power_only"
    assert config.belief.estimation_method == "local_gp"

