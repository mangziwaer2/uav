from dataclasses import replace
import json

import numpy as np
import pytest

from uav_em.advisor import build_prompt
from uav_em.belief import ElectromagneticBelief
from uav_em.config import ProjectConfig, MapConfig, BeliefConfig
from uav_em.context import build_context
from uav_em.environment import Geometry, Measurement


def config():
    base = ProjectConfig()
    return replace(base, map=MapConfig(obstacles_m=()),
                   sensor=replace(base.sensor, mode="power_only"),
                   belief=BeliefConfig(estimation_method="local_gp", support_radius_m=80.))


def context_at(belief, time=5.):
    cfg = belief.config
    return build_context((40., 300.), cfg.map.goal_m, [[40., 300.], [200., 300.]],
                         belief.snapshot(time), cfg, 100., [], [], None)


def test_nearest_point_age_is_not_replaced_by_latest_point_age():
    cfg = config()
    belief = ElectromagneticBelief(cfg, Geometry(cfg.map))
    belief.update(Measurement((40., 300.), 0., -70.))
    belief.update(Measurement((60., 300.), 5., -80.))
    ctx = context_at(belief)
    estimate = ctx["electromagnetic_estimate"]
    columns = estimate["sample_columns"]
    local = [dict(zip(columns, sample)) for sample in estimate["local_samples"]]
    center = next(s for s in local if s["x_m"] == 40 and s["y_m"] == 300)
    assert center["nearest_support_age_seconds"] == 5.
    assert center["nearest_support_distance_m"] == 0.
    assert center["local_training_point_count"] == 2
    assert estimate["estimator"]["support_count_is_not_independent_sample_count"]
    assert all(len(sample) == len(columns) for sample in estimate["coarse_samples"] + estimate["local_samples"])
    assert ctx["measurement_evidence"]["recent_points"][-1]["interference_dbm"] == -80.


def test_no_new_observation_decay_is_not_reported_as_measured_environment_change():
    cfg = config()
    belief = ElectromagneticBelief(cfg, Geometry(cfg.map))
    belief.update(Measurement((40., 300.), 0., -60.))
    first, later = context_at(belief, 0.), context_at(belief, 20.)
    assert later["measurement_evidence"]["recent_points"] == first["measurement_evidence"]["recent_points"]
    assert later["electromagnetic_estimate"]["current_grid_estimate_dbm"] < first["electromagnetic_estimate"]["current_grid_estimate_dbm"]
    assert later["electromagnetic_estimate"]["temporal_decay_is_not_a_new_measurement"]
    assert later["measurement_evidence"]["along_track_change_may_be_spatial_or_temporal"]
    assert later["route"]["field_evaluation_time_seconds"] == 20.
    assert not later["route"]["field_at_arrival_times_predicted"]
    prompt = build_prompt(later)
    assert "不表示测到干扰下降或源关闭" in prompt


def test_unknown_cells_and_expired_evidence_are_not_low_power_observations():
    cfg = config()
    belief = ElectromagneticBelief(cfg, Geometry(cfg.map))
    belief.update(Measurement((40., 300.), 0., -60.))
    old = context_at(belief, 36.)
    assert not old["measurement_evidence"]["recent_points"]
    est = old["electromagnetic_estimate"]
    for sample in est["coarse_samples"] + est["local_samples"]:
        item = dict(zip(est["sample_columns"], sample))
        assert item["estimate_dbm"] is None
        assert item["nearest_support_distance_m"] is None
        assert item["nearest_support_age_seconds"] is None
        assert item["local_training_point_count"] == 0
    assert not context_at(belief, -1.)["measurement_evidence"]["recent_points"]


def test_same_evidence_same_context_despite_different_hidden_emitters():
    cfg = config()
    changed = replace(cfg, electromagnetic=replace(cfg.electromagnetic, fields=()))
    contexts = []
    for setting in (cfg, changed):
        belief = ElectromagneticBelief(setting, Geometry(setting.map))
        for t in range(16):
            belief.update(Measurement((40. + 3 * t, 300.), float(t), -90. + t / 2))
        contexts.append(context_at(belief, 15.))
    assert contexts[0] == contexts[1]
    evidence = contexts[0]["measurement_evidence"]["recent_points"]
    assert len(evidence) == 8 and evidence[0]["time_seconds"] == 0 and evidence[-1]["time_seconds"] == 15
    for item in evidence:
        assert set(item) == {"position_m", "time_seconds", "interference_dbm"}
    assert not contexts[0]["directional_sensing"]["available"]
    json.dumps(contexts[0], allow_nan=False)


def test_main_geometry_is_public_and_local_samples_skip_blocked_cells():
    cfg = config()
    building = {"id": "wall", "footprint_m": [[60, 280], [80, 280], [80, 320], [60, 320]], "height_m": 45.}
    cfg = replace(cfg, map=replace(cfg.map, geometry_mode="fixed_altitude", buildings=(building,)))
    geometry = Geometry(cfg.map, cfg.aircraft.clearance_m, cfg.aircraft.altitude_m, cfg.aircraft.vertical_clearance_m)
    belief = ElectromagneticBelief(cfg, geometry)
    belief.update(Measurement((40., 300.), 0., -70.))
    ctx = context_at(belief)
    assert ctx["map"]["known_buildings"][0]["id"] == "wall"
    assert all(geometry.free(s[:2]) for s in ctx["electromagnetic_estimate"]["local_samples"])
    assert ctx["information_sources"]["map"] == "known_geometry_prior"
