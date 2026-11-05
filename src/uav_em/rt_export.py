"""离线射线追踪适配；不导入可选引擎，不将源参数交给在线决策。"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .environment import ElectromagneticWorld


def load_rt_settings(path):
    """离线源配置：米、dBm、Hz；仅支持经过几何检查的内置街谷场景。"""
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"scene_name", "origin_xy_m", "bandwidth_hz", "samples_per_tx", "max_depth", "seed", "transmitters"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("invalid RT settings schema")
    if settings["scene_name"] != "simple_street_canyon":
        raise ValueError("only the verified simple_street_canyon scene is supported")
    origin = np.asarray(settings["origin_xy_m"], dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("origin_xy_m must contain two finite metre coordinates")
    for name, minimum, maximum in (("samples_per_tx", 1, 100000000), ("max_depth", 0, 10), ("seed", 0, 2**31 - 1)):
        value = settings[name]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"invalid {name}")
    if (isinstance(settings["bandwidth_hz"], bool) or not np.isfinite(settings["bandwidth_hz"])
            or settings["bandwidth_hz"] <= 0):
        raise ValueError("bandwidth_hz must be finite and positive")
    sources = settings["transmitters"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 8:
        raise ValueError("one to eight independent transmitters required")
    names = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"name", "position_m", "power_dbm"}:
            raise ValueError("invalid transmitter schema")
        position = np.asarray(source["position_m"], dtype=float)
        if (not isinstance(source["name"], str) or not source["name"] or source["name"] in names
                or position.shape != (3,) or not np.isfinite(position).all() or position[2] <= 0
                or isinstance(source["power_dbm"], bool) or not np.isfinite(source["power_dbm"])
                or not -100 <= source["power_dbm"] <= 60):
            raise ValueError("invalid transmitter name, position or power")
        names.add(source["name"])
    return settings


def combine_received_watts(received_watts, background_dbm):
    """同频带独立源非相干叠加；输入[源,y,x]瓦，背景dBm只加一次。"""
    received = np.asarray(received_watts, dtype=float)
    if (received.ndim != 3 or min(received.shape) < 1 or not np.isfinite(received).all()
            or np.any(received < 0) or not np.isfinite(background_dbm)):
        raise ValueError("received_watts must be finite nonnegative [source,y,x] watts")
    total_mw = received.sum(axis=0) * 1000 + 10 ** (background_dbm / 10)
    powers = 10 * np.log10(total_mw)
    if not np.isfinite(powers).all() or np.any(powers < -300) or np.any(powers > 100):
        raise ValueError("total power outside supported finite dBm range")
    return powers


def rectangular_building(identifier, vertices, origin_xy_m):
    """只接收轴对齐矩形棱柱，不把任意网格包围盒冒充真实楼体。"""
    vertices = np.asarray(vertices, dtype=float)
    origin = np.asarray(origin_xy_m, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 8
            or not np.isfinite(vertices).all() or origin.shape != (2,) or not np.isfinite(origin).all()):
        raise ValueError("invalid prism vertices or scene origin")
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    if np.any(upper - lower <= 1e-5):
        raise ValueError("building must have nonzero width, length and height")
    at_lower = np.isclose(vertices, lower, atol=1e-4, rtol=0)
    at_upper = np.isclose(vertices, upper, atol=1e-4, rtol=0)
    corners = {tuple(row) for row in at_upper.tolist()}
    if not np.all(at_lower | at_upper) or len(corners) != 8:
        raise ValueError("only verified axis-aligned rectangular prisms are supported")
    minimum = lower[:2] - origin
    maximum = upper[:2] - origin
    return {"id": identifier,
            "footprint_m": [[minimum[0], minimum[1]], [maximum[0], minimum[1]],
                            [maximum[0], maximum[1]], [minimum[0], maximum[1]]],
            "base_m": float(lower[2]), "height_m": float(upper[2] - lower[2]),
            "height_source": "Sionna RT mesh, verified rectangular prism"}


def validate_cell_centers(centers, horizontal, vertical, altitude_m, origin_xy_m):
    """校验单元中心与节点一致，拒绝转置、翻转或半格偏移。"""
    expected_x, expected_y = np.meshgrid(horizontal, vertical)
    expected = np.stack([expected_x + origin_xy_m[0], expected_y + origin_xy_m[1],
                         np.full_like(expected_x, altitude_m)], axis=-1)
    values = np.asarray(centers, dtype=float)
    if values.shape != expected.shape or not np.allclose(values, expected, atol=1e-4, rtol=0):
        raise ValueError("radio map cell centers do not match project [y,x] grid coordinates")


def write_rt_bundle(directory, config, received_watts, buildings, provenance):
    """新目录保存总功率、仿真配置和仅离线使用的源清单，拒绝覆盖。"""
    directory = Path(directory)
    horizontal = np.arange(config.map.columns) * config.map.resolution_m
    vertical = np.arange(config.map.rows) * config.map.resolution_m
    received = np.asarray(received_watts, dtype=float)
    if received.ndim != 3 or received.shape[1:] != (len(vertical), len(horizontal)):
        raise ValueError("received map shape does not match project grid")
    powers = combine_received_watts(received, config.electromagnetic.background_dbm)
    grid = {"x_m": horizontal.tolist(), "y_m": vertical.tolist(), "power_dbm": powers.tolist(),
            "altitude_m": config.aircraft.altitude_m,
            "carrier_frequency_hz": config.electromagnetic.carrier_frequency_hz}
    public_map = {"source": provenance["scene_name"], "geometry_source": "verified engine mesh",
                  "license": provenance["license"], "scene_sha256": provenance["scene_sha256"]}
    config = replace(config, map=replace(config.map, buildings=tuple(buildings), provenance=public_map),
                     sensor=replace(config.sensor, mode="power_only"),
                     electromagnetic=replace(config.electromagnetic, fields=(), power_grid=grid))
    config.validate()
    ElectromagneticWorld(config)
    value = asdict(config)
    value["electromagnetic"]["power_grid"] = {"path": "power_grid.npz", "provenance": {
        "generator": "Sionna RT", "engine_version": provenance["engine_version"],
        "quantity": "total_received_interference_power_dbm", "time_model": "static",
        "spatial_model": "radio-map cell mean at grid nodes, linear-power interpolation"}}
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "power_grid.npz", **grid)
    np.savez_compressed(directory / "source_received_watts.npz", received_watts=received)
    (directory / "simulation.json").write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                                         allow_nan=False), encoding="utf-8")
    metadata = {**provenance, "background_dbm": config.electromagnetic.background_dbm,
                "power_range_dbm": [float(powers.min()), float(powers.max())],
                "zero_signal_cells_per_source": (received == 0).sum(axis=(1, 2)).tolist(),
                "power_grid_sha256": hashlib.sha256((directory / "power_grid.npz").read_bytes()).hexdigest(),
                "limitations": ["simulation truth, not field measurements or full-wave EM",
                                "static single-height, co-channel independent sources",
                                "cell-averaged received power, not coherent pointwise fading",
                                "zero signal may include missed finite-sample rays; background is not proof of safety",
                                "power-only local sensing; no synthetic AoA or spatial disk scan",
                                "source fitting uses a different approximate forward model"]}
    (directory / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                       allow_nan=False), encoding="utf-8")
    load_config(directory / "simulation.json")
    return directory / "simulation.json"
