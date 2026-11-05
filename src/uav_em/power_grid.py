"""外部静态总接收干扰功率栅格；只供真值采样，不伪造方向或源参数。"""

import json
from pathlib import Path

import numpy as np


def load_power_grid(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            required = {"x_m", "y_m", "power_dbm", "altitude_m", "carrier_frequency_hz"}
            if set(archive.files) != required:
                raise ValueError("NPZ power grid requires x_m/y_m/power_dbm/altitude_m/carrier_frequency_hz only")
            return {name: archive[name].tolist() for name in required}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_power_grid(value, config):
    required = {"x_m", "y_m", "power_dbm", "altitude_m", "carrier_frequency_hz"}
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - required - {"provenance"}:
        raise ValueError("invalid total received power grid schema")
    for name in required:
        array = np.asarray(value[name])
        if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            raise ValueError("power grid must contain finite numeric values, not masks or nodata")
    horizontal, vertical, powers = (np.asarray(value[name], dtype=float) for name in ("x_m", "y_m", "power_dbm"))
    if (horizontal.ndim != 1 or vertical.ndim != 1 or min(horizontal.size, vertical.size) < 2
            or np.any(np.diff(horizontal) <= 0) or np.any(np.diff(vertical) <= 0)
            or powers.shape != (vertical.size, horizontal.size)):
        raise ValueError("power grid requires increasing axes and power_dbm[y,x]")
    if np.any(powers < -300) or np.any(powers > 100):
        raise ValueError("power grid dBm outside supported [-300,100] range")
    for name, expected in (("altitude_m", config.aircraft.altitude_m),
                           ("carrier_frequency_hz", config.electromagnetic.carrier_frequency_hz)):
        if np.ndim(value[name]) != 0 or not np.isclose(value[name], expected, rtol=0, atol=1e-6):
            raise ValueError(f"power grid {name} must match the simulation")
    maximum = (np.array([config.map.columns, config.map.rows]) - 1) * config.map.resolution_m
    if horizontal[0] > 0 or vertical[0] > 0 or horizontal[-1] < maximum[0] or vertical[-1] < maximum[1]:
        raise ValueError("power grid must cover the full simulation domain; extrapolation is forbidden")
    if config.sensor.mode != "power_only" or config.electromagnetic.fields:
        raise ValueError("power grid requires power_only sensing and empty synthetic fields")
    if not isinstance(value.get("provenance", {}), dict):
        raise ValueError("power grid provenance must be an object")


class PowerGrid:
    """已包含背景的总功率；在毫瓦域双线性插值，静态切片不模拟任意移动源。"""

    def __init__(self, value):
        self.horizontal = np.asarray(value["x_m"], dtype=float)
        self.vertical = np.asarray(value["y_m"], dtype=float)
        self.power_mw = 10 ** (np.asarray(value["power_dbm"], dtype=float) / 10)

    def sample(self, points):
        points = np.asarray(points, dtype=float)
        if (points.shape[-1:] != (2,) or not np.isfinite(points).all()
                or np.any(points < [self.horizontal[0], self.vertical[0]])
                or np.any(points > [self.horizontal[-1], self.vertical[-1]])):
            raise ValueError("power grid query outside available coordinates")
        columns = np.clip(np.searchsorted(self.horizontal, points[..., 0], side="right") - 1, 0, len(self.horizontal) - 2)
        rows = np.clip(np.searchsorted(self.vertical, points[..., 1], side="right") - 1, 0, len(self.vertical) - 2)
        fraction_x = (points[..., 0] - self.horizontal[columns]) / (self.horizontal[columns + 1] - self.horizontal[columns])
        fraction_y = (points[..., 1] - self.vertical[rows]) / (self.vertical[rows + 1] - self.vertical[rows])
        lower = self.power_mw[rows, columns] * (1 - fraction_x) + self.power_mw[rows, columns + 1] * fraction_x
        upper = self.power_mw[rows + 1, columns] * (1 - fraction_x) + self.power_mw[rows + 1, columns + 1] * fraction_x
        return 10 * np.log10(lower * (1 - fraction_y) + upper * fraction_y)
