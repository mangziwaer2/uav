"""离线裁取GeoJSON建筑到局部米制；不联网、不获取或推测电磁源。"""

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .buildings import Building


def meter_value(value):
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\s*", value)
        if not match:
            raise ValueError("building height strings must use metres, not floors or feet")
        value = float(match.group(1))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid building height in metres")
    return value


def import_buildings(path, config, coordinates, source, license_name, origin_lonlat=None, default_height_m=None):
    """lonlat用小范围等距圆柱近似；原点在地图西南角，限5km，不作精确GIS测绘。"""
    path = Path(path)
    if coordinates not in ("local_m", "lonlat") or not source.strip() or not license_name.strip():
        raise ValueError("explicit coordinate mode, source and license are required")
    if config.map.buildings or config.map.obstacles_m:
        raise ValueError("import template must have empty buildings and obstacles_m; no silent geometry replacement")
    if default_height_m is not None and meter_value(default_height_m) <= 0:
        raise ValueError("default height must be positive")
    if coordinates == "lonlat":
        if (origin_lonlat is None or len(origin_lonlat) != 2 or not all(math.isfinite(item) for item in origin_lonlat)
                or not -180 <= origin_lonlat[0] <= 180 or not -75 <= origin_lonlat[1] <= 75):
            raise ValueError("lonlat import requires a southwest origin, latitude within +/-75 degrees")
    elif origin_lonlat is not None:
        raise ValueError("local_m input cannot also specify geographic origin")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("type") != "FeatureCollection" or not isinstance(value.get("features"), list):
        raise ValueError("building input must be a GeoJSON FeatureCollection")
    maximum = (np.array([config.map.columns, config.map.rows]) - 1) * config.map.resolution_m
    buildings, excluded, imputed = [], 0, 0
    for index, feature in enumerate(value["features"]):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("invalid GeoJSON feature")
        geometry = feature.get("geometry")
        properties = feature.get("properties") or {}
        if not isinstance(properties, dict) or not isinstance(geometry, dict) or geometry.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError("only Polygon/MultiPolygon building features are supported")
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for part, polygon in enumerate(polygons):
            if not isinstance(polygon, list) or not polygon:
                raise ValueError("building polygon requires an outer ring")
            rings = []
            for ring in polygon:
                points = np.asarray(ring)
                if points.dtype.kind not in "iuf" or points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                    raise ValueError("GeoJSON coordinates must be finite horizontal pairs")
                points = points.astype(float)
                if coordinates == "lonlat":
                    if np.any(np.abs(points[:, 0]) > 180) or np.any(np.abs(points[:, 1]) > 90):
                        raise ValueError("invalid longitude/latitude")
                    origin = np.asarray(origin_lonlat)
                    points = np.radians(points - origin) * [6371008.8 * math.cos(math.radians(origin[1])), 6371008.8]
                rings.append(points.tolist())
            outer = np.asarray(rings[0])
            if np.any(np.max(outer, axis=0) < 0) or np.any(np.min(outer, axis=0) > maximum):
                excluded += 1
                continue
            if coordinates == "lonlat" and any(np.linalg.norm(ring, axis=1).max() > 5000 for ring in map(np.asarray, rings)):
                raise ValueError("local geographic projection limited to 5km; preproject larger datasets to local metres")
            height = properties.get("height_m", properties.get("height"))
            assumed = height is None
            if assumed:
                if default_height_m is None:
                    raise ValueError("missing building height; provide an explicit default or complete the dataset")
                height = default_height_m
            base = meter_value(properties.get("base_m", properties.get("min_height", 0)))
            roof = meter_value(height)
            if roof <= base:
                raise ValueError("roof height must exceed base height")
            identifier = str(feature.get("id", properties.get("id", f"feature_{index}")))
            building = Building.from_dict({"id": f"{identifier}:{part}", "footprint_m": rings[0], "holes_m": rings[1:],
                                          "height_m": roof - base, "base_m": base,
                                          "height_source": "assumed_default" if assumed else "dataset"})
            buildings.append(asdict(building))
            imputed += int(assumed)
    if not buildings:
        raise ValueError("no buildings intersect the requested map extent")
    provenance = {"source": source, "license": license_name, "input_file": path.name,
                  "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "coordinate_mode": coordinates, "origin_lonlat": origin_lonlat,
                  "projection": "local_equirectangular_sphere_5km_limit" if coordinates == "lonlat" else "provided_local_metres",
                  "outside_buildings_excluded": excluded, "assumed_height_count": imputed,
                  "default_roof_height_m": default_height_m,
                  "boundary_policy": "retain complete intersecting footprints, no silent clipping or bounding rectangles"}
    result = replace(config, map=replace(config.map, geometry_mode="fixed_altitude", buildings=tuple(buildings), provenance=provenance))
    result.validate()
    from .environment import ElectromagneticWorld
    ElectromagneticWorld(result)
    return result
