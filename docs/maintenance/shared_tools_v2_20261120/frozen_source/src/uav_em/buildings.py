"""局部米制简单多边形棱柱；支持凹多边形和庭院，不依赖GIS运行环境。"""

from dataclasses import dataclass
import math

import numpy as np


TOLERANCE = 1e-8


def cross(first, second):
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def point_segment_distance(points, start, end):
    points, start, end = np.asarray(points), np.asarray(start), np.asarray(end)
    direction = end - start
    fraction = np.clip(np.sum((points - start) * direction, axis=-1)
                       / np.maximum(np.sum(direction * direction, axis=-1), 1e-30), 0, 1)
    return np.linalg.norm(points - start - fraction[..., None] * direction, axis=-1)


def segments_touch(start, end, first, second):
    start, end, first, second = map(np.asarray, (start, end, first, second))
    direction, edge = end - start, second - first
    first_side, second_side = cross(direction, first - start), cross(direction, second - start)
    start_side, end_side = cross(edge, start - first), cross(edge, end - first)
    crossing = (((first_side > 0) & (second_side < 0)) | ((first_side < 0) & (second_side > 0)))
    crossing &= (((start_side > 0) & (end_side < 0)) | ((start_side < 0) & (end_side > 0)))
    distance = np.minimum.reduce(np.broadcast_arrays(
        point_segment_distance(start, first, second), point_segment_distance(end, first, second),
        point_segment_distance(first, start, end), point_segment_distance(second, start, end)))
    return crossing | (distance <= TOLERANCE)


def ring_edges(ring):
    return zip(ring, (*ring[1:], ring[0]))


def ring_contains(points, ring):
    points = np.asarray(points, dtype=float)
    inside = np.zeros(points.shape[:-1], dtype=bool)
    boundary = np.zeros_like(inside)
    for first, second in ring_edges(ring):
        first, second = np.asarray(first), np.asarray(second)
        boundary |= point_segment_distance(points, first, second) <= TOLERANCE
        if first[1] != second[1]:
            spans = (first[1] > points[..., 1]) != (second[1] > points[..., 1])
            crossing_x = first[0] + (points[..., 1] - first[1]) * (second[0] - first[0]) / (second[1] - first[1])
            inside ^= spans & (points[..., 0] < crossing_x)
    return inside | boundary, boundary


def normalize_ring(value):
    try:
        ring = tuple(tuple(coordinate for coordinate in point) for point in value)
    except TypeError as exception:
        raise ValueError("building ring must contain coordinate pairs") from exception
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3 or any(len(point) != 2 or any(isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float)) or not math.isfinite(coordinate)
            for coordinate in point) for point in ring):
        raise ValueError("building ring requires at least three finite coordinate pairs")
    if len(set(ring)) != len(ring):
        raise ValueError("building ring has duplicate vertices")
    area = sum(cross(np.asarray(first), np.asarray(second)) for first, second in ring_edges(ring)) / 2
    if abs(area) < TOLERANCE:
        raise ValueError("building ring has zero area")
    edges = list(ring_edges(ring))
    for index, vertex in enumerate(ring):
        incoming = np.asarray(ring[index - 1]) - vertex
        outgoing = np.asarray(ring[(index + 1) % len(ring)]) - vertex
        if abs(cross(incoming, outgoing)) < TOLERANCE and np.dot(incoming, outgoing) > 0:
            raise ValueError("building ring has overlapping adjacent edges")
    for index, (start, end) in enumerate(edges):
        for other in range(index + 1, len(edges)):
            if other in (index + 1, len(edges) - 1 if index == 0 else -1):
                continue
            if segments_touch(start, end, *edges[other]):
                raise ValueError("building ring self-intersects")
    return ring


@dataclass(frozen=True)
class Building:
    """height_m为楼体高度，base_m为统一地面基准以上的基底；楼顶=base+height。"""

    id: str
    footprint_m: tuple
    holes_m: tuple
    height_m: float
    base_m: float = 0.0
    loss_db: float = 6.0
    height_source: str = "provided"

    @classmethod
    def from_dict(cls, value):
        required = {"id", "footprint_m", "height_m"}
        if not isinstance(value, dict) or not required.issubset(value) or set(value) - {
                *required, "holes_m", "base_m", "loss_db", "height_source"}:
            raise ValueError("invalid building schema")
        if not isinstance(value["id"], str) or not value["id"]:
            raise ValueError("building id must be a nonempty string")
        for name, default in (("height_m", 0), ("base_m", 0), ("loss_db", 6)):
            number = value.get(name, default)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
                raise ValueError("building height/base/loss must be finite and nonnegative")
        if value["height_m"] <= 0 or not isinstance(value.get("height_source", "provided"), str):
            raise ValueError("building height must be positive and have a source label")
        outer = normalize_ring(value["footprint_m"])
        holes = tuple(normalize_ring(ring) for ring in value.get("holes_m", ()))
        for index, hole in enumerate(holes):
            contained, boundary = ring_contains(np.asarray(hole), outer)
            if not contained.all() or boundary.any():
                raise ValueError("building hole must lie strictly inside footprint")
            for other in (outer, *holes[:index]):
                if any(segments_touch(*first, *second) for first in ring_edges(hole) for second in ring_edges(other)):
                    raise ValueError("building rings intersect")
                if other is not outer and (ring_contains(np.asarray(hole[0]), other)[0]
                                           or ring_contains(np.asarray(other[0]), hole)[0]):
                    raise ValueError("building holes overlap")
        return cls(value["id"], outer, holes, value["height_m"], value.get("base_m", 0),
                   value.get("loss_db", 6), value.get("height_source", "provided"))

    def contains(self, points):
        inside, boundary = ring_contains(points, self.footprint_m)
        for hole in self.holes_m:
            in_hole, on_hole = ring_contains(points, hole)
            inside &= ~in_hole | on_hole
        return inside | boundary

    def segment_blocked(self, start, end, clearance_m=0.0):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        blocked = self.contains(start) | self.contains(end)
        for ring in (self.footprint_m, *self.holes_m):
            for first, second in ring_edges(ring):
                blocked |= segments_touch(start, end, first, second)
                if clearance_m > 0:
                    distance = np.minimum.reduce(np.broadcast_arrays(
                        point_segment_distance(start, first, second), point_segment_distance(end, first, second),
                        point_segment_distance(first, start, end), point_segment_distance(second, start, end)))
                    blocked |= distance <= clearance_m + TOLERANCE
        return blocked

    def blocks_altitude(self, altitude_m, vertical_clearance_m):
        return self.base_m - vertical_clearance_m <= altitude_m <= self.base_m + self.height_m + vertical_clearance_m

    def ray_intersection(self, points, altitude_m, center, source_height_m):
        points, center = np.asarray(points, dtype=float), np.asarray(center, dtype=float)
        difference = source_height_m - altitude_m
        if abs(difference) < 1e-12:
            if not self.base_m <= altitude_m <= self.base_m + self.height_m:
                return np.zeros(points.shape[:-1], dtype=bool)
            enter, leave = 0.0, 1.0
        else:
            first, second = ((self.base_m - altitude_m) / difference,
                             (self.base_m + self.height_m - altitude_m) / difference)
            enter, leave = max(0.0, min(first, second)), min(1.0, max(first, second))
            if enter > leave or leave <= 0 or enter >= 1:
                return np.zeros(points.shape[:-1], dtype=bool)
        return self.segment_blocked(points + (center - points) * enter,
                                    points + (center - points) * leave)
