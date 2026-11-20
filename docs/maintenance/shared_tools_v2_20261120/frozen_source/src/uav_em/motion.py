"""Polyline execution preserving corners for geometry and energy checks."""
import numpy as np

def advance_route(route: list, distance_m: float) -> tuple[list, list]:
    """返回剩余折线和本次实际经过的折线，保留拐点供连续几何复核。"""
    if not route or distance_m < 0:
        raise ValueError("advance requires a route and nonnegative distance")
    remaining = [list(point) for point in route]
    traversed = [remaining[0]]
    while len(remaining) > 1 and distance_m > 1e-9:
        start, end = np.asarray(remaining[0]), np.asarray(remaining[1])
        length = float(np.linalg.norm(end - start))
        if length <= distance_m + 1e-9:
            distance_m -= length
            traversed.append(end.tolist())
            remaining.pop(0)
        else:
            point = (start + (end - start) * distance_m / length).tolist()
            traversed.append(point)
            remaining[0] = point
            break
    return remaining, traversed

