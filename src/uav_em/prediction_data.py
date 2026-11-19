"""Causal mixed-policy collection. Policies accept geometry and point sensor only."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .equivalent_sources import FrozenGP
from .planning import path_length
from .scenarios import geometry_route, free_graph
from .short_loop import LoopSettings, PointGP, admissible, search_route
from .simulation import advance_route


POLICIES = ('geometry', 'left', 'right', 'gp_numerical')


@dataclass(frozen=True)
class DataSettings:
    max_seconds: float = 80.
    step_seconds: float = 1.
    review_seconds: float = 8.
    detour_m: float = 100.
    length_factor: float = 2.
    proposals: int = 350
    horizons: tuple = (0., 4., 8., 12.)
    distances: tuple = (30., 60., 120.)
    angles: tuple = (-60., -30., 0., 30., 60.)


def public_map(geometry, vertical_clearance_m):
    return dict(bounds_m=[[0., 0.], geometry.high.tolist()],
        clearance_m=geometry.clearance_m, geometry_mode=geometry.config.geometry_mode,
        vertical_clearance_m=vertical_clearance_m,
        obstacles_m=geometry.config.obstacles_m,
        buildings=[dict(id=b.id, footprint_m=np.asarray(b.footprint_m).tolist(), height_m=b.height_m,
                        base_m=b.base_m, holes_m=[np.asarray(h).tolist() for h in b.holes_m])
                   for b in geometry.buildings])


def query_tasks(geometry, position, goal, now, settings=DataSettings()):
    """No route, field, source or future measurement argument is available."""
    position, goal = np.asarray(position, float), np.asarray(goal, float)
    delta = goal-position
    forward = delta / np.linalg.norm(delta) if np.linalg.norm(delta) > 1e-8 else np.array([1., 0.])
    side = np.array([-forward[1], forward[0]])
    locations = [(position, 0., 0.)]
    for distance in settings.distances:
        for angle in settings.angles:
            theta = math.radians(angle)
            p = position + distance*(math.cos(theta)*forward + math.sin(theta)*side)
            if geometry.free(p):
                locations.append((p, distance, angle))
    return [dict(position_m=p.tolist(), time_seconds=float(now+horizon), horizon_seconds=horizon,
                 radial_distance_m=distance, angle_degrees=angle,
                 direct_geometry_free=bool(geometry.edge_free(position, p)))
            for p, distance, angle in locations for horizon in settings.horizons]


def prediction_input(geometry, aircraft, history, position, goal, now, settings=DataSettings()):
    if not history or any(p['time_seconds'] > now for p in history):
        raise ValueError('nonempty causal history required')
    times = [p['time_seconds'] for p in history]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('strictly ordered history required')
    if times[-1] != now or not np.allclose(history[-1]['position_m'], position):
        raise ValueError('latest measurement must be at this state')
    # Explicit whitelist: extra simulator fields cannot become estimator features.
    history = [{k: p[k] for k in ('position_m', 'time_seconds', 'interference_dbm')} for p in history]
    return dict(time_seconds=float(now), position_m=list(position), goal_m=list(goal),
        speed_mps=aircraft.speed_mps, altitude_m=aircraft.altitude_m,
        map=public_map(geometry, aircraft.vertical_clearance_m), history=history,
        queries=query_tasks(geometry, position, goal, now, settings))


def baseline_prediction(value, geometry, belief, background=-100.):
    gp = FrozenGP(value['history'], geometry, value['time_seconds'], belief, background)
    q = value['queries']
    mu, sd, support = gp.predict(np.array([x['position_m'] for x in q]),
                                 np.array([x['time_seconds'] for x in q]))
    return dict(gp_mean_dbm=mu.tolist(), gp_std_db=sd.tolist(), gp_supported=support.tolist(),
                persistence_dbm=[value['history'][-1]['interference_dbm']]*len(q))


def smooth(geometry, route):
    if not route:
        return []
    result, index = [route[0]], 0
    while index < len(route)-1:
        index = next(k for k in range(len(route)-1, index, -1)
                     if geometry.edge_free(route[index], route[k]))
        result.append(route[index])
    return result


def collection_route(geometry, graph, start, goal, policy, settings=DataSettings()):
    if policy not in POLICIES:
        raise ValueError('unknown policy')
    direct = smooth(geometry, geometry_route(geometry, graph, start, goal))
    if policy not in ('left', 'right') or not direct:
        return direct
    delta = np.asarray(goal)-start
    side = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
    anchor = (np.asarray(start)+goal)/2 + (1 if policy == 'left' else -1)*settings.detour_m*side
    ordered = sorted(graph, key=lambda node: (float(np.linalg.norm(geometry.position(node)-anchor)), node))
    # Connectivity is geometric, independent of RF. Keep the first feasible anchor.
    for node in ordered:
        point = geometry.position(node)
        first = geometry_route(geometry, graph, start, point)
        second = geometry_route(geometry, graph, point, goal)
        if first and second:
            # Smooth the two legs separately to preserve the intentional detour.
            return smooth(geometry, first) + smooth(geometry, second)[1:]
    return direct


def collect_episode(geometry, aircraft, belief, observe, start, goal, policy, seed,
                    settings=DataSettings(), background=-100.):
    """The only hidden-environment capability is observe(current_position, now)."""
    if policy not in POLICIES:
        raise ValueError('unknown policy')
    graph = free_graph(geometry)
    direct = collection_route(geometry, graph, start, goal, 'geometry', settings)
    maximum = settings.length_factor*path_length(direct)
    route = collection_route(geometry, graph, start, goal, policy, settings)
    estimator = PointGP(geometry, belief, background)
    now, spent, energy, next_review = 0., 0., aircraft.initial_energy_wh, 0.
    position = np.asarray(start, float)
    samples, motions, decisions, searches, frames = [], [], [], [], []
    termination = 'time_limit'
    while now <= settings.max_seconds+1e-8:
        measurement = observe(position, now)
        if (measurement.directional_scan is not None or measurement.time_seconds != now
                or not np.allclose(measurement.position_m, position)):
            raise ValueError('sensor must supply only current scalar point power')
        estimator.observe(measurement)
        at_end = np.linalg.norm(position-goal) < 1e-7 or now >= settings.max_seconds-1e-8
        if now >= next_review-1e-8 or at_end:
            value = prediction_input(geometry, aircraft, estimator.points, position, goal, now, settings)
            baseline = baseline_prediction(value, geometry, belief, background)
            samples.append(dict(input=value, baseline=baseline))
            if policy == 'gp_numerical' and not at_end:
                # Initial search plus at most ten later reviews under the 80s horizon.
                if not decisions or measurement.interference_dbm > -80.:
                    snapshot, _ = estimator.freeze(now)
                    proposed, records = search_route(snapshot, position.tolist(), list(goal), list(goal),
                        LoopSettings(proposals_per_review=settings.proposals), seed+1000*len(decisions))
                    searches.extend(dict(time_seconds=now, **r) for r in records)
                    accepted = admissible(proposed, geometry, position, goal, spent, maximum, energy, aircraft)
                    if accepted:
                        route = proposed
                    decisions.append(dict(time_seconds=now, accepted=accepted,
                        reason='initial_or_current_point_above_soft_reference', route_after=route))
            frames.append(dict(time_seconds=now, position_m=position.tolist(), route_m=route,
                               observation_count=len(estimator.points), sample_index=len(samples)-1))
            next_review = now+settings.review_seconds
        if at_end:
            termination = 'goal_reached' if np.linalg.norm(position-goal) < 1e-7 else 'time_limit'
            break
        if not admissible(route, geometry, position, goal, spent, maximum, energy, aircraft):
            termination = 'route_or_energy_limit'
            break
        step = min(settings.step_seconds, settings.max_seconds-now, next_review-now,
                   path_length(route)/aircraft.speed_mps)
        if step <= 1e-9:
            raise RuntimeError('nonprogressing collection')
        remaining, traversed = advance_route(route, step*aircraft.speed_mps)
        distance = path_length(traversed)
        if not all(geometry.edge_free(a, b) for a, b in zip(traversed, traversed[1:])):
            raise RuntimeError('invalid executed edge')
        motions.append(dict(start_time_seconds=now, end_time_seconds=now+step, traversed_m=traversed))
        spent += distance
        energy -= step*aircraft.flight_power_w/3600
        now += step
        route, position = remaining, np.asarray(remaining[0])
    return dict(policy=policy, observations=estimator.points, samples=samples, frames=frames,
        motions=motions, decisions=decisions, searches=searches,
        metrics=dict(success=termination == 'goal_reached', termination=termination,
            simulation_seconds=now, executed_length_m=spent, maximum_length_m=maximum,
            energy_used_wh=aircraft.initial_energy_wh-energy,
            rrt_sample_proposals=sum(s['iterations'] for s in searches)))
