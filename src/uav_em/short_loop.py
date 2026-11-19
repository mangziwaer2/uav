"""Small causal flight loop with replaceable estimator, policy and path planner.

The online objects receive measurements and public geometry, never a world object.
Truth grids and flight exposure are computed after an episode has been frozen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

import numpy as np

from .belief import MapSnapshot
from .environment import ElectromagneticWorld, Geometry
from .equivalent_sources import FrozenGP
from .planning import path_length, plan
from .simulation import advance_route


@dataclass(frozen=True)
class LoopSettings:
    review_seconds: float = 8.
    step_seconds: float = 1.
    delay_seconds: float = .5
    max_seconds: float = 140.
    max_reviews: int = 16
    max_llm_calls: int = 12
    proposals_per_review: int = 700
    length_factor: float = 1.25
    target_distance_m: float = 140.


class PointGP:
    """Estimator seam: observed points in, snapshot/arbitrary-time query out."""
    def __init__(self, geometry, settings, background=-100.):
        self.geometry, self.settings, self.background = geometry, settings, background
        self.points = []

    def observe(self, measurement):
        point = {k: asdict(measurement)[k] for k in ('position_m', 'time_seconds', 'interference_dbm')}
        if self.points and point['time_seconds'] <= self.points[-1]['time_seconds']:
            raise ValueError('each observation must be new and ordered')
        self.points.append(point)

    def freeze(self, now):
        gp = FrozenGP(self.points, self.geometry, now, self.settings, self.background)
        xy = self.geometry.grid().reshape(-1, 2)
        mu, sd, support = gp.predict(xy, np.full(len(xy), now))
        shape = self.geometry.grid().shape[:-1]
        snapshot = MapSnapshot(now, mu.reshape(shape), sd.reshape(shape), support.reshape(shape),
                               np.full(shape, -1.), len(self.points), self.geometry,
                               recent_point_measurements=tuple(self.points[-16:]))
        snapshot.estimation_details = dict(method='local_level_gp', calibrated=False,
            support_radius_m=self.settings.support_radius_m,
            maximum_age_seconds=self.settings.max_age_seconds)
        return snapshot, gp


def targets(geometry, position, goal, distance=140.):
    """Public goal-oriented positions; free endpoints are not RF-safe corridors."""
    position, goal = np.asarray(position, float), np.asarray(goal, float)
    delta = goal - position
    remaining = float(np.linalg.norm(delta))
    result = [dict(id='goal', position_m=goal.tolist(), meaning='plan directly to goal')]
    if remaining <= distance:
        return result
    forward = delta / remaining
    side = np.array([-forward[1], forward[0]])
    for name, angle in (('right_wide', -70), ('right', -40), ('forward', 0), ('left', 40), ('left_wide', 70)):
        theta = math.radians(angle)
        point = position + distance * (math.cos(theta) * forward + math.sin(theta) * side)
        if geometry.free(point):
            result.append(dict(id=name, position_m=point.tolist(),
                goal_distance_reduction_m=remaining - float(np.linalg.norm(goal - point)),
                meaning='unmeasured intermediate endpoint; RRT must check the route'))
    return result


def route_forecast(route, gp, now, speed, delay=.5):
    """GP supported summaries along actual future arrival times; null outside support."""
    positions, times, durations = [], [], []
    elapsed = delay
    for a, b in zip(route, route[1:]):
        a, b = np.asarray(a), np.asarray(b)
        duration = float(np.linalg.norm(b - a) / speed)
        count = max(1, int(math.ceil(duration)))
        for k in range(count):
            positions.append(a + (b - a) * ((k + .5) / count))
            times.append(now + elapsed + duration * ((k + .5) / count))
            durations.append(duration / count)
        elapsed += duration
    if not positions:
        return dict(supported_fraction=0., mean_supported_dbm=None, max_supported_dbm=None)
    mu, sd, support = gp.predict(np.asarray(positions), np.asarray(times))
    dt = np.asarray(durations)
    return dict(supported_fraction=float(dt[support].sum() / dt.sum()),
                mean_supported_dbm=float(np.average(mu[support], weights=dt[support])) if support.any() else None,
                max_supported_dbm=float(mu[support].max()) if support.any() else None,
                meaning='model prediction at arrival times; unsupported is unknown, not safe',
                samples=[dict(position_m=point.tolist(), arrival_seconds=float(t),
                    mean_dbm=float(m) if ok else None, std_db=float(s), supported=bool(ok))
                    for point, t, m, s, ok in zip(positions, times, mu, sd, support)])


def public_context(snapshot, gp, points, position, goal, route, speed, energy, spent_length,
                   maximum_length, settings, calls):
    now, geometry = snapshot.time_seconds, snapshot.geometry
    choices = targets(geometry, position, goal, settings.target_distance_m)
    for item in choices:
        # Only straight-line forecast. Buildings may require a different executed route.
        item['direct_segment_geometry_free'] = geometry.edge_free(position, item['position_m'])
        item['straight_segment_forecast'] = route_forecast(
            [position, item['position_m']], gp, now, speed, settings.delay_seconds)
    route_info = route_forecast(route, gp, now, speed, settings.delay_seconds)
    return dict(time_seconds=now,
        vehicle=dict(position_m=list(position), goal_m=list(goal), speed_mps=speed, remaining_energy_wh=energy),
        map=dict(bounds_m=[[0., 0.], geometry.high.tolist()], clearance_m=geometry.clearance_m,
            blocking_buildings=[dict(id=b.id, footprint_m=b.footprint_m) for b in geometry.blocking_buildings],
            obstacles_m=geometry.config.obstacles_m),
        measurement_evidence=dict(recent_points=points[-24:], total_received=len(points),
            prefix_sha256=hashlib.sha256(json.dumps(points, sort_keys=True).encode()).hexdigest(),
            origin='noisy scalar power only at already visited positions'),
        electromagnetic_estimate=dict(estimator='local_level_gp', current_snapshot_only=False,
            uncertainty_is_calibrated=False, reference_dbm=-80.,
            meaning='future GP query, no source localization, unknown is not low power'),
        route=dict(points=route, length_m=path_length(route), forecast=route_info),
        waypoint_replanning=dict(candidates=choices, keep_route_until_revised=True,
            meaning='choose target ID; short waypoint plus a geometric route to goal, fly only to next review'),
        remaining_budgets=dict(llm_calls=max(0, settings.max_llm_calls-calls),
            route_length_m=maximum_length-spent_length, proposals_per_review=settings.proposals_per_review),
        available_actions=['continue', 'replan'],
        constraints=dict(interference_is_soft_cost=True, geometry_and_energy_are_hard=True,
            maximum_total_length_m=maximum_length, no_probe=True))


def validate_response(value, request_id, context):
    """LLM chooses existing IDs; no generated code, continuous weights or secret fields."""
    if not isinstance(value, dict) or set(value) != {'request_id', 'decision', 'strategy', 'memory'}:
        raise ValueError('response envelope')
    if value['request_id'] != request_id or value['strategy'] != {'mode': 'baseline'}:
        raise ValueError('identity or strategy')
    d = value['decision']
    if not isinstance(d, dict) or set(d) - {'action', 'target_id', 'reason'}:
        raise ValueError('decision fields')
    if d.get('action') not in ('continue', 'replan') or not isinstance(d.get('reason'), str):
        raise ValueError('action or reason')
    if d['action'] == 'replan':
        allowed = {item['id'] for item in context['waypoint_replanning']['candidates']}
        if d.get('target_id') not in allowed:
            raise ValueError('target is not from this request')
    elif 'target_id' in d:
        raise ValueError('continue cannot change target')
    if not isinstance(value['memory'], str) or len(value['memory']) > 1200:
        raise ValueError('memory')
    return json.dumps(d, ensure_ascii=False), value['strategy'], value['memory']


def numerical_decision(context):
    forecast = context['route']['forecast']
    recent = context['measurement_evidence']['recent_points']
    high = recent[-1]['interference_dbm'] > -80. or (forecast.get('max_supported_dbm') or -100.) > -80.
    return (dict(action='replan', target_id='goal', reason='supported power above soft reference; refresh GP route')
            if high else dict(action='continue', reason='retain remaining route; no supported high-power forecast'))


def search_route(snapshot, position, goal, target, settings, seed):
    endpoints = [position] + ([target] if not np.allclose(target, goal) else []) + [goal]
    count = len(endpoints)-1
    records, route = [], []
    budget = settings.proposals_per_review // count
    for index, (a, b) in enumerate(zip(endpoints, endpoints[1:])):
        solved = plan(snapshot, a, b, 'rrt_star', 'balanced', -80., budget, seed+index)
        records.append(asdict(solved))
        if not solved.path:
            return [], records
        route.extend(solved.path if not route else solved.path[1:])
    return route, records


def admissible(route, geometry, position, goal, spent_length, maximum_length, energy, aircraft):
    if not route or not np.allclose(route[0], position) or not np.allclose(route[-1], goal):
        return False
    length = path_length(route)
    return bool(spent_length + length <= maximum_length + 1e-7
        and energy - length / aircraft.speed_mps * aircraft.flight_power_w / 3600 >= aircraft.reserve_energy_wh
        and all(geometry.edge_free(a, b) for a, b in zip(route, route[1:])))


def run_loop(config, settings=LoopSettings(), advisor=None, emit=None):
    """Point sensor -> estimator -> policy -> RRT -> short execution, no RF truth feedback."""
    if config.sensor.mode != 'power_only':
        raise ValueError('only point-power sensing is allowed')
    world = ElectromagneticWorld(config)
    geometry, aircraft = world.geometry, config.aircraft
    estimator = PointGP(geometry, config.belief, config.electromagnetic.background_dbm)
    position, goal = np.asarray(config.map.start_m, float), np.asarray(config.map.goal_m, float)
    maximum = settings.length_factor * float(np.linalg.norm(goal-position))
    route, frames, decisions, motions, searches = [], [], [], [], []
    now, energy, spent, next_review = 0., aircraft.initial_energy_wh, 0., 0.
    calls, fallbacks = 0, 0
    termination = 'time_limit'
    while now <= settings.max_seconds + 1e-9:
        estimator.observe(world.observe(position, now))
        snapshot, gp = estimator.freeze(now)
        # Preserve even an initial planning failure or time-limit observation.
        frames.append(dict(time_seconds=now, position_m=position.tolist(), route_m=route,
            estimate_dbm=snapshot.estimate_dbm.tolist(), supported=snapshot.supported.tolist(),
            observed_positions_m=[p['position_m'] for p in estimator.points],
            estimate_issued_seconds=snapshot.time_seconds))
        if np.linalg.norm(position-goal) < 1e-7:
            termination = 'goal_reached';break
        if now >= settings.max_seconds-1e-9:
            break
        if not route:
            initial, records = search_route(snapshot, position.tolist(), goal.tolist(), goal.tolist(), settings, config.seed)
            searches.extend(dict(time_seconds=now, purpose='initial', **r) for r in records)
            if not admissible(initial, geometry, position, goal, spent, maximum, energy, aircraft):
                termination = 'initial_plan_failed';break
            route = initial
        if now >= next_review-1e-9 and len(decisions) < settings.max_reviews:
            context = public_context(snapshot, gp, estimator.points, position.tolist(), goal.tolist(), route,
                aircraft.speed_mps, energy, spent, maximum, settings, calls)
            model_raw, error, wall = None, None, 0.
            backend = 'numerical'
            if advisor is not None and calls < settings.max_llm_calls:
                calls += 1;backend = 'online_llm';tick=time.monotonic()
                try:
                    model_raw = advisor.decide(context)
                    decision = json.loads(model_raw)
                except Exception as exc:
                    error=repr(exc);fallbacks+=1;decision=numerical_decision(context)
                wall=time.monotonic()-tick
            else:
                decision=numerical_decision(context)
                if advisor is not None:
                    backend='budget_fallback';fallbacks+=1
            accepted, rejection = 'continue', None
            if decision['action'] == 'replan':
                target = next(c['position_m'] for c in context['waypoint_replanning']['candidates']
                              if c['id'] == decision['target_id'])
                candidate, records = search_route(snapshot, position.tolist(), goal.tolist(), target,
                                                  settings, config.seed+1000*(len(decisions)+1))
                searches.extend(dict(time_seconds=now, purpose=decision['target_id'], **r) for r in records)
                if admissible(candidate, geometry, position, goal, spent, maximum, energy, aircraft):
                    route=candidate;accepted='replan'
                else:
                    rejection='candidate failed geometry/search/total-length/energy check; retain route'
            entry=dict(index=len(decisions),time_seconds=now,context=context,backend=backend,
                proposed=decision,accepted_action=accepted,rejection=rejection,model_error=error,
                raw_response=model_raw,wall_seconds=wall,route_after=route)
            decisions.append(entry)
            if emit:emit('decision',entry)
            next_review=now+settings.review_seconds
            delay=min(settings.delay_seconds, settings.max_seconds-now)
            if energy-delay*aircraft.hover_power_w/3600 < aircraft.reserve_energy_wh:
                termination='energy_reserve';break
            energy-=delay*aircraft.hover_power_w/3600
            now+=delay
        frames.append(dict(time_seconds=now, position_m=position.tolist(), route_m=route,
            estimate_dbm=snapshot.estimate_dbm.tolist(), supported=snapshot.supported.tolist(),
            observed_positions_m=[p['position_m'] for p in estimator.points],
            estimate_issued_seconds=snapshot.time_seconds))
        step=min(settings.step_seconds, settings.max_seconds-now, path_length(route)/aircraft.speed_mps)
        if next_review > now:step=min(step,next_review-now)
        if step<=1e-9:break
        if energy-step*aircraft.flight_power_w/3600 < aircraft.reserve_energy_wh:
            termination='energy_reserve';break
        remaining, traversed=advance_route(route,step*aircraft.speed_mps)
        distance=path_length(traversed)
        if spent+distance > maximum+1e-7 or not all(geometry.edge_free(a,b) for a,b in zip(traversed,traversed[1:])):
            raise RuntimeError('execution rejected invalid path')
        now+=step;spent+=distance;energy-=step*aircraft.flight_power_w/3600
        motions.append(dict(end_time_seconds=now,traversed_m=traversed))
        route=remaining;position=np.asarray(route[0])
    return dict(schema_version=1,settings=asdict(settings),observations=estimator.points,
        frames=frames,decisions=decisions,motions=motions,searches=searches,
        metrics=dict(success=termination=='goal_reached',termination=termination,
            simulation_seconds=now,executed_length_m=spent,energy_used_wh=aircraft.initial_energy_wh-energy,
            llm_calls=calls,decision_count=len(decisions),fallbacks=fallbacks,
            model_errors=sum(d['model_error'] is not None for d in decisions),
            rejected_replans=sum(d['rejection'] is not None for d in decisions),
            rrt_sample_proposals=sum(s['iterations'] for s in searches),
            planning_calls=len(searches),planner_seconds=sum(s['elapsed_seconds'] for s in searches),
            llm_wall_seconds=sum(d['wall_seconds'] for d in decisions)))
