"""Exhaust the same finite one-splice action set offered to the route selector."""

from dataclasses import dataclass
from itertools import groupby

from .passive_closed_loop import QueryLimit
from .passive_local_repair import LocalRepair, ResolvedRepair, resolve_repair, waypoint_domain, xy
from .passive_repair_pricing import RepairComparison


@dataclass(frozen=True)
class RepairSearch:
    route_id: int
    edit: LocalRepair | None
    selected: ResolvedRepair
    comparison: RepairComparison | None
    tried: int
    new_legal: int
    evidence_insufficient: int
    exhausted: bool
    time_only: bool


def _edits(route, waypoints):
    for left in range(len(route) - 1):
        for right in range(left + 1, len(route)):
            if right > left + 1:
                yield LocalRepair(left, right)
            for point in waypoints:
                if point != route[left] and point != route[right]:
                    yield LocalRepair(left, right, point)


@dataclass(frozen=True)
class RepairNeighborhood:
    originals: tuple[ResolvedRepair, ...]
    original_aliases: tuple[dict[int, tuple[LocalRepair, ...]], ...]
    expanded: tuple[tuple[ResolvedRepair, dict[int, tuple[LocalRepair, ...]]], ...]
    tried: int
    now: float
    speed: float
    waypoints: tuple


def normalize_route(route):
    return tuple(point for point, _ in groupby(map(xy, route)))


def build_repair_neighborhood(routes, waypoints, *, geometry, aircraft, position, now, flown_m, energy_wh):
    """Enumerate the common legal action domain without querying a predictor."""
    waypoints = tuple(sorted(waypoint_domain(waypoints)))
    state = dict(geometry=geometry, aircraft=aircraft, position=position, now=now,
                 flown_m=flown_m, energy_wh=energy_wh)
    routes = tuple(map(normalize_route, routes))
    if any(len(set(route)) != len(route) for route in routes):
        raise ValueError("original routes must not repeat vertices")
    starters = tuple(resolve_repair(route, None, **state) for route in routes)
    original = {row.route: index for index, row in enumerate(starters)}
    original_aliases = [{} for _ in starters]
    expanded = {}
    tried = 0
    for index, starter in enumerate(starters):
        route = starter.route
        for proposal in _edits(route, waypoints):
            tried += 1
            resolved = resolve_repair(route, proposal, **state)
            if resolved.status != "repaired" or len(set(resolved.route)) != len(resolved.route):
                continue
            if resolved.route in original:
                sources = original_aliases[original[resolved.route]]
            else:
                _, sources = expanded.setdefault(resolved.route, (resolved, {}))
            sources.setdefault(index, []).append(proposal)

    def freeze_aliases(sources):
        return {index: tuple(edits) for index, edits in sources.items()}

    rows = tuple((resolved, freeze_aliases(sources)) for resolved, sources in expanded.values())
    return RepairNeighborhood(starters, tuple(map(freeze_aliases, original_aliases)), rows,
                              tried, now, aircraft.speed_mps, waypoints)


def search_neighborhood(pricing, neighborhood):
    """Keep the best completed legal choice if the common query ledger is exhausted."""
    starters = neighborhood.originals
    if ((pricing.now, pricing.speed) != (neighborhood.now, neighborhood.speed)
            or pricing.routes != tuple(row.route for row in starters)):
        raise ValueError("pricing and legal action domain must use the same checkpoint")

    def rank(price, order):
        return round(price.score, 8), price.time_s, order

    route_id = min(range(len(starters)), key=lambda index: rank(pricing.menu[index], index))
    time_only = pricing.menu[route_id].status == "time_only"
    selected, edit, comparison = starters[route_id], None, None
    best = rank(pricing.menu[route_id], route_id)

    insufficient = 0
    exhausted = False
    for order, (resolved, sources) in enumerate(neighborhood.expanded, start=len(starters)):
        if time_only:
            insufficient += 1
            candidate = (round(resolved.time_s, 8), resolved.time_s, order)
            if candidate < best:
                route_id = next(iter(sources))
                selected, edit, best = resolved, sources[route_id][0], candidate
            continue
        try:
            comparisons = pricing.compare_references(resolved.route, sources)
        except QueryLimit:
            exhausted = True
            break
        usable = next(((index, evidence) for index, evidence in zip(sources, comparisons)
                       if evidence.price.score is not None), None)
        if usable is None:
            insufficient += 1
            continue
        index, evidence = usable
        candidate = rank(evidence.price, order)
        if candidate < best:
            route_id, selected, edit, comparison, best = index, resolved, sources[index][0], evidence, candidate
    return RepairSearch(route_id, edit, selected, comparison, neighborhood.tried, len(neighborhood.expanded),
                        insufficient, exhausted, time_only)


def search_repairs(pricing, waypoints, *, geometry, aircraft, position, flown_m, energy_wh):
    neighborhood = build_repair_neighborhood(pricing.routes, waypoints, geometry=geometry, aircraft=aircraft,
                                             position=position, now=pricing.now, flown_m=flown_m, energy_wh=energy_wh)
    return search_neighborhood(pricing, neighborhood)
