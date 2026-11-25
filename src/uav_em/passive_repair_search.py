"""Exhaust the same finite one-splice action set offered to the route selector."""

from dataclasses import dataclass

from .passive_closed_loop import QueryLimit
from .passive_local_repair import LocalRepair, ResolvedRepair, resolve_repair, waypoint_domain
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


def search_repairs(pricing, waypoints, *, geometry, aircraft, position, flown_m, energy_wh):
    """Keep the best completed legal choice if the common query ledger is exhausted."""
    if pricing.speed != aircraft.speed_mps:
        raise ValueError("route pricing and repair must use the same aircraft speed")
    waypoints = tuple(sorted(waypoint_domain(waypoints)))
    state = dict(geometry=geometry, aircraft=aircraft, position=position, now=pricing.now,
                 flown_m=flown_m, energy_wh=energy_wh)

    def rank(price, order):
        return round(price.score, 8), price.time_s, order

    starters = [resolve_repair(route, None, **state) for route in pricing.routes]
    route_id = min(range(len(starters)), key=lambda index: rank(pricing.menu[index], index))
    time_only = pricing.menu[route_id].status == "time_only"
    selected, edit, comparison = starters[route_id], None, None
    best = rank(pricing.menu[route_id], route_id)
    original = set(pricing.routes)
    expanded = {}
    tried = 0
    for index, route in enumerate(pricing.routes):
        for proposal in _edits(route, waypoints):
            tried += 1
            resolved = resolve_repair(route, proposal, **state)
            if resolved.status != "repaired" or resolved.route in original:
                continue
            _, sources = expanded.setdefault(resolved.route, (resolved, {}))
            sources.setdefault(index, proposal)

    insufficient = 0
    exhausted = False
    for order, (resolved, sources) in enumerate(expanded.values(), start=len(starters)):
        if time_only:
            insufficient += 1
            candidate = (round(resolved.time_s, 8), resolved.time_s, order)
            if candidate < best:
                route_id = next(iter(sources))
                selected, edit, best = resolved, sources[route_id], candidate
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
            route_id, selected, edit, comparison, best = index, resolved, sources[index], evidence, candidate
    return RepairSearch(route_id, edit, selected, comparison, tried, len(expanded), insufficient, exhausted,
                        time_only)
