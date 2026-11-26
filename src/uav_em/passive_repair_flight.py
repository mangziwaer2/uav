"""Execute the public repair policy with injected point sensing and reply delivery."""

from base64 import b64encode
from dataclasses import asdict
import math

from .environment import Measurement
from .passive_closed_loop import CountedPosterior, QueryLimit
from .passive_event_choice import alert
from .passive_local_repair import waypoint_domain
from .passive_model_advice import Flight
from .passive_repair_actions import action_table_digest
from .passive_repair_geometry import FIXED_DOMAIN, ONLINE_DOMAIN, OnlineRepairGeometry, repair_domain
from .passive_repair_pricing import LockedPrice
from .passive_repair_review import RepairReview
from .passive_repair_tool_policy import run_numerical_tools
from .passive_repair_tools import TOOL_DOMAIN, RepairToolWorkspace
from .passive_route_exchange import RouteExchange
from .planning import path_length


def _candidate(route, speed, price=None):
    return dict(route=[list(point) for point in route], time_s=path_length(route) / speed,
                unknown_s=None if price is None else price.unknown_s,
                compared_s=None if price is None else price.priced_s,
                costs=dict(risk=None if price is None or price.tail_q is None else price.score),
                price=None if price is None else asdict(price))


def _exchange(request, receipt):
    if request is None:
        return None
    return dict(request_id=request.request_id, request_sha256=request.request_sha256,
                request_b64=b64encode(request.wire).decode("ascii"), received_sha256=receipt.received_sha256,
                session_id=receipt.session_id, raw_reply_b64=b64encode(receipt.raw_reply).decode("ascii"),
                raw_reply_sha256=receipt.raw_sha256, status=receipt.status, elapsed_s=receipt.elapsed_s,
                reason=None if receipt.selection is None else receipt.selection.reason)


def _review_record(flight, arm, result):
    table = result.table
    rows = [_candidate(route, flight.speed, table.rows[index].price if table else None)
            for index, route in enumerate(result.routes)]
    choice = result.selected
    selected = result.reference_id
    repair = None
    if choice is not None and choice.status == "repaired":
        selected = len(rows)
        rows.append(_candidate(choice.route, flight.speed, table.rows[result.action_id].price))
        base = rows[result.reference_id]["route"]
        edit = result.edit
        repair = dict(replaced=base[edit.left:edit.right + 1], waypoint=edit.waypoint_xy_m,
                      replacement=rows[selected]["route"][edit.left:edit.left + (2 if edit.waypoint_xy_m is None else 3)])
    public = table.public() if table else None
    return dict(t=flight.now, candidates=rows, selected=selected, action_id=result.action_id,
                authority=arm, status=result.status, action_count=len(table.rows) if table else 0,
                action_inputs=public, action_table_sha256=action_table_digest(public) if table else None,
                incumbent=result.incumbent_id, reference=result.reference_id, advanced_m=0.0, repair=repair,
                resolution=None if choice is None else choice.status,
                edit=None if result.edit is None else asdict(result.edit),
                locked_intervals=table.locked_intervals if table else (), new_targets=result.new_targets,
                comparison=None if result.comparison is None else asdict(result.comparison),
                evidence_status=result.evidence_status, splice_attempts=table.splice_attempts if table else None,
                exchange=_exchange(result.request, result.receipt))


def _tool_record(workspace, speed, now, status, selected, arm, trace):
    def price_for(route_id):
        if route_id == 0 and 0 in workspace.pricing:
            return workspace.pricing[0].menu[0]
        result = workspace.comparisons.get((route_id, 0))
        return None if result is None else LockedPrice(**result["comparison"]["price"])

    evidence = workspace.comparisons.get((selected, 0))
    return dict(t=now, candidates=[_candidate(route.route, speed, price_for(route_id))
                                   for route_id, route in enumerate(workspace.routes)],
                authority=arm, status=status, selected=selected, action_id=selected, incumbent=0,
                advanced_m=0.0, comparison=None if evidence is None else evidence["comparison"],
                tool_trace=trace)


def fly_repair(geometry, aircraft, settings, sensor, *, arm, separation_m, min_post_s, waypoints=None,
               request_prefix=None, transport=None, tool_geometry_limit=None, tool_query_limit=None):
    """Run one arm. The caller must freeze scientific inputs and resources before using live replies."""
    if arm not in ("numerical", "llm"):
        raise ValueError("unknown repair decision arm")
    if arm == "llm" and (not isinstance(request_prefix, str) or not request_prefix.strip() or not callable(transport)):
        raise ValueError("LLM flight requires a request scope and receiving adapter")
    if arm == "numerical" and (request_prefix is not None or transport is not None):
        raise ValueError("numerical flight does not receive LLM replies")
    tool_mode = tool_geometry_limit is not None
    if tool_mode and (waypoints is not None or type(tool_geometry_limit) is not int or tool_geometry_limit <= 0):
        raise ValueError("tool mode needs online geometry and an explicit geometry budget")
    if tool_query_limit is not None and (not tool_mode or type(tool_query_limit) is not int or tool_query_limit <= 0):
        raise ValueError("an explicit query budget requires tool mode")
    online = OnlineRepairGeometry(geometry, aircraft) if waypoints is None else None
    waypoints = () if online is not None else tuple(sorted(waypoint_domain(waypoints)))
    flight = Flight(geometry, aircraft, settings)
    if tool_query_limit is not None:
        flight.ledger["query_limit"] = tool_query_limit
    if tool_mode:
        flight.ledger.update(geometry_limit=tool_geometry_limit, geometry_attempts=0)
    review = RepairReview(separation_m=separation_m, min_post_s=min_post_s, direction_memory=online is not None)
    tool_exchange = RouteExchange() if tool_mode and arm == "llm" else None
    consultations = 0

    def observe(position, clock):
        sample = sensor(tuple(position), clock)
        if (tuple(sample.position_m) != tuple(position) or sample.time_seconds != clock
                or not math.isfinite(sample.interference_dbm)):
            raise ValueError("point sensor must return a finite current-position measurement")
        return Measurement(tuple(position), clock, float(sample.interference_dbm))

    while not flight.stop:
        flight._observe(observe)
        frame = flight.frames[-1]
        frame.update(energy_left_wh=flight.energy, advanced_m=0.0, incumbent_before=flight.incumbent)
        record = None
        try:
            posterior = flight._fit((flight.active,))[flight.active]
            flight._display(posterior)
            if flight.stop:
                break
            counted = CountedPosterior(posterior, flight.ledger)
            if not flight.incumbent:
                routes, _ = repair_domain(flight, online, waypoints)
                if not routes:
                    flight.stop = "no_legal_remaining_route"
                    break
                selected = min(range(len(routes)), key=lambda index: (path_length(routes[index]), index))
                record = dict(t=flight.now, candidates=[_candidate(route, flight.speed) for route in routes],
                              selected=selected, authority="initial", status="initial", advanced_m=0.0)
            else:
                peak, high = alert(flight.incumbent, counted, flight.now, flight.speed)
                points = [(row["t"], *row["xy"], row["measured_dbm"], (row["forecast"] or {}).get("error_db"))
                          for row in flight.frames]
                queries_before = flight.ledger["queries"]
                request_id = f"{request_prefix}-{consultations + 1:03d}"
                if tool_mode:
                    status, targets = "rearmed", ()
                    if high:
                        workspace = RepairToolWorkspace(
                            online, flight.incumbent, counted, now=flight.now, flown_m=flight.flown,
                            energy_wh=flight.energy, points=points, peak_dbm=peak, min_post_s=min_post_s,
                            ledger=flight.ledger, geometry_limit=tool_geometry_limit)
                        if arm == "llm":
                            status, selected = workspace.review(request_id, tool_exchange, transport)
                            trace = [dict(exchange=_exchange(request, receipt), feedback=feedback)
                                     for request, receipt, feedback in workspace.turns]
                            consultations += 1
                        else:
                            status, selected, trace = run_numerical_tools(workspace)
                        record = _tool_record(workspace, flight.speed, flight.now, status, selected, arm, trace)
                else:
                    routes, guides = repair_domain(flight, online, waypoints) if high else ((), ())
                    result = review.check(flight.incumbent, routes, guides, counted, peak_dbm=peak, arm=arm,
                                          geometry=geometry, aircraft=aircraft, position=flight.position, now=flight.now,
                                          flown_m=flight.flown, energy_wh=flight.energy, points=points,
                                          request_id=request_id, transport=transport)
                    status, targets = result.status, result.new_targets
                    if targets:
                        consultations += int(result.request is not None)
                        flight.ledger["pricings"] += int(result.table is not None)
                        record = _review_record(flight, arm, result)
                frame["alert"] = dict(peak_dbm=peak, active=high, state=status, new_targets=targets)
                if record is not None:
                    record["pricing_queries"] = flight.ledger["queries"] - queries_before

            if record is not None:
                frame["decision"] = len(flight.decisions)
                flight.decisions.append(record)
                if record["selected"] is None:
                    flight.stop = record["status"]
                    break
                flight.ledger["reviews"] += 1
                flight.incumbent = frame["remaining"] = record["candidates"][record["selected"]]["route"]
            flight._advance({flight.active: posterior})
            frame["advanced_m"] = flight.flown - frame["flown_m"]
            if record is not None:
                record["advanced_m"] = frame["advanced_m"]
        except QueryLimit:
            if not flight.stop:
                flight.stop = "compute_budget"
    domain = TOOL_DOMAIN if tool_mode else ONLINE_DOMAIN if online else FIXED_DOMAIN
    return dict(flight.log(arm), consultations=consultations, route_domain=domain)
