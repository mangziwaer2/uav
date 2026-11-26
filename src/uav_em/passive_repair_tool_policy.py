"""Deterministic operator control over the existing online geometry proposal set."""

from .passive_route_exchange import ToolCall


def run_numerical_tools(workspace):
    """Use the same legal actions and R0-locked prices without model receipts."""
    incumbent = workspace.routes[0].route
    start, goal = incumbent[0], incumbent[-1]
    shortest = workspace.shortest_route
    guides = workspace.planner.guides(start, incumbent)
    if shortest and shortest != incumbent:
        lead = shortest[1] if len(shortest) > 2 else tuple((a + b) / 2 for a, b in zip(start, goal))
        guides = tuple(dict.fromkeys((lead, *guides)))

    steps = []

    def call(tool, args, reason):
        feedback = workspace.execute(ToolCall(tool, args, reason))
        steps.append(dict(feedback=feedback, reason=reason))
        return feedback

    selected, best = 0, (0.0, workspace.routes[0].time_s, 0)
    for point in guides:
        if point in (start, goal):
            continue
        generated = call("generate", dict(operator="via", waypoint_xy_m=point), "deterministic geometry sweep")
        if generated["status"] == "geometry_budget":
            return generated["status"], None, steps
        if generated["status"] != "generated":
            continue
        route_id = generated["route_id"]
        compared = call("compare", dict(route_id=route_id, reference_id=0), "compare against R0's locked window")
        if compared["status"] == "compute_budget":
            return compared["status"], None, steps
        if compared["status"] == "comparable":
            comparison = compared["comparison"]
            margin = comparison["delta_t_s"] + comparison["delta_tail_q"]
        elif compared["status"] == "no_baseline_evidence":
            margin = compared["comparison"]["delta_t_s"]
        else:
            continue
        rank = (round(margin, 8), generated["time_s"], route_id)
        if rank < best:
            selected, best = route_id, rank

    committed = call("commit", dict(route_id=selected), "best eligible R0-relative score")
    return committed["status"], committed.get("route_id") if committed["status"] == "committed" else None, steps
