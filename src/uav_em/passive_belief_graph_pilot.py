"""Single-use P8 developer simulation; no external model or environment service."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .conditional_policy import _digest, _edge_free, _integrate, _length, _log_power
from .motion import advance_route
from .strategy_graph import StrategyBudget, _seal, evaluate_strategy, prepare_task
from .strategy_search import VERSION, search_dynamic, search_open_loop


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/passive_belief_graph_P8_20261124"
CONTRACT = ROOT / "docs/passive_belief_graph_P8_contract.md"
TEMPLATE = Path(__file__).with_name("passive_belief_graph_replay.html")
LIMITS = dict(geometry_tests=400, route_expansions=1000, route_relaxations=2000,
              power_queries=2000, base_receiver_terms=4000, likelihood_terms=100,
              search_states=100, action_evaluations=1000, belief_updates=100, policy_evaluations=12)
CASES = (("A1", 43601, 0), ("A2", 43602, 0), ("B1", 43603, 1), ("B2", 43604, 1), ("O", 43605, None))
WINDOW_S = 240
HORIZON_S = 15
GRID = (42, 26)
DISPLAY_STEP_S = .5


def _world(identifier, center, velocity, *, peak=-67., scale=22., rate=.05):
    return dict(id=identifier, weight=.5, background_dbm=-102., bases=[dict(
        center_m=center, velocity_mps=velocity, peak_dbm=peak, scale_m=scale,
        rate_db_per_second=rate, reference_time_seconds=0.)])


def public_task():
    return dict(version="corridor-task-v1", now_seconds=0., start_m=[0., 10.], goal_m=[180., 10.],
                bounds_m=[[-10., -58.], [190., 65.]], obstacles_m=[[65., -12., 100., 12.]],
                initial_distance_m=180., spent_length_m=0., speed_mps=15., flight_power_w=350.,
                remaining_energy_wh=17., reserve_energy_wh=15., measurement_noise_db=1.5,
                sensor_interval_seconds=1., history=[dict(time_seconds=0., position_m=[0., 10.],
                                                         interference_dbm=-84.2)],
                worlds=[_world("H1", [58., 32.], [-1.3, .3]), _world("H2", [45., -33.], [1.7, -.2])],
                depth=1, heading_degrees=[0.], corner_offset_m=.25)


def _cost_cut(means, noise, weights, scores):
    deltas = [first - second for first, second in zip(*scores, strict=True)]
    if means[0] == means[1] or deltas[0] * deltas[1] >= 0 or min(weights) <= 0:
        return None
    target = math.log(abs(deltas[0] / deltas[1])) - math.log(weights[1] / weights[0])
    return math.fsum(means) / 2 + noise ** 2 * target / (means[1] - means[0])


def _graph(body):
    budget = StrategyBudget(**LIMITS)
    prepared = prepare_task(body, budget)
    start = prepared["nodes"][prepared["root_id"]]
    if not prepared["preparation_complete"] or len(start["step_edges"]) != 1:
        raise ValueError("required ordinary shared prefix is unavailable")
    prefix = prepared["edges"][start["step_edges"][0]]
    checkpoint = prepared["nodes"][prefix["to_id"]]
    terminals = [prepared["edges"][idx] for idx in checkpoint["terminal_edges"]]
    if len(terminals) != 2 or checkpoint["step_edges"]:
        raise ValueError("required two-route, one-point graph is unavailable")
    for edge in start["terminal_edges"]:
        del prepared["edges"][edge]
    start["terminal_edges"] = []
    noise, means = body["measurement_noise_db"], checkpoint["signal_means_dbm"]
    scores = [[edge["duration_seconds"] + q for q in edge["q_by_world"]] for edge in terminals]
    optimum = _cost_cut(means, noise, [world["weight"] for world in body["worlds"]], scores)
    if optimum is not None and not any(math.isclose(row["threshold_dbm"], optimum) for row in checkpoint["cuts"]):
        budget.reserve(likelihood_terms=2 * len(means))
        z = [(optimum - mean) / (noise * math.sqrt(2)) for mean in means]
        checkpoint["cuts"].append(dict(threshold_dbm=optimum,
                                       below_by_world=[math.erfc(-value) / 2 for value in z],
                                       above_by_world=[math.erfc(value) / 2 for value in z]))
    prepared["work"] = budget.summary()
    prepared.pop("prepared_id")
    return _seal(prepared)


def _menu(graph):
    task = graph["task"]
    root = graph["nodes"][graph["root_id"]]
    prefix = graph["edges"][root["step_edges"][0]]
    node = graph["nodes"][prefix["to_id"]]
    options = []
    for label, identifier in zip(("R0", "R1"), node["terminal_edges"], strict=True):
        edge = graph["edges"][identifier]
        elapsed = prefix["duration_seconds"] + edge["duration_seconds"]
        length = prefix["length_m"] + edge["length_m"]
        options.append(dict(id=label, path=edge["path_m"], full_seconds=elapsed, full_length_m=length,
                            energy_wh=elapsed * task["flight_power_w"] / 3600,
                            score_by_hypothesis=[elapsed + start + end for start, end in zip(
                                prefix["q_by_world"], edge["q_by_world"], strict=True)]))
    public = dict(history=task["history"], building_boxes_m=task["obstacles_m"], goal_m=task["goal_m"],
                  common_prefix=prefix["path_m"], maximum_length_m=task["maximum_length_m"],
                  maximum_seconds=HORIZON_S, initial_energy_wh=task["remaining_energy_wh"],
                  reserve_energy_wh=task["reserve_energy_wh"],
                  ordinary_point=dict(t_s=node["time_seconds"], xy=node["position_m"],
                                      predictive_means_dbm=node["signal_means_dbm"],
                                      noise_std_db=task["measurement_noise_db"]),
                  prior=[world["weight"] for world in task["worlds"]], routes=options,
                  cuts=[dict(id=f"C{index}", **cut) for index, cut in enumerate(node["cuts"])])
    return dict(menu_id=_digest(public), **public)


def _prompt(menu):
    return ("Independent decision for a bounded synthetic UAV corridor graph. Use only this message; no tools, "
            "workspace, prior conversations, or network. Choose one complete preflight policy minimizing prior "
            "expected T+Q, not just time. The hypotheses are mutually exclusive and their columns share the same "
            "order throughout. Follow the common prefix, then the chosen cut classifies the one ordinary noisy "
            "power reading: below includes equality, above otherwise. Choose a route ID for both bins; they may "
            "match. Both routes have already passed common hard constraints. Future readings are not given. "
            "No extra sensing or route changes. Reply with only one JSON line, exactly fields menu_id, cut, "
            "below, above, reason, using the supplied menu ID and option IDs. Reason at most 240 characters.\n"
            + json.dumps(menu, allow_nan=False, separators=(",", ":")))


def _policy(graph, menu, reply):
    if (type(reply) is not dict or set(reply) != {"menu_id", "cut", "below", "above", "reason"}
            or reply["menu_id"] != menu["menu_id"] or not isinstance(reply["reason"], str)
            or not 1 <= len(reply["reason"].strip()) <= 240):
        raise ValueError("invalid or stale graph policy")
    prefix_id = graph["nodes"][graph["root_id"]]["step_edges"][0]
    checkpoint = graph["nodes"][graph["edges"][prefix_id]["to_id"]]
    routes = dict(zip((row["id"] for row in menu["routes"]), checkpoint["terminal_edges"], strict=True))
    cuts = {row["id"]: row["threshold_dbm"] for row in menu["cuts"]}
    if any(not isinstance(reply[field], str) for field in ("cut", "below", "above")):
        raise ValueError("policy requires cut and route IDs")
    if reply["cut"] not in cuts or any(reply[branch] not in routes for branch in ("below", "above")):
        raise ValueError("unknown cut or route")

    def leaf(branch):
        return dict(edge_id=routes[reply[branch]], threshold_dbm=None, below=None, above=None)

    return dict(version=VERSION, task_id=graph["task_id"], root=dict(
        edge_id=prefix_id, threshold_dbm=cuts[reply["cut"]], below=leaf("below"), above=leaf("above")))


def _data(path):
    return json.loads(path.read_bytes())


def _write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources():
    paths = [Path(__file__), TEMPLATE, *[Path(__file__).with_name(name + ".py") for name in (
        "conditional_policy", "motion", "strategy_graph", "strategy_search")]]
    return {path.name: _sha(path) for path in paths}


def freeze():
    OUT.mkdir(exist_ok=False)
    (OUT / "contract.md").write_bytes(CONTRACT.read_bytes())
    _write(OUT / "public_task.json", public_task())
    _write(OUT / "freeze.json", dict(contract_sha256=_sha(OUT / "contract.md"),
                                     public_task_sha256=_sha(OUT / "public_task.json"), sources=_sources(),
                                     cases=CASES, per_planner_limits=LIMITS, llm_calls=1, reply_s=WINDOW_S,
                                     max_prefixes=5, max_tails=10, max_points_per_path=16, horizon_s=HORIZON_S,
                                     max_truth_integration_nodes=5 * 2 * 34, max_display_values=5 * 31 * math.prod(GRID)))


def _check_freeze():
    record = _data(OUT / "freeze.json")
    if (record["sources"] != _sources() or record["contract_sha256"] != _sha(OUT / "contract.md")
            or record["public_task_sha256"] != _sha(OUT / "public_task.json")):
        raise ValueError("frozen P8 inputs changed")


def prepare():
    _check_freeze()
    _write(OUT / "prepare_started.json", dict(started_unix_s=time.time()))
    graph = _graph(_data(OUT / "public_task.json"))
    results = {name: search(graph, StrategyBudget(**LIMITS)) for name, search in (
        ("fixed", search_open_loop), ("condition", search_dynamic))}
    if not all(row["complete"] and row["best"] for row in results.values()):
        _write(OUT / "incomplete_planning.json", results)
        raise ValueError("no full numeric reference; execution stopped")
    menu = _menu(graph)
    _write(OUT / "prepared.json", graph)
    _write(OUT / "numerical.json", results)
    _write(OUT / "public_menu.json", menu)
    with (OUT / "request.txt").open("x", encoding="utf-8") as stream:
        stream.write(_prompt(menu))
    names = ("prepared.json", "numerical.json", "public_menu.json", "request.txt")
    _write(OUT / "ready.json", dict(sha256={name: _sha(OUT / name) for name in names},
                                    ready_unix_s=time.time(), menu_id=menu["menu_id"]))


def _receipt(graph, menu, ready, fallback):
    path = OUT / "reply.txt"
    elapsed = path.stat().st_mtime - ready["ready_unix_s"] if path.exists() else None
    status, reply, evaluated = "missing", None, fallback
    if path.exists():
        actual = path.read_bytes()
        try:
            if not 0 <= elapsed <= WINDOW_S:
                status = "outside_window"
            elif len(actual) <= 1024 and len(actual.splitlines()) == 1:
                reply = json.loads(actual)
                proposal = _policy(graph, menu, reply)
                evaluated = evaluate_strategy(graph, proposal, StrategyBudget(**LIMITS))
                status = "accepted"
            else:
                status = "malformed"
        except (ValueError, OverflowError):
            status = "invalid"
    return dict(status=status, reply=reply, local_delivery_s=elapsed,
                raw_sha256=_sha(path) if path.exists() else None), evaluated


def _assessment_world(seed, family, task):
    rng = random.Random(seed)
    if family is None:
        world = _world("outside", [76., -18.], [.8, .25], peak=-65., scale=13.)
    else:
        world = deepcopy(task["worlds"][family])
        base = world["bases"][0]
        base["center_m"] = [position + rng.uniform(-2.5, 2.5) for position in base["center_m"]]
        base["velocity_mps"] = [speed + rng.uniform(-.25, .25) for speed in base["velocity_mps"]]
        base["peak_dbm"] += rng.uniform(-1., 1.)
        base["scale_m"] *= rng.uniform(.94, 1.06)
    return world, [rng.gauss(0., task["measurement_noise_db"]) for _ in range(HORIZON_S + 1)]


def _power(world, xy, stamp):
    return -80. + 10. * _log_power(world, xy, stamp) / math.log(10.)


def _feedback(menu, reading):
    point = menu["ordinary_point"]
    z = [(reading - mean) / point["noise_std_db"] for mean in point["predictive_means_dbm"]]
    log_likelihood = [-.5 * value ** 2 - math.log(point["noise_std_db"] * math.sqrt(2 * math.pi)) for value in z]
    weights = [math.log(prior) + score for prior, score in zip(menu["prior"], log_likelihood, strict=True)]
    probs = [math.exp(weight - max(weights)) for weight in weights]
    return dict(t_s=point["t_s"], xy=point["xy"], measured_dbm=reading, predictive_log_likelihood=log_likelihood,
                z=z, posterior=[prob / math.fsum(probs) for prob in probs])


def _trace(path, task, world, noise, feedback):
    if path[0] != task["start_m"] or path[-1] != task["goal_m"] or any(
            not _edge_free(first, second, task) for first, second in zip(path, path[1:])):
        raise ValueError("executed route fails public geometry checks")
    length = _length(path)
    duration = length / task["speed_mps"]
    energy = duration * task["flight_power_w"] / 3600
    if (length > task["maximum_length_m"] + 1e-8 or duration > HORIZON_S + 1e-8
            or task["remaining_energy_wh"] - energy < task["reserve_energy_wh"] - 1e-9):
        raise ValueError("executed route fails resource checks")
    points = [*task["history"], dict(time_seconds=feedback["t_s"], position_m=feedback["xy"],
                                     interference_dbm=feedback["measured_dbm"])]
    for stamp in range(2, math.floor(duration) + 1):
        xy = advance_route(path, stamp * task["speed_mps"])[0][0]
        points.append(dict(time_seconds=stamp, position_m=xy, interference_dbm=_power(world, xy, stamp) + noise[stamp]))
    exposure = _integrate(world, path, task["now_seconds"], task["speed_mps"])
    return dict(path=path, points=points, T=duration, Q=exposure, T_plus_Q=duration + exposure,
                length_m=length, energy_wh=energy, stop="arrived")


def _display(world, task):
    width, height = GRID
    lower, upper = task["bounds_m"]
    abscissas = [lower[0] + (index + .5) * (upper[0] - lower[0]) / width for index in range(width)]
    ordinates = [upper[1] - (index + .5) * (upper[1] - lower[1]) / height for index in range(height)]
    times = [index * DISPLAY_STEP_S for index in range(round(HORIZON_S / DISPLAY_STEP_S) + 1)]
    return [dict(dbm=[round(_power(world, (west, north), stamp)) for north in ordinates for west in abscissas],
                 centers=[[position + speed * stamp for position, speed in zip(
                     base["center_m"], base["velocity_mps"], strict=True)] for base in world["bases"]])
            for stamp in times]


def _evaluate_case(spec, graph, menu, policies):
    label, seed, family = spec
    task = graph["task"]
    world, noise = _assessment_world(seed, family, task)
    point = menu["ordinary_point"]
    feedback = _feedback(menu, _power(world, point["xy"], point["t_s"]) + noise[1])
    route_labels = {tuple(map(tuple, row["path"])): row["id"] for row in menu["routes"]}
    traces, choices = {}, {}
    for arm, review in policies.items():
        root = review["policy"]["root"]
        bin_id = "below" if feedback["measured_dbm"] <= root["threshold_dbm"] else "above"
        remainder = graph["edges"][root[bin_id]["edge_id"]]["path_m"]
        route_id = route_labels[tuple(map(tuple, remainder))]
        choices[arm] = dict(route_id=route_id, bin=bin_id, threshold_dbm=root["threshold_dbm"])
        if route_id not in traces:
            traces[route_id] = _trace(menu["common_prefix"] + remainder[1:], task, world, noise, feedback)
    return dict(id=label, feedback=feedback, choices=choices, traces=traces), _display(world, task)


def execute():
    _check_freeze()
    ready = _data(OUT / "ready.json")
    if any(expected != _sha(OUT / name) for name, expected in ready["sha256"].items()):
        raise ValueError("sealed P8 checkpoint changed")
    _write(OUT / "execution_started.json", dict(started_unix_s=time.time()))
    graph, menu, numeric = (_data(OUT / name) for name in ("prepared.json", "public_menu.json", "numerical.json"))
    policies = {name: row["best"] for name, row in numeric.items()}
    receipt, policies["llm"] = _receipt(graph, menu, ready, policies["condition"])
    _write(OUT / "receipt.json", receipt)
    _write(OUT / "policies.json", policies)
    cases, truth = [], {}
    for spec in CASES:
        case, heat = _evaluate_case(spec, graph, menu, policies)
        cases.append(case)
        truth[case["id"]] = heat
        _write(OUT / f"case_{case['id']}.json", case)
    resources = dict(common_prefixes=len(cases), unique_tails=sum(len(case["traces"]) for case in cases),
                     llm_replies=int(receipt["raw_sha256"] is not None), display_values=len(cases) * 31 * math.prod(GRID))
    result = dict(cases=cases, receipts=receipt, resources=resources,
                  model_expected_tq={name: row["expected_score_seconds"] for name, row in policies.items()},
                  status="closed")
    _write(OUT / "results.json", result)
    payload = dict(result=result, menu=menu, bounds=graph["task"]["bounds_m"], truth=truth,
                   dimensions=GRID, grid_step_s=DISPLAY_STEP_S)
    replay = TEMPLATE.read_text(encoding="utf-8").replace("{{REPLAY_DATA}}", json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c"))
    with (OUT / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(replay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "prepare", "execute"))
    args = parser.parse_args()
    {"freeze": freeze, "prepare": prepare, "execute": execute}[args.stage]()
