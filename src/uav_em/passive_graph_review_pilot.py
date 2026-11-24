"""Accept one independent public review proposal and fly one P7 suffix."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from .environment import ElectromagneticWorld
from .motion import advance_route
from .passive_gp_changepoint_audit import OUT as AUDIT, _score
from .passive_map import FixedPriorPosterior, LIMIT_S, candidates
from .passive_map_pilot import ROOT, _write
from .passive_model_advice import Flight
from .passive_model_advice_publish import evaluate
from .passive_route_actions_branch_replay import render
from .passive_route_actions_pilot import LABELS, OUT as P7, STATE_KEYS, _config, _replay, _unique
from .planning import path_length


OUT = ROOT / "outputs/passive_graph_review_P7_20261124"
CONTRACT = ROOT / "docs/passive_graph_review_P7_pilot_contract.md"
DEADLINE_S = 360
MAX_INPUT = 10_000
MAX_REPLY = 1200
CHECKPOINT_S = 24
POINT_S = 25

PROMPT = """Independent simulated flight decision. Use ONLY the JSON below. No tools, files, network or prior runs.
You are NOT the route selector. Decide whether a risk explanation is worth testing with the one listed ordinary point before a fork. Optimize flight time plus linear power exposure, not just distance. Larger dBm means stronger interference. A proxy or GP support is not safety, and unknown is not low risk. The point has no reading yet. Do not infer sources or invent evidence.
Node and route IDs are zero-based. Edges are consecutive ordered nodes of route paths, with arrival seconds and complete goal-reaching memberships implied by those paths. Route columns: route ID, node IDs, arrival seconds, nominal T+Q proxy, regime-mixture T+Q proxy, unknown seconds, seconds from the ordinary sample until this route leaves the incumbent (negative means missed), nominal proxy delta after removing one-sided support price differences. Candidate forks are pairs with the incumbent. Point rows are [ID/second,x,y,dBm]. Residual rows are [point ID,issue second,predicted dBm,observed SD dB,error dB,standardized error,support].
Return exactly one JSON line with fields packet_id,action,fork_id,point_ids,revision_id,sample_id,reason. For action review, choose one listed fork ID, cite one to three distinct existing point IDs, set revision_id to gp_regime and sample_id to S25. Otherwise action abstain, null fork_id, revision_id and sample_id, and empty point_ids. Short reason in English. Do not choose a route or design a threshold. Do not wrap the line in a code block.
"""


def _read(path):
    return json.loads(path.read_bytes())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _put(name, raw):
    with (OUT / name).open("xb") as stream:
        stream.write(raw)


def _hashes():
    source = ROOT / "src/uav_em"
    paths = [path for path in source.iterdir() if path.suffix in (".py", ".js", ".html")]
    paths += [ROOT / "requirements.txt", CONTRACT, AUDIT / "forecasts.json", AUDIT / "public_inputs.json"]
    paths += [P7 / name for name in ("checkpoint_state.json", "public_checkpoint.json", "review.json", "ready.json")]
    return {str(path.relative_to(ROOT)): _sha(path) for path in sorted(paths)}


def _case():
    return next(case for case in _read(AUDIT / "forecasts.json") if case["t_s"] == CHECKPOINT_S)


def _packet(public, frozen):
    nodes = []
    for row in public["routes"]:
        for node in row[1]:
            if node not in nodes:
                nodes.append(node)
    routes = []
    for rid, path, arrivals, _, _, unknown, *evidence in public["routes"]:
        routes.append([rid, [nodes.index(node) for node in path], arrivals,
                       round(frozen["nominal_prices"][rid], 2), round(frozen["costs"]["expected_proxy_tq"][rid], 2),
                       unknown, round(frozen["lead_after_sample_s"][rid], 3), evidence[3]])
    forecasts = frozen["mixture"]
    weights = [math.exp(value) for value in forecasts["log_weights"]]
    means = forecasts["means_dbm"]
    spread = forecasts["observation_std_db"]
    mixed_mean = sum(weight * mean for weight, mean in zip(weights, means))
    mix_std = math.sqrt(sum(weight * ((mean - mixed_mean) ** 2 + std ** 2)
                            for weight, mean, std in zip(weights, means, spread)))
    previous = public["incumbent_idx"]
    forks = {f"F{idx}": [previous, idx] for idx in frozen["eligible"] if idx != previous}
    body = dict(t_s=CHECKPOINT_S, cap_m=public["cap_m"], flown_m=public["flown_m"], energy_wh=public["energy_wh"],
                reserve_wh=public["reserve_wh"], incumbent=previous, nodes=[[round(value, 3) for value in node] for node in nodes],
                routes=routes, forks=forks, points=[[round(value, 3) for value in row] for row in public["point_rows"]],
                residuals=public["residual_rows"], ordinary_sample=dict(id="S25", time_s=POINT_S,
                    xy=[round(value, 3) for value in forecasts["xy_m"]],
                    nominal_mean_sd=[round(frozen["nominal"][key][0], 3) for key in ("means_dbm", "observation_std_db")],
                    mixture_mean_sd=[round(mixed_mean, 3), round(mix_std, 3)],
                    reset_prior_weight=round(weights[0], 5), longest_segment_weight=round(weights[-1], 5)))
    body["packet_id"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return body


def _resolve(raw, elapsed, packet):
    if not math.isfinite(elapsed) or not 0 <= elapsed < DEADLINE_S:
        return dict(status="timeout")
    if not raw:
        return dict(status="missing_reply")
    try:
        if len(raw) > MAX_REPLY or len(raw.decode("utf-8").splitlines()) != 1:
            raise ValueError("expected one bounded UTF-8 line")
        reply = json.loads(raw, object_pairs_hook=_unique)
        if (not isinstance(reply, dict) or set(reply) != {"packet_id", "action", "fork_id", "point_ids",
                                                          "revision_id", "sample_id", "reason"}
                or reply["packet_id"] != packet["packet_id"]
                or not isinstance(reply["reason"], str) or not reply["reason"].strip()):
            raise ValueError("stale or malformed review reply")
        if reply["action"] == "abstain" and reply["point_ids"] == [] and all(
                reply[key] is None for key in ("fork_id", "revision_id", "sample_id")):
            return dict(status="accepted_abstain", proposal=reply)
        pts = reply["point_ids"]
        if (reply["action"] != "review" or not isinstance(reply["fork_id"], str) or reply["fork_id"] not in packet["forks"]
                or reply["revision_id"] != "gp_regime" or reply["sample_id"] != "S25" or not isinstance(pts, list)
                or not 1 <= len(pts) <= 3 or any(type(idx) is not int or not 0 <= idx <= CHECKPOINT_S for idx in pts)
                or len(pts) != len(set(pts))):
            raise ValueError("unlicensed fork, model revision, ordinary point, or evidence")
        return dict(status="accepted_review", proposal=reply)
    except ValueError as error:
        return dict(status="invalid_reply", error=str(error))


def prepare():
    original = _read(P7 / "ready.json")["hashes"]
    for name in ("checkpoint_state.json", "public_checkpoint.json", "review.json"):
        if _sha(P7 / name) != original[name]:
            raise ValueError("original P7 checkpoint bytes changed")
    review, public, frozen = _read(P7 / "review.json"), _read(P7 / "public_checkpoint.json"), _case()
    if (review["numeric_proxy_idx"] != frozen["nominal_best"] or public["incumbent_idx"] != frozen["incumbent"]
            or [row["route_xy_m"] for row in review["routes"]] != frozen["routes"]
            or frozen["nominal_best"] not in frozen["eligible"] or public["t_s"] != CHECKPOINT_S):
        raise ValueError("frozen audit and original legal action domain disagree")
    packet = _packet(public, frozen)
    request = (PROMPT + json.dumps(packet, separators=(",", ":"), allow_nan=False)).encode()
    if len(request) > MAX_INPUT:
        raise ValueError("public request exceeds its budget")
    OUT.mkdir(exist_ok=False)
    _put("contract.md", CONTRACT.read_bytes())
    _write(OUT / "freeze.json", dict(sha256=_hashes(), calls=1, suffixes=1, deadline_s=DEADLINE_S))
    _write(OUT / "public_checkpoint.json", packet)
    _put("review.json", (P7 / "review.json").read_bytes())
    _put("request.txt", request)
    _write(OUT / "ready.json", dict(origin_s=time.monotonic(), request_sha256=_sha(OUT / "request.txt"),
                                    packet_sha256=_sha(OUT / "public_checkpoint.json"),
                                    review_sha256=_sha(OUT / "review.json")))


def _choose(frozen, score, status):
    fallback, conditional = frozen["nominal_best"], score["conditional_action"]
    wins_score = score["log_density"]["mixture"] >= score["log_density"]["nominal"]
    checked = conditional if wins_score else fallback
    if status != "accepted_review":
        return fallback, checked, status
    if not wins_score:
        return fallback, checked, "review_not_supported_by_new_point"
    if conditional == fallback:
        return fallback, checked, "review_no_action_change"
    return conditional, checked, "review_changed_action"


def _commit(flight, frozen, action):
    eligible = frozen["eligible"]
    entries = []
    for idx in eligible:
        full = frozen["routes"][idx]
        tail, _ = advance_route(full, flight.speed)
        if math.dist(tail[0], flight.position) > 1e-8:
            raise ValueError("conditioned route missed the common point")
        tail[0] = flight.position
        length = path_length(tail)
        duration = length / flight.speed
        if (tail[-1] != flight.goal or length + flight.flown > flight.cap + 1e-8 or duration + flight.now > LIMIT_S + 1e-8
                or flight.energy - duration * flight.aircraft.flight_power_w / 3600 < flight.aircraft.reserve_energy_wh
                or not all(flight.geometry.edge_free(first, second) for first, second in zip(tail, tail[1:]))):
            raise ValueError("illegal full continuation after the ordinary point")
        entries.append(dict(route=tail, checkpoint_route=full, checkpoint_route_id=idx))
    reference = eligible.index(frozen["incumbent"])
    selected = eligible.index(action)
    flight._apply(dict(candidates=entries, selected=selected, reference=reference, qualified=len(entries),
                       changed=reference != selected))


def _publish(world, log, selected, metrics, review):
    data = dict(n_route=dict(log=log, evaluation=metrics))
    page = render(_replay(data, world.geometry.config), review,
                  dict(arms=dict(n_route=dict(route_idx=selected))), choice_s=POINT_S)
    for key, label in LABELS.items():
        if key != "n_route":
            page = page.replace(f'<option value="{key}">{label}</option>', "")
    page = page.replace(LABELS["n_route"], "LLM + P7").replace("P7 \u00b7 \u5206\u5c94\u4e0e\u5b9e\u9645\u822a\u8ff9",
                                                                       "P7 \u00b7 LLM \u98ce\u9669\u590d\u6838")
    page = page.replace('<a href="index.html">\u539f\u56de\u653e</a>',
                        '<a href="../passive_route_actions_P7_20261124/branches.html">P7 \u539f\u56de\u653e</a>'
                        '<a href="receipt.json">LLM \u56de\u6267</a><a href="public_flight.json">\u98de\u884c\u65e5\u5fd7</a>')
    page = page.replace('24 \u79d2\u51b3\u7b56</h2>', '24 \u79d2\u5019\u9009 / 25 \u79d2\u51b3\u7b56</h2>')
    _put("index.html", page.encode())


def execute(text, session):
    arrival = time.monotonic()
    ready = _read(OUT / "ready.json")
    if (_read(OUT / "freeze.json")["sha256"] != _hashes() or CONTRACT.read_bytes() != (OUT / "contract.md").read_bytes()
            or ready["request_sha256"] != _sha(OUT / "request.txt")
            or ready["packet_sha256"] != _sha(OUT / "public_checkpoint.json")
            or ready["review_sha256"] != _sha(OUT / "review.json")):
        raise ValueError("frozen pilot inputs changed")
    raw = text.encode()
    elapsed = arrival - ready["origin_s"]
    packet = _read(OUT / "public_checkpoint.json")
    receipt = dict(_resolve(raw, elapsed, packet), elapsed_s=elapsed, session=session, bytes=len(raw),
                   raw_sha256=hashlib.sha256(raw).hexdigest(), request_sha256=ready["request_sha256"])
    if text and not session:
        raise ValueError("independent decision session is required for a nonempty reply")
    _put("raw_reply.txt", raw)
    _write(OUT / "receipt.json", receipt)
    _write(OUT / "progress.jsonl", dict(status="one_suffix_started", receipt_status=receipt["status"]))
    state = _read(P7 / "checkpoint_state.json")
    frozen = _case()
    config = _config()
    world = ElectromagneticWorld(config)
    world.random.bit_generator.state = state["noise_state"]
    flight = Flight(world.geometry, config.aircraft, config.belief)
    for key in STATE_KEYS:
        setattr(flight, key, state["flight"][key])
    original = _read(P7 / "public_checkpoint.json")
    if (flight.stop or flight.now != CHECKPOINT_S or len(flight.points) != 25
            or [[p["time_seconds"], *p["position_m"], p["interference_dbm"]] for p in flight.points] != original["point_rows"]
            or candidates(flight.position, flight.goal, flight.incumbent, flight.geometry, flight.aircraft,
                          flight.now, flight.flown, flight.energy, flight.cap) != frozen["routes"]):
        raise ValueError("restored public flight state or route legality changed")
    initial_ledger = dict(flight.ledger)
    posterior = flight._fit((flight.active,))[flight.active]
    flight._advance({flight.active: posterior})
    flight._observe(world.observe)
    score = _score(frozen, flight.points[-1])
    selected, strong_checked, status = _choose(frozen, score, receipt["status"])
    _commit(flight, frozen, selected)
    while True:
        posterior = flight._fit((flight.active,))[flight.active]
        flight._display(posterior)
        if flight.stop:
            break
        flight._advance({flight.active: posterior})
        flight._observe(world.observe)
    log = flight.log("llm_review")
    _write(OUT / "public_flight.json", log)
    evaluation = evaluate(world, log)
    _write(OUT / "results.json", dict(outcome=status, route_idx=selected, observation=score,
                                    nominal_idx=frozen["nominal_best"], mixture_fixed_idx=frozen["mixture_best"],
                                    strong_conditioned_idx=score["conditional_action"], strong_checked_idx=strong_checked,
                                    metrics=evaluation["metrics"], inherited_resources=initial_ledger,
                                    new_resources={key: value - initial_ledger[key] for key, value in flight.ledger.items()},
                                    reused_audit_resources=frozen["resources"], repeated_prefix_flights=0,
                                    new_suffixes=1, new_points=len(flight.points) - 25, receipt=receipt))
    _publish(world, log, selected, evaluation, _read(OUT / "review.json"))
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n" + json.dumps(dict(status="closed", flight_stop=flight.stop)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_subparsers(dest="command", required=True)
    choices.add_parser("prepare")
    go = choices.add_parser("execute")
    go.add_argument("--reply", default="")
    go.add_argument("--session", default="")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        execute(args.reply, args.session)
