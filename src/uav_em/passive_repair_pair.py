"""Audit and publish a closed tool-mode pair without planning or sensing."""

from base64 import b64decode
import json
import math

from .motion import advance_route
from .passive_local_repair import check_remaining_route, xy
from .passive_model_advice_publish import evaluate
from .passive_repair_replay import _check_execution, _same_polyline, render_repair_replay
from .passive_repair_tools import TOOL_DOMAIN, _parse_call
from .passive_route_exchange import RouteRequest, _sha256


def _receipt(turn, frame, prefix, candidates, registered, previous, sessions):
    exchange = turn["exchange"]
    wire = b64decode(exchange["request_b64"], validate=True)
    reply = b64decode(exchange["raw_reply_b64"], validate=True)
    if _sha256(wire) != exchange["request_sha256"] or _sha256(reply) != exchange["raw_reply_sha256"]:
        raise ValueError("tool exchange bytes do not match their recorded hashes")
    packet = json.loads(wire)["input"]
    feedback = turn["feedback"]
    if (packet["request_id"] != exchange["request_id"] or packet["state"]["t_s"] != frame["t"]
            or packet["state"]["points"] != prefix or packet["last_feedback"] != previous
            or [(row["route_id"], row["xy_m"]) for row in packet["routes"]] != list(enumerate(candidates[:registered]))):
        raise ValueError("tool request does not match its recorded checkpoint")
    elapsed = exchange["elapsed_s"]
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid model exchange duration")
    if exchange["status"] == "accepted":
        request = RouteRequest(exchange["request_id"], wire, exchange["request_sha256"], registered)
        call = _parse_call(reply, request)
        session = exchange["session_id"]
        if (exchange["received_sha256"] != request.request_sha256 or not session or session in sessions
                or feedback is None or (call.tool, call.args, call.reason)
                != (feedback["tool"], feedback["args"], exchange["reason"])):
            raise ValueError("accepted reply does not match its tool step")
        sessions.add(session)
    elif feedback is not None:
        raise ValueError("rejected receipt cannot have executed feedback")
    return elapsed


def _inspect_tools(event, frame, prefix, candidates, sessions):
    if not candidates or not _same_polyline(candidates[0], frame["incumbent_before"]):
        raise ValueError("tool review must start from the existing commitment")
    trace, authority = event["tool_trace"], event["authority"]
    registered, previous = 1, None
    comparable, elapsed = set(), []
    terminal = {"committed", "compute_budget", "geometry_budget", "commit_verification_failed"}
    if not trace:
        raise ValueError("tool review is missing its trace")
    for index, turn in enumerate(trace):
        feedback = turn["feedback"]
        if authority == "llm":
            elapsed.append(_receipt(turn, frame, prefix, candidates, registered, previous, sessions))
        elif "exchange" in turn or feedback is None:
            raise ValueError("numerical trace must contain tool feedback without model receipts")
        status = feedback["status"] if feedback else turn["exchange"]["status"]
        if (status in terminal or feedback is None) != (index == len(trace) - 1):
            raise ValueError("tool review did not end at its first terminal response")
        if status == "generated":
            if (feedback["tool"] != "generate" or feedback["route_id"] != registered or registered >= len(candidates)
                    or not _same_polyline(feedback["xy_m"], candidates[registered])):
                raise ValueError("generated route is not registered in trace order")
            registered += 1
        if status == "comparable":
            comparable.add((feedback["route_id"], feedback["reference_id"]))
        if authority == "llm":
            previous = json.loads(json.dumps(feedback))
    selected = event["selected"]
    if (registered != len(candidates) or status != event["status"]
            or (status == "committed") != (selected is not None)
            or (selected is not None and (feedback["tool"] != "commit" or selected != feedback["route_id"]
                                          or not _same_polyline(feedback["xy_m"], candidates[selected])))):
        raise ValueError("terminal tool response does not match the recorded commitment")
    return dict(t=frame["t"], generated=registered - 1, comparable_pairs=len(comparable), selected=selected,
                status=status, advanced_m=frame["advanced_m"], model_rounds=len(elapsed), model_exchange_s=math.fsum(elapsed))


def _inspect_arm(log, geometry, aircraft):
    frames, measurements = log["frames"], log["points"]
    if (log["route_domain"] != TOOL_DOMAIN or not log["stop"] or not frames or len(frames) != len(measurements)):
        raise ValueError("expected a closed tool-mode flight log")
    _check_execution(geometry.config, log, frames)
    incumbent, prefix, checks, sessions = [], [], [], set()
    for index, (frame, sample) in enumerate(zip(frames, measurements)):
        now, position, power = frame["t"], frame["xy"], frame["measured_dbm"]
        expected_energy = aircraft.initial_energy_wh - now * aircraft.flight_power_w / 3600
        if (sample["time_seconds"] != now or xy(sample["position_m"]) != xy(position)
                or sample["interference_dbm"] != power or not math.isfinite(power) or frame["prefix_size"] != index + 1
                or not math.isclose(frame["energy_left_wh"], expected_energy, rel_tol=0, abs_tol=1e-8)
                or not _same_polyline(frame["incumbent_before"], incumbent)):
            raise ValueError("checkpoint history or resources disagree with execution")
        prefix.append([now, *position, power, (frame["forecast"] or {}).get("error_db")])
        event = None
        if "decision" in frame:
            event = log["decisions"][frame["decision"]]
            candidates = [row["route"] for row in event["candidates"]]
            for candidate in candidates:
                route = tuple(map(xy, candidate))
                if (len(route) < 2 or route[0] != xy(position) or route[-1] != xy(geometry.config.goal_m)
                        or len(set(route)) != len(route) or check_remaining_route(
                            route, geometry=geometry, aircraft=aircraft, now=now, flown_m=frame["flown_m"],
                            energy_wh=frame["energy_left_wh"]).status != "legal"):
                    raise ValueError("recorded candidate is not a legal complete remainder")
            if index == 0 and event["authority"] == "initial" and event["status"] == "initial" and event["selected"] is not None:
                pass
            elif index > 0 and event["authority"] == log["mode"] and frame["alert"]["active"]:
                checks.append(_inspect_tools(event, frame, prefix, candidates, sessions))
                if event["selected"] is None and (index != len(frames) - 1 or event["status"] != log["stop"]):
                    raise ValueError("uncommitted tool review must end the recorded flight")
            else:
                raise ValueError("decision does not match its arm or trigger")
        if (event is None or event["selected"] is None) and not _same_polyline(frame["remaining"], incumbent):
            raise ValueError("remainder changed without a recorded commitment")
        if index + 1 < len(frames):
            incumbent, _ = advance_route(frame["remaining"], frame["advanced_m"])
    return checks


def publish_tool_pair(out, world, logs, *, synthetic):
    """Write a new colocated replay and evidence file for two already closed arms."""
    labels = {"numerical": "数值组", "llm": "模拟回执" if synthetic else "LLM 调度"}
    if logs.keys() != labels.keys() or any(log["mode"] != mode for mode, log in logs.items()):
        raise ValueError("expected one numerical arm and one LLM arm")
    for limit, counter in (("geometry_limit", "geometry_attempts"), ("query_limit", "queries")):
        cap = logs["numerical"]["resources"].get(limit)
        if type(cap) is not int or cap <= 0 or any(
                log["resources"].get(limit) != cap or not 0 <= log["resources"][counter] <= cap for log in logs.values()):
            raise ValueError("tool pair must share explicit whole-flight resource limits")
    checks = {mode: _inspect_arm(log, world.geometry, world.config.aircraft) for mode, log in logs.items()}
    numerical, llm = (logs[mode]["frames"][0] for mode in labels)
    if (numerical["measured_dbm"] != llm["measured_dbm"]
            or not _same_polyline(numerical["remaining"], llm["remaining"])):
        raise ValueError("tool pair must share the same initial measurement and commitment")
    arms = {mode: dict(log=logs[mode], evaluation=evaluate(world, logs[mode])) for mode in labels}
    metrics = {mode: entry["evaluation"]["metrics"] for mode, entry in arms.items()}
    complete = all(row["stop"] == "arrived" and row["legal"] for row in metrics.values())
    delta = metrics["llm"]["T_plus_Q"] - metrics["numerical"]["T_plus_Q"] if complete else None
    summary = "两组均合法到达才比较全程 T+Q；本次不计算差值。" if delta is None else f"全程 Δ(T+Q)（LLM − N）：{delta:+.2f}"
    report = dict(domain=TOOL_DOMAIN, synthetic=synthetic, paired_delta_tq=delta, replay="replay.html",
                  arms={mode: dict(log=logs[mode], metrics=metrics[mode], checkpoints=checks[mode]) for mode in labels})
    evidence = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    page = render_repair_replay(world.geometry.config, arms, labels, synthetic=synthetic,
                                evidence_href="evidence.json", pair_summary=summary)
    out.mkdir(parents=True, exist_ok=False)
    for name, body in (("evidence.json", evidence), ("replay.html", page)):
        with (out / name).open("x", encoding="utf-8") as stream:
            stream.write(body)
    return report
