"""Publish the closed P5 flights from immutable logs without sampling or planning."""

import hashlib
import json
import math
from pathlib import Path

from .environment import ElectromagneticWorld
from .passive_closed_loop_pilot import _executed_samples
from .passive_map import LIMIT_S
from .passive_map_pilot import _truth_q, _write
from .passive_model_advice import CHECKPOINT, parse_reply
from .passive_model_advice_pilot import LABELS, OUT, _forecast_metrics, _frozen, _replay, _scenario


def reconstruct(frames, speed):
    traversed = []
    for previous, frame in zip(frames, frames[1:]):
        moved = frame["moved"]
        if not moved or moved[0] != previous["xy"] or moved[-1] != frame["xy"]:
            raise ValueError("broken executed trace")
        clock = previous["t"]
        for first, second in zip(moved, moved[1:]):
            duration = math.dist(first, second) / speed
            traversed.append((first, second, clock, duration))
            clock += duration
        if not math.isclose(clock, frame["t"], rel_tol=0, abs_tol=1e-8):
            raise ValueError("recorded positions, speed and times disagree")
    return traversed


def evaluate(world, log):
    frames, aircraft = log["frames"], world.config.aircraft
    traversed = reconstruct(frames, aircraft.speed_mps)
    last = frames[-1]
    total_time = last["t"]
    cap = 1.25 * math.dist(world.geometry.config.start_m, world.geometry.config.goal_m)
    length = sum(math.dist(first, second) for first, second, _, _ in traversed)
    energy = total_time * aircraft.flight_power_w / 3600
    if (not math.isclose(length, last["flown_m"], abs_tol=1e-7)
            or not math.isclose(log["energy_left_wh"], aircraft.initial_energy_wh - energy, abs_tol=1e-8)
            or not math.isclose(log["cap_m"], cap, abs_tol=1e-8)):
        raise ValueError("executed resources disagree with public flight log")
    q = _truth_q(world, _executed_samples(traversed))
    legal = (length <= cap + 1e-7 and total_time <= LIMIT_S + 1e-8
             and log["energy_left_wh"] >= aircraft.reserve_energy_wh - 1e-8
             and all(world.geometry.edge_free(first, second) for first, second, _, _ in traversed))
    if log["stop"] == "arrived" and last["xy"] != list(world.geometry.config.goal_m):
        raise ValueError("recorded arrival does not match goal")
    grid = world.geometry.grid().reshape(-1, 2)
    trajectory = [dict(dbm=world.truth(grid, frame["t"]).round(2).tolist(),
                       sources=world.source_positions_for_evaluation(frame["t"])) for frame in frames]
    metrics = dict(T=total_time, Q=q, T_plus_Q=total_time + q, length_m=length, cap_m=cap, energy_wh=energy,
                   remaining_wh=log["energy_left_wh"], stop=log["stop"], legal=legal)
    return dict(metrics=metrics, trajectory=trajectory)


def publish():
    def load(name):
        return json.loads((OUT / name).read_text(encoding="utf-8"))

    if load("freeze.json") != _frozen():
        raise ValueError("frozen P5 inputs changed")
    progress = [json.loads(line) for line in (OUT / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    if ([row["mode"] for row in progress if row["status"] == "flight_closed"] != ["numerical", "llm"]
            or progress[-1] != dict(status="failed", error="'committed_s'")):
        raise ValueError("requires exactly the original two closed P5 arms and publication failure")
    receipt = load("receipt.json")
    if receipt["status"] != "timeout" or (OUT / "raw_reply.txt").read_bytes():
        raise ValueError("timely reply evidence disagrees with original fallback")
    late_reply = parse_reply((OUT / "late_reply.txt").read_text(encoding="utf-8").rstrip("\n"))
    packet = load("public_checkpoint.json")
    world = ElectromagneticWorld(_scenario())
    arms, results = {}, {}
    for mode in ("numerical", "llm"):
        log = load(f"public_{mode}.json")
        if (log["points"][:25] != packet["point_measurements"] or log["frames"][24]["t"] != CHECKPOINT
                or log["model"] != receipt["selected"]):
            raise ValueError("public prefix or fallback mismatch")
        evaluation = evaluate(world, log)
        arms[mode] = dict(log=log, evaluation=evaluation)
        results[mode] = dict(metrics=evaluation["metrics"], model=log["model"], new_forecasts=_forecast_metrics(log),
                             resources=log["resources"], reviews=[
                                 {key: row[key] for key in ("t", "changed", "model", "selected", "reference")}
                                 for row in log["decisions"]])
    traces = [[(frame["t"], frame["moved"]) for frame in entry["log"]["frames"]] for entry in arms.values()]
    page = _replay(arms, world.geometry.config)
    fallback = "LLM 通道 · 超时后数值回退"
    page = page.replace(json.dumps(LABELS), json.dumps(dict(fixed=LABELS["fixed"], event=fallback)))
    page = page.replace(LABELS["event"], fallback)
    page = page.replace('<div class="controls">', '<p>执行器在截止前未收到回复。LLM 通道按合同使用数值回退；'
                        '<a href="late_reply.txt">独立模型回执</a>仅供对照，未参与飞行。</p><div class="controls">')
    page = page.replace('<a href="raw_reply.txt">原始回执</a>',
                        '<a href="receipt.json">超时记录</a><a href="findings.md">结果解读</a>')
    _write(OUT / "results.json", dict(arms=results, same_trajectory=traces[0] == traces[1], reply=receipt,
                                      late_reply=late_reply, late_reply_executed=False,
                                      weighted_predictive_loss=packet["weighted_predictive_loss"]))
    with (OUT / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(page)
    names = ("freeze.json", "launch_addendum.md", "progress.jsonl", "request.txt", "public_checkpoint.json",
             "raw_reply.txt", "receipt.json", "late_reply.txt", "public_numerical.json", "public_llm.json",
             "index.html", "results.json")
    _write(OUT / "publication.json", dict(status="recovered_from_closed_logs", flights_rerun=0, calls_added=0,
                                          publisher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                          files={name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                                                 for name in names}))


if __name__ == "__main__":
    publish()
