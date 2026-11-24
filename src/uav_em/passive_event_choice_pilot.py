"""Run the frozen two-seed E1 comparison with synchronous independent replies."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import select
import sys
import time

from .config import ProjectConfig
from .environment import ElectromagneticWorld
from .passive_closed_loop import CountedPosterior, QueryLimit
from .passive_decision_replay import DATA, _payload, render
from .passive_event_choice import ALERT_DBM, HORIZON_S, MAX_EVENTS, alert, menu, prices
from .passive_map_pilot import _write
from .passive_model_advice import Flight
from .passive_model_advice_publish import evaluate
from .passive_trigger_pilot import _world


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/passive_event_choice_E1_20261124"
CONTRACT = ROOT / "docs/passive_event_choice_E1_contract.md"
SEEDS = (43621, 43641)
LABELS = dict(numerical="N · 窗口全枚举", llm="LLM · 事件选路")
MAX_WAIT_S, MAX_TOTAL_WAIT_S = 180, 720
MAX_REQUEST_BYTES, MAX_REPLY_BYTES = 24_000, 800
PROMPT = """You are the independent online route selector for an observation-limited civil UAV simulation.
Use ONLY the JSON in this message. No tools, files, external information, or previous decisions.
Select exactly one listed complete legal route ID (the incumbent is also a valid explicit choice).
Minimize time and power exposure. Near-window costs are already in linear power units. All route scores integrate Q only on the same observed-support time intervals; an unpriced interval is UNKNOWN, NOT low risk. Individual Q is context only, not comparable without considering support. Larger dBm means more power. Do not infer true sources, invent evidence, or change a route. Rows are [id, xy_path, remaining_goal_s, window_s, common_Q, compared_s, unknown_s, individual_Q, edit, delta_common_TQ]. First-waypoint groups are [xy,route_IDs]. Last readings are [second,x,y,dBm,prior_error_dB]. Select one, even when evidence is weak. Return exactly one JSON line: {"request_id":"...","route_id":0,"reason":"short evidence-based reason"}. No abstention or markdown.
"""


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _layout(layout):
    buildings = tuple(dict(id=f"B{index}", height_m=height,
                           footprint_m=[[left, bottom], [right, bottom], [right, top], [left, top]])
                      for index, ((left, bottom, right, top), height) in enumerate(
                          zip(layout.obstacles_m, (40, 45)), start=1))
    return replace(layout, geometry_mode="fixed_altitude", obstacles_m=(), buildings=buildings)


def _public():
    config = ProjectConfig()
    return dict(map=asdict(_layout(config.map)), aircraft=asdict(config.aircraft), belief=asdict(config.belief),
                sensor="ordinary position/time/noisy total dBm", packet_keys=["request_id", "t_s", "goal", "buildings",
                    "flown_m", "cap_m", "energy_wh", "reserve_wh", "alert_dbm", "window_s", "incumbent", "points",
                    "routes", "first_waypoints"],
                window_s=HORIZON_S, alert_dbm=ALERT_DBM)


def _frozen():
    paths = [path for path in Path(__file__).parent.iterdir() if path.suffix in (".py", ".js", ".html")]
    paths.append(ROOT / "requirements.txt")
    return dict(contract_sha256=_sha(CONTRACT.read_bytes()), seeds=list(SEEDS), arms=list(LABELS),
                public_sha256=_sha(_payload(_public()).encode()), max_events=MAX_EVENTS,
                implementation={str(path.relative_to(ROOT)): _sha(path.read_bytes()) for path in sorted(paths)})


def freeze():
    OUT.mkdir(exist_ok=False)
    (OUT / "contract.md").write_bytes(CONTRACT.read_bytes())
    (OUT / "public_inputs.json").write_text(_payload(_public()), encoding="utf-8")
    _write(OUT / "freeze.json", _frozen())


def parse_reply(raw, packet):
    def unique_keys(pairs):
        values = dict(pairs)
        if len(values) != len(pairs):
            raise ValueError("duplicate reply keys")
        return values

    if not raw or len(raw) > MAX_REPLY_BYTES or len(raw.splitlines()) != 1:
        raise ValueError("invalid reply size or lines")
    reply = json.loads(raw, object_pairs_hook=unique_keys)
    if (not isinstance(reply, dict) or set(reply) != {"request_id", "route_id", "reason"}
            or reply["request_id"] != packet["request_id"] or type(reply["route_id"]) is not int
            or not 0 <= reply["route_id"] < len(packet["routes"])
            or not isinstance(reply["reason"], str) or not reply["reason"].strip()):
        raise ValueError("reply must explicitly select one frozen route")
    return reply


def _packet(flight, rows, reference, peak, request_id):
    def rounded(value):
        return None if value is None else round(value, 3)

    points = [[item["t"], *item["xy"], rounded(item["measured_dbm"]),
               rounded((item.get("forecast") or {}).get("error_db"))] for item in flight.frames[-12:]]
    routes = [[index, row["route"], rounded(row["goal_s"]), rounded(row["time_s"]),
               rounded(row["costs"]["risk"] - row["time_s"]) if row["compared_s"] else None,
               rounded(row["compared_s"]), rounded(row["unknown_s"]), rounded(row["individual_q"]),
               row["edit"], rounded(row["delta_common"])] for index, row in enumerate(rows)]
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(tuple(row["route"][1]), []).append(index)
    packet = dict(request_id=request_id, t_s=flight.now, goal=flight.goal, buildings=flight.geometry.config.buildings,
                  flown_m=flight.flown, cap_m=flight.cap, energy_wh=flight.energy,
                  reserve_wh=flight.aircraft.reserve_energy_wh, alert_dbm=rounded(peak), window_s=HORIZON_S,
                  incumbent=reference, points=points, routes=routes, first_waypoints=list(groups.items()))
    if set(packet) != set(_public()["packet_keys"]):
        raise ValueError("packet changed the public allowlist")
    return packet


def _request(packet, resources, *, output=OUT, prompt=PROMPT):
    name = packet["request_id"]
    message = (prompt + _payload(packet)).encode()
    if len(message) > MAX_REQUEST_BYTES:
        return None, "request_budget"
    (output / f"request_{name}.txt").write_bytes(message)
    _write(output / f"packet_{name}.json", packet)
    wait = min(MAX_WAIT_S, MAX_TOTAL_WAIT_S - resources["wait_s"])
    start = time.monotonic()
    print(_payload(dict(status="awaiting_independent_reply", request=name, t_s=packet["t_s"],
                        path=str(output / f"request_{name}.txt"), limit_s=wait)), flush=True)
    raw = sys.stdin.buffer.readline(4096) if wait > 0 and select.select([sys.stdin], [], [], wait)[0] else b""
    elapsed = time.monotonic() - start
    resources["wait_s"] += elapsed
    (output / f"raw_{name}.txt").write_bytes(raw)
    reply, status = None, "timeout"
    if raw and elapsed <= wait:
        try:
            reply = parse_reply(raw, packet)
            status = "accepted"
        except (ValueError, UnicodeError):
            status = "invalid_reply"
    receipt = dict(status=status, elapsed_s=elapsed, request_sha256=_sha(message), raw_sha256=_sha(raw), response=reply)
    _write(output / f"receipt_{name}.json", receipt)
    print(_payload(dict(status=status, request=name, route=None if reply is None else reply["route_id"],
                        elapsed_s=round(elapsed, 2))), flush=True)
    return reply, status


def fly(world, arm, case, choose, *, price=prices):
    flight = Flight(world.geometry, world.config.aircraft, world.config.belief)
    events, issued, spent = 0, 0, False
    while not flight.stop:
        flight._observe(world.observe)
        try:
            posterior = flight._fit((flight.active,))[flight.active]
            counted = CountedPosterior(posterior, flight.ledger)
            flight._display(counted)
            if flight.stop:
                break
            initial = not flight.incumbent
            peak, high = (None, False) if initial else alert(flight.incumbent, counted, flight.now, flight.speed)
            spent = spent and high
            monitor = dict(peak_dbm=peak, active=high, state="rearmed" if not high else "already_reviewed")
            flight.frames[-1]["alert"] = monitor
            if initial or (high and not spent and events < MAX_EVENTS):
                routes, child = menu(flight)
                if not routes:
                    flight.stop = "no_legal_remaining_route"
                    break
                if initial or len(routes) > 1:
                    rows, numerical, reference = price(routes, counted, flight.now, flight.speed, flight.incumbent, child)
                    flight.ledger["pricings"] += 1
                    selected, authority, request_id = numerical, "numerical", None
                    if not initial:
                        events += 1
                        spent = True
                        monitor["state"] = "reviewed"
                        if arm == "llm":
                            issued += 1
                            request_id = f"E{case:02d}-{issued:02d}"
                            reply, status = choose(_packet(flight, rows, reference, peak, request_id))
                            if reply is None:
                                monitor["state"] = status
                                flight.stop = status
                                flight.decisions.append(dict(t=flight.now, candidates=rows, selected=None,
                                                             numerical=numerical, authority=status, request_id=request_id))
                                break
                            selected, authority = reply["route_id"], "llm"
                    flight._apply(dict(candidates=rows, selected=selected, reference=reference,
                                       qualified=len(rows), changed=reference is not None and selected != reference,
                                       numerical=numerical, authority=authority, request_id=request_id))
                    flight.decisions[-1]["reason"] = "initial" if initial else "early_risk"
                else:
                    monitor["state"] = "no_alternative"
            elif high and not spent:
                monitor["state"] = "event_budget"
            flight._advance({flight.active: posterior})
        except QueryLimit:
            flight.stop = "compute_budget"
    return dict(flight.log(arm), event_reviews=events, consultations=issued)


def _publish(seed, arms, config):
    grid = dict(rows=config.rows, cols=config.columns, spacing=config.resolution_m,
                obstacles=[[*part["footprint_m"][0], *part["footprint_m"][2]] for part in config.buildings],
                start=config.start_m, goal=config.goal_m)
    data = DATA + _payload(dict(grid=grid, arms=arms)) + "</script>"
    page = render(data, LABELS, f"E1 · 预警分岔试跑 · {seed}", "results.json")
    page = page.replace('>原回放</a>', '>结果数据</a>').replace('<th>T (s)</th><th>未知 (s)</th>',
                          '<th>窗口 T (s)</th><th>窗内未知 (s)</th>')
    (OUT / f"replay_{seed}.html").write_text(page, encoding="utf-8")


def run(digest):
    frozen = json.loads((OUT / "freeze.json").read_bytes())
    if (frozen != _frozen() or digest != frozen["contract_sha256"]
            or _sha((OUT / "contract.md").read_bytes()) != digest
            or _sha((OUT / "public_inputs.json").read_bytes()) != frozen["public_sha256"]):
        raise ValueError("E1 requires the frozen contract and public inputs")
    (OUT / "started").touch(exist_ok=False)
    results, resources, session_log = {}, {"wait_s": 0}, OUT / "progress.jsonl"
    for case, seed in enumerate(SEEDS, start=1):
        config = _world(seed).config
        config = replace(config, map=_layout(config.map))
        arms = {}
        for arm in LABELS:
            world = ElectromagneticWorld(config)
            log = fly(world, arm, case, lambda packet: _request(packet, resources))
            _write(OUT / f"public_{seed}_{arm}.json", log)
            evaluation = evaluate(world, log)
            arms[arm] = dict(log=log, evaluation=evaluation)
            with session_log.open("a", encoding="utf-8") as stream:
                stream.write(_payload(dict(seed=seed, arm=arm, stop=log["stop"], consultations=log["consultations"])) + "\n")
        _publish(seed, arms, config.map)
        results[seed] = {arm: dict(metrics=entry["evaluation"]["metrics"], events=[
            {key: event[key] for key in ("t", "selected", "numerical", "authority", "request_id")}
            for event in entry["log"]["decisions"]], resources=entry["log"]["resources"]) for arm, entry in arms.items()}
    _write(OUT / "results.json", dict(scenes=results, **resources))
    links = "".join(f'<li><a href="replay_{seed}.html">场景 {seed} · 全部候选与实际航迹</a></li>' for seed in SEEDS)
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><title>E1 事件选路</title><h1>E1 事件选路</h1>'
            f'<ul>{links}</ul><a href="results.json">结果</a> · <a href="contract.md">冻结合同</a>')
    (OUT / "index.html").write_text(html, encoding="utf-8")
    with session_log.open("a", encoding="utf-8") as stream:
        stream.write('{"status":"closed"}\n')
    print(_payload(dict(status="closed", output=str(OUT))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--authorized-contract-sha256")
    args = parser.parse_args()
    freeze() if args.freeze else run(args.authorized_contract_sha256)
