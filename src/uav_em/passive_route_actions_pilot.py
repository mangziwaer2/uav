"""Freeze one P7 simulation, accept bounded independent replies, then fly its suffixes."""

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import time

from . import passive_closed_loop_pilot as p4
from .environment import ElectromagneticWorld
from .passive_closed_loop import CountedPosterior, QueryLimit
from .passive_decision_evidence import build_decision_evidence
from .passive_map import FixedPriorPosterior
from .passive_map_pilot import ROOT, TEMPLATE, _write
from .passive_model_advice import CHECKPOINT, Flight
from .passive_route_actions import build_route_action_menu, resolve_route_action
from .passive_trigger_pilot import _world


OUT = ROOT / "outputs/passive_route_actions_P7_20261124"
CONTRACT = ROOT / "docs/passive_route_actions_P7_contract.md"
SEED = 43537
MODES = ("route", "waypoint")
DEADLINE_S = 360
INPUT_BYTES = 14_000
REPLY_BYTES = 1_000
LABELS = dict(llm_waypoint="LLM · 中间目标", llm_route="LLM · 直接选路", n_guard="N · 原软门控",
              n_route="N · 全域枚举", n_waypoint="N · 中间目标域")
STATE_KEYS = ("position", "energy", "now", "flown", "incumbent", "moved", "pending", "points", "frames",
              "decisions", "traversed", "ledger", "stop")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(name):
    return json.loads((OUT / name).read_bytes())


def _put(name, content):
    with (OUT / name).open("xb") as stream:
        stream.write(content)


def _log(status, **details):
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(status=status, **details), allow_nan=False) + "\n")


def _config():
    base = _world(SEED).config
    buildings = tuple(dict(id=f"B{index}", height_m=height,
                           footprint_m=[[left, bottom], [right, bottom], [right, top], [left, top]])
                      for index, ((left, bottom, right, top), height) in enumerate(
                          zip(base.map.obstacles_m, (40, 45)), start=1))
    return replace(base, map=replace(base.map, geometry_mode="fixed_altitude", obstacles_m=(), buildings=buildings))


def _freeze_record():
    source = ROOT / "src/uav_em"
    modules = (*p4.MODULES, "passive_model_advice", "passive_route_actions", "passive_decision_evidence",
               "passive_route_actions_pilot", "event_inference")
    paths = (TEMPLATE, ROOT / "requirements.txt", *(source / (name + ".py") for name in modules))
    return dict(contract_sha256=_hash(CONTRACT), seed=SEED, noise_seed=SEED + 1, checkpoint_s=CHECKPOINT,
                implementation={str(path.relative_to(ROOT)): _hash(path) for path in paths})


def _verify_freeze():
    frozen = _read("freeze.json")
    if frozen != _freeze_record() or _hash(OUT / "contract.md") != frozen["contract_sha256"]:
        raise ValueError("frozen contract or source changed")


def freeze():
    OUT.mkdir(exist_ok=False)
    _put("contract.md", CONTRACT.read_bytes())
    _write(OUT / "freeze.json", _freeze_record())


def _prefix(flight, observe):
    while not flight.stop:
        flight._observe(observe)
        posterior = flight._fit((flight.active,))
        if flight.stop or flight.now == CHECKPOINT:
            flight._display(posterior[flight.active])
            return None if flight.stop else posterior[flight.active]
        if flight.now in (0, 12):
            review = flight._menu(posterior)
            if review:
                flight._apply(review[flight.active])
        flight._display(posterior[flight.active])
        if not flight.stop:
            flight._advance(posterior)
    return None


def _packet(flight, menu, evidence):
    rows = []
    for row in evidence["routes"]:
        pair = row["comparison"]
        tail = pair["candidate_after_fork"] if pair else {}
        sample = pair["next_ordinary_point_before_fork"] if pair else None
        rows.append([row["idx"], row["route_xy_m"], [round(stamp, 3) for stamp in row["waypoint_arrival_s"]],
                     *(round(value, 3) if value is not None else None for value in (
                         row["travel_s"], row["risk_proxy_tq"], row["unknown_s"], pair["fork_s"] if pair else None,
                         tail.get("joint_s"), tail.get("own_only_s"),
                         pair["delta_without_one_side_tq"] if pair else None)), sample])
    return dict(menu_id=menu["menu_id"], t_s=flight.now, position_xy_m=flight.position, goal_xy_m=flight.goal,
                flown_m=flight.flown, cap_m=flight.cap, energy_wh=flight.energy,
                reserve_wh=flight.aircraft.reserve_energy_wh, buildings=flight.geometry.config.buildings,
                point_rows=[[point["t_s"], *point["xy_m"], point["measured_dbm"]] for point in evidence["points"]],
                residual_rows=[[row["point_idx"], row["issued_s"], round(row["predicted_dbm"], 3),
                                round(row["observation_std_db"], 3), round(row["error_db"], 3),
                                round(row["standardized_error"], 3), row["support"]]
                               for row in evidence["forecast_residuals"][-8:]],
                incumbent_idx=menu["incumbent_idx"], routes=rows,
                waypoints=[[point["waypoint_idx"], point["xy_m"], point["completed_route_idx"]]
                           for point in menu["waypoints"]])


def _prompt(mode, packet):
    return (
        "Independent decision for a simulated flight. Use only this message. No tools, files, network, or questions. "
        "Minimize elapsed time plus power exposure while reaching the goal. All listed full remainders are legal. "
        "You may keep the incumbent. You do not have to imitate a numerical rule. Larger dBm is stronger. "
        "A GP-supported forecast is not a safety label. Unknown is not low interference; proxy cost is not truth. "
        "Point columns: [t,x,y,measured dBm]. Residual columns: [point index,issued t,predicted dBm,obs std,"
        "measured minus predicted dB,standardized residual,support]. Route columns: [id,xy path,arrival times,"
        "travel seconds,T+Q proxy,unknown seconds,fork time vs incumbent,joint supported seconds after fork,"
        "own-only supported seconds after fork,proxy change excluding one-sided support,next planned ordinary "
        "point before fork (not a reading)]. Negative proxy change is lower. "
        "Waypoint columns: [id,xy,route id selected by the shared whole-route proxy when using that waypoint]. "
        f"Choose mode {mode}: propose only one {mode} id or keep. Reply only one JSON line with exactly "
        f'{{"menu_id":"{packet["menu_id"]}","action":"{mode}","idx":0,"reason":"brief evidence"}}. '
        'For keeping use "action":"keep","idx":null. At most 1000 UTF-8 bytes.\n'
        + json.dumps(packet, allow_nan=False, separators=(",", ":"))
    )


def prepare():
    _verify_freeze()
    _put("progress.jsonl", b'{"status":"prefix_started"}\n')
    config = _config()
    world = ElectromagneticWorld(config)
    flight = Flight(world.geometry, config.aircraft, config.belief)
    try:
        posterior = _prefix(flight, world.observe)
        if posterior is None:
            _write(OUT / "prefix_only.json", flight.log("prefix_only"))
            _log("closed_without_checkpoint", stop=flight.stop)
            return
        counted = CountedPosterior(posterior, flight.ledger)
        flight.ledger["pricings"] += 1
        menu = build_route_action_menu(world.geometry, config.aircraft, counted, position=flight.position,
                                       incumbent=flight.incumbent, now=flight.now, flown_m=flight.flown,
                                       energy_wh=flight.energy)
        evidence = build_decision_evidence([row["route_xy_m"] for row in menu["routes"]], menu["incumbent_idx"],
                                           counted, flight.now, flight.speed, measurements=flight.points,
                                           frozen_forecasts=[frame["forecast"] for frame in flight.frames[1:]],
                                           next_sample_s=flight.now + 1)
    except QueryLimit:
        flight.stop = "compute_budget"
        _write(OUT / "prefix_only.json", flight.log("prefix_only"))
        _log("closed_without_checkpoint", stop=flight.stop)
        return
    if not menu["waypoints"] or not any(row["comparison"] for row in evidence["routes"]):
        _write(OUT / "prefix_only.json", flight.log("prefix_only"))
        _log("closed_without_checkpoint", stop="no_choice")
        return
    packet = _packet(flight, menu, evidence)
    requests = {f"request_{mode}.txt": _prompt(mode, packet).encode("utf-8") for mode in MODES}
    if any(len(value) > INPUT_BYTES for value in requests.values()):
        _write(OUT / "prefix_only.json", flight.log("prefix_only"))
        _log("closed_without_checkpoint", stop="input_budget")
        return
    _write(OUT / "public_checkpoint.json", packet)
    _write(OUT / "review.json", menu)
    _write(OUT / "checkpoint_state.json", dict(flight={key: getattr(flight, key) for key in STATE_KEYS},
                                              noise_state=world.random.bit_generator.state))
    for name, request in requests.items():
        _put(name, request)
    bound = ("public_checkpoint.json", "review.json", "checkpoint_state.json", *requests)
    hashes = {name: _hash(OUT / name) for name in bound}
    _write(OUT / "ready.json", dict(hashes=hashes, origin_s=time.monotonic()))
    _log("checkpoint_open", routes=len(menu["routes"]), waypoints=len(menu["waypoints"]))


def _ready():
    _verify_freeze()
    ready = _read("ready.json")
    if any(_hash(OUT / name) != sha for name, sha in ready["hashes"].items()):
        raise ValueError("frozen checkpoint or request changed")
    return ready


def _unique(pairs):
    decoded = dict(pairs)
    if len(decoded) != len(pairs):
        raise ValueError("duplicate response keys")
    return decoded


def _resolve(raw, elapsed, menu, mode):
    selected = menu["numeric_proxy_idx" if mode == "route" else "numeric_waypoint_route_idx"]
    receipt = dict(status="missing_reply", route_idx=selected, elapsed_s=elapsed, output_bytes=len(raw))
    if elapsed >= DEADLINE_S:
        return dict(receipt, status="timeout")
    if not raw:
        return receipt
    try:
        scannable = raw.decode("utf-8")
        if len(raw) > REPLY_BYTES or len(scannable.splitlines()) != 1:
            raise ValueError("expected a single bounded UTF-8 JSON line")
        reply = json.loads(scannable, object_pairs_hook=_unique)
        selected = resolve_route_action(menu, reply, mode=mode)
    except ValueError as error:
        return dict(receipt, status="invalid_reply", error=str(error))
    return dict(receipt, status="accepted", route_idx=selected, proposal=reply)


def submit(mode, text):
    ready = _ready()
    if (OUT / "execution_started.json").exists():
        raise ValueError("execution already closed the reply window")
    raw = text.encode("utf-8")
    elapsed = time.monotonic() - ready["origin_s"]
    _put(f"reply_{mode}.txt", raw)
    receipt = _resolve(raw, elapsed, _read("review.json"), mode)
    _write(OUT / f"receipt_{mode}.json", dict(receipt, raw_sha256=_hash(OUT / f"reply_{mode}.txt"),
                                              request_sha256=ready["hashes"][f"request_{mode}.txt"]))
    _log("receipt", mode=mode, reply_status=receipt["status"], elapsed_s=elapsed)


def _finish(flight, menu, selected, observe, posterior):
    reference = menu["incumbent_idx"]
    flight._apply(dict(candidates=[dict(route=row["route_xy_m"]) for row in menu["routes"]], selected=selected,
                       reference=reference, changed=reference != selected, qualified=1))
    try:
        while not flight.stop:
            flight._advance({flight.active: posterior})
            flight._observe(observe)
            posterior = flight._fit((flight.active,))[flight.active]
            flight._display(posterior)
    except QueryLimit:
        flight.stop = "compute_budget"


def _replay(arms, layout):
    grid = dict(rows=layout.rows, cols=layout.columns, spacing=layout.resolution_m,
                obstacles=[[*part["footprint_m"][0], *part["footprint_m"][2]] for part in layout.buildings],
                start=layout.start_m, goal=layout.goal_m)
    page = p4._replay(arms, grid)
    options = lambda labels: "".join(f'<option value="{key}">{label}</option>' for key, label in labels.items())
    edits = {f"P4 · 接入 LLM 前的数值闭环 · {p4.SEED}": "P7 · 路线与中间目标试飞",
             options(p4.LABELS): options(LABELS), json.dumps(p4.LABELS): json.dumps(LABELS),
             '<a href="public_fixed.json">': '<a href="public_checkpoint.json">',
             '<a href="public_event.json">': '<a href="review.json">',
             f"{p4.LABELS['fixed']} 日志": "公开决策包", f"{p4.LABELS['event']} 日志": "候选与数值对照",
             'arm.addEventListener("change",()=>{pause();configure();});':
                 'arm.addEventListener("change",()=>{pause();configure(Number(byId("time").value.split(" / ")[0]));});',
             '\nconfigure();': f'\nconfigure({CHECKPOINT:g});',
             '  const reason={initial:': '  if (decision?.t===frame.t && log.proposal?.waypoint_xy_m) {\n'
                 '    dot(belief,log.proposal.waypoint_xy_m,8,"#ffd169");\n'
                 '    text(belief,log.proposal.waypoint_xy_m,"W");\n  }\n  const reason={initial:'}
    for before, after in edits.items():
        if before not in page:
            raise ValueError(f"P7 replay marker missing: {before}")
        page = page.replace(before, after)
    return page


def execute():
    ready = _ready()
    elapsed = time.monotonic() - ready["origin_s"]
    menu = _read("review.json")
    receipts = {}
    for mode in MODES:
        path = OUT / f"receipt_{mode}.json"
        if not path.exists():
            if elapsed < DEADLINE_S:
                raise ValueError("await both single receipts or the frozen deadline")
            receipts[mode] = _resolve(b"", elapsed, menu, mode)
            continue
        receipt = _read(path.name)
        raw_path = OUT / f"reply_{mode}.txt"
        if (receipt["raw_sha256"] != _hash(raw_path)
                or receipt["request_sha256"] != ready["hashes"][f"request_{mode}.txt"]
                or _resolve(raw_path.read_bytes(), receipt["elapsed_s"], menu, mode) != {
                    key: value for key, value in receipt.items() if key not in ("raw_sha256", "request_sha256")}):
            raise ValueError("receipt mismatch")
        receipts[mode] = receipt
    _write(OUT / "execution_started.json", dict(receipts=receipts))
    config = _config()
    state = _read("checkpoint_state.json")
    choices = dict(llm_waypoint=receipts["waypoint"]["route_idx"], llm_route=receipts["route"]["route_idx"],
                   n_guard=menu["numeric_guarded_idx"], n_route=menu["numeric_proxy_idx"],
                   n_waypoint=menu["numeric_waypoint_route_idx"])
    executed = {}
    for selected in dict.fromkeys(choices.values()):
        world = ElectromagneticWorld(config)
        world.random.bit_generator.state = deepcopy(state["noise_state"])
        flight = Flight(world.geometry, config.aircraft, config.belief)
        for key, value in deepcopy(state["flight"]).items():
            setattr(flight, key, value)
        posterior = FixedPriorPosterior(flight.points, flight.geometry, flight.now, flight.settings[flight.active])
        _finish(flight, menu, selected, world.observe, posterior)
        log = flight.log(f"route_{selected}")
        _write(OUT / f"public_route_{selected}.json", log)
        executed[selected] = world, log, flight.traversed
        _log("flight_closed", route_idx=selected, stop=flight.stop, **flight.ledger)
    evaluations = {selected: p4._evaluate(*record) for selected, record in executed.items()}
    arms, metrics = {}, {}
    for name, selected in choices.items():
        log = dict(executed[selected][1])
        if name.startswith("llm_"):
            receipt = receipts[name.removeprefix("llm_")]
            reply = receipt.get("proposal", {})
            peak = menu["waypoints"][reply["idx"]]["xy_m"] if reply.get("action") == "waypoint" else None
            log["proposal"] = dict(status=receipt["status"], waypoint_xy_m=peak)
        evaluation = evaluations[selected]
        arms[name] = dict(log=log, evaluation=evaluation)
        metrics[name] = dict(route_idx=selected, **evaluation["metrics"], resources=log["resources"])
    _write(OUT / "results.json", dict(arms=metrics, receipts=receipts, distinct_flown_suffixes=len(executed),
                                      common_prefix_points=len(state["flight"]["points"]),
                                      restore_fits=len(executed)))
    _put("index.html", _replay(arms, config.map).encode("utf-8"))
    _log("complete", distinct_flown_suffixes=len(executed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("freeze", "prepare", "execute"):
        stages.add_parser(name)
    receipt = stages.add_parser("submit")
    receipt.add_argument("mode", choices=MODES)
    receipt.add_argument("reply")
    args = parser.parse_args()
    if args.stage == "submit":
        submit(args.mode, args.reply)
    else:
        {"freeze": freeze, "prepare": prepare, "execute": execute}[args.stage]()
