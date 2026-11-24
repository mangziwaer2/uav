"""Freeze and execute a single E3 simulated flight pair."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from uuid import UUID

import numpy as np

from .config import ProjectConfig
from .environment import ElectromagneticWorld
from .passive_branch_event import BranchOpportunity, price_branches
from .passive_closed_loop import CountedPosterior, MAX_POINTS, MAX_QUERIES, QueryLimit
from .passive_decision_replay import DATA, _payload, render
from .passive_event_choice import HORIZON_S, alert, menu
from .passive_event_choice_pilot import _layout, _sha, parse_reply
from .passive_map import LIMIT_S
from .passive_map_pilot import _write
from .passive_model_advice import Flight
from .passive_model_advice_publish import evaluate
from .planning import path_length


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/passive_branch_event_E3_20261124"
CONTRACT = ROOT / "docs/passive_branch_event_E3_contract.md"
SEED = 43821
LABELS = {"llm": "LLM · 分岔选路", "numerical": "强 N · 共同上尾定价"}
PACKET_KEYS = ("request_id", "t_s", "goal", "buildings", "flown_m", "cap_m", "energy_wh", "reserve_wh",
               "peak_dbm", "window_s", "incumbent", "new_first_targets", "points", "routes")
PROMPT = """You are the independent route selector in a civilian UAV simulation. Use only the JSON below. Do not use tools, files, other sessions or outside information. Explicitly select one listed legal complete route, including the incumbent when appropriate. Minimize actual time T plus integrated normalized linear interference Q. Route rows are [id, XY path, full remaining T, offset to split, difference in T versus incumbent, comparable seconds after split, difference in expected Q, difference in pointwise upper-tail Q, unsupported window seconds, supported but not comparable seconds, outside window seconds]. Differences are candidate minus incumbent on the same support; lower is better. Null means no comparable Q, not zero exposure. All Q values are partial predictions, not whole-route costs. Real points are [second,x,y,dBm,prior prediction error dB]. Larger dBm is stronger; only infer direction from supported evidence. Do not infer hidden sources or treat unknown as low or high measured power. Weigh the travel-time cost, observed trends and forecast uncertainty. Return exactly one line: {"request_id":"...","route_id":0,"reason":"short evidence-based reason"}. No markdown or abstention.
"""


def _public():
    base = ProjectConfig()
    return dict(map=asdict(_layout(base.map)), aircraft=asdict(base.aircraft), belief=asdict(base.belief),
                sensor="ordinary position/time/noisy total dBm", packet_keys=PACKET_KEYS, prompt=PROMPT,
                resources=dict(point_fits=MAX_POINTS, queries=MAX_QUERIES, simulated_s=LIMIT_S,
                               routes_per_review=126, reviews_per_frame=1, request_bytes=64_000,
                               reply_bytes=800, consultations_per_flight=None, wait_s=None))


def _frozen():
    paths = [path for path in Path(__file__).parent.iterdir() if path.suffix in (".py", ".js", ".html")]
    paths.append(ROOT / "requirements.txt")
    return dict(contract_sha256=_sha(CONTRACT.read_bytes()), public_sha256=_sha(_payload(_public()).encode()),
                seed=SEED, arms=list(LABELS),
                implementation={str(path.relative_to(ROOT)): _sha(path.read_bytes()) for path in sorted(paths)})


def freeze():
    OUT.mkdir(exist_ok=False)
    (OUT / "contract.md").write_bytes(CONTRACT.read_bytes())
    (OUT / "public_inputs.json").write_text(_payload(_public()), encoding="utf-8")
    frozen = _frozen()
    _write(OUT / "freeze.json", frozen)
    print(frozen["contract_sha256"], flush=True)


def _world():
    rng = np.random.default_rng(SEED)
    fields = tuple(dict(center_m=(np.asarray(center) + rng.uniform(-12, 12, 2)).tolist(), velocity_mps=velocity,
                        sigma_m=sigma, peak_dbm=peak, start_seconds=start, end_seconds=end)
                   for center, velocity, sigma, peak, start, end in (
                       ((227, 361), (0.70, -0.30), 45, -66, 0, 90),
                       ((492, 392), (-0.55, 0.50), 37, -68, 19, 65),
                   ))
    base = ProjectConfig(seed=SEED + 1)
    return ElectromagneticWorld(replace(base, map=_layout(base.map), electromagnetic=replace(
        base.electromagnetic, fields=fields, background_dbm=-100, measurement_noise_db=1)))


def _packet(flight, rows, reference, peak, targets, request_id):
    def rounded(value):
        return None if value is None else round(value, 3)

    points = [[frame["t"], *frame["xy"], rounded(frame["measured_dbm"]),
               rounded((frame.get("forecast") or {}).get("error_db"))] for frame in flight.frames[-12:]]
    metrics = ("time_s", "split_s", "delta_t_s", "post_common_s", "delta_expected_q", "delta_tail_q", "unknown_s",
               "supported_not_compared_s", "outside_s")
    packet = dict(request_id=request_id, t_s=flight.now, goal=flight.goal, buildings=flight.geometry.config.buildings,
                  flown_m=flight.flown, cap_m=flight.cap, energy_wh=flight.energy,
                  reserve_wh=flight.aircraft.reserve_energy_wh, peak_dbm=rounded(peak), window_s=HORIZON_S,
                  incumbent=reference, new_first_targets=targets, points=points,
                  routes=[[index, row["route"], *(rounded(row[key]) for key in metrics)] for index, row in enumerate(rows)])
    if tuple(packet) != PACKET_KEYS:
        raise ValueError("unexpected public input field")
    return packet


def _request(packet, sessions):
    name = packet["request_id"]
    message = PROMPT + _payload(packet)
    if len(message.encode()) > 64_000:
        return None, "request_budget"
    (OUT / f"request_{name}.txt").write_text(message, encoding="utf-8")
    _write(OUT / f"packet_{name}.json", packet)
    print(_payload(dict(status="awaiting", t=packet["t_s"], request=name, path=str(OUT / f"request_{name}.txt"))), flush=True)
    started = time.monotonic()
    envelope = sys.stdin.buffer.readline(8192)
    elapsed = time.monotonic() - started
    raw, session, reply = b"", None, None
    status = "invalid_reply" if envelope else "missing_reply"
    try:
        submission = json.loads(envelope)
        raw = submission["reply"].encode()
        session = str(UUID(submission["session_id"]))
        if session in sessions:
            raise ValueError("decision session was reused")
        sessions.add(session)
        reply, status = parse_reply(raw, packet), "accepted"
    except (AttributeError, TypeError, KeyError, ValueError, UnicodeError):
        pass
    (OUT / f"raw_{name}.txt").write_bytes(raw)
    _write(OUT / f"receipt_{name}.json", dict(status=status, elapsed_s=elapsed, request_sha256=_sha(message.encode()),
                                             raw_sha256=_sha(raw), session_id=session, response=reply))
    return reply, status


def fly(world, arm, choose):
    flight = Flight(world.geometry, world.config.aircraft, world.config.belief)
    opportunities = BranchOpportunity(world.geometry.config.resolution_m)
    issued = 0
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
            routes = menu(flight)[0] if initial or high else []
            if (initial or high) and not routes:
                flight.stop = "no_legal_remaining_route"
                break
            state, targets = ("initial", []) if initial else opportunities.check(high, flight.incumbent, routes, flight.speed)
            frame = flight.frames[-1]
            frame["alert"] = dict(peak_dbm=peak, active=high, state=state, new_first_targets=targets)
            if initial or targets:
                rows, numerical, reference = price_branches(routes, counted, flight.now, flight.speed, flight.incumbent)
                flight.ledger["pricings"] += 1
                selected, authority, request_id = numerical, "numerical", None
                if initial:
                    selected, authority = min(range(len(routes)), key=lambda idx: (path_length(routes[idx]), idx)), "initial"
                    numerical = selected
                elif arm == "llm":
                    issued += 1
                    request_id = f"E3-{issued:02d}"
                    reply, status = choose(_packet(flight, rows, reference, peak, targets, request_id))
                    selected, authority = (reply["route_id"], "llm") if reply else (None, status)
                    if reply is None:
                        flight.stop = status
                        frame["alert"]["state"] = status
                flight.decisions.append(dict(t=flight.now, candidates=rows, selected=selected, reference=reference,
                                             numerical=numerical, reason=state, authority=authority,
                                             request_id=request_id, changed=reference is not None and selected is not None
                                             and selected != reference))
                frame["decision"] = len(flight.decisions) - 1
                if selected is None:
                    break
                flight.ledger["reviews"] += 1
                flight.incumbent = frame["remaining"] = routes[selected]
            flight._advance({flight.active: posterior})
        except QueryLimit:
            flight.stop = "compute_budget"
    return dict(flight.log(arm), consultations=issued)


def _publish(world, arms):
    layout = world.geometry.config
    grid = dict(rows=layout.rows, cols=layout.columns, spacing=layout.resolution_m, start=layout.start_m,
                goal=layout.goal_m, obstacles=[[*part["footprint_m"][0], *part["footprint_m"][2]] for part in layout.buildings])
    page = render(DATA + _payload(dict(grid=grid, arms=arms)) + "</script>", LABELS,
                  "E3 · 新首分岔试飞", "results.json")
    page = page.replace(">原回放</a>", '>结果数据</a><a href="contract.md">冻结合同</a>')
    page = page.replace("<th>T (s)</th><th>未知 (s)</th>", "<th>余程 T (s)</th><th>窗内未知 (s)</th>")
    page = page.replace('rearmed:"未预警"', 'initial:"起飞", new_branch:"新首分岔", missing_reply:"无回执", rearmed:"未预警"')
    page = page.replace('snapshot.authority === "llm" ? "LLM" : "N"',
                        'snapshot.authority === "llm" ? "LLM" : snapshot.authority === "initial" ? "共同起飞" : "N"')
    page = page.replace('<h2>全程实际结果</h2>', '''<section class="summary" aria-labelledby="branch-title">
<h2 id="branch-title">分岔后的可比证据</h2><p id="branch-meta"></p>
<table><thead><tr><th>路线</th><th>ΔT (s)</th><th>分岔后可比 (s)</th><th>期望 ΔQ</th>
<th>上尾 ΔQ</th><th>支持但未比 (s)</th><th>窗外 (s)</th></tr></thead><tbody id="branch-rows"></tbody></table>
</section><h2>全程实际结果</h2>''')
    page = page.replace("  updateDecisionControls(mode, events, snapshot);", '''  updateDecisionControls(mode, events, snapshot);
  const branchRows = byId("branch-rows");
  branchRows.replaceChildren();
  byId("branch-meta").textContent = decision?.reference == null ? "共同起飞，无原路对照。" :
    `相对该次原路 R${decision.reference}；未比较不代表零暴露。`;
  if (decision?.reference != null) for (const [id, item] of decision.candidates.entries()) {
    const entry = branchRows.insertRow();
    for (const value of [routeLabel(id), item.delta_t_s, item.post_common_s, item.delta_expected_q, item.delta_tail_q,
                         item.supported_not_compared_s, item.outside_s]) {
      entry.insertCell().textContent = typeof value === "number" ? fmt(value) : value ?? "—";
    }
  }''')
    page = page.replace('stops[metrics.stop],', '(stops[metrics.stop] ?? metrics.stop),')
    (OUT / "index.html").write_text(page, encoding="utf-8")


def execute(digest):
    frozen = json.loads((OUT / "freeze.json").read_bytes())
    if (frozen != _frozen() or digest != frozen["contract_sha256"]
            or digest != _sha((OUT / "contract.md").read_bytes())
            or _sha((OUT / "public_inputs.json").read_bytes()) != frozen["public_sha256"]):
        raise ValueError("E3 contract, implementation or public inputs were not frozen")
    (OUT / "started").touch(exist_ok=False)
    logs, sessions = {}, set()
    for arm in LABELS:
        logs[arm] = fly(_world(), arm, lambda packet: _request(packet, sessions))
        _write(OUT / f"public_{arm}.json", logs[arm])
    world = _world()
    arms = {arm: dict(log=log, evaluation=evaluate(world, log)) for arm, log in logs.items()}
    results = {arm: dict(metrics=entry["evaluation"]["metrics"], consultations=logs[arm]["consultations"],
                         events=[{key: event[key] for key in ("t", "selected", "reference", "numerical", "authority",
                                                             "changed", "reason", "request_id")}
                                 for event in logs[arm]["decisions"]]) for arm, entry in arms.items()}
    metrics = {arm: data["metrics"] for arm, data in results.items()}
    paired = all(row["legal"] and row["stop"] == "arrived" for row in metrics.values())
    delta = {key: metrics["llm"][key] - metrics["numerical"][key] for key in ("T", "Q", "T_plus_Q")} if paired else None
    _write(OUT / "results.json", dict(seed=SEED, arms=results, llm_minus_n=delta,
                                     wait_s=sum(json.loads(path.read_bytes())["elapsed_s"]
                                                for path in OUT.glob("receipt_*.json"))))
    _publish(world, arms)
    (OUT / "closed").touch(exist_ok=False)
    print(_payload(dict(status="closed", replay=str(OUT / "index.html"), llm_minus_n=delta)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--execute", metavar="CONTRACT_SHA256")
    args = parser.parse_args()
    freeze() if args.freeze else execute(args.execute)
