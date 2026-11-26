"""Freeze a one-shot static readback challenge; never fly or execute its reply."""

import json

import numpy as np

from .config import AircraftConfig, MapConfig
from .environment import Geometry
from .passive_closed_loop import CountedPosterior
from .passive_repair_checkpoint import _implementation_hashes, _json, _profile_record, _write
from .passive_repair_flight import _exchange
from .passive_repair_geometry import OnlineRepairGeometry
from .passive_repair_receiver import isolated_stdio_transport
from .passive_repair_tools import RepairToolWorkspace, _parse_call
from .passive_route_exchange import RouteExchange, RouteRequest, ToolCall, _sha256


PROBE_ID = "RC1-20261125"
TIMEOUT_S = 600
PROBE_INSTRUCTIONS = (
    "This is an isolated readback test, not a flight or a route recommendation. Only read the supplied input. "
    "Find the first observation power (state.points[0][3]), the y coordinate of the last comparison route "
    "at its first interior vertex, and the status of that last comparison. Return one JSON line with keys "
    "request_id, tool, args and reason. Copy input.request_id. Set tool to commit and args.route_id to the "
    "last comparison route_id. This tag is readback only and will never execute. Format the reason as "
    "first_dbm=<power with 3 decimal places>;last_via_y_m=<y with 2 decimal places>;last_status=<status> "
    "with no extra text or spaces. Do not guess, call external tools, or execute anything."
)


class _SupportedFixture:
    now = 59.0

    def predict(self, positions, times):
        return np.full(len(times), -74.), np.full(len(times), 2.), np.zeros(len(times), dtype=int)


def build_receiver_probe():
    """Build a public-only pressure packet and compute its deterministic readback."""
    start, goal, position = (20, 200), (1300, 200), (905, 200)
    aircraft = AircraftConfig()
    layout = MapConfig(columns=76, rows=26, resolution_m=20, start_m=start, goal_m=goal)
    geometry = Geometry(layout, aircraft.clearance_m)
    ledger = dict(queries=0, query_limit=11038)
    prefix = [(clock, 20 + 15 * clock, 200, -73.125 if clock == 0 else -74., None if clock == 0 else 0.)
              for clock in range(60)]
    workspace = RepairToolWorkspace(OnlineRepairGeometry(geometry, aircraft), [position, goal],
                                    CountedPosterior(_SupportedFixture(), ledger), now=59., flown_m=885.,
                                    energy_wh=aircraft.initial_energy_wh - 59 * aircraft.flight_power_w / 3600,
                                    points=prefix, peak_dbm=-74., min_post_s=0.5, ledger=ledger, geometry_limit=767)
    for index in range(160):
        generated = workspace.execute(ToolCall("generate", dict(operator="via", waypoint_xy_m=[1050, 220 + index / 4]), "fixture"))
        if generated["status"] != "generated" or generated["route_id"] != index + 1:
            raise ValueError("public probe lost one of its distinct legal routes")
        last = workspace.execute(ToolCall("compare", dict(route_id=index + 1, reference_id=0), "fixture"))
        if last["status"] != "comparable":
            raise ValueError("public probe lost its comparable evidence")
    draft = workspace._request(PROBE_ID, last)
    packet = json.loads(draft.wire)["input"]
    wire = json.dumps(dict(instructions=PROBE_INSTRUCTIONS, input=packet), allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode()
    tail = packet["comparisons"][-1]
    route_id = tail["route_id"]
    ordinate = packet["routes"][route_id]["xy_m"][1][1]
    first_dbm, status = packet["state"]["points"][0][3], tail["status"]
    reason = f"first_dbm={first_dbm:.3f};last_via_y_m={ordinate:.2f};last_status={status}"
    witness = dict(route_id=route_id, reason=reason)
    return RouteRequest(draft.request_id, wire, _sha256(wire), len(packet["routes"])), witness


def freeze_receiver_probe(out, contract, executable, *, profile):
    request, witness = build_receiver_probe()
    contract_bytes = contract.read_bytes()
    manifest = dict(probe=PROBE_ID, request_sha256=request.request_sha256, wire_bytes=len(request.wire),
                    readback=witness, contract_sha256=_sha256(contract_bytes), timeout_s=TIMEOUT_S, attempts=1,
                    profile=_profile_record(profile), executable_sha256=_sha256(executable.read_bytes()),
                    implementation=_implementation_hashes())
    out.mkdir(parents=True)
    manifest_bytes = _json(manifest)
    for name, data in (("contract.md", contract_bytes), ("public-request.json", request.wire), ("manifest.json", manifest_bytes)):
        _write(out / name, data)
    return _sha256(manifest_bytes)


def run_frozen_receiver_probe(out, executable, *, profile, approved_manifest_sha256):
    """One dispatch after an external approval pins the exact frozen manifest."""
    manifest_bytes = (out / "manifest.json").read_bytes()
    if _sha256(manifest_bytes) != approved_manifest_sha256:
        raise ValueError("receiver probe is not bound to the approved manifest")
    manifest = json.loads(manifest_bytes)
    request, witness = build_receiver_probe()
    if (manifest["probe"] != PROBE_ID or manifest["timeout_s"] != TIMEOUT_S or manifest["attempts"] != 1
            or manifest["contract_sha256"] != _sha256((out / "contract.md").read_bytes())
            or manifest["request_sha256"] != request.request_sha256 or manifest["wire_bytes"] != len(request.wire)
            or (out / "public-request.json").read_bytes() != request.wire or manifest["readback"] != witness
            or manifest["profile"] != _profile_record(profile) or manifest["implementation"] != _implementation_hashes()
            or manifest["executable_sha256"] != _sha256(executable.read_bytes())):
        raise ValueError("frozen receiver probe inputs changed")
    _write(out / "started.json", _json(dict(manifest_sha256=approved_manifest_sha256)))
    transport = isolated_stdio_transport(out / "rounds", executable, profile, timeout_s=TIMEOUT_S)
    receipt = RouteExchange().submit(request, transport, parser=_parse_call)
    call = receipt.selection
    matched = (call is not None and call.tool == "commit" and call.args == {"route_id": witness["route_id"]}
               and type(call.args["route_id"]) is int and call.reason == witness["reason"])
    status = "passed" if matched else "readback_mismatch" if call is not None else receipt.status
    result = dict(probe=PROBE_ID, status=status, new_points=0, advanced_m=0, executed_tool_calls=0,
                  exchange=_exchange(request, receipt))
    _write(out / "result.json", _json(result))
    return result
