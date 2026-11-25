"""Synchronous exact-byte exchange for a public route decision, without a numerical fallback."""

from dataclasses import dataclass, replace
import hashlib
import json
import time
from uuid import UUID

from .passive_local_repair import LocalRepair, resolve_repair, waypoint_domain, xy


PUBLIC_FIELDS = frozenset(("request_id", "t_s", "goal", "buildings", "flown_m", "cap_m", "energy_wh",
                           "reserve_wh", "peak_dbm", "window_s", "incumbent", "new_first_targets", "points",
                           "routes"))
REPAIR_WAYPOINTS = "repair_waypoints_xy_m"


def _sha256(wire):
    return hashlib.sha256(wire).hexdigest()


@dataclass(frozen=True)
class RouteRequest:
    request_id: str
    wire: bytes
    request_sha256: str
    route_count: int


def seal_route_request(packet, instructions):
    """Serialize an allowlisted public packet once; pass the resulting bytes unchanged."""
    if not isinstance(packet, dict) or not PUBLIC_FIELDS <= packet.keys() <= PUBLIC_FIELDS | {REPAIR_WAYPOINTS}:
        raise ValueError("unexpected public route input fields")
    if REPAIR_WAYPOINTS in packet:
        waypoint_domain(packet[REPAIR_WAYPOINTS])
    request_id, routes = packet["request_id"], packet["routes"]
    if (not isinstance(instructions, str) or not instructions.strip() or not isinstance(request_id, str)
            or not request_id.strip() or not isinstance(routes, list) or not routes
            or any(not isinstance(row, (list, tuple)) or len(row) < 2 or type(row[0]) is not int or row[0] != index
                   for index, row in enumerate(routes))):
        raise ValueError("instructions, request ID and consecutive legal route IDs required")
    if any(not isinstance(point, (list, tuple)) or len(point) != 5
           or type(point[0]) not in (int, float) or not 0 <= point[0] <= packet["t_s"]
           for point in packet["points"]):
        raise ValueError("route requests can only contain past or current five-column observations")
    wire = json.dumps(dict(instructions=instructions, input=packet), allow_nan=False,
                      ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return RouteRequest(request_id, wire, _sha256(wire), len(routes))


@dataclass(frozen=True)
class RouteSelection:
    route_id: int
    reason: str
    repair: LocalRepair | None


@dataclass(frozen=True)
class ReceivedReply:
    received_sha256: str
    session_id: str | None
    raw_reply: bytes


def receive_route_request(wire, expected_sha256, open_session):
    """The receiving adapter hashes the actual bytes before opening a fresh session."""
    received_sha256 = _sha256(wire)
    if received_sha256 != expected_sha256:
        return ReceivedReply(received_sha256, None, b"")
    session_id, raw_reply = open_session(wire)
    return ReceivedReply(received_sha256, session_id, raw_reply)


def _unique_keys(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate reply key")
    return result


def _parse_reply(raw, request):
    line = raw.decode("utf-8").removesuffix("\n").removesuffix("\r")
    if "\n" in line or "\r" in line:
        raise ValueError("expected one reply line")
    reply = json.loads(line, object_pairs_hook=_unique_keys)
    required = {"request_id", "route_id", "reason"}
    if (not isinstance(reply, dict) or not required <= reply.keys() <= required | {"repair"}
            or reply["request_id"] != request.request_id):
        raise ValueError("reply does not match the request")
    route_id, reason = reply["route_id"], reply["reason"]
    if (type(route_id) is not int or not 0 <= route_id < request.route_count
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("explicit legal route selection and reason required")
    repair = None
    if "repair" in reply:
        edit = reply["repair"]
        anchors = {"left", "right"}
        if (not isinstance(edit, dict) or not anchors <= edit.keys() <= anchors | {"waypoint_xy_m"}
                or any(type(edit[key]) is not int for key in anchors)):
            raise ValueError("invalid repair parameters")
        via = xy(edit["waypoint_xy_m"]) if "waypoint_xy_m" in edit else None
        repair = LocalRepair(edit["left"], edit["right"], via)
    return RouteSelection(route_id, reason, repair)


@dataclass(frozen=True)
class RouteReceipt:
    request_id: str
    request_sha256: str
    received_sha256: str | None
    session_id: str | None
    raw_reply: bytes
    status: str
    selection: RouteSelection | None
    elapsed_s: float

    @property
    def raw_sha256(self):
        return _sha256(self.raw_reply)


class RouteExchange:
    def __init__(self):
        self._sessions = set()

    def submit(self, request, exchange):
        """Exchange once through a trusted receiving adapter and retain the raw response."""
        start = time.monotonic()
        delivered = None

        def finish(status, session_id=None, selection=None):
            return RouteReceipt(request.request_id, request.request_sha256,
                                delivered.received_sha256 if delivered is not None else None, session_id,
                                delivered.raw_reply if delivered is not None else b"", status, selection,
                                time.monotonic() - start)

        if _sha256(request.wire) != request.request_sha256:
            return finish("request_mismatch")
        try:
            delivered = exchange(request.wire, request.request_sha256)
        except OSError:
            return finish("delivery_error")
        if delivered is None:
            return finish("missing_reply")
        if delivered.received_sha256 != request.request_sha256:
            return finish("delivery_mismatch")
        try:
            session_id = str(UUID(delivered.session_id))
        except (AttributeError, TypeError, ValueError):
            return finish("invalid_session")
        if session_id in self._sessions:
            return finish("reused_session", session_id)
        self._sessions.add(session_id)
        if not delivered.raw_reply:
            return finish("missing_reply", session_id)
        try:
            selection = _parse_reply(delivered.raw_reply, request)
        except (ValueError, RecursionError):
            return finish("invalid_reply", session_id)
        return finish("accepted", session_id, selection)


def resolve_route_receipt(request, receipt, *, geometry, aircraft, position, now, flown_m, energy_wh):
    """Resolve an accepted choice against the exact menu that was sent, with no GP veto."""
    if (receipt.status != "accepted" or receipt.selection is None or receipt.request_id != request.request_id
            or receipt.request_sha256 != request.request_sha256 or receipt.received_sha256 != request.request_sha256
            or _sha256(request.wire) != request.request_sha256):
        raise ValueError("route receipt is not bound to this frozen request")
    packet = json.loads(request.wire)["input"]
    if (packet["t_s"] != now or packet["flown_m"] != flown_m or packet["energy_wh"] != energy_wh
            or xy(packet["goal"]) != xy(geometry.config.goal_m)):
        raise ValueError("route request does not match the current flight state")
    selection = receipt.selection
    edit = selection.repair
    outside_domain = (REPAIR_WAYPOINTS in packet and edit is not None and edit.waypoint_xy_m is not None
                      and edit.waypoint_xy_m not in waypoint_domain(packet[REPAIR_WAYPOINTS]))
    resolved = resolve_repair(packet["routes"][selection.route_id][1], None if outside_domain else edit,
                              geometry=geometry, aircraft=aircraft, position=position, now=now,
                              flown_m=flown_m, energy_wh=energy_wh)
    return replace(resolved, status="outside_repair_domain") if outside_domain else resolved
