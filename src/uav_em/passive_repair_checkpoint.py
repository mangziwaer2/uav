"""One-shot checkpoint acceptance, without sensing or flight execution."""

from dataclasses import replace
from html import escape
import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic

from .config import AircraftConfig, BeliefConfig, MapConfig
from .environment import Geometry
from .passive_closed_loop import CountedPosterior
from .passive_event_choice import alert
from .passive_map import FixedPriorPosterior
from .passive_model_advice import MODELS
from .passive_repair_flight import _exchange
from .passive_repair_geometry import OnlineRepairGeometry
from .passive_repair_tools import RepairToolWorkspace
from .passive_route_exchange import RouteExchange, _sha256, receive_route_request


CHECKPOINT_ID = "LC1-20261125"
FOLLOWUP_ID = "LC2-20261125"
LIMITS = dict(geometry=8, queries=192, seconds=360, min_post_s=0.5)
ISOLATION = ("The user message is the complete synthetic simulation checkpoint. Use only that message. "
             "Do not use tools, access files, consult other sessions or send commentary. Return exactly "
             "one raw JSON line with the requested tool call, without markdown. The caller executes it.")
ISOLATED_FLAGS = ("--strict-config", "--disable", "shell_tool", "--disable", "view_image", "--disable", "apps",
                  "--disable", "browser_use", "--disable", "computer_use", "-c", 'web_search="disabled"')


def extract_public(record):
    from base64 import b64decode

    if record.get("synthetic") is not True or record.get("scripted") is not True:
        raise ValueError("LC1 requires the closed synthetic interface record")
    log = record["arms"]["llm"]["log"]
    frame = next(row for row in log["frames"] if row["t"] == 1)
    event = next(row for row in log["decisions"] if row["t"] == 1)
    first = event["tool_trace"][0]["exchange"]
    wire = b64decode(first["request_b64"], validate=True)
    if first["request_sha256"] != _sha256(wire):
        raise ValueError("closed record first request hash mismatch")
    packet = json.loads(wire)["input"]
    state = packet["state"]
    if (packet["last_feedback"] is not None or packet["comparisons"] or len(packet["routes"]) != 1
            or state["t_s"] != 1 or len(state["points"]) != 2 or state["flown_m"] != frame["flown_m"]
            or state["energy_wh"] != frame["energy_left_wh"] or frame["active_model"] != "nominal"
            or packet["routes"][0]["xy_m"] != frame["incumbent_before"]):
        raise ValueError("closed record does not match the frozen first checkpoint")
    forecast = frame["forecast"]
    public_geometry = state["geometry"]
    return dict(
        t_s=state["t_s"], points=state["points"], incumbent=frame["incumbent_before"],
        start=state["points"][0][1:3], goal=state["goal_xy_m"], flown_m=state["flown_m"],
        cap_m=state["cap_m"], energy_wh=state["energy_wh"], peak_dbm=state["peak_dbm"],
        old_forecast={key: forecast[key] for key in ("issued_s", "arrival_s", "xy", "dbm", "error_db")},
        geometry={key: public_geometry[key] for key in ("max_xy_m", "mode", "buildings", "obstacles_m")},
        aircraft=dict(speed_mps=public_geometry["speed_mps"], altitude_m=public_geometry["altitude_m"],
                      clearance_m=public_geometry["clearance_m"],
                      vertical_clearance_m=public_geometry["vertical_clearance_m"],
                      flight_power_w=state["flight_power_w"], reserve_energy_wh=state["reserve_wh"]),
        resolution_m=20,
    )


def extract_followup_public(record, previous):
    if record.get("synthetic") is not True or previous["t_s"] != 1:
        raise ValueError("LC2 requires the synthetic prefix and the published LC1 layout")
    log = record["arms"]["numerical"]["log"]
    points, forecasts = [], []
    for frame in log["frames"]:
        if frame["t"] > 2:
            break
        clock, position, forecast = frame["t"], frame["xy"], frame["forecast"]
        if frame["active_model"] != "nominal" or frame["prefix_size"] != len(points) + 1:
            raise ValueError("unexpected checkpoint prefix or posterior version")
        points.append([clock, *position, frame["measured_dbm"], None if forecast is None else forecast["error_db"]])
        if forecast is not None:
            forecasts.append({key: forecast[key] for key in ("issued_s", "arrival_s", "xy", "dbm", "error_db")})
        if clock == 2:
            break
    if ([row[0] for row in points] != [0, 1, 2] or points[:2] != previous["points"]
            or log["cap_m"] != previous["cap_m"] or not frame["alert"]["active"]
            or frame["incumbent_before"] != [position, previous["goal"]]):
        raise ValueError("checkpoint is not the preselected straight-route prefix")
    return dict(t_s=clock, points=points, forecasts=forecasts, incumbent=frame["incumbent_before"],
                start=previous["start"], goal=previous["goal"], cap_m=log["cap_m"], flown_m=frame["flown_m"],
                energy_wh=frame["energy_left_wh"], peak_dbm=frame["alert"]["peak_dbm"],
                geometry=previous["geometry"], aircraft=previous["aircraft"], resolution_m=previous["resolution_m"])


def build_workspace(public):
    now, aircraft = public["t_s"], AircraftConfig(**public["aircraft"])
    layout, cell = public["geometry"], public["resolution_m"]
    width, height = layout["max_xy_m"]
    config = MapConfig(columns=int(width / cell) + 1, rows=int(height / cell) + 1, resolution_m=cell,
                       start_m=public["start"], goal_m=public["goal"], geometry_mode=layout["mode"],
                       obstacles_m=layout["obstacles_m"], buildings=layout["buildings"])
    geometry = Geometry(config, aircraft.clearance_m, aircraft.altitude_m, aircraft.vertical_clearance_m)
    points = [(clock, east, north, power, error) for clock, east, north, power, error in public["points"]]
    prefix = [dict(time_seconds=clock, position_m=(east, north), interference_dbm=power)
              for clock, east, north, power, _ in points]
    settings = replace(BeliefConfig(), gp_time_scale_seconds=MODELS["nominal"])
    forecasts = public["forecasts"] if "forecasts" in public else [public["old_forecast"]]
    if len(forecasts) != len(points) - 1 or points[-1][0] != now:
        raise ValueError("expected one causal forecast per observed arrival")
    for index, old in enumerate(forecasts, 1):
        arrival, east, north, power, error = points[index]
        issued = points[index - 1][0]
        if (old["issued_s"] != issued or old["arrival_s"] != arrival or tuple(old["xy"]) != (east, north)
                or not issued < arrival <= now):
            raise ValueError("frozen forecast does not verify against the observed arrival")
        earlier = FixedPriorPosterior(prefix[:index], geometry, issued, settings)
        mean, _, support = earlier.predict([old["xy"]], [arrival])
        residual = power - float(mean[0])
        if (support[0] >= 2 or not math.isclose(float(mean[0]), old["dbm"], abs_tol=1e-9)
                or not math.isclose(residual, old["error_db"], abs_tol=1e-9)
                or not math.isclose(residual, error, abs_tol=1e-9)):
            raise ValueError("frozen forecast does not verify against the observed arrival")
    posterior = FixedPriorPosterior(prefix, geometry, now, settings)
    peak, active = alert(public["incumbent"], posterior, now, aircraft.speed_mps)
    if not active or not math.isclose(peak, public["peak_dbm"], abs_tol=1e-9):
        raise ValueError("frozen alert cannot be reconstructed")
    ledger = dict(queries=0, query_limit=LIMITS["queries"], geometry_attempts=0, pricings=0)
    workspace = RepairToolWorkspace(
        OnlineRepairGeometry(geometry, aircraft), public["incumbent"], CountedPosterior(posterior, ledger),
        now=now, flown_m=public["flown_m"], energy_wh=public["energy_wh"], points=points, peak_dbm=peak,
        min_post_s=LIMITS["min_post_s"], ledger=ledger, geometry_limit=LIMITS["geometry"])
    if workspace.public["cap_m"] != public["cap_m"]:
        raise ValueError("checkpoint changed the original whole-route length cap")
    return workspace


def _write(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def _json(data):
    return (json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def minimal_profile(config_path):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    settings = {key: config[key] for key in ("model_provider", "model", "model_reasoning_effort")}
    provider = config["model_providers"][settings["model_provider"]]
    allowed = ("name", "base_url", "wire_api", "requires_openai_auth", "experimental_bearer_token")
    connection = {key: provider[key] for key in allowed if key in provider}
    return settings, connection


def _profile_code(profile):
    settings, connection = profile
    lines = [f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in settings.items()]
    lines += [f'[model_providers.{json.dumps(settings["model_provider"])}]']
    lines += [f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in connection.items()]
    return ("\n".join(lines) + "\n").encode()


def _profile_record(profile):
    settings, connection = profile
    public_connection = {key: value for key, value in connection.items() if key != "experimental_bearer_token"}
    fingerprint = _sha256(_profile_code((settings, public_connection)))
    return dict(settings=settings, public_config_sha256=fingerprint, flags=list(ISOLATED_FLAGS))


def _implementation_hashes():
    return {path.name: _sha256(path.read_bytes()) for path in sorted(Path(__file__).parent.glob("*.py"))}


def freeze(record_path, contract, out, executable, *, previous=None, profile=None):
    source = record_path.read_bytes()
    record = json.loads(source)
    checkpoint_id = CHECKPOINT_ID
    extra = {}
    if previous is None and profile is None:
        public = _json(extract_public(record))
    elif previous is not None and profile is not None:
        older = (previous / "public.json").read_bytes()
        older_manifest = json.loads((previous / "manifest.json").read_bytes())
        if older_manifest["checkpoint"] != CHECKPOINT_ID or older_manifest["public_sha256"] != _sha256(older):
            raise ValueError("LC1 public geometry does not match its published freeze")
        public = _json(extract_followup_public(record, json.loads(older)))
        checkpoint_id, extra = FOLLOWUP_ID, dict(layout_sha256=_sha256(older), profile=_profile_record(profile))
    else:
        raise ValueError("LC2 requires both the original public geometry and the isolated model profile")
    workspace = build_workspace(json.loads(public))
    contract_data = contract.read_bytes()
    first = workspace._request(checkpoint_id, None)
    manifest = dict(checkpoint=checkpoint_id, limits=LIMITS, source_sha256=_sha256(source),
                    contract_sha256=_sha256(contract_data), public_sha256=_sha256(public),
                    first_request_sha256=first.request_sha256, implementation=_implementation_hashes(),
                    receiver_sha256=_sha256(executable.read_bytes()), **extra)
    out.mkdir(parents=True)
    for name, data in (("contract.md", contract_data), ("public.json", public), ("first-request.json", first.wire),
                       ("manifest.json", _json(manifest))):
        _write(out / name, data)
    return manifest


def parse_session(events):
    session, final, complete = None, None, False
    for line in events.splitlines():
        row = json.loads(line)
        kind = row["type"]
        if kind == "thread.started" and session is None:
            session = row["thread_id"]
        elif kind == "turn.completed" and not complete:
            complete = True
        elif kind in ("item.started", "item.updated", "item.completed"):
            item = row["item"]
            if item["type"] == "error":
                continue
            if item["type"] not in ("reasoning", "agent_message"):
                raise ValueError("non-message activity in isolated session")
            if item["type"] == "agent_message" and kind == "item.completed":
                if final is not None:
                    raise ValueError("expected one model message")
                final = item["text"].encode("utf-8")
        elif kind not in ("turn.started", "error"):
            raise ValueError(f"unexpected session event: {kind}")
    if not session or not complete or final is None:
        raise ValueError("incomplete isolated session")
    return session, final


def _isolated_transport(out, executable, *, profile=None):
    rounds = out / "rounds"
    rounds.mkdir()
    deadline, turn = None, 0
    excluded = {"CODEX_THREAD_ID", "CODEX_SESSION_ID", "CODEX_TOOL_CALL_ID", "PYTHONPATH", "PWD", "OLDPWD"}
    env = {key: val for key, val in os.environ.items() if key not in excluded}
    if profile is not None:
        allowed = {"PATH", "HOME", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTPS_PROXY", "NO_PROXY"}
        env = {key: val for key, val in env.items() if key in allowed}

    def transport(wire, digest):
        nonlocal deadline, turn
        turn += 1
        prefix = f"{turn:02d}"
        _write(rounds / f"{prefix}-request.json", wire)

        def open_session(exact):
            nonlocal deadline
            if deadline is None:
                deadline = monotonic() + LIMITS["seconds"]
            remaining = deadline - monotonic()
            events, session, raw = b"", None, b""
            result = dict(received_sha256=_sha256(exact), status="wall_clock")
            if remaining > 0:
                with TemporaryDirectory(prefix="uav-checkpoint-") as directory:
                    work, child_env, flags = Path(directory), env, ()
                    if profile is not None:
                        home, work = work / "home", work / "empty"
                        home.mkdir(mode=0o700)
                        work.mkdir(mode=0o700)
                        descriptor = os.open(home / "config.toml", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with os.fdopen(descriptor, "wb") as stream:
                            stream.write(_profile_code(profile))
                        child_env, flags = dict(env, CODEX_HOME=str(home)), ISOLATED_FLAGS
                    command = [str(executable), "exec", "--json", "--ephemeral", "--skip-git-repo-check",
                               "--ignore-rules", "--sandbox", "read-only", "--color", "never",
                               *flags, "-c", f"developer_instructions={json.dumps(ISOLATION)}", exact.decode("utf-8")]
                    try:
                        process = subprocess.run(command, cwd=work, env=child_env, stdin=subprocess.DEVNULL,
                                                 capture_output=True, timeout=max(0.0, deadline - monotonic()), check=False)
                        events = process.stdout
                        result.update(status="process_failed", exit_code=process.returncode)
                        if process.returncode == 0:
                            session, raw = parse_session(events)
                            result["status"] = "delivered"
                    except subprocess.TimeoutExpired as error:
                        events = error.stdout or b""
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        result.update(status="receiver_rejected", detail=str(error))
            _write(rounds / f"{prefix}-session.jsonl", events)
            _write(rounds / f"{prefix}-reply.txt", raw)
            _write(rounds / f"{prefix}-receiver.json", _json(dict(result, session_id=session)))
            return session, raw

        return receive_route_request(wire, digest, open_session)

    return transport


def run_frozen(out, executable, *, profile=None):
    manifest_raw = (out / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    checkpoint_id = FOLLOWUP_ID if profile is not None else CHECKPOINT_ID
    for name, key in (("contract.md", "contract_sha256"), ("public.json", "public_sha256"),
                      ("first-request.json", "first_request_sha256")):
        if _sha256((out / name).read_bytes()) != manifest[key]:
            raise ValueError(f"changed checkpoint freeze: {name}")
    if (_implementation_hashes() != manifest["implementation"] or LIMITS != manifest["limits"]
            or manifest["checkpoint"] != checkpoint_id or _sha256(executable.read_bytes()) != manifest["receiver_sha256"]
            or (profile is not None and _profile_record(profile) != manifest["profile"])):
        raise ValueError("changed checkpoint implementation, resources or receiver")
    workspace = build_workspace(json.loads((out / "public.json").read_bytes()))
    if workspace._request(checkpoint_id, None).wire != (out / "first-request.json").read_bytes():
        raise ValueError("rebuilt first request differs from freeze")
    _write(out / "started.json", _json(dict(manifest_sha256=_sha256(manifest_raw))))
    status, selected = workspace.review(checkpoint_id, RouteExchange(), _isolated_transport(out, executable, profile=profile))
    evidence = dict(checkpoint=checkpoint_id, status=status, selected=selected, advanced_m=0, new_points=0,
                    resources=workspace.ledger, state=workspace.public,
                    routes=[workspace._route_row(index, route) for index, route in enumerate(workspace.routes)],
                    turns=[dict(exchange=_exchange(request, receipt), feedback=feedback)
                           for request, receipt, feedback in workspace.turns])
    _write(out / "evidence.json", _json(evidence))
    _write(out / "index.html", render_checkpoint(evidence).encode())
    return evidence


def render_checkpoint(evidence):
    state, selected = evidence["state"], evidence["selected"]
    label = escape(evidence["checkpoint"].split("-")[0])
    buildings = state["geometry"]["buildings"]
    routes = evidence["routes"]
    all_xy = [point for route in routes for point in route["xy_m"]]
    all_xy += [point for building in buildings for point in building["footprint_m"]]
    all_xy += [point[1:3] for point in state["points"]]
    east, north = zip(*all_xy)
    x0, y0 = min(east) - 15, max(north) + 15
    width, height = max(east) - x0 + 15, y0 - min(north) + 15

    def xy(point):
        return f"{point[0] - x0:.3f},{y0 - point[1]:.3f}"

    def coords(points):
        return " ".join(map(xy, points))

    drawing = [f'<polygon class="building" points="{coords(building["footprint_m"])}"/>' for building in buildings]
    for route in sorted(routes, key=lambda row: row["route_id"] == selected):
        route_id = route["route_id"]
        style = "chosen" if route_id == selected else "incumbent" if route_id == 0 else "candidate"
        drawing.append(f'<polyline class="{style}" points="{coords(route["xy_m"])}"><title>R{route_id}</title></polyline>')
    for clock, east, north, power, _ in state["points"]:
        x, y = xy((east, north)).split(",")
        drawing.append(f'<circle cx="{x}" cy="{y}" r="2"><title>t={clock}s, {power:.2f} dBm</title></circle>')
    rows = []
    for index, turn in enumerate(evidence["turns"], 1):
        exchange, feedback = turn["exchange"], turn["feedback"]
        detail = escape(json.dumps(feedback, ensure_ascii=False, indent=2))
        tool, status = (feedback["tool"], feedback["status"]) if feedback else ("-", exchange["status"])
        reason = escape(exchange["reason"] or "")
        rows.append(f'<li><strong>{escape(tool)} / {escape(status)}</strong><p>{reason}</p>'
                    f'<details><summary>工具反馈</summary><pre>{detail}</pre></details>'
                    f'<a href="rounds/{index:02d}-request.json">请求</a> · '
                    f'<a href="rounds/{index:02d}-reply.txt">原始回复</a> · '
                    f'<a href="rounds/{index:02d}-receiver.json">接收凭证</a> · '
                    f'<a href="rounds/{index:02d}-session.jsonl">原始会话</a></li>')
    final = "未提交" if selected is None else f"提交 R{selected}（未飞）"
    metrics = "".join(f'<li>R{row["route_id"]}：{row["distance_m"]:.2f} m / {row["time_s"]:.2f} s</li>' for row in routes)
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{label} 检查点回看</title>
<style>body{{font:16px system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#18252a}}
svg{{width:100%;max-height:440px;background:#edf3f3}}.building{{fill:#919da3}}polyline{{fill:none;stroke-width:1.4}}
.incumbent{{stroke:#435b69;stroke-dasharray:4 2}}.candidate{{stroke:#c2702c}}.chosen{{stroke:#087a68;stroke-width:2}}
circle{{fill:#152631}}li{{padding:12px 0;border-bottom:1px solid #ccd6d9}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
p{{overflow-wrap:anywhere}}a{{color:#126652}}</style>
<h1>{label} · 静止检查点</h1><p>t={state['t_s']:g} s · {final} · 新点测 0 · 推进 0 m · {escape(evidence['status'])}</p>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="公开建筑、已发生点测、原路与候选余程">{''.join(drawing)}</svg>
<p>灰色是建筑，黑点是已发生点测，灰蓝虚线是原路，橙色是生成候选，绿色是最终提交。所有余程均未飞，不显示真场。</p>
<h2>完整余程</h2><ul>{metrics}</ul>
<h2>模型与工具交互</h2><ol>{''.join(rows)}</ol>
<p><a href="evidence.json">完整证据</a> · <a href="first-request.json">冻结首包</a> ·
<a href="contract.md">合同</a> · <a href="manifest.json">冻结哈希</a></p></html>"""
