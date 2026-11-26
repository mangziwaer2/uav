"""Freeze and run the independent EP1 static prediction contract exactly once."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from time import monotonic

import numpy as np
import scipy

from .equivalent_prediction import (FrozenMixture, INSTRUCTIONS, RESOURCE_KEYS, RollingGP, charged_resources,
                                    fit_structures, make_request, numerical_selection, parse_reply,
                                    pilot_feedback, validate_request, wire_bytes)
from .equivalent_prediction_samples import sample_order, synthetic_table
from .event_inference import FAMILIES, digest
from .passive_repair_checkpoint import _profile_record, minimal_profile
from .passive_repair_receiver import isolated_stdio_transport


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/equivalent_prediction_EP1_contract.md"
CONTRACT_HASH = "d1089682d307dfcc94b20f04e34967efa7440f469ef27b20923e94e80d9b39fc"
OUTPUT = ROOT / "outputs/equivalent_prediction_EP1_20261126"
MODEL = "gpt-6.2-sol"
ARMS = ("numerical_pair", "numerical_all", "llm_semantic", "llm_anonymous", "gp")
MODEL_SECONDS = 8640.


def exclusive_write(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def put_json(path, value):
    exclusive_write(path, wire_bytes(value))


class Journal:
    def __init__(self, path):
        self.path = path
        path.mkdir()
        self.sequence = 0
        self.previous = None

    def emit(self, kind, payload):
        record = dict(sequence=self.sequence, kind=kind, previous_sha256=self.previous,
                      utc=datetime.now(timezone.utc).isoformat(), payload=payload)
        raw = wire_bytes(record)
        exclusive_write(self.path / f"{self.sequence:04d}-{kind}.json", raw)
        self.previous = sha256(raw).hexdigest()
        self.sequence += 1


def environment_record():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


def implementation_files():
    return sorted([*Path(__file__).parent.glob("*.py"), ROOT / "requirements.txt",
                   ROOT / "tests/test_equivalent_prediction.py", ROOT / "tests/test_equivalent_prediction_ep1.py",
                   ROOT / "tests/test_equivalent_prediction_report.py"])


def frozen_profile(path):
    settings, connection = minimal_profile(path)
    return dict(settings, model=MODEL, model_reasoning_effort="medium"), connection


def freeze(out, executable, profile):
    raw_contract = CONTRACT.read_bytes()
    if sha256(raw_contract).hexdigest() != CONTRACT_HASH:
        raise ValueError("EP1 contract differs from the recorded research design")
    executable = executable.resolve(strict=True)
    files = implementation_files()
    manifest = dict(experiment="EP1", contract_sha256=CONTRACT_HASH, prompt_sha256=sha256(INSTRUCTIONS.encode()).hexdigest(),
                    environment=environment_record(), implementation={str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                                                                      for path in files},
                    receiver_sha256=sha256(executable.read_bytes()).hexdigest(), profile=_profile_record(profile),
                    order=[dict(layer=layer, repetition=rep) for layer, rep in sample_order()],
                    arms=list(ARMS), limits=dict(llm_requests=48, request_seconds=180, model_seconds=MODEL_SECONDS,
                                                fit_jobs=144, solver_calls=288, residual_calls=11520,
                                                jacobian_calls=11520, dictionary_kernel_terms=131040, gp_decompositions=144))
    out.mkdir()
    exclusive_write(out / "contract.md", raw_contract)
    exclusive_write(out / "prompt.txt", INSTRUCTIONS.encode())
    snapshot = out / "implementation"
    snapshot.mkdir()
    for path in files:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        exclusive_write(target, path.read_bytes())
    raw = wire_bytes(manifest)
    exclusive_write(out / "manifest.json", raw)
    return sha256(raw).hexdigest()


def verify_frozen(out, expected, executable, profile):
    raw = (out / "manifest.json").read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("wrong EP1 manifest binding")
    manifest = json.loads(raw)
    if (manifest["contract_sha256"] != CONTRACT_HASH
            or sha256((out / "contract.md").read_bytes()).hexdigest() != CONTRACT_HASH
            or sha256(CONTRACT.read_bytes()).hexdigest() != CONTRACT_HASH
            or manifest["prompt_sha256"] != sha256(INSTRUCTIONS.encode()).hexdigest()
            or (out / "prompt.txt").read_bytes() != INSTRUCTIONS.encode()
            or manifest["environment"] != environment_record()
            or manifest["profile"] != _profile_record(profile)
            or manifest["receiver_sha256"] != sha256(executable.read_bytes()).hexdigest()):
        raise ValueError("frozen EP1 contract, environment, prompt or receiver changed")
    for name, expected_hash in manifest["implementation"].items():
        if any(sha256(path.read_bytes()).hexdigest() != expected_hash
               for path in (ROOT / name, out / "implementation" / name)):
            raise ValueError(f"frozen EP1 implementation changed: {name}")
    return manifest


def select_llm(request, directory, executable, profile, remaining_s):
    validate_request(request)
    directory.mkdir()
    put_json(directory / "public.json", request)
    transport = isolated_stdio_transport(directory / "receiver", executable, profile, timeout_s=180,
                                         total_timeout_s=remaining_s)
    wire = wire_bytes(request)
    started = monotonic()
    status = dict(status="failed", request_id=request["request_id"], wire_sha256=sha256(wire).hexdigest(),
                  model=profile[0].get("model", MODEL),
                  reasoning_effort=profile[0].get("model_reasoning_effort", "medium"),
                  model_unavailable=False, integrity_error=False)
    try:
        reply = transport(wire, status["wire_sha256"])
        if reply.received_sha256 != status["wire_sha256"] or not reply.session_id:
            raise ValueError("isolated reply has no matching session and input digest")
        response = parse_reply(request, reply.raw_reply)
        status.update(status="selected", response=response, ids=response["ids"], session_id=reply.session_id,
                      raw_reply_sha256=sha256(reply.raw_reply).hexdigest())
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        status["reason"] = str(error)
    status["wall_seconds"] = monotonic() - started
    receiver = directory / "receiver"
    metadata = json.loads((receiver / "0001-receiver.json").read_bytes())
    status["receiver"] = metadata
    status["integrity_error"] = metadata["status"] in {"delivery_mismatch", "invalid_session_events", "credential_redacted"}
    event_text = (receiver / "0001-session.jsonl").read_text(errors="replace")
    diagnostic = event_text + (receiver / "0001-stderr.txt").read_text(errors="replace")
    lowered = diagnostic.lower()
    status["model_unavailable"] = status["status"] != "selected" and any(value in lowered for value in (
        "model_not_found", "unknown model", "unsupported model", "model is not supported", "model does not exist",
        "model not found", "invalid model", "no available channel", "model_not_available"))
    status["usage"] = None
    status.update(returned_model_id=None, system_fingerprint=None, upstream_identity_verified=False,
                  price=None, visible_network_error_events=0, visible_reconnect_events=0, network_attempts=None)
    for line in event_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            status["usage"] = event.get("usage")
        if isinstance(event.get("model"), str):
            status["returned_model_id"] = event["model"]
        if isinstance(event.get("system_fingerprint"), str):
            status["system_fingerprint"] = event["system_fingerprint"]
        message = event.get("message", "")
        if event.get("type") == "error" and isinstance(message, str):
            status["visible_network_error_events"] += int("status " in message or "stream" in message.lower())
            status["visible_reconnect_events"] += int(message.startswith("Reconnecting..."))
    put_json(directory / "selection.json", status)
    return status


def evaluate_case(table, directory, advisor):
    directory.mkdir()
    journal = Journal(directory / "journal")
    measurements = table["measurements"]
    put_json(directory / "private_table.json", table)
    pilot_fits = fit_structures(measurements[:8])
    journal.emit("pilot_fits_frozen", pilot_fits)
    feedback = pilot_feedback(pilot_fits, measurements[:8], measurements[8:12],
                              lambda kind, payload: journal.emit(f"pilot_{kind}", payload))
    semantic = make_request(measurements[:12], feedback, table["aliases"], semantic=True)
    anonymous = make_request(measurements[:12], feedback, table["aliases"], semantic=False)
    validate_request(semantic)
    validate_request(anonymous)
    selections = dict(numerical_pair=dict(status="selected", ids=numerical_selection(semantic)),
                      numerical_all=dict(status="selected", ids=["A", "B", "C"]),
                      gp=dict(status="selected", ids=[]))
    for arm, request in (("llm_semantic", semantic), ("llm_anonymous", anonymous)):
        selections[arm] = advisor(arm, deepcopy(request), directory / arm)
        journal.emit(f"{arm}_selection", selections[arm])
        if selections[arm].get("integrity_error") or selections[arm].get("model_unavailable"):
            break
    put_json(directory / "selections.json", selections)
    journal.emit("selections_frozen", selections)
    final_fits = fit_structures(measurements[:12])
    journal.emit("formal_fits_frozen", final_fits)
    forecasts, arms = {}, {}
    for arm in ARMS:
        choice = selections.get(arm, dict(status="not_requested", reason="batch stopped before this selection"))
        if choice["status"] != "selected":
            spent = charged_resources(pilot_fits, {}, [], pilot=True) if choice["status"] != "not_requested" else {}
            if "wall_seconds" in choice:
                spent.update(model_requests=1, model_wall_seconds=choice["wall_seconds"])
            arms[arm] = dict(choice, rows=[], resources=spent)
            continue
        if arm == "gp":
            forecasts[arm] = RollingGP(measurements[:12])
            resources = dict(gp_decompositions=0)
        else:
            families = [table["aliases"][alias] for alias in choice["ids"]]
            forecasts[arm] = FrozenMixture([final_fits[family]["fit"]["model"] for family in families],
                                           measurements[:12], update_weights=True)
            resources = charged_resources(pilot_fits, final_fits, families, pilot=arm != "numerical_all")
            caps = ((6, 240, 240, 3276) if arm == "numerical_all" else (10, 400, 400, 4992))
            if any(resources[key] > cap for key, cap in zip(RESOURCE_KEYS, caps)):
                raise ValueError("per-arm EP1 resource limit exceeded")
            if "wall_seconds" in choice:
                resources.update(model_requests=1, model_wall_seconds=choice["wall_seconds"])
        arms[arm] = dict(status="complete", ids=choice["ids"], rows=[], resources=resources,
                         selection=choice)
    for sample in measurements[12:]:
        issued = {arm: forecast.issue(sample["position_m"], sample["time_seconds"])
                  for arm, forecast in forecasts.items()}
        journal.emit("predictions_frozen", issued)
        journal.emit("observation_revealed", sample)
        for arm, forecast in forecasts.items():
            row = forecast.observe(sample)
            arms[arm]["rows"].append(row)
        journal.emit("scores_and_updates", {arm: dict(score=arms[arm]["rows"][-1],
                                                     log_weights=forecast.log_weights.tolist()
                                                     if isinstance(forecast, FrozenMixture) else None)
                                           for arm, forecast in forecasts.items()})
    arms["gp"]["resources"]["gp_decompositions"] = forecasts["gp"].decompositions
    physical = {key: sum(fit["resources"][key] for pool in (pilot_fits, final_fits) for fit in pool.values())
                for key in RESOURCE_KEYS}
    physical.update(fit_jobs=6, gp_decompositions=6)
    result = dict(case_id=directory.name, layer=table["layer"], repetition=table["repetition"], arms=arms,
                  status="complete" if all(arm["status"] == "complete" for arm in arms.values()) else "incomplete",
                  physical_resources=physical, last_journal_sha256=journal.previous,
                  stop_batch=any(choice.get("model_unavailable") or choice.get("integrity_error")
                                 for choice in selections.values()))
    put_json(directory / "result.json", result)
    return result


def run(out, expected, executable, profile):
    from .equivalent_prediction_report import publish_report

    manifest = verify_frozen(out, expected, executable, profile)
    put_json(out / "started.json", dict(manifest_sha256=expected, utc=datetime.now(timezone.utc).isoformat()))
    results = []
    calls, elapsed = 0, 0.

    def advisor(arm, request, directory):
        nonlocal calls, elapsed
        if calls >= 48 or elapsed >= MODEL_SECONDS:
            return dict(status="budget_exhausted", reason="EP1 total model budget exhausted")
        calls += 1
        selection = select_llm(request, directory, executable, profile, MODEL_SECONDS - elapsed)
        elapsed += selection["wall_seconds"]
        return selection

    stop = "completed"
    try:
        for index, record in enumerate(manifest["order"], 1):
            table = synthetic_table(record["layer"], record["repetition"])
            result = evaluate_case(table, out / f"case_{index:02d}", advisor)
            results.append(result)
            print(f"EP1 {index:02d}/24 {result['status']}; LLM requests={calls}, seconds={elapsed:.1f}", flush=True)
            if result["stop_batch"]:
                stop = "model_or_integrity_stop"
                break
        totals = {key: sum(result["physical_resources"][key] for result in results)
                  for key in (*RESOURCE_KEYS, "fit_jobs", "gp_decompositions")}
        if any(value > manifest["limits"][key] for key, value in totals.items()):
            raise ValueError("EP1 aggregate numerical resource violation")
    except Exception as error:
        stop = "integrity_stop"
        put_json(out / "failure.json", dict(type=type(error).__name__, reason=str(error)))
        print(f"EP1 stopped: {type(error).__name__}: {error}", flush=True)
    put_json(out / "closed.json", dict(stop=stop, completed_cases=len(results), llm_requests=calls,
                                       model_wall_seconds=elapsed, manifest_sha256=expected))
    summary = publish_report(out, results, dict(manifest, closure=json.loads((out / "closed.json").read_bytes())))
    print(f"EP1 report: {out / 'index.html'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    execute = commands.add_parser("run")
    execute.add_argument("digest")
    args = parser.parse_args()
    profile = frozen_profile(args.profile)
    if args.command == "freeze":
        print(freeze(args.out, args.executable, profile), flush=True)
    else:
        run(args.out, args.digest, args.executable, profile)


if __name__ == "__main__":
    main()
