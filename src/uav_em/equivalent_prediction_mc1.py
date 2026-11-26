"""Frozen three-configuration calibration using only public prediction prefixes."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .equivalent_prediction import (FrozenMixture, INSTRUCTIONS, RESOURCE_KEYS, RollingGP,
                                    charged_resources, fit_structures, make_request, numerical_selection,
                                    pilot_feedback, validate_request, wire_bytes)
from .equivalent_prediction_ep1 import Journal, environment_record, exclusive_write, put_json, select_llm
from .equivalent_prediction_samples import sample_order, synthetic_table
from .passive_repair_checkpoint import _profile_record, minimal_profile


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/model_calibration_MC1_contract.md"
PLAN = ROOT / "docs/reference/llm_calibration_plan.json"
OUTPUT = ROOT / "outputs/model_calibration_MC1_20261126"
CANDIDATES = (
    dict(key="luna", model="gpt-5.6-luna", reasoning_effort="low"),
    dict(key="sol56", model="gpt-5.6-sol", reasoning_effort="medium"),
    dict(key="sol61", model="gpt-6.1-sol", reasoning_effort="medium"),
)
ARMS = ("numerical_pair", "numerical_all", "gp", "luna", "sol56", "sol61")
LIMITS = dict(llm_requests=36, model_seconds=6480, fit_jobs=72, solver_calls=144,
              residual_calls=5760, jacobian_calls=5760, dictionary_kernel_terms=65520,
              forecast_kernel_terms=1872, gp_decompositions=72)


def candidate_profiles(config):
    settings, connection = minimal_profile(config)
    return {item["key"]: (dict(settings, model=item["model"], model_reasoning_effort=item["reasoning_effort"]),
                           deepcopy(connection)) for item in CANDIDATES}


def source_files():
    return sorted([*Path(__file__).parent.glob("*.py"), ROOT / "requirements.txt",
                   *ROOT.glob("tests/test_equivalent_prediction*.py"),
                   ROOT / "tests/test_passive_repair_checkpoint.py", ROOT / "tests/test_passive_repair_receiver.py"])


def freeze(out, executable, profiles):
    plan = json.loads(PLAN.read_bytes())
    expected = [(item["model"], item["reasoning_effort"]) for item in CANDIDATES]
    if [(item["requested_model_id"], item["requested_reasoning_effort"]) for item in plan["candidates"]] != expected:
        raise ValueError("MC1 configurations differ from the declared plan")
    files = source_files()
    order = []
    keys = [item["key"] for item in CANDIDATES]
    for layer, rep in sample_order(namespace="MC1-v1", repetitions=3):
        rotation = (layer + rep) % 3
        order.append(dict(layer=layer, repetition=rep, model_order=keys[rotation:] + keys[:rotation]))
    manifest = dict(experiment="MC1", contract_sha256=sha256(CONTRACT.read_bytes()).hexdigest(),
                    plan_sha256=sha256(PLAN.read_bytes()).hexdigest(), prompt_sha256=sha256(INSTRUCTIONS.encode()).hexdigest(),
                    environment=environment_record(), candidates=list(CANDIDATES), arms=list(ARMS), order=order,
                    profiles={key: _profile_record(value) for key, value in profiles.items()},
                    receiver_sha256=sha256(executable.read_bytes()).hexdigest(), limits=LIMITS,
                    source={str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in files})
    out.mkdir()
    exclusive_write(out / "contract.md", CONTRACT.read_bytes())
    exclusive_write(out / "configuration.json", PLAN.read_bytes())
    exclusive_write(out / "prompt.txt", INSTRUCTIONS.encode())
    for path in files:
        target = out / "implementation" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        exclusive_write(target, path.read_bytes())
    raw = wire_bytes(manifest)
    exclusive_write(out / "manifest.json", raw)
    return sha256(raw).hexdigest()


def verify(out, expected, executable, profiles):
    raw = (out / "manifest.json").read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("MC1 manifest binding mismatch")
    manifest = json.loads(raw)
    for filename, current, field in (("contract.md", CONTRACT, "contract_sha256"),
                                      ("configuration.json", PLAN, "plan_sha256")):
        if any(sha256(path.read_bytes()).hexdigest() != manifest[field] for path in (out / filename, current)):
            raise ValueError("MC1 contract or model plan changed")
    if (manifest["profiles"] != {key: _profile_record(value) for key, value in profiles.items()}
            or manifest["environment"] != environment_record()
            or manifest["receiver_sha256"] != sha256(executable.read_bytes()).hexdigest()
            or manifest["prompt_sha256"] != sha256(INSTRUCTIONS.encode()).hexdigest()
            or (out / "prompt.txt").read_bytes() != INSTRUCTIONS.encode()
            or manifest["candidates"] != list(CANDIDATES) or manifest["limits"] != LIMITS):
        raise ValueError("MC1 receiver, prompt, environment or model profile changed")
    for name, fingerprint in manifest["source"].items():
        if any(sha256(path.read_bytes()).hexdigest() != fingerprint
               for path in (ROOT / name, out / "implementation" / name)):
            raise ValueError(f"MC1 frozen source changed: {name}")
    return manifest


def evaluate_case(table, directory, advisor, model_order):
    if sorted(model_order) != sorted(item["key"] for item in CANDIDATES):
        raise ValueError("each MC1 configuration must appear exactly once")
    directory.mkdir()
    journal = Journal(directory / "journal")
    put_json(directory / "private_table.json", table)
    history = table["measurements"]
    pilot = fit_structures(history[:8])
    journal.emit("pilot_fits_frozen", pilot)
    feedback = pilot_feedback(pilot, history[:8], history[8:12],
                              lambda kind, payload: journal.emit(f"pilot_{kind}", payload))
    request = make_request(history[:12], feedback, table["aliases"], semantic=True)
    validate_request(request)
    put_json(directory / "public.json", request)
    selections = dict(numerical_pair=dict(status="selected", ids=numerical_selection(request)),
                      numerical_all=dict(status="selected", ids=["A", "B", "C"]), gp=dict(status="selected", ids=[]))
    for key in model_order:
        selections[key] = advisor(key, deepcopy(request), directory / key)
        journal.emit(f"{key}_selection", selections[key])
        if selections[key].get("integrity_error"):
            raise ValueError("MC1 isolated model integrity failure")
    put_json(directory / "selections.json", selections)
    journal.emit("selections_frozen", selections)
    fitted = fit_structures(history[:12])
    journal.emit("formal_fits_frozen", fitted)
    forecasts, arms = {}, {}
    forecast_terms = 4 * sum(len(item["fit"]["model"]["bases"]) for item in pilot.values())
    for key in ARMS:
        selected = selections[key]
        if selected["status"] != "selected":
            attempted = "receiver" in selected
            resources = charged_resources(pilot, {}, [], pilot=True) if attempted else {}
            if attempted:
                resources.update(model_requests=1, model_wall_seconds=selected["wall_seconds"])
            arms[key] = dict(status=selected["status"], reason=selected.get("reason"), ids=[], rows=[],
                             resources=resources, selection=selected)
            continue
        if key == "gp":
            forecasts[key] = RollingGP(history[:12])
            resources = dict(gp_decompositions=6)
        else:
            families = [table["aliases"][alias] for alias in selected["ids"]]
            models = [fitted[family]["fit"]["model"] for family in families]
            forecasts[key] = FrozenMixture(models, history[:12], update_weights=True)
            resources = charged_resources(pilot, fitted, families, pilot=key != "numerical_all")
            caps = (6, 240, 240, 3276) if key == "numerical_all" else (10, 400, 400, 4992)
            if any(resources[field] > cap for field, cap in zip(RESOURCE_KEYS, caps)):
                raise ValueError("MC1 per-arm numeric resource violation")
            forecast_terms += 6 * sum(len(model["bases"]) for model in models)
            if "wall_seconds" in selected:
                resources.update(model_requests=1, model_wall_seconds=selected["wall_seconds"])
        arms[key] = dict(status="complete", ids=selected["ids"], rows=[], resources=resources, selection=selected)
    for sample in history[12:]:
        issued = {key: predictor.issue(sample["position_m"], sample["time_seconds"])
                  for key, predictor in forecasts.items()}
        journal.emit("predictions_frozen", issued)
        journal.emit("observation_revealed", sample)
        for key, predictor in forecasts.items():
            arms[key]["rows"].append(predictor.observe(sample))
        journal.emit("scores_and_updates", {key: dict(score=arms[key]["rows"][-1],
                                                     log_weights=predictor.log_weights.tolist()
                                                     if isinstance(predictor, FrozenMixture) else None)
                                           for key, predictor in forecasts.items()})
    physical = {field: sum(item["resources"][field] for pool in (pilot, fitted) for item in pool.values())
                for field in RESOURCE_KEYS}
    physical.update(fit_jobs=6, forecast_kernel_terms=forecast_terms, gp_decompositions=forecasts["gp"].decompositions)
    result = dict(case_id=directory.name, layer=table["layer"], repetition=table["repetition"], arms=arms,
                  physical_resources=physical, last_journal_sha256=journal.previous,
                  status="complete" if all(item["status"] == "complete" for item in arms.values()) else "incomplete")
    put_json(directory / "result.json", result)
    return result


def run(out, expected, executable, profiles):
    from .equivalent_prediction_mc1_report import publish_mc1

    manifest = verify(out, expected, executable, profiles)
    put_json(out / "started.json", dict(manifest_sha256=expected, utc=datetime.now(timezone.utc).isoformat()))
    counters = {item["key"]: dict(calls=0, seconds=0., disabled=None) for item in CANDIDATES}
    results = []

    def advisor(key, request, directory):
        counter = counters[key]
        if counter["disabled"]:
            return dict(status="disabled_unavailable", reason=counter["disabled"])
        if counter["calls"] >= 12 or counter["seconds"] >= 2160:
            return dict(status="budget_exhausted", reason="MC1 configuration budget exhausted")
        counter["calls"] += 1
        selected = select_llm(request, directory, executable, profiles[key], 2160 - counter["seconds"])
        counter["seconds"] += selected["wall_seconds"]
        if selected["model_unavailable"]:
            counter["disabled"] = "model channel unavailable; no further requests to this configuration"
        return selected

    stop = "completed"
    try:
        for index, record in enumerate(manifest["order"], 1):
            table = synthetic_table(record["layer"], record["repetition"], namespace="MC1-v1")
            result = evaluate_case(table, out / f"case_{index:02d}", advisor, record["model_order"])
            results.append(result)
            states = ", ".join(f"{key}:{result['arms'][key]['status']}" for key in counters)
            print(f"MC1 {index:02d}/12 {states}", flush=True)
            if all(counter["disabled"] for counter in counters.values()):
                stop = "all_models_unavailable"
                break
        totals = {field: sum(result["physical_resources"][field] for result in results)
                  for field in (*RESOURCE_KEYS, "fit_jobs", "forecast_kernel_terms", "gp_decompositions")}
        if any(value > LIMITS[field] for field, value in totals.items()):
            raise ValueError("MC1 aggregate numeric resource violation")
    except Exception as error:
        stop = "integrity_stop"
        put_json(out / "failure.json", dict(type=type(error).__name__, reason=str(error)))
        print(f"MC1 stopped: {type(error).__name__}: {error}", flush=True)
    put_json(out / "closed.json", dict(stop=stop, completed_cases=len(results), configurations=counters,
                                       llm_requests=sum(item["calls"] for item in counters.values()),
                                       model_wall_seconds=sum(item["seconds"] for item in counters.values()),
                                       manifest_sha256=expected))
    report = publish_mc1(out, results, dict(manifest, closure=json.loads((out / "closed.json").read_bytes())))
    print(f"MC1 static report: {out / 'index.html'}", flush=True)
    return report


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
    profiles = candidate_profiles(args.profile)
    if args.command == "freeze":
        print(freeze(args.out, args.executable, profiles), flush=True)
    else:
        run(args.out, args.digest, args.executable, profiles)


if __name__ == "__main__":
    main()
