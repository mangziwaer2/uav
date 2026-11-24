"""Single-use P5 runner with a bounded external model reply at the public checkpoint."""

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import select
import sys
import time

import numpy as np

from . import passive_closed_loop_pilot as p4
from .environment import ElectromagneticWorld
from .passive_model_advice import CHECKPOINT, MAX_INPUT_BYTES, MAX_REPLY_BYTES, Flight, numerical_choice, parse_reply
from .passive_map_pilot import ROOT, TEMPLATE, _write
from .passive_trigger_pilot import _world


OUT = ROOT / "outputs/passive_model_advice_P5_20261124"
CONTRACT = ROOT / "docs/passive_model_advice_P5_contract.md"
SEED = 43511
REPLY_TIMEOUT_S = 240
LABELS = dict(fixed="N · 数值模型决策", event="LLM · 单次模型决策")


def _scenario():
    base = _world(SEED).config
    buildings = tuple(dict(id=f"B{index}", height_m=height,
                           footprint_m=[[left, bottom], [right, bottom], [right, top], [left, top]])
                      for index, ((left, bottom, right, top), height) in enumerate(
                          zip(base.map.obstacles_m, (40, 45)), start=1))
    return replace(base, map=replace(base.map, geometry_mode="fixed_altitude", obstacles_m=(), buildings=buildings))


def _prompt(packet):
    instructions = (
        'You are the independent simulation decision maker. Use only the JSON below. No tools, files, internet, '
        'or questions. Choose exactly one GP temporal scale (nominal, short, long) for the remainder of this '
        'synthetic flight. All other model parameters, legal routes and numerical routing rules are fixed; do not '
        'propose geometry or override constraints. Decide from already observed noisy points, pre-issued one-step '
        'forecasts and route evidence. Larger dBm means stronger interference. GP support is not a safety label; '
        'unknown is not known low power. Route scores are predictive proxies, not future ground truth. In each '
        'review.rows item the columns are [whole-route T+Q_tail, unsupported seconds, paired change vs incumbent '
        'excluding one-sided support, seconds of two-route support]; null means the paired check was not needed. '
        'A route replacement needs both costs to improve. Indices are zero-based. The weighted prediction score '
        'is lower-is-better and uses the 12 pre-issued forecasts with a 6 s half-life. It is historical evidence '
        'not proof of future route cost. Return ONLY one JSON line, at most 1000 UTF-8 bytes, with exactly '
        'these keys: {"model":"nominal","reason":"one concise reason based on the packet"}.\n'
    )
    return instructions + json.dumps(packet, allow_nan=False, separators=(",", ":"))


def _request(packet, fallback, report):
    prompt = _prompt(packet)
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > MAX_INPUT_BYTES:
        return dict(selected=fallback, status="input_budget", calls=0, input_bytes=prompt_bytes, output_bytes=0,
                    elapsed_s=0.0)
    (OUT / "request.txt").write_text(prompt, encoding="utf-8")
    report("checkpoint_open", t=CHECKPOINT, input_bytes=prompt_bytes)
    print("P5 public checkpoint ready; awaiting one raw JSON reply on stdin.", flush=True)
    origin = time.monotonic()
    ready, _, _ = select.select([sys.stdin], [], [], REPLY_TIMEOUT_S)
    raw = sys.stdin.readline(MAX_REPLY_BYTES + 2) if ready else ""
    elapsed = time.monotonic() - origin
    (OUT / "raw_reply.txt").write_text(raw, encoding="utf-8")
    receipt = dict(selected=fallback, status="timeout" if not ready else "missing_reply", calls=1,
                   input_bytes=prompt_bytes, output_bytes=len(raw.rstrip("\r\n").encode("utf-8")), elapsed_s=elapsed)
    if raw:
        try:
            reply = parse_reply(raw.removesuffix("\n").removesuffix("\r"))
        except ValueError as error:
            receipt.update(status="invalid_reply", error=str(error))
        else:
            receipt.update(status="accepted", selected=reply["model"], reason=reply["reason"])
    return receipt


def _forecast_metrics(log):
    forecasts = [frame["forecast"] for frame in log["frames"] if frame["t"] > CHECKPOINT and frame["forecast"]]
    return dict(n=len(forecasts), mae_db=float(np.mean([row["absolute_error_db"] for row in forecasts]))
                if forecasts else None,
                first_issue_s=forecasts[0]["issued_s"] if forecasts else None)


def _replay(arms, layout):
    grid = dict(rows=layout.rows, cols=layout.columns, spacing=layout.resolution_m,
                obstacles=[[*item["footprint_m"][0], *item["footprint_m"][2]] for item in layout.buildings],
                start=layout.start_m, goal=layout.goal_m)
    page = p4._replay(dict(fixed=arms["numerical"], event=arms["llm"]), grid)
    replacements = {
        f"P4 · 接入 LLM 前的数值闭环 · {p4.SEED}": "P5 · 一次性 GP 模型决策",
        json.dumps(p4.LABELS): json.dumps(LABELS),
        p4.LABELS["fixed"]: LABELS["fixed"],
        p4.LABELS["event"]: LABELS["event"],
        "public_fixed.json": "public_numerical.json",
        "public_event.json": "public_llm.json",
        '${frame.prefix_size} 个原始点${risk}': '${frame.prefix_size} 个原始点 · GP ${frame.active_model}${risk}',
        '<a href="progress.jsonl">运行记录</a>': ('<a href="public_checkpoint.json">24 秒公开决策包</a>'
                                                    '<a href="raw_reply.txt">原始回执</a>'
                                                    '<a href="progress.jsonl">运行记录</a>'),
    }
    for old, new in replacements.items():
        if old not in page:
            raise ValueError(f"P5 replay marker missing: {old}")
        page = page.replace(old, new)
    return page


def _frozen():
    source = Path(__file__).parent
    modules = (*p4.MODULES, "passive_model_advice", "passive_model_advice_pilot")
    paths = (TEMPLATE, ROOT / "requirements.txt", *[source / (name + ".py") for name in modules])
    return dict(contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(), seed=SEED, noise_seed=SEED + 1,
                checkpoint_s=CHECKPOINT, implementation={str(path.relative_to(ROOT)):
                    hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


def freeze():
    OUT.mkdir(exist_ok=False)
    (OUT / "contract.md").write_bytes(CONTRACT.read_bytes())
    _write(OUT / "freeze.json", _frozen())


def run(digest):
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    if frozen != _frozen() or digest != frozen["contract_sha256"] or (OUT / "contract.md").read_bytes() != CONTRACT.read_bytes():
        raise ValueError("execution requires frozen contract and unchanged implementation")
    progress = OUT / "progress.jsonl"
    with progress.open("x", encoding="utf-8") as stream:
        stream.write('{"status":"started"}\n')

    def report(status, **data):
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(status=status, **data), allow_nan=False) + "\n")

    try:
        config = _scenario()
        world = ElectromagneticWorld(config)
        shared = Flight(world.geometry, config.aircraft, config.belief)
        checkpoint = shared.run(world.observe, pause=True)
        if checkpoint is None:
            _write(OUT / "public_prefix_only.json", shared.log("prefix_only"))
            report("closed_without_call", stop=shared.stop)
            return
        posteriors, menu = checkpoint
        packet = shared.packet(menu)
        numerical, scores = numerical_choice(packet)
        packet["weighted_predictive_loss"] = scores
        _write(OUT / "public_checkpoint.json", packet)
        rng_state = deepcopy(world.random.bit_generator.state)
        receipt = _request(packet, numerical, report)
        _write(OUT / "receipt.json", receipt)
        report("checkpoint_closed", status_reply=receipt["status"], elapsed_s=receipt["elapsed_s"])
        flights = {}
        for mode, model in (("numerical", numerical), ("llm", receipt["selected"])):
            branch_world = ElectromagneticWorld(config)
            branch_world.random.bit_generator.state = deepcopy(rng_state)
            branch = deepcopy(shared)
            origin = time.monotonic()
            branch.resume(model, posteriors[model], menu[model], branch_world.observe)
            log = branch.log(mode)
            _write(OUT / f"public_{mode}.json", log)
            flights[mode] = (branch_world, log, branch.traversed)
            report("flight_closed", mode=mode, model=model, stop=branch.stop,
                   elapsed_s=time.monotonic() - origin, **branch.ledger)
        arms, results = {}, {}
        for mode, (branch_world, log, traversed) in flights.items():
            evaluation = p4._evaluate(branch_world, log, traversed)
            arms[mode] = dict(log=log, evaluation=evaluation)
            results[mode] = dict(metrics=evaluation["metrics"], model=log["model"],
                                 new_forecasts=_forecast_metrics(log), resources=log["resources"],
                                 reviews=[{key: row[key] for key in ("t", "changed", "model", "selected", "reference")}
                                          for row in log["decisions"]])
        traces = [[(frame["t"], frame["moved"]) for frame in entry["log"]["frames"]] for entry in arms.values()]
        _write(OUT / "results.json", dict(arms=results, same_trajectory=traces[0] == traces[1], reply=receipt,
                                          weighted_predictive_loss=scores))
        (OUT / "index.html").write_text(_replay(arms, config.map), encoding="utf-8")
        report("complete")
    except Exception as error:
        report("failed", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--authorized-contract-sha256")
    options = parser.parse_args()
    if options.freeze:
        freeze()
    else:
        run(options.authorized_contract_sha256)
