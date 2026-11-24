"""Recover P7 scoring and replay from closed flight logs, without resampling."""

import json
from pathlib import Path

from .environment import ElectromagneticWorld
from .passive_map_pilot import _write
from .passive_model_advice_publish import evaluate
from .passive_route_actions_pilot import CHECKPOINT, MODES, OUT, _config, _hash, _put, _read, _ready, _replay, _resolve


def publish():
    ready = _ready()
    execution = _read("execution_started.json")
    menu, packet = _read("review.json"), _read("public_checkpoint.json")
    receipts = execution["receipts"]
    for mode, receipt in receipts.items():
        raw_name = f"reply_{mode}.txt"
        verified = _resolve((OUT / raw_name).read_bytes(), receipt["elapsed_s"], menu, mode)
        if (receipt != _read(f"receipt_{mode}.json") or receipt["raw_sha256"] != _hash(OUT / raw_name)
                or receipt["request_sha256"] != ready["hashes"][f"request_{mode}.txt"]
                or any(receipt.get(key) != value for key, value in verified.items())):
            raise ValueError("receipt no longer matches the original closed execution")
    choices = dict(llm_waypoint=receipts["waypoint"]["route_idx"], llm_route=receipts["route"]["route_idx"],
                   n_guard=menu["numeric_guarded_idx"], n_route=menu["numeric_proxy_idx"],
                   n_waypoint=menu["numeric_waypoint_route_idx"])
    progress = [json.loads(line) for line in (OUT / "progress.jsonl").read_bytes().splitlines()]
    if [row["route_idx"] for row in progress if row["status"] == "flight_closed"] != list(dict.fromkeys(choices.values())):
        raise ValueError("not all planned unique suffixes are closed")

    config = _config()
    traces = {}
    for selected in dict.fromkeys(choices.values()):
        log = _read(f"public_route_{selected}.json")
        point_rows = [[point["time_seconds"], *point["position_m"], point["interference_dbm"]]
                      for point in log["points"][:len(packet["point_rows"])]]
        checkpoint = log["decisions"][-1]
        if (point_rows != packet["point_rows"] or checkpoint["t"] != CHECKPOINT or checkpoint["selected"] != selected
                or checkpoint["candidates"][selected]["route"] != menu["routes"][selected]["route_xy_m"]):
            raise ValueError("prefix or chosen remainder differs from the recorded checkpoint")
        traces[selected] = log, evaluate(ElectromagneticWorld(config), log)

    arms, metrics = {}, {}
    for name, selected in choices.items():
        original, evaluation = traces[selected]
        log = dict(original)
        if name.startswith("llm_"):
            receipt = receipts[name.removeprefix("llm_")]
            reply = receipt.get("proposal", {})
            waypoint = menu["waypoints"][reply["idx"]]["xy_m"] if reply.get("action") == "waypoint" else None
            log["proposal"] = dict(status=receipt["status"], waypoint_xy_m=waypoint)
        arms[name] = dict(log=log, evaluation=evaluation)
        metrics[name] = dict(route_idx=selected, **evaluation["metrics"], resources=log["resources"])

    notes = dict(publication_failure="KeyError: 'committed_s' after both suffixes were saved",
                 model_prompt_copy=dict(modes=list(MODES), point_index=11, field="x_m", frozen=198.04133706149003,
                                        sent=198.04133706146103, exact_byte_transport=False),
                 reruns=0, added_calls=0)
    for mode in MODES:
        frozen = (OUT / f"request_{mode}.txt").read_bytes()
        if frozen.count(b"198.04133706149003") != 1:
            raise ValueError("cannot identify the documented prompt transfer difference")
        _put(f"transcribed_request_{mode}.txt", frozen.replace(b"198.04133706149003", b"198.04133706146103"))
    _write(OUT / "results.json", dict(arms=metrics, receipts=receipts, distinct_flown_suffixes=len(traces),
                                      common_prefix_points=len(packet["point_rows"]), restore_fits=len(traces),
                                      execution_notes=notes))
    page = _replay(arms, config.map)
    page = page.replace('<div class="controls">', '<p>两次 LLM 回执均被执行为保持。数值全域枚举采用另一条余程。'
                        '<a href="findings.md">结果与局限</a></p><div class="controls">')
    page = page.replace('<a href="progress.jsonl">运行记录</a>', '<a href="progress.jsonl">运行记录</a>'
                        '<a href="reply_route.txt">直接选路回执</a><a href="reply_waypoint.txt">中间目标回执</a>')
    _put("index.html", page.encode("utf-8"))
    names = ("contract.md", "freeze.json", "ready.json", "execution_started.json", "public_checkpoint.json",
             "review.json", "progress.jsonl", "results.json", "index.html", *(f"public_route_{idx}.json" for idx in traces),
             *(f"{kind}_{mode}{ending}" for mode in MODES for kind, ending in
               (("request", ".txt"), ("transcribed_request", ".txt"), ("reply", ".txt"), ("receipt", ".json"))))
    _write(OUT / "publication.json", dict(status="recovered_from_closed_logs", **notes,
                                          publisher_sha256=_hash(Path(__file__)),
                                          files={name: _hash(OUT / name) for name in names}))


if __name__ == "__main__":
    publish()
