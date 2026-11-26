"""Pure MC1 reporting; accepts scored records, never fits or calls a model.

``arms`` holds six-arm metrics and, for LLM arms, request accounting and the
``task_eligible`` gate. ``comparisons`` uses ``<LLM>_minus_<reference>`` keys.
Missing pairs remain descriptive, with no imputation or replacement calls.
"""

from contextlib import ExitStack
from html import escape
import json
import math
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .equivalent_prediction_report import (
    _arm_summary, _bootstrap_indices, _finite_number, _format, _json_safe,
    _mean, _resource_totals, _table,
)


ARMS = ("numerical_pair", "numerical_all", "gp", "luna", "sol56", "sol61")
LLM_ARMS = ("luna", "sol56", "sol61")
LABELS = dict(numerical_pair="N-选二", numerical_all="N-全三", gp="滚动 GP",
              luna="luna", sol56="sol56", sol61="sol61")
COMPARISONS = {f"{arm}_minus_{reference}": (arm, reference)
               for arm in LLM_ARMS for reference in ("numerical_pair", "sol56")}
EXPECTED_CASES = 12
CASES_PER_LAYER = 3
BOOTSTRAP_SEED = 438271
BOOTSTRAP_REPLICATES = 10000
NLL_MARGIN = .05
NOTICE = "本轮未试飞：MC1 静止预测六臂曲线，仅汇总已评分点测；不代表合法到达或实际全程 T+Q 收益。"
TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens")
IDENTITY_FIELDS = ("returned_model_id", "upstream_model_id", "upstream_version",
                   "model_version", "snapshot", "system_fingerprint", "provider",
                   "interface", "identity_verification", "upstream_identity_verified")
NETWORK_FIELDS = ("visible_network_error_events", "visible_reconnect_events", "network_attempts")
PLOT_FIELDS = ("time_seconds", "observed_dbm", "mean_dbm", "lower90_dbm", "upper90_dbm")
NOT_REQUESTED = {"missing", "not_requested", "disabled", "disabled_unavailable", "budget_exhausted"}


def _selection(entry):
    source = entry.get("selection", entry)
    if source is None:
        source = {}
    if not isinstance(source, dict):
        raise ValueError("selection must be a dict or null")
    selection = {key: value for key, value in source.items()
                 if key not in {"rows", "resources", "selection"}}
    selection["requested_model_id"] = source.get("requested_model_id") or source.get("model")
    for key in (*IDENTITY_FIELDS, *NETWORK_FIELDS, "reasoning_effort", "wall_seconds", "usage"):
        selection.setdefault(key, None)
    selection["price"] = None
    return selection


def _numeric_resources(resources):
    return isinstance(resources, dict) and all(
        _numeric_resources(value) if isinstance(value, dict)
        else _finite_number(value) and value >= 0
        for value in resources.values())


def _summarize_arm(arm, entry):
    summary = _arm_summary(arm, entry)
    if summary["status"] != "complete":
        summary["rows"] = []
    elif summary["valid"]:
        error = None
        rows = summary["rows"]
        if arm == "gp" and summary["ids"]:
            error = "GP 不选择结构 ID"
        elif not _numeric_resources(summary["resources"]):
            error = "resources 必须为非负有限数值 dict"
        elif any(any(not _finite_number(row.get(key)) for key in PLOT_FIELDS)
                 or row["lower90_dbm"] > row["upper90_dbm"] for row in rows):
            error = "六个评分点必须含有限时间、观测及有效预测区间"
        elif any(left["time_seconds"] >= right["time_seconds"]
                 for left, right in zip(rows, rows[1:])):
            error = "评分时间必须严格递增"
        if error:
            summary.update(valid=False, metrics=None, validation_error=error)
    if arm in LLM_ARMS:
        summary["selection"] = _selection(entry or {})
        summary["requested"] = (summary["status"] not in NOT_REQUESTED
                                or "receiver" in summary["selection"])
    return summary


def _aggregate_arm(cases, arm):
    entries = [case["arms"][arm] for case in cases]
    layers = []
    names = ("mean_nll", "mean_squared_error_db2", "mae_db", "coverage90", "mean_interval_width_db")
    for layer in range(4):
        metrics = [case["arms"][arm]["metrics"] for case in cases
                   if case["layer"] == layer and case["arms"][arm]["valid"]]
        layers.append(dict(layer=layer, valid_cases=len(metrics),
                           metrics={name: _mean([entry[name] for entry in metrics]) for name in names}))
    valid = [entry for entry in entries if entry["valid"]]
    metrics = {name: _mean([layer["metrics"][name] for layer in layers if layer["valid_cases"]])
               for name in names}
    metrics.update(rmse_db=math.sqrt(metrics["mean_squared_error_db2"]) if valid else None,
                   point_count=6 * len(valid),
                   all_mismatch_points=sum(entry["metrics"]["all_mismatch_points"] for entry in valid),
                   all_mismatch_cases=sum(entry["metrics"]["all_mismatch_points"] > 0 for entry in valid))
    statuses = sorted({entry["status"] for entry in entries})
    summary = dict(valid_cases=len(valid), expected_cases=EXPECTED_CASES,
                   valid_rate=len(valid) / EXPECTED_CASES, metrics=metrics, layers=layers,
                   status_counts={status: sum(entry["status"] == status for entry in entries) for status in statuses},
                   missing_count=EXPECTED_CASES - len(cases) + sum(entry["status"] == "missing" for entry in entries),
                   disabled_count=sum(entry["status"].startswith("disabled") for entry in entries),
                   not_requested_count=sum(entry["status"] == "not_requested" for entry in entries),
                   budget_exhausted_count=sum(entry["status"] == "budget_exhausted" for entry in entries),
                   independently_chargeable_resource_totals=_resource_totals([entry["resources"] for entry in entries]))
    if arm not in LLM_ARMS:
        return summary
    requested = [entry for entry in entries if entry["requested"]]
    usage = [entry["selection"]["usage"] for entry in requested]
    totals = _resource_totals([record for record in usage if isinstance(record, dict)])
    recorded = {key: sum(isinstance(record, dict) and _finite_number(record.get(key))
                         and record[key] >= 0 for record in usage) for key in TOKEN_FIELDS}
    for key in TOKEN_FIELDS:
        totals[key] = sum(record[key] for record in usage
                          if isinstance(record, dict) and _finite_number(record.get(key)) and record[key] >= 0
                          ) if recorded[key] else None
    walls = [entry["selection"]["wall_seconds"] for entry in requested
             if _finite_number(entry["selection"]["wall_seconds"]) and entry["selection"]["wall_seconds"] >= 0]
    network_totals, network_missing = {}, {}
    for key in NETWORK_FIELDS:
        values = [entry["selection"][key] for entry in requested
                  if _finite_number(entry["selection"][key]) and entry["selection"][key] >= 0]
        network_totals[key] = sum(values) if values else None
        network_missing[key] = len(requested) - len(values)
    summary.update(requested_count=len(requested), valid_count=len(valid),
                   failed_count=sum(not entry["valid"] for entry in requested),
                   usage_totals=totals, usage_field_records=recorded,
                   usage_missing_records={key: len(requested) - count for key, count in recorded.items()},
                   network_telemetry_totals=network_totals, network_telemetry_missing_records=network_missing,
                   model_wall_seconds=dict(recorded_count=len(walls), missing_count=len(requested) - len(walls),
                                           total=sum(walls) if walls else None,
                                           median=float(np.median(walls)) if walls else None,
                                           p95=float(np.percentile(walls, 95)) if walls else None),
                   price=None)
    return summary


def _comparison(cases_by_layer, indices, name, left_arm, right_arm):
    layer_means, pair_counts, paired_ids, samples = [], [], [], []
    same_pair_count = 0
    for layer, cases in enumerate(cases_by_layer):
        values = np.array([case["delta_nll"][name] if case["delta_nll"][name] is not None else np.nan
                           for case in cases], dtype=float)
        paired = [case for case, value in zip(cases, values) if math.isfinite(value)]
        paired_ids.extend(case["case_id"] for case in paired)
        pair_counts.append(len(paired))
        layer_means.append(_mean([case["delta_nll"][name] for case in paired]))
        same_pair_count += sum(set(case["arms"][left_arm]["ids"]) == set(case["arms"][right_arm]["ids"])
                               for case in paired)
        if paired:
            resampled = values[indices[layer]]
            counts = np.isfinite(resampled).sum(axis=1)
            means = np.full(BOOTSTRAP_REPLICATES, np.nan)
            np.divide(np.nansum(resampled, axis=1), counts, out=means, where=counts > 0)
            samples.append(means)
    draws = np.mean(samples, axis=0) if samples else np.array([])
    usable = draws[np.isfinite(draws)]
    return dict(left_arm=left_arm, right_arm=right_arm, paired_cases=sum(pair_counts),
                paired_case_ids=paired_ids, pairs_by_layer=pair_counts, layer_mean_delta_nll=layer_means,
                mean_delta_nll=_mean([value for value in layer_means if value is not None]),
                ci95=np.percentile(usable, [2.5, 97.5]).tolist() if len(usable) else None,
                bootstrap_usable_replicates=len(usable), same_pair_count=same_pair_count,
                descriptive_only=pair_counts != [CASES_PER_LAYER] * 4)


def summarize_mc1(results: list, manifest: dict):
    """Return an order-independent, JSON-safe MC1 summary without doing I/O.

    The twelve independent cases occupy unique (layer, repetition) slots.
    Six points first form a case mean; all contrasts share case bootstrap
    indices, stratified by layer. Incomplete contrasts weight represented
    layers equally and discard resamples missing a represented paired layer.
    Prices are unavailable: eligibility never selects a cheapest model.
    """
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dict")
    candidates = manifest.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("manifest.candidates must be a list")
    configurations = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("key") not in LLM_ARMS:
            raise ValueError("invalid MC1 candidate key")
        key = candidate["key"]
        if key in configurations:
            raise ValueError("duplicate MC1 candidate key")
        configurations[key] = candidate
    results = list(results)
    if len(results) > EXPECTED_CASES:
        raise ValueError("MC1 accepts at most twelve independent cases")
    cases, seen_ids, seen_slots = [], set(), set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("each case must be a dict")
        case_id, layer, repetition = (result.get(key) for key in ("case_id", "layer", "repetition"))
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError("case_id must be a unique nonempty string")
        if type(layer) is not int or layer not in range(4):
            raise ValueError("layer must be an integer from 0 to 3")
        if type(repetition) is not int or repetition not in range(3) or (layer, repetition) in seen_slots:
            raise ValueError("repetition must be 0..2, unique within its layer")
        seen_ids.add(case_id)
        seen_slots.add((layer, repetition))
        arms = result.get("arms", {})
        if not isinstance(arms, dict) or set(arms) - set(ARMS):
            raise ValueError("arms must contain only the six MC1 arm keys")
        physical = result.get("physical_resources", {})
        if not _numeric_resources(physical):
            raise ValueError("physical_resources must be a nonnegative numeric dict")
        case = dict(case_id=case_id, layer=layer, repetition=repetition,
                    status=result.get("status", "missing"), reason=result.get("reason"),
                    physical_resources=physical,
                    arms={arm: _summarize_arm(arm, arms.get(arm)) for arm in ARMS})
        case["complete"] = case["status"] == "complete" and all(entry["valid"] for entry in case["arms"].values())
        case["delta_nll"], case["pairing_errors"] = {}, {}
        for name, (left_arm, right_arm) in COMPARISONS.items():
            left, right = case["arms"][left_arm], case["arms"][right_arm]
            delta = None
            if left["valid"] and right["valid"]:
                if [(row["time_seconds"], row["observed_dbm"]) for row in left["rows"]] != [
                        (row["time_seconds"], row["observed_dbm"]) for row in right["rows"]]:
                    case["pairing_errors"][name] = "两臂评分时间或实测值不一致，不能配对"
                else:
                    delta = left["metrics"]["mean_nll"] - right["metrics"]["mean_nll"]
            case["delta_nll"][name] = delta if delta is not None and math.isfinite(delta) else None
        cases.append(case)
    cases.sort(key=lambda case: (case["layer"], case["repetition"], case["case_id"]))
    by_layer = [[case for case in cases if case["layer"] == layer] for layer in range(4)]
    indices = _bootstrap_indices(by_layer)
    comparisons = {name: _comparison(by_layer, indices, name, *arms) for name, arms in COMPARISONS.items()}
    closure = manifest.get("closure")
    closure_completed = isinstance(closure, dict) and closure.get("stop") == "completed"
    design_complete = len(cases) == EXPECTED_CASES
    arms = {arm: _aggregate_arm(cases, arm) for arm in ARMS}
    for arm in LLM_ARMS:
        reasons = []
        if not design_complete:
            reasons.append("预定12个独立场不完整")
        if not closure_completed:
            reasons.append("manifest.closure.stop != 'completed'")
        for reference in ("numerical_pair", "sol56"):
            contrast = comparisons[f"{arm}_minus_{reference}"]
            delta = contrast["mean_delta_nll"]
            if contrast["paired_cases"] != EXPECTED_CASES:
                reasons.append(f"相对 {reference} 未取得12个有效配对场")
            elif delta > NLL_MARGIN:
                reasons.append(f"相对 {reference} 的 NLL 均差 > +0.05 nat/点")
        arms[arm].update(configuration=configurations.get(arm), task_eligible=not reasons,
                         eligibility_reasons=reasons)
    return _json_safe(dict(
        experiment="MC1", notice=NOTICE, manifest=manifest, expected_cases=EXPECTED_CASES,
        received_cases=len(cases), missing_cases=EXPECTED_CASES - len(cases),
        cases_by_layer=[len(layer) for layer in by_layer], design_complete=design_complete,
        complete_cases=sum(case["complete"] for case in cases), closure_completed=closure_completed,
        cases=cases, arms=arms, comparisons=comparisons,
        physical_resource_totals=_resource_totals([case["physical_resources"] for case in cases]),
        bootstrap=dict(generator="PCG64", seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES,
                       unit="case", shared_indices_across_arms=True, percentiles=[2.5, 97.5],
                       incomplete_policy="仅描述：已有配对层等权；剔除任何已有配对层在抽样后无配对的重复"),
        task_eligible={arm: arms[arm]["task_eligible"] for arm in LLM_ARMS},
        cost_selection="undetermined_missing_prices", selected_model=None, price=None,
    ))


def _evidence_links(case, arm):
    if "receiver" not in case["arms"][arm]["selection"]:
        return []
    prefix = f"{quote(case['case_id'], safe='').replace('.', '%2E')}/{arm}"
    return [("selection.json", f"{prefix}/selection.json"),
            ("原始回执", f"{prefix}/receiver/0001-reply.txt")]


def _comparison_rows(report):
    rows = []
    for contrast in report["comparisons"].values():
        interval = contrast["ci95"]
        rows.append([f"{LABELS[contrast['left_arm']]} − {LABELS[contrast['right_arm']]}",
                     str(contrast["paired_cases"]), "/".join(map(str, contrast["pairs_by_layer"])),
                     _format(contrast["mean_delta_nll"]),
                     "—" if interval is None else f"[{_format(interval[0])}, {_format(interval[1])}]",
                     "仅描述" if contrast["descriptive_only"] else "完整配对",
                     str(contrast["bootstrap_usable_replicates"])])
    return rows


def _model_rows(report):
    rows = []
    for arm in LLM_ARMS:
        entry = report["arms"][arm]
        configuration = entry["configuration"] or {}
        wall, usage = entry["model_wall_seconds"], entry["usage_totals"]
        rows.append([arm, configuration.get("model"), configuration.get("reasoning_effort"),
                     *(str(entry[key]) for key in ("requested_count", "valid_count", "failed_count", "disabled_count", "not_requested_count")),
                     _format(wall["median"]), _format(wall["p95"]),
                     *(str(usage[key]) if usage[key] is not None else "unknown" for key in TOKEN_FIELDS),
                     "是" if entry["task_eligible"] else "否"])
    return rows


COMPARISON_HEADERS = ["比较", "配对场数", "各层分母", "NLL 均差", "95% 区间", "范围", "区间保留次数"]
MODEL_HEADERS = ["候选", "配置请求名", "推理设置", "请求", "有效", "失败", "disabled", "未请求",
                 "等待中位 s", "等待 p95 s", "input_tokens", "cached_input_tokens", "output_tokens", "task_eligible"]
METHOD_NOTE = ("每场六点平均 NLL（nat/点，越低越好），先配对作差，再四层等权、层内等权。"
               "PCG64(438271)，10000 次场级配对分层 bootstrap，95% 百分位区间；六点不是六个独立场。"
               "缺场仅描述已有配对层，不补场；缺少已有配对层的抽样重复剔除。")
COST_NOTE = ("cost_selection='undetermined_missing_prices'；price=null，未选择最便宜模型。"
             "质量、墙钟和 token 量分别报告，不能据此判断性价比。原始 output_tokens 可能已含推理 token，"
             "不另行相加；cached_input_tokens 是输入的子集。缺失用量和失败请求费用均未知。"
             "token 合计仅覆盖已报告记录，各字段缺失数见详情。请求名是网关别名，底层版本仅按回执记录，缺失为 null。")
GATE_NOTE = ("每个配置独立判断 task_eligible：要求12场合法有效配对、正常闭批，相对 N-选二及 sol56 的 NLL 均差均 ≤ +0.05；"
             "其他配置失败或 disabled 导致的场级 incomplete 不否决完整配置。"
             "sol56 自身差为0。这是校准筛选规则，不是统计非劣效证明，不代表 EP2 或飞行收益。")


def _md_table(headers, rows):
    def cell(value):
        return escape(str(value)).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "|" + "---|" * len(headers),
                      *("| " + " | ".join(map(cell, row)) + " |" for row in rows)])


def _findings(report):
    lines = ["# MC1 三模型静止校准汇总", "", NOTICE, "",
             f"收到 {report['received_cases']}/12 场；六臂均完成 {report['complete_cases']}/12 场；"
             f"正常闭批：{report['closure_completed']}。", "", METHOD_NOTE, "", GATE_NOTE, "",
             _md_table(COMPARISON_HEADERS, _comparison_rows(report)), "", COST_NOTE, "",
             _md_table(MODEL_HEADERS, _model_rows(report)), ""]
    for arm in LLM_ARMS:
        for reason in report["arms"][arm]["eligibility_reasons"]:
            lines.append(f"- {arm}：{escape(reason)}")
    for case in report["cases"]:
        lines.extend(["", f"## {escape(case['case_id'])} · 层 {case['layer']} · repetition {case['repetition']}", "",
                      f"场状态：{escape(str(case['status']))}；{escape(str(case['reason'] or ''))}", "",
                      _md_table(ARM_HEADERS, _arm_rows(case)), ""])
        for arm in LLM_ARMS:
            links = _evidence_links(case, arm)
            if links:
                lines.append(f"- {arm}：" + " · ".join(f"[{label}]({href})" for label, href in links))
    lines.extend(["", "实际公共资源按 physical_resources 每场计一次；各臂应计资源含失败消耗，不能相加冒充实际公共工作。",
                  "", "完整逐场状态、原始 usage、模型身份 null、分层指标、全失配点/场计数及资源见 "
                  "[summary.json](summary.json)；[六臂静止预测曲线](index.html)。", ""])
    return "\n".join(lines)


ARM_HEADERS = ["臂", "选择", "状态", "原因", "NLL", "RMSE dB", "MAE dB", "覆盖90%", "宽度 dB", "全失配点"]


def _arm_rows(case):
    rows = []
    for arm, entry in case["arms"].items():
        metrics = entry["metrics"] or {}
        rows.append([LABELS[arm], str(entry["ids"]), entry["status"],
                     entry.get("validation_error") or entry["reason"] or "",
                     *[_format(metrics.get(key)) for key in ("mean_nll", "rmse_db", "mae_db", "coverage90",
                                                             "mean_interval_width_db", "all_mismatch_points")]])
    return rows


def _prediction_svg(case):
    series = {arm: entry["rows"] for arm, entry in case["arms"].items() if entry["valid"]}
    rows = [row for arm_rows in series.values() for row in arm_rows]
    if not rows:
        return "<p>无可绘制的有效已记录预测与观测；未补造预测点。</p>"
    t0 = min(row["time_seconds"] for row in rows)
    dt = max(1., max(row["time_seconds"] for row in rows) - t0)
    p0 = min(row[key] for row in rows for key in PLOT_FIELDS[1:]) - 2.
    dp = max(4., max(row[key] for row in rows for key in PLOT_FIELDS[1:]) + 2. - p0)

    def x(time):
        return 72. + 760. * (time - t0) / dt

    def y(power):
        return 285. - 235. * (power - p0) / dp

    def points(arm_rows, key):
        return " ".join(f"{x(row['time_seconds']):.2f},{y(row[key]):.2f}" for row in arm_rows)

    parts = ["<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 920 390' role='img'>",
             f"<title>{escape(case['case_id'])}：六臂冻结预测与观测</title>",
             "<desc>黑点为已记录点测，彩线为预测均值，阴影为90%观测区间。本轮未试飞。</desc>",
             "<path d='M72 45V285H850' fill='none' stroke='#475569'/>",
             "<text x='10' y='25'>功率 (dBm，越大干扰越强)</text><text x='785' y='330'>时间 (s)</text>"]
    for tick in range(5):
        power = p0 + dp * tick / 4
        parts.append(f"<path d='M72 {y(power):.2f}H850' stroke='#e2e8f0'/>"
                     f"<text x='8' y='{y(power) + 4:.2f}'>{power:.1f}</text>")
    for time in sorted({row["time_seconds"] for row in rows}):
        parts.append(f"<text x='{x(time):.2f}' y='307' text-anchor='middle'>{time:g}</text>")
    colors = ("#2563eb", "#d97706", "#15803d", "#9333ea", "#dc2626", "#0891b2")
    for index, (arm, color) in enumerate(zip(ARMS, colors)):
        arm_rows = series.get(arm, [])
        if arm_rows:
            polygon = points(arm_rows, "lower90_dbm") + " " + points(list(reversed(arm_rows)), "upper90_dbm")
            parts.extend([f"<polygon points='{polygon}' fill='{color}' fill-opacity='.08'/>",
                          f"<polyline data-arm='{arm}' points='{points(arm_rows, 'mean_dbm')}' "
                          f"fill='none' stroke='{color}' stroke-width='2'/>"])
        parts.append(f"<text x='{72 + index % 3 * 260}' y='{350 + index // 3 * 24}' fill='{color}'>"
                     f"{escape(LABELS[arm])}{'' if arm_rows else '（无有效预测）'}</text>")
    for time, power in sorted({(row["time_seconds"], row["observed_dbm"]) for row in rows}):
        parts.append(f"<circle cx='{x(time):.2f}' cy='{y(power):.2f}' r='3.5' fill='#0f172a'/>")
    return "".join(parts) + "</svg>"


def _html(report):
    parts = ["<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>MC1 三模型静止校准汇总</title><style>",
             "body{font:16px system-ui,sans-serif;max-width:1280px;margin:2rem auto;padding:0 1rem;color:#0f172a;background:#f8fafc}",
             "section{background:white;padding:1rem;margin:1rem 0;border:1px solid #cbd5e1;border-radius:8px}",
             "table{border-collapse:collapse;width:100%}th,td{border:1px solid #cbd5e1;padding:.45rem;text-align:left}",
             ".scroll{overflow:auto}svg{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}text{font-size:12px}",
             "</style></head><body><h1>MC1 三模型静止校准汇总</h1>", f"<p><strong>{escape(NOTICE)}</strong></p>",
             f"<p>收到 {report['received_cases']}/12 场；六臂均完成 {report['complete_cases']}/12 场；"
             f"正常闭批：{report['closure_completed']}。</p>",
             f"<p>{escape(METHOD_NOTE)}</p><p>{escape(GATE_NOTE)}</p>",
             _table(COMPARISON_HEADERS, _comparison_rows(report)), f"<p>{escape(COST_NOTE)}</p>",
             _table(MODEL_HEADERS, _model_rows(report)),
             "<ul>" + "".join(f"<li>{arm}：{escape(reason)}</li>" for arm in LLM_ARMS
                              for reason in report["arms"][arm]["eligibility_reasons"]) + "</ul>",
             "<p><a href='summary.json'>JSON 指标与资源</a> · <a href='findings.md'>文字结果</a></p>"]
    for case in report["cases"]:
        parts.extend([f"<section><h2>{escape(case['case_id'])} · 层 {case['layer']} · repetition {case['repetition']}</h2>",
                      f"<p>场状态：{escape(str(case['status']))}；{escape(str(case['reason'] or ''))}</p>",
                      _prediction_svg(case), _table(ARM_HEADERS, _arm_rows(case))])
        for arm in LLM_ARMS:
            links = _evidence_links(case, arm)
            if links:
                parts.append(f"<p>{arm}：" + " · ".join(
                    f"<a href='{escape(href, quote=True)}'>{label}</a>" for label, href in links) + "</p>")
        parts.append("<details><summary>逐场配对差、资源与原始模型记录</summary><pre>" + escape(json.dumps(
            case, ensure_ascii=False, indent=2, allow_nan=False)) + "</pre></details></section>")
    parts.extend(["<p>实际公共资源按 physical_resources 每场计一次；各臂应计资源含失败消耗，不能相加冒充实际公共工作。</p>",
                  "<details><summary>完整统计、分层指标、用量缺失与资源汇总</summary><pre>" + escape(json.dumps(
                      {key: value for key, value in report.items() if key != "cases"},
                      ensure_ascii=False, indent=2, allow_nan=False)) + "</pre></details></body></html>"])
    return "".join(parts)


def publish_mc1(out: Path, results: list, manifest: dict):
    """Exclusively publish three new documents, preserving existing evidence."""
    summary = summarize_mc1(results, manifest)
    documents = {"summary.json": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                 "findings.md": _findings(summary), "index.html": _html(summary)}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with ExitStack() as stack:
            streams = {}
            for name in documents:
                path = out / name
                streams[name] = stack.enter_context(path.open("x", encoding="utf-8"))
                created.append(path)
            for name, body in documents.items():
                streams[name].write(body)
    except OSError:
        for path in created:
            path.unlink()
        raise
    return summary
