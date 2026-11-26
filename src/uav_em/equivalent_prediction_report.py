"""Summarize already scored EP1 tables without fitting, sensing, or model calls."""

from contextlib import ExitStack
from html import escape
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np


ARMS = ("numerical_pair", "numerical_all", "llm_semantic", "llm_anonymous", "gp")
LABELS = {
    "numerical_pair": "N-选二",
    "numerical_all": "N-全三",
    "llm_semantic": "LLM-语义",
    "llm_anonymous": "LLM-匿名",
    "gp": "滚动 GP",
}
COMPARISONS = {
    f"llm_semantic_minus_{arm}": arm
    for arm in ("numerical_pair", "numerical_all", "llm_anonymous", "gp")
}
BOOTSTRAP_SEED = 438271
BOOTSTRAP_REPLICATES = 10000
EXPECTED_CASES = 24
CASES_PER_LAYER = 6
POINTS_PER_CASE = 6
NOTICE = "静止预测图表：未试飞，仅合成点测；不代表合法到达或实际全程 T+Q 收益。"


def _json_safe(value):
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def _finite_number(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _mean(values):
    return math.fsum(value / len(values) for value in values) if values else None


def _arm_summary(arm, entry):
    if entry is None:
        entry = {"status": "missing", "reason": "预定臂缺失"}
    if not isinstance(entry, dict):
        raise ValueError(f"{arm} must be a dict")
    summary = {
        "status": entry.get("status", "missing"),
        "reason": entry.get("reason"),
        "ids": entry.get("ids", []),
        "resources": entry.get("resources", {}),
        "rows": entry.get("rows", []),
        "valid": False,
        "metrics": None,
    }
    if summary["status"] != "complete":
        return summary
    rows, ids = summary["rows"], summary["ids"]
    invalid = None
    if not isinstance(rows, list) or len(rows) != POINTS_PER_CASE:
        invalid = "完整臂必须包含六个评分点"
    elif not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        invalid = "ids 必须为字符串列表"
    elif len(set(ids)) != len(ids) or (arm != "gp" and len(ids) != (3 if arm == "numerical_all" else 2)):
        invalid = "结构选择数量无效或重复"
    elif not isinstance(summary["resources"], dict):
        invalid = "resources 必须为 dict"
    else:
        numeric_keys = ("nll", "squared_error", "absolute_error", "interval_width_db")
        for row in rows:
            if not isinstance(row, dict) or any(not _finite_number(row.get(key)) for key in numeric_keys):
                invalid = "评分缺失或包含非有限数值"
                break
            if any(row[key] < 0 for key in numeric_keys[1:]):
                invalid = "误差和区间宽度不得为负"
                break
            if any(not isinstance(row.get(key), (bool, np.bool_)) for key in ("covered90", "all_mismatch")):
                invalid = "覆盖和全失配标记必须为 bool"
                break
    if invalid is not None:
        summary["validation_error"] = invalid
        return summary
    squared_error = _mean([float(row["squared_error"]) for row in rows])
    summary["valid"] = True
    summary["metrics"] = {
        "point_count": len(rows),
        "mean_nll": _mean([float(row["nll"]) for row in rows]),
        "mean_squared_error_db2": squared_error,
        "rmse_db": math.sqrt(squared_error),
        "mae_db": _mean([float(row["absolute_error"]) for row in rows]),
        "coverage90": _mean([int(row["covered90"]) for row in rows]),
        "mean_interval_width_db": _mean([float(row["interval_width_db"]) for row in rows]),
        "all_mismatch_points": sum(int(row["all_mismatch"]) for row in rows),
    }
    return summary


def _resource_totals(resources):
    keys = sorted({key for resource in resources if isinstance(resource, dict) for key in resource})
    totals = {}
    for key in keys:
        values = [resource[key] for resource in resources if isinstance(resource, dict) and key in resource]
        if values and all(isinstance(value, dict) for value in values):
            totals[key] = _resource_totals(values)
        elif values and all(_finite_number(value) for value in values):
            totals[key] = sum(values)
    return totals


def _aggregate_arm(cases, arm):
    valid_cases = [case for case in cases if case["arms"][arm]["valid"]]
    layers = []
    metric_names = ("mean_nll", "mean_squared_error_db2", "mae_db", "coverage90", "mean_interval_width_db")
    for layer in range(4):
        entries = [case["arms"][arm]["metrics"] for case in valid_cases if case["layer"] == layer]
        metrics = {name: _mean([entry[name] for entry in entries]) for name in metric_names}
        layers.append({"layer": layer, "valid_cases": len(entries), "metrics": metrics})
    metrics = {name: _mean([layer["metrics"][name] for layer in layers if layer["valid_cases"]])
               for name in metric_names}
    metrics["rmse_db"] = math.sqrt(metrics["mean_squared_error_db2"]) if valid_cases else None
    metrics["all_mismatch_points"] = sum(case["arms"][arm]["metrics"]["all_mismatch_points"] for case in valid_cases)
    metrics["point_count"] = POINTS_PER_CASE * len(valid_cases)
    return {
        "valid_cases": len(valid_cases),
        "expected_cases": EXPECTED_CASES,
        "valid_rate": len(valid_cases) / EXPECTED_CASES,
        "received_case_valid_rate": len(valid_cases) / len(cases) if cases else None,
        "metrics": metrics,
        "layers": layers,
        "independently_chargeable_resource_totals": _resource_totals([case["arms"][arm]["resources"] for case in cases]),
    }


def _bootstrap_indices(cases_by_layer):
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    counts = [len(cases) for cases in cases_by_layer]
    if len(set(counts)) == 1 and counts[0]:
        draws = generator.integers(0, counts[0], size=(BOOTSTRAP_REPLICATES, 4, counts[0]))
        return [draws[:, layer, :] for layer in range(4)]
    indices = [np.empty((BOOTSTRAP_REPLICATES, count), dtype=np.int64) for count in counts]
    for replicate in range(BOOTSTRAP_REPLICATES):
        for layer, count in enumerate(counts):
            if count:
                indices[layer][replicate] = generator.integers(0, count, size=count)
    return indices


def _comparison(cases_by_layer, indices, comparison, other_arm):
    layer_means, pair_counts, paired_case_ids, samples = [], [], [], []
    same_pair_count = 0 if other_arm in ("numerical_pair", "llm_anonymous") else None
    for layer, cases in enumerate(cases_by_layer):
        values = np.array([case["delta_nll"][comparison] if case["delta_nll"][comparison] is not None else np.nan
                           for case in cases], dtype=np.float64)
        paired = [case for case, value in zip(cases, values) if math.isfinite(value)]
        paired_case_ids.extend(case["case_id"] for case in paired)
        pair_counts.append(len(paired))
        layer_means.append(_mean([case["delta_nll"][comparison] for case in paired]))
        if same_pair_count is not None:
            same_pair_count += sum(set(case["arms"]["llm_semantic"]["ids"]) == set(case["arms"][other_arm]["ids"])
                                   for case in paired)
        if paired:
            resampled = values[indices[layer]]
            counts = np.isfinite(resampled).sum(axis=1)
            means = np.full(BOOTSTRAP_REPLICATES, np.nan)
            np.divide(np.nansum(resampled, axis=1), counts, out=means, where=counts > 0)
            samples.append(means)
    bootstrap_means = np.mean(samples, axis=0) if samples else np.array([])
    usable_samples = bootstrap_means[np.isfinite(bootstrap_means)]
    return {
        "left_arm": "llm_semantic",
        "right_arm": other_arm,
        "paired_cases": sum(pair_counts),
        "paired_case_ids": paired_case_ids,
        "pairs_by_layer": pair_counts,
        "layer_mean_delta_nll": layer_means,
        "mean_delta_nll": _mean([value for value in layer_means if value is not None]),
        "ci95": np.percentile(usable_samples, [2.5, 97.5]).tolist() if len(usable_samples) else None,
        "bootstrap_usable_replicates": len(usable_samples),
        "same_pair_count": same_pair_count,
        "descriptive_only": pair_counts != [CASES_PER_LAYER] * 4,
    }


def summarize(results):
    """Return JSON-safe per-case metrics, paired NLL contrasts, and EP1 gates.

    Comparisons use keys ``llm_semantic_minus_<other arm>``. All comparisons
    share case resamples within each layer. In incomplete batches, estimates
    weight represented layers equally; draws lacking a pair in a represented
    layer are excluded from descriptive intervals and counted explicitly.
    """
    results = list(results)
    if len(results) > EXPECTED_CASES:
        raise ValueError("EP1 accepts at most 24 cases")
    cases, seen_ids = [], set()
    for result in results:
        case_id, layer = result.get("case_id"), result.get("layer")
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError("case_id must be a unique nonempty string")
        if type(layer) is not int or layer not in range(4):
            raise ValueError("layer must be an integer from 0 to 3")
        seen_ids.add(case_id)
        arms = result.get("arms", {})
        if not isinstance(arms, dict):
            raise ValueError("arms must be a dict")
        case = {"case_id": case_id, "layer": layer, "status": result.get("status", "missing"),
                "reason": result.get("reason"), "arms": {arm: _arm_summary(arm, arms.get(arm)) for arm in ARMS}}
        for resource_key in ("resources", "shared_resources", "physical_resources"):
            if resource_key in result:
                case[resource_key] = result[resource_key]
        case["complete"] = case["status"] == "complete" and all(entry["valid"] for entry in case["arms"].values())
        case["delta_nll"] = {}
        for comparison, other_arm in COMPARISONS.items():
            semantic, other = case["arms"]["llm_semantic"], case["arms"][other_arm]
            delta = semantic["metrics"]["mean_nll"] - other["metrics"]["mean_nll"] if semantic["valid"] and other["valid"] else None
            case["delta_nll"][comparison] = delta if delta is not None and math.isfinite(delta) else None
        cases.append(case)
    cases.sort(key=lambda case: (case["layer"], case["case_id"]))
    cases_by_layer = [[case for case in cases if case["layer"] == layer] for layer in range(4)]
    if any(len(layer_cases) > CASES_PER_LAYER for layer_cases in cases_by_layer):
        raise ValueError("each EP1 layer accepts at most six cases")
    indices = _bootstrap_indices(cases_by_layer)
    comparisons = {name: _comparison(cases_by_layer, indices, name, arm) for name, arm in COMPARISONS.items()}
    complete_design = len(cases) == EXPECTED_CASES and all(case["complete"] for case in cases)
    primary_gates = {}
    for arm in ("numerical_pair", "numerical_all"):
        comparison = comparisons[f"llm_semantic_minus_{arm}"]
        primary_gates[arm] = (comparison["paired_cases"] == EXPECTED_CASES
                              and comparison["mean_delta_nll"] <= -0.05 and comparison["ci95"][1] < 0)
    gp = comparisons["llm_semantic_minus_gp"]
    anonymous = comparisons["llm_semantic_minus_llm_anonymous"]
    gp_gate = gp["paired_cases"] == EXPECTED_CASES and gp["mean_delta_nll"] <= 0.05
    semantic_evidence = complete_design and anonymous["ci95"] is not None and anonymous["ci95"][1] < 0
    reasons = []
    if not complete_design:
        reasons.append("预定24场或预定臂缺失、失败、评分无效，或场状态未完成")
    for arm, passed in primary_gates.items():
        if not passed:
            reasons.append(f"LLM-语义相对{LABELS[arm]}未同时满足均差≤−0.05及95%区间上界<0")
    if not gp_gate:
        reasons.append("LLM-语义相对GP未满足完整配对均差≤0.05")
    return _json_safe({
        "experiment": "EP1",
        "notice": NOTICE,
        "received_cases": len(cases),
        "expected_cases": EXPECTED_CASES,
        "missing_cases": EXPECTED_CASES - len(cases),
        "cases_by_layer": [len(layer_cases) for layer_cases in cases_by_layer],
        "complete_cases": sum(case["complete"] for case in cases),
        "bootstrap": {"generator": "PCG64", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
                      "unit": "case", "shared_indices_across_arms": True, "percentiles": [2.5, 97.5],
                      "incomplete_policy": "仅描述：已有配对的层等权；排除任何已有配对层在抽样后无配对的重复，逐比较报告保留次数"},
        "cases": cases,
        "arms": {arm: _aggregate_arm(cases, arm) for arm in ARMS},
        "actual_shared_resource_totals": _resource_totals([
            case.get("physical_resources", case.get("shared_resources", {})) for case in cases]),
        "comparisons": comparisons,
        "promotion": {"passed": complete_design and all(primary_gates.values()) and gp_gate,
                      "complete_design": complete_design, "primary_gates": primary_gates, "gp_gate": gp_gate,
                      "semantic_positive_evidence": semantic_evidence, "reasons": reasons},
    })


def _format(value):
    return "—" if value is None else f"{value:.4f}"


def _comparison_rows(report):
    rows = []
    for comparison in report["comparisons"].values():
        interval = comparison["ci95"]
        rows.append([f"LLM-语义 − {LABELS[comparison['right_arm']]}", str(comparison["paired_cases"]),
                     "/".join(map(str, comparison["pairs_by_layer"])), _format(comparison["mean_delta_nll"]),
                     "—" if interval is None else f"[{_format(interval[0])}, {_format(interval[1])}]",
                     "—" if comparison["same_pair_count"] is None else str(comparison["same_pair_count"])])
    return rows


def _findings(report):
    lines = ["# EP1 静止预测结果", "", NOTICE, "",
             f"收到 {report['received_cases']}/24 场，五臂均完成 {report['complete_cases']}/24 场；晋级判据："
             + ("通过。" if report["promotion"]["passed"] else "未通过。"), "",
             "主终点为每场六点平均 NLL（nat/点，越低越好），先算配对差，再四层等权、层内等权。", "",
             "分层 bootstrap：PCG64(438271)，10,000 次；所有臂共用场索引，95% 百分位区间。", "",
             "| 比较 | 配对场数 | 各层分母 | NLL 均差 | 95% 区间 | 同选一对 |",
             "|---|---:|---|---:|---|---:|"]
    lines.extend("| " + " | ".join(row) + " |" for row in _comparison_rows(report))
    lines.extend(["", "不完整结果仅作描述，不补场、不代选；仅对已有配对的层等权。抽样后缺配对层的重复不用于区间。", ""])
    for comparison in report["comparisons"].values():
        lines.append(f"- 相对 {LABELS[comparison['right_arm']]}：区间保留 {comparison['bootstrap_usable_replicates']}/10000 次重复。")
    lines.extend(["", "语义消融正面证据：" + ("有。" if report["promotion"]["semantic_positive_evidence"] else "未达到合同判据。"), ""])
    lines.extend(f"- {reason}" for reason in report["promotion"]["reasons"])
    lines.extend(["", "| 场 | 层 | 臂 | 状态 | NLL | RMSE (dB) | MAE (dB) | 覆盖90% | 宽度 (dB) | 全失配点 |",
                  "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"])
    for case in report["cases"]:
        for arm, entry in case["arms"].items():
            metrics = entry["metrics"] or {}
            status = str(entry.get("validation_error") or entry.get("reason") or entry["status"])
            cells = [case["case_id"], str(case["layer"]), LABELS[arm], status]
            cells.extend(_format(metrics.get(key)) for key in ("mean_nll", "rmse_db", "mae_db", "coverage90", "mean_interval_width_db", "all_mismatch_points"))
            lines.append("| " + " | ".join(escape(cell).replace("|", "\\|").replace("\n", " ").replace("\r", " ") for cell in cells) + " |")
    lines.extend(["", "各臂独立应计资源（含失败臂已消耗部分；不可相加冒充实际共享工作）：", "", "```json",
                  json.dumps({arm: entry["independently_chargeable_resource_totals"] for arm, entry in report["arms"].items()}, ensure_ascii=False, indent=2, allow_nan=False),
                  "```", "", "实际共享资源（优先汇总 physical_resources，兼容 shared_resources；未提供不表示零工作）：", "", "```json",
                  json.dumps(report["actual_shared_resource_totals"], ensure_ascii=False, indent=2, allow_nan=False), "```", "",
                  "逐场原始资源、状态与指标见 [summary.json](summary.json)，逐场预测与观测见 [静止预测图表](index.html)。", "",
                  "本设计只有一套合成规则，每场一次选择；预测改善不能直接归因于规划或飞行收益。", ""])
    return "\n".join(lines)


def _table(headers, rows):
    heading = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<div class='scroll'><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>"


def _prediction_svg(case):
    colors = ("#2563eb", "#d97706", "#15803d", "#9333ea", "#dc2626")
    plot_keys = ("time_seconds", "observed_dbm", "mean_dbm", "lower90_dbm", "upper90_dbm")
    series = {arm: [row for row in entry["rows"] if isinstance(row, dict)
                    and all(_finite_number(row.get(key)) for key in plot_keys)
                    and row["lower90_dbm"] <= row["upper90_dbm"]]
              for arm, entry in case["arms"].items() if isinstance(entry["rows"], list)}
    rows = [row for arm_rows in series.values() for row in arm_rows]
    if not rows:
        return "<p>无可绘制的已记录预测与观测；未补造预测点。</p>"
    minimum_time = min(row["time_seconds"] for row in rows)
    time_span = max(1., max(row["time_seconds"] for row in rows) - minimum_time)
    minimum_power = min(row[key] for row in rows for key in plot_keys[1:]) - 2.
    power_span = max(4., max(row[key] for row in rows for key in plot_keys[1:]) + 2. - minimum_power)

    def horizontal(time_seconds):
        return 72. + 760. * (time_seconds - minimum_time) / time_span

    def vertical(power_dbm):
        return 285. - 235. * (power_dbm - minimum_power) / power_span

    def points(arm_rows, key):
        return " ".join(f"{horizontal(row['time_seconds']):.2f},{vertical(row[key]):.2f}" for row in arm_rows)

    elements = ["<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 900 355' role='img'>",
                f"<title>{escape(case['case_id'])}：冻结预测与合成观测</title>",
                "<desc>实心黑点为合成观测，彩线为各臂预测均值，阴影为90%观测区间。未试飞。</desc>",
                "<path d='M72 45V285H850' fill='none' stroke='#475569'/>",
                "<text x='10' y='25'>功率 (dBm)</text><text x='785' y='330'>时间 (s)</text>"]
    for tick in range(5):
        power_dbm = minimum_power + power_span * tick / 4
        ordinate = vertical(power_dbm)
        elements.append(f"<path d='M72 {ordinate:.2f}H850' stroke='#e2e8f0'/><text x='8' y='{ordinate + 4:.2f}'>{power_dbm:.1f}</text>")
    for time_seconds in sorted({row["time_seconds"] for row in rows}):
        elements.append(f"<text x='{horizontal(time_seconds):.2f}' y='307' text-anchor='middle'>{time_seconds:g}</text>")
    for arm, color in zip(ARMS, colors):
        arm_rows = sorted(series.get(arm, []), key=lambda row: row["time_seconds"])
        if not arm_rows:
            continue
        polygon = points(arm_rows, "lower90_dbm") + " " + points(list(reversed(arm_rows)), "upper90_dbm")
        elements.append(f"<polygon points='{polygon}' fill='{color}' fill-opacity='.08'/>")
        elements.append(f"<polyline points='{points(arm_rows, 'mean_dbm')}' fill='none' stroke='{color}' stroke-width='2'/>")
    observations = sorted({(row["time_seconds"], row["observed_dbm"]) for row in rows})
    for time_seconds, observed_dbm in observations:
        elements.append(f"<circle cx='{horizontal(time_seconds):.2f}' cy='{vertical(observed_dbm):.2f}' r='3.5' fill='#0f172a'/>")
    for index, (arm, color) in enumerate(zip(ARMS, colors)):
        elements.append(f"<text x='{72 + index * 153}' y='350' fill='{color}'>{escape(LABELS[arm])}</text>")
    return "".join(elements) + "</svg>"


def _html(report):
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>EP1 静止预测结果</title><style>",
             "body{font:16px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#0f172a;background:#f8fafc}",
             "section{background:white;padding:1rem;margin:1rem 0;border:1px solid #cbd5e1;border-radius:8px}",
             "table{border-collapse:collapse;width:100%}th,td{border:1px solid #cbd5e1;padding:.45rem;text-align:left}",
             ".scroll{overflow:auto}svg{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}text{font-size:12px}",
             "</style><body><h1>EP1 静止预测结果</h1>", f"<p><strong>{escape(NOTICE)}</strong></p>",
             f"<p>收到 {report['received_cases']}/24 场，五臂均完成 {report['complete_cases']}/24 场。晋级判据：",
             "通过" if report["promotion"]["passed"] else "未通过", "。</p>",
             "<p>每场六点平均 NLL，nat/点，越低越好。四层等权、层内等权；PCG64(438271)，10000 次配对分层 bootstrap，95% 百分位区间。</p>",
             "<p>不完整结果仅作描述。已有配对的层等权，抽样后缺配对层的重复排除；保留次数见统计详情。观测黑点、预测彩线、90%观测区间阴影。</p>",
             _table(["比较", "配对场数", "各层分母", "NLL 均差", "95% 区间", "同选一对"], _comparison_rows(report)),
             "<p>语义消融正面证据：" + ("有" if report["promotion"]["semantic_positive_evidence"] else "未达到合同判据") + "。</p>",
             "<ul>" + "".join(f"<li>{escape(reason)}</li>" for reason in report["promotion"]["reasons"]) + "</ul>",
             "<p><a href='summary.json'>JSON 指标与资源</a> · <a href='findings.md'>文字结果</a></p>"]
    for case in report["cases"]:
        parts.append(f"<section><h2>{escape(case['case_id'])} · 层 {case['layer']}</h2>")
        parts.append(f"<p>场状态：{escape(str(case['status']))}；{escape(str(case.get('reason') or ''))}</p>")
        parts.append(_prediction_svg(case))
        arm_rows = []
        for arm, entry in case["arms"].items():
            metrics = entry["metrics"] or {}
            arm_rows.append([LABELS[arm], str(entry["ids"]), entry["status"], entry.get("validation_error") or entry["reason"] or "",
                             *[_format(metrics.get(key)) for key in ("mean_nll", "rmse_db", "mae_db", "coverage90", "mean_interval_width_db", "all_mismatch_points")]])
        parts.append(_table(["臂", "选择", "状态", "原因", "NLL", "RMSE dB", "MAE dB", "覆盖90%", "宽度 dB", "全失配点"], arm_rows))
        parts.append("<details><summary>逐场配对差和资源</summary><pre>" + escape(json.dumps(
            {"delta_nll": case["delta_nll"], "resources": {arm: entry["resources"] for arm, entry in case["arms"].items()},
             "physical_resources": case.get("physical_resources", case.get("shared_resources"))}, ensure_ascii=False, indent=2, allow_nan=False)) + "</pre></details></section>")
    parts.append("<details><summary>完整统计、有效率与资源汇总</summary><pre>" + escape(json.dumps(
        {key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2, allow_nan=False)) + "</pre></details>")
    parts.append("<p>本设计只有一套合成规则，每场一次选择；解释范围限于本设计。</p></body></html>")
    return "".join(parts)


def publish_report(out: Path, results: list, manifest: dict):
    """Exclusively write a self-contained static report from already scored data."""
    report = summarize(results)
    report["manifest"] = _json_safe(manifest)
    closure = report["manifest"].get("closure")
    closure_completed = isinstance(closure, dict) and closure.get("stop") == "completed"
    report["promotion"]["closure_completed"] = closure_completed
    report["promotion"]["passed"] = report["promotion"]["passed"] and closure_completed
    if not closure_completed:
        report["promotion"]["reasons"].append("批次未以 manifest.closure.stop == 'completed' 正常闭批")
    documents = {
        "summary.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "findings.md": _findings(report),
        "index.html": _html(report),
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reserved = []
    try:
        with ExitStack() as stack:
            streams = {}
            for name in documents:
                path = out / name
                streams[name] = stack.enter_context(path.open("x", encoding="utf-8"))
                reserved.append(path)
            for name, body in documents.items():
                streams[name].write(body)
    except FileExistsError:
        for path in reserved:
            path.unlink()
        raise
    return report
