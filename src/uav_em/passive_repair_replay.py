"""Render recorded repair decisions without sensing, planning or evaluating them."""

from html import escape
import math

from .motion import advance_route
from .passive_decision_replay import DATA, _payload, render


def _same_polyline(first, second):
    return len(first) == len(second) and all(math.dist(a, b) <= 1e-7 for a, b in zip(first, second))


def _check_execution(layout, log, frames):
    if (not _same_polyline(frames[0]["moved"], [layout.start_m]) or frames[0]["t"] != 0
            or frames[0]["flown_m"] != 0 or math.dist(frames[0]["xy"], layout.start_m) > 1e-7
            or (log["stop"] == "arrived") != (math.dist(frames[-1]["xy"], layout.goal_m) <= 1e-7)):
        raise ValueError("recorded takeoff or stop disagrees with the actual trace")
    linked = 0
    for index, frame in enumerate(frames):
        next_frame = frames[index + 1] if index + 1 < len(frames) else None
        actual = next_frame["flown_m"] - frame["flown_m"] if next_frame else 0.0
        if actual < 0 or not math.isclose(frame["advanced_m"], actual, abs_tol=1e-7):
            raise ValueError("recorded advancement does not match the actual trace")
        if "decision" in frame:
            if frame["decision"] != linked or linked >= len(log["decisions"]):
                raise ValueError("broken frame-to-decision link")
            event = log["decisions"][linked]
            linked += 1
            selected = event["selected"]
            if (event["t"] != frame["t"] or not math.isclose(event["advanced_m"], actual, abs_tol=1e-7)
                    or (selected is None and actual > 0)
                    or (selected is not None and (type(selected) is not int or not 0 <= selected < len(event["candidates"])
                        or not _same_polyline(frame["remaining"], event["candidates"][selected]["route"])))):
                raise ValueError("recorded commit does not match the frame")
        if next_frame:
            if actual <= 0 or next_frame["t"] <= frame["t"] or not frame["remaining"]:
                raise ValueError("recorded trajectory has no forward progress")
            _, expected = advance_route(frame["remaining"], actual)
            moved = next_frame["moved"]
            if (not moved or not _same_polyline([moved[0], moved[-1]], [frame["xy"], next_frame["xy"]])
                    or not _same_polyline(moved, expected)
                    or not math.isclose(sum(math.dist(a, b) for a, b in zip(moved, moved[1:])), actual, abs_tol=1e-7)):
                raise ValueError("actual trace diverges from the recorded commitment")
    if linked != len(log["decisions"]):
        raise ValueError("unlinked decision in replay")


def _replay_arm(layout, entry):
    log, evaluation = entry["log"], entry["evaluation"]
    frames, metrics = log["frames"], evaluation["metrics"]
    if (not frames or len(frames) != len(evaluation["trajectory"]) or log["stop"] != metrics["stop"]
            or not math.isclose(metrics["T"], frames[-1]["t"])
            or not math.isclose(metrics["length_m"], frames[-1]["flown_m"])):
        raise ValueError("evaluation must match the recorded execution")
    _check_execution(layout, log, frames)
    decisions = []
    for event in log["decisions"]:
        row = {key: value for key, value in event.items()
               if key not in ("exchange", "authority", "action_inputs", "tool_trace")}
        if "tool_trace" in event:
            row["tool_steps"] = [turn["feedback"] or dict(status=turn["exchange"]["status"])
                                 for turn in event["tool_trace"]]
        row["controller"] = event["authority"]
        decisions.append(row)
    return dict(log=dict(frames=frames, points=log["points"], decisions=decisions), evaluation=evaluation)


def render_repair_replay(layout, arms, labels, *, synthetic, evidence_href, pair_summary=None):
    grid = dict(rows=layout.rows, cols=layout.columns, spacing=layout.resolution_m, start=layout.start_m,
                goal=layout.goal_m, obstacles=layout.obstacles_m,
                buildings=[item["footprint_m"] for item in layout.buildings])
    payload = dict(grid=grid, arms={mode: _replay_arm(layout, entry) for mode, entry in arms.items()})
    has_tools = any("tool_steps" in event for arm in payload["arms"].values() for event in arm["log"]["decisions"])
    title = ("合成工具链回放 · 非试飞结果" if has_tools else "合成执行回放 · 非试飞结果") if synthetic else "整程改路双图回放"
    page = render(DATA + _payload(payload) + "</script>", labels, title, evidence_href)
    start, end = page.index("  const action = snapshot?"), page.index('\n}\n\ndecisionTime.addEventListener')
    page = page[:start] + '''  const controller = {initial:"共同起飞", llm:"LLM 通道", numerical:"数值组"};
  const action = candidate ? `${controller[decision.controller]} 承诺 ${routeLabel(candidate.id)} · 下一步实际推进 ${fmt(decision.advanced_m)} m`
    : decision ? `未提交路线 · ${stopLabel(decision.status)}` : "无决策";
  const comparison = decision?.comparison;
  const coverage = comparison ? ` · 分岔后可比 ${fmt(comparison.post_common_s)} s` : "";
  const shared = decision?.action_count ? ` · 共用表 ${decision.action_count} 个动作${decision.action_id == null ? "" : `，选择 A${decision.action_id}`}` : "";
  const price = !candidate ? "无新承诺余程" : candidate.total == null ? "所选余程未定价" : "只积分冻结窗内共同支持";
  const history = snapshot && frame.t > snapshot.t ? " · 历史快照" : "";
  byId("decision-info").textContent = snapshot ?
    `t=${fmt(snapshot.t, 1)} s · ${action}${shared} · ${price}${coverage}${history}` : "无候选快照";''' + page[end:]
    edits = {
        ">原回放</a>": ">执行证据（含原始报文字节）</a>",
        "<th>T (s)</th><th>未知 (s)</th>": "<th>余程 T (s)</th><th>窗内未知 (s)</th>",
        "<h2>全程实际结果</h2>": "<h2>已执行航段 · 未到达不计为全程结果</h2>",
        "stops[metrics.stop],": "stopLabel(metrics.stop),",
        '${stops[evaluation.metrics.stop] ?? evaluation.metrics.stop}': '${stopLabel(evaluation.metrics.stop)}',
        "全程 T+Q ${fmt(evaluation.metrics.T_plus_Q)}":
            '${evaluation.metrics.stop === "arrived" && evaluation.metrics.legal ? "本次全程" : "停止时已飞前缀"} '
            'T+Q ${fmt(evaluation.metrics.T_plus_Q)}',
        'const decisionData =': '''function stopLabel(status) {
  const failures = {compute_budget:"查询或拟合预算不足", geometry_budget:"几何搜索资源耗尽",
    point_budget:"点测预算用尽", missing_reply:"缺少回执",
    invalid_reply:"无效回执", invalid_session:"无效会话", reused_session:"会话重复", delivery_error:"转交中断",
    request_mismatch:"请求不一致", delivery_mismatch:"接收内容不一致"};
  return stops[status] ?? failures[status] ?? status;
}
const decisionData =''',
        'proxy(exposure), proxy(route.total),\n                         selected ? "采用" : "—"':
            'exposure === null ? "未定价" : fmt(exposure), route.total === null ? "未定价" : fmt(route.total),\n'
            '                         selected ? "承诺" : "—"',
        'const alertText =': 'Object.assign(alertState, {new_branch:"发现新分岔", compute_budget:stopLabel("compute_budget"),\n'
                             '    missing_reply:stopLabel("missing_reply")});\n  const alertText =',
        '${alertState[frame.alert.state]}': '${alertState[frame.alert.state] ?? stopLabel(frame.alert.state)}',
        '  const candidate = snapshot?.routes.find': '''  const repair = decision?.repair;
  if (repair) {
    path(belief, repair.replaced, "#f28698", 3, true);
    path(belief, repair.replacement, "#a0f5e5", 2, true);
  }
  const candidate = snapshot?.routes.find''',
        '  const pointCount =': '''  if (repair?.waypoint) {
    dot(belief, repair.waypoint, 6, "#ffd169");
    text(belief, repair.waypoint, "引导点", 10, 14);
  }
  const pointCount =''',
        '\n}\nfunction path(ctx,': '''
  for (const outline of grid.buildings) {
    ctx.beginPath();ctx.moveTo(...xy(ctx, outline[0]));
    for (const point of outline.slice(1)) ctx.lineTo(...xy(ctx, point));
    ctx.closePath();ctx.fillStyle="#465257";ctx.fill();ctx.strokeStyle="#dce7e0";ctx.lineWidth=1.5;ctx.stroke();
  }
}
function path(ctx,''',
        '<span><i class="unknown"></i>GP 未支持</span>': '''<span><i class="dash" style="border-color:#ce5269"></i>被替换子段</span>
  <span><i class="dot" style="background:#ffd169"></i>引导点</span><span><i class="unknown"></i>GP 未支持</span>''',
    }
    if synthetic:
        edits["逐时真场 · 事后观察"] = "合成真场 · 事后绘制"
        edits["飞行 <select"] = "夹具 <select"
    if pair_summary is not None:
        edits["<h2>全程实际结果</h2>"] += f"<p>{escape(pair_summary)}</p>"
        edits["<a href=\"../index.html\">其他场景</a>"] = ""
    if has_tools:
        edits['\nconfigure();'] = '\nconfigure(arms[arm.value].log.decisions.find(event => event.tool_steps)?.t ?? 0);'
        edits['<p id="decision-info"></p>'] = '''<p id="decision-info"></p>
<details id="tool-details" open hidden><summary>本次检查点的工具调用</summary><ol id="tool-steps"></ol></details>'''
        edits[' : "无候选快照";'] = ''' : "无候选快照";
  const steps = decision?.tool_steps ?? [];
  byId("tool-details").hidden = !steps.length;
  const statuses = {blocked_waypoint:"引导点不可飞", blocked_first_edge:"首段不可飞", generated:"生成合法余程",
    comparable:"同窗可比", committed:"提交余程", resources:"超出硬资源", soft_detour_limit:"超出本轮软绕行余量",
    no_baseline_evidence:"参照无证据", missing_support:"缺共同支持", insufficient_post_support:"分岔后支持不足",
    duplicate:"同轨，复用已有路线", geometry_budget:"几何搜索资源耗尽"};
  byId("tool-steps").replaceChildren(...steps.map(step => {
    const item = document.createElement("li");
    const coordinates = step.waypoint_xy_m ? ` · 引导点 (${step.waypoint_xy_m.map(value => fmt(value, 1)).join(", ")})` : "";
    const reference = step.reference_id == null ? "" : ` / 参照 ${routeLabel(step.reference_id)}`;
    const route = step.route_id == null ? "" : ` · ${routeLabel(step.route_id)}${reference}`;
    const args = step.args ? ` · ${JSON.stringify(step.args)}` : "";
    const evidence = step.comparison;
    const risk = evidence ? ` · ΔT ${fmt(evidence.delta_t_s)} s，分岔后可比 ${fmt(evidence.post_common_s)} s，` +
      (evidence.delta_tail_q == null ? "ΔQ 未定价" : `期望 ΔQ ${fmt(evidence.delta_expected_q)}，上尾 ΔQ ${fmt(evidence.delta_tail_q)}`) : "";
    item.style.overflowWrap = "anywhere";
    item.textContent = `${step.tool ?? "回执"}：${statuses[step.status] ?? stopLabel(step.status)}${coordinates}${route}${risk}${args}`;
    return item;
  }));'''
    for before, after in edits.items():
        if before not in page:
            raise ValueError(f"missing repair replay marker: {before[:100]}")
        page = page.replace(before, after, 1)
    return page
