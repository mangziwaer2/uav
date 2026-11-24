"""Publish complete decision overlays from already recorded double-map replays."""

from html import escape
import json
from pathlib import Path


OUT = Path(__file__).resolve().parents[2] / "outputs"
TEMPLATE = Path(__file__).with_name("passive_map_replay.html")
SCRIPT = Path(__file__).with_name("passive_decision_replay.js")
DATA = '<script type="application/json" id="data">'
P7_LABELS = dict(n_route="N · 全域枚举", n_waypoint="N · 中间目标域", n_guard="N · 原软门控",
                 llm_route="LLM · 直接选路", llm_waypoint="LLM · 中间目标")
P3_LABELS = dict(risk="风险 T+Q", mean="均值 T+Q", shortest="最短航时")
P7 = "passive_route_actions_P7_20261124"
GRAPH = "passive_graph_review_P7_20261124"
P3 = "passive_pricing_P3_20261123"


def _snapshot(clock, candidates, selected, *, price_clock=None):
    vertices, edges = {}, {}
    for candidate in candidates:
        route = candidate["xy"]
        for point in route[1:-1]:
            vertices.setdefault(tuple(point), len(vertices))
        for start, end in zip(route, route[1:]):
            if start != end:
                edges.setdefault((tuple(start), tuple(end)), [start, end])
        candidate["waypoints"] = [vertices[tuple(point)] for point in route[1:-1]]
    if not candidates or (selected is not None and selected not in {row["id"] for row in candidates}):
        raise ValueError("recorded decision must refer to one of its own candidates")
    return dict(t=clock, price_t=clock if price_clock is None else price_clock, routes=candidates, selected=selected,
                edges=list(edges.values()), vertices=list(vertices))


def _snapshots(mode, log, menu):
    review = {} if menu is None else {row["route_idx"]: row for row in menu["routes"]}
    menu_clock = None if menu is None else menu["t_s"]
    snapshots = []
    for decision in log["decisions"]:
        selected = decision["selected"]
        if isinstance(selected, dict):
            selected = selected[mode]
        candidates, price_clock = [], decision["t"]
        for index, row in enumerate(decision["candidates"]):
            route_id = row.get("checkpoint_route_id", index)
            published = review.get(index, {}) if decision["t"] == menu_clock else {}
            if published and published["route_xy_m"] != row["route"]:
                raise ValueError("frozen menu geometry differs from recorded decision")
            if "checkpoint_route_id" in row:
                published = review[route_id]
                if row["checkpoint_route"] != published["route_xy_m"]:
                    raise ValueError("conditioned candidate differs from its frozen parent")
                price_clock = menu_clock
            costs = row.get("costs", {})
            candidates.append(dict(id=route_id, xy=row["route"],
                                   travel_s=row.get("time_s", published.get("travel_s")),
                                   unknown_s=row.get("unknown_s", row.get("coverage", {}).get(
                                       "unknown", published.get("unknown_s"))),
                                   compared_s=row.get("compared_s"),
                                   total=costs.get("mean" if mode == "mean" else "risk", published.get("risk_proxy_tq"))))
        snapshot = _snapshot(decision["t"], candidates, None if selected is None else candidates[selected]["id"],
                             price_clock=price_clock)
        if "authority" in decision:
            snapshot.update(authority=decision["authority"], numerical=decision["numerical"])
        snapshots.append(snapshot)
    if menu is not None and not any(item["t"] == menu_clock for item in snapshots):
        candidates = [dict(id=row["route_idx"], xy=row["route_xy_m"], travel_s=row["travel_s"],
                           unknown_s=row["unknown_s"], total=row["risk_proxy_tq"]) for row in review.values()]
        snapshots.append(_snapshot(menu_clock, candidates, None))
    clocks = {frame["t"] for frame in log["frames"]}
    if any(item["t"] not in clocks for item in snapshots):
        raise ValueError("snapshot has no matching recorded flight frame")
    return sorted(snapshots, key=lambda item: item["t"])


def _payload(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def render(published, labels, title, original_href, *, menu=None):
    raw = published.split(DATA, 1)[1].split("</script>", 1)[0]
    arms = json.loads(raw)["arms"]
    if arms.keys() != labels.keys():
        raise ValueError("display groups and recorded flight groups differ")
    snapshots = {mode: _snapshots(mode, arms[mode]["log"], menu) for mode in labels}
    page = TEMPLATE.read_text(encoding="utf-8")
    first_option = page.index('<option value="local">')
    option_end = page.index('</select>', first_option)
    options = "".join(f'<option value="{escape(mode)}">{escape(label)}</option>' for mode, label in labels.items())
    page = page[:first_option] + options + page[option_end:]
    controls = '''  <label>候选快照 <select id="decision-time"></select></label>
  <label>预览路线 <select id="decision-route"></select></label>
'''
    table = '''<section class="summary" aria-labelledby="decision-title">
<h2 id="decision-title">合法候选余程</h2><p id="decision-info"></p>
<table><thead><tr><th>路线</th><th>中间点</th><th>T (s)</th><th>未知 (s)</th>
<th>Q 代理</th><th>T+Q 代理</th><th>记录动作</th></tr></thead><tbody id="decision-rows"></tbody></table>
</section>'''
    legend = '''<span><i class="solid"></i>已飞轨迹</span><span><i class="dash" style="border-color:#007c83"></i>已承诺余程</span>
  <span><i class="dash" style="border-color:#687981"></i>快照候选</span><span><i class="dash" style="border-color:#b57515"></i>预览路线</span>'''
    script = SCRIPT.read_text(encoding="utf-8")
    draw_start, draw_end = page.index("function draw() {"), page.index("function pause() {")
    page = page[:draw_start] + script + "\n" + page[draw_end:]
    edits = {
        "P1 · 被动全图估计回放": escape(title),
        "冻结决策快照 · 仅当时信息": "GP 与候选余程 · 仅当时信息",
        '</div>\n<div class="stats"': controls + '</div>\n<div class="stats"',
        '<span><i class="solid"></i>执行余程</span><span><i class="dash"></i>另一评分器首选</span>': legend,
        "未用于该组定价": "GP 未支持",
        '<div class="summary"><table>': table + '\n<h2>全程实际结果</h2><div class="summary"><table>',
        page[page.index("<footer>"):page.index("</footer>")]:
            f'<footer><a href="{escape(original_href)}">原回放</a><a href="../index.html">其他场景</a>',
        "{{REPLAY_DATA}}": raw,
        '<script>\n"use strict";': '<script type="application/json" id="decision-data">'
                                   + _payload(snapshots) + '</script>\n<script>\n"use strict";',
        'const labels = {local:"Local · 近场 GP", full:"Full · 全图 GP"};': "const labels = " + _payload(labels) + ";",
        "  const event=forward?log.decisions.find(item=>item.t>time):log.decisions.filter(item=>item.t<time).at(-1);":
            "  const events=decisionData[arm.value];\n"
            "  const event=forward?events.find(item=>item.t>time):events.filter(item=>item.t<time).at(-1);",
        "const {log}=arms[arm.value], time=log.frames[Number(range.value)].t;":
            "const time=arms[arm.value].log.frames[Number(range.value)].t;",
        'arm.addEventListener("change",()=>{pause();configure();});':
            'arm.addEventListener("change",()=>{pause();configure(Number(byId("time").value.split(" / ")[0]));});',
        "</style>": '#decision-rows tr[data-selected="true"]{background:#e0f1ec;font-weight:650}\n</style>',
    }
    for before, after in edits.items():
        if before not in page:
            raise ValueError(f"missing decision replay marker: {before[:100]}")
        page = page.replace(before, after, 1 if before != "P1 · 被动全图估计回放" else 2)
    return page


def publish(root=OUT):
    target = root / "decision_replays_20261124"
    cases = [
        ("p7", f"{P7}/index.html", P7_LABELS, "P7 · 全部分岔 · 43537", P7),
        ("p7_llm", f"{GRAPH}/index.html", {"n_route": "LLM 复核 · 数值默认执行"},
         "P7 · LLM 复核与完整分岔 · 43537", GRAPH),
        *((f"p3_{seed}", f"{P3}/replay_{seed}.html", P3_LABELS,
           f"P3 · 全部分岔 · {seed}", None) for seed in (43401, 43411)),
    ]
    pages = {}
    for name, source, labels, title, review_dir in cases:
        menu = json.loads((root / review_dir / "review.json").read_bytes()) if review_dir else None
        pages[name] = render((root / source).read_text(encoding="utf-8"), labels, title, f"../{source}", menu=menu)
    target.mkdir(exist_ok=True)
    for name, page in pages.items():
        (target / f"{name}.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    publish()
