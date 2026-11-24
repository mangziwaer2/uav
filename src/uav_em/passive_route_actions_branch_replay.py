"""Render the P7 branch overlay from published replay and public candidate data."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P7 = ROOT / "outputs/passive_route_actions_P7_20261124"
SCRIPT = Path(__file__).with_name("passive_route_actions_branches.js")
DATA_START = '<script type="application/json" id="data">'


def _branch_data(review, results):
    routes = review["routes"]
    segments, seen = [], set()
    for row in routes:
        route = row["route_xy_m"]
        for start, end in zip(route, route[1:]):
            edge = tuple(sorted((tuple(start), tuple(end))))
            if start != end and edge not in seen:
                seen.add(edge)
                segments.append([start, end])
    return dict(t_s=review["t_s"], routes=routes, segments=segments, waypoints=review["waypoints"],
                selected={mode: metrics["route_idx"] for mode, metrics in results["arms"].items()},
                waypoint_reachable=review["waypoint_reachable_route_indices"])


def render(page, review, results, *, choice_s=None):
    frozen = json.loads(page.split(DATA_START, 1)[1].split("</script>", 1)[0])
    branch = _branch_data(review, results)
    routes = {row["route_idx"]: row["route_xy_m"] for row in branch["routes"]}
    if frozen["arms"].keys() != branch["selected"].keys():
        raise ValueError("result and replay arms disagree")
    actual_choice_s = branch["t_s"] if choice_s is None else choice_s
    if choice_s is not None:
        branch["choice_s"] = choice_s
    for mode, arm in frozen["arms"].items():
        decision = next(row for row in arm["log"]["decisions"] if row["t"] == actual_choice_s)
        selected = branch["selected"][mode]
        if choice_s is None:
            matches = decision["selected"] == selected and decision["candidates"][selected]["route"] == routes[selected]
        else:
            picked = decision["candidates"][decision["selected"]]
            matches = picked["checkpoint_route_id"] == selected and picked["checkpoint_route"] == routes[selected]
        if not matches:
            raise ValueError("candidate overlay does not match the executed decision")

    payload = json.dumps(branch, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    controls = ('<label><input type="checkbox" id="branch-layer" checked> 24 秒候选与中间点</label>\n'
                '  <label>预览 <select id="branch-focus"><option value="">无</option></select></label>\n')
    keys = ('<span><i class="solid"></i>已飞</span><span><i class="planned"></i>选定未飞</span>'
            '<span><i class="candidate"></i>24 秒合法候选</span><span><i class="preview"></i>预览路线</span>')
    comparison = '''<section id="branch-review" class="summary branch-review" aria-labelledby="branch-title">
<h2 id="branch-title">24 秒决策</h2><p id="branch-choice"></p>
<table><thead><tr><th scope="col">候选</th><th scope="col">经过中间点</th><th scope="col">余程 (s)</th>
<th scope="col">未知 (s)</th><th scope="col">预测 T+Q</th><th scope="col">本组</th></tr></thead>
<tbody id="branch-routes"></tbody></table>
<details><summary>中间点与数值补全</summary><dl id="branch-waypoints"></dl></details>
</section>'''
    style = '''.planned,.candidate,.preview{display:inline-block;width:20px;border-top:3px dashed #007c83;
vertical-align:middle;margin-right:5px}.candidate{border-color:#6d7b82}.preview{border-color:#a96c11}
#branch-layer{accent-color:#17746f}.branch-review{padding-block:12px;margin-bottom:14px}
.branch-review p{margin:0 0 8px}.branch-review tr[data-picked="true"]{background:#e0f1ec;font-weight:650}
.branch-review tr[data-preview="true"]{outline:2px dashed #aa6b17;outline-offset:-2px}
.branch-review details{margin-top:10px}.branch-review summary{cursor:pointer}
#branch-waypoints{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;margin:7px 0}
#branch-waypoints dd{margin:0}'''
    route_draw_start = page.index('  if (decision?.t===frame.t && decision.changed) path(belief,')
    route_draw_end = page.index('  const reason={initial:', route_draw_start)
    page = page[:route_draw_start] + '''  drawBranchOverlay(log, frame, index);
  for (const point of log.points.slice(0,frame.prefix_size)) dot(belief,point.position_m,3.5,color(point.interference_dbm,-105,-65));
  drawBranchMarkers(frame, mode);
  dot(belief,frame.xy,7,"#fff");
''' + page[route_draw_end:]
    edits = {
        "P7 · 路线与中间目标试飞": "P7 · 分岔与实际航迹",
        "</style>": style + "\n</style>",
        '<option value="n_route">': '<option value="n_route" selected>',
        '</div>\n<div class="stats"': controls + '</div>\n<div class="stats"',
        '<span><i class="solid"></i>执行余程</span><span><i class="dash"></i>换路前余程</span>': keys,
        '<div class="summary"><table>': comparison + '\n<div class="summary"><table>',
        '<footer><a href="contract.md">': '<footer><a href="index.html">原回放</a><a href="contract.md">',
        '<script>\n"use strict";': ('<script type="application/json" id="branch-data">' + payload
                                    + '</script>\n<script>\n"use strict";'),
        'function draw() {': SCRIPT.read_text(encoding="utf-8") + '\nfunction draw() {',
        'configure(24);': 'setupBranchReview();\nconfigure(branchData.t_s);',
    }
    for before, after in edits.items():
        if before not in page:
            raise ValueError(f"published replay marker missing: {before}")
        page = page.replace(before, after)
    return page


def publish():
    publication = json.loads((P7 / "publication.json").read_bytes())
    sources = {}
    for name in ("index.html", "review.json", "results.json"):
        content = (P7 / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != publication["files"][name]:
            raise ValueError(f"published P7 original changed: {name}")
        sources[name] = content
    page = render(sources["index.html"].decode("utf-8"), json.loads(sources["review.json"]),
                  json.loads(sources["results.json"]))
    with (P7 / "branches.html").open("x", encoding="utf-8") as stream:
        stream.write(page)


if __name__ == "__main__":
    publish()
