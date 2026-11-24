const decisionData = JSON.parse(byId("decision-data").textContent);
const decisionTime = byId("decision-time"), routeFocus = byId("decision-route");
let shownMode, shownSnapshot;

function routeLabel(id) { return `R${id}`; }
function waypointLabel(id) { return `W${id}`; }
function proxy(value) { return value === null ? "未记录" : fmt(value); }

function updateDecisionControls(mode, events, snapshot) {
  if (shownMode !== mode) {
    decisionTime.replaceChildren(...events.map(item => new Option(`${fmt(item.t, 1)} s · ${item.routes.length} 条`, item.t)));
    decisionTime.disabled = !events.length;
    shownMode = mode;
  }
  if (shownSnapshot === snapshot) return;
  shownSnapshot = snapshot;
  const routes = snapshot?.routes ?? [];
  decisionTime.value = snapshot?.t ?? "";
  routeFocus.replaceChildren(new Option("无", ""), ...routes.map(route => new Option(routeLabel(route.id), route.id)));
  routeFocus.disabled = !routes.length;
  const body = byId("decision-rows");
  body.replaceChildren();
  for (const route of routes) {
    const row = body.insertRow(), selected = route.id === snapshot.selected;
    row.dataset.selected = String(selected);
    const exposure = route.total !== null && route.travel_s !== null && route.compared_s !== 0
      ? route.total - route.travel_s : null;
    for (const value of [routeLabel(route.id), route.waypoints.map(waypointLabel).join(" → ") || "直达",
                         proxy(route.travel_s), proxy(route.unknown_s), proxy(exposure), proxy(route.total),
                         selected ? "采用" : "—"]) row.insertCell().textContent = value;
  }
}

function draw() {
  const mode = arm.value, {log, evaluation} = arms[mode];
  const index = Number(range.value), frame = log.frames[index], current = evaluation.trajectory[index];
  const events = decisionData[mode], snapshot = events.filter(item => item.t <= frame.t).at(-1);
  const decision = log.decisions.filter(item => item.t <= frame.t).at(-1);
  const flown = log.frames.slice(0, index + 1).flatMap(item => item.moved);
  updateDecisionControls(mode, events, snapshot);
  heat(truth, current.dbm);
  endpoints(truth);
  path(truth, flown, "#a0f5e5", 3);
  for (const source of current.sources) {
    dot(truth, source.position_m, 8, "#ffd169");
    text(truth, source.position_m, `✚ ${source.id}`, 11, -9);
  }
  dot(truth, frame.xy, 7, "#fff");
  byId("truth-info").textContent = `t=${fmt(frame.t, 1)} s · 活动源 ${current.sources.length} · 当前位置点测 ${fmt(frame.measured_dbm, 1)} dBm`;
  byId("time").value = `${fmt(frame.t, 1)} / ${fmt(log.frames.at(-1).t, 1)} s`;
  byId("stats").textContent = `${labels[mode]} · 已飞 ${fmt(frame.flown_m, 1)} m · 全程 T+Q ${fmt(evaluation.metrics.T_plus_Q)} · ${stops[evaluation.metrics.stop] ?? evaluation.metrics.stop}`;
  byId("previous").disabled = !events.some(item => item.t < frame.t);
  byId("next").disabled = !events.some(item => item.t > frame.t);

  const gp = frame.gp ?? decision?.gp, gpTime = frame.gp ? frame.t : decision?.t;
  const std = powerLayer.value === "std";
  if (gp) heat(belief, std ? gp.std_db : gp.dbm, gp.support, 1, std);
  else belief.clearRect(0, 0, belief.canvas.width, belief.canvas.height);
  endpoints(belief);
  for (const edge of snapshot?.edges ?? []) path(belief, edge, "#d6e0e3", 1.6, true);
  const candidate = snapshot?.routes.find(route => route.id === snapshot.selected);
  path(belief, frame.remaining ?? candidate?.xy ?? [], "#a0f5e5", 4, true);
  const preview = routeFocus.value === "" ? undefined : snapshot?.routes.find(route => route.id === Number(routeFocus.value));
  if (preview) path(belief, preview.xy, "#ffc268", 3, true);
  path(belief, flown, "#a0f5e5", 4);
  const highlighted = new Set((preview ?? candidate)?.waypoints);
  const vertices = snapshot?.vertices ?? [];
  for (const [id, point] of vertices.entries()) {
    dot(belief, point, 3.5, highlighted.has(id) ? (preview ? "#ffc268" : "#a0f5e5") : "#d6e0e3");
    if (highlighted.has(id) || !preview) text(belief, point, waypointLabel(id), 8, -5);
  }
  const pointCount = frame.prefix_size ?? log.points.filter(point => point.time_seconds <= frame.t).length;
  for (const point of log.points.slice(0, pointCount)) dot(belief, point.position_m, 3.5, color(point.interference_dbm, -105, -65));
  dot(belief, frame.xy, 7, "#fff");
  const alertState = {rearmed:"未预警", already_reviewed:"持续预警", reviewed:"预警复核", no_alternative:"无备选",
                      event_budget:"复核预算用尽", timeout:"等待超时", invalid_reply:"无效回执", request_budget:"请求预算用尽"};
  const alertText = frame.alert ? ` · ${alertState[frame.alert.state]}${frame.alert.peak_dbm === null ? "" : ` ${fmt(frame.alert.peak_dbm, 1)} dBm`}` : "";
  byId("belief-info").textContent = `${gp ? `GP t=${fmt(gpTime, 1)} s` : "未记录 GP"} · ${pointCount} 个原始点${alertText}${std ? " · 色标 0–15 dB" : ""}`;
  const action = snapshot?.authority ? (candidate ? `${snapshot.authority === "llm" ? "LLM" : "N"} 采用 ${routeLabel(candidate.id)} · 同菜单 N ${routeLabel(snapshot.numerical)}`
    : `未执行 · ${alertState[snapshot.authority] ?? snapshot.authority}`) :
    snapshot?.selected === null ? "待决策" : candidate ? `采用 ${routeLabel(candidate.id)}` : "无决策";
  const history = snapshot && frame.t > snapshot.t ? " · 历史候选，不代表当前位置仍可执行" : "";
  const pricing = snapshot?.authority ? "只积分窗内共同支持，不给未知定价" : snapshot && snapshot.price_t !== snapshot.t
    ? `表中 T、Q 与未知时长来自 ${fmt(snapshot.price_t, 1)} s 冻结的名义定价`
    : mode === "mean" ? "均值代理" : mode === "shortest" ? "仅按 T 选路；风险代理供比较" : "风险代理";
  byId("decision-info").textContent = snapshot ?
    `t=${fmt(snapshot.t, 1)} s · ${snapshot.routes.length} 条候选 · ${action} · ${pricing}${history}` : "无候选快照";
}

decisionTime.addEventListener("change", () => { pause(); configure(Number(decisionTime.value)); });
routeFocus.addEventListener("change", draw);
