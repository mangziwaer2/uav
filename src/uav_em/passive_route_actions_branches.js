const branchData = JSON.parse(byId("branch-data").textContent);
const branchLayer = byId("branch-layer");
const branchFocus = byId("branch-focus");
const branchRoutes = new Map(branchData.routes.map(route => [route.route_idx, route]));
const waypointByPosition = new Map(branchData.waypoints.map(point => [point.xy_m.join(","), point.waypoint_idx]));
const waypointName = index => `W${index}`;
const routeName = index => `R${index}`;
const routeWaypoints = route => route.route_xy_m.slice(1, -1)
  .map(point => waypointByPosition.get(point.join(","))).filter(index => index !== undefined);
const waypointChain = route => routeWaypoints(route).map(waypointName).join(" \u2192 ");

function drawBranchOverlay(log, frame, frameIndex) {
  if (frame.t >= branchData.t_s && branchLayer.checked) {
    for (const segment of branchData.segments) path(belief, segment, "#b9c3c8", 1.6, true);
    if (branchFocus.value !== "") {
      const preview = branchRoutes.get(Number(branchFocus.value));
      path(belief, preview.route_xy_m, "#ffc268", 3, true);
    }
  }
  path(belief, frame.remaining, "#a0f5e5", 4, true);
  const flown = log.frames.slice(0, frameIndex + 1).flatMap(item => item.moved);
  path(belief, flown, "#a0f5e5", 4);
}

function drawBranchMarkers(frame, mode) {
  const available = frame.t >= branchData.t_s;
  const chosen = branchRoutes.get(branchData.selected[mode]);
  branchLayer.disabled = !available;
  branchFocus.disabled = !available || !branchLayer.checked;
  byId("branch-review").hidden = !available;
  if (!available) return;

  const decided = frame.t >= (branchData.choice_s ?? branchData.t_s);
  const selectedWaypoints = new Set(decided ? routeWaypoints(chosen) : []);
  if (branchLayer.checked) {
    const canvasScale = belief.canvas.width / Math.max(belief.canvas.clientWidth, 1);
    const scale = Math.min(3, Math.max(1, canvasScale));
    for (const point of branchData.waypoints) {
      const selected = selectedWaypoints.has(point.waypoint_idx);
      const [left, top] = xy(belief, point.xy_m);
      dot(belief, point.xy_m, (selected ? 4 : 3) * scale, selected ? "#a0f5e5" : "#f4f5f5");
      belief.font = `bold ${12 * scale}px system-ui`;
      belief.lineWidth = 3 * scale;
      belief.strokeStyle = "#243139";
      belief.fillStyle = selected ? "#a0f5e5" : "#fff";
      belief.strokeText(waypointName(point.waypoint_idx), left + 7 * scale, top - 5 * scale);
      belief.fillText(waypointName(point.waypoint_idx), left + 7 * scale, top - 5 * scale);
    }
  }

  byId("branch-choice").textContent = decided ? `本组采用 ${routeName(chosen.route_idx)} · ${waypointChain(chosen)} · `
    + `${branchData.routes.length} 条合法候选 / ${branchData.waypoint_reachable.length} 条中间点可达余程`
    : `${branchData.routes.length} 条合法候选 · 正在经过共同点测段`;
  for (const row of byId("branch-routes").rows) {
    const picked = decided && Number(row.dataset.route) === chosen.route_idx;
    row.dataset.picked = String(picked);
    row.dataset.preview = String(branchLayer.checked && row.dataset.route === branchFocus.value);
    row.cells[5].textContent = picked ? "采用" : "\u2014";
  }
}

function setupBranchReview() {
  for (const route of branchData.routes) {
    const choice = document.createElement("option");
    choice.value = route.route_idx;
    choice.textContent = `${routeName(route.route_idx)} · ${waypointChain(route)}`;
    branchFocus.append(choice);

    const row = byId("branch-routes").insertRow();
    row.dataset.route = route.route_idx;
    for (const value of [routeName(route.route_idx), waypointChain(route), fmt(route.travel_s),
                         fmt(route.unknown_s), fmt(route.risk_proxy_tq), ""]) {
      row.insertCell().textContent = value;
    }
  }
  for (const point of branchData.waypoints) {
    const term = document.createElement("dt");
    const result = document.createElement("dd");
    term.textContent = `${waypointName(point.waypoint_idx)} (${point.xy_m.join(", ")})`;
    result.textContent = `${point.route_indices.map(routeName).join("、")} \u2192 ${routeName(point.completed_route_idx)}`;
    byId("branch-waypoints").append(term, result);
  }
  branchLayer.addEventListener("change", draw);
  branchFocus.addEventListener("change", draw);
  window.addEventListener("resize", draw);
}
