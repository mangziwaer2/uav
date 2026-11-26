"""Attribute repair outcomes after closure, without sensing or repricing any route."""

from itertools import chain, permutations, product
import math
from types import SimpleNamespace

from .motion import advance_route
from .passive_closed_loop_pilot import _executed_samples
from .passive_event_choice import HORIZON_S
from .passive_local_repair import check_remaining_route, xy
from .passive_map import LIMIT_S, _common_time
from .passive_model_advice_publish import reconstruct
from .passive_repair_actions import ACTION_COLUMNS, SOURCE_COLUMNS, action_table_digest
from .passive_repair_geometry import FIXED_DOMAIN, ONLINE_DOMAIN, OnlineRepairGeometry, repair_domain
from .passive_repair_search import build_repair_neighborhood, normalize_route


EPS = 1e-8


def _close(first, second):
    return math.isclose(first, second, rel_tol=0, abs_tol=EPS)


def _score(segments, duration, truth):
    exposure = math.fsum(float(weight) * 10 ** ((float(truth(point, clock)) + 80) / 10)
                         for point, clock, weight in _executed_samples(segments))
    if not math.isfinite(exposure):
        raise ValueError("nonfinite true exposure")
    return dict(T=duration, Q=exposure, T_plus_Q=duration + exposure)


def _remaining_score(route, now, speed, truth):
    duration = 0.0
    segments = []
    for first, second in zip(route, route[1:]):
        step = math.dist(first, second) / speed
        segments.append((first, second, now + duration, step))
        duration += step
    return _score(segments, duration, truth)


def _reference_routes(state, goal):
    state = dict(state)
    start = xy(state.pop('position'))
    geometry = state['geometry']
    limit = 1.25 * math.dist(geometry.config.start_m, goal) - state['flown_m']
    grid = product(*(range(0, math.floor(bound) + 1, 40) for bound in geometry.high))
    points = [point for point in grid if point not in (start, goal)
              and math.dist(start, point) + math.dist(point, goal) <= limit + EPS and geometry.free(point)]
    for vias in chain(((point,) for point in points), permutations(points, 2)):
        route = (start, *vias, goal)
        if check_remaining_route(route, **state).status == 'legal':
            yield route


def _action_rows(event, neighborhood, incumbent_id):
    packet = event['action_inputs']
    if packet is None:
        if event['action_table_sha256'] is not None or event['action_count'] or event['action_id'] is not None:
            raise ValueError("incomplete frozen table has action evidence")
        return None
    if action_table_digest(packet) != event['action_table_sha256']:
        raise ValueError("frozen action table hash mismatch")
    table = packet['action_table']
    rows = table['rows']
    intervals = tuple(map(tuple, packet['locked_intervals_s']))
    originals = tuple(row.route for row in neighborhood.originals)
    if (table['version'] != 1 or tuple(table['columns']) != ACTION_COLUMNS
            or tuple(table['source_columns']) != SOURCE_COLUMNS
            or tuple((index, normalize_route(route)) for index, route in packet['routes']) != tuple(enumerate(originals))
            or tuple(map(xy, packet['repair_waypoints_xy_m'])) != neighborhood.waypoints
            or len(rows) != event['action_count'] or len(rows) != len(originals) + len(neighborhood.expanded)):
        raise ValueError("frozen table does not match the complete legal domain")
    if (intervals != tuple(map(tuple, event['locked_intervals'])) or table['time_only'] != (not intervals)
            or any(not 0 <= left < right <= HORIZON_S for left, right in intervals)
            or any(first[1] > second[0] for first, second in zip(intervals, intervals[1:]))):
        raise ValueError("invalid frozen comparison intervals")
    domain = (*zip(neighborhood.originals, neighborhood.original_aliases), *neighborhood.expanded)
    for index, (row, (resolved, aliases)) in enumerate(zip(rows, domain)):
        if len(row) != len(ACTION_COLUMNS) or row[0] != index or not _close(row[1], resolved.time_s):
            raise ValueError("invalid frozen action ID or time")
        expected = {base: {(edit.left, edit.right,
                            None if edit.waypoint_xy_m is None else neighborhood.waypoints.index(edit.waypoint_xy_m))
                           for edit in edits} for base, edits in aliases.items()}
        baseline = 0
        actual = {}
        for source in row[-1]:
            base, edits = source[:2]
            if len(source) != len(SOURCE_COLUMNS) or not 0 <= base < len(originals):
                raise ValueError("invalid frozen action source")
            if not edits:
                if index >= len(originals) or base != incumbent_id:
                    raise ValueError("invalid frozen original reference")
                baseline += 1
                continue
            if base in actual or len(edits) != len(set(map(tuple, edits))):
                raise ValueError("duplicate frozen action source")
            actual[base] = set(map(tuple, edits))
        if actual != expected or baseline != int(index < len(originals)):
            raise ValueError("frozen action sources do not reconstruct the legal domain")
    prices = tuple(dict(zip(ACTION_COLUMNS, row)) for row in rows)
    for row in prices:
        priced = row['status'] in ('baseline', 'comparable')
        if any((value is not None and math.isfinite(value) and value >= 0) != priced
               or (not priced and value is not None) for value in (row['expected_q'], row['tail_q'])):
            raise ValueError("inconsistent frozen Q evidence")
    return prices


def _is_lower(candidate, reference):
    return candidate['Q'] <= 0.8 * reference['Q'] and candidate['T_plus_Q'] <= reference['T_plus_Q'] - 1


def _summarize(routes, scores, incumbent):
    qualified = [route for route in routes if _is_lower(scores[route], incumbent)]
    best = min(qualified, key=lambda route: (scores[route]['T_plus_Q'], route)) if qualified else None
    return dict(legal_count=len(routes), low_exposure_count=len(qualified),
                best_low_exposure=None if best is None else dict(route=best, **scores[best]))


def _proxy_evidence(candidate, selected, routes, rows, intervals, speed, min_post_s):
    alternative, choice = rows[candidate], rows[selected]
    split = _common_time(routes[candidate], routes[selected], speed)
    end = max(alternative['time_s'], choice['time_s'])
    post = sum(max(0., min(right, end) - left) for left, right in intervals if left >= split - EPS)
    keys = ('expected_q', 'tail_q')
    if post + EPS < min_post_s or any(row[key] is None for row in (alternative, choice) for key in keys):
        return dict(action_id=candidate, status='missing_comparable_evidence', post_common_s=None)
    delta_t = alternative['time_s'] - choice['time_s']
    expected, tail = (delta_t + alternative[key] - choice[key] for key in keys)
    status = ('both_favor_alternative' if max(expected, tail) < -EPS else
              'both_favor_selection' if min(expected, tail) > EPS else 'no_proxy_consensus')
    return dict(action_id=candidate, status=status, post_common_s=post, delta_expected=expected, delta_tail=tail)


def _proxy_first(rows, risk):
    def rank(row):
        return round(row['time_s'] + row[risk], 8), row['time_s'], row['action_id']

    best = min((row for row in rows if row[risk] is not None), key=rank, default=None)
    return best['action_id'] if best is not None else None


def _audit_frame(frame, event, kind, geometry, aircraft, online, waypoints, truth, min_post_s):
    position, goal = list(frame['xy']), list(geometry.config.goal_m)
    before = frame['incumbent_before']
    incumbent = normalize_route(before or frame['remaining'])
    cap = 1.25 * math.dist(geometry.config.start_m, goal)
    public_state = SimpleNamespace(position=position, goal=goal, incumbent=before,
                                   geometry=geometry, aircraft=aircraft, now=frame['t'], flown=frame['flown_m'],
                                   energy=frame['energy_left_wh'], cap=cap, speed=aircraft.speed_mps)
    state = dict(geometry=geometry, aircraft=aircraft, position=position, now=frame['t'],
                 flown_m=frame['flown_m'], energy_wh=frame['energy_left_wh'])
    original, guides = repair_domain(public_state, online, waypoints)
    neighborhood = build_repair_neighborhood(dict.fromkeys(map(normalize_route, original)), guides, **state)
    originals = tuple(row.route for row in neighborhood.originals)
    actions = (*originals, *(row.route for row, _ in neighborhood.expanded))
    selected = event['action_id'] if kind == 'review' else None
    pipeline = None
    if kind == 'review':
        if not incumbent or event['incumbent'] != originals.index(incumbent):
            raise ValueError("review incumbent disagrees with the unflown commitment")
        rows = _action_rows(event, neighborhood, event['incumbent'])
        if selected is None:
            if event['selected'] is not None or frame['advanced_m'] > EPS:
                raise ValueError("unexpected execution without a selected action")
            pipeline = event['status']
        elif (rows is None or not 0 <= selected < len(rows) or event['selected'] is None
              or normalize_route(event['candidates'][event['selected']]['route']) != actions[selected]
              or normalize_route(frame['remaining']) != actions[selected]):
            raise ValueError("executed commitment does not match the selected frozen action")
        elif frame['advanced_m'] <= EPS:
            pipeline = 'committed_without_advance'
    else:
        rows = None
    report = dict(t=frame['t'], kind=kind, prefix_size=frame['prefix_size'], gp_version=frame.get('active_model'),
                  trigger=(frame.get('alert') or {}).get('state'), selected_action_id=selected,
                  advanced_m=frame['advanced_m'], pipeline_fault=pipeline,
                  action_table_sha256=event.get('action_table_sha256') if event else None,
                  first_by_proxy=None if rows is None else dict(expected=_proxy_first(rows, 'expected_q'),
                                                               tail=_proxy_first(rows, 'tail_q')))
    if not incumbent:
        return dict(report, availability='no_legal_commitment', decision_evidence='not_available')
    reference = tuple(dict.fromkeys((*actions, *_reference_routes(state, xy(goal)))))
    scores = {route: _remaining_score(route, frame['t'], aircraft.speed_mps, truth) for route in reference}
    baseline = scores[incumbent]
    scopes = {name: _summarize(routes, scores, baseline) for name, routes in
              (('original', originals), ('shared', actions), ('reference', reference))}
    availability = next((name for name, summary in scopes.items() if summary['low_exposure_count']), 'not_found')
    comparisons = []
    if pipeline:
        evidence = 'pipeline_fault'
    elif kind != 'review':
        evidence = 'not_triggered' if availability in ('original', 'shared') else 'not_available'
    else:
        chosen = scores[actions[selected]]
        intervals = event['action_inputs']['locked_intervals_s']
        comparisons = [_proxy_evidence(index, selected, actions, rows, intervals, aircraft.speed_mps, min_post_s)
                       for index, route in enumerate(actions) if index != selected and _is_lower(scores[route], baseline)
                       and scores[route]['T_plus_Q'] <= chosen['T_plus_Q'] - 1]
        for comparison in comparisons:
            comparison['counterfactual'] = scores[actions[comparison['action_id']]]
        evidence = ('both_proxies_favored_alternative' if any(row['status'] == 'both_favor_alternative'
                    for row in comparisons) else 'missing_or_different_online_order' if comparisons
                    else 'no_better_low_exposure_action_than_selection')
    return dict(report, availability=availability, decision_evidence=evidence, domains=scopes,
                incumbent=dict(route=incumbent, **baseline), comparisons=comparisons,
                selected_counterfactual=None if selected is None else scores[actions[selected]])


def audit_closed_repair(log, *, geometry, aircraft, truth, min_post_s, waypoints=None):
    """Return actual performance and separate counterfactuals at the predeclared frames."""
    if not log.get('stop') or not log.get('frames') or not 0 < min_post_s <= HORIZON_S:
        raise ValueError("closed repair log and valid support threshold required")
    domain = log.get('route_domain', FIXED_DOMAIN)
    if domain not in (FIXED_DOMAIN, ONLINE_DOMAIN) or (domain == ONLINE_DOMAIN) != (waypoints is None):
        raise ValueError("audit inputs do not match the logged route domain")
    online = OnlineRepairGeometry(geometry, aircraft) if domain == ONLINE_DOMAIN else None
    frames, events = log['frames'], log['decisions']
    start, goal = map(xy, (geometry.config.start_m, geometry.config.goal_m))
    cap, speed = 1.25 * math.dist(start, goal), aircraft.speed_mps
    if xy(frames[0]['xy']) != start or not _close(frames[0]['t'], 0) or not _close(log['cap_m'], cap):
        raise ValueError("flight starts outside its recorded public state")
    segments = reconstruct(frames, speed)
    for index, frame in enumerate(frames):
        elapsed = frame['t']
        if (not _close(frame['flown_m'], elapsed * speed)
                or not _close(frame['energy_left_wh'], aircraft.initial_energy_wh - elapsed * aircraft.flight_power_w / 3600)):
            raise ValueError("checkpoint resources do not match the executed trace")
        actual = 0. if index + 1 == len(frames) else frames[index + 1]['flown_m'] - frame['flown_m']
        if actual < 0 or not _close(actual, frame['advanced_m']):
            raise ValueError("checkpoint advancement does not match the executed trace")
        if actual > EPS:
            _, moved = advance_route(frame['remaining'], actual)
            recorded = frames[index + 1]['moved']
            if len(moved) != len(recorded) or any(math.dist(first, second) > EPS for first, second in zip(moved, recorded)):
                raise ValueError("actual advancement diverges from the committed route")
    last = frames[-1]
    if not _close(last['energy_left_wh'], log['energy_left_wh']):
        raise ValueError("terminal energy disagrees with the executed trace")
    ended_at_goal = xy(last['xy']) == goal
    if (log['stop'] == 'arrived') != ended_at_goal:
        raise ValueError("terminal position and arrival status disagree")
    legal = (last['flown_m'] <= cap + EPS and last['t'] <= LIMIT_S + EPS
             and last['energy_left_wh'] >= aircraft.reserve_energy_wh - EPS
             and all(geometry.edge_free(first, second) for first, second, _, _ in segments))
    metrics = dict(_score(segments, last['t'], truth), eligible_for_pairing=ended_at_goal and legal,
                   legal_prefix=legal, stop=log['stop'], length_m=last['flown_m'])
    checks, linked = [], []
    high_before = saw_high = audited_persistent = False
    for index, frame in enumerate(frames):
        alert = frame.get('alert') or {}
        high = bool(alert.get('active'))
        event = None
        if 'decision' in frame:
            if frame['decision'] != len(linked) or len(linked) >= len(events):
                raise ValueError("invalid frame-to-decision link")
            event = events[frame['decision']]
            if event['t'] != frame['t'] or not _close(event['advanced_m'], frame['advanced_m']):
                raise ValueError("frame and decision clocks or advancement disagree")
            linked.append(event)
        if event is not None and event['authority'] != 'initial':
            if not high or event['authority'] != log['mode']:
                raise ValueError("repair decision does not match the recorded trigger or arm")
            checks.append((index, event, 'review'))
        elif high and high_before and not audited_persistent and alert['state'] in ('already_reviewed', 'no_alternative'):
            checks.append((index, None, 'persistent_alert_without_new_branch'))
            audited_persistent = True
        high_before, saw_high = high, saw_high or high
    if len(linked) != len(events):
        raise ValueError("unlinked decisions in closed repair log")
    if not saw_high:
        checks.insert(0, (0, None, 'no_alert_start'))
    return dict(mode=log['mode'], route_domain=domain, actual_prefix=metrics, total_frames=len(frames),
                audited_frame_indices=[index for index, _, _ in checks], unreviewed_frames=len(frames) - len(checks),
                checkpoints=[_audit_frame(frames[index], event, kind, geometry, aircraft, online, waypoints, truth, min_post_s)
                             for index, event, kind in checks])


def paired_repair_delta(numerical, llm):
    """No full-mission delta for a pair containing a stopped or illegal arm."""
    if (numerical['mode'], llm['mode']) != ('numerical', 'llm'):
        raise ValueError("expected one numerical and one LLM report")
    baseline, candidate = numerical['actual_prefix'], llm['actual_prefix']
    return (candidate['T_plus_Q'] - baseline['T_plus_Q']
            if candidate['eligible_for_pairing'] and baseline['eligible_for_pairing'] else None)
