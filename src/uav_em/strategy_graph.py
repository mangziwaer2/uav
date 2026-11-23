"""Paid finite corridor graph and model-conditioned conditional policies.

This library uses declared hypotheses only; no environment or sensor is called.
"""
from copy import deepcopy
import heapq
import itertools
import math

from .conditional_policy import (BudgetExhausted, PolicyError, _digest, _edge_free,
    _integrate, _keys, _log_power, _number, _point, _query_count, make_snapshot)


LIMITS = dict(geometry_tests=200000, route_expansions=200000,
    route_relaxations=2000000, power_queries=2000000, base_receiver_terms=8000000,
    likelihood_terms=100000, search_states=100000, action_evaluations=1000000,
    belief_updates=1000000, policy_evaluations=10000)


class StrategyBudget:
    def __init__(self, **limits):
        if set(limits)-set(LIMITS):
            raise PolicyError('unknown strategy budget counter')
        self.limits = dict(LIMITS, **limits)
        if any(type(v) is not int or v < 0 for v in self.limits.values()):
            raise PolicyError('nonnegative integer strategy budgets required')
        self._spent = dict.fromkeys(self.limits, 0)

    def summary(self):
        return dict(self._spent)

    def reserve(self, **counts):
        if set(counts)-set(self.limits) or any(type(v) is not int or v < 0 for v in counts.values()):
            raise PolicyError('invalid strategy work reservation')
        if any(self._spent[k]+v > self.limits[k] for k, v in counts.items()):
            raise BudgetExhausted('complete strategy operation exceeds remaining budget', work=counts)
        for k, v in counts.items():
            self._spent[k] += v


FIELDS = {'version', 'now_seconds', 'start_m', 'goal_m', 'speed_mps',
    'sensor_interval_seconds', 'measurement_noise_db', 'bounds_m', 'obstacles_m',
    'spent_length_m', 'initial_distance_m', 'remaining_energy_wh', 'reserve_energy_wh',
    'flight_power_w', 'history', 'worlds', 'depth', 'heading_degrees', 'corner_offset_m'}


def _task(body):
    _keys(body, FIELDS, 'corridor task'); _digest(body)
    if body['version'] != 'corridor-task-v1':
        raise PolicyError('unsupported corridor task')
    initial = _number(body['initial_distance_m'], positive=True)
    maximum = _number(1.25*initial, positive=True)
    if type(body['depth']) is not int or not 0 <= body['depth'] <= 3:
        raise PolicyError('zero to three normal observation layers required')
    angles = body['heading_degrees']
    if (type(angles) is not list or len(angles) not in (1, 3)
            or any(not -40 <= _number(x) <= 40 for x in angles)
            or angles != sorted(set(angles)) or 0. not in angles
            or len(angles) == 3 and angles[0] != -angles[2]):
        raise PolicyError('symmetric absolute headings [-theta,0,theta] or [0] required')
    _number(body['corner_offset_m'], positive=True)
    # Reuse the strict established common-data validator with an unused dummy
    # catalogue. No geometry or prediction is evaluated by make_snapshot.
    common = {k:deepcopy(v) for k, v in body.items() if k not in
        ('initial_distance_m', 'depth', 'heading_degrees', 'corner_offset_m')}
    common.update(version='conditional-task-v1', maximum_length_m=maximum)
    start = _point(body['start_m']); goal = _point(body['goal_m'])
    step = _number(body['speed_mps'], positive=True)*_number(body['sensor_interval_seconds'], positive=True)
    checkpoint = [start[0]+step, start[1]]
    if checkpoint == goal:
        checkpoint = [start[0]-step, start[1]]
    common.update(prefixes=[dict(id='validation', path_m=[start, checkpoint])],
        continuations=[dict(id='validation', prefix_id='validation', path_m=[checkpoint, goal])],
        thresholds_dbm=[])
    make_snapshot(common)
    if len(body['worlds']) > 4 or len(body['obstacles_m']) > 3 or goal[0] <= start[0]:
        raise PolicyError('finite domain: at most four worlds/three buildings and goal to the right')
    if body['spent_length_m'] == 0 and not math.isclose(initial, math.dist(start, goal), rel_tol=1e-12, abs_tol=1e-9):
        raise PolicyError('initial D0 must bind initial start-goal distance')
    result = deepcopy(body)
    result['maximum_length_m'] = maximum
    return result


def _free(a, b, task, budget):
    budget.reserve(geometry_tests=1)
    return _edge_free(a, b, task)


def _mode_edge(a, b, boxes, mode):
    for box, sign in zip(boxes, mode):
        middle = (box[0]+box[2])/2.
        if min(a[0], b[0]) <= middle <= max(a[0], b[0]):
            if a[0] == b[0]:
                ys = (a[1], b[1])
            else:
                ys = (a[1]+(middle-a[0])/(b[0]-a[0])*(b[1]-a[1]),)
            if sign == 1 and min(ys) <= box[3] or sign == -1 and max(ys) >= box[1]:
                return False
    return True


def _simplify(path):
    result = []
    for p in path:
        result.append(p)
        while len(result) >= 3:
            a, b, c = result[-3:]
            cross = (b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])
            dot = sum((b[k]-a[k])*(c[k]-b[k]) for k in (0, 1))
            if cross != 0 or dot < 0:
                break
            result.pop(-2)
    return result


def _corridors(task, start, budget):
    vertices = [start, task['goal_m']]
    pad = task['corner_offset_m']
    for x0, y0, x1, y1 in task['obstacles_m']:
        for p in ([x0-pad, y0-pad], [x0-pad, y1+pad], [x1+pad, y0-pad], [x1+pad, y1+pad]):
            if p not in vertices and _free(p, p, task, budget):
                vertices.append(p)
    adjacency = [[] for _ in vertices]
    for i, a in enumerate(vertices):
        for j in range(i+1, len(vertices)):
            b = vertices[j]
            if _free(a, b, task, budget):
                length = math.dist(a, b)
                adjacency[i].append((j, length)); adjacency[j].append((i, length))
    routes = {}
    for mode in itertools.product((-1, 1), repeat=len(task['obstacles_m'])):
        queue = [(0., (0,))]; best = {0:(0., (0,))}
        while queue:
            budget.reserve(route_expansions=1)
            distance, chain = heapq.heappop(queue)
            i = chain[-1]
            if (distance, chain) != best.get(i):
                continue
            if i == 1:
                path = _simplify([deepcopy(vertices[k]) for k in chain])
                signature = _digest(path)
                fresh = signature not in routes
                route = routes.setdefault(signature, dict(path_m=path, modes=[]))
                route['modes'].append(list(mode))
                # Publish each finished mode immediately. A later mode can
                # exhaust its geometry budget without erasing this route.
                if fresh:
                    yield route
                break
            for j, length in adjacency[i]:
                budget.reserve(route_relaxations=1)
                if j in chain or not _mode_edge(vertices[i], vertices[j], task['obstacles_m'], mode):
                    continue
                candidate = distance+length, chain+(j,)
                if candidate < best.get(j, (math.inf, ())):
                    best[j] = candidate; heapq.heappush(queue, candidate)


def _resource_ok(task, depth, length):
    future = depth*task['speed_mps']*task['sensor_interval_seconds']+length
    return (task['spent_length_m']+future <= task['maximum_length_m']+1e-8
        and task['remaining_energy_wh']-future/task['speed_mps']*task['flight_power_w']/3600.
            >= task['reserve_energy_wh']-1e-10)


def _prediction_cut(task, node, budget):
    worlds = task['worlds']; count = len(worlds)
    budget.reserve(power_queries=count, base_receiver_terms=sum(len(w['bases']) for w in worlds))
    means = [-80.+10.*_log_power(w, node['position_m'], node['time_seconds'])/math.log(10.) for w in worlds]
    distinct = sorted(set(means))
    cuts = [(a+b)/2. for a, b in zip(distinct, distinct[1:])] or distinct
    budget.reserve(likelihood_terms=2*count*len(cuts))
    rows = []
    for threshold in cuts:
        z = [(threshold-m)/task['measurement_noise_db'] for m in means]
        rows.append(dict(threshold_dbm=threshold,
            below_by_world=[math.erfc(-v/math.sqrt(2.))/2. for v in z],
            above_by_world=[math.erfc(v/math.sqrt(2.))/2. for v in z]))
    node.update(signal_means_dbm=means, cuts=rows)


def _edge_prediction(task, node, edge, budget):
    receivers = _query_count(edge['path_m'], task['speed_mps'])
    budget.reserve(power_queries=receivers*len(task['worlds']),
        base_receiver_terms=receivers*sum(len(w['bases']) for w in task['worlds']))
    edge['q_by_world'] = [_integrate(w, edge['path_m'], node['time_seconds'], task['speed_mps']) for w in task['worlds']]


def _seal(value):
    result = deepcopy(value)
    result['prepared_id'] = _digest(value)
    return result


def prepare_task(body, budget):
    """Compile known geometry, then cache complete paid hypothetical costs.

    Exhaustion retains complete legal terminal routes; partial work is explicit.
    """
    if not isinstance(budget, StrategyBudget):
        raise PolicyError('shared StrategyBudget required')
    task = _task(body); before = budget.summary()
    nodes, edges = {}, {}
    root_key = (0, 0, 0)
    start = task['start_m']; goal = task['goal_m']
    axis = math.atan2(goal[1]-start[1], goal[0]-start[0])
    theta = math.radians(max(task['heading_degrees']))
    step = task['speed_mps']*task['sensor_interval_seconds']
    stopped = None

    def add_node(key):
        depth, straights, net = key
        along = step*(straights+(depth-straights)*math.cos(theta))
        across = step*net*math.sin(theta)
        position = [start[0]+along*math.cos(axis)-across*math.sin(axis),
                    start[1]+along*math.sin(axis)+across*math.cos(axis)]
        identifier = 'n_'+ '_'.join(map(str, key))
        if identifier not in nodes:
            nodes[identifier] = dict(id=identifier, depth=depth, position_m=position,
                time_seconds=task['now_seconds']+depth*task['sensor_interval_seconds'],
                spent_length_m=task['spent_length_m']+depth*step,
                terminal_edges=[], step_edges=[], cuts=[], signal_means_dbm=None)
        return identifier

    root = add_node(root_key)
    keys = [root_key]
    try:
        for key in keys:
            identifier = add_node(key); node = nodes[identifier]
            for route in _corridors(task, node['position_m'], budget):
                path = route['path_m']
                length = math.fsum(math.dist(a, b) for a, b in zip(path, path[1:]))
                if not _resource_ok(task, key[0], length):
                    continue
                edge_id = identifier+'_t_'+_digest(path)[:16]
                edges[edge_id] = dict(id=edge_id, from_id=identifier, to_id=None,
                    path_m=path, modes=route['modes'], length_m=length,
                    duration_seconds=length/task['speed_mps'])
                node['terminal_edges'].append(edge_id)
            if key[0] >= task['depth']:
                continue
            for angle in task['heading_degrees']:
                next_key = (key[0]+1, key[1]+int(angle == 0), key[2]+(1 if angle > 0 else -1 if angle < 0 else 0))
                next_id = add_node(next_key); target = nodes[next_id]
                path = [node['position_m'], target['position_m']]
                if (not _free(*path, task, budget)
                        or not _resource_ok(task, key[0]+1, math.dist(target['position_m'], goal))):
                    if next_key not in keys:
                        nodes.pop(next_id, None)
                    continue
                edge_id = identifier+'_s_'+next_id
                edges[edge_id] = dict(id=edge_id, from_id=identifier, to_id=next_id,
                    path_m=deepcopy(path), modes=None, length_m=step,
                    duration_seconds=task['sensor_interval_seconds'])
                node['step_edges'].append(edge_id)
                if next_key not in keys:
                    keys.append(next_key)
    except BudgetExhausted as exc:
        stopped = dict(stage='geometry', error=str(exc))
    prediction_failures = []
    try:
        # Root terminals are evaluated before any exploratory cache entry.
        for edge in edges.values():
            try:
                _edge_prediction(task, nodes[edge['from_id']], edge, budget)
            except (ValueError, OverflowError) as exc:
                prediction_failures.append(dict(edge_id=edge['id'], error=str(exc)))
        for identifier, node in nodes.items():
            if identifier != root:
                try:
                    _prediction_cut(task, node, budget)
                except (ValueError, OverflowError) as exc:
                    prediction_failures.append(dict(node_id=identifier, error=str(exc)))
    except BudgetExhausted as exc:
        stopped = dict(stage='prediction', error=str(exc))
    # Drop incomplete entries and steps whose whole target has no legal future.
    usable = {i:e for i, e in edges.items() if 'q_by_world' in e}
    for node in sorted(nodes.values(), key=lambda n:-n['depth']):
        node['terminal_edges'] = [i for i in node['terminal_edges'] if i in usable]
        retained = []
        for i in node['step_edges']:
            edge = usable.get(i)
            target = nodes.get(edge['to_id']) if edge else None
            if target and target['cuts'] and (target['terminal_edges'] or target['step_edges']):
                retained.append(i)
            else:
                usable.pop(i, None)
        node['step_edges'] = retained
    work = {k:v-before[k] for k, v in budget.summary().items()}
    return _seal(dict(version='corridor-prepared-v1', task=task, task_id=_digest(task),
        root_id=root, nodes=nodes, edges=usable, preparation_complete=stopped is None and not prediction_failures,
        stopped_details=stopped, prediction_failures=prediction_failures, work=work,
        scope='finite declared geometry and hypothetical worlds only; no sensing or truth'))


def validate_prepared(prepared):
    if type(prepared) is not dict or 'prepared_id' not in prepared:
        raise PolicyError('bound prepared graph required')
    body = {k:v for k, v in prepared.items() if k != 'prepared_id'}
    if _digest(body) != prepared['prepared_id'] or body.get('version') != 'corridor-prepared-v1':
        raise PolicyError('prepared graph changed after binding')
    if _digest(body['task']) != body['task_id']:
        raise PolicyError('prepared task binding changed')
    return deepcopy(prepared)


def _compile(prepared, policy):
    _keys(policy, {'version', 'task_id', 'root'}, 'corridor policy'); _digest(policy)
    if policy['version'] != 'corridor-policy-v1' or policy['task_id'] != prepared['task_id']:
        raise PolicyError('policy task binding changed')
    decisions = []
    def walk(decision, node_id):
        _keys(decision, {'edge_id', 'threshold_dbm', 'below', 'above'}, 'strategy decision')
        node = prepared['nodes'][node_id]
        identifier = decision['edge_id']
        if type(identifier) is not str or identifier not in node['terminal_edges']+node['step_edges']:
            raise PolicyError('selected edge is not a complete legal cached action')
        edge = prepared['edges'][identifier]
        decisions.append((decision, edge))
        if edge['to_id'] is None:
            if any(decision[k] is not None for k in ('threshold_dbm', 'below', 'above')):
                raise PolicyError('terminal edge cannot contain future branches')
        else:
            _number(decision['threshold_dbm'])
            target = prepared['nodes'][edge['to_id']]
            if decision['threshold_dbm'] not in [c['threshold_dbm'] for c in target['cuts']]:
                raise PolicyError('threshold outside shared predicted cuts')
            walk(decision['below'], edge['to_id']); walk(decision['above'], edge['to_id'])
    walk(policy['root'], prepared['root_id'])
    return decisions


def _evaluate(prepared, policy, budget):
    decisions = _compile(prepared, policy)
    steps = sum(e['to_id'] is not None for _, e in decisions)
    charge = dict(policy_evaluations=1, action_evaluations=len(decisions), belief_updates=2*steps)
    budget.reserve(**charge)
    worlds = prepared['task']['worlds']; weights = [w['weight'] for w in worlds]
    branches = []
    def walk(decision, belief, depth):
        edge = prepared['edges'][decision['edge_id']]
        if edge['to_id'] is None:
            return (list(edge['q_by_world']), [edge['duration_seconds']]*len(worlds),
                    [edge['length_m']]*len(worlds), edge['length_m'])
        target = prepared['nodes'][edge['to_id']]
        cut = next(c for c in target['cuts'] if c['threshold_dbm'] == decision['threshold_dbm'])
        children = []
        for label in ('below', 'above'):
            probabilities = cut[label+'_by_world']
            masses = [w*p for w, p in zip(belief, probabilities)]; total = math.fsum(masses)
            posterior = [m/total for m in masses] if total else None
            branches.append(dict(depth=depth+1, node_id=target['id'], label=label,
                threshold_dbm=decision['threshold_dbm'], conditional_probability=total,
                posterior_weights=posterior))
            children.append(walk(decision[label], posterior if posterior is not None else belief, depth+1))
        low, high = cut['below_by_world'], cut['above_by_world']
        def combine(immediate, index):
            return [math.fsum((x, p*a, r*b)) for x, p, a, r, b in
                zip(immediate, low, children[0][index], high, children[1][index])]
        return (combine(edge['q_by_world'], 0), combine([edge['duration_seconds']]*len(worlds), 1),
                combine([edge['length_m']]*len(worlds), 2), edge['length_m']+max(c[3] for c in children))
    q, flight, length, worst = walk(policy['root'], weights, 0)
    average = lambda values:math.fsum(w*x for w, x in zip(weights, values))
    task = prepared['task']
    result = dict(policy=deepcopy(policy), policy_id=_digest(policy), task_id=prepared['task_id'],
        expected_score_seconds=average(q)+average(flight), expected_q_seconds=average(q),
        expected_flight_seconds=average(flight), expected_length_m=average(length), worst_length_m=worst,
        minimum_remaining_energy_wh=task['remaining_energy_wh']-worst/task['speed_mps']*task['flight_power_w']/3600.,
        worlds=[dict(id=w['id'], weight=w['weight'], q_seconds=a, flight_seconds=b,
                     length_m=c, score_seconds=a+b) for w, a, b, c in zip(worlds, q, flight, length)],
        branches=branches, work=charge,
        scope='bin-conditioned finite hypothetical model expectation; no new evidence or flight')
    if task['spent_length_m']+worst > task['maximum_length_m']+1e-8 or result['minimum_remaining_energy_wh'] < task['reserve_energy_wh']-1e-10:
        raise PolicyError('complete strategy exceeds cumulative resources', work=charge)
    _digest(result)
    return result


def evaluate_strategy(prepared, policy, budget):
    return _evaluate(validate_prepared(prepared), deepcopy(policy), budget)


def assess_strategy(prepared, policy, worlds, measurement_noise_db, budget):
    """Offline artificial reassessment after freezing a policy, without mutation."""
    p = validate_prepared(prepared); policy = deepcopy(policy)
    decisions = _compile(p, policy)
    body = {k:deepcopy(v) for k, v in p['task'].items() if k != 'maximum_length_m'}
    body.update(worlds=deepcopy(worlds), measurement_noise_db=measurement_noise_db)
    task = _task(body)
    before = budget.summary()
    # Only selected edges/checkpoints are recomputed, exactly once in this cache.
    selected = {e['id']:deepcopy(e) for _, e in decisions}
    touched = {e['to_id'] for e in selected.values() if e['to_id'] is not None}
    for edge in selected.values():
        _edge_prediction(task, p['nodes'][edge['from_id']], edge, budget)
    for identifier in touched:
        node = p['nodes'][identifier]
        cuts = sorted({d['threshold_dbm'] for d, e in decisions if e['to_id'] == identifier})
        budget.reserve(power_queries=len(worlds), base_receiver_terms=sum(len(w['bases']) for w in worlds),
                       likelihood_terms=2*len(worlds)*len(cuts))
        means = [-80.+10.*_log_power(w, node['position_m'], node['time_seconds'])/math.log(10.) for w in worlds]
        node['cuts'] = []
        for cut in cuts:
            z = [(cut-m)/measurement_noise_db for m in means]
            node['cuts'].append(dict(threshold_dbm=cut,
                below_by_world=[math.erfc(-v/math.sqrt(2.))/2. for v in z],
                above_by_world=[math.erfc(v/math.sqrt(2.))/2. for v in z]))
    p['edges'].update(selected)
    p['task']['worlds'] = task['worlds']; p['task']['measurement_noise_db'] = measurement_noise_db
    # task_id deliberately remains the frozen planner binding for _compile;
    # this internal copy is never published as a valid prepared planning cache.
    result = _evaluate(p, policy, budget)
    result.update(assessment_model_id=_digest(dict(worlds=worlds, measurement_noise_db=measurement_noise_db)),
        original_policy_id=_digest(policy),
        work={k:v-before[k] for k, v in budget.summary().items()},
        scope='offline declared artificial alternative model only; not simulator truth; original inputs unchanged')
    return result
