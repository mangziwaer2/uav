"""One normal-observation conditional policy over a frozen finite belief.

No simulator, sensor, hidden configuration, fitting service or advisor is used.
All expectations are conditional on the explicitly supplied hypothetical worlds.
"""
from copy import deepcopy
import hashlib
import json
import math


TASK_VERSION = 'conditional-task-v1'
MAX_QUADRATURE_SECONDS = .5
_LN10 = math.log(10.)


class PolicyError(ValueError):
    def __init__(self, message, *, work=None):
        super().__init__(message)
        self.work = dict(work or {})


class BudgetExhausted(RuntimeError):
    def __init__(self, message, *, work=None):
        super().__init__(message)
        self.work = dict(work or {})


def _digest(value):
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()
    except (ValueError, TypeError, OverflowError) as exc:
        raise PolicyError('finite JSON value required') from exc
    return hashlib.sha256(raw).hexdigest()


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise PolicyError('exact '+label+' fields required')


def _number(value, *, positive=False, nonnegative=False):
    if type(value) not in (int, float):
        raise PolicyError('finite numeric value required; bool is not a number')
    try:
        valid = math.isfinite(value)
    except OverflowError as exc:
        raise PolicyError('finite representable numeric value required') from exc
    if not valid:
        raise PolicyError('finite numeric value required; bool is not a number')
    if positive and value <= 0 or nonnegative and value < 0:
        raise PolicyError('numeric value outside positive/nonnegative domain')
    return float(value)


def _point(value):
    if type(value) is not list or len(value) != 2:
        raise PolicyError('two dimensional JSON point required')
    return [_number(v) for v in value]


def _identifier(value):
    if type(value) is not str or not 1 <= len(value) <= 96:
        raise PolicyError('bounded nonempty identifier required')


def _bounded_list(value, low, high, label):
    if type(value) is not list or not low <= len(value) <= high:
        raise PolicyError('bounded '+label+' list required')


def _length(path):
    try:
        length = math.fsum(math.dist(a, b) for a, b in zip(path, path[1:]))
    except OverflowError as exc:
        raise PolicyError('finite complete path length required') from exc
    if not math.isfinite(length):
        raise PolicyError('finite path length required')
    return length


def _path(value):
    _bounded_list(value, 2, 16, 'path')
    for point in value:
        _point(point)
    if any(math.dist(a, b) <= 1e-10 for a, b in zip(value, value[1:])):
        raise PolicyError('positive length consecutive path segments required')
    _length(value)


_BODY_FIELDS = {'version', 'now_seconds', 'start_m', 'goal_m', 'speed_mps',
    'sensor_interval_seconds', 'measurement_noise_db', 'bounds_m', 'obstacles_m',
    'spent_length_m', 'maximum_length_m', 'remaining_energy_wh', 'reserve_energy_wh',
    'flight_power_w', 'history', 'worlds', 'prefixes', 'continuations', 'thresholds_dbm'}


def make_snapshot(body):
    """Validate a public catalogue without rejecting unused infeasible routes."""
    _keys(body, _BODY_FIELDS, 'task snapshot')
    _digest(body)
    if body['version'] != TASK_VERSION:
        raise PolicyError('unsupported task snapshot version')
    now = _number(body['now_seconds'], nonnegative=True)
    for name in ('speed_mps', 'sensor_interval_seconds', 'measurement_noise_db',
                 'maximum_length_m', 'flight_power_w'):
        _number(body[name], positive=True)
    for name in ('spent_length_m', 'remaining_energy_wh', 'reserve_energy_wh'):
        _number(body[name], nonnegative=True)
    _number(now+body['sensor_interval_seconds'], positive=True)
    if (body['remaining_energy_wh'] < body['reserve_energy_wh']
            or body['spent_length_m'] > body['maximum_length_m']):
        raise PolicyError('current resources already exceed hard limits')
    start, goal = _point(body['start_m']), _point(body['goal_m'])
    _bounded_list(body['bounds_m'], 2, 2, 'bounds')
    low, high = map(_point, body['bounds_m'])
    if any(a >= b for a, b in zip(low, high)):
        raise PolicyError('strict rectangular map bounds required')
    if any(not a <= p <= b for point in (start, goal)
           for a, p, b in zip(low, point, high)):
        raise PolicyError('current endpoint outside public map')
    _bounded_list(body['obstacles_m'], 0, 64, 'obstacles')
    for box in body['obstacles_m']:
        _bounded_list(box, 4, 4, 'rectangle')
        values = [_number(x) for x in box]
        if values[0] >= values[2] or values[1] >= values[3]:
            raise PolicyError('strict obstacle rectangle required')
    _bounded_list(body['history'], 1, 64, 'history')
    previous = -math.inf
    for row in body['history']:
        _keys(row, {'position_m', 'time_seconds', 'interference_dbm'}, 'history observation')
        _point(row['position_m']); _number(row['interference_dbm'])
        stamp = _number(row['time_seconds'], nonnegative=True)
        if not previous < stamp <= now:
            raise PolicyError('strict chronological observed prefix required')
        previous = stamp
    last = body['history'][-1]
    if last['time_seconds'] != now or last['position_m'] != body['start_m']:
        raise PolicyError('last observation must bind current time and position')
    _bounded_list(body['worlds'], 1, 8, 'worlds')
    world_ids = []
    for world in body['worlds']:
        _keys(world, {'id', 'weight', 'background_dbm', 'bases'}, 'hypothetical world')
        _identifier(world['id']); world_ids.append(world['id'])
        if _number(world['weight'], nonnegative=True) > 1:
            raise PolicyError('world weight cannot exceed one')
        _number(world['background_dbm'])
        _bounded_list(world['bases'], 0, 4, 'equivalent bases')
        for base in world['bases']:
            _keys(base, {'center_m', 'scale_m', 'peak_dbm', 'velocity_mps',
                        'rate_db_per_second', 'reference_time_seconds'}, 'equivalent base')
            _point(base['center_m']); _point(base['velocity_mps'])
            _number(base['scale_m'], positive=True); _number(base['peak_dbm'])
            _number(base['rate_db_per_second'])
            if _number(base['reference_time_seconds']) > now:
                raise PolicyError('hypothesis reference cannot be a future observation')
    if len(world_ids) != len(set(world_ids)) or not math.isclose(
            math.fsum(w['weight'] for w in body['worlds']), 1., rel_tol=0., abs_tol=1e-12):
        raise PolicyError('unique worlds with normalized nonnegative weights required')
    _bounded_list(body['prefixes'], 1, 8, 'mission prefixes')
    prefix_ids = []
    for prefix in body['prefixes']:
        _keys(prefix, {'id', 'path_m'}, 'prefix')
        _identifier(prefix['id']); prefix_ids.append(prefix['id']); _path(prefix['path_m'])
        duration = _length(prefix['path_m'])/body['speed_mps']
        if prefix['path_m'][0] != body['start_m'] or not math.isclose(
                duration, body['sensor_interval_seconds'], rel_tol=1e-10, abs_tol=1e-9):
            raise PolicyError('prefix must end at the next normal point measurement')
    if len(prefix_ids) != len(set(prefix_ids)):
        raise PolicyError('unique mission prefix IDs required')
    _bounded_list(body['continuations'], 1, 64, 'continuations')
    continuation_ids = []
    for route in body['continuations']:
        _keys(route, {'id', 'prefix_id', 'path_m'}, 'continuation')
        _identifier(route['id']); continuation_ids.append(route['id']); _path(route['path_m'])
        if route['prefix_id'] not in prefix_ids:
            raise PolicyError('continuation must bind a known prefix')
        prefix = next(p for p in body['prefixes'] if p['id'] == route['prefix_id'])
        if route['path_m'][0] != prefix['path_m'][-1] or route['path_m'][-1] != body['goal_m']:
            raise PolicyError('continuation endpoints must bind checkpoint and goal')
    if len(continuation_ids) != len(set(continuation_ids)):
        raise PolicyError('unique continuation IDs required')
    if any(not 1 <= sum(c['prefix_id'] == pid for c in body['continuations']) <= 8
           for pid in prefix_ids):
        raise PolicyError('each prefix requires one to eight continuations')
    _bounded_list(body['thresholds_dbm'], 0, 4, 'shared thresholds')
    cuts = [_number(x) for x in body['thresholds_dbm']]
    if any(a >= b for a, b in zip(cuts, cuts[1:])):
        raise PolicyError('strictly increasing unique shared thresholds required')
    result = deepcopy(body)
    result['snapshot_id'] = _digest(body)
    return result


def _edge_free(start, end, snapshot):
    low, high = snapshot['bounds_m']
    if any(not a <= p <= b for point in (start, end) for a, p, b in zip(low, point, high)):
        return False
    for box in snapshot['obstacles_m']:
        enter, leave = 0., 1.
        for axis in (0, 1):
            delta = end[axis]-start[axis]
            if delta == 0:
                if not box[axis] <= start[axis] <= box[axis+2]:
                    enter, leave = 1., 0.
                    break
            else:
                first = (box[axis]-start[axis])/delta
                second = (box[axis+2]-start[axis])/delta
                enter = max(enter, min(first, second)); leave = min(leave, max(first, second))
        if enter <= leave:
            return False
    return True


def _query_count(path, speed):
    count = 0
    for a, b in zip(path, path[1:]):
        duration = math.dist(a, b)/speed
        if not math.isfinite(duration/MAX_QUADRATURE_SECONDS):
            raise PolicyError('path quadrature requires a finite complete allowance')
        count += max(1, math.ceil(duration/MAX_QUADRATURE_SECONDS))
    return count


def _log_power(world, point, stamp):
    terms = [(world['background_dbm']+80.)*_LN10/10.]
    for base in world['bases']:
        elapsed = stamp-base['reference_time_seconds']
        center = [p+elapsed*v for p, v in zip(base['center_m'], base['velocity_mps'])]
        distance = math.dist(point, center)/base['scale_m']
        terms.append((base['peak_dbm']+elapsed*base['rate_db_per_second']+80.)*_LN10/10.
                     -.5*distance*distance)
    peak = max(terms)
    value = peak+math.log(math.fsum(math.exp(t-peak) for t in terms))
    if not math.isfinite(value):
        raise PolicyError('nonfinite hypothetical world prediction')
    return value


def _integrate(world, path, start_time, speed):
    elapsed, q = 0., []
    for a, b in zip(path, path[1:]):
        duration = math.dist(a, b)/speed
        count = max(1, math.ceil(duration/MAX_QUADRATURE_SECONDS))
        for index in range(count):
            fraction = (index+.5)/count
            point = [x+fraction*(y-x) for x, y in zip(a, b)]
            power = math.exp(_log_power(world, point, start_time+elapsed+fraction*duration))
            q.append(power*(duration/count))
        elapsed += duration
    value = math.fsum(q)
    if not math.isfinite(value):
        raise PolicyError('finite complete-route exposure required')
    return value
