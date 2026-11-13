"""由短程序组合预设数值算子；解释AST，不执行任意Python。

程序只得到当前public上下文。它计算一个普通修复提议，搜索/接受/预算仍由
RepairSession执行；库函数不会获取新观测、读取文件或调用额外规划器。
"""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import time
import numpy as np

from .agent_navigation import FileDecisionAdvisor
from .route_repair import parse_proposal


LIBRARY_VERSION = 'repair_operators_v1'
API_DESCRIPTION = '''Restricted Python-shaped composition, maximum 12 assignment statements / 32 library calls.
Only assignments, variables, literal strings/integers and the following function calls are allowed.
No imports, attributes, indexing, arithmetic, loops, comprehensions, filesystem, exec/eval or arbitrary functions.
Final variable MUST be decision. These are prewritten numeric operators, not newly generated implementations.
- whole(): interval covering CURRENT remaining route; requires feasible incumbent.
- select_interval(metric="interference"|"uncertainty"|"detour"|"front", padding=0..3):
  interference ranks segment interference_cost/length times supported_fraction; uncertainty ranks
  uncertainty_cost/length; detour ranks adjacent two-edge excess over chord; front selects first segment.
  Padding expands by anchor count, clipped to remaining route. Returns ordered current anchor IDs.
- corridors(interval, side="both"|"left"|"right"|"tube", scale="near"|"local"|"wide"):
  creates clipped axis-aligned boxes around endpoint chord and/or parallel lateral corridors.
  Lateral distance = GP support_radius * (0.5|1|1.5); half-width = max(distance/2,1m).
  Left/right relative to interval start->end. Geometric preferences ONLY, not measured safe corridors.
- sampling(boxes, weighting="cost"|"uniform", concentration="moderate"|"focused"):
  focus=0.3|0.6. Cost weights=1/mean(1+2*clip((mean-reference)/10,0,4)+0.4*clip(std/15,0,2))
  using only public local/coarse samples inside each box; duplicate positions use local sample.
  Unknown uses no observed interference term and prior std=15, not a safety claim. No samples: density=1.4.
- uniform(): focus=0, no boxes.
- quota(mode="small"|"balanced"|"remaining"):
  caps=100|350|samples_left; also capped by edges_left/(estimated_edges_per_sample*remaining_rounds).
  estimated_edges_per_sample=max(16, used_edges/max(used_samples,1)) after a search, else64.
  remaining mode divides by1; other modes divide by current+remaining rounds; minimum1 if work remains.
- waypoint(interval, side="left"|"right", scale="near"|"local"|"wide"):
  midpoint of chord shifted by same support-based lateral distance and clipped to map.
- keep(), shortcut(interval), local_rrt(interval, distribution, samples),
  via(interval, point, distribution, samples), global_rrt(distribution, samples): ordinary v1 proposals.
- stop_if_no_gain(proposal): keep if latest RRT search feedback has reason no_objective_improvement
  on this same snapshot; otherwise returns proposal. This heuristic may stop too early.
All computed numbers come ONLY from the same public context visible to the direct-values arm.
All proposals go through the SAME parser/search/objective/acceptor and shared resource ceilings.
'''


class OperatorLibrary:
    def __init__(self, context):
        self.ctx = deepcopy(context)
        self.r = self.ctx['route_repair']
        self.points = self.r['anchors']

    def whole(self):
        if not self.r['incumbent_feasible'] or len(self.points) < 2:
            raise ValueError('interval requires feasible remaining route')
        return [self.points[0]['id'], self.points[-1]['id']]

    def _indices(self, interval):
        self.whole()
        ids = [a['id'] for a in self.points]
        if not isinstance(interval, list) or len(interval) != 2 or any(v not in ids for v in interval):
            raise ValueError('invalid library interval')
        i, j = (ids.index(v) for v in interval)
        if i >= j:
            raise ValueError('ordered interval required')
        return i, j

    def select_interval(self, metric='interference', padding=1):
        self.whole()
        if type(padding) is not int or not 0 <= padding <= 3:
            raise ValueError('padding must be integer0..3')
        segments = self.r['segments']
        if metric == 'front':
            i, j = 0, 1
        elif metric in ('interference', 'uncertainty'):
            field = metric + '_cost'
            def score(s):
                value = s['cost'][field] / max(s['cost']['length_m'], 1.)
                return value * s['supported_fraction'] if metric == 'interference' else value
            best = max(segments, key=score)
            i, j = best['index'], best['index'] + 1
        elif metric == 'detour':
            if len(self.points) == 2:
                i, j = 0, 1
            else:
                xy = np.array([a['position_m'] for a in self.points])
                excess = [np.linalg.norm(xy[k+1]-xy[k]) + np.linalg.norm(xy[k+2]-xy[k+1])
                          - np.linalg.norm(xy[k+2]-xy[k]) for k in range(len(xy)-2)]
                i = int(np.argmax(excess)); j = i + 2
        else:
            raise ValueError('unknown interval metric')
        return [self.points[max(0, i-padding)]['id'], self.points[min(len(self.points)-1, j+padding)]['id']]

    def _frame(self, interval, scale):
        i, j = self._indices(interval)
        a, b = (np.array(self.points[k]['position_m'], float) for k in (i, j))
        d = b-a; length = float(np.linalg.norm(d))
        if length < 1e-8:
            raise ValueError('coincident interval endpoints')
        if scale not in ('near', 'local', 'wide'):
            raise ValueError('invalid corridor scale')
        support = self.ctx['electromagnetic_estimate']['estimator']['local_prediction_radius_m']
        distance = support * {'near':.5, 'local':1., 'wide':1.5}[scale]
        return a, b, np.array([-d[1], d[0]])/length, distance

    def corridors(self, interval, side='both', scale='local'):
        a, b, normal, distance = self._frame(interval, scale)
        options = {'both':[-1, 1], 'left':[1], 'right':[-1], 'tube':[0]}
        if side not in options:
            raise ValueError('invalid corridor side')
        low_bound, high_bound = np.array(self.ctx['map']['bounds_m'], float)
        width = max(1., distance/2)
        boxes = []
        for sign in options[side]:
            shift = normal*distance*sign
            low = np.maximum(low_bound, np.minimum(a+shift, b+shift)-width)
            high = np.minimum(high_bound, np.maximum(a+shift, b+shift)+width)
            if np.all(high > low):
                boxes.append({'low_m':low.tolist(), 'high_m':high.tolist(), 'weight':1.})
        return boxes

    def sampling(self, boxes, weighting='cost', concentration='focused'):
        if not isinstance(boxes, list) or not 1 <= len(boxes) <= 3:
            raise ValueError('sampling requires1..3 nonempty boxes')
        if weighting not in ('cost', 'uniform') or concentration not in ('moderate', 'focused'):
            raise ValueError('invalid sampling mode')
        result = deepcopy(boxes)
        em = self.ctx['electromagnetic_estimate']
        columns = em['sample_columns']; sample_map = {}
        for sample in em['coarse_samples'] + em.get('local_samples', []):
            row = dict(zip(columns, sample));sample_map[(row['x_m'], row['y_m'])] = row
        for box in result:
            densities = []
            for xy, row in sample_map.items():
                if np.all(np.array(xy) >= box['low_m']) and np.all(np.array(xy) <= box['high_m']):
                    mean = row['estimate_dbm']
                    risk = float(np.clip((mean-em['interference_reference_dbm'])/10, 0, 4)) if row['supported'] and mean is not None else 0.
                    densities.append(1. + 2.*risk + .4*float(np.clip(row['uncertainty_db']/15, 0, 2)))
            box['weight'] = 1./float(np.mean(densities) if densities else 1.4) if weighting == 'cost' else 1.
        return {'focus_probability': {'moderate':.3, 'focused':.6}[concentration], 'regions':result}

    def uniform(self):
        return {'focus_probability':0., 'regions':[]}

    def quota(self, mode='balanced'):
        if mode not in ('small', 'balanced', 'remaining'):
            raise ValueError('invalid quota mode')
        w = self.r['resources'];left = w['samples_left']
        if left < 1 or w['edges_left'] < 1:
            raise ValueError('no remaining search work')
        ratio = max(16., w['edges']/max(w['samples'], 1)) if w['samples'] else 64.
        rounds = 1 if mode == 'remaining' else 1+self.r['rounds_left']
        cap = {'small':100, 'balanced':350, 'remaining':left}[mode]
        return max(1, min(left, cap, int(w['edges_left']/(ratio*rounds))))

    def waypoint(self, interval, side='left', scale='local'):
        a, b, normal, distance = self._frame(interval, scale)
        if side not in ('left', 'right'):
            raise ValueError('invalid waypoint side')
        low, high = self.ctx['map']['bounds_m']
        return np.clip((a+b)/2 + normal*distance*(1 if side == 'left' else -1), low, high).tolist()

    def _proposal(self, op, **kwargs):
        return {'route_version':self.r['route_version'], 'operator':op, 'reason':'预设算子组合计算', **kwargs}

    def keep(self):
        return self._proposal('keep')

    def shortcut(self, interval):
        self._indices(interval)
        return self._proposal('shortcut', start_id=interval[0], end_id=interval[1])

    def local_rrt(self, interval, distribution, samples):
        self._indices(interval)
        return self._proposal('local_rrt', start_id=interval[0], end_id=interval[1], sampling=distribution, sample_budget=samples)

    def via(self, interval, point, distribution, samples):
        self._indices(interval)
        return self._proposal('via', start_id=interval[0], end_id=interval[1], waypoint_m=point, sampling=distribution, sample_budget=max(2, samples))

    def global_rrt(self, distribution, samples):
        return self._proposal('global_rrt', sampling=distribution, sample_budget=samples)

    def stop_if_no_gain(self, proposal):
        history = self.r['previous_feedback']
        stalled = (history and history[-1].get('operator') in ('local_rrt', 'via', 'global_rrt')
                   and history[-1]['reason'] == 'no_objective_improvement')
        return self.keep() if self.r['incumbent_feasible'] and stalled else deepcopy(proposal)


FUNCTIONS = ('whole', 'select_interval', 'corridors', 'sampling', 'uniform', 'quota',
             'waypoint', 'keep', 'shortcut', 'local_rrt', 'via', 'global_rrt', 'stop_if_no_gain')


def compile_program(source, context, reason):
    """确定性解释有限调用图；返回普通提议和每步计算记录，不调用eval/exec。"""
    started = time.perf_counter()
    if not isinstance(source, str) or not source.strip() or len(source) > 3000:
        raise ValueError('program length must be1..3000 characters')
    tree = ast.parse(source, mode='exec')
    if not 1 <= len(tree.body) <= 12 or sum(1 for _ in ast.walk(tree)) > 250:
        raise ValueError('program AST too large')
    if sum(isinstance(n, ast.Call) for n in ast.walk(tree)) > 32:
        raise ValueError('program call limit')
    allowed = (ast.Module, ast.Assign, ast.Name, ast.Store, ast.Load, ast.Call, ast.keyword, ast.Constant)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError('only straight-line library-call assignments allowed')
    lib = OperatorLibrary(context); variables = {};calls = []
    def evaluate(node):
        if isinstance(node, ast.Constant):
            if type(node.value) is int and 0 <= node.value <= 100:
                return node.value
            if isinstance(node.value, str) and len(node.value) <= 80:
                return node.value
            raise ValueError('only short string/small integer literals allowed')
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError('unknown variable: '+node.id)
            return deepcopy(variables[node.id])
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise ValueError('function is not a preset operator')
            if len(calls) >= 32 or any(k.arg is None for k in node.keywords):
                raise ValueError('call limit or keyword expansion')
            args = [evaluate(a) for a in node.args];kwargs = {k.arg:evaluate(k.value) for k in node.keywords}
            if len(kwargs) != len(node.keywords):
                raise ValueError('duplicate keyword')
            result = getattr(lib, node.func.id)(*args, **kwargs)
            calls.append({'operator':node.func.id, 'args':deepcopy(args), 'kwargs':deepcopy(kwargs), 'result':deepcopy(result)})
            return result
        raise ValueError('unsupported expression')
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            raise ValueError('single variable assignment required')
        name = statement.targets[0].id
        if name.startswith('_') or name in FUNCTIONS or name in variables:
            raise ValueError('reserved/reassigned variable')
        variables[name] = evaluate(statement.value)
    if not isinstance(variables.get('decision'), dict):
        raise ValueError('program must assign decision proposal')
    proposal = deepcopy(variables['decision']);proposal['reason'] = reason
    proposal = parse_proposal(proposal, context)
    return proposal, {'library_version':LIBRARY_VERSION, 'program':source, 'calls':calls,
                      'resolved_proposal':proposal, 'compute_seconds':time.perf_counter()-started,
                      'arbitrary_python_executed':False}


class FileProgramPolicy(FileDecisionAdvisor):
    def __init__(self, public_root, mission_id, timeout_seconds=600):
        super().__init__(public_root, mission_id, timeout_seconds, validator=self._validate)

    def _validate(self, value, identifier, context):
        if not isinstance(value, dict) or set(value) != {'request_id', 'decision', 'strategy', 'memory'} or value['request_id'] != identifier:
            raise ValueError('response identity/schema mismatch')
        if value['strategy'] != {'mode':'baseline'} or not isinstance(value['memory'], str) or len(value['memory']) > 1200:
            raise ValueError('invalid strategy/memory')
        decision = value['decision']
        if not isinstance(decision, dict) or set(decision) != {'program', 'reason'}:
            raise ValueError('program decision requires program/reason only')
        proposal, trace = compile_program(decision['program'], context, decision['reason'])
        self.last_call['program_execution'] = trace
        return json.dumps(proposal, ensure_ascii=False, allow_nan=False), {'mode':'baseline'}, value['memory']


class LibraryNumericalPolicy:
    """与模型共享算子的确定性自适应对照；规则在试验前冻结，不称最强基线。"""
    last_call = {}

    def decide(self, context):
        lib = OperatorLibrary(context);r = context['route_repair']
        if not r['incumbent_feasible']:
            p = lib.global_rrt(lib.uniform(), lib.quota('remaining'))
        elif (r['previous_feedback'] and r['previous_feedback'][-1].get('operator') in ('local_rrt','via','global_rrt')
              and r['previous_feedback'][-1]['reason'] == 'no_objective_improvement'):
            p = lib.keep()
        elif r['round'] == 1:
            p = lib.shortcut(lib.whole())
        elif any(s['cost']['interference_cost'] > 1e-8 for s in r['segments']) and r['resources']['edges_left'] > 128:
            interval = lib.select_interval('interference', 1)
            p = lib.local_rrt(interval, lib.sampling(lib.corridors(interval)), lib.quota('balanced'))
        else:
            p = lib.keep()
        return json.dumps(p, ensure_ascii=False)
