"""有共享预算和可控采样混合的RRT*；仅读当前估计，不持久化树或读取真值。"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import time
import numpy as np
from .planning import PlanResult, path_length


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class WorkBudget:
    """每个决策点共享计数：采样提议次数、未缓存的线段几何/代价评价次数。"""
    sample_limit: int = 2100
    edge_limit: int = 10000
    samples: int = 0
    edges: int = 0
    cache_hits: int = 0
    reserved_edges: int = 0
    verification_edges: int = 0

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.sample_limit, self.edge_limit)):
            raise ValueError('positive integer resource limits required')

    def take_sample(self):
        if self.samples >= self.sample_limit:
            raise BudgetExhausted('sample_budget_exhausted')
        self.samples += 1

    def take_edge(self):
        if self.edges >= self.edge_limit - self.reserved_edges:
            raise BudgetExhausted('edge_budget_exhausted')
        self.edges += 1

    def summary(self):
        return {**asdict(self), 'samples_left': self.sample_limit-self.samples,
                'edges_left': self.edge_limit-self.edges-self.reserved_edges,
                'total_edges_left': self.edge_limit-self.edges}


class FixedObjective:
    """固定J=长度+2×超参考惩罚+0.4×不确定性，单位m的研究性代价。

    沿线段精确分割最近邻栅格的半格边界，逐格常值积分，避免仅改变
    航路分段数量就改变分数。几何与积分合为一个预算边评价；对称缓存
    在同一快照/决策内共享。字段采样、LLM及数组操作另计墙钟，不冒充等CPU。
    """
    def __init__(self, snapshot, reference_dbm, budget):
        self.snapshot, self.geometry = snapshot, snapshot.geometry
        self.reference, self.budget = reference_dbm, budget
        self.cache = {}

    def quadrature(self, start, end):
        a,b=np.asarray(start,float),np.asarray(end,float)
        delta=b-a;length=float(np.linalg.norm(delta));cuts=[0.,1.]
        resolution=self.geometry.config.resolution_m
        for axis,size in enumerate((self.geometry.config.columns,self.geometry.config.rows)):
            if abs(delta[axis])>1e-12:
                crossings=((np.arange(size-1)+.5)*resolution-a[axis])/delta[axis]
                cuts.extend(crossings[(crossings>0)&(crossings<1)].tolist())
        cuts=np.unique(cuts); weights=np.diff(cuts)*length
        points=a+((cuts[:-1]+cuts[1:])/2)[:,None]*delta
        nodes=np.rint(points/resolution).astype(int)
        nodes=np.clip(nodes,[0,0],[self.geometry.config.columns-1,self.geometry.config.rows-1])
        return weights,nodes

    def edge(self, start, end):
        a,b=tuple(float(x) for x in start),tuple(float(x) for x in end)
        key=tuple(sorted((a,b)))
        if key in self.cache:
            self.budget.cache_hits+=1
            return self.cache[key]
        self.budget.take_edge()
        if not self.geometry.edge_free(a,b):
            value=None
        else:
            weights,nodes=self.quadrature(a,b)
            mean=self.snapshot.estimate_dbm[nodes[:,1],nodes[:,0]]
            sd=self.snapshot.uncertainty_db[nodes[:,1],nodes[:,0]]
            length=float(weights.sum())
            interference=float(weights@np.clip((mean-self.reference)/10,0,4))*2
            uncertainty=float(weights@np.clip(sd/15,0,2))*.4
            value={'length_m':length,'interference_cost':interference,'uncertainty_cost':uncertainty,
                   'objective':length+interference+uncertainty}
        self.cache[key]=value
        return value

    def edge_cost(self,start,end):
        value=self.edge(start,end)
        return value['objective'] if value is not None else math.inf

    def route_metrics(self,route):
        if not route:return None
        result=dict(length_m=0.,interference_cost=0.,uncertainty_cost=0.,objective=0.)
        if not all(self.geometry.free(p) for p in route):return None
        for a,b in zip(route,route[1:]):
            value=self.edge(a,b)
            if value is None:return None
            for k in result:result[k]+=value[k]
        return result


def validate_sampling(value, high):
    if not isinstance(value,dict) or set(value)!={'focus_probability','regions'}:
        raise ValueError('sampling requires focus_probability and regions')
    p=value['focus_probability'];regions=value['regions'];high=np.asarray(high,float)
    if isinstance(p,bool) or not isinstance(p,(float,int)) or not math.isfinite(p) or not 0<=p<=.6:
        raise ValueError('focus_probability must be in [0,0.6]')
    if not isinstance(regions,list) or len(regions)>3 or (p>0 and not regions):
        raise ValueError('positive focus requires 1 to 3 regions')
    parsed=[]
    for region in regions:
        if not isinstance(region,dict) or set(region)!={'low_m','high_m','weight'}:
            raise ValueError('region requires low_m/high_m/weight')
        low,upper=region['low_m'],region['high_m'];weight=region['weight']
        for point in (low,upper):
            if not isinstance(point,list) or len(point)!=2 or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in point):
                raise ValueError('region coordinates must be finite pairs')
        if (np.any(np.asarray(low)<0) or np.any(np.asarray(upper)>high) or np.any(np.asarray(upper)<=low)
                or isinstance(weight,bool) or not isinstance(weight,(float,int)) or not math.isfinite(weight) or not 0<weight<=100):
            raise ValueError('invalid region bounds or weight')
        parsed.append({'low_m':list(low),'high_m':list(upper),'weight':float(weight)})
    return {'focus_probability':float(p),'regions':parsed}


def sample_point(rng,goal,high,sampling):
    draw=rng.random()
    if draw<.1:return np.asarray(goal), 'goal'
    if draw<.1+sampling['focus_probability']:
        regions=sampling['regions'];weights=np.array([r['weight'] for r in regions]);weights/=weights.sum()
        region=regions[int(rng.choice(len(regions),p=weights))]
        return rng.uniform(region['low_m'],region['high_m']), 'focus'
    return rng.uniform(np.zeros(2),high), 'global'


def repair_rrt_star(problem,start,goal,sample_budget,seed,sampling):
    """扩展标准RRT*采样与计数；保持选父/重接/后代代价更新，耗尽仍可返回已有解。"""
    started=time.perf_counter();start,goal=np.asarray(start,float),np.asarray(goal,float)
    result=PlanResult('rrt_star',iteration_unit='sample_proposals')
    stats={'goal':0,'focus':0,'global':0}
    geometry=problem.geometry
    sampling=validate_sampling(sampling,geometry.high)
    if type(sample_budget) is not int or sample_budget<0:raise ValueError('nonnegative integer sample budget required')
    if not geometry.free(start) or not geometry.free(goal):
        result.reason='invalid_endpoint';return result,stats
    if np.linalg.norm(start-goal)<1e-9:
        result.path=[start.tolist()];result.reason='goal_reached';return result,stats
    rng=np.random.default_rng(seed);nodes=[start];parents=[-1];costs=[0.];children=[set()];connectors=[]
    step=3*geometry.config.resolution_m
    try:
        for _ in range(sample_budget):
            problem.budget.take_sample();result.iterations+=1
            proposed,mode=sample_point(rng,goal,geometry.high,sampling);stats[mode]+=1
            positions=np.asarray(nodes);distances=np.linalg.norm(positions-proposed,axis=1);nearest=int(np.argmin(distances))
            if distances[nearest]<1e-9:continue
            candidate=nodes[nearest]+(proposed-nodes[nearest])*min(1,step/distances[nearest])
            if not math.isfinite(problem.edge_cost(nodes[nearest],candidate)):continue
            distances=np.linalg.norm(positions-candidate,axis=1)
            if distances.min()<1e-7:continue
            radius=max(step,6*step*math.sqrt(math.log(len(nodes)+2)/(len(nodes)+1)))
            near=set(np.flatnonzero(distances<=radius).tolist())|{nearest}
            edges={i:problem.edge_cost(nodes[i],candidate) for i in sorted(near)}
            parent=min(sorted(near),key=lambda i:costs[i]+edges[i]);cost=costs[parent]+edges[parent]
            if not math.isfinite(cost):continue
            current=len(nodes);nodes.append(candidate);parents.append(parent);costs.append(cost);children.append(set());children[parent].add(current)
            ancestors=set();ancestor=parent
            while ancestor>=0:ancestors.add(ancestor);ancestor=parents[ancestor]
            for i in sorted(near-ancestors-{0}):
                new=cost+edges[i]
                if new<costs[i]-1e-9:
                    children[parents[i]].remove(i);parents[i]=current;children[current].add(i);delta=new-costs[i];pending=[i]
                    while pending:
                        desc=pending.pop();costs[desc]+=delta;pending.extend(children[desc])
            if np.linalg.norm(candidate-goal)<=step:
                edge=problem.edge_cost(candidate,goal)
                if math.isfinite(edge):
                    connectors.append((current,edge))
                    if result.first_solution_iteration is None:result.first_solution_iteration=result.iterations
    except BudgetExhausted as exc:
        result.reason=str(exc)
    if connectors:
        current,edge=min(connectors,key=lambda item:costs[item[0]]+item[1]);total=costs[current]+edge;reverse=[goal.tolist()]
        while current>=0:reverse.append(nodes[current].tolist());current=parents[current]
        result.path=list(reversed(reverse));result.cost=total;result.length_m=path_length(result.path)
        result.reason='solved' if result.reason=='no_path_within_budget' else 'solved_before_'+result.reason
    result.elapsed_seconds=time.perf_counter()-started
    return result,stats
