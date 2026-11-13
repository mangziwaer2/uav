"""人工共同目标v2：有限航程下的近期风险；只读当前GP/公开几何。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import time
import numpy as np
from .planning import PlanResult, path_length
from .repair_search import FixedObjective, WorkBudget, BudgetExhausted, repair_rrt_star
from .repair_process import canonical_path
from .task_readiness import gaussian_positive_part

VERSION = 'prefix_risk_v2'

@dataclass(frozen=True)
class ObjectiveSettings:
    horizon_seconds: float = 10.
    unknown_excess_db: float = 10.
    risk_tolerance_db_seconds: float = 1.
    mission_length_ratio: float = 1.25
    search_samples: int = 700
    sample_limit: int = 2100
    edge_limit: int = 10000

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (self.horizon_seconds,self.unknown_excess_db,self.mission_length_ratio)):
            raise ValueError('finite positive horizon/unknown penalty/ratio required')
        if self.mission_length_ratio<1 or not math.isfinite(self.risk_tolerance_db_seconds) or self.risk_tolerance_db_seconds<0:
            raise ValueError('invalid ratio/tolerance')
        WorkBudget(self.sample_limit,self.edge_limit)
        if type(self.search_samples) is not int or not 0<self.search_samples<=self.sample_limit:
            raise ValueError('invalid sample quota')


def cut_route(route, distance_m):
    """保留所有几何拐点，在累计距离处准确切开；零长边不影响结果。"""
    if not route or not math.isfinite(distance_m) or distance_m<0:raise ValueError('route and nonnegative distance required')
    front=[list(route[0])];left=distance_m
    for i,(a,b) in enumerate(zip(route,route[1:])):
        a,b=np.asarray(a,float),np.asarray(b,float);length=float(np.linalg.norm(b-a))
        if length<=1e-10:continue
        if left<length-1e-10:
            point=(a+(b-a)*left/length).tolist()
            if left>1e-10:front.append(point)
            return front,[point]+deepcopy(route[i+1:])
        front.append(b.tolist());left-=length
    return front,[list(route[-1])]


def prefix_risk(problem, route, speed_mps, settings=ObjectiveSettings()):
    if not math.isfinite(speed_mps) or speed_mps<=0:raise ValueError('positive speed required')
    prefix,tail=cut_route(route,speed_mps*settings.horizon_seconds)
    expected=known_seconds=unknown_seconds=0.;field_nodes=0
    for a,b in zip(prefix,prefix[1:]):
        weights,nodes=problem.quadrature(a,b);duration=weights/speed_mps
        rows,cols=nodes[:,1],nodes[:,0];s=problem.snapshot;mask=s.supported[rows,cols]
        # Unknown numerical priors are never passed to the supported-risk expectation.
        excess=gaussian_positive_part(s.estimate_dbm[rows[mask],cols[mask]],
                                     s.uncertainty_db[rows[mask],cols[mask]],problem.reference)
        expected+=float(duration[mask]@excess);known_seconds+=float(duration[mask].sum())
        unknown_seconds+=float(duration[~mask].sum());field_nodes+=len(nodes)
    return {'risk_db_seconds':expected+settings.unknown_excess_db*unknown_seconds,
            'supported_expected_excess_db_seconds':expected,'unknown_seconds':unknown_seconds,
            'supported_seconds':known_seconds,'evaluated_seconds':known_seconds+unknown_seconds,
            'unknown_penalty_db_seconds':settings.unknown_excess_db*unknown_seconds,
            'tail_length_m':path_length(tail),'field_nodes':field_nodes,
            'arrival_field_predicted':False,'uncertainty_calibrated':False,
            'unknown_is_measurement':False}


def select_candidate(records, method, settings=ObjectiveSettings()):
    valid=[r for r in records if r['feasible']]
    if not valid:return None
    # Stable ordering retains incumbent on exact ties.
    if method=='pool_j':return min(valid,key=lambda r:r['metrics']['objective'])
    if method!='prefix_v2':raise ValueError('unknown objective method')
    minimum=min(r['prefix']['risk_db_seconds'] for r in valid)
    near=[r for r in valid if r['prefix']['risk_db_seconds']<=minimum+settings.risk_tolerance_db_seconds+1e-9]
    return min(near,key=lambda r:r['metrics']['length_m'])


def candidate_paths(route, radius_m, high):
    """固定前段/全段侧移池，几何偏好不代表观测到源方向。"""
    yield 'incumbent',deepcopy(route)
    for label,distance in (('whole',path_length(route)),('front150',150.),('front300',300.)):
        front,tail=cut_route(route,distance);a,b=np.asarray(front[0]),np.asarray(front[-1]);delta=b-a
        if np.linalg.norm(delta)<1e-9:continue
        normal=np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
        yield label+'_chord',[a.tolist(),b.tolist()]+deepcopy(tail[1:])
        for scale in (.5,1.,1.5):
            for side,sign in (('left',1),('right',-1)):
                offset=normal*radius_m*scale*sign
                mid=np.clip((a+b)/2+offset,0,high)
                first,second=np.clip(a+offset,0,high),np.clip(b+offset,0,high)
                yield f'{label}_via_{side}_{scale:g}',[a.tolist(),mid.tolist(),b.tolist()]+deepcopy(tail[1:])
                yield f'{label}_parallel_{side}_{scale:g}',[a.tolist(),first.tolist(),second.tolist(),b.tolist()]+deepcopy(tail[1:])


class ObjectiveOptimizer:
    """两方法共用候选/预算；与v1独立，不改变历史接受器或GP。"""
    policy=None
    last_call={}

    def __init__(self, method, settings=ObjectiveSettings()):
        if method not in ('pool_j','prefix_v2'):raise ValueError('unknown method')
        self.method,self.settings=method,settings
        self.reference_length_m=None;self.flown_m=0.;self.last_time=None

    def optimize(self,snapshot,context,route,max_length_m,seed,call_limit):
        started=time.perf_counter();settings=self.settings
        if self.last_time is not None and snapshot.time_seconds<=self.last_time:
            raise ValueError('optimizer requires strictly advancing mission time')
        interval=context.get('execution_feedback',{}).get('interval')
        if interval:
            flown=float(interval['flown_m'])
            if not math.isfinite(flown) or flown<0:raise ValueError('invalid executed distance')
            self.flown_m+=flown
        self.last_time=snapshot.time_seconds
        work=WorkBudget(settings.sample_limit,settings.edge_limit)
        problem=FixedObjective(snapshot,context['electromagnetic_estimate']['interference_reference_dbm'],work)
        start=context['vehicle']['position_m'];goal=context['vehicle']['goal_m'];speed=context['vehicle']['speed_mps']
        searches=[];records=[];seen=set();field_nodes=0;initial_missing=not route
        # Reserved edges cover route scoring and independent final verification after RRT.
        work.reserved_edges=256

        def search():
            reserve=work.reserved_edges;work.reserved_edges+=512
            try:
                result,stats=repair_rrt_star(problem,start,goal,settings.search_samples,seed,{'focus_probability':0.,'regions':[]})
            finally:work.reserved_edges=reserve
            searches.append({'purpose':'shared_current_j_candidate','profile':'balanced','sampling_counts':stats,**asdict(result)})
            return result.path

        def record(identifier,path):
            nonlocal field_nodes
            fingerprint=canonical_path(path)
            if fingerprint in seen:return
            seen.add(fingerprint);metrics=None;reason=None;prefix=None
            try:metrics=problem.route_metrics(path)
            except BudgetExhausted as exc:reason=str(exc)
            if metrics is None:reason=reason or 'geometry_invalid'
            elif not (np.allclose(path[0],start,atol=1e-8,rtol=0) and np.allclose(path[-1],goal,atol=1e-8,rtol=0)):
                reason='endpoint_mismatch'
            elif metrics['length_m']>max_length_m+1e-8:reason='energy_limit'
            else:
                prefix=prefix_risk(problem,path,speed,settings);field_nodes+=prefix['field_nodes']
            records.append({'id':identifier,'path':deepcopy(path),'metrics':metrics,'prefix':prefix,'feasible':reason is None,'failure':reason})

        base=deepcopy(route) if route else search()
        if base:
            radius=context['electromagnetic_estimate']['estimator']['local_prediction_radius_m']
            for identifier,path in candidate_paths(base,radius,problem.geometry.high):record(identifier,path)
            if not initial_missing:
                proposal=search()
                if proposal:record('shared_rrt',proposal)
        valid=[r for r in records if r['feasible']]
        if self.reference_length_m is None and valid:
            self.reference_length_m=min(r['metrics']['length_m'] for r in valid)
        cap=settings.mission_length_ratio*self.reference_length_m if self.reference_length_m is not None else None
        for r in valid:
            if self.flown_m+r['metrics']['length_m']>cap+1e-7:
                r.update(feasible=False,failure='mission_length_limit')
        chosen=select_candidate(records,self.method,settings)
        path=chosen['path'] if chosen else []
        work.reserved_edges=0
        for a,b in zip(path,path[1:]):
            work.take_edge();work.verification_edges+=1
            if not problem.geometry.edge_free(a,b):raise RuntimeError('final geometry invalid')
        initial=next((r['metrics'] for r in records if r['id']=='incumbent'),None)
        elapsed=time.perf_counter()-started
        changed=bool(chosen and (initial_missing or canonical_path(path)!=canonical_path(route)))
        report={'schema_version':2,'contract_version':VERSION,'method':self.method,'snapshot_time_seconds':snapshot.time_seconds,
            'objective':self.method,'initial_metrics':initial,'best_metrics':chosen['metrics'] if chosen else None,
            'selected_prefix':chosen['prefix'] if chosen else None,'selected_candidate_id':chosen['id'] if chosen else None,
            'selection_settings':asdict(settings),'mission_reference_length_m':self.reference_length_m,
            'mission_distance_cap_m':cap,'executed_distance_m':self.flown_m,
            'trace':[],'searches':searches,'resources':work.summary(),'actual_advisor_calls':0,'advisor_failures':0,
            'policy_errors':0,'advisor_seconds':0.,'numeric_compute_seconds':elapsed,'numeric_recovery':False,
            'call_budget_exhausted':False,'accepted_proposals':int(changed),'accepted_model_proposals':0,'accepted_recovery_proposals':0,
            'has_final_route':bool(path),'final_geometry_verified':bool(path),'initial_validation_failure':None,
            'candidates':records,'risk_field_nodes':field_nodes,'new_observations':0,
            'no_probe_action':True,'current_j_search_and_v2_selection_are_distinct':True}
        return PlanResult('rrt_star',path=path,cost=chosen['metrics']['objective'] if chosen else None,
            length_m=path_length(path) if path else None,iterations=work.samples,iteration_unit='sample_proposals',
            elapsed_seconds=elapsed,reason='objective_pool_selected' if path else 'objective_pool_no_feasible_route'),report
