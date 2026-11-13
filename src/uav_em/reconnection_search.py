"""同快照多接回点/分批独立RRT*竞争；研究者编写，不读取未来场。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import math
import time
import numpy as np
from .aligned_search import PowerTimeSearchObjective
from .balanced_objective import BalancedOptimizer, PowerTimeSettings, _BalancedSession, choose, paired_seed, power_time_score
from .decision_objective import ObjectiveSettings, prefix_risk
from .planning import path_length
from .repair_program import OperatorLibrary
from .repair_process import canonical_path
from .repair_search import BudgetExhausted, repair_rrt_star
from .route_repair import anchors

VERSION='reconnection_competition_v1'
MODES=('single_C','single_batched','multi_equal','multi_feedback')
MAX_BATCHES=6
POST_SAMPLE_CAP=700


class EdgeSlice:
    """局部硬边额度只限制本次搜索，不重置/扩大共享台账。"""
    def __init__(self,shared,edge_quota):
        if type(edge_quota) is not int or edge_quota<0:raise ValueError('nonnegative integer edge quota required')
        self.shared=shared;self.start_edges=shared.edges;self.edge_quota=edge_quota
    def take_sample(self):self.shared.take_sample()
    def take_edge(self):
        if self.shared.edges-self.start_edges>=self.edge_quota:raise BudgetExhausted('batch_edge_budget_exhausted')
        self.shared.take_edge()
    @property
    def cache_hits(self):return self.shared.cache_hits
    @cache_hits.setter
    def cache_hits(self,value):self.shared.cache_hits=value


def reconnections(ctx):
    """同一局部风险入口；原局部终点、后延两锚点、goal。按几何端点去重。"""
    lib=OperatorLibrary(ctx);i,j=map(lambda x:int(x[1:]),lib.select_interval('interference',1))
    points=ctx['route_repair']['anchors'];ends=[]
    for end in (j,min(j+2,len(points)-1),len(points)-1):
        xy=np.asarray(points[end]['position_m'])
        if (end>i and np.linalg.norm(xy-points[i]['position_m'])>1e-8
            and not any(np.allclose(xy,points[k]['position_m'],rtol=0,atol=1e-8) for k in ends)):
            ends.append(end)
    if not ends:raise ValueError('no distinct forward reconnection endpoint')
    return [dict(id=f'reconnect_{end}',start_index=i,end_index=end,
        start_m=deepcopy(points[i]['position_m']),end_m=deepcopy(points[end]['position_m'])) for end in ends]


def select_arm(mode,arms,history,step):
    if mode in ('single_C','single_batched'):return arms[0]
    if mode=='multi_equal' or step<len(arms):return arms[step%len(arms)]
    # Compare WHOLE remaining route costs, not unequal local leg costs.
    solved=[(min(h['candidate_C_seconds'] for h in history if h['arm_id']==a['id'] and h['candidate_C_seconds'] is not None),a)
        for a in arms if any(h['arm_id']==a['id'] and h['candidate_C_seconds'] is not None for h in history)]
    if solved:return min(solved,key=lambda pair:(pair[0],pair[1]['end_index']))[1]
    return min(arms,key=lambda a:(sum(h['search_samples'] for h in history if h['arm_id']==a['id']),a['end_index']))


def run_competition(snapshot,context,pool_report,maximum,scene_seed,mode,
                    settings=PowerTimeSettings(),limits=ObjectiveSettings()):
    if mode not in MODES:raise ValueError('unknown competition mode')
    started=time.perf_counter();chosen=choose(pool_report['candidates'])
    session=_BalancedSession(snapshot,context,anchors(chosen['path'],80.),maximum,limits,scene_seed,pool_report,settings,'balanced')
    initial=deepcopy(session.work.summary());base=deepcopy(session.best)
    ctx=session.request(1,1);arms=reconnections(ctx)
    if mode in ('single_C','single_batched'):arms=arms[:1]
    lib=OperatorLibrary(ctx);history=[];candidate_records=[];score_nodes=0
    search_problem=PowerTimeSearchObjective(session.snapshot,session.speed,session.work,settings)
    initial_samples=session.work.samples
    initial_edges_left=session.work.summary()['edges_left']
    # Fixed equal shares, including scoring margin; unused shares return to the shared ledger.
    edge_share=max(0,(initial_edges_left-128*MAX_BATCHES)//MAX_BATCHES)
    if edge_share<1:edge_share=max(0,(initial_edges_left-128)//MAX_BATCHES)
    batches=1 if mode=='single_C' else MAX_BATCHES
    for step in range(batches):
        left=min(POST_SAMPLE_CAP-(session.work.samples-initial_samples),session.work.summary()['samples_left'])
        available=session.work.summary()['edges_left']
        reserve=min(128,max(0,available//2))
        final_batch=step==batches-1
        edge_quota=max(0,available-reserve) if final_batch else min(edge_share,max(0,available-reserve))
        quota=left if final_batch else min(left,max(1,math.ceil(POST_SAMPLE_CAP/MAX_BATCHES)))
        if left<1 or edge_quota<1:break
        arm=select_arm(mode,arms,history,step)
        i,j=arm['start_index'],arm['end_index']
        interval=[f'a{i}',f'a{j}'];boxes=lib.corridors(interval,side='both',scale='local')
        sampling=lib.sampling(boxes) if boxes else lib.uniform()
        repeat=sum(h['arm_id']==arm['id'] for h in history)
        purpose='::'.join((canonical_path(base),arm['id'],f'repeat{repeat}'))
        seed=paired_seed(scene_seed,snapshot.time_seconds,VERSION,purpose)
        before=deepcopy(session.work.summary());old_score=choose(session.records)['balance']['score_seconds']
        prior_reserve=session.work.reserved_edges;session.work.reserved_edges+=reserve
        slice_budget=EdgeSlice(session.work,edge_quota)
        search_problem.budget=slice_budget;search_problem.cache={}
        try:
            plan,stats=repair_rrt_star(search_problem,arm['start_m'],arm['end_m'],quota,seed,sampling)
        finally:session.work.reserved_edges=prior_reserve
        after_search=deepcopy(session.work.summary())
        search=dict(purpose='reconnect_rrt',actual_seed=seed,seed_stage=VERSION,seed_purpose=purpose,
            arm_id=arm['id'],batch=step,repeat=repeat,search_objective='power_time_C',search_cost_unit='seconds',
            batch_edge_quota=edge_quota,batch_sample_quota=quota,search_cache_empty_at_start=True,
            sampling=deepcopy(sampling),sampling_counts=stats,budget_before=before,budget_after=after_search,**asdict(plan))
        session.searches.append(search)
        path=anchors(base[:i]+plan.path+base[j+1:],80.) if plan.path else []
        balance=None;metrics=None;failure=None;duplicate=None;error=None
        try:
            if not path:failure='no_path_within_batch'
            elif not session.endpoints(path):failure='route_endpoints_invalid'
            elif path_length(path)>maximum+1e-8:failure='energy_or_mission_length_exceeded'
            else:
                metrics=session.problem.route_metrics(path)
                if metrics is None:failure='geometry_invalid'
                elif session.work.edge_limit-session.work.edges<len(path)-1:failure='insufficient_final_verification_budget'
                else:
                    balance=power_time_score(session.problem,path,session.speed,settings)
                    score_nodes+=balance['field_nodes']
                    duplicate=next((r['id'] for r in session.records if r['feasible'] and canonical_path(r['path'])==canonical_path(path)),None)
                    if duplicate is None:
                        risk=prefix_risk(session.problem,path,session.speed);score_nodes+=risk['field_nodes']
                        session.records.append(dict(id=f'reconnection_candidate_{step}',path=path,feasible=True,failure=None,
                            metrics=metrics,prefix=risk,balance=balance,arm_id=arm['id']))
        except BudgetExhausted as exc:failure=str(exc);balance=None
        selected=choose(session.records)
        accepted=canonical_path(selected['path'])!=canonical_path(session.best)
        if accepted:
            session.best=deepcopy(selected['path']);session.best_metrics=deepcopy(selected['metrics'])
            session.accepted_proposals+=1
        session.work.reserved_edges=max(session.work.reserved_edges,len(session.best)-1)
        trace=dict(batch=step,arm_id=arm['id'],start_index=i,end_index=j,repeat=repeat,
            selected_using='whole_route_numeric_C_only',frozen_base_fingerprint=canonical_path(base),
            edge_quota=edge_quota,sample_quota=quota,search_edges=after_search['edges']-before['edges'],
            search_samples=after_search['samples']-before['samples'],resources_before=before,resources_after=session.work.summary(),
            candidate_C_seconds=balance['score_seconds'] if balance else None,candidate_failure=failure,duplicate_of=duplicate,
            previous_best_C_seconds=old_score,best_C_seconds=selected['balance']['score_seconds'],accepted=accepted,error=error)
        history.append(trace)
        candidate_records.append(dict(batch=step,arm_id=arm['id'],path=path,feasible=failure is None,
            failure=failure,metrics=metrics,balance=balance,duplicate_of=duplicate))
        session.trace.append(dict(error=None,feedback=dict(accepted=accepted,reason=failure or ('improved' if accepted else 'no_objective_improvement'))))
    session.work.reserved_edges=0
    for a,b in zip(session.best,session.best[1:]):
        session.work.take_edge();session.work.verification_edges+=1
        if not snapshot.geometry.edge_free(a,b):raise ValueError('final competition geometry invalid')
    selected=choose(session.records)
    return dict(contract_version=VERSION,competition_mode=mode,path=session.best,candidates=session.records,
        competition_candidates=candidate_records,competition_trace=history,reconnection_options=arms,frozen_base_route=base,
        trace=session.trace,searches=session.searches,resources=session.work.summary(),initial_resources=initial,
        starting_resources_already_paid=deepcopy(pool_report['resources']),selected_candidate_id=selected['id'],
        selected_balance=selected['balance'],selected_prefix=selected['prefix'],best_metrics=selected['metrics'],
        final_geometry_verified=True,policy_errors=0,batch_errors=0,actual_advisor_calls=0,model_calls=0,new_measurements=0,
        score_field_nodes=score_nodes,grid_nodes_built=search_problem.grid_nodes_built,search_field_nodes=search_problem.field_nodes,
        grid_build_seconds=search_problem.grid_build_seconds,numeric_compute_seconds=time.perf_counter()-started,
        search_sample_cap=POST_SAMPLE_CAP,tree_reuse=False)


class ReconnectionOptimizer:
    policy=None
    last_call={}
    def __init__(self,scene_seed,mode,settings=PowerTimeSettings(),limits=ObjectiveSettings()):
        if mode not in MODES:raise ValueError('unknown mode')
        self.base=BalancedOptimizer(scene_seed,settings,limits=limits)
        self.scene_seed,self.mode,self.settings,self.limits=scene_seed,mode,settings,limits
    def optimize(self,snapshot,context,route,max_length_m,seed,call_limit):
        started=time.perf_counter()
        result,report=self.base.optimize(snapshot,context,route,max_length_m,seed,0)
        if not result.path:return result,report
        maximum=min(max_length_m,report['mission_distance_cap_m']-report['executed_distance_m'])
        extra=run_competition(snapshot,context,report,maximum,self.scene_seed,self.mode,self.settings,self.limits)
        searches=report['searches']+extra['searches'];trace=report['trace']+extra['trace']
        nodes=report['score_field_nodes']+extra['score_field_nodes'];initial=deepcopy(report['resources'])
        report.update(extra,searches=searches,trace=trace,score_field_nodes=nodes,selection_rule='balanced',
            primitive_initial_resources=initial,accepted_proposals=int(not route or canonical_path(route)!=canonical_path(extra['path'])),
            numeric_compute_seconds=time.perf_counter()-started)
        result.path=extra['path'];result.length_m=path_length(result.path);result.cost=extra['selected_balance']['score_seconds']
        result.iterations=extra['resources']['samples'];result.elapsed_seconds=report['numeric_compute_seconds']
        return result,report
