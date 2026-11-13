"""离线生成入口/接回/调度表达式的受控执行器；无在线LLM或自由代码执行。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import math,time
import numpy as np
from .aligned_search import PowerTimeSearchObjective
from .balanced_objective import BalancedOptimizer,PowerTimeSettings,_BalancedSession,choose,paired_seed,power_time_score
from .decision_objective import ObjectiveSettings,prefix_risk
from .planning import path_length
from .repair_program import OperatorLibrary
from .repair_process import canonical_path
from .repair_search import BudgetExhausted,repair_rrt_star
from .reconnection_search import EdgeSlice,MAX_BATCHES,POST_SAMPLE_CAP,VERSION as STREAM_VERSION
from .route_repair import anchors
from .structural_policy import ExpressionPolicy

VERSION='offline_generated_structure_v2'


def segments_public(session,ctx):
 base=session.best;rows=[]
 for k,(a,b) in enumerate(zip(base,base[1:])):
  balance=power_time_score(session.problem,[a,b],session.speed,session.score_settings)
  seg=ctx['route_repair']['segments'][k];length=seg['cost']['length_m'];seconds=balance['remaining_flight_seconds']
  detour=0.
  if k+2<len(base):detour=path_length(base[k:k+3])-float(np.linalg.norm(np.array(base[k+2])-base[k]))
  rows.append(dict(index=k,fraction=k/max(1,len(base)-2),supported=seg['supported_fraction'],
   power_density=balance['power_proxy_seconds']/max(seconds,1e-9),uncertainty=seg['cost']['uncertainty_cost']/max(length,1e-9),
   age=float(seg['mean_supported_age_seconds'] or 0.),seconds=seconds,detour=detour))
 return rows


def build_options(session,ctx,policy):
 base=session.best;segments=segments_public(session,ctx)
 ranks=[policy.value('entry_score',row) for row in segments];selected=max(range(len(ranks)),key=lambda k:ranks[k])
 offset=policy.value('entry_offset',dict(index=selected,count=len(segments)))
 if offset!=int(offset) or not -3<=offset<=0:raise ValueError('entry_offset must be integer -3..0')
 i=max(0,selected+int(offset));balances=[power_time_score(session.problem,[a,b],session.speed,session.score_settings) for a,b in zip(base,base[1:])]
 initial_C=sum(b['score_seconds'] for b in balances);options=[];seen=[]
 original_i=int(OperatorLibrary(ctx).select_interval('interference',1)[0][1:])
 for j in range(i+1,len(base)):
  point=np.array(base[j]);a=np.array(base[i]);chord_seconds=float(np.linalg.norm(point-a))/session.speed
  if chord_seconds<1e-9 or any(np.allclose(point,p,atol=1e-8,rtol=0) for p in seen):continue
  seen.append(point);session.work.take_edge();free=bool(session.problem.geometry.edge_free(a,point))
  chord=power_time_score(session.problem,[base[i],base[j]],session.speed,session.score_settings)
  outside=sum(b['score_seconds'] for b in balances[:i]+balances[j:]);span=sum(b['remaining_flight_seconds'] for b in balances[i:j])
  feature=dict(end_index=j,fraction=j/(len(base)-1),span_seconds=span,chord_seconds=chord_seconds,chord_free=free,
   chord_C=outside+chord['score_seconds'],base_C=initial_C,
   supported=sum(b['supported_seconds'] for b in balances[i:j])/max(span,1e-9),
   unknown_seconds=sum(b['unknown_seconds'] for b in balances[i:j]))
  rank=policy.value('exit_score',feature)
  options.append(dict(id=f'reconnect_{j}' if i==original_i else f'reconnect_{i}_{j}',start_index=i,end_index=j,
   start_m=deepcopy(base[i]),end_m=deepcopy(base[j]),features=feature,exit_rank=rank))
 if not options:raise ValueError('no forward options')
 options=sorted(options,key=lambda x:(-x['exit_rank'],x['end_index']))[:policy.program['max_arms']]
 return options,dict(segments=segments,entry_ranks=ranks,selected_segment=selected,entry_offset=offset)


def arm_features(arm,history,batch,best_C,initial_C,arms):
 h=[r for r in history if r['arm_id']==arm['id']];solved=[r['candidate_C_seconds'] for r in h if r['candidate_C_seconds'] is not None]
 f=arm['features']
 return dict(attempts=len(h),successes=len(solved),best_candidate_C=min(solved) if solved else 1e12,best_C=best_C,initial_C=initial_C,
  gain=max(0.,initial_C-min(solved)) if solved else 0.,last_gain=max(0.,h[-1]['previous_best_C_seconds']-h[-1]['best_C_seconds']) if h else 0.,
  edges_spent=sum(r['search_edges'] for r in h),samples_spent=sum(r['search_samples'] for r in h),
  chord_free=f['chord_free'],chord_C=f['chord_C'],chord_seconds=f['chord_seconds'],span_seconds=f['span_seconds'],
  supported=f['supported'],unknown_seconds=f['unknown_seconds'],batch=batch,arms=arms)


def run_generated(snapshot,context,pool_report,maximum,scene_seed,program,settings=PowerTimeSettings(),limits=ObjectiveSettings()):
 started=time.perf_counter();policy=ExpressionPolicy(program);chosen=choose(pool_report['candidates'])
 session=_BalancedSession(snapshot,context,anchors(chosen['path'],80.),maximum,limits,scene_seed,pool_report,settings,'balanced')
 initial=deepcopy(session.work.summary());base=deepcopy(session.best);initial_C=chosen['balance']['score_seconds']
 ctx=session.request(1,1);history=[];candidate_records=[];policy_trace=[];score_nodes=0;errors=[];build={};arms=[];stop_reason='batch_limit'
 search_problem=PowerTimeSearchObjective(session.snapshot,session.speed,session.work,settings);initial_samples=session.work.samples
 try:
  arms,build=build_options(session,ctx,policy)
  for step in range(MAX_BATCHES):
   left=min(POST_SAMPLE_CAP-(session.work.samples-initial_samples),session.work.summary()['samples_left'])
   available=session.work.summary()['edges_left'];reserve=min(128,max(0,available//2));available_search=max(0,available-reserve)
   if available<1:stop_reason='resource_stop';break
   best_C=choose(session.records)['balance']['score_seconds']
   stop_features=dict(batch=step,arms=len(arms),best_C=best_C,initial_C=initial_C,gain=max(0.,initial_C-best_C),samples_left=left,edges_left=available_search,batches_left=MAX_BATCHES-step)
   stop=policy.value('stop',stop_features)
   if stop:
    policy_trace.append(dict(batch=step,stop_features=stop_features,stop=True));stop_reason='generated_stop';break
   priorities=[policy.value('priority',arm_features(a,history,step,best_C,initial_C,len(arms))) for a in arms]
   arm=arms[max(range(len(arms)),key=lambda k:priorities[k])];i,j=arm['start_index'],arm['end_index']
   af=arm_features(arm,history,step,best_C,initial_C,len(arms))
   op_features={k:af[k] for k in ('chord_free','chord_C','best_C','attempts','successes','batch')}
   operation=policy.value('operation',op_features)
   if operation not in (0,1):raise ValueError('operation must be0(shortcut) or1(RRT*)')
   spent_edges=sum(r['search_edges'] for r in history);spent_samples=sum(r['search_samples'] for r in history)
   qfeatures=dict(samples_left=left,edges_left=available_search,batches_left=MAX_BATCHES-step,batch=step,attempts=af['attempts'],edges_per_sample=max(1.,spent_edges/max(spent_samples,1)) if spent_samples else 64.)
   raw_sample=policy.value('sample_quota',qfeatures);raw_edges=policy.value('edge_quota',qfeatures)
   if raw_sample<0 or raw_edges<0:raise ValueError('nonnegative quotas required')
   quota=min(left,int(raw_sample));edge_quota=min(available_search,int(raw_edges))
   policy_trace.append(dict(batch=step,stop_features=stop_features,stop=False,priority_features=[arm_features(a,history,step,best_C,initial_C,len(arms)) for a in arms],
    priorities=priorities,selected_arm_id=arm['id'],operation_features=op_features,operation=operation,quota_features=qfeatures,raw_sample_quota=raw_sample,raw_edge_quota=raw_edges,sample_quota=quota,edge_quota=edge_quota))
   if operation==1 and (quota<1 or edge_quota<1):stop_reason='generated_zero_quota';break
   before=deepcopy(session.work.summary());old_score=best_C;local=[]
   if operation==0:local=[base[i],base[j]]
   else:
    interval=[f'a{i}',f'a{j}'];lib=OperatorLibrary(ctx);boxes=lib.corridors(interval,side='both',scale='local');sampling=lib.sampling(boxes) if boxes else lib.uniform()
    repeat=sum(r['arm_id']==arm['id'] and r['operation']==1 for r in history)
    purpose='::'.join((canonical_path(base),arm['id'],f'repeat{repeat}'));seed=paired_seed(scene_seed,snapshot.time_seconds,STREAM_VERSION,purpose)
    prior_reserve=session.work.reserved_edges;session.work.reserved_edges+=reserve
    search_problem.budget=EdgeSlice(session.work,edge_quota);search_problem.cache={}
    try:plan,stats=repair_rrt_star(search_problem,arm['start_m'],arm['end_m'],quota,seed,sampling)
    finally:session.work.reserved_edges=prior_reserve
    local=plan.path
    session.searches.append(dict(purpose='generated_reconnect_rrt',actual_seed=seed,seed_stage=STREAM_VERSION,seed_purpose=purpose,arm_id=arm['id'],batch=step,
     search_objective='power_time_C',batch_edge_quota=edge_quota,batch_sample_quota=quota,sampling=deepcopy(sampling),sampling_counts=stats,budget_before=before,budget_after=session.work.summary(),**asdict(plan)))
   after_search=deepcopy(session.work.summary());path=anchors(base[:i]+local+base[j+1:],80.) if local else []
   balance=None;metrics=None;failure=None;duplicate=None
   try:
    if not path:failure='no_path_within_batch'
    elif not session.endpoints(path):failure='route_endpoints_invalid'
    elif path_length(path)>maximum+1e-8:failure='energy_or_mission_length_exceeded'
    else:
     metrics=session.problem.route_metrics(path)
     if metrics is None:failure='geometry_invalid'
     elif session.work.edge_limit-session.work.edges<len(path)-1:failure='insufficient_final_verification_budget'
     else:
      balance=power_time_score(session.problem,path,session.speed,settings);score_nodes+=balance['field_nodes']
      duplicate=next((r['id'] for r in session.records if r['feasible'] and canonical_path(r['path'])==canonical_path(path)),None)
      if duplicate is None:
       risk=prefix_risk(session.problem,path,session.speed);score_nodes+=risk['field_nodes']
       session.records.append(dict(id=f'generated_candidate_{step}',path=path,feasible=True,failure=None,metrics=metrics,prefix=risk,balance=balance,arm_id=arm['id']))
   except BudgetExhausted as exc:failure=str(exc);balance=None
   selected=choose(session.records);accepted=canonical_path(selected['path'])!=canonical_path(session.best)
   if accepted:session.best=deepcopy(selected['path']);session.best_metrics=deepcopy(selected['metrics']);session.accepted_proposals+=1
   session.work.reserved_edges=max(session.work.reserved_edges,len(session.best)-1)
   trace=dict(batch=step,arm_id=arm['id'],start_index=i,end_index=j,operation=operation,
    search_edges=after_search['edges']-before['edges'],search_samples=after_search['samples']-before['samples'],edge_quota=edge_quota,sample_quota=quota,
    candidate_C_seconds=balance['score_seconds'] if balance else None,candidate_failure=failure,duplicate_of=duplicate,
    previous_best_C_seconds=old_score,best_C_seconds=selected['balance']['score_seconds'],accepted=accepted,
    resources_before=before,resources_after=session.work.summary())
   history.append(trace);candidate_records.append(dict(batch=step,arm_id=arm['id'],path=path,feasible=failure is None,failure=failure,metrics=metrics,balance=balance,duplicate_of=duplicate))
   session.trace.append(dict(error=None,feedback=dict(accepted=accepted,reason=failure or ('improved' if accepted else 'no_objective_improvement'))))
 except (ValueError,TypeError,BudgetExhausted) as exc:
  error=f'{type(exc).__name__}: {exc}';errors.append(error);stop_reason='policy_error'
  session.trace.append(dict(error=error,feedback=dict(accepted=False,reason='policy_error')))
 session.work.reserved_edges=0
 for a,b in zip(session.best,session.best[1:]):
  session.work.take_edge();session.work.verification_edges+=1
  if not snapshot.geometry.edge_free(a,b):raise ValueError('final generated geometry invalid')
 selected=choose(session.records)
 return dict(contract_version=VERSION,path=session.best,candidates=session.records,generated_program=deepcopy(policy.program),
  expression_trace=policy_trace,build_trace=build,generated_trace=history,generated_candidates=candidate_records,
  reconnection_options=arms,frozen_base_route=base,trace=session.trace,searches=session.searches,resources=session.work.summary(),
  initial_resources=initial,starting_resources_already_paid=deepcopy(pool_report['resources']),selected_candidate_id=selected['id'],selected_balance=selected['balance'],selected_prefix=selected['prefix'],best_metrics=selected['metrics'],
  final_geometry_verified=True,policy_errors=len(errors),policy_error_details=errors,actual_advisor_calls=0,model_calls=0,new_measurements=0,
  score_field_nodes=score_nodes,grid_nodes_built=search_problem.grid_nodes_built,search_field_nodes=search_problem.field_nodes,
  numeric_compute_seconds=time.perf_counter()-started,stop_reason=stop_reason,arbitrary_python_executed=False,search_sample_cap=POST_SAMPLE_CAP)


class GeneratedOptimizer:
 policy=None
 last_call={}
 def __init__(self,scene_seed,program,settings=PowerTimeSettings(),limits=ObjectiveSettings()):
  self.base=BalancedOptimizer(scene_seed,settings,limits=limits);self.scene_seed,self.program,self.settings,self.limits=scene_seed,ExpressionPolicy(program).program,settings,limits
 def optimize(self,snapshot,context,route,max_length_m,seed,call_limit):
  started=time.perf_counter();result,report=self.base.optimize(snapshot,context,route,max_length_m,seed,0)
  if not result.path:return result,report
  maximum=min(max_length_m,report['mission_distance_cap_m']-report['executed_distance_m'])
  extra=run_generated(snapshot,context,report,maximum,self.scene_seed,self.program,self.settings,self.limits)
  searches=report['searches']+extra['searches'];trace=report['trace']+extra['trace'];nodes=report['score_field_nodes']+extra['score_field_nodes']
  report.update(extra,searches=searches,trace=trace,score_field_nodes=nodes,selection_rule='balanced',
   accepted_proposals=int(not route or canonical_path(route)!=canonical_path(extra['path'])),numeric_compute_seconds=time.perf_counter()-started)
  result.path=extra['path'];result.length_m=path_length(result.path);result.cost=extra['selected_balance']['score_seconds'];result.iterations=extra['resources']['samples'];result.elapsed_seconds=report['numeric_compute_seconds']
  return result,report
