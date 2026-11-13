"""冻结快照上的区段修复、数值反馈与统一接受器；GP和任务目标不由LLM修改。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import time
import numpy as np
from .agent_navigation import FileDecisionAdvisor, strict_json, visible_context
from .planning import PlanResult, path_length
from .repair_search import BudgetExhausted, WorkBudget, FixedObjective, repair_rrt_star, validate_sampling

UNIFORM={'focus_probability':0.,'regions':[]}
OPERATORS=('keep','shortcut','local_rrt','via','global_rrt')


@dataclass(frozen=True)
class RepairSettings:
    """每决策总预算；跨轮次/双腿/恢复共用。间距m仅生成连接点，不是测量范围。"""
    max_rounds: int = 3
    sample_limit: int = 2100
    edge_limit: int = 10000
    anchor_spacing_m: float = 80.

    def validate(self):
        if type(self.max_rounds) is not int or not 1<=self.max_rounds<=3:
            raise ValueError('max_rounds must be an integer in [1,3]')
        WorkBudget(self.sample_limit,self.edge_limit)
        if isinstance(self.anchor_spacing_m,bool) or not math.isfinite(self.anchor_spacing_m) or self.anchor_spacing_m<=0:
            raise ValueError('positive anchor spacing required')


def anchors(route, spacing):
    """保留每个原始拐点；长边等距插点，不裁掉中间几何信息。"""
    if not route:return []
    result=[list(route[0])]
    for a,b in zip(route,route[1:]):
        a,b=np.asarray(a,float),np.asarray(b,float);length=float(np.linalg.norm(b-a))
        if length<1e-8:continue
        count=max(1,int(math.ceil(length/spacing)))
        result.extend((a+(b-a)*j/count).tolist() for j in range(1,count+1))
    return result


def parse_proposal(raw,context):
    p=strict_json(raw) if isinstance(raw,str) else deepcopy(raw)
    required={'route_version','operator','reason'}
    allowed=required|{'start_id','end_id','waypoint_m','sample_budget','sampling'}
    if not isinstance(p,dict) or not required<=set(p)<=allowed:
        raise ValueError('proposal fields invalid')
    repair=context['route_repair'];op=p['operator']
    if type(p['route_version']) is not int or p['route_version']!=repair['route_version']:
        raise ValueError('stale route_version')
    if not isinstance(op,str) or op not in OPERATORS:raise ValueError('unsupported operator')
    if not isinstance(p['reason'],str) or not p['reason'].strip() or len(p['reason'])>600:
        raise ValueError('short nonempty reason required')
    if op=='keep':
        if set(p)!=required or not repair['incumbent_feasible']:raise ValueError('keep requires a feasible incumbent and no search fields')
        return p
    if op=='global_rrt':
        if 'start_id' in p or 'end_id' in p:raise ValueError('global search has no local anchor ids')
    else:
        ids={a['id']:i for i,a in enumerate(repair['anchors'])}
        if not repair['incumbent_feasible'] or p.get('start_id') not in ids or p.get('end_id') not in ids or ids[p['start_id']]>=ids[p['end_id']]:
            raise ValueError('local repair requires ordered ids of a feasible current route')
    if op=='via':
        point=p.get('waypoint_m')
        if not isinstance(point,list) or len(point)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in point):
            raise ValueError('via needs one finite waypoint')
        high=np.asarray(context['map']['bounds_m'][1]);point=np.asarray(point)
        if np.any(point<0) or np.any(point>high):raise ValueError('waypoint outside map')
    elif 'waypoint_m' in p:raise ValueError('waypoint only applies to via')
    if op=='shortcut':
        if any(k in p for k in ('sampling','sample_budget')):raise ValueError('shortcut uses no samples')
    else:
        n=p.get('sample_budget');left=repair['resources']['samples_left']
        if type(n) is not int or not (2 if op=='via' else 1)<=n<=left:
            raise ValueError('sample_budget exceeds remaining resources or too small')
        p['sampling']=validate_sampling(p.get('sampling'),context['map']['bounds_m'][1])
    return p


def repair_response(value,identifier,context):
    if not isinstance(value,dict) or set(value)!={'request_id','decision','strategy','memory'} or value['request_id']!=identifier:
        raise ValueError('response identity/schema mismatch')
    if value['strategy']!={'mode':'baseline'}:raise ValueError('no external strategy modules')
    memory=value['memory']
    if not isinstance(memory,str) or len(memory)>1200:raise ValueError('memory too long')
    proposal=parse_proposal(value['decision'],context)
    return json.dumps(proposal,ensure_ascii=False,allow_nan=False),{'mode':'baseline'},memory


class FileRepairPolicy(FileDecisionAdvisor):
    """文件咨询真实在线LLM；本类不产生固定模型响应。"""
    def __init__(self,public_root,mission_id,timeout_seconds=600):
        super().__init__(public_root,mission_id,timeout_seconds,validator=repair_response)


class NumericalRepairPolicy:
    """同接口的固定数值操作选择，仅为开发对照，不是LLM或强基线已证优。"""
    last_call={}
    def decide(self,ctx):
        r=ctx['route_repair'];round_id=r['round'];left=r['resources']['samples_left']
        p={'route_version':r['route_version'],'operator':'keep','reason':'固定数值操作选择'}
        if left==0 and r['incumbent_feasible']:
            return json.dumps(p,ensure_ascii=False)
        if not r['incumbent_feasible'] or round_id==1:
            p.update(operator='global_rrt',sample_budget=min(left,max(1,left//(r['rounds_left']+1))),sampling=deepcopy(UNIFORM))
        elif round_id==2 and len(r['anchors'])>2:
            # Pick the supported segment with highest extra cost per metre; no truth/direction.
            segments=r['segments'];seg=max(segments,key=lambda x:x['cost']['interference_cost']/max(x['cost']['length_m'],1))
            i=max(0,seg['index']-1);j=min(len(r['anchors'])-1,seg['index']+2)
            a=np.asarray(r['anchors'][i]['position_m']);b=np.asarray(r['anchors'][j]['position_m']);high=np.asarray(ctx['map']['bounds_m'][1])
            low=np.maximum(0,np.minimum(a,b)-120);upper=np.minimum(high,np.maximum(a,b)+120)
            p.update(operator='local_rrt',start_id=f'a{i}',end_id=f'a{j}',sample_budget=min(left,max(1,left//2)),
                     sampling={'focus_probability':.6,'regions':[{'low_m':low.tolist(),'high_m':upper.tolist(),'weight':1.}]})
        elif len(r['anchors'])>2:
            p.update(operator='shortcut',start_id='a0',end_id=f"a{len(r['anchors'])-1}")
        return json.dumps(p,ensure_ascii=False)


class RepairSession:
    """一份快照、一个目标、共享资源及当前最佳；id仅在route_version内有效。"""
    def __init__(self,snapshot,context,route,max_length_m,settings,seed):
        settings.validate();self.settings=settings;self.snapshot=deepcopy(snapshot);self.context=deepcopy(context)
        self.work=WorkBudget(settings.sample_limit,settings.edge_limit)
        self.problem=FixedObjective(self.snapshot,context['electromagnetic_estimate']['interference_reference_dbm'],self.work)
        self.start=list(context['vehicle']['position_m']);self.goal=list(context['vehicle']['goal_m'])
        self.max_length=max_length_m;self.seed=seed;self.version=0;self.searches=[];self.trace=[]
        self.best=None;self.best_metrics=None;self.accepted_proposals=0;self.validation_failure=None
        if route:
            candidate=anchors(route,settings.anchor_spacing_m)
            try:
                metrics=self.problem.route_metrics(candidate)
                if metrics is not None and metrics['length_m']<=max_length_m+1e-8 and self.endpoints(candidate):
                    if self.work.edge_limit-self.work.edges >= len(candidate)-1:
                        self.best,self.best_metrics=candidate,metrics
                        self.work.reserved_edges=len(candidate)-1
            except BudgetExhausted as exc:self.validation_failure=str(exc)
        self.initial_metrics=deepcopy(self.best_metrics)

    def endpoints(self,path):
        return bool(path and np.allclose(path[0],self.start,atol=1e-8,rtol=0) and np.allclose(path[-1],self.goal,atol=1e-8,rtol=0))

    def request(self,round_id,max_rounds):
        route=self.best or [];summary=[]
        for i,(a,b) in enumerate(zip(route,route[1:])):
            w,nodes=self.problem.quadrature(a,b);total=max(float(w.sum()),1e-12)
            mask=self.snapshot.supported[nodes[:,1],nodes[:,0]];age=self.snapshot.age_seconds[nodes[:,1],nodes[:,0]]
            values=self.snapshot.estimate_dbm[nodes[:,1],nodes[:,0]]
            summary.append({'index':i,'start_id':f'a{i}','end_id':f'a{i+1}',
                'cost':self.problem.edge(a,b),'supported_fraction':float(w@mask/total),
                'max_supported_mean_dbm':float(values[mask].max()) if mask.any() else None,
                'mean_supported_age_seconds':float(w[mask]@age[mask]/max(float(w[mask].sum()),1e-12)) if mask.any() else None})
        ctx=visible_context(self.context)
        # Preserve the measured evidence but suppress irrelevant old policy capability menus.
        for key in ('source_inference','source_refinement','profiles','waypoint_replanning','available_actions','probe_candidates','active_probe','stage_plan'):
            ctx.pop(key,None)
        ctx['route']={'points':[{'position_m':p} for p in route], 'fixed_metrics':deepcopy(self.best_metrics),
                      'field_at_arrival_times_predicted':False}
        ctx['route_repair']={'round':round_id,'rounds_left':max_rounds-round_id,'route_version':self.version,
            'snapshot_time_seconds':self.snapshot.time_seconds,'incumbent_feasible':self.best is not None,
            'anchors':[{'id':f'a{i}','position_m':p} for i,p in enumerate(route)],'segments':summary,
            'best_metrics':deepcopy(self.best_metrics),'initial_metrics':deepcopy(self.initial_metrics),
            'fixed_objective':{'length_weight':1.,'interference_weight':2.,'uncertainty_weight':.4,
                'reference_dbm':self.problem.reference,'integration':'exact_nearest_grid_line_integral',
                'not_future_truth_or_calibrated_safety':True},
            'maximum_flight_length_m':self.max_length,'resources':self.work.summary(),
            'operators':list(OPERATORS),'previous_feedback':[deepcopy(t['feedback']) for t in self.trace],
            'same_snapshot_no_new_measurements_within_rounds':True}
        return ctx

    def search(self,a,b,quota,sampling,purpose):
        # Reserve work for scoring the candidate and final verification. Searching
        # until the final edge would otherwise discard an already discovered route.
        incumbent_reserve=self.work.reserved_edges
        extra=min(128,max(0,(self.work.edge_limit-self.work.edges-incumbent_reserve)//2))
        self.work.reserved_edges=incumbent_reserve+extra
        before=self.work.summary()
        try:
            result,stats=repair_rrt_star(self.problem,a,b,quota,self.seed+len(self.searches),sampling)
        finally:
            self.work.reserved_edges=incumbent_reserve
        self.searches.append({'purpose':purpose,'sampling':deepcopy(sampling),'sampling_counts':stats,
            'budget_before':before,'budget_after':self.work.summary(),**asdict(result)})
        return result.path

    def apply(self,p):
        prior=deepcopy(self.best_metrics);version=self.version;candidate=None;failure=None;metrics=None
        op=p['operator'];base=self.best
        try:
            if op=='keep':return {'operator':op,'accepted':False,'reason':'kept_incumbent','best_metrics':prior,
                                   'resources':self.work.summary(),'route_version_before':version,'route_version_after':version}
            if op=='global_rrt':candidate=self.search(self.start,self.goal,p['sample_budget'],p['sampling'],'global_rrt')
            else:
                i=int(p['start_id'][1:]);j=int(p['end_id'][1:]);a,b=base[i],base[j]
                if op=='shortcut':local=[a,b]
                elif op=='local_rrt':local=self.search(a,b,p['sample_budget'],p['sampling'],'local_rrt')
                else:
                    via=p['waypoint_m']
                    if not self.problem.geometry.free(via):
                        failure='waypoint_intersects_geometry';local=[]
                    else:
                        n=p['sample_budget'];first=self.search(a,via,n//2,p['sampling'],'via_leg1')
                        second=self.search(via,b,n-n//2,p['sampling'],'via_leg2') if first else []
                        local=first+second[1:] if first and second else []
                candidate=base[:i]+local+base[j+1:] if local else []
            if not candidate:failure=failure or 'no_path_within_budget'
            elif not self.endpoints(candidate):failure='route_endpoints_invalid'
            else:
                candidate=anchors(candidate,self.settings.anchor_spacing_m)
                metrics=self.problem.route_metrics(candidate)
                if metrics is None:failure='geometry_invalid'
                elif metrics['length_m']>self.max_length+1e-8:failure='energy_budget_exceeded'
                elif self.work.edge_limit-self.work.edges<len(candidate)-1:failure='insufficient_final_verification_budget'
        except BudgetExhausted as exc:failure=str(exc)
        accepted=(failure is None and metrics is not None and (prior is None or metrics['objective']<prior['objective']-1e-7))
        if accepted:
            self.best=candidate;self.best_metrics=metrics;self.version+=1;self.accepted_proposals+=1
            self.work.reserved_edges=len(candidate)-1
        return {'operator':op,'accepted':accepted,'reason':failure or ('improved' if accepted else 'no_objective_improvement'),
                'candidate_metrics':metrics,'previous_best_metrics':prior,'best_metrics':deepcopy(self.best_metrics),
                'candidate_minus_previous_best':metrics['objective']-prior['objective'] if metrics is not None and prior is not None else None,
                'route_version_before':version,'route_version_after':self.version,'resources':self.work.summary(),
                'repaired_interval':[p.get('start_id'),p.get('end_id')] if op!='global_rrt' else None,
                'preserves_route_outside_interval':op in ('shortcut','local_rrt','via'),
                'no_new_measurements':True}


class RouteRepairOptimizer:
    def __init__(self,policy=None,settings=RepairSettings(),numeric_policy=None):
        if policy is not None and numeric_policy is not None:
            raise ValueError('choose external model policy or numeric policy')
        settings.validate();self.policy=policy;self.numeric_policy=numeric_policy;self.settings=settings;self.last_call={}

    def optimize(self,snapshot,context,route,max_length_m,seed,call_limit):
        started=time.perf_counter();session=RepairSession(snapshot,context,route,max_length_m,self.settings,seed)
        calls=0;failures=0;advisor_seconds=0.;recovery=False
        # Numeric control has no LLM calls. Its iterations use the same operation interface.
        policy=self.policy if self.policy is not None else (self.numeric_policy or NumericalRepairPolicy())
        rounds=min(self.settings.max_rounds,call_limit) if self.policy is not None else self.settings.max_rounds
        for round_id in range(1,rounds+1):
            if session.work.summary()["edges_left"]<=0 and session.best is None:break
            if session.work.samples>=session.work.sample_limit and session.best is None:break
            ctx=session.request(round_id,rounds);raw=None;proposal=None;error=None
            t=time.perf_counter()
            try:
                raw=policy.decide(deepcopy(ctx));proposal=parse_proposal(raw,ctx)
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}';failures+=1
            elapsed=time.perf_counter()-t
            if self.policy is not None:calls+=1;advisor_seconds+=elapsed
            runtime=deepcopy(getattr(policy,'last_call',{}))
            feedback=(session.apply(proposal) if error is None else {'accepted':False,'reason':'policy_error',
                       'error':error,'best_metrics':deepcopy(session.best_metrics),'resources':session.work.summary()})
            session.trace.append({'round':round_id,'context':ctx,'raw_response':raw,'proposal':proposal,
                                  'error':error,'model_runtime':runtime,'advisor_seconds':elapsed,'feedback':feedback})
            if error is None and proposal['operator']=='keep':break
        # Resource-limited numeric recovery only when no feasible incumbent exists.
        # It cannot receive fresh budgets after an error or policy-call exhaustion.
        if session.best is None and session.work.samples<session.work.sample_limit and session.work.summary()["edges_left"]>0:
            recovery=True
            p={'operator':'global_rrt','sample_budget':session.work.sample_limit-session.work.samples,'sampling':deepcopy(UNIFORM)}
            feedback=session.apply(p)
            session.trace.append({'round':None,'backend':'numeric_recovery','feedback':feedback,'proposal':p})
        self.last_call=deepcopy(getattr(policy,'last_call',{})) if self.policy is not None and calls else {}
        # Final geometry verification is independent of search edge-cache acceptance
        # and explicitly consumes the work reserved when the incumbent was accepted.
        verified=session.best is not None
        if session.best is not None:
            for a,b in zip(session.best,session.best[1:]):
                session.work.edges+=1;session.work.verification_edges+=1
                verified = session.problem.geometry.edge_free(a,b) and verified
            session.work.reserved_edges=0
            if not verified:raise RuntimeError('repair search returned invalid geometry')
        result=PlanResult('rrt_star' ,path=session.best or [],cost=session.best_metrics['objective'] if session.best_metrics else None,
            length_m=path_length(session.best) if session.best else None,reason='repair_best_feasible' if session.best else 'repair_no_feasible_route',
            iterations=session.work.samples,iteration_unit='sample_proposals',elapsed_seconds=time.perf_counter()-started-advisor_seconds)
        report={'schema_version':1,'snapshot_time_seconds':snapshot.time_seconds,'objective':'fixed_balanced_exact_grid_integral',
            'initial_metrics':session.initial_metrics,'best_metrics':session.best_metrics,'trace':session.trace,
            'searches':session.searches,'resources':session.work.summary(),'actual_advisor_calls':calls,'advisor_failures':failures if self.policy is not None else 0,'policy_errors':failures,
            'advisor_seconds':advisor_seconds,'numeric_compute_seconds':result.elapsed_seconds,
            'numeric_recovery':recovery,
            'call_budget_exhausted':self.policy is not None and call_limit<=0,'accepted_proposals':session.accepted_proposals,
            'accepted_model_proposals':sum(bool(t['feedback'].get('accepted')) for t in session.trace if t.get('backend')!='numeric_recovery') if self.policy is not None else 0,
            'accepted_recovery_proposals':sum(bool(t['feedback'].get('accepted')) for t in session.trace if t.get('backend')=='numeric_recovery'),
            'has_final_route':bool(session.best),'final_geometry_verified':verified,'initial_validation_failure':session.validation_failure}
        return result,report
