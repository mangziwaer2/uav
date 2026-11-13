"""公开证据状态、有限修复过程与数值基线；不读取world，不执行生成Python。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import time
import numpy as np

from .planning import PlanResult, path_length
from .repair_program import OperatorLibrary, LibraryNumericalPolicy
from .repair_search import BudgetExhausted
from .route_repair import RepairSession, RepairSettings, anchors, parse_proposal

CONTRACT_VERSION = 'repair_process_v1'
OUTCOMES = ('improved','no_gain','geometry_failure','search_failure','resource_stop','duplicate','stopped','policy_error')
METRICS = ('whole','supported_interference','unknown_front','stale','detour','evidence')
ACTION_KEYS = {'operator','metric','padding','side','scale','allocation'}
MAP_KEYS = ('bounds_m','obstacles_m','clearance_m','geometry_mode','altitude_m',
            'vertical_clearance_m','known_buildings','geometry_meaning')
VEHICLE_KEYS = ('position_m','goal_m','altitude_m','speed_mps','remaining_energy_wh','reserve_energy_wh')


def _subset(value, keys):
    return {key:deepcopy(value[key]) for key in keys if key in value}


def route_evidence(problem, route):
    """当前内部GP的沿路支持摘要；均值不外泄，非新的测量或计费搜索。"""
    length=known=age_sum=sd_sum=0.;segments=[]
    for i,(a,b) in enumerate(zip(route,route[1:])):
        weights,nodes=problem.quadrature(a,b);s=problem.snapshot
        mask=s.supported[nodes[:,1],nodes[:,0]];ages=s.age_seconds[nodes[:,1],nodes[:,0]]
        sd=s.uncertainty_db[nodes[:,1],nodes[:,0]]
        total=float(weights.sum());supported=float(weights@mask)
        age=float(weights[mask]@ages[mask]/supported) if supported>0 else None
        segments.append({'index':i,'supported_fraction':supported/max(total,1e-12),
                         'supported_age_seconds':age,'mean_std_db':float(weights@sd/max(total,1e-12))})
        length+=total;known+=supported;age_sum+=supported*(age or 0.);sd_sum+=float(weights@sd)
    return {'length_m':length,'supported_fraction':known/max(length,1e-12),
            'unknown_fraction':1.-known/max(length,1e-12),
            'mean_supported_age_seconds':age_sum/known if known else None,
            'mean_std_db':sd_sum/max(length,1e-12),'segments':segments,
            'origin':'same_frozen_gp_line_integrals_not_new_measurements'}


@dataclass(frozen=True)
class RepairState:
    """冻结字段合同；to_dict返回独立副本。附带的公开context用于已有数值库。"""
    data: dict

    @classmethod
    def from_context(cls, context, previous_process_feedback=()):
        r=context['route_repair'];now=r['snapshot_time_seconds'];em=context['electromagnetic_estimate']
        public_em=_subset(em,('interference_reference_dbm','sample_columns','coarse_samples','local_samples','estimator'))
        columns=public_em['sample_columns'];allowed_columns=('x_m','y_m','estimate_dbm','uncertainty_db','supported',
            'nearest_support_age_seconds','nearest_support_distance_m','local_training_point_count')
        if any(c not in allowed_columns for c in columns):raise ValueError('noncontract EM sample column')
        for field in ('coarse_samples','local_samples'):
            sanitized=[]
            for row in public_em.get(field,[]):
                entry=dict(zip(columns,row))
                if not entry['supported']:entry['estimate_dbm']=None
                sanitized.append([entry[c] for c in columns])
            public_em[field]=sanitized
        evidence=_subset(context.get('measurement_evidence',{}),('origin','recent_points','power_noise_std_db'))
        evidence['recent_points']=[_subset(p,('position_m','time_seconds','interference_dbm'))
                                   for p in evidence.get('recent_points',[])]
        if any(p['time_seconds']>now+1e-8 for p in evidence['recent_points']):
            raise ValueError('future measurement in RepairState')
        # Only current-route anchors and historical numeric feedback, never arbitrary context modules.
        repair=_subset(r,('round','rounds_left','route_version','snapshot_time_seconds','incumbent_feasible',
            'anchors','segments','best_metrics','initial_metrics','fixed_objective','maximum_flight_length_m','resources'))
        public={'map':_subset(context['map'],MAP_KEYS),'vehicle':_subset(context['vehicle'],VEHICLE_KEYS),
                'electromagnetic_estimate':public_em,'measurement_evidence':evidence,'route_repair':repair}
        # Previous feedback is reduced to explicit numeric contract fields.
        repair['previous_feedback']=[_subset(f,('operator','accepted','reason','candidate_metrics',
            'previous_best_metrics','best_metrics','resources','candidate_minus_previous_best',
            'route_version_before','route_version_after')) for f in r.get('previous_feedback',[])]
        state={'contract_version':CONTRACT_VERSION,'snapshot_time_seconds':now,'route_version':r['route_version'],
               'public_context':public,'task_objective':'unchanged_current_gp_J_not_future_exposure',
               'no_inference_of_true_source':True,'new_observations_within_process':0,
               'previous_process_feedback':[RepairFeedback.from_dict(f).to_dict() for f in previous_process_feedback]}
        json.dumps(state,allow_nan=False)
        return cls(state)

    def to_dict(self):return deepcopy(self.data)


@dataclass(frozen=True)
class RepairFeedback:
    """可供后续选择器读取的数值反馈；拒绝加入事后真值评价字段。"""
    data: dict

    @classmethod
    def from_dict(cls, value):
        fields={'contract_version','candidate_id','outcome','accepted','reason','candidate_metrics','best_metrics',
            'candidate_evidence','duplicate_of','work_delta','resources','route_version_before','route_version_after',
            'no_new_observations','no_inference_of_true_source','future_exposure_evaluation_included'}
        if set(value)!=fields or value['contract_version']!=CONTRACT_VERSION or value['outcome'] not in OUTCOMES:
            raise ValueError('invalid feedback schema')
        if value['no_new_observations'] is not True or value['no_inference_of_true_source'] is not True or value['future_exposure_evaluation_included'] is not False:
            raise ValueError('feedback violates information boundary')
        for key in ('candidate_metrics','best_metrics'):
            if value[key] is not None and set(value[key])!={'length_m','interference_cost','uncertainty_cost','objective'}:
                raise ValueError('noncontract metric')
        if type(value['accepted']) is not bool or any(v<0 for v in value['work_delta'].values()):
            raise ValueError('invalid acceptance/resource feedback')
        if value['route_version_after']<value['route_version_before']:raise ValueError('route version regressed')
        json.dumps(value,allow_nan=False)
        return cls(deepcopy(value))

    def to_dict(self):return deepcopy(self.data)


def canonical_path(route):
    """消除共线插点后的路径指纹，仅去除同向插点，不抹掉折返。"""
    result=[]
    for p in route:
        p=np.array(p,float)
        if result and np.linalg.norm(p-result[-1])<1e-9:continue
        while len(result)>1:
            first=result[-1]-result[-2];second=p-result[-1]
            cross=first[0]*second[1]-first[1]*second[0]
            if abs(cross)>1e-8*max(1.,np.linalg.norm(first)*np.linalg.norm(second)) or first@second<0:break
            result.pop()
        result.append(p)
    value=[[round(float(v),8) for v in p] for p in result]
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def select_interval(context, metric='evidence', padding=1):
    """已支持干扰为零时明确回退，不将先验低均值称作安全。"""
    if metric not in METRICS or type(padding) is not int or not 0<=padding<=3:
        raise ValueError('invalid interval selector')
    lib=OperatorLibrary(context);r=context['route_repair'];lib.whole();segments=r['segments']
    risk=[s['cost']['interference_cost']/max(s['cost']['length_m'],1.)*s['supported_fraction'] for s in segments]
    xy=np.array([a['position_m'] for a in r['anchors']])
    detours=[float(np.linalg.norm(xy[k+1]-xy[k])+np.linalg.norm(xy[k+2]-xy[k+1])-np.linalg.norm(xy[k+2]-xy[k]))
             for k in range(max(0,len(xy)-2))]
    requested=metric;fallback=None
    if metric=='evidence':
        metric='supported_interference' if max(risk,default=0)>1e-8 else ('detour' if max(detours,default=0)>1e-6 else 'unknown_front')
    if metric=='supported_interference' and max(risk,default=0)<=1e-8:
        metric='unknown_front';fallback='no_supported_interference_signal'
    if metric=='whole':interval=lib.whole()
    elif metric=='detour':interval=lib.select_interval('detour',padding)
    else:
        if metric=='supported_interference':scores=risk
        elif metric=='unknown_front':scores=[(1-s['supported_fraction'])/(1+.25*s['index']) for s in segments]
        else:scores=[s['supported_fraction']*float(s['mean_supported_age_seconds'] or 0.) for s in segments]
        i=int(np.argmax(scores));j=i+1
        interval=[r['anchors'][max(0,i-padding)]['id'],r['anchors'][min(len(xy)-1,j+padding)]['id']]
    return interval,{'requested_metric':requested,'resolved_metric':metric,'fallback':fallback,
        'supported_interference_discriminative':max(risk,default=0)>1e-8,
        'all_segment_interference_scores':risk,'interval':interval,'not_safe_direction_or_source_bearing':True}


def allocate(context, mode, slots_left):
    r=context['route_repair'];work=r['resources'];left=work['samples_left'];edges=work['edges_left']
    if mode not in ('fixed','feedback'):raise ValueError('invalid allocation mode')
    if edges<32 or left<2:return 0,{'reason':'insufficient_shared_resources','samples':0}
    previous=r['previous_feedback'];ratio=64.
    if mode=='feedback' and previous:
        # Marginal search cost, rather than cached initial/final route validation, informs allocation.
        for a,b in zip([{'resources':{'samples':0,'edges':0}}]+previous[:-1],previous):
            ds=b['resources']['samples']-a['resources']['samples'];de=b['resources']['edges']-a['resources']['edges']
            if ds>0:ratio=max(16.,de/ds)
    slots=max(1,slots_left)
    n=min(left,350) if mode=='fixed' else min(left,700,max(2,int((edges-16)/(ratio*slots))))
    return int(n),{'mode':mode,'samples':int(n),'estimated_edges_per_sample':ratio,
                   'planned_remaining_search_slots':slots,'shared_resources':deepcopy(work)}


def validate_process(program):
    """最多6节点、只向前跳转；无自由条件表达式、循环或Python执行。"""
    if not isinstance(program,dict) or set(program)!={'version','nodes'} or program['version']!=CONTRACT_VERSION:
        raise ValueError('process schema/version invalid')
    nodes=program['nodes']
    if not isinstance(nodes,list) or not 1<=len(nodes)<=6:raise ValueError('process needs1..6 nodes')
    for i,node in enumerate(nodes):
        if not isinstance(node,dict) or set(node)!={'action','next'}:raise ValueError('invalid process node')
        action=node['action'];transitions=node['next']
        if not isinstance(action,dict) or set(action)!=ACTION_KEYS:raise ValueError('invalid action keys')
        if action['operator'] not in ('shortcut','local_rrt','via','global_rrt','keep'):raise ValueError('invalid operator')
        if action['metric'] not in METRICS or type(action['padding']) is not int or not 0<=action['padding']<=3:
            raise ValueError('invalid selector')
        if action['side'] not in ('both','left','right','tube'):raise ValueError('invalid side')
        if action['operator']=='via' and action['side'] not in ('left','right'):raise ValueError('via requires explicit geometric side')
        if action['scale'] not in ('near','local','wide') or action['allocation'] not in ('fixed','feedback'):
            raise ValueError('invalid construction/allocation')
        if not isinstance(transitions,dict) or 'default' not in transitions or not set(transitions)<=(set(OUTCOMES)|{'default'}):
            raise ValueError('invalid transitions')
        for target in transitions.values():
            if target!='stop' and (type(target) is not int or not i<target<len(nodes)):
                raise ValueError('only forward bounded transitions allowed')
    return deepcopy(program)


def baseline_process(adaptive=True):
    def action(op,metric='evidence',padding=1,side='both',scale='local'):
        return dict(operator=op,metric=metric,padding=padding,side=side,scale=scale,allocation='feedback' if adaptive else 'fixed')
    actions=[action('shortcut','whole'),action('local_rrt'),action('local_rrt',padding=3,scale='wide'),
             action('via',padding=3,side='left'),action('via',padding=3,side='right'),action('global_rrt','whole')]
    nodes=[{'action':a,'next':{'default':i+1 if i+1<len(actions) else 'stop','resource_stop':'stop'}} for i,a in enumerate(actions)]
    if adaptive:
        # Failure enlarges the interval; a solved-but-no-gain search changes geometric construction.
        nodes[1]['next'].update(improved=5,no_gain=3,duplicate=3)
        nodes[2]['next'].update(improved=5)
        nodes[3]['next'].update(improved=5)
    return validate_process({'version':CONTRACT_VERSION,'nodes':nodes})


def _proposal(state, action, slots_left):
    context=state.to_dict()['public_context'];lib=OperatorLibrary(context);r=context['route_repair']
    op=action['operator'];meta={}
    if op=='keep':return lib.keep(),meta
    if not r['incumbent_feasible']:
        n,allocation=allocate(context,action['allocation'],1)
        return (lib.global_rrt(lib.uniform(),n) if n else None),{'bootstrap':True,'allocation':allocation}
    interval,selection=select_interval(context,action['metric'],action['padding']);meta['selection']=selection
    if op=='shortcut':return lib.shortcut(interval),meta
    n,allocation=allocate(context,action['allocation'],slots_left);meta['allocation']=allocation
    if not n:return None,meta
    boxes=lib.corridors(interval,side=action['side'],scale=action['scale'])
    distribution=lib.sampling(boxes) if boxes else lib.uniform()
    meta['construction']={'boxes':boxes,'side':action['side'],'scale':action['scale'],
                          'meaning':'geometric_search_preference_not_measured_safe_region'}
    if op=='global_rrt':return lib.global_rrt(distribution,n),meta
    if op=='local_rrt':return lib.local_rrt(interval,distribution,n),meta
    point=lib.waypoint(interval,side=action['side'],scale=action['scale'])
    return lib.via(interval,point,distribution,n),meta


def _candidate(proposal, base, searches, spacing):
    op=proposal['operator']
    if op=='keep':return []
    if op=='global_rrt':return anchors(searches[0]['path'],spacing) if searches else []
    i,j=(int(proposal[k][1:]) for k in ('start_id','end_id'))
    if op=='shortcut':local=[base[i],base[j]]
    elif op=='local_rrt':local=searches[0]['path'] if searches else []
    else:local=(searches[0]['path']+searches[1]['path'][1:] if len(searches)==2 and all(s['path'] for s in searches) else [])
    return anchors(base[:i]+local+base[j+1:],spacing) if local else []


def feedback_outcome(feedback, duplicate_of=None):
    if feedback['accepted']:return 'improved'
    if duplicate_of is not None:return 'duplicate'
    reason=feedback['reason']
    if reason in ('geometry_invalid','waypoint_intersects_geometry','route_endpoints_invalid'):return 'geometry_failure'
    if reason in ('edge_budget_exhausted','sample_budget_exhausted','insufficient_final_verification_budget','insufficient_shared_resources'):return 'resource_stop'
    if reason=='no_objective_improvement':return 'no_gain'
    if reason=='kept_incumbent':return 'stopped'
    if reason=='policy_error':return 'policy_error'
    return 'search_failure'


class ProcessRepairOptimizer:
    """人工有限过程的数值控制；policy=None，不将其标为模型决策。"""
    policy=None
    last_call={}

    def __init__(self, method='adaptive_process', program=None, settings=RepairSettings()):
        if method not in ('library_numeric','fixed_process','adaptive_process'):raise ValueError('unknown numeric process method')
        settings.validate();self.settings=settings;self.method=method
        self.program=validate_process(program) if program is not None else baseline_process(method=='adaptive_process')

    def optimize(self,snapshot,context,route,max_length_m,seed,call_limit):
        started=time.perf_counter();session=RepairSession(snapshot,context,route,max_length_m,self.settings,seed)
        candidates=[];seen={};process_trace=[];node_id=0;initial_missing=session.best is None
        if session.best:
            seen[canonical_path(session.best)]='incumbent'
            candidates.append({'id':'incumbent','path':deepcopy(session.best),'metrics':deepcopy(session.best_metrics),
                               'evidence':route_evidence(session.problem,session.best),'duplicate_of':None})
        for step in range(1,7):
            ctx=session.request(step,6)
            state=RepairState.from_context(ctx,[t['feedback'] for t in process_trace]);node=self.program['nodes'][node_id]
            before=session.work.summary();base=deepcopy(session.best or []);search_count=len(session.searches)
            construction={};error=None;raw=None;proposal=None;duplicate=None;candidate=[]
            try:
                if self.method=='library_numeric':
                    raw=LibraryNumericalPolicy().decide(state.to_dict()['public_context']);proposal=parse_proposal(raw,ctx)
                else:
                    slots=sum(n['action']['operator'] not in ('keep','shortcut') for n in self.program['nodes'][node_id:])
                    proposal,construction=_proposal(state,node['action'],slots)
                    if proposal is not None:proposal=parse_proposal(proposal,ctx);raw=json.dumps(proposal,ensure_ascii=False)
                if proposal is None:
                    feedback={'accepted':False,'reason':'insufficient_shared_resources','resources':session.work.summary(),
                              'best_metrics':deepcopy(session.best_metrics)}
                else:
                    if proposal['operator']=='shortcut':
                        candidate=_candidate(proposal,base,[],self.settings.anchor_spacing_m)
                        duplicate=seen.get(canonical_path(candidate))
                    if duplicate is not None:
                        feedback={'operator':proposal['operator'],'accepted':False,'reason':'duplicate_candidate',
                            'candidate_metrics':None,'previous_best_metrics':deepcopy(session.best_metrics),
                            'best_metrics':deepcopy(session.best_metrics),'resources':session.work.summary()}
                    else:
                        feedback=session.apply(proposal)
                        candidate=_candidate(proposal,base,session.searches[search_count:],self.settings.anchor_spacing_m)
                        duplicate=seen.get(canonical_path(candidate)) if candidate else None
            except (ValueError,TypeError,BudgetExhausted) as exc:
                error=f'{type(exc).__name__}: {exc}'
                feedback={'accepted':False,'reason':'policy_error','error':error,'resources':session.work.summary(),
                          'best_metrics':deepcopy(session.best_metrics)}
            identifier=f'candidate_{step}' if candidate else None
            if candidate:
                fingerprint=canonical_path(candidate);seen.setdefault(fingerprint,identifier)
                candidates.append({'id':identifier,'path':candidate,'metrics':feedback.get('candidate_metrics'),
                    'evidence':route_evidence(session.problem,candidate) if feedback.get('candidate_metrics') is not None else None,
                    'duplicate_of':duplicate,'fingerprint':fingerprint,'accepted':bool(feedback['accepted'])})
            outcome=feedback_outcome(feedback,duplicate)
            delta={key:session.work.summary()[key]-before[key] for key in ('samples','edges','cache_hits')}
            structured={'contract_version':CONTRACT_VERSION,'candidate_id':identifier,'outcome':outcome,
                'accepted':bool(feedback['accepted']),'reason':feedback['reason'],'candidate_metrics':feedback.get('candidate_metrics'),
                'best_metrics':deepcopy(session.best_metrics),'candidate_evidence':candidates[-1]['evidence'] if candidate else None,
                'duplicate_of':duplicate,'work_delta':delta,'resources':session.work.summary(),
                'route_version_before':state.data['route_version'],'route_version_after':session.version,
                'no_new_observations':True,'no_inference_of_true_source':True,
                'future_exposure_evaluation_included':False}
            structured=RepairFeedback.from_dict(structured).to_dict()
            session.trace.append({'round':step,'context':ctx,'raw_response':raw,'proposal':proposal,
                                  'error':error,'model_runtime':{},'advisor_seconds':0.,'feedback':feedback})
            next_node=node['next'].get(outcome,node['next']['default'])
            if self.method=='library_numeric':next_node='stop' if proposal and proposal['operator']=='keep' else (step if step<6 else 'stop')
            process_trace.append({'step':step,'node':node_id,'state':state.to_dict(),'construction':construction,
                                  'feedback':structured,'next':next_node})
            if outcome in ('resource_stop','stopped','policy_error') or next_node=='stop':break
            node_id=next_node
        verified=session.best is not None
        if session.best:
            for a,b in zip(session.best,session.best[1:]):
                if session.work.edges>=session.work.edge_limit:raise RuntimeError('final verification exceeds shared cap')
                session.work.edges+=1;session.work.verification_edges+=1
                verified=session.problem.geometry.edge_free(a,b) and verified
            session.work.reserved_edges=0
            if not verified:raise RuntimeError('invalid final process route')
        elapsed=time.perf_counter()-started
        result=PlanResult('rrt_star',path=session.best or [],cost=session.best_metrics['objective'] if session.best_metrics else None,
            length_m=path_length(session.best) if session.best else None,reason='process_best_feasible' if session.best else 'process_no_feasible_route',
            iterations=session.work.samples,iteration_unit='sample_proposals',elapsed_seconds=elapsed)
        report={'schema_version':1,'contract_version':CONTRACT_VERSION,'method':self.method,
            'snapshot_time_seconds':snapshot.time_seconds,'objective':'fixed_balanced_exact_grid_integral',
            'initial_metrics':session.initial_metrics,'best_metrics':session.best_metrics,'trace':session.trace,
            'searches':session.searches,'resources':session.work.summary(),'actual_advisor_calls':0,'advisor_failures':0,
            'policy_errors':sum(t['error'] is not None for t in session.trace),'advisor_seconds':0.,'numeric_compute_seconds':elapsed,
            'numeric_recovery':initial_missing,'call_budget_exhausted':False,'accepted_proposals':session.accepted_proposals,
            'accepted_model_proposals':0,'accepted_recovery_proposals':int(initial_missing and session.best is not None),
            'has_final_route':bool(session.best),'final_geometry_verified':verified,'initial_validation_failure':session.validation_failure,
            'process_program':deepcopy(self.program),'process_trace':process_trace,'candidates':candidates,
            'stop_reason':process_trace[-1]['feedback']['outcome'] if process_trace else 'empty',
            'max_process_actions':6,'arbitrary_python_executed':False,'new_observations':0}
        return result,report
