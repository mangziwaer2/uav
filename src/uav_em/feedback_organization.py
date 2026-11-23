"""Two bounded diagnostic source jobs with shared local task information.

No simulator, propagation truth, flight, observation or birth is available here.
All resulting compound worlds remain out of the active hypothesis bank.
"""
from copy import deepcopy
import json
import math

import numpy as np

from .event_inference import digest
from . import feedback_revision as tool


PACKET_FIELDS = {'version', 'time_seconds', 'history', 'history_sha256', 'models',
    'gp_soft_constraints', 'anchor_cache', 'known_map', 'state', 'source_receipt',
    'limitations', 'packet_id'}
MODES = ('static', 'translate', 'strength_change', 'translate_strength')
LIMITATIONS = [
    'Diagnostic candidates only: no birth, inherited weight, new evidence or executed route.',
    'Known history, soft points and seeds have already used this prefix; tail errors are retrospective.',
    'Local ambiguity is an allocation heuristic, not calibrated uncertainty or value of information.',
    'Frozen-weight substitution is conditional algebra, not candidate credibility or task benefit.'
]
COUNTERS = ('solver_calls', 'parameter_evaluations', 'objective_calls', 'gradient_calls', 'kernel_terms',
    'cache_hits', 'constraint_calls', 'constraint_jacobian_calls')


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('finite numeric scalar required')
    return float(value)


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError('exact public fields required')


def _array(value, shape):
    if not isinstance(value, list):
        raise ValueError('public numeric arrays must be lists')
    def walk(v):
        if isinstance(v, list):
            for child in v: walk(child)
        else: _finite(v)
    walk(value)
    a=np.asarray(value, float)
    if not a.size and 0 in shape:a=a.reshape(shape)
    if a.shape != tuple(shape): raise ValueError('numeric array shape mismatch')
    return a


def _sha(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('SHA256 required')


def bind(value, key='request_id'):
    result=deepcopy(value)
    result[key]=digest({k:v for k,v in result.items() if k!=key})
    return result


def validate_packet(packet):
    """Validate one already public prefix, never query an external provider."""
    try:
        _keys(packet, PACKET_FIELDS)
        if packet['version']!='D2-M8':raise ValueError('unsupported diagnostic packet version')
        _sha(packet['packet_id']);_sha(packet['history_sha256'])
        if digest({k:v for k,v in packet.items() if k!='packet_id'})!=packet['packet_id']:
            raise ValueError('packet content changed')
        now=_finite(packet['time_seconds'])
        if now<0 or not isinstance(packet['history'],list) or len(packet['history'])<4:
            raise ValueError('at least four causal observations required')
        if digest(packet['history'])!=packet['history_sha256']:raise ValueError('history binding changed')
        last=-math.inf
        for point in packet['history']:
            _keys(point, {'position_m','time_seconds','interference_dbm'})
            _array(point['position_m'],(2,));_finite(point['interference_dbm'])
            t=_finite(point['time_seconds'])
            if not 0<=t<=now or t<=last:raise ValueError('strict causal prefix required')
            last=t
        if last!=now:raise ValueError('history must end at current time')
        soft=packet['gp_soft_constraints']
        # The original GP constraint packet also contains descriptive metadata;
        # only these numerical public fields enter tool calculations.
        for k in ('position_m','time_seconds','mean_dbm','weights','strength','prefix_end_seconds','independent_evidence_points'):
            if k not in soft:raise ValueError('missing GP soft field')
        n=len(soft['time_seconds'])
        _array(soft['position_m'],(n,2));tt=_array(soft['time_seconds'],(n,))
        _array(soft['mean_dbm'],(n,));weights=_array(soft['weights'],(n,))
        if (soft['prefix_end_seconds']!=now or soft['independent_evidence_points']!=0
                or np.any(tt>now) or np.any(weights<0) or _finite(soft['strength'])<0
                or (n and not math.isclose(float(weights.sum()),1.,abs_tol=1e-10))):
            raise ValueError('causal GP soft constraints required')
        models=packet['models']
        if not isinstance(models,list) or not models:raise ValueError('source worlds required')
        allowed={'id','family','std_db','born_index','scored_points','reference_time_seconds',
            'bases','peaks_dbm','fit_mse','weight','log_weight'}
        required=allowed-{'scored_points','fit_mse','log_weight'}
        for model in models:
            if not isinstance(model,dict) or not required<=set(model)<=allowed:
                raise ValueError('source model public fields required')
            if (not isinstance(model['id'],str) or not model['id']
                    or model['family'] not in ('static','translate','strength_change','unknown_prior')
                    or _finite(model['std_db'])<0 or _finite(model['weight'])<0
                    or _finite(model['reference_time_seconds'])>now
                    or type(model['born_index']) is not int or not 0<=model['born_index']<len(packet['history'])):
                raise ValueError('invalid causal source world')
            if 'scored_points' in model and (type(model['scored_points']) is not int or model['scored_points']<0):
                raise ValueError('invalid source evidence count')
            if not isinstance(model['bases'],list) or len(model['bases'])!=len(model['peaks_dbm']):
                raise ValueError('misaligned source parameters')
            _array(model['peaks_dbm'],(len(model['bases']),))
            for basis in model['bases']:
                _keys(basis,{'center_m','scale_m','velocity_mps','rate_db_per_second'})
                _array(basis['center_m'],(2,));_array(basis['velocity_mps'],(2,))
                if _finite(basis['scale_m'])<=0:raise ValueError('positive basis scale required')
                _finite(basis['rate_db_per_second'])
        ids=[m['id'] for m in models]
        if len(set(ids))!=len(ids) or not math.isclose(sum(m['weight'] for m in models),1.,abs_tol=1e-10):
            raise ValueError('normalized distinct source worlds required')
        anchor=packet['anchor_cache']
        _keys(anchor,{'time_seconds','route_a','route_b','baseline_delta_c','worlds','history_sha256'})
        if anchor['time_seconds']!=now or anchor['history_sha256']!=packet['history_sha256']:
            raise ValueError('anchor prefix mismatch')
        _finite(anchor['baseline_delta_c']);_keys(anchor['worlds'],ids)
        for model in models:
            world=anchor['worlds'][model['id']]
            _keys(world,{'weight','seed_delta_q'})
            if world['weight']!=model['weight']:raise ValueError('anchor world weight mismatch')
            _finite(world['seed_delta_q'])
        for route in (anchor['route_a'],anchor['route_b']):
            _keys(route,{'id','queries','gp_weight','gp_fixed_power'})
            if not isinstance(route['id'],str) or not route['id']:raise ValueError('route ID required')
            q=route['queries'];_keys(q,{'position_m','time_seconds','dt_seconds','duration_seconds'})
            n=len(q['time_seconds']);_array(q['position_m'],(n,2))
            tt=_array(q['time_seconds'],(n,));dt=_array(q['dt_seconds'],(n,))
            gw=_array(route['gp_weight'],(n,));gp=_array(route['gp_fixed_power'],(n,))
            if (not n or np.any(tt<=now) or np.any(np.diff(tt)<=0) or np.any(dt<=0)
                    or np.any(gw<0) or np.any(gw>1) or np.any(gp < -1e-9)
                    or not math.isclose(float(dt.sum()),_finite(q['duration_seconds']),abs_tol=1e-8)):
                raise ValueError('invalid frozen future route cache')
        if anchor['route_a']['id']==anchor['route_b']['id']:raise ValueError('distinct anchor routes required')
        if (not isinstance(packet['known_map'],dict) or not isinstance(packet['state'],dict)
                or not isinstance(packet['source_receipt'],dict)):
            raise ValueError('known map and state required')
        digest(packet)  # also reject non-JSON/nonfinite metadata inputs
        return packet
    except (KeyError,TypeError,IndexError,OverflowError) as exc:
        raise ValueError('invalid public diagnostic packet') from exc


def _indices(modes):
    result=[]
    for i,mode in enumerate(modes):
        result.append(4*i)
        if mode in ('translate','translate_strength'):result.extend((4*i+1,4*i+2))
        if mode in ('strength_change','translate_strength'):result.append(4*i+3)
    return result


def _priority(analysis, modes, weight, gap):
    ix=_indices(modes);units=np.tile([4.,3.,3.,.5],len(modes))[ix]
    H=np.asarray(analysis['weighted_fit_jacobian'],float)[:,ix]*units
    g=np.asarray(analysis['task_gradient'],float)[ix]*units
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(g)):
        raise ValueError('finite task derivatives required')
    variance=float(g@np.linalg.solve(H.T@H+np.eye(len(ix)),g))
    ambiguity=math.sqrt(max(0.,variance))
    return ambiguity,weight*ambiguity/(1.+abs(gap))


def _seed_cost(model,packet):
    return tool.estimate_seed_analysis_terms(model,packet['history'],
        packet['gp_soft_constraints'],packet['anchor_cache'])


def build_options(packet):
    """Pay once for complete seed derivatives and share all mode allocations."""
    validate_packet(packet)
    reconstruction=packet['source_receipt'].get('cache_reconstruction_kernel_terms',0)
    if type(reconstruction) is not int or not 0<=reconstruction<=40000:
        raise ValueError('bounded cache reconstruction accounting required')
    options=[];excluded=[];analyses={};work=dict(kernel_terms=reconstruction,
        analysis_kernel_terms=0,cache_reconstruction_kernel_terms=reconstruction,total_kernel_terms=reconstruction,worlds=0)
    for seed in sorted(packet['models'],key=lambda m:(-m['weight'],m['id'])):
        reason=None
        if not 1<=len(seed['bases'])<=2:reason='requires_one_or_two_bases'
        elif seed['weight']<=0:reason='zero_weight'
        elif work['worlds']>=8:reason='common_world_limit'
        if reason is None:
            try:
                choices=tool.mechanism_options(seed)
                if min(seed['peaks_dbm']) < -140.:raise ValueError('seed amplitude below frozen domain')
                if _seed_cost(seed,packet)+work['kernel_terms']>40000:reason='common_kernel_limit'
            except (ValueError,TypeError) as exc:reason='ineligible_seed: '+str(exc)
        if reason is not None:
            excluded.append(dict(target_id=seed['id'],reason=reason));continue
        analysis=tool.analyze_seed(seed,packet['history'],packet['gp_soft_constraints'],packet['anchor_cache'])
        actual=analysis['resources']['kernel_terms']
        if type(actual) is not int or actual<0 or work['kernel_terms']+actual>40000:
            raise ValueError('common analysis exceeded its reserved complete-world allowance')
        work['kernel_terms']+=actual;work['analysis_kernel_terms']+=actual
        work['total_kernel_terms']=work['kernel_terms'];work['worlds']+=1;analyses[seed['id']]=analysis
        for modes in choices:
            ambiguity,priority=_priority(analysis,modes,seed['weight'],packet['anchor_cache']['baseline_delta_c'])
            options.append(dict(target_id=seed['id'],component_modes=list(modes),
                active_parameters=_indices(modes),local_ambiguity_seconds=ambiguity,priority=priority))
    options.sort(key=lambda x:(-x['priority'],x['target_id'],tuple(x['component_modes'])))
    value=dict(packet_id=packet['packet_id'],options=options,excluded_targets=excluded,
        analyses=analyses,resources=work,limitations=deepcopy(LIMITATIONS))
    return bind(value,'common_id')


def _unique(pairs):
    value={}
    for k,v in pairs:
        if k in value:raise ValueError('duplicate JSON key')
        value[k]=v
    return value


def keep_response(request,reason='no further diagnostic job'):
    return dict(request_id=request['request_id'],action='keep',target_id=None,
        component_modes=[],start=None,reason=reason)


def validate_response(request,raw):
    if digest({k:v for k,v in request.items() if k!='request_id'})!=request['request_id']:
        raise ValueError('request changed')
    if isinstance(raw,str):
        raw=json.loads(raw,object_pairs_hook=_unique,
            parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    _keys(raw,{'request_id','action','target_id','component_modes','start','reason'})
    try: digest(raw)
    except (TypeError,OverflowError) as exc:raise ValueError('finite JSON decision required') from exc
    if raw['request_id']!=request['request_id']:raise ValueError('stale decision')
    if not isinstance(raw['reason'],str) or not 1<=len(raw['reason'])<=1000:
        raise ValueError('bounded reason required')
    if raw['action']=='keep':
        if raw['target_id'] is not None or raw['component_modes']!=[] or raw['start'] is not None:
            raise ValueError('keep requires no job')
    elif raw['action']=='fit':
        if not isinstance(raw['component_modes'],list) or not raw['component_modes']:
            raise ValueError('mechanism list required')
        matched=next((x for x in request['common']['options'] if x['target_id']==raw['target_id']
            and x['component_modes']==raw['component_modes']),None)
        if matched is None:raise ValueError('target/modes outside common options')
        job_key=(raw['target_id'],tuple(raw['component_modes']))
        seen={(x['target_id'],tuple(x['component_modes'])) for x in request['coverage']}
        if raw['start']=='original':
            if job_key in seen:raise ValueError('original job already covered')
        elif raw['start']=='previous':
            c=request['continuation']
            if (request['phase']!='revise' or c is None or c['target_id']!=raw['target_id']
                    or c['component_modes']!=raw['component_modes']):
                raise ValueError('continuation requires previous legal candidate')
        else:raise ValueError('unknown diagnostic start')
    else:raise ValueError('unknown diagnostic action')
    return deepcopy(raw)


def numeric_response(request):
    seen={(x['target_id'],tuple(x['component_modes'])) for x in request['coverage']}
    untried=[x for x in request['common']['options'] if (x['target_id'],tuple(x['component_modes'])) not in seen]
    chosen=untried[0] if untried else None
    previous=request['continuation']
    if (previous is not None and previous['numeric_continue_eligible']
            and previous['priority']>(chosen['priority'] if chosen else -math.inf)):
        return dict(request_id=request['request_id'],action='fit',target_id=previous['target_id'],
            component_modes=deepcopy(previous['component_modes']),start='previous',
            reason='continue incomplete legal improvement with greater remaining local task ambiguity')
    if chosen is None:return keep_response(request)
    return dict(request_id=request['request_id'],action='fit',target_id=chosen['target_id'],
        component_modes=deepcopy(chosen['component_modes']),start='original',
        reason='highest untried shared local task ambiguity, stable target/mode tie break')


def _feedback_summary(fit,feedback):
    return dict(task=deepcopy(feedback),fit_objective=fit['fit_objective'],seed_objective=fit['seed_objective'],
        solver=deepcopy(fit.get('solver',{})),stop_reason=fit.get('stop_reason'),
        legal=fit.get('legal'),improved=fit.get('improved'),candidate_id=fit['model']['id'],
        candidate_is_active=False,independent_future_evidence=False)


def _continuation(plan,fit,feedback,packet):
    seed=next(m for m in packet['models'] if m['id']==plan['target_id'])
    ambiguity,priority=_priority(feedback,plan['component_modes'],seed['weight'],packet['anchor_cache']['baseline_delta_c'])
    trace=fit.get('solver',{})
    improved=fit['fit_objective']<fit['seed_objective']-1e-8
    unfinished=not trace.get('solver_success',False)
    return dict(target_id=plan['target_id'],component_modes=deepcopy(plan['component_modes']),
        local_ambiguity_seconds=ambiguity,priority=priority,
        numeric_continue_eligible=bool(improved and unfinished),legal_candidate_id=fit['model']['id'])


def request_for(packet,common,phase,coverage,feedback,continuation,resources):
    if phase not in ('probe','revise'):raise ValueError('unknown phase')
    value={k:deepcopy(packet[k]) for k in ('version','time_seconds','history','history_sha256',
        'models','gp_soft_constraints','anchor_cache','known_map','state','packet_id')}
    value.update(schema='feedback_revision_v1',phase=phase,
        common={k:deepcopy(common[k]) for k in ('common_id','options','excluded_targets','resources')},
        feedback=deepcopy(feedback),coverage=deepcopy(coverage),continuation=deepcopy(continuation),
        resources_so_far=deepcopy(resources),limitations=deepcopy(LIMITATIONS))
    return bind(value)


def _add_resources(total,resources):
    for key in COUNTERS:
        v=resources.get(key,0)
        if type(v) is not int or v<0:raise ValueError('nonnegative integer tool accounting required')
        total[key]+=v


def run_fixed(packet,method,advisor=None,emit=None,*,common=None):
    """Run at most two source diagnostics; a keep retains prior diagnostics."""
    if method not in ('numeric','llm'):raise ValueError('numeric or llm organization required')
    if method=='llm' and advisor is None:raise ValueError('explicit advisor required')
    validate_packet(packet);original=digest(packet)
    if common is None:common=build_options(packet)
    if (common['packet_id']!=packet['packet_id']
            or common['common_id']!=digest({k:v for k,v in common.items() if k!='common_id'})):
        raise ValueError('common table binding changed')
    original_common=digest(common)
    def check_live():
        if digest(packet)!=original or digest(common)!=original_common:
            raise ValueError('public diagnostic inputs changed while waiting')
        validate_packet(packet)
    rounds=[];coverage=[];feedback=None;continuation=None;retained=None
    work={k:0 for k in COUNTERS};feedback_work={k:0 for k in COUNTERS}
    for index,phase in enumerate(('probe','revise')):
        check_live()
        request=request_for(packet,common,phase,coverage,feedback,continuation,
            dict(fitting=work,feedback=feedback_work,common=common['resources']))
        if method=='llm':
            try:
                envelope=deepcopy(advisor(deepcopy(request)))
            except TimeoutError as exc:
                envelope=dict(raw=None,timely=False,actual_llm_request=None,
                    invocation_accounting='unknown_after_advisor_exception',advisor_error=str(exc))
            check_live()
            if not isinstance(envelope,dict):
                envelope=dict(raw=None,timely=None,actual_llm_request=None,
                    original_envelope=envelope,invocation_accounting='invalid_envelope_unknown')
            if type(envelope.get('actual_llm_request')) is not bool:
                envelope.setdefault('invocation_accounting','not_reported_by_advisor')
        else:
            envelope=dict(raw=numeric_response(request),timely=True,actual_llm_request=False,
                origin='numeric_shared_feedback')
        error=None
        try:
            if envelope.get('timely') is not True:raise ValueError('response not timely')
            plan=validate_response(request,envelope.get('raw'))
        except (ValueError,TypeError,KeyError) as exc:
            error=str(exc);plan=keep_response(request,'invalid or unavailable decision; no extra job')
        fit=None;result_feedback=None;tool_error=None
        if plan['action']=='fit':
            check_live()
            seed=next(m for m in packet['models'] if m['id']==plan['target_id'])
            start=retained['model'] if plan['start']=='previous' else None
            fit=tool.fit_mechanism(seed,plan['component_modes'],packet['history'],
                packet['gp_soft_constraints'],f"m8_{packet['time_seconds']:g}_{method}_{index}",start_model=start)
            _add_resources(work,fit['resources'])
            result_feedback=tool.task_feedback(fit,packet['anchor_cache'])
            _add_resources(feedback_work,result_feedback['resources'])
            feedback=_feedback_summary(fit,result_feedback)
            continuation=_continuation(plan,fit,result_feedback,packet)
            retained=fit
            coverage.append(dict(target_id=plan['target_id'],component_modes=deepcopy(plan['component_modes']),
                start=plan['start'],candidate_id=fit['model']['id']))
        record=dict(index=index,phase=phase,request=request,envelope=deepcopy(envelope),
            accepted_plan=plan,parser_accepted=error is None,fallback_error=error,
            fit=fit,task_feedback=result_feedback,tool_error=tool_error,
            resources_after=dict(fitting=deepcopy(work),feedback=deepcopy(feedback_work),common=deepcopy(common['resources'])))
        rounds.append(record)
        if emit:emit(phase,deepcopy(record))
        check_live()
        if plan['action']=='keep':break
    calls=sum(r['envelope'].get('actual_llm_request') is True for r in rounds)
    unknown=sum(type(r['envelope'].get('actual_llm_request')) is not bool for r in rounds)
    return dict(packet_id=packet['packet_id'],method=method,rounds=rounds,common=deepcopy(common),
        retained_fit=deepcopy(retained),coverage=coverage,
        resources=dict(fitting=work,feedback=feedback_work,common=deepcopy(common['resources'])),
        actual_model_calls=calls,unknown_model_call_receipts=unknown,
        model_call_accounting_complete=not unknown,active_pool_changed=False,new_versions_born=0,
        limitations=deepcopy(LIMITATIONS))
