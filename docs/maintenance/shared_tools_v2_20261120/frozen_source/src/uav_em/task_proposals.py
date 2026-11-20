"""D2-M5 bounded two-round task-related source proposals at fixed states."""
from copy import deepcopy
from dataclasses import asdict
import math
import shutil
from types import SimpleNamespace

from . import task_allocation as shared
from .config import load_config
from .environment import ElectromagneticWorld
from .event_demo import FileAdvisor, control, read, save, sha
from .event_inference import Budget, FAMILIES, REGIONS, digest, fit_job, validate_plan
from .event_llm_inference import family_evidence, pilot_feedback, ranked_families, rebase_model
from .feasible_scenario import best, evaluate
from .joint_prediction import JointSettings
from .repair_search import WorkBudget
from .source_geometry import refine_fit

ROOT = shared.ROOT
M4 = ROOT/'outputs/interference_task_allocation_d2_m4_20261120'
METHODS = shared.METHODS


def bind(value):
    return dict(value, request_id=digest({k:v for k,v in value.items() if k!='request_id'}))


def make_packet(original, event):
    packet=deepcopy(original)
    prior=deepcopy(event['bank_before_birth'])
    now=packet['time_seconds']
    if event['time_seconds']!=now or packet['history'][-1]['time_seconds']!=now:
        raise ValueError('state/prefix mismatch')
    if (not prior or any(m['family'] not in FAMILIES or not m['bases']
                         or m['reference_time_seconds']>now for m in prior)
            or not math.isclose(math.fsum(m['weight'] for m in prior),1.,abs_tol=1e-8)):
        raise ValueError('normalized causal source-only prior required')
    if packet['history_sha256']!=digest(packet['history']):raise ValueError('history changed')
    packet.pop('allowed_target_ids');packet.pop('operation');packet.pop('request_id')
    packet.update(version='D2-M5',prior_models=prior,
        family_evidence=family_evidence(SimpleNamespace(models=prior),packet['recent_source_predictions']),
        operation=dict(families=list(FAMILIES),regions=list(REGIONS),max_bases=[1,2],
            pilot_jobs=1,refine_jobs=2,max_geometry_calls=1,
            birth_mass=.25,independent_new_evidence=0,
            constraint='fixed original route pool; no global power continuation or manual parameters/weights'))
    return bind(packet)


def request_for(packet, phase, feedback=None):
    if phase not in ('pilot','refine'):raise ValueError('unknown phase')
    value={k:deepcopy(v) for k,v in packet.items() if k!='request_id'}
    value.update(phase=phase,required_jobs=1 if phase=='pilot' else 2,
        pilot_training_cutoff_seconds=packet['history'][-4]['time_seconds'],
        feedback=deepcopy(feedback),fixed_state_request_id=packet['request_id'])
    return bind(value)


def validate(request, raw):
    if request['request_id']!=digest({k:v for k,v in request.items() if k!='request_id'}):
        raise ValueError('request changed')
    if isinstance(raw,str):
        import json
        text=raw.strip()
        if text.startswith('```') and text.endswith('```'):text='\n'.join(text.splitlines()[1:-1])
        raw=json.loads(text,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if not isinstance(raw,dict) or set(raw)!={'request_id','action','jobs','geometry_job','reason'}:
        raise ValueError('exact proposal schema required')
    if raw['request_id']!=request['request_id']:raise ValueError('wrong request binding')
    if not isinstance(raw['reason'],str) or not 1<=len(raw['reason'])<=1000:raise ValueError('bounded reason required')
    if raw['action']=='keep':
        if request['phase']!='refine' or raw['jobs']!=[] or raw['geometry_job'] is not None:
            raise ValueError('keep only in refine with no jobs or geometry')
    elif raw['action']=='propose':
        validate_plan(request,dict(request_id=raw['request_id'],stage='inference',jobs=raw['jobs'],reason=raw['reason']))
        if len(raw['jobs'])!=request['required_jobs']:raise ValueError('wrong job count')
        index=raw['geometry_job']
        if index is not None and (request['phase']!='refine' or type(index) is not int or index not in (0,1)):
            raise ValueError('at most one full-fit geometry target')
    else:raise ValueError('unknown action')
    return deepcopy(raw)


def family_order(packet):
    family={m['id']:m['family'] for m in packet['models']}
    priorities={f:0. for f in FAMILIES}
    for world in packet['world_actions']:priorities[family[world['id']]]+=world['allocation_priority']
    evidence_order=ranked_families(packet['family_evidence'])
    return sorted(FAMILIES,key=lambda f:(-priorities[f],evidence_order.index(f)))


def numerical_choice(request):
    order=family_order(request)
    if request['phase']=='pilot':families=order[:1]
    else:
        feedback=request['feedback'];pilot=feedback['job']['family']
        families=([pilot,next(f for f in order if f!=pilot)]
            if feedback.get('tail_rmse_db') is not None and feedback['tail_rmse_db']>4.
            else [f for f in order if f!=pilot])
    return dict(request_id=request['request_id'],action='propose',
        jobs=[dict(family=f,region='task_disk',max_bases=2) for f in families],
        geometry_job=0 if request['phase']=='refine' else None,
        reason='task-priority family ordering, birth-after NLL ties, known-tail 4dB feedback rule; fixed common tools')


def execute_jobs(packet, phase, plan, budget, index):
    budget.reserve(plan['jobs'])
    public=dict(goal_m=packet['state']['goal_m'],map=dict(bounds_m=packet['known_map']['bounds_m']))
    history=packet['history'][:-3] if phase=='pilot' else packet['history']
    models=[] if phase=='pilot' else packet['prior_models']
    bank=SimpleNamespace(history=deepcopy(history),models=deepcopy(models),
                         log_weights=[m['log_weight'] for m in models])
    seed=next(f for f in packet['fits'] if f['phase']==phase)
    soft=deepcopy(seed['gp_soft_constraints'])
    if soft['prefix_end_seconds']!=history[-1]['time_seconds'] or soft['independent_evidence_points']!=0:
        raise ValueError('cached GP prefix mismatch')
    fits=[]
    for j,job in enumerate(plan['jobs']):
        try:
            fit=fit_job(bank,public,job,budget,f'm5_state{index}_{phase}{j}',soft=soft)
            if phase=='refine' and plan['geometry_job']==j and fit['model']['bases']:
                fit=refine_fit(fit,public,'geometry')
        except (ValueError,FloatingPointError) as exc:fit=dict(job=deepcopy(job),error=str(exc))
        fits.append(fit)
    return fits


def commit_models(packet,fits,keep=False):
    if keep or len(fits)!=3 or any('model' not in f or not f['model']['bases'] for f in fits):
        return deepcopy(packet['models']),False
    prior=deepcopy(packet['prior_models'])
    for m in prior:
        m['weight']*=.75;m['log_weight']+=math.log(.75)
    for fit in fits:
        m=rebase_model(fit['model'],packet['time_seconds'],len(packet['history'])-1)
        prior.append(dict(m,weight=.25/3,log_weight=math.log(.25/3)))
    return prior,True


def prepare(out):
    out.mkdir(exist_ok=False)
    for name in ('inputs','public/requests','public/responses','runs','predictions','evaluation','replay','reports/frozen_source'):
        (out/name).mkdir(parents=True)
    shutil.copy2(ROOT/'configs/local_power.json',out/'scenario.json')
    shutil.copy2(ROOT/'docs/task_proposals_public_v1.md',out/'public/contract.md')
    ep=read(shared.M3/'runs/geometry_amplitude/episode.json');old=read(M4/'protocol.json')
    for i,now in enumerate(shared.TIMES):
        path=M4/'inputs'/f'{i:02d}.json'
        if sha(path)!=old['inputs_sha256'][path.name]:raise ValueError('old input changed')
        event=next(e for e in ep['inferences'] if e['time_seconds']==now)
        save(out/'inputs'/path.name,make_packet(read(path),event))
    sources=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'tests').rglob('*.py'))
    sources += [ROOT/p for p in ('requirements.txt','docs/task_proposals_pilot_v1.md','docs/task_proposals_public_v1.md')]
    hashes={}
    for path in sources:
        rel=path.relative_to(ROOT);dest=out/'reports/frozen_source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);hashes[str(rel)]=sha(path)
    previous=deepcopy(old['previous_originals'])
    previous['original_m4_sha256']={str(p.relative_to(ROOT)):sha(p) for p in M4.rglob('*') if p.is_file()}
    protocol=dict(version='D2-M5',methods=list(METHODS),times_seconds=list(shared.TIMES),source_sha256=hashes,
        scenario_sha256=sha(out/'scenario.json'),public_contract_sha256=sha(out/'public/contract.md'),
        inputs_sha256={p.name:sha(p) for p in (out/'inputs').glob('*.json')},previous_originals=previous,
        joint_settings=asdict(JointSettings()),max_flights=0,max_sensor_points=0,max_actual_model_calls=6,max_bridge_requests=6,
        max_jobs=18,max_solver_calls=42,max_refinement_calls=6,max_residual_calls=1680,max_jacobian_calls=1680,
        max_dictionary_kernel_terms=240000,max_kernel_terms=30000,max_diagnostic_kernel_terms=36,
        max_planner_receiver_queries=72000,max_edges=3000,max_evaluation_points=1200,
        max_display_receiver_queries=3024,max_replay_truth_points=1008,
        interpretation='three fixed-state two-round mechanism-proposal comparisons; no new flight or independent future evidence',
        design_path='task_proposals_pilot_v1.md')
    save(out/'protocol.json',protocol)
    save(out/'reports/literature_reading.json',read(ROOT/'docs/references/task_proposals_20261120/reading.json'))
    control(out/'public/control.json',dict(status='prepared',max_requests=6))
    shared.check(out)
    print('PREPARED',out,flush=True)


def run_one(out,method):
    protocol=shared.check(out)
    if method not in METHODS:raise ValueError('undeclared method')
    bridge=None
    if method=='allocation_I':
        session=read(out/'reports/session.json')
        if read(out/'public/session_ready.json')['agent_id']!=session['agent_id']:raise ValueError('session not bound')
        bridge=FileAdvisor(out/'public',session['agent_id'])
    folder=out/'runs'/method;folder.mkdir(exist_ok=False)
    cfg=load_config(out/'scenario.json');geometry=ElectromagneticWorld(cfg).geometry
    counts={k:0 for k in ('jobs','solver_calls','refinement_calls','residual_calls','jacobian_calls',
        'dictionary_kernel_terms','kernel_terms','diagnostic_kernel_terms','planner_receiver_queries','edges','actual_model_calls','bridge_requests')}
    control(out/'public/control.json',dict(status='running' if bridge else 'prepared',max_requests=6))
    for i in range(3):
        packet=read(out/'inputs'/f'{i:02d}.json');budget=Budget();rounds=[];all_fits=[];feedback=None;failed=False;keep=False
        for phase in ('pilot','refine'):
            request=request_for(packet,phase,feedback);error=None
            envelope=bridge.ask('inference',request) if bridge else dict(raw=numerical_choice(request),timely=True,actual_llm_request=False)
            counts['actual_model_calls']+=int(envelope['actual_llm_request']);counts['bridge_requests']+=int(bridge is not None)
            try:
                if not envelope['timely']:raise ValueError('response not timely')
                plan=validate(request,envelope['raw'])
            except (ValueError,TypeError,KeyError) as exc:
                error=str(exc);plan=None;failed=True
            fits=[]
            if plan and plan['action']=='propose':
                fits=execute_jobs(packet,phase,plan,budget,i);all_fits.extend(fits)
            elif plan:keep=True
            if phase=='pilot':
                failed_fit=dict(job=dict(family=FAMILIES[0],region='task_disk',max_bases=2),error='pilot unavailable')
                feedback=pilot_feedback(fits[0] if fits else failed_fit,SimpleNamespace(history=packet['history']))
                counts['diagnostic_kernel_terms']+=feedback.get('diagnostic_kernel_terms',0)
            rounds.append(dict(request=request,envelope=envelope,accepted_plan=plan,parser_accepted=error is None,
                fallback_error=error,fits=fits,pilot_feedback=deepcopy(feedback) if phase=='pilot' else None,
                resources_after=asdict(budget)))
        models,committed=commit_models(packet,all_fits,keep=keep or failed)
        counts['jobs']+=budget.reserved_jobs
        for key in ('solver_calls','residual_calls','jacobian_calls','dictionary_kernel_terms'):counts[key]+=getattr(budget,key)
        for fit in all_fits:
            if 'geometry_refinement' in fit:
                counts['refinement_calls']+=1
                for key,value in fit['geometry_refinement']['resources'].items():counts[key]+=value
        forecast=shared.forecast_for(packet,models,cfg,geometry);work=WorkBudget(1,500)
        candidates=[evaluate(c['path'],packet['state'],forecast,geometry,work,'joint',c['id']) for c in packet['candidates']]
        chosen=best(candidates)
        counts['planner_receiver_queries']+=forecast.counts['receiver_queries'];counts['edges']+=work.edges
        label=', '.join(f"{f['job']['family']}/{f['job']['region']}/{f['job']['max_bases']}" for f in all_fits)
        record=dict(method=method,index=i,time_seconds=packet['time_seconds'],request_id=packet['request_id'],
            public_prefix_sha256=packet['history_sha256'],rounds=rounds,pool_committed=committed,
            choice=dict(action='propose' if committed else 'keep',target_id=label if committed else None,
                reason='committed three new worlds' if committed else 'original pool retained'),
            source_models=models,forecast_sha256=forecast.fingerprint,candidates=candidates,selected=chosen,
            next_15m_position=shared.endpoint(chosen['path']),
            original_next_15m_position=shared.endpoint(next(c for c in packet['candidates'] if c['id']==packet['original_selected'])['path']),
            receiver_work=deepcopy(forecast.counts),resources=work.summary())
        save(folder/f'{i:02d}.json',record)
        print('PROPOSED',method,i,packet['time_seconds'],committed,label,chosen['id'],flush=True)
    for key,value in counts.items():assert value<=protocol['max_'+key],(key,value)
    save(folder/'status.json',dict(status='completed',counts=counts,record_sha256={f'{i:02d}.json':sha(folder/f'{i:02d}.json') for i in range(3)}))
    if bridge:control(out/'public/control.json',dict(status='completed',max_requests=6,bridge_requests=bridge.count))


finalize=shared.finalize
