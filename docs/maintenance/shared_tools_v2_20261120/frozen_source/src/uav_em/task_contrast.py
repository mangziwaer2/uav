"""D2-M7: cached task contrasts and bounded scalar dynamic witnesses, never births."""
from copy import deepcopy
import html
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import minimize

from . import task_allocation as shared
from .equivalent_field import kernel
from .event_demo import FileAdvisor,control,read,save,sha
from .event_inference import digest

ROOT=shared.ROOT
M6=ROOT/'outputs/interference_constrained_geometry_d2_m6_20261120'
METHODS=('contrast_N','contrast_I')
BINS=(0.,5.,10.,20.,math.inf)


def audit(record):
    """Pure cached algebra, preserving saved linear-power expectations."""
    feasible=[c for c in record['candidates'] if c['feasible']]
    chosen=min(feasible,key=lambda c:(c['score']['score_seconds'],sum(math.dist(a,b) for a,b in zip(c['path'],c['path'][1:])),c['id']))
    if chosen['id']!=record['selected']['id']:raise ValueError('cached argmin changed')
    now=record['time_seconds'];base=chosen['score'];rows=[]
    def bins(candidate):
        s=candidate['score'];q=s['queries'];tt=np.asarray(q['time_seconds'])-now
        powers=np.asarray(s['expected_normalized_power']);dt=np.asarray(q['dt_seconds'])
        if not np.all(np.isfinite(powers)) or np.any(powers<0):raise ValueError('invalid expected power')
        return np.asarray([powers[:,(tt>=lo)&(tt<hi)]@dt[(tt>=lo)&(tt<hi)] for lo,hi in zip(BINS,BINS[1:])]).T
    base_bins=bins(chosen)
    for c in feasible:
        if c['id']==chosen['id']:continue
        s=c['score'];dT=s['remaining_flight_seconds']-base['remaining_flight_seconds'];worlds=[]
        delta_bins=bins(c)-base_bins
        if [w['id'] for w in s['scenario_scores']]!=[w['id'] for w in base['scenario_scores']]:raise ValueError('world order changed')
        for i,(w,a) in enumerate(zip(s['scenario_scores'],base['scenario_scores'])):
            dq=w['power_proxy_seconds']-a['power_proxy_seconds']
            if not math.isclose(float(delta_bins[i].sum()),dq,abs_tol=1e-7):raise ValueError('cache integral mismatch')
            worlds.append(dict(id=w['id'],weight=w['weight'],delta_q=dq,weighted_delta_q=w['weight']*dq,
                delta_c=dT+dq,delta_q_by_arrival_bin=delta_bins[i].tolist()))
        dC=s['score_seconds']-base['score_seconds']
        if not math.isclose(dC,dT+math.fsum(w['weighted_delta_q'] for w in worlds),abs_tol=1e-7):raise ValueError('cache mixture mismatch')
        rows.append(dict(route_b=c['id'],delta_t=dT,delta_c=dC,worlds=worlds,
            next_15m_distance_m=math.dist(shared.endpoint(chosen['path']),shared.endpoint(c['path']))))
    return dict(route_a=chosen['id'],pairs=rows,arrival_bins_relative_seconds=[[lo,None if math.isinf(hi) else hi] for lo,hi in zip(BINS,BINS[1:])],
        arrival_bins_absolute_seconds=[[now+lo,None if math.isinf(hi) else now+hi] for lo,hi in zip(BINS,BINS[1:])],binning='cached integration midpoint')


def make_request(packet,record):
    now=packet['time_seconds']
    history=packet['history']
    if (now!=record['time_seconds'] or history[-1]['time_seconds']!=now
            or any(not math.isfinite(p['time_seconds']) or p['time_seconds']>now for p in history)
            or any(a['time_seconds']>=b['time_seconds'] for a,b in zip(history,history[1:]))):
        raise ValueError('noncausal prefix')
    for m in record['source_models']:
        if (not math.isfinite(m['reference_time_seconds']) or m['reference_time_seconds']>now
                or type(m['born_index']) is not int or not 0<=m['born_index']<len(history)):
            raise ValueError('noncausal source version')
    for key in ('recent_source_predictions','recent_joint_predictions'):
        for item in packet[key]:
            issued=item['issued'];observed=item['observation']['time_seconds'];predicted=issued['time_seconds']
            if (not all(math.isfinite(t) for t in (observed,predicted)) or observed>now or predicted>now
                    or issued.get('forecast_at_seconds',predicted)>predicted):
                raise ValueError('noncausal recent prediction')
    soft=deepcopy(packet['fits'][-1]['gp_soft_constraints'])
    if soft['prefix_end_seconds']!=now or soft['independent_evidence_points']!=0 or any(t>now for t in soft['time_seconds']):
        raise ValueError('noncausal or independent GP soft points')
    value=dict(version='D2-M7',cached_record_sha256=digest(record),time_seconds=now,history=deepcopy(packet['history']),state=deepcopy(packet['state']),
        known_map=deepcopy(packet['known_map']),models=deepcopy(record['source_models']),gp_soft_constraints=soft,
        recent_source_predictions=deepcopy(packet['recent_source_predictions']),
        recent_joint_predictions=deepcopy(packet['recent_joint_predictions']),consultation_reasons=deepcopy(packet['consultation_reasons']),
        contrasts=audit(record),allowed_world_ids=[m['id'] for m in record['source_models'] if m['family']=='strength_change' and m['weight']>=.01],
        operation=dict(action='profile_rate',rate_bounds=[-.5,.5],max_parameter_evaluations=40,max_kernel_terms=20000,
            real_mse_slack_db2=1.,gp_mse_slack_db2=1.,births=0,weight_changes=0,route_changes=0),
        interpretation='conditional diagnostic only; relative fit compatibility is not absolute trust; no new independent evidence')
    value['request_id']=digest(value);return value


def validate(request,raw):
    if request['request_id']!=digest({k:v for k,v in request.items() if k!='request_id'}):raise ValueError('request changed')
    if isinstance(raw,str):
        raw=json.loads(raw,object_pairs_hook=_unique_pairs)
    if not isinstance(raw,dict) or set(raw)!={'request_id','action','world_id','basis_index','route_b','reason'}:raise ValueError('exact response schema required')
    if raw['request_id']!=request['request_id']:raise ValueError('wrong request binding')
    if not isinstance(raw['reason'],str) or not 1<=len(raw['reason'])<=2000:raise ValueError('bounded reason required')
    if raw['action']=='keep':
        if any(raw[k] is not None for k in ('world_id','basis_index','route_b')):raise ValueError('keep requires null targets')
    elif raw['action']=='profile_rate':
        if raw['world_id'] not in request['allowed_world_ids']:raise ValueError('world not eligible')
        m=next(m for m in request['models'] if m['id']==raw['world_id'])
        if type(raw['basis_index']) is not int or not 0<=raw['basis_index']<len(m['bases']):raise ValueError('invalid basis')
        if raw['route_b'] not in [p['route_b'] for p in request['contrasts']['pairs']]:raise ValueError('invalid route')
    else:raise ValueError('invalid action')
    return deepcopy(raw)


def _unique_pairs(items):
    result={}
    for k,v in items:
        if k in result:raise ValueError('duplicate response key')
        result[k]=v
    return result


def keep(request,reason):
    return dict(request_id=request['request_id'],action='keep',world_id=None,basis_index=None,route_b=None,reason=reason)


def numerical_choice(request):
    options=[]
    for p in request['contrasts']['pairs']:
        for w in p['worlds']:
            if w['id'] in request['allowed_world_ids']:
                priority=w['weight']*abs(w['delta_q'])/(1+max(0.,p['delta_c']))*(.25+min(p['next_15m_distance_m']/15.,1.))
                options.append((-priority,w['id'],p['route_b']))
    if not options:return keep(request,'no eligible dynamic world')
    _,wid,b=min(options);m=next(m for m in request['models'] if m['id']==wid)
    basis=min(range(len(m['bases'])),key=lambda i:(-m['peaks_dbm'][i],i))
    return dict(request_id=request['request_id'],action='profile_rate',world_id=wid,basis_index=basis,route_b=b,
        reason='cached weighted contrast / nominal gap, weighted by next-action distance; highest peak basis')


class LimitReached(ValueError):pass


class RateProblem:
    """Analytic one-rate profile; GP cancels exactly in the changed source component."""
    def __init__(self,request,record,plan):
        if request['cached_record_sha256']!=digest(record):raise ValueError('cached record binding changed')
        self.model=deepcopy(next(m for m in request['models'] if m['id']==plan['world_id']))
        self.index=plan['basis_index'];self.seed=self.model['bases'][self.index]['rate_db_per_second']
        if not -.5<=self.seed<=.5:raise ValueError('seed outside rate bounds')
        self.pair=next(p for p in request['contrasts']['pairs'] if p['route_b']==plan['route_b'])
        self.world=next(w for w in self.pair['worlds'] if w['id']==plan['world_id'])
        self.routes=[next(c for c in record['candidates'] if c['id']==cid)['score'] for cid in (request['contrasts']['route_a'],plan['route_b'])]
        history=request['history'][-12:];soft=request['gp_soft_constraints'];self.nreal=len(history);self.soft_weights=np.asarray(soft['weights'])
        self.xy=np.asarray([p['position_m'] for p in history]+soft['position_m']+[p for s in self.routes for p in s['queries']['position_m']])
        self.times=np.asarray([p['time_seconds'] for p in history]+soft['time_seconds']+[t for s in self.routes for t in s['queries']['time_seconds']])
        self.labels=np.asarray([p['interference_dbm'] for p in history]+soft['mean_dbm']);self.nfit=len(self.labels)
        self.dtref=self.times-self.model['reference_time_seconds']
        self.gp_dtref=self.dtref[self.nreal:self.nfit].tolist()
        self.route_coeff=np.concatenate([-np.asarray(self.routes[0]['queries']['dt_seconds'])*(1-np.asarray(self.routes[0]['gp_weight'])),
                                         np.asarray(self.routes[1]['queries']['dt_seconds'])*(1-np.asarray(self.routes[1]['gp_weight']))])
        self.work=dict(parameter_evaluations=0,kernel_terms=0,objective_calls=0,gradient_calls=0,constraint_calls=0,constraint_gradient_calls=0,solver_calls=0)
        self.trace=[];self.max_parameters=request['operation']['max_parameter_evaluations'];self.max_terms=request['operation']['max_kernel_terms']
        self.slacks=np.array([request['operation']['real_mse_slack_db2'],request['operation']['gp_mse_slack_db2']])
        self.baseline=None

    def evaluate(self,rate):
        if self.work['parameter_evaluations']>=self.max_parameters:raise LimitReached('parameter budget exhausted')
        terms=len(self.times)*len(self.model['bases'])
        if self.work['kernel_terms']+terms>self.max_terms:raise LimitReached('kernel budget exhausted')
        self.work['parameter_evaluations']+=1;self.work['kernel_terms']+=terms
        bases=deepcopy(self.model['bases']);bases[self.index]['rate_db_per_second']=rate
        matrix=kernel(bases,self.xy,self.times,self.model['reference_time_seconds'])
        contribution=matrix*10**((np.asarray(self.model['peaks_dbm'])+80)/10)
        total=.01+contribution.sum(axis=1);mu=-80+10*np.log10(total)
        dmu=contribution[:,self.index]/total*self.dtref;residual=mu[:self.nfit]-self.labels
        losses=np.array([np.mean(residual[:self.nreal]**2),np.dot(self.soft_weights,residual[self.nreal:]**2)])
        gradients=np.array([np.mean(2*residual[:self.nreal]*dmu[:self.nreal]),
            np.dot(self.soft_weights,2*residual[self.nreal:]*dmu[self.nreal:self.nfit])])
        factor=math.exp(.5*(math.log(10)/10*self.model['std_db'])**2)
        qpart=float(self.route_coeff@total[self.nfit:])*factor
        dqdr=float(self.route_coeff@(contribution[self.nfit:,self.index]*self.dtref[self.nfit:]))*factor*math.log(10)/10
        if not np.all(np.isfinite(np.r_[losses,gradients,qpart,dqdr])):raise ValueError('nonfinite profile')
        if self.baseline is None:
            if rate!=self.seed:raise ValueError('seed must be evaluated first')
            self.baseline=dict(losses=losses.copy(),qpart=qpart)
        delta=qpart-self.baseline['qpart'];world_c=self.world['delta_c']+delta
        mixture_c=self.pair['delta_c']+self.model['weight']*delta
        row=dict(rate_db_per_second=float(rate),real_mse_db2=float(losses[0]),gp_mse_db2=float(losses[1]),
            world_delta_c_seconds=world_c,frozen_weight_delta_c_seconds=mixture_c,
            feasible=bool(-.5<=rate<=.5 and np.all(losses<=self.baseline['losses']+self.slacks+1e-7)))
        self.trace.append(row)
        return row,losses,gradients,qpart,dqdr


def profile(request,record,plan):
    plan=validate(request,plan)
    if plan['action']=='keep':return None
    problem=RateProblem(request,record,plan);results=[]
    for sign in (1.,-1.):
        cache={}
        def ev(x):
            r=float(x[0])
            if r not in cache:
                if len(cache)>=20:raise LimitReached('direction parameter budget exhausted')
                cache[r]=problem.evaluate(r)
            return cache[r]
        def objective(x):
            if problem.work['objective_calls']>=40:raise LimitReached('objective budget exhausted')
            problem.work['objective_calls']+=1
            return sign*ev(x)[3]
        def gradient(x):
            if problem.work['gradient_calls']>=40:raise LimitReached('gradient budget exhausted')
            problem.work['gradient_calls']+=1
            return np.array([sign*ev(x)[4]])
        def constraint(x):
            problem.work['constraint_calls']+=1
            return problem.baseline['losses']+problem.slacks-ev(x)[1]
        def constraint_gradient(x):
            problem.work['constraint_gradient_calls']+=1
            return -ev(x)[2][:,None]
        try:
            ev([problem.seed]);problem.work['solver_calls']+=1
            solved=minimize(objective,[problem.seed],jac=gradient,method='SLSQP',bounds=[(-.5,.5)],
                constraints=[dict(type='ineq',fun=constraint,jac=constraint_gradient)],options=dict(ftol=1e-8,maxiter=20))
            results.append(dict(direction='min' if sign==1 else 'max',success=bool(solved.success),status=int(solved.status),message=str(solved.message)))
        except (ValueError,FloatingPointError) as exc:
            results.append(dict(direction='min' if sign==1 else 'max',success=False,error=str(exc)))
    legal=[r for r in problem.trace if r['feasible']]
    if not legal:raise ValueError('seed witness missing')
    ends=[min(legal,key=lambda r:(r['world_delta_c_seconds'],r['rate_db_per_second'])),max(legal,key=lambda r:(r['world_delta_c_seconds'],-r['rate_db_per_second']))]
    wi=[e['world_delta_c_seconds'] for e in ends];mi=[e['frozen_weight_delta_c_seconds'] for e in ends]
    return dict(plan=plan,route_a=request['contrasts']['route_a'],route_b=plan['route_b'],world_id=plan['world_id'],
        seed_rate_db_per_second=problem.seed,baseline_real_mse_db2=float(problem.baseline['losses'][0]),baseline_gp_mse_db2=float(problem.baseline['losses'][1]),
        nominal_delta_c_seconds=problem.pair['delta_c'],world_delta_c_interval_seconds=wi,frozen_weight_delta_c_interval_seconds=mi,
        world_ranking_crossed=wi[0]<-1e-7 and wi[1]>1e-7,frozen_weight_ranking_crossed=mi[0]<-1e-7 and mi[1]>1e-7,
        next_15m_distance_m=problem.pair['next_15m_distance_m'],endpoints=deepcopy(ends),trace=problem.trace,solvers=results,
        model_reference_time_seconds=problem.model['reference_time_seconds'],gp_time_offsets_from_model_reference_seconds=problem.gp_dtref,
        gp_has_nonzero_time_offset=any(t!=0 for t in problem.gp_dtref),
        resources=problem.work,births=0,weight_changes=0,route_changes=0,interval_is_global_bound=False)


def check(out):
    p=read(out/'protocol.json')
    for rel,h in p['source_sha256'].items():
        if sha(ROOT/rel)!=h or sha(out/'reports/frozen_source'/rel)!=h:raise ValueError('source changed: '+rel)
    for rel,h in p['inputs_sha256'].items():
        if sha(out/rel)!=h:raise ValueError('input changed: '+rel)
    for rel,h in p['previous_originals'].items():
        if sha(ROOT/rel)!=h:raise ValueError('original changed: '+rel)
    return p


def prepare(out):
    out.mkdir(exist_ok=False)
    for name in ('inputs','public/requests','public/responses','runs','review','reports/frozen_source'):(out/name).mkdir(parents=True)
    shutil.copy2(ROOT/'docs/task_contrast_public_v1.md',out/'public/contract.md')
    for i in range(3):
        a=M6/'inputs'/f'{i:02d}.json';r=M6/'runs/allocation_I'/f'{i:02d}.json'
        original=read(M6/'reports/completion_manifest.json')
        for src,name in ((a,f'packet_{i:02d}.json'),(r,f'record_{i:02d}.json')):
            if sha(src)!=original[str(src.relative_to(M6))]:raise ValueError('M6 changed')
            shutil.copy2(src,out/'inputs'/name)
        save(out/'inputs'/f'request_{i:02d}.json',make_request(read(a),read(r)))
    sources=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'tests').rglob('*.py'))
    sources += [ROOT/p for p in ('requirements.txt','docs/task_contrast_pilot_v1.md','docs/task_contrast_public_v1.md','tools/verify_task_contrast.py')]
    hashes={}
    for src in sources:
        rel=src.relative_to(ROOT);dest=out/'reports/frozen_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest);hashes[str(rel)]=sha(src)
    previous={}
    old=read(M6/'protocol.json')['previous_originals']
    for key,files in old.items():
        base=ROOT/'outputs/interference_joint_gp_d1_20261119_0940' if key=='original_d1_sha256' else ROOT
        previous.update({str((base/name).relative_to(ROOT)):h for name,h in files.items()})
    previous.update({str(p.relative_to(ROOT)):sha(p) for p in M6.rglob('*') if p.is_file()})
    files=list((out/'inputs').glob('*.json'))+[out/'public/contract.md']
    protocol=dict(version='D2-M7',source_sha256=hashes,inputs_sha256={str(p.relative_to(out)):sha(p) for p in files},previous_originals=previous,
        methods=list(METHODS),max_jobs=6,max_solver_calls=12,max_parameter_evaluations=240,max_objective_calls=240,max_gradient_calls=240,max_kernel_terms=120000,
        max_bridge_requests=3,max_actual_model_calls=3,max_flights=0,max_sensor_points=0,max_truth_queries=0,max_gp_queries=0,
        max_route_changes=0,max_weight_changes=0,scope='shared M6 I states; conditional diagnostic, no prospective task benefit')
    save(out/'protocol.json',protocol);control(out/'public/control.json',dict(status='prepared',max_requests=3));check(out)
    print('PREPARED',out,flush=True)


def run_one(out,method):
    check(out)
    if method not in METHODS:raise ValueError('undeclared method')
    bridge=None
    if method=='contrast_I':
        session=read(out/'reports/session.json')
        if read(out/'public/session_ready.json')['agent_id']!=session['agent_id']:raise ValueError('session not bound')
        bridge=FileAdvisor(out/'public',session['agent_id']);control(out/'public/control.json',dict(status='running',max_requests=3))
    folder=out/'runs'/method;folder.mkdir(exist_ok=False);totals=dict(jobs=0,bridge_requests=0,actual_model_calls=0)
    for i in range(3):
        request=read(out/'inputs'/f'request_{i:02d}.json');record=read(out/'inputs'/f'record_{i:02d}.json')
        envelope=bridge.ask('inference',request) if bridge else dict(raw=numerical_choice(request),timely=True,actual_llm_request=False)
        error=None
        try:
            if not envelope['timely']:raise ValueError(envelope.get('error') or 'untimely response')
            plan=validate(request,envelope['raw'])
        except (ValueError,TypeError,KeyError) as exc:
            error=str(exc);plan=keep(request,'response fallback: '+error)
        result=profile(request,record,plan)
        totals['jobs']+=int(result is not None);totals['bridge_requests']+=int(bridge is not None);totals['actual_model_calls']+=int(envelope['actual_llm_request'])
        if result:
            for k,v in result['resources'].items():totals[k]=totals.get(k,0)+v
        save(folder/f'{i:02d}.json',dict(method=method,index=i,time_seconds=request['time_seconds'],request_id=request['request_id'],
            accepted_plan=plan,envelope=envelope,fallback=error is not None,error=error,result=result,bank_sha256=digest(request['models'])))
        print('PROFILED',method,i,plan['world_id'],plan['route_b'],None if result is None else result['frozen_weight_delta_c_interval_seconds'],flush=True)
    save(folder/'status.json',dict(status='completed',counts=totals,record_sha256={f'{i:02d}.json':sha(folder/f'{i:02d}.json') for i in range(3)}))
    if bridge:control(out/'public/control.json',dict(status='completed',max_requests=3,bridge_requests=bridge.count))


def finalize(out):
    protocol=check(out);rows=[];totals={}
    for method in METHODS:
        status=read(out/'runs'/method/'status.json')
        if status['status']!='completed':raise ValueError('incomplete method')
        for name,h in status['record_sha256'].items():
            if sha(out/'runs'/method/name)!=h:raise ValueError('run changed')
        for k,v in status['counts'].items():totals[k]=totals.get(k,0)+v
        rows.extend(read(out/'runs'/method/f'{i:02d}.json') for i in range(3))
    for k,v in totals.items():
        if 'max_'+k in protocol and v>protocol['max_'+k]:raise ValueError('batch budget exceeded '+k)
    save(out/'reports/witness_freeze.json',{str(p.relative_to(out)):sha(p) for p in (out/'runs').rglob('*') if p.is_file()})
    result=dict(rows=rows,counts=totals,new_flights=0,new_sensor_points=0,new_gp_queries=0,new_truth_queries=0,
        births=0,weight_changes=0,route_changes=0,scope=protocol['scope'])
    save(out/'reports/results.json',result);render(out,result)
    check(out);save(out/'reports/completion_manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(dict(counts=totals,summary=[dict(method=r['method'],time=r['time_seconds'],plan=r['accepted_plan'],
        result=None if r['result'] is None else {k:r['result'][k] for k in ('frozen_weight_delta_c_interval_seconds','world_ranking_crossed','frozen_weight_ranking_crossed','next_15m_distance_m')}) for r in rows]),ensure_ascii=False,indent=2),flush=True)


def render(out,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),dpi=110)
    for ax,row in zip(axes.ravel(),result['rows']):
        r=row['result'];ax.axhline(0,color='#555',lw=1)
        if r:
            valid=[x for x in r['trace'] if x['feasible']];bad=[x for x in r['trace'] if not x['feasible']]
            for points,color,label in ((bad,'#aaa','incompatible trial'),(valid,'#006cba','compatible witness')):
                ax.scatter([x['rate_db_per_second'] for x in points],[x['frozen_weight_delta_c_seconds'] for x in points],c=color,s=18,label=label)
            ax.set_title(f"{row['method']} t={row['time_seconds']:g}s | {r['route_a']} -> {r['route_b']}\n{r['world_id']} B{r['plan']['basis_index']} | next={r['next_15m_distance_m']:.2f}m",fontsize=9)
        else:ax.set_title(f"{row['method']} t={row['time_seconds']:g}s | keep")
        ax.set(xlabel='rate (dB/s)',ylabel='conditional C(B)-C(A) (s)');ax.grid(alpha=.2)
    fig.suptitle('M7: reached witnesses only | fixed states | NO NEW FLIGHT / TRUTH / BANK UPDATE')
    fig.tight_layout();fig.savefig(out/'review/intervals.png');plt.close(fig)
    data=[]
    for row in result['rows']:
        i=row['index'];packet=read(out/'inputs'/f'packet_{i:02d}.json');record=read(out/'inputs'/f'record_{i:02d}.json')
        data.append(dict(row=row,packet=dict(known_map=packet['known_map'],history=packet['history']),
            models=record['source_models'],candidates=[dict(id=c['id'],path=c['path']) for c in record['candidates'] if c['feasible']]))
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>M7任务排序相容见证</title><style>body{font:16px/1.7 system-ui;max-width:1300px;margin:auto}svg{border:1px solid #ccc;width:100%}pre{white-space:pre-wrap}img{width:100%}select{font-size:17px}</style><h1>M7：已有观测能否确定路线排序？</h1><p>固定公开状态；无新飞行/点测/真值查询。蓝线为原最优A，橙线为受检B，紫色为所选互斥假设的等效基元（非真定位）。下方是实际达到的相容参数见证，不是全局边界，不更新源池或执行路线。</p><p><a href="../../../docs/task_contrast_results_20261120.md">结果说明</a> · <a href="../reports/results.json">原始结果</a></p><select id="pick"></select><svg id="map" viewBox="0 0 800 600"></svg><pre id="detail"></pre><img src="intervals.png"><script>const data=PAYLOAD;const pick=document.querySelector('#pick'),map=document.querySelector('#map');data.forEach((d,i)=>pick.add(new Option(d.row.method+' | '+d.row.time_seconds+' s',i)));function draw(){const d=data[+pick.value],r=d.row.result;let svg='';for(const b of d.packet.known_map.buildings)svg+='<polygon points="'+b.footprint_m.map(p=>p[0]+','+(600-p[1])).join(' ')+'" fill="#888"/>';const ids=r?[r.route_a,r.route_b]:[];ids.forEach((id,i)=>{const p=d.candidates.find(c=>c.id===id).path;svg+='<polyline points="'+p.map(q=>q[0]+','+(600-q[1])).join(' ')+'" fill="none" stroke="'+['#006cba','#dd7c12'][i]+'" stroke-width="3"/>'});svg+='<polyline points="'+d.packet.history.map(p=>p.position_m[0]+','+(600-p.position_m[1])).join(' ')+'" fill="none" stroke="#222" stroke-width="2"/>';if(r){const m=d.models.find(m=>m.id===r.world_id);m.bases.forEach((b,i)=>{const dt=d.row.time_seconds-m.reference_time_seconds,x=b.center_m[0]+dt*b.velocity_mps[0],y=600-b.center_m[1]-dt*b.velocity_mps[1];svg+='<circle cx="'+x+'" cy="'+y+'" r="'+b.scale_m+'" fill="none" stroke="#a000ab" stroke-dasharray="4 4"/><circle cx="'+x+'" cy="'+y+'" r="5" fill="#a000ab"/><text x="'+(x+7)+'" y="'+y+'">B'+i+'</text>'})}map.innerHTML=svg;const view={time_seconds:d.row.time_seconds,method:d.row.method,plan:d.row.accepted_plan,fallback:d.row.fallback,result:r?Object.fromEntries(Object.entries(r).filter(([k])=>k!=='trace')):null};document.querySelector('#detail').textContent=JSON.stringify(view,null,2)}pick.onchange=draw;draw();</script>'''.replace('PAYLOAD',payload)
    (out/'review/index.html').write_text(page)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><a href="review/index.html">M7任务排序相容见证（无新飞行）</a>')
