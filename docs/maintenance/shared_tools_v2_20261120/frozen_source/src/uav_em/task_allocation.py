"""D2-M4 fixed-state computation allocation; no new flights or point sensors."""
from copy import deepcopy
from dataclasses import asdict
import html
import json
import math
from pathlib import Path
import shutil

import numpy as np

from .config import load_config
from .environment import ElectromagneticWorld
from .event_demo import FileAdvisor, control, read, save, sha
from .event_inference import digest
from .event_llm_inference import rebase_model
from .feasible_scenario import best, evaluate
from .joint_experiment import numerical_samples
from .joint_prediction import JointSettings
from .motion import advance_route
from .point_gp import FrozenGP, training_points
from .repair_search import WorkBudget
from .scenario_planning import ScenarioForecast
from .source_geometry import refine_fit

ROOT=Path(__file__).resolve().parents[2]
M3=ROOT/'outputs/interference_source_geometry_d2_m3_20261120'
TIMES=(20.5,25.,33.5)
METHODS=('allocation_N','allocation_I')


def endpoint(path):
    remaining,_=advance_route(deepcopy(path),min(15.,sum(math.dist(a,b) for a,b in zip(path,path[1:]))))
    return remaining[0]


def make_packet(episode,now,known_map):
    decision=next(d for d in episode['decisions'] if d['time_seconds']==now)
    event=next(e for e in episode['inferences'] if e['time_seconds']==now)
    request=decision['public']['request'];history=deepcopy(request['history'])
    if not history or history[-1]['time_seconds']!=now or any(p['time_seconds']>now for p in history):
        raise ValueError('noncausal public history')
    fits=[]
    for rnd in event['rounds']:
        for fit in rnd['fits']:
            if 'model' not in fit or not fit['model']['bases']:continue
            keep=('job','model','window','bounds_dbm','gp_soft_constraints','fit_objective','gp_shape_mse')
            item={k:deepcopy(fit[k]) for k in keep}
            if item['window'][-1]['time_seconds']>now:raise ValueError('future fit prefix')
            item['phase']=rnd['request']['phase']
            item['pilot_feedback']=deepcopy(rnd['pilot_feedback'])
            if item['phase']=='pilot' and item['window'][-1]['time_seconds']!=rnd['request']['pilot_training_cutoff_seconds']:
                raise ValueError('pilot cutoff mismatch')
            fits.append(item)
    if len(fits)!=3:raise ValueError('three shared fitted hypotheses required')
    candidates=[]
    for candidate in decision['candidates']:
        c={k:deepcopy(candidate[k]) for k in ('id','path','feasible','reason')}
        if c['feasible']:
            c['score']={k:deepcopy(candidate['score'][k]) for k in ('score_seconds','power_proxy_seconds','remaining_flight_seconds','scenario_scores')}
        candidates.append(c)
    valid=[c for c in candidates if c['feasible']]
    selected=next(c for c in valid if c['id']==decision['selected_id'])
    selected_end=endpoint(selected['path']);worlds=[]
    for world in selected['score']['scenario_scores']:
        def score(c):return next(s['score_seconds'] for s in c['score']['scenario_scores'] if s['id']==world['id'])
        chosen=min(valid,key=lambda c:(score(c),sum(math.dist(a,b) for a,b in zip(c['path'],c['path'][1:])),c['id']))
        distance=math.dist(endpoint(chosen['path']),selected_end)
        regret=max(0.,score(selected)-score(chosen))
        worlds.append(dict(id=world['id'],weight=world['weight'],selected=chosen['id'],
            next_15m_position=endpoint(chosen['path']),distance_to_mixture_endpoint_m=distance,
            predicted_regret_seconds=regret,allocation_priority=world['weight']*regret*distance/15.))
    packet=dict(version='D2-M4',time_seconds=now,history=history,history_sha256=digest(history),
        known_map=deepcopy(known_map),state=deepcopy(request['state']),models=deepcopy(request['models']),
        gp_local_samples=deepcopy(decision['public']['gp_local_samples']),fits=fits,candidates=candidates,
        original_selected=decision['selected_id'],world_actions=worlds,
        recent_source_predictions=[deepcopy(r) for r in episode['prediction_records'] if r['observation']['time_seconds']<=now][-6:],
        recent_joint_predictions=[deepcopy(r) for r in episode['joint_prediction_records'] if r['observation']['time_seconds']<=now][-6:],
        consultation_reasons=deepcopy(decision['public']['consultation_event']['reasons']),
        allowed_target_ids=[f['model']['id'] for f in fits],
        operation=dict(max_geometry_calls=1,actions=['keep','refine_geometry'],center_per_axis_radius_m=40.,scale_ratio=math.sqrt(2.),
            max_residual_calls=40,max_jacobian_calls=40,max_kernel_terms=5000,
            constraints='same fit objective and prefix; original task region; no weight, source count, dynamics or route-candidate changes'),
        interpretation='fixed observed state; mutually exclusive worlds; conditional weights are not absolute trust; no future or truth information')
    packet['request_id']=digest(packet)
    return packet


def validate(packet,raw):
    if packet['request_id']!=digest({k:v for k,v in packet.items() if k!='request_id'}):raise ValueError('request changed')
    if isinstance(raw,str):
        text=raw.strip()
        if text.startswith('```') and text.endswith('```'):text='\n'.join(text.splitlines()[1:-1])
        raw=json.loads(text)
    if not isinstance(raw,dict) or set(raw)!={'request_id','action','target_id','reason'}:raise ValueError('exact response schema required')
    if raw['request_id']!=packet['request_id']:raise ValueError('wrong request binding')
    if raw['action'] not in ('keep','refine_geometry'):raise ValueError('unsupported action')
    if raw['action']=='keep' and raw['target_id'] is not None:raise ValueError('keep requires null target')
    if raw['action']=='refine_geometry' and raw['target_id'] not in packet['allowed_target_ids']:raise ValueError('target outside current newborn fits')
    if not isinstance(raw['reason'],str) or not 1<=len(raw['reason'])<=2000:raise ValueError('bounded reason required')
    return deepcopy(raw)


def numerical_choice(packet):
    priorities={w['id']:w['allocation_priority'] for w in packet['world_actions']}
    fit=min(packet['fits'],key=lambda f:(-priorities[f['model']['id']],-f['fit_objective'],f['model']['id']))
    return dict(request_id=packet['request_id'],action='refine_geometry',target_id=fit['model']['id'],
        reason='maximize conditional weighted route regret times next-15m action distance; tie by original fit loss then ID')


def apply_choice(packet,choice):
    choice=validate(packet,choice);models=deepcopy(packet['models']);updated=None
    if choice['action']=='refine_geometry':
        fit=next(f for f in packet['fits'] if f['model']['id']==choice['target_id'])
        public=dict(goal_m=packet['state']['goal_m'],map=dict(bounds_m=packet['known_map']['bounds_m']))
        updated=refine_fit(fit,public,'geometry')
        if updated['geometry_refinement']['accepted']:
            replacement=rebase_model(updated['model'],packet['time_seconds'],len(packet['history'])-1)
            for i,model in enumerate(models):
                if model['id']==choice['target_id']:
                    models[i]=dict(replacement,weight=model['weight'],log_weight=model['log_weight'])
    return models,updated


def prepare(out):
    out.mkdir(exist_ok=False)
    for name in ('inputs','public/requests','public/responses','runs','predictions','evaluation','replay','reports/frozen_source'):
        (out/name).mkdir(parents=True)
    shutil.copy2(ROOT/'configs/local_power.json',out/'scenario.json')
    shutil.copy2(ROOT/'docs/task_allocation_public_v1.md',out/'public/contract.md')
    cfg=load_config(out/'scenario.json')
    known_map=dict(bounds_m=[[0.,0.],[(cfg.map.columns-1)*cfg.map.resolution_m,(cfg.map.rows-1)*cfg.map.resolution_m]],
        buildings=deepcopy(cfg.map.buildings),altitude_m=cfg.aircraft.altitude_m,vertical_clearance_m=cfg.aircraft.vertical_clearance_m)
    ep=read(M3/'runs/geometry_amplitude/episode.json')
    if tuple(d['time_seconds'] for d in ep['decisions'][1:])!=TIMES:raise ValueError('fixed state selection changed')
    for i,now in enumerate(TIMES):save(out/'inputs'/f'{i:02d}.json',make_packet(ep,now,known_map))
    sources=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'tests').rglob('*.py'))
    sources += [ROOT/p for p in ('requirements.txt','docs/task_allocation_pilot_v1.md','docs/task_allocation_public_v1.md')]
    hashes={}
    for path in sources:
        rel=path.relative_to(ROOT);dest=out/'reports/frozen_source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);hashes[str(rel)]=sha(path)
    originals=dict(read(M3/'protocol.json'))
    previous={k:v for k,v in originals.items() if k.startswith('original_') and k.endswith('_sha256')}
    previous['original_m3_sha256']={str(p.relative_to(ROOT)):sha(p) for p in M3.rglob('*') if p.is_file()}
    protocol=dict(version='D2-M4',methods=list(METHODS),times_seconds=list(TIMES),source_sha256=hashes,
        scenario_sha256=sha(out/'scenario.json'),public_contract_sha256=sha(out/'public/contract.md'),
        inputs_sha256={p.name:sha(p) for p in (out/'inputs').glob('*.json')},previous_originals=previous,
        joint_settings=asdict(JointSettings()),max_flights=0,max_sensor_points=0,max_actual_model_calls=3,max_bridge_requests=3,
        max_refinement_calls=6,max_residual_calls=240,max_jacobian_calls=240,max_kernel_terms=30000,
        max_planner_receiver_queries=72000,max_edges=3000,max_evaluation_points=1200,
        max_display_receiver_queries=3024,max_replay_truth_points=1008,
        interpretation='three fixed-state local computation-allocation comparisons; no closed-loop, generalization, or full N/I/P/J result')
    save(out/'protocol.json',protocol)
    save(out/'reports/literature_reading.json',read(ROOT/'docs/references/task_allocation_20261120/reading.json'))
    control(out/'public/control.json',dict(status='prepared',max_requests=3))
    check(out)
    print('PREPARED',out,flush=True)


def check(out):
    protocol=read(out/'protocol.json')
    for name,value in protocol['source_sha256'].items():
        if sha(ROOT/name)!=value or sha(out/'reports/frozen_source'/name)!=value:raise ValueError('frozen source changed: '+name)
    if sha(out/'scenario.json')!=protocol['scenario_sha256'] or sha(out/'public/contract.md')!=protocol['public_contract_sha256']:
        raise ValueError('frozen configuration or public contract changed')
    for name,value in protocol['inputs_sha256'].items():
        if sha(out/'inputs'/name)!=value:raise ValueError('frozen input changed')
    for key,files in protocol['previous_originals'].items():
        base=ROOT/'outputs/interference_joint_gp_d1_20261119_0940' if key=='original_d1_sha256' else ROOT
        for name,value in files.items():
            if sha(base/name)!=value:raise ValueError('previous original changed: '+name)
    return protocol


def forecast_for(packet,models,cfg,geometry):
    now=packet['time_seconds'];history=packet['history']
    gp=FrozenGP(training_points(history,geometry,now,cfg.belief),geometry,now,cfg.belief,cfg.electromagnetic.background_dbm)
    version=digest(dict(history=history,now=now,settings=asdict(cfg.belief)))
    return ScenarioForecast(history,models,now,gp,gp_version=version,joint_settings=JointSettings())


def run_one(out,method):
    protocol=check(out)
    if method not in METHODS:raise ValueError('undeclared method')
    bridge=None
    if method=='allocation_I':
        session=read(out/'reports/session.json')
        if read(out/'public/session_ready.json')['agent_id']!=session['agent_id']:raise ValueError('session not bound')
        bridge=FileAdvisor(out/'public',session['agent_id'])
    folder=out/'runs'/method;folder.mkdir(exist_ok=False)
    cfg=load_config(out/'scenario.json');geometry=ElectromagneticWorld(cfg).geometry
    counts=dict(refinement_calls=0,residual_calls=0,jacobian_calls=0,kernel_terms=0,planner_receiver_queries=0,edges=0,actual_model_calls=0,bridge_requests=0)
    control(out/'public/control.json',dict(status='running' if bridge else 'prepared',max_requests=3))
    for i in range(3):
        packet=read(out/'inputs'/f'{i:02d}.json');error=None
        envelope=bridge.ask('inference',packet) if bridge else dict(raw=numerical_choice(packet),timely=True,actual_llm_request=False)
        try:
            if not envelope['timely']:raise ValueError('response not timely')
            choice=validate(packet,envelope['raw'])
        except (ValueError,TypeError,KeyError) as exc:
            error=str(exc);choice=dict(request_id=packet['request_id'],action='keep',target_id=None,reason='invalid or unavailable response; retain original state')
        models,fit=apply_choice(packet,choice)
        if fit:
            counts['refinement_calls']+=1
            for key in ('residual_calls','jacobian_calls','kernel_terms'):counts[key]+=fit['geometry_refinement']['resources'][key]
        forecast=forecast_for(packet,models,cfg,geometry);work=WorkBudget(1,500)
        candidates=[evaluate(c['path'],packet['state'],forecast,geometry,work,'joint',c['id']) for c in packet['candidates']]
        chosen=best(candidates)
        counts['planner_receiver_queries']+=forecast.counts['receiver_queries'];counts['edges']+=work.edges
        counts['actual_model_calls']+=int(envelope['actual_llm_request']);counts['bridge_requests']+=int(bridge is not None)
        record=dict(method=method,index=i,time_seconds=packet['time_seconds'],request_id=packet['request_id'],public_prefix_sha256=packet['history_sha256'],
            choice=choice,envelope=envelope,parser_accepted=error is None,fallback_error=error,fit=fit,source_models=models,
            forecast_sha256=forecast.fingerprint,candidates=candidates,selected=chosen,
            next_15m_position=endpoint(chosen['path']),original_next_15m_position=endpoint(next(c for c in packet['candidates'] if c['id']==packet['original_selected'])['path']),
            receiver_work=deepcopy(forecast.counts),resources=work.summary())
        save(folder/f'{i:02d}.json',record)
        print('ALLOCATED',method,i,packet['time_seconds'],choice['action'],choice['target_id'],chosen['id'],flush=True)
    for key,value in counts.items():assert value<=protocol['max_'+key]
    save(folder/'status.json',dict(status='completed',counts=counts,record_sha256={f'{i:02d}.json':sha(folder/f'{i:02d}.json') for i in range(3)}))
    if bridge:control(out/'public/control.json',dict(status='completed',max_requests=3,bridge_requests=bridge.count))


def finalize(out):
    protocol=check(out)
    if any((out/'evaluation').iterdir()) or any((out/'predictions').iterdir()):raise ValueError('refuse repeated scientific finalization')
    totals={};records={}
    for method in METHODS:
        status=read(out/'runs'/method/'status.json');assert status['status']=='completed'
        for name,value in status['record_sha256'].items():assert sha(out/'runs'/method/name)==value
        for key,value in status['counts'].items():totals[key]=totals.get(key,0)+value
        records[method]=[read(out/'runs'/method/f'{i:02d}.json') for i in range(3)]
    for key,value in totals.items():assert value<=protocol['max_'+key]
    save(out/'reports/allocation_freeze.json',{str(p.relative_to(out)):sha(p) for tree in ('runs','public') for p in (out/tree).rglob('*') if p.is_file()})
    cfg=load_config(out/'scenario.json');world=ElectromagneticWorld(cfg)
    xy=np.array([[x,y] for y in np.linspace(0,600,16) for x in np.linspace(0,800,21)])
    grids={};display_count=0
    for i in range(3):
        packet=read(out/'inputs'/f'{i:02d}.json')
        for method,models in [('baseline',packet['models'])]+[(m,records[m][i]['source_models']) for m in METHODS]:
            f=forecast_for(packet,models,cfg,world.geometry);p=f.predict(xy,np.full(len(xy),packet['time_seconds']),'joint')
            field=(-80+10*np.log10(p['weights']@p['expected_normalized_power'])).reshape(16,21)
            grids[f'{i:02d}_{method}']=field;display_count+=len(xy)
    assert display_count<=protocol['max_display_receiver_queries']
    np.savez_compressed(out/'predictions/display.npz',**grids)
    save(out/'reports/prediction_freeze.json',{'predictions/display.npz':sha(out/'predictions/display.npz')})
    cache={};rows=[];truth_count=0;truth_grids=[]
    for i in range(3):
        packet=read(out/'inputs'/f'{i:02d}.json');now=packet['time_seconds']
        baseline=next(c for c in packet['candidates'] if c['id']==packet['original_selected']);variants={}
        for method,candidate in [('baseline',baseline)]+[(m,records[m][i]['selected']) for m in METHODS]:
            key=digest(dict(time_seconds=now,path=candidate['path']))
            if key not in cache:
                samples=numerical_samples(dict(motions=[dict(start_time_seconds=now+.5,traversed_m=candidate['path'])],decisions=[]),cfg.aircraft.speed_mps)
                assert truth_count+len(samples)<=protocol['max_evaluation_points']
                for sample in samples:sample['truth_dbm']=float(world.truth(sample['position_m'],sample['time_seconds']))
                truth_count+=len(samples)
                q=math.fsum(s['dt_seconds']*10**((s['truth_dbm']+80)/10) for s in samples);t=math.fsum(s['dt_seconds'] for s in samples)
                cache[key]=dict(path=candidate['path'],samples=samples,true_flight_T=t,true_flight_Q=q,true_flight_T_plus_Q=t+q)
            variants[method]=dict(cache[key],selected_id=candidate['id'],predicted_score=candidate['score']['score_seconds'],route_cache_key=key)
        rows.append(dict(index=i,time_seconds=now,variants=variants,gains={m:variants['baseline']['true_flight_T_plus_Q']-variants[m]['true_flight_T_plus_Q'] for m in METHODS}))
        truth_grids.append(world.truth(xy,now).reshape(16,21))
    assert len(xy)*len(truth_grids)<=protocol['max_replay_truth_points']
    np.savez_compressed(out/'evaluation/display_truth.npz',truth_dbm=np.array(truth_grids))
    result=dict(rows=rows,counts=totals,evaluation_truth_points=truth_count,unique_evaluated_routes=len(cache),
        display_receiver_queries=display_count,replay_truth_points=len(xy)*len(truth_grids),new_flights=0,new_sensor_points=0,
        scope=protocol['interpretation'],initial_common_hover_excluded=True)
    save(out/'reports/results.json',result)
    render(out,cfg,records,result,grids,truth_grids)
    check(out)
    save(out/'reports/completion_manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def render(out,cfg,records,result,grids,truth_grids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Circle
    from PIL import Image
    images=[]
    protocol=read(out/'protocol.json');version=protocol['version']
    title='固定状态的两阶段机制提议' if version=='D2-M5' else '固定状态的单次细化分配'
    design=protocol.get('design_path','task_allocation_pilot_v1.md')
    page=f'<!doctype html><meta charset="utf-8"><title>{version}任务相关假设工具组织</title><style>body{{font:17px/1.7 system-ui;max-width:1350px;margin:auto}}img{{width:100%}}td,th{{padding:6px;border:1px solid #bbb}}</style><h1>{version}：{title}</h1><p>无新飞行、无新点测。三个既有状态的冻结整路反事实，不是闭环收益；左上真图仅用于评价。共享输入/工具/预算，N为任务相关数值组织，I为独立LLM组织；连续参数由数值工具求解。初始共同悬停不计入本页反事实指标。底色为联合混合预测，紫色叉号及虚圆只展示一个替代世界的基元，不是真源定位；其他世界互斥而不叠加。</p><p><a href="../reports/results.json">完整结果</a> · <a href="../protocol.json">合同和预算</a> · <a href="../../../docs/{design}">设计</a></p>'
    for i,row in enumerate(result['rows']):
        packet=read(out/'inputs'/f'{i:02d}.json');now=packet['time_seconds'];fig,axes=plt.subplots(2,2,figsize=(14,10),dpi=100)
        trail=np.array([p['position_m'] for p in packet['history']])
        variants=[('truth',truth_grids[i]),('baseline',grids[f'{i:02d}_baseline'])]+[(m,grids[f'{i:02d}_{m}']) for m in METHODS]
        for ax,(method,field) in zip(axes.ravel(),variants):
            im=ax.imshow(field,origin='lower',extent=(0,800,0,600),vmin=-100,vmax=-55,cmap='YlOrRd')
            for b in cfg.map.buildings:ax.add_patch(Polygon(b['footprint_m'],color='#303030' if b['height_m']+cfg.aircraft.vertical_clearance_m>=cfg.aircraft.altitude_m else '#abc4cf',zorder=3))
            ax.plot(*trail.T,color='#125ec9',lw=2,zorder=4);ax.plot(*cfg.map.goal_m,'b*',ms=10,zorder=5)
            if method=='truth':
                for j,source in enumerate(cfg.electromagnetic.fields,1):
                    if source['start_seconds']<=now<source['end_seconds']:
                        center=np.array(source['center_m'])+(now-source['start_seconds'])*np.array(source['velocity_mps'])
                        ax.plot(*center,'k^',ms=9,zorder=6);ax.annotate(f'T{j}',center,xytext=(5,-10),textcoords='offset points')
                ax.set_title('True field at frozen state (evaluation only)')
            else:
                candidate=row['variants'][method];route=np.array(candidate['path'])
                ax.plot(*route.T,'--',color='#007c91',lw=2,zorder=5)
                models=packet['models'] if method=='baseline' else records[method][i]['source_models']
                model=max(models,key=lambda m:m['weight'])
                for j,basis in enumerate(model['bases'],1):
                    center=np.array(basis['center_m'])+(now-model['reference_time_seconds'])*np.array(basis['velocity_mps'])
                    ax.plot(*center,'X',color='#a000ab',mec='white',ms=9,zorder=6);ax.add_patch(Circle(center,basis['scale_m'],fill=False,color='#a000ab',ls=':',zorder=4))
                    ax.annotate(f'B{j}',center,xytext=(5,5),textcoords='offset points')
                action='shared original' if method=='baseline' else records[method][i]['choice']['action']
                action+=f" | shown world={model['id']} w={model['weight']:.3f}"
                ax.set_title(f"{method}: {candidate['selected_id']} | true remaining T+Q={candidate['true_flight_T_plus_Q']:.2f}\n{action}",fontsize=10)
            ax.set(xlim=(0,800),ylim=(0,600),xlabel='x (m)',ylabel='y (m)')
        fig.suptitle(f'{version} | t={now:g}s | FROZEN-ROUTE COUNTERFACTUAL: NO NEW FLIGHT\nBlue: previously observed trail. Dashed: full hypothetical remaining route. Purple: ONE alternative source world.',fontsize=12)
        fig.tight_layout(rect=(0,0,.93,.94));fig.colorbar(im,cax=fig.add_axes([.95,.16,.012,.65]),label='dBm (higher = stronger)')
        name=f'state_{i:02d}.png';fig.savefig(out/'replay'/name)
        fig.canvas.draw();images.append(Image.fromarray(np.array(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        page+=f'<h2>t={now:g}秒</h2><table><tr><th>组织</th><th>分配动作</th><th>冻结剩余T+Q</th><th>相对原路线收益</th></tr>'
        for method in ('baseline',)+METHODS:
            action='共同原状态' if method=='baseline' else str(records[method][i]['choice'])
            page+=f"<tr><td>{method}</td><td>{html.escape(action)}</td><td>{row['variants'][method]['true_flight_T_plus_Q']:.3f}</td><td>{row['gains'].get(method,0.):+.3f}</td></tr>"
        page+='</table>'
        if version=='D2-M5':
            for method in METHODS:
                rec=records[method][i]
                page+=f'<details><summary>{method} 两阶段请求、反馈与工具接受</summary><ul>'
                for rnd in rec['rounds']:
                    text=dict(phase=rnd['request']['phase'],plan=rnd['accepted_plan'],timely=rnd['envelope']['timely'],
                        fallback=rnd['fallback_error'],pilot_tail_rmse_db=(rnd['pilot_feedback'] or {}).get('tail_rmse_db'),
                        geometry=[f['geometry_refinement']['accepted'] for f in rnd['fits'] if 'geometry_refinement' in f])
                    page+='<li>'+html.escape(json.dumps(text,ensure_ascii=False))+'</li>'
                page+='</ul><p>全部世界及基元：</p><pre>'+html.escape(json.dumps(rec['source_models'],ensure_ascii=False,indent=2))+'</pre></details>'
        page+=f'<img src="{name}">'
    images[0].save(out/'replay/fixed_states.gif',save_all=True,append_images=images[1:],duration=2000,loop=0)
    (out/'replay/index.html').write_text(page+'<p>仅三个已见开发状态；不作泛化或LLM优越结论。</p>')
    (out/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><a href="replay/index.html">{version}固定状态诊断（无新飞行）</a>')
