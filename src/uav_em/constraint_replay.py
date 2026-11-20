"""D2-M6 conditional replay: fixed M5 plans, optional explicit-region solver."""
from copy import deepcopy
import html
import json
import math
from pathlib import Path
import shutil

import numpy as np

from . import task_allocation as shared
from . import task_proposals
from .config import load_config
from .environment import ElectromagneticWorld
from .event_demo import read,save,sha
from .event_inference import digest
from .feasible_scenario import best,evaluate
from .joint_experiment import numerical_samples
from .repair_search import WorkBudget
from .source_geometry import constrained_refine_fit

ROOT=shared.ROOT
M5=ROOT/'outputs/interference_task_proposals_d2_m5_20261120'
METHODS=shared.METHODS


def restore_seed(fit):
    result=deepcopy(fit);trace=result.pop('geometry_refinement')
    result['model']=deepcopy(trace['seed_model']);result['fit_objective']=trace['previous_objective']
    soft=result.get('gp_soft_constraints')
    result['gp_shape_mse']=((result['fit_objective']-result['model']['fit_mse'])/soft['strength']
        if soft and soft['position_m'] else None)
    return result


def replay_fits(packet,original):
    fits=[];updates=[]
    public=dict(goal_m=packet['state']['goal_m'],map=dict(bounds_m=packet['known_map']['bounds_m']))
    for ri,rnd in enumerate(original['rounds']):
        for j,old in enumerate(rnd['fits']):
            result=deepcopy(old)
            if 'geometry_refinement' in old:
                if rnd['request']['phase']!='refine' or rnd['accepted_plan']['geometry_job']!=j:
                    raise ValueError('only original full geometry target may be recomputed')
                seed=restore_seed(old);result=constrained_refine_fit(seed,public)
                updates.append(dict(round_index=ri,job_index=j,original_fit_sha256=digest(old),
                    seed_fit_sha256=digest(seed),old_geometry=deepcopy(old['geometry_refinement']),
                    new_geometry=deepcopy(result['geometry_refinement'])))
            fits.append(result)
    models,committed=task_proposals.commit_models(packet,fits,keep=not original['pool_committed'])
    return fits,models,committed,updates


def prepare(out):
    out.mkdir(exist_ok=False)
    for name in ('inputs','public','runs','predictions','evaluation','replay','reports/frozen_source'):
        (out/name).mkdir(parents=True)
    shutil.copy2(M5/'scenario.json',out/'scenario.json')
    shutil.copy2(ROOT/'docs/constrained_geometry_pilot_v1.md',out/'public/contract.md')
    old=read(M5/'protocol.json')
    for i in range(3):
        src=M5/'inputs'/f'{i:02d}.json'
        if sha(src)!=old['inputs_sha256'][src.name]:raise ValueError('M5 input changed')
        shutil.copy2(src,out/'inputs'/src.name)
        for method in METHODS:
            src=M5/'runs'/method/f'{i:02d}.json'
            if sha(src)!=read(M5/'runs'/method/'status.json')['record_sha256'][src.name]:raise ValueError('M5 record changed')
            shutil.copy2(src,out/'inputs'/f'{method}_{i:02d}.json')
    sources=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'tests').rglob('*.py'))
    sources += [ROOT/p for p in ('requirements.txt','docs/constrained_geometry_pilot_v1.md')]
    hashes={}
    for path in sources:
        rel=path.relative_to(ROOT);dest=out/'reports/frozen_source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);hashes[str(rel)]=sha(path)
    previous=deepcopy(old['previous_originals'])
    previous['original_m5_sha256']={str(p.relative_to(ROOT)):sha(p) for p in M5.rglob('*') if p.is_file()}
    protocol=dict(version='D2-M6',methods=list(METHODS),times_seconds=list(shared.TIMES),source_sha256=hashes,
        scenario_sha256=sha(out/'scenario.json'),public_contract_sha256=sha(out/'public/contract.md'),
        inputs_sha256={p.name:sha(p) for p in (out/'inputs').glob('*.json')},previous_originals=previous,
        max_refinement_calls=5,max_solver_calls=5,max_residual_calls=200,max_jacobian_calls=200,max_kernel_terms=25000,
        max_planner_receiver_queries=72000,max_edges=3000,max_actual_model_calls=0,max_bridge_requests=0,
        max_display_receiver_queries=2016,max_evaluation_points=1200,max_new_evaluated_routes=6,max_replay_truth_points=0,
        max_flights=0,max_sensor_points=0,max_base_fits=0,
        interpretation='fixed M5 proposals, conditional shared-tool replay; not a new prospective LLM or closed-loop experiment')
    save(out/'protocol.json',protocol)
    save(out/'reports/literature_reading.json',read(ROOT/'docs/references/constrained_geometry_20261120/reading.json'))
    shared.check(out);print('PREPARED',out,flush=True)


def run_one(out,method):
    protocol=shared.check(out)
    if method not in METHODS:raise ValueError('undeclared method')
    folder=out/'runs'/method;folder.mkdir(exist_ok=False)
    cfg=load_config(out/'scenario.json');geometry=ElectromagneticWorld(cfg).geometry
    counts={k:0 for k in ('refinement_calls','solver_calls','residual_calls','jacobian_calls','kernel_terms',
        'planner_receiver_queries','edges','actual_model_calls','bridge_requests')}
    for i in range(3):
        packet=read(out/'inputs'/f'{i:02d}.json');original=read(out/'inputs'/f'{method}_{i:02d}.json')
        fits,models,committed,updates=replay_fits(packet,original)
        counts['refinement_calls']+=len(updates)
        for update in updates:
            for key,value in update['new_geometry']['resources'].items():counts[key]+=value
        forecast=shared.forecast_for(packet,models,cfg,geometry);work=WorkBudget(1,500)
        candidates=[evaluate(c['path'],packet['state'],forecast,geometry,work,'joint',c['id']) for c in packet['candidates']]
        chosen=best(candidates)
        counts['planner_receiver_queries']+=forecast.counts['receiver_queries'];counts['edges']+=work.edges
        record=dict(method=method,index=i,time_seconds=packet['time_seconds'],
            original_record_sha256=sha(out/'inputs'/f'{method}_{i:02d}.json'),
            frozen_proposals=[dict(phase=r['request']['phase'],request_id=r['request']['request_id'],plan=r['accepted_plan']) for r in original['rounds']],
            original_decisions_reused=True,new_actual_model_calls=0,pool_committed=committed,fits=fits,
            geometry_updates=updates,source_models=models,forecast_sha256=forecast.fingerprint,
            candidates=candidates,selected=chosen,next_15m_position=shared.endpoint(chosen['path']),
            old_next_15m_position=original['next_15m_position'],receiver_work=deepcopy(forecast.counts),resources=work.summary())
        save(folder/f'{i:02d}.json',record)
        print('REPLAYED',method,i,'geometry',[(u['old_geometry']['accepted'],u['new_geometry']['accepted']) for u in updates],chosen['id'],flush=True)
    for key,value in counts.items():assert value<=protocol['max_'+key],(key,value)
    save(folder/'status.json',dict(status='completed',counts=counts,
        record_sha256={f'{i:02d}.json':sha(folder/f'{i:02d}.json') for i in range(3)}))


def finalize(out):
    protocol=shared.check(out)
    if any((out/'evaluation').iterdir()) or any((out/'predictions').iterdir()):raise ValueError('refuse repeated finalization')
    records={};totals={}
    for method in METHODS:
        status=read(out/'runs'/method/'status.json');assert status['status']=='completed'
        for name,value in status['record_sha256'].items():assert sha(out/'runs'/method/name)==value
        records[method]=[read(out/'runs'/method/f'{i:02d}.json') for i in range(3)]
        for key,value in status['counts'].items():totals[key]=totals.get(key,0)+value
    for key,value in totals.items():assert value<=protocol['max_'+key]
    save(out/'reports/route_freeze.json',{str(p.relative_to(out)):sha(p) for p in (out/'runs').rglob('*') if p.is_file()})
    cfg=load_config(out/'scenario.json');world=ElectromagneticWorld(cfg)
    xy=np.array([[x,y] for y in np.linspace(0,600,16) for x in np.linspace(0,800,21)])
    grids={};display_count=0
    for i in range(3):
        packet=read(out/'inputs'/f'{i:02d}.json')
        for method in METHODS:
            forecast=shared.forecast_for(packet,records[method][i]['source_models'],cfg,world.geometry)
            pred=forecast.predict(xy,np.full(len(xy),packet['time_seconds']),'joint')
            grids[f'{i:02d}_{method}']=(-80+10*np.log10(pred['weights']@pred['expected_normalized_power'])).reshape(16,21)
            display_count+=len(xy)
    assert display_count<=protocol['max_display_receiver_queries']
    np.savez_compressed(out/'predictions/display.npz',**grids)
    save(out/'reports/prediction_freeze.json',{'predictions/display.npz':sha(out/'predictions/display.npz')})
    # Evaluation caches become readable only after all new routes/fields are frozen.
    old_result=read(M5/'reports/results.json');cache={}
    for row in old_result['rows']:
        for variant in row['variants'].values():
            key=digest(dict(time_seconds=row['time_seconds'],path=variant['path']))
            assert key==variant['route_cache_key'];cache[key]=deepcopy(variant)
    old_grids=np.load(M5/'predictions/display.npz');truth_grids=np.load(M5/'evaluation/display_truth.npz')['truth_dbm']
    rows=[];new_points=0;new_routes=0;reused=0
    for i,oldrow in enumerate(old_result['rows']):
        now=oldrow['time_seconds'];variants={k:deepcopy(v) for k,v in oldrow['variants'].items()}
        for method in METHODS:
            chosen=records[method][i]['selected'];key=digest(dict(time_seconds=now,path=chosen['path']))
            if key in cache:reused+=1
            else:
                samples=numerical_samples(dict(motions=[dict(start_time_seconds=now+.5,traversed_m=chosen['path'])],decisions=[]),cfg.aircraft.speed_mps)
                assert new_points+len(samples)<=protocol['max_evaluation_points'] and new_routes<protocol['max_new_evaluated_routes']
                for sample in samples:sample['truth_dbm']=float(world.truth(sample['position_m'],sample['time_seconds']))
                new_points+=len(samples);new_routes+=1
                t=math.fsum(s['dt_seconds'] for s in samples);q=math.fsum(s['dt_seconds']*10**((s['truth_dbm']+80)/10) for s in samples)
                cache[key]=dict(path=chosen['path'],samples=samples,true_flight_T=t,true_flight_Q=q,true_flight_T_plus_Q=t+q,route_cache_key=key)
            variants['constrained_'+method]=dict(cache[key],selected_id=chosen['id'],predicted_score=chosen['score']['score_seconds'])
        rows.append(dict(index=i,time_seconds=now,variants=variants,
            gain_vs_same_original_method={m:variants[m]['true_flight_T_plus_Q']-variants['constrained_'+m]['true_flight_T_plus_Q'] for m in METHODS}))
    result=dict(rows=rows,counts=totals,new_flights=0,new_sensor_points=0,new_decision_calls=0,
        display_receiver_queries=display_count,new_evaluation_truth_points=new_points,new_evaluated_routes=new_routes,
        reused_route_integrals=reused,reused_truth_grid_points=int(truth_grids.size),new_replay_truth_points=0,
        initial_common_hover_excluded=True,scope=protocol['interpretation'])
    save(out/'reports/results.json',result)
    render(out,cfg,records,result,grids,old_grids,truth_grids)
    shared.check(out)
    save(out/'reports/completion_manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(dict(counts=totals,new_truth_points=new_points,new_routes=new_routes,
        rows=[dict(time=r['time_seconds'],costs={m:v['true_flight_T_plus_Q'] for m,v in r['variants'].items()},gain=r['gain_vs_same_original_method']) for r in rows]),indent=2),flush=True)


def render(out,cfg,records,result,grids,old_grids,truth_grids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Polygon
    from PIL import Image
    images=[]
    page='<!doctype html><meta charset="utf-8"><title>M6区域约束工具复核</title><style>body{font:17px/1.7 system-ui;max-width:1500px;margin:auto}img{width:100%}td,th{padding:6px;border:1px solid #bbb}pre{white-space:pre-wrap}</style><h1>M6：固定原提议，复核区域约束工具</h1><p>固定状态冻结剩余路线反事实，无新飞行/点测/决策LLM。M5原计划、基础拟合与原keep均固定，只重算已指定的几何作业。底图为实际联合预测，紫色只标一个被修订的互斥世界，不是真定位。</p><p><a href="../../../docs/constrained_geometry_results_20261120.md">结果与归因边界</a> · <a href="../reports/results.json">完整数据</a> · <a href="fixed_states.gif">三状态GIF</a></p>'
    for i,row in enumerate(result['rows']):
        packet=read(out/'inputs'/f'{i:02d}.json');now=packet['time_seconds'];trail=np.array([p['position_m'] for p in packet['history']])
        fig,axes=plt.subplots(2,3,figsize=(18,11),dpi=100)
        panels=[('truth','True field (evaluation only)',truth_grids[i]),('allocation_N','M5 numerical',old_grids[f'{i:02d}_allocation_N']),
            ('constrained_allocation_N','M6 numerical',grids[f'{i:02d}_allocation_N']),('baseline','Original common pool',old_grids[f'{i:02d}_baseline']),
            ('allocation_I','M5 LLM proposals',old_grids[f'{i:02d}_allocation_I']),('constrained_allocation_I','M6 same LLM proposals',grids[f'{i:02d}_allocation_I'])]
        for ax,(key,label,field) in zip(axes.ravel(),panels):
            im=ax.imshow(field,origin='lower',extent=(0,800,0,600),vmin=-100,vmax=-55,cmap='YlOrRd')
            for b in cfg.map.buildings:ax.add_patch(Polygon(b['footprint_m'],color='#303030' if b['height_m']+cfg.aircraft.vertical_clearance_m>=cfg.aircraft.altitude_m else '#abc4cf',zorder=3))
            ax.plot(*trail.T,color='#125ec9',lw=2,zorder=4);ax.plot(*cfg.map.goal_m,'b*',ms=9,zorder=5)
            if key=='truth':
                for j,s in enumerate(cfg.electromagnetic.fields,1):
                    if s['start_seconds']<=now<s['end_seconds']:
                        c=np.asarray(s['center_m'])+(now-s['start_seconds'])*np.asarray(s['velocity_mps'])
                        ax.plot(*c,'k^',ms=8,zorder=6);ax.annotate(f'T{j}',c,xytext=(4,3),textcoords='offset points')
                ax.set_title(label)
            else:
                variant=row['variants'][key];ax.plot(*np.asarray(variant['path']).T,'--',color='#007c91',lw=2,zorder=5)
                if key=='baseline':models=packet['models'];target=None
                else:
                    method=key.replace('constrained_','');new=key.startswith('constrained_')
                    old=read(out/'inputs'/f'{method}_{i:02d}.json')
                    models=records[method][i]['source_models'] if new else old['source_models']
                    target=next((f['model']['id'] for r in old['rounds'] for f in r['fits'] if 'geometry_refinement' in f),None)
                model=next((m for m in models if m['id']==target),max(models,key=lambda m:m['weight']))
                for j,b in enumerate(model['bases'],1):
                    c=np.asarray(b['center_m'])+(now-model['reference_time_seconds'])*np.asarray(b['velocity_mps'])
                    ax.plot(*c,'X',color='#a000ab',mec='white',ms=8,zorder=6);ax.add_patch(Circle(c,b['scale_m'],fill=False,ls=':',color='#a000ab',zorder=4))
                    ax.annotate(f'B{j}',c,xytext=(4,4),textcoords='offset points')
                ax.set_title(f"{label}: {variant['selected_id']} | true T+Q={variant['true_flight_T_plus_Q']:.2f}\nworld={model['id']} w={model['weight']:.3f}",fontsize=10)
            ax.set(xlim=(0,800),ylim=(0,600),xlabel='x (m)',ylabel='y (m)')
        fig.suptitle(f'M6 | t={now:g}s | FIXED M5 PROPOSALS | FROZEN-ROUTE COUNTERFACTUAL: NO NEW FLIGHT',fontsize=14)
        fig.subplots_adjust(left=.045,right=.91,bottom=.06,top=.9,wspace=.18,hspace=.28)
        fig.colorbar(im,cax=fig.add_axes([.935,.14,.012,.69]),label='dBm (higher = stronger)')
        name=f'state_{i:02d}.png';fig.savefig(out/'replay'/name)
        fig.canvas.draw();images.append(Image.fromarray(np.array(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        page+=f'<h2>t={now:g}秒</h2><table><tr><th>方法</th><th>路线</th><th>T</th><th>Q</th><th>T+Q</th></tr>'
        for key,v in row['variants'].items():page+=f"<tr><td>{key}</td><td>{v['selected_id']}</td><td>{v['true_flight_T']:.3f}</td><td>{v['true_flight_Q']:.3f}</td><td>{v['true_flight_T_plus_Q']:.3f}</td></tr>"
        page+='</table>'
        for method in METHODS:
            page+='<details><summary>'+method+' 几何求解回执</summary><pre>'+html.escape(json.dumps([
                dict(old_accepted=u['old_geometry']['accepted'],new={k:u['new_geometry'].get(k) for k in
                    ('accepted','solver_success','region_eligible','previous_objective','candidate_objective','error','resources')})
                for u in records[method][i]['geometry_updates']],ensure_ascii=False,indent=2))+'</pre></details>'
        page+=f'<img src="{name}">'
    images[0].save(out/'replay/fixed_states.gif',save_all=True,append_images=images[1:],duration=2400,loop=0)
    (out/'replay/index.html').write_text(page+'<p>仅三个已见开发状态；工具效果不能直接当作新的LLM组织或闭环收益。</p>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><a href="replay/index.html">M6约束工具复核（无新飞行）</a>')
