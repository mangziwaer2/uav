"""Frozen development study D1. Invoked only through run_experiment.py joint."""
from copy import deepcopy
from dataclasses import asdict
import html
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .config import load_config
from .environment import ElectromagneticWorld
from .event_demo import FileAdvisor, read, save, sha, control, execution_samples
from .event_inference import digest
from .joint_prediction import JointSettings
from .point_gp import FrozenGP, training_points
from .scenario_loop import run as fly
from .scenario_planning import ScenarioForecast
from .equivalent_field import normalized_expected_power
from .motion import advance_route

ROOT=Path(__file__).resolve().parents[2]
METHODS={
    'source_numeric':('mixture_numeric','mixture',False,False,False),
    'joint_point_numeric':('mixture_numeric','joint',False,False,False),
    'joint_N':('mixture_numeric','joint',True,False,False),
    'joint_I':('mixture_numeric','joint',True,True,False),
    'joint_P':('mixture_llm','joint',True,False,True),
    'joint_J':('mixture_llm','joint',True,True,True),
    'gp_numeric':('gp_numeric','gp',False,False,False),
    'persistence_numeric':('persistence_numeric','persistence',False,False,False),
}


def prepare(out):
    out.mkdir(exist_ok=False)
    for name in ('runs','public','reports/frozen_source','predictions','evaluation','replay'):
        (out/name).mkdir(parents=True)
    shutil.copy2(ROOT/'configs/local_power.json',out/'scenario.json')
    public_text=(ROOT/'docs/event_demo_public_contract.md').read_text()
    public_text+='\n\nD1新增：所有核心方法共享因果GP软约束与联合预测，GP派生点不是新增传感证据。前方支持内均值越过−78dBm会请求咨询；该值是开发任务线。路径接口包含joint_settings、GP支持及该次推断工具结果；用这些信息组织池外搜索。同步回调冻结物理状态。只处理你被指定的本方法public目录；禁止读取其他方法。\n'
    contracts={}
    for method in ('joint_I','joint_P','joint_J'):
        folder=out/'public'/method
        for tree in ('requests','responses'):(folder/tree).mkdir(parents=True)
        (folder/'contract.md').write_text(public_text)
        cap=8 if method=='joint_I' else 4 if method=='joint_P' else 12
        control(folder/'control.json',dict(status='prepared',max_requests=cap))
        contracts[method]=sha(folder/'contract.md')
    sources=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'tests').rglob('*.py'))
    sources += [ROOT/'requirements.txt',ROOT/'docs/joint_gp_pilot_v1.md']
    hashes={}
    for path in sources:
        rel=path.relative_to(ROOT);dest=out/'reports/frozen_source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        hashes[str(rel)]=sha(path)
    original=ROOT/'outputs/interference_scenario_loop_20261118'
    old={str(p.relative_to(ROOT)):sha(p) for p in original.rglob('*') if p.is_file()}
    save(out/'protocol.json',dict(version='D1',methods=list(METHODS),source_sha256=hashes,
        scenario_sha256=sha(out/'scenario.json'),public_contract_sha256=contracts,
        joint_settings=asdict(JointSettings()),max_flights=8,max_events_per_flight=4,
        max_actual_model_calls=24,max_bridge_requests=24,max_sensor_points=824,
        max_jobs=72,max_solver_calls=144,max_residual_calls=5760,max_samples=28000,max_edges=400000,
        max_evaluation_points=4000,max_replay_truth_points=86016,max_display_receiver_queries=86016,
        max_monitor_receiver_queries=9000,max_soft_gp_queries=960,max_prequential_queries=816,
        max_planner_receiver_queries=40000000,max_frames_per_flight=32,display_shape=[16,21],
        original_c1_sha256=old,
        interpretation='single seen development scene; current numerical feedback planner, not a final strong optimization baseline; no generalization or superiority claim',
        timing='supported forward mean crossing; synchronous frozen callbacks; fixed 0.5s hover'))
    print('PREPARED',out,flush=True)


def check(out):
    protocol=read(out/'protocol.json')
    for rel,expected in protocol['source_sha256'].items():
        if sha(ROOT/rel)!=expected:raise ValueError('source changed after freeze: '+rel)
    if sha(out/'scenario.json')!=protocol['scenario_sha256']:raise ValueError('scenario changed')
    for method,expected in protocol['public_contract_sha256'].items():
        if sha(out/'public'/method/'contract.md')!=expected:raise ValueError('public contract changed')
    return protocol


def run_one(out,method):
    protocol=check(out)
    if method not in protocol['methods']:raise ValueError('undeclared method')
    folder=out/'runs'/method;folder.mkdir(exist_ok=False)
    underlying,predictor,soft,infer_llm,path_llm=METHODS[method]
    bridge=None
    if infer_llm or path_llm:
        public=out/'public'/method
        session=read(public/'session_ready.json');agent=session['agent_id']
        if read(out/'reports'/f'{method}_session.json')['agent_id']!=agent:raise ValueError('session binding mismatch')
        bridge=FileAdvisor(public,agent)
        control(public/'control.json',dict(status='running',max_requests=8 if method=='joint_I' else 4 if method=='joint_P' else 12))
    def emit(row):
        with (folder/'decisions.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print('DECISION',method,row['index'],'t',row['time_seconds'],row['selected_id'],flush=True)
    started=time.monotonic()
    try:
        ep=fly(load_config(out/'scenario.json'),underlying,
            advisor=(lambda info,index:bridge.ask('path',info)) if path_llm else None,
            inference_advisor=(lambda request:bridge.ask('inference',request)) if infer_llm else None,
            timing='forward',joint_settings=JointSettings(**protocol['joint_settings']),
            forecast_method=predictor,soft_fit=soft,emit=emit)
        ep['study_method']=method;ep['wall_seconds']=time.monotonic()-started
        save(folder/'episode.json',ep)
        save(folder/'status.json',dict(status='completed',episode_sha256=sha(folder/'episode.json')))
        if bridge:control(out/'public'/method/'control.json',dict(status='completed',bridge_requests=bridge.count))
        print('COMPLETE',method,ep['metrics'],flush=True)
    except BaseException as exc:
        save(folder/'failure.json',dict(error=repr(exc)))
        if bridge:control(out/'public'/method/'control.json',dict(status='failed',error=repr(exc)))
        raise


def display_indices(ep,limit=32):
    frames=ep['frames'];mandatory={0,len(frames)-1}
    for i,f in enumerate(frames):
        if f['phase']=='after_decision':
            mandatory.add(i)
            mandatory.add(i-1)
    if len(mandatory)>limit:raise ValueError('cannot omit an event to fit replay budget')
    extras=[i for i in range(len(frames)) if i not in mandatory]
    count=min(limit-len(mandatory),len(extras))
    if count:mandatory.update(extras[int(i)] for i in np.linspace(0,len(extras)-1,count))
    return sorted(mandatory)


def display_predictions(ep,cfg,geometry,settings):
    high=geometry.high
    xy=np.array([[x,y] for y in np.linspace(0,high[1],16) for x in np.linspace(0,high[0],21)])
    indices=display_indices(ep);images=[];supports=[];gains=[];metadata=[];query_count=0
    for index in indices:
        frame=ep['frames'][index];now=frame['time_seconds'];observed=frame['observation_time_seconds']
        prefix=[p for p in ep['observations'] if p['time_seconds']<=observed]
        gp=FrozenGP(training_points(prefix,geometry,observed,cfg.belief),geometry,observed,cfg.belief,cfg.electromagnetic.background_dbm)
        version=digest(dict(history=prefix,now=observed,settings=asdict(cfg.belief)))
        forecast=ScenarioForecast(prefix,frame['hypotheses'],observed,gp,gp_version=version,joint_settings=settings)
        d=forecast.predict(xy,np.full(len(xy),now),ep['planner_method'])
        powers=d['expected_normalized_power'] if 'expected_normalized_power' in d else normalized_expected_power(d['means'],d['stds'],-80.)
        image=-80.+10*np.log10(d['weights']@powers)
        if d['supported'] is None:
            _,_,supported=gp.predict(xy,np.full(len(xy),now))
        else:supported=d['supported']
        if ep['planner_method']=='gp':image=np.where(supported,image,np.nan)
        images.append(image.reshape(16,21));supports.append(np.asarray(supported).reshape(16,21))
        gains.append(np.asarray(d.get('gp_weight',np.zeros(len(xy)))).reshape(16,21))
        metadata.append(dict(index=index,time_seconds=now,phase=frame['phase'],forecast_sha256=forecast.fingerprint,
            top_hypotheses=sorted(frame['hypotheses'],key=lambda m:(-m['weight'],m['id']))[:3],
            quantity='expected linear power converted to dBm; uncalibrated uncertainty included',
            unsupported='source/unknown alternatives' if ep['planner_method']=='joint' else 'declared reference convention'))
        query_count+=len(xy)
    return xy,dict(indices=np.array(indices),actual_dbm=np.array(images),supported=np.array(supports),gp_weight=np.array(gains)),metadata,query_count


def numerical_samples(ep,speed):
    # Each old execution sample spans <=10m; split further to respect <=0.5s.
    rows=[]
    for motion in ep['motions']:
        elapsed=0.
        for a,b in zip(motion['traversed_m'],motion['traversed_m'][1:]):
            a,b=np.asarray(a),np.asarray(b);duration=float(np.linalg.norm(b-a))/speed
            count=max(1,math.ceil(duration/.5))
            for i in range(count):
                fraction=(i+.5)/count
                rows.append(dict(position_m=(a+fraction*(b-a)).tolist(),time_seconds=motion['start_time_seconds']+elapsed+fraction*duration,
                                 dt_seconds=duration/count,kind='flight'))
            elapsed+=duration
    rows += [dict(position_m=d['public']['request']['state']['position_m'],time_seconds=d['time_seconds']+.25,
                  dt_seconds=.5,kind='fixed_hover') for d in ep['decisions']]
    return rows


def audit_before_truth(out,protocol,episodes):
    counts=dict(flights=len(episodes),sensor_points=0,jobs=0,solver_calls=0,residual_calls=0,samples=0,edges=0,
                actual_model_calls=0,bridge_requests=0,monitor_receiver_queries=0,soft_gp_queries=0,prequential_queries=0,
                planner_receiver_queries=0)
    for method,ep in episodes.items():
        assert len(ep['decisions'])<=protocol['max_events_per_flight']
        assert len(ep['joint_prediction_records'])==len(ep['observations'])-1
        counts['sensor_points']+=len(ep['observations']);counts['prequential_queries']+=len(ep['joint_prediction_records'])
        for name,metric in [('jobs','inferred_jobs'),('solver_calls','solver_calls'),('residual_calls','residual_calls'),
                            ('samples','total_samples'),('edges','total_edges'),('actual_model_calls','actual_model_calls')]:
            counts[name]+=ep['metrics'][metric]
        for index,row in enumerate(ep['joint_prediction_records'],1):
            assert row['issued']['history_sha256']==digest(ep['observations'][:index])
            assert row['issued']['forecast_at_seconds']<row['observation']['time_seconds']
            assert row['issued']['predicted_before_observation']
        counts['monitor_receiver_queries']+=sum(len(r['queries']['time_seconds']) for r in ep['trigger_records'])
        for event in ep['inferences']:
            for rnd in event['rounds']:
                soft=rnd['request'].get('gp_soft_constraints')
                if soft:
                    counts['soft_gp_queries']+=soft['gp_queries']
                    assert soft['independent_evidence_points']==0
                    expected=rnd['request']['pilot_training_cutoff_seconds'] if rnd['request']['phase']=='pilot' else event['time_seconds']
                    assert soft['prefix_end_seconds']==expected
        for d in ep['decisions']:
            counts['planner_receiver_queries']+=d['receiver_work']['receiver_queries']+d['public']['additional_gp_receiver_queries']
            valid=[c for c in d['candidates'] if c['feasible']]
            winner=min(valid,key=lambda c:(c['score']['score_seconds'],sum(math.dist(a,b) for a,b in zip(c['path'],c['path'][1:])),c['id']))
            assert winner['id']==d['selected_id']
            for c in valid:
                s=c['score'];value=math.fsum(x['weight']*x['power_proxy_seconds'] for x in s['scenario_scores'])
                assert math.isclose(value,s['power_proxy_seconds'],rel_tol=1e-10,abs_tol=1e-8)
                assert math.isclose(s['score_seconds'],s['remaining_flight_seconds']+value,rel_tol=1e-10)
            if ep['planner_method']=='joint':
                assert len({c['score']['forecast_sha256'] for c in valid})==1
        public=out/'public'/method
        if public.exists():
            requests=list((public/'requests').glob('*.json'));counts['bridge_requests']+=len(requests)
            actual=0
            for path in requests:
                req=read(path);host=read(public/'responses'/f'{path.stem}.host.json')
                assert req['request_id']==host['request_id']
                if host['started']:
                    receipt=read(public/'responses'/f'{path.stem}.started.json')
                    assert receipt['agent_id']==host['agent_id'] and receipt['request_id']==req['request_id']
                    actual+=1
                if host['raw'] is not None:assert sha(public/'responses'/f'{path.stem}.raw.txt')==host['raw_sha256']
            assert actual==ep['metrics']['actual_model_calls']
    for key,value in counts.items():
        assert value<=protocol['max_'+key],(key,value,protocol['max_'+key])
    for rel,expected in protocol['original_c1_sha256'].items():assert sha(ROOT/rel)==expected
    return dict(passed=True,counts=counts,original_c1_unchanged=True,
        scope='saved causal prefixes, solver/route arithmetic, shared forecast identities, receipts, resource bounds and original hashes; no independent scientific rerun')


def finalize(out):
    protocol=check(out)
    if any((out/'evaluation').iterdir()) or any((out/'predictions').iterdir()):raise ValueError('refuse duplicate evaluation')
    episodes={}
    for method in protocol['methods']:
        if read(out/'runs'/method/'status.json')['status']!='completed':raise ValueError('all flights must finish before truth')
        episodes[method]=read(out/'runs'/method/'episode.json')
    audit=audit_before_truth(out,protocol,episodes);save(out/'reports/pre_truth_audit.json',audit)
    frozen=[p for tree in ('runs','public') for p in (out/tree).rglob('*') if p.is_file()]
    save(out/'reports/flight_freeze.json',{str(p.relative_to(out)):sha(p) for p in frozen})
    cfg=load_config(out/'scenario.json');world=ElectromagneticWorld(cfg);settings=JointSettings(**protocol['joint_settings'])
    display_queries=0;displays={}
    for method,ep in episodes.items():
        xy,arrays,metadata,count=display_predictions(ep,cfg,world.geometry,settings);display_queries+=count
        np.savez_compressed(out/'predictions'/f'{method}.npz',**arrays)
        save(out/'predictions'/f'{method}.json',metadata);displays[method]=(xy,arrays)
    assert display_queries<=protocol['max_display_receiver_queries']
    save(out/'reports/prediction_freeze.json',{str(p.relative_to(out)):sha(p) for p in (out/'predictions').iterdir()})
    # Every flight and display prediction is now frozen. First evaluation truth queries follow.
    rows=[];truth_count=grid_count=warning_count=0
    for method,ep in episodes.items():
        samples=numerical_samples(ep,cfg.aircraft.speed_mps)
        for s in samples:s['truth_dbm']=float(world.truth(s['position_m'],s['time_seconds']))
        truth_count+=len(samples)
        flight_q=math.fsum(s['dt_seconds']*10**((s['truth_dbm']+80.)/10.) for s in samples if s['kind']=='flight')
        hover_q=math.fsum(s['dt_seconds']*10**((s['truth_dbm']+80.)/10.) for s in samples if s['kind']=='fixed_hover')
        predictions=ep['joint_prediction_records'];errors=[r['residual_db'] for r in predictions]
        high=[r for r in predictions if r['observation']['interference_dbm']>=-78.]
        warning=dict(tp=0,fp=0,fn=0,tn=0,supported_points=0,alarms=[])
        warning_samples=[]
        for trigger in ep['trigger_records']:
            q=trigger['queries'];truth=[float(world.truth(p,t)) for p,t in zip(q['position_m'],q['time_seconds'])]
            warning_count+=len(truth)
            for i,(mu,ok,y) in enumerate(zip(q['mean_dbm'],q['supported'],truth)):
                if not ok:continue
                predicted,actual=mu>=-78.,y>=-78.;warning['supported_points']+=1
                warning['tp' if predicted and actual else 'fp' if predicted else 'fn' if actual else 'tn']+=1
            if trigger['requested']:
                crossed=[t for t,y in zip(q['time_seconds'],truth) if y>=-78.]
                warning['alarms'].append(dict(time_seconds=trigger['time_seconds'],predicted_lead_seconds=trigger['predicted_lead_seconds'],
                    frozen_route_truth_lead_seconds=min(crossed)-trigger['time_seconds'] if crossed else None))
            warning_samples.append(dict(forecast_at_seconds=trigger['time_seconds'],queries=q,truth_dbm=np.asarray(truth).tolist()))
        changed=sum(d['selected_id'].startswith('proposal') for d in ep['decisions'])
        row=dict(method=method,**ep['metrics'],Q=flight_q+hover_q,flight_Q=flight_q,fixed_hover_Q=hover_q,
            mission_T_plus_Q=ep['metrics']['simulation_seconds']+flight_q+hover_q,
            joint_prequential_rmse_db=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            joint_prequential_mae_db=float(np.mean(np.abs(errors))) if errors else None,
            high_observed_points=len(high),high_mean_underestimate_db=float(np.mean([max(0.,r['residual_db']) for r in high])) if high else None,
            extra_route_commits=changed,wall_seconds=ep['wall_seconds'],warning=warning)
        save(out/'evaluation'/f'{method}.json',dict(metrics=row,samples=samples,warning_samples=warning_samples));rows.append(row)
        xy,arrays=displays[method];grids=[]
        for index in arrays['indices']:
            grids.append(world.truth(xy,ep['frames'][int(index)]['time_seconds']).reshape(16,21));grid_count+=len(xy)
        np.savez_compressed(out/'evaluation'/f'{method}_grids.npz',truth_dbm=np.asarray(grids))
    assert truth_count<=protocol['max_evaluation_points'] and grid_count<=protocol['max_replay_truth_points']
    assert warning_count<=protocol['max_monitor_receiver_queries']
    save(out/'reports/results.json',dict(rows=rows,evaluation_truth_points=truth_count,replay_truth_points=grid_count,
        warning_evaluation_truth_points=warning_count,display_queries=display_queries,
        limitations=['one seen development scene, one flight per method','current numerical feedback planner needs stronger formal baselines',
                     'synchronous callbacks freeze physical time','no generalization, safety or LLM superiority conclusion']))
    save(out/'reports/verification.json',audit)
    render(out)
    write_findings(out,rows)
    print(json.dumps(dict(rows=rows,verification=audit['passed']),ensure_ascii=False,indent=2),flush=True)


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image
    cfg=load_config(out/'scenario.json');high=[(cfg.map.columns-1)*cfg.map.resolution_m,(cfg.map.rows-1)*cfg.map.resolution_m]
    results=read(out/'reports/results.json')
    for row in results['rows']:
        method=row['method'];ep=read(out/'runs'/method/'episode.json');arrays=np.load(out/'predictions'/f'{method}.npz')
        truth=np.load(out/'evaluation'/f'{method}_grids.npz')['truth_dbm'];metadata=read(out/'predictions'/f'{method}.json');images=[]
        for k,index in enumerate(arrays['indices']):
            f=ep['frames'][int(index)];now=f['time_seconds'];fig,axes=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
            fig.subplots_adjust(top=.79,bottom=.13,right=.9,wspace=.12)
            trail=np.asarray([p['position_m'] for p in ep['observations'] if p['time_seconds']<=now])
            decisions=[d for d in ep['decisions'] if d['time_seconds']<=now];last=decisions[-1] if decisions else None
            for n,(ax,z) in enumerate(zip(axes,[truth[k],arrays['actual_dbm'][k]])):
                cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#b9bdc2')
                im=ax.imshow(np.ma.masked_invalid(z),origin='lower',extent=(0,high[0],0,high[1]),vmin=-100,vmax=-55,cmap=cmap)
                for b in cfg.map.buildings:ax.add_patch(Polygon(b['footprint_m'],color='#303030' if b['height_m']+cfg.aircraft.vertical_clearance_m>=cfg.aircraft.altitude_m else '#abc4cf',zorder=3))
                ax.plot(*trail.T,color='#125ec9',lw=2,zorder=4)
                if f['route_m']:ax.plot(*np.asarray(f['route_m']).T,'--',color='#007c91',lw=1.4,zorder=4)
                ax.plot(*f['position_m'],'ko',ms=4,zorder=5);ax.plot(*cfg.map.goal_m,'b*',ms=10,zorder=5)
                for d in decisions:ax.plot(*d['public']['request']['state']['position_m'],'o',mfc='none',mec='#a000b0',ms=7,zorder=5)
                ax.set(xlim=(0,high[0]),ylim=(0,high[1]),xlabel='x (m)',ylabel='y (m)')
                ax.set_title('True field (evaluation only)' if n==0 else f"Planner: {ep['planner_method']} | cyan: GP support",fontsize=10)
                if n==1 and np.any(arrays['supported'][k]) and not np.all(arrays['supported'][k]):
                    ax.contour(np.linspace(0,high[0],21),np.linspace(0,high[1],16),arrays['supported'][k],levels=[.5],colors=['cyan'],linewidths=.8)
            weights=', '.join(f"{m['family']}:{m['weight']:.2f}" for m in metadata[k]['top_hypotheses'])
            last_text=f"commit t={last['time_seconds']:g}: {last['selected_id']}" if last else 'no warning event'
            fig.suptitle(f"{method} | t={now:.1f}s | {f['phase']} | {last_text}\n{weights}\nBlue: flown; dashed: committed route; circles: warning consultations. Synchronous; latency not charged.",fontsize=10)
            bar=fig.add_axes([.92,.18,.014,.5]);fig.colorbar(im,cax=bar,label='dBm (higher = stronger)')
            fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        images[0].save(out/'replay'/f'{method}.gif',save_all=True,append_images=images[1:],duration=360,loop=0)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>D1联合预测与LLM决策开发实验</title><style>body{font:16px/1.7 system-ui;max-width:1250px;margin:25px auto;padding:12px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}</style><h1>D1：联合预测、前方预警与LLM决策</h1><p>一个已见开发场景，各方法一次飞行。原件独立冻结；仅描述性比较。墙钟等待未计入物理飞行。蓝线已飞、虚线已提交、紫圈咨询，右图为该方法当前预测；当前预测可能与最近决策冻结版本不同，日志保留其版本。</p><p><a href="../reports/findings.md">结果解读</a> · <a href="../reports/results.json">完整指标</a> · <a href="../reports/verification.json">核验</a> · <a href="../protocol.json">冻结合同</a></p><table><tr><th>方法</th><th>到达</th><th>航程m</th><th>真实Q</th><th>真实T+Q</th><th>调用</th><th>预序RMSE</th></tr>'
    for r in results['rows']:page+=f"<tr><td>{r['method']}</td><td>{r['success']}</td><td>{r['executed_length_m']:.2f}</td><td>{r['Q']:.2f}</td><td>{r['mission_T_plus_Q']:.2f}</td><td>{r['actual_model_calls']}</td><td>{r['joint_prequential_rmse_db']:.2f}</td></tr>"
    page+='</table>'
    for r in results['rows']:page+=f"<article><h2>{r['method']}</h2><a href=\"../runs/{r['method']}/episode.json\">原始记录</a><img src=\"{r['method']}.gif\"></article>"
    (out/'replay/index.html').write_text(page+'</html>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><a href="replay/index.html">D1：实验指标与双图回放</a>')


def write_findings(out,rows):
    lookup={r['method']:r for r in rows};base=lookup['joint_N']['mission_T_plus_Q']
    lines=['# D1开发实验结果','', '单个已见开发场景、每方法一次。数值参考沿用已有反馈规划器，尚不足以充当论文的最终强优化基线。', '',
           '| 方法 | 到达 | 航程m | Q | T+Q | 相对N | 真实LLM调用 |', '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['method']} | {r['success']} | {r['executed_length_m']:.2f} | {r['Q']:.2f} | {r['mission_T_plus_Q']:.2f} | {(r['mission_T_plus_Q']/base-1)*100:+.2f}% | {r['actual_model_calls']} |")
    lines+=['','GP收益与LLM收益分开：source_numeric→joint_point_numeric检查联合预测；joint_point_numeric→joint_N检查软拟合；I/P/J对N仅描述当前闭环差异。航迹和点测可能不同，不能冒充同状态因果对照。', '',
            '预序指标均先冻结旧预测再接收点测。预警点级混淆矩阵只统计GP支持内，标签评价的是触发时冻结路线；它与改路后的实际暴露不同。', '',
            '当前等待冻结物理状态，未评价实际LLM时延。GP方差未校准，−78dBm是开发预警线，没有硬安全或真实通信设备失效含义。原C1原件未改。', '', '逐事件：']
    for r in rows:
        ep=read(out/'runs'/r['method']/'episode.json')
        lines.append('\n'+r['method']+':')
        for d in ep['decisions']:
            lines.append(f"- t={d['time_seconds']:g}s，触发预测提前量={d['public']['consultation_event']['predicted_lead_seconds']:.2f}s，共同最优={d['common_selected_id']}，最终={d['selected_id']}，提议={d['extra']['plan']['action']}，回退={d['fallback']}。")
    (out/'reports/findings.md').write_text('\n'.join(lines)+'\n')
