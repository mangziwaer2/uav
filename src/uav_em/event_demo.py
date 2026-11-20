"""One bounded development demonstration; entry is run_experiment.py demo."""
from copy import deepcopy
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .config import load_config
from .environment import ElectromagneticWorld
from .equivalent_field import kernel, power_and_jacobian, normalized_expected_power
from .scenario_loop import run as fly

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('gp_numeric', 'mixture_numeric', 'mixture_llm')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    """Atomically publish complete JSON, retaining exclusive-create semantics."""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix='.'+path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        # A same-filesystem hard link publishes the closed file atomically and
        # fails if the destination exists; replace/rename would overwrite it.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def control(path, value):
    temporary = path.with_suffix('.next')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def prepare(out):
    out.mkdir(exist_ok=False)
    for name in ('runs', 'public/requests', 'public/responses', 'reports/frozen_source',
                 'predictions', 'evaluation', 'replay'):
        (out/name).mkdir(parents=True)
    original = ROOT/'outputs/interference_scenario_loop_20261118/scenarios/dynamic.json'
    shutil.copy2(original, out/'scenario.json')
    shutil.copy2(ROOT/'docs/event_demo_public_contract.md', out/'public/contract.md')
    sources = sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'tests').rglob('*.py'))
    sources += [ROOT/p for p in ('requirements.txt', 'docs/event_demo_v1.md', 'docs/event_demo_public_contract.md')]
    hashes = {}
    for path in sources:
        rel = path.relative_to(ROOT)
        dest = out/'reports/frozen_source'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[str(rel)] = sha(path)
    save(out/'protocol.json', dict(source_sha256=hashes, scenario_sha256=sha(out/'scenario.json'),
        public_contract_sha256=sha(out/'public/contract.md'), methods=list(METHODS),
        max_flights=3, max_events_per_flight=4, max_bridge_requests=12, max_actual_model_calls=12,
        max_sensor_points=330, max_jobs=24, max_solver_calls=48, max_residual_calls=1920,
        max_samples=10500, max_edges=150000, max_evaluation_points=2000,
        max_replay_truth_points=36288, max_frames_per_flight=36,
        display_shape=[16,21], independent_llm_sessions=1,
        interpretation='one seen development scene; descriptive pilot, not a superiority or generalization test',
        timing='power_rise, synchronous frozen callback, 0.5s fixed hover; wall latency not physically charged'))
    control(out/'public/control.json', dict(status='prepared', max_requests=12))
    print('PREPARED', out, flush=True)


def check(out):
    protocol = read(out/'protocol.json')
    for rel, expected in protocol['source_sha256'].items():
        if sha(ROOT/rel) != expected:
            raise ValueError('source changed after freeze: '+rel)
    if sha(out/'scenario.json') != protocol['scenario_sha256'] or sha(out/'public/contract.md') != protocol['public_contract_sha256']:
        raise ValueError('frozen input changed')
    return protocol


class FileAdvisor:
    """Sequential public mailbox with actual worker-start receipts, no hidden state."""
    def __init__(self, public, agent_id, clock=time.monotonic, sleeper=time.sleep):
        self.public, self.agent_id, self.clock, self.sleeper = public, agent_id, clock, sleeper
        self.count = 0

    def ask(self, kind, request):
        if self.count >= 12:
            raise ValueError('new request budget exhausted')
        index = self.count
        self.count += 1
        request_id = request['request_id'] if kind == 'inference' else request['request']['request_id']
        name = f'{index:02d}'
        started_path = self.public/'responses'/f'{name}.started.json'
        raw_path = self.public/'responses'/f'{name}.raw.txt'
        ready_path = self.public/'responses'/f'{name}.ready'
        issued = self.clock()
        save(self.public/'requests'/f'{name}.json', dict(index=index, kind=kind,
            request_id=request_id, request=deepcopy(request), issued_monotonic=issued))
        print('REQUEST', index, kind, 't', request.get('time_seconds', request.get('request',{}).get('time_seconds')), flush=True)
        started = received = raw = error = None
        receipt_json_retries = 0
        while True:
            now = self.clock()
            if started is None and now-issued > 60.:
                error = 'undispatched timeout'; break
            if started is not None and now-started > 180.:
                error = 'response timeout'; break
            if started is None and started_path.exists():
                try:
                    receipt = read(started_path)
                    stamp = receipt['started_monotonic']
                    if (receipt['agent_id'] != self.agent_id or receipt['request_id'] != request_id
                            or type(stamp) not in (int, float) or not math.isfinite(stamp)
                            or not issued <= stamp <= now):
                        raise ValueError('invalid start receipt binding/clock')
                    started = stamp
                    if started-issued > 60.:
                        error = 'late dispatch'; break
                except json.JSONDecodeError:
                    # A legacy writer may still expose an unfinished receipt.
                    # Retry only inside the unchanged dispatch deadline.
                    receipt_json_retries += 1
                except (KeyError, TypeError, ValueError) as exc:
                    error = str(exc); break
            if started is not None and raw_path.exists() and ready_path.exists():
                received = now
                raw = raw_path.read_text()
                break
            self.sleeper(.15)
        timely = bool(received is not None and started is not None and received-started <= 180.)
        envelope = dict(raw=raw, timely=timely, started=started is not None,
            actual_llm_request=started is not None, agent_id=self.agent_id,
            issued_monotonic=issued, started_monotonic=started, received_monotonic=received,
            wall_seconds=self.clock()-issued, error=error, kind=kind, index=index,
            request_id=request_id, raw_sha256=sha(raw_path) if raw is not None else None,
            receipt_json_retries=receipt_json_retries)
        save(self.public/'responses'/f'{name}.host.json', envelope)
        print('RECEIVED', index, 'timely', timely, 'error', error, flush=True)
        return envelope


def run_one(out, method):
    check(out)
    if method not in METHODS:
        raise ValueError('undeclared method')
    folder = out/'runs'/method
    folder.mkdir(exist_ok=False)
    bridge = None
    if method == 'mixture_llm':
        agent_id = read(out/'reports/session.json')['agent_id']
        if read(out/'public/session_ready.json')['agent_id'] != agent_id:
            raise ValueError('decision session not ready/bound')
        bridge = FileAdvisor(out/'public', agent_id)
        control(out/'public/control.json', dict(status='running', max_requests=12))
    def emit(row):
        with (folder/'decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        print('DECISION', method, row['index'], 't', row['time_seconds'], row['selected_id'], flush=True)
    started = time.monotonic()
    try:
        episode = fly(load_config(out/'scenario.json'), method,
            advisor=(lambda info, index: bridge.ask('path', info)) if bridge else None,
            emit=emit, timing='power_rise',
            inference_advisor=(lambda request: bridge.ask('inference', request)) if bridge else None)
        episode['wall_seconds'] = time.monotonic()-started
        save(folder/'episode.json', episode)
        save(folder/'status.json', dict(status='completed', episode_sha256=sha(folder/'episode.json')))
        if bridge:
            control(out/'public/control.json', dict(status='completed', bridge_requests=bridge.count))
        print('COMPLETE', method, episode['metrics'], flush=True)
    except BaseException as exc:
        save(folder/'failure.json', dict(error=repr(exc)))
        if bridge:
            control(out/'public/control.json', dict(status='failed', error=repr(exc)))
        raise


def freeze_flights(out):
    check(out)
    for method in METHODS:
        if read(out/'runs'/method/'status.json')['status'] != 'completed':
            raise ValueError('all declared flights must finish before truth evaluation')
    files = [p for tree in ('runs','public') for p in (out/tree).rglob('*') if p.is_file()]
    files += [out/'reports/session.json']
    save(out/'reports/flight_freeze.json', {str(p.relative_to(out)):sha(p) for p in files})


def frame_indices(episode):
    frames = episode['frames']
    targets = list(np.arange(0., episode['metrics']['simulation_seconds']+1e-8, 4.))
    targets += [episode['metrics']['simulation_seconds']]
    indices = {min(range(len(frames)),key=lambda i:abs(frames[i]['time_seconds']-t)) for t in targets}
    for event in episode['decisions']:
        indices.update(i for i, f in enumerate(frames) if abs(f['time_seconds']-event['time_seconds'])<1e-8
                       or abs(f['time_seconds']-event['time_seconds']-.5)<1e-8)
    if len(indices)>36:
        raise ValueError('display frame budget exceeded')
    return sorted(indices)


def model_mean(model, xy, now, latest_power):
    if model['family']=='persistence':
        return np.full(len(xy),latest_power)
    return power_and_jacobian(kernel(model['bases'],xy,np.full(len(xy),now),model['reference_time_seconds']),model['peaks_dbm'])[0]


def frozen_display(episode, method, high):
    """Compute display predictions from saved states, before any evaluation truth."""
    indices = frame_indices(episode)
    xy = np.array([[x,y] for y in np.linspace(0,high[1],16) for x in np.linspace(0,high[0],21)])
    actual=[];alternatives=[];metadata=[];terms=0
    for index in indices:
        frame=episode['frames'][index];now=frame['time_seconds'];models=frame['hypotheses']
        means=[model_mean(m,xy,now,frame['latest_power_dbm']) for m in models]
        terms+=len(xy)*sum(len(m['bases']) for m in models)
        mixture=sum(m['weight']*normalized_expected_power(mu,np.full(len(xy),m['std_db']),-80.) for m,mu in zip(models,means))
        if method=='gp_numeric':
            mu=np.asarray(frame['estimate_dbm'])[::2,::2]
            sd=np.asarray(frame['uncertainty_db'])[::2,::2]
            support=np.asarray(frame['supported'])[::2,::2]
            expected=normalized_expected_power(np.where(support,mu,-100.),np.where(support,sd,15.),-80.)
            image=-80.+10.*np.log10(expected)
            image=np.where(support,image,np.nan)
        else:
            image=(-80.+10.*np.log10(mixture)).reshape(16,21)
        top=sorted([m for m in models if m['family']!='persistence'],key=lambda m:(-m['weight'],m['id']))[:2]
        maps=[model_mean(m,xy,now,frame['latest_power_dbm']).reshape(16,21) for m in top]
        terms+=len(xy)*sum(len(m['bases']) for m in top)
        maps += [np.full((16,21),np.nan)]*(2-len(maps))
        actual.append(image);alternatives.append(maps)
        metadata.append(dict(frame_index=index, alternatives=top,
            persistence_mass=sum(m['weight'] for m in models if m['family']=='persistence'),
            actual_prediction='GP expectation (unsupported hidden)' if method=='gp_numeric' else 'alternative-mixture expected linear power, converted to dBm'))
    return xy,dict(indices=np.asarray(indices),actual_dbm=np.asarray(actual),alternative_mean_dbm=np.asarray(alternatives)),metadata,terms


def execution_samples(episode,speed):
    samples=[]
    for motion in episode['motions']:
        elapsed=0.
        for first,second in zip(motion['traversed_m'],motion['traversed_m'][1:]):
            a,b=np.asarray(first),np.asarray(second);length=float(np.linalg.norm(b-a));dt=length/speed
            count=max(1,math.ceil(length/10.))
            for i in range(count):
                f=(i+.5)/count
                samples.append(dict(position_m=(a+f*(b-a)).tolist(),
                    time_seconds=motion['start_time_seconds']+elapsed+f*dt,dt_seconds=dt/count,kind='flight'))
            elapsed+=dt
    for decision in episode['decisions']:
        samples.append(dict(position_m=decision['public']['request']['state']['position_m'],
            time_seconds=decision['time_seconds']+.25,dt_seconds=.5,kind='fixed_hover'))
    return samples


def audit_cached(out, results):
    """Cache arithmetic and causal bindings, not independent propagation/fitting."""
    protocol=check(out)
    for name in ('flight_freeze','prediction_freeze'):
        for rel, expected in read(out/'reports'/f'{name}.json').items():
            if sha(out/rel)!=expected:raise ValueError('frozen result changed: '+rel)
    counts=dict(flights=0,observations=0,events=0,bridge_requests=0,actual_calls=0,jobs=0,solves=0,residuals=0,samples=0,edges=0)
    for row in results:
        ep=read(out/'runs'/row['method']/'episode.json');counts['flights']+=1
        assert len(ep['prediction_records'])==len(ep['observations'])-1
        counts['observations']+=len(ep['observations']);counts['events']+=len(ep['decisions'])
        assert len(ep['decisions'])<=4
        for decision in ep['decisions']:
            assert decision['public']['consultation_event']['requested']
            prefix=[p for p in ep['observations'] if p['time_seconds']<=decision['time_seconds']]
            assert prefix==decision['public']['request']['history']
            valid=[c for c in decision['candidates'] if c['feasible']]
            selected=min(valid,key=lambda c:(c['score']['score_seconds'],
                sum(math.dist(a,b) for a,b in zip(c['path'],c['path'][1:])),c['id']))
            assert selected['id']==decision['selected_id']
            for c in valid:
                s=c['score'];q=math.fsum(v['weight']*v['power_proxy_seconds'] for v in s['scenario_scores'])
                assert math.isclose(q,s['power_proxy_seconds'],rel_tol=1e-10,abs_tol=1e-8)
                assert math.isclose(s['score_seconds'],s['remaining_flight_seconds']+q,rel_tol=1e-10)
        for key,metric in [('jobs','inferred_jobs'),('solves','solver_calls'),('residuals','residual_calls'),('samples','total_samples'),('edges','total_edges')]:
            counts[key]+=ep['metrics'][metric]
        counts['actual_calls']+=ep['metrics']['actual_model_calls']
        length=math.fsum(math.dist(a,b) for m in ep['motions'] for a,b in zip(m['traversed_m'],m['traversed_m'][1:]))
        assert math.isclose(length,ep['metrics']['executed_length_m'],rel_tol=1e-10)
        data=read(out/'evaluation'/f"{row['method']}.json")
        q=math.fsum(s['dt_seconds']*10.**((s['truth_dbm']+80.)/10.) for s in data['samples'])
        assert math.isclose(q,row['Q'],rel_tol=1e-10)
        assert math.isclose(row['mission_T_plus_Q'],ep['metrics']['simulation_seconds']+q,rel_tol=1e-10)
    requests=sorted((out/'public/requests').glob('*.json'));counts['bridge_requests']=len(requests)
    actual=0
    for path in requests:
        request=read(path);host=read(out/'public/responses'/f'{path.stem}.host.json')
        assert request['request_id']==host['request_id']
        if host['started']:
            receipt=read(out/'public/responses'/f'{path.stem}.started.json')
            assert receipt['request_id']==host['request_id'] and receipt['agent_id']==host['agent_id']
            actual+=1
        if host['raw'] is not None:
            raw=out/'public/responses'/f'{path.stem}.raw.txt'
            assert sha(raw)==host['raw_sha256'] and raw.read_text()==host['raw']
    assert actual==counts['actual_calls']<=12 and counts['bridge_requests']<=12
    assert counts['observations']<=330 and counts['jobs']<=24 and counts['solves']<=48 and counts['residuals']<=1920
    assert counts['samples']<=10500 and counts['edges']<=150000
    assert counts['flights']==3
    return dict(passed=True,counts=counts,scope='saved arithmetic, selection, measurement-prefix, original receipt and hash bindings only; no independent GP/fit/physics/RRT rerun')


def finalize(out):
    protocol=check(out)
    if list((out/'evaluation').iterdir()) or list((out/'predictions').iterdir()):
        raise ValueError('refuse duplicate evaluation/prediction run')
    freeze_flights(out)
    cfg=load_config(out/'scenario.json')
    high=[(cfg.map.columns-1)*cfg.map.resolution_m,(cfg.map.rows-1)*cfg.map.resolution_m]
    display_terms=0
    for method in METHODS:
        episode=read(out/'runs'/method/'episode.json')
        xy,arrays,metadata,terms=frozen_display(episode,method,high);display_terms+=terms
        np.savez_compressed(out/'predictions'/f'{method}.npz',**arrays)
        save(out/'predictions'/f'{method}.json',metadata)
    save(out/'reports/prediction_freeze.json',{str(p.relative_to(out)):sha(p) for p in (out/'predictions').iterdir()})
    # First evaluation world queries occur only after every flight and display prediction froze.
    world=ElectromagneticWorld(cfg);rows=[];evaluation_count=grid_count=0
    for method in METHODS:
        episode=read(out/'runs'/method/'episode.json');samples=execution_samples(episode,cfg.aircraft.speed_mps)
        for s in samples:
            s['truth_dbm']=float(world.truth(s['position_m'],s['time_seconds']));evaluation_count+=1
        flight_q=math.fsum(s['dt_seconds']*10.**((s['truth_dbm']+80.)/10.) for s in samples if s['kind']=='flight')
        hover_q=math.fsum(s['dt_seconds']*10.**((s['truth_dbm']+80.)/10.) for s in samples if s['kind']=='fixed_hover')
        changed=sum(not np.allclose(d['route_after'],d['public']['request']['state']['route_m'])
            if len(d['route_after'])==len(d['public']['request']['state']['route_m']) else True for d in episode['decisions'])
        row=dict(method=method,**episode['metrics'],Q=flight_q+hover_q,flight_Q=flight_q,fixed_hover_Q=hover_q,
            mission_T_plus_Q=episode['metrics']['simulation_seconds']+flight_q+hover_q,
            llm_extra_route_commits=changed if method=='mixture_llm' else 0,wall_seconds=episode['wall_seconds'],
            inference_parser_accepted=sum(r['parser_accepted'] for e in episode['inferences'] for r in e['rounds']),
            inference_fallbacks=sum(not r['parser_accepted'] for e in episode['inferences'] for r in e['rounds']))
        save(out/'evaluation'/f'{method}.json',dict(metrics=row,samples=samples));rows.append(row)
        indices=np.load(out/'predictions'/f'{method}.npz')['indices'];grids=[];sources=[]
        for index in indices:
            now=episode['frames'][int(index)]['time_seconds']
            grids.append(world.truth(xy,now).reshape(16,21));grid_count+=len(xy)
            sources.append(world.source_positions_for_evaluation(now))
        np.savez_compressed(out/'evaluation'/f'{method}_grids.npz',truth_dbm=np.asarray(grids))
        save(out/'evaluation'/f'{method}_sources.json',sources)
    assert evaluation_count<=protocol['max_evaluation_points'] and grid_count<=protocol['max_replay_truth_points']
    results=dict(rows=rows,evaluation_truth_points=evaluation_count,replay_truth_points=grid_count,
        display_basis_terms=display_terms,conclusion='descriptive single-scene pilot; no generalization or isolated inference-LLM benefit claim',
        physical_model_latency_charged=False,unforewarned_source_onset='cannot promise prediction before evidence; inspect reaction after measurement')
    save(out/'reports/results.json',results)
    save(out/'reports/verification.json',audit_cached(out,rows))
    render(out)
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image
    cfg=load_config(out/'scenario.json');high=[(cfg.map.columns-1)*cfg.map.resolution_m,(cfg.map.rows-1)*cfg.map.resolution_m]
    for method in METHODS:
        ep=read(out/'runs'/method/'episode.json');pred=np.load(out/'predictions'/f'{method}.npz')
        metadata=read(out/'predictions'/f'{method}.json');truth=np.load(out/'evaluation'/f'{method}_grids.npz')['truth_dbm']
        true_sources=read(out/'evaluation'/f'{method}_sources.json');images=[]
        for k,index in enumerate(pred['indices']):
            frame=ep['frames'][int(index)];now=frame['time_seconds'];meta=metadata[k]
            fig,axes=plt.subplots(1,4,figsize=(16,4.4),sharex=True,sharey=True)
            fig.subplots_adjust(top=.75,bottom=.14,wspace=.08,right=.94)
            models=meta['alternatives'];titles=['Truth (evaluation only)', 'Current planner forecast\nexpected power; gray unknown']
            titles += [f"Alternative {i+1}: {m['family']}\nw={m['weight']:.3f}; future points={m['scored_points']}" for i,m in enumerate(models)]
            titles += ['No fitted alternative']*(4-len(titles))
            fields=[truth[k],pred['actual_dbm'][k],*pred['alternative_mean_dbm'][k]]
            trail=np.asarray([o['position_m'] for o in ep['observations'] if o['time_seconds']<=now])
            decisions=[d for d in ep['decisions'] if d['time_seconds']<=now];last=decisions[-1] if decisions else None
            for n,(ax,z,title) in enumerate(zip(axes,fields,titles)):
                cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#bec3ca')
                im=ax.imshow(np.ma.masked_invalid(z),origin='lower',extent=(0,*high[:1],0,high[1]),vmin=-100,vmax=-55,cmap=cmap)
                for b in cfg.map.buildings:
                    ax.add_patch(Polygon(b['footprint_m'],color='#303030' if b['height_m']+cfg.aircraft.vertical_clearance_m>=cfg.aircraft.altitude_m else '#b9cddd',zorder=3))
                ax.plot(*trail.T,color='royalblue',lw=2,zorder=4)
                if frame['route_m']:ax.plot(*np.asarray(frame['route_m']).T,'--',color='#007e91',lw=1.3,zorder=4)
                ax.plot(*frame['position_m'],'ko',ms=4,zorder=5);ax.plot(*cfg.map.goal_m,'b*',ms=10,zorder=5)
                for d in decisions:ax.plot(*d['public']['request']['state']['position_m'],'o',mfc='none',mec='#c000c0',ms=7,zorder=5)
                if last and last['extra']['plan']['action']=='via':ax.plot(*last['extra']['plan']['waypoint_m'],'D',color='#a000b0',ms=5,zorder=5)
                if n==0:
                    for source in true_sources[k]:ax.plot(*source['position_m'],'*',color='black',ms=10,zorder=5)
                if n>=2 and n-2<len(models):
                    model=models[n-2]
                    for b in model['bases']:
                        center=np.asarray(b['center_m'])+(now-model['reference_time_seconds'])*np.asarray(b['velocity_mps'])
                        ax.plot(*center,'x',color='#8a2387',ms=8,zorder=5)
                ax.set_title(title,fontsize=9);ax.set_xlabel('x (m)');ax.set_xlim(0,high[0]);ax.set_ylim(0,high[1])
            last_text=f"last event t={last['time_seconds']:g}: {last['selected_id']}" if last else 'no rise event yet'
            fig.suptitle(f"{method} | t={now:.1f}s | power={frame['latest_power_dbm']:.1f} dBm | events={len(decisions)}/4 | {last_text}\n"
                f"Blue: flown; dashed: executed remaining route; circles: rise events; diamond: proposal (may be rejected)\n"
                f"Alternatives are separate possible worlds (mean fields); persistence mass={meta['persistence_mass']:.3f}. Synchronous prototype; wall wait not simulated.",fontsize=10)
            bar=fig.add_axes([.95,.17,.012,.5]);fig.colorbar(im,cax=bar,label='dBm (higher = stronger)')
            fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        destination=out/'replay'/f'{method}_dynamic.gif'
        if destination.exists():raise FileExistsError(destination)
        images[0].save(destination,save_all=True,append_images=images[1:],duration=420,loop=0)
        print('REPLAY',destination.name,len(images),'frames',flush=True)
    results=read(out/'reports/results.json');style='<style>body{font:16px/1.7 system-ui;max-width:1500px;margin:25px auto;padding:15px}img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbc;padding:7px}article{margin:30px 0}</style>'
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>事件飞行回放</title>'+style+'<h1>简单功率上升事件：实际飞行回放</h1><p>一个已见动态开发场景，三次新飞行。联合LLM组织等效推断和路径；非正式泛化或单模块因果收益。等待冻结仿真状态，实际墙钟不折算飞行。</p><p>四图：事后真场、当前规划预测、两张独立等效解释。它们不是同时存在的源。蓝线已飞，虚线是已提交剩余路线，紫圈触发，紫菱形仅提议，是否执行看事件文字/虚线。</p><p>无征兆突现源不能保证预知，关注异常点测后的反应；事件最多4次，额度耗尽也须作为限制。GP灰区未知；混合图含未校准不确定性的期望功率，替代解释图为均值。</p><table><tr><th>方法</th><th>到达</th><th>航程m</th><th>Q含固定悬停</th><th>任务T+Q</th><th>实际LLM调用</th><th>事件</th></tr>'
    for row in results['rows']:
        text+=f"<tr><td>{row['method']}</td><td>{row['success']}</td><td>{row['executed_length_m']:.2f}</td><td>{row['Q']:.2f}</td><td>{row['mission_T_plus_Q']:.2f}</td><td>{row['actual_model_calls']}</td><td>{row['decision_count']}</td></tr>"
    text+='</table><p><a href="../reports/results.json">完整指标</a> · <a href="../reports/verification.json">缓存核验及范围</a> · <a href="../reports/findings.md">逐事件解读</a></p>'
    for method in METHODS:
        text+=f'<article><h2>{html.escape(method)}</h2><a href="{method}_dynamic.gif">原GIF</a> · <a href="../runs/{method}/episode.json">完整飞行/解释日志</a><img src="{method}_dynamic.gif"></article>'
    (out/'replay/index.html').write_text(text+'</html>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><a href="replay/index.html">三次实际事件飞行：直接看回放及指标</a>')
