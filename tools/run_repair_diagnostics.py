"""冻结候选后才评价真值的瓶颈诊断；不是新闭环策略试验。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json
from uav_em.belief import MapSnapshot
from uav_em.config import load_config
from uav_em.environment import Geometry
from uav_em.repair_diagnostics import SEARCH_PROFILES,geometric_candidates,search_candidate

SOURCE=ROOT/'outputs/interference_program_comparison_20261110'
DEFAULT=ROOT/'outputs/interference_repair_diagnostics_20261110'
REFERENCES={'ReEvo':'2402.01145','EoH':'2401.02051','VLM-RRT':'2505.23267',
            'IGV-RRT':'2603.21887','Trajectory_Adaptation':'2504.12755'}

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_case(out,path):
    case=read(path);cfg=load_config(out/'scenarios'/f"{case['scene']}.json")
    geometry=Geometry(cfg.map,cfg.aircraft.clearance_m,cfg.aircraft.altitude_m,cfg.aircraft.vertical_clearance_m)
    with np.load(path.with_suffix('.npz')) as data:
        snapshot=MapSnapshot(case['time_seconds'],**{key:data[key].copy() for key in data.files},
            observation_count=case['observation_count'],geometry=geometry,estimation_details=case['estimation_details'])
    return case,snapshot,cfg

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    for part in ('cases','scenarios','candidates','evaluations','reports/runtime_source','figures','replay','logs','literature'):
        (out/part).mkdir(parents=True)
    inputs={}
    paths=[*(SOURCE/'paired/cases').glob('*'),*(SOURCE/'scenarios').glob('*.json')]
    if len([p for p in paths if p.suffix=='.npz'])!=4:raise ValueError('exactly four recorded snapshots required')
    for path in paths:
        target=out/('scenarios' if path.parent.name=='scenarios' else 'cases')/path.name
        shutil.copy2(path,target);inputs[str(target.relative_to(out))]=sha(target)
    sources={}
    for path in [*(ROOT/'src/uav_em').glob('*.py'),Path(__file__).resolve()]:
        relative=path.relative_to(ROOT);target=out/'reports/runtime_source'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);sources[str(relative)]=sha(path)
    save_json(out/'protocol.json',dict(study='frozen_snapshot_repair_bottleneck_diagnostic',
        source_study=str(SOURCE.relative_to(ROOT)),inputs_sha256=inputs,sources_sha256=sources,
        cases=sorted(p.stem for p in (out/'cases').glob('*.json')),
        candidate_rule='incumbent + fixed interval/side/offset geometry + six RRT profiles x three seeds',
        profiles=list(SEARCH_PROFILES),seed_offsets=[0,101,202],
        independent_search_limits={'rounds':1,'sample_quota':700,'sample_cap':2100,'edge_cap':10000},
        pool_budget='geometric evaluation separate; every search independent; aggregate work reported; not matched online budget',
        evaluation={'length_ratio_limit':1.25,'material_reduction_fraction':.1,'step_seconds':.25,
                    'departure_delay_seconds':.5,'no_new_observations':True,'hover_exposure_included':False},
        objective='unchanged current GP J; no source/GP/weight retuning',
        model_calls=0,truth_boundary='all candidates saved and hashed before evaluate; no truth feedback',
        limitation='four development snapshots; fixed-route open-loop counterfactual; no closed-loop or LLM advantage claim'))
    save_json(out/'status.json',{'status':'prepared'})

def check(out):
    protocol=read(out/'protocol.json')
    for relative,expected in protocol['inputs_sha256'].items():
        if sha(out/relative)!=expected:raise ValueError('input changed: '+relative)
    for relative,expected in protocol['sources_sha256'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('runtime source changed: '+relative)
        if sha(out/'reports/runtime_source'/relative)!=expected:raise ValueError('frozen source changed')
    return protocol

def propose(out):
    protocol=check(out)
    if (out/'candidate_manifest.json').exists() or any((out/'candidates').iterdir()):
        raise FileExistsError('refuse to overwrite candidates')
    for name in protocol['cases']:
        case,snapshot,_=load_case(out,out/'cases'/f'{name}.json');started=time.perf_counter()
        records,public,pool_work=geometric_candidates(snapshot,case['context'],case['route'],case['max_length_m'])
        for profile in protocol['profiles']:
            for offset in protocol['seed_offsets']:
                records.append(search_candidate(snapshot,case['context'],case['route'],case['max_length_m'],profile,case['seed']+offset))
            print(name,profile,'done',flush=True)
        save_json(out/'candidates'/f'{name}.json',{'case':name,'records':records,'public_context':public,
            'geometric_pool_resources':pool_work,'elapsed_seconds':time.perf_counter()-started})
    files={str(path.relative_to(out)):sha(path) for path in sorted((out/'candidates').glob('*.json'))}
    save_json(out/'candidate_manifest.json',{'sha256':files,'all_candidates_frozen_before_truth_evaluation':True})
    save_json(out/'status.json',{'status':'candidates_frozen'})

def verify_candidates(out):
    manifest=read(out/'candidate_manifest.json')
    if len(manifest['sha256'])!=4:raise ValueError('incomplete candidate freeze')
    for relative,expected in manifest['sha256'].items():
        if sha(out/relative)!=expected:raise ValueError('frozen candidates modified')
    return manifest

def evaluate(out):
    protocol=check(out);verify_candidates(out)
    # Only now instantiate hidden worlds. Never send evaluations back to propose.
    from uav_em.environment import ElectromagneticWorld
    from uav_em.diagnostic_evaluation import evaluate_fixed_route,diagnose_candidates
    if any((out/'evaluations').iterdir()):raise FileExistsError('refuse to overwrite evaluations')
    settings=protocol['evaluation']
    for name in protocol['cases']:
        case,snapshot,cfg=load_case(out,out/'cases'/f'{name}.json');world=ElectromagneticWorld(cfg)
        data=read(out/'candidates'/f'{name}.json');records=data['records']
        for record in records:
            if record['feasible']:
                record['evaluation']=evaluate_fixed_route(world,snapshot,record['path'],
                    settings['departure_delay_seconds'],settings['step_seconds'])
        diagnostic=diagnose_candidates(records,settings['length_ratio_limit'],settings['material_reduction_fraction'])
        searches=[r['report'] for r in records if r['family']=='rrt_search']
        save_json(out/'evaluations'/f'{name}.json',{'case':name,'time_seconds':snapshot.time_seconds,
            'records':records,'diagnosis':diagnostic,'geometric_pool_resources':data['geometric_pool_resources'],
            'aggregate_search_resources':{'runs':len(searches),
                'samples':sum(r['resources']['samples'] for r in searches),
                'edges':sum(r['resources']['edges'] for r in searches),
                'compute_seconds':sum(r['numeric_compute_seconds'] for r in searches),
                'model_calls':sum(r['actual_advisor_calls'] for r in searches),
                'policy_errors':sum(r['policy_errors'] for r in searches)},
            'candidate_source_sha256':sha(out/'candidates'/f'{name}.json')})
        print('evaluated',name,json.dumps(diagnostic,ensure_ascii=False),flush=True)
    verify_candidates(out);save_json(out/'status.json',{'status':'evaluated'})

class PaperText(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'):self.skip+=1
        if tag in ('p','section','h1','h2','h3','h4','li','tr'):self.parts.append('\n')
    def handle_endtag(self,tag):
        if tag in ('script','style'):self.skip=max(0,self.skip-1)
    def handle_data(self,data):
        if not self.skip:self.parts.append(data)

def literature(out):
    """留存全文来源与读取材料；下载本身不冒充已审阅全文。"""
    def fetch(item):
        name,identifier=item;url='https://arxiv.org/html/'+identifier
        target=out/'literature'/name;target.mkdir(exist_ok=False)
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 research-method-review'})
            with urllib.request.urlopen(request,timeout=35) as response:raw=response.read();resolved=response.url
            (target/'paper.html').write_bytes(raw);parser=PaperText();parser.feed(raw.decode('utf-8'))
            text='\n'.join(line.strip() for line in ''.join(parser.parts).splitlines() if line.strip())
            (target/'paper.txt').write_text(text)
            return {'name':name,'requested_url':url,'resolved_url':resolved,'sha256':sha(target/'paper.html'),
                    'text_characters':len(text),'status':'downloaded_not_yet_method_reviewed'}
        except Exception as exc:
            return {'name':name,'requested_url':url,'status':'failed','error':repr(exc)}
    with ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(fetch,REFERENCES.items()))
    save_json(out/'literature/sources.json',results)
    for result in results:print(json.dumps(result,ensure_ascii=False),flush=True)

def plot_case(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Rectangle
    from uav_em.environment import ElectromagneticWorld
    from uav_em.visualization import save_gif
    name=data['case'];case,snapshot,cfg=load_case(out,out/'cases'/f'{name}.json')
    d=data['diagnosis'];by_id={r['id']:r for r in data['records']}
    selections=[('incumbent','Incumbent','#222222'),
                (d['best_current_j_id'],'Lowest frozen GP J','#1775cf'),
                (d['posthoc_lowest_exposure_id'],'Posthoc exposure reference','#d95319')]
    # Keep duplicate choices visible in legends, offset line styles only, never coordinates.
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for family,marker,color in [('incumbent','*','#222222'),('geometric','o','#8499ac'),('rrt_search','^','#6c65b0')]:
        rs=[r for r in data['records'] if r['family']==family and r['feasible']]
        axes[0].scatter([r['metrics']['objective'] for r in rs],[r['evaluation']['future_excess_db_seconds'] for r in rs],
                        marker=marker,c=color,label=family,alpha=.6,s=36)
    for identifier,label,color in selections:
        record=by_id[identifier]
        axes[0].scatter(record['metrics']['objective'],record['evaluation']['future_excess_db_seconds'],
                        s=140,facecolors='none',edgecolors=color,linewidths=2,label=label)
    for r in data['records']:
        if r['feasible'] and r['metrics']['length_m']>d['length_cap_m']+1e-8:
            axes[0].scatter(r['metrics']['objective'],r['evaluation']['future_excess_db_seconds'],c='red',marker='x',s=28)
    axes[0].set(xlabel='Current fixed GP objective J',ylabel='Future excess (dB s)',
                title='Finite pool; red x: > 1.25 x incumbent length');axes[0].legend(fontsize=7)
    high=snapshot.geometry.high
    def geometry(ax):
        for building in snapshot.geometry.buildings:
            ax.add_patch(Polygon(building.footprint_m,facecolor='#444444' if building in snapshot.geometry.blocking_buildings else '#aaaaaa',alpha=.6))
        for box in cfg.map.obstacles_m:ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],color='#444444'))
        ax.set(xlim=(0,high[0]),ylim=(0,high[1]),xlabel='x (m)',ylabel='y (m)',aspect='equal')
    geometry(axes[1])
    for idx,(identifier,label,color) in enumerate(selections):
        route=np.array(by_id[identifier]['path']);axes[1].plot(*route.T,color=color,linestyle=('-','--',':')[idx],label=label,lw=2)
    axes[1].legend(fontsize=7);axes[1].set_title('Posthoc reference is NOT online planner input')
    fig.suptitle(f'{name} @ {snapshot.time_seconds:.1f}s | fixed-route counterfactual, NOT closed loop')
    fig.savefig(out/'figures'/f'{name}.png',dpi=145);plt.close(fig)

    world=ElectromagneticWorld(cfg);grid=snapshot.geometry.grid()
    duration=max(by_id[identifier]['evaluation']['flight_seconds'] for identifier,_,_ in selections)
    offsets=np.linspace(0,duration,36);truths=[world.truth(grid,snapshot.time_seconds+.5+t) for t in offsets]
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),layout='constrained')
    cmap=plt.get_cmap('inferno').copy();cmap.set_bad('#d6d6d6')
    image=axes[0].imshow(truths[0],origin='lower',extent=[0,*high[:1],0,high[1]],vmin=-100,vmax=-60,cmap=cmap,alpha=.85)
    # GP remains frozen through this animation; grey explicitly marks unsupported cells.
    axes[1].imshow(np.ma.masked_where(~snapshot.supported,snapshot.estimate_dbm),origin='lower',
        extent=[0,high[0],0,high[1]],vmin=-100,vmax=-60,cmap=cmap,alpha=.85)
    fig.colorbar(image,ax=list(axes),label='Total interference power (dBm)',shrink=.65)
    markers=[]
    for ax in axes:
        geometry(ax)
        local=[]
        for idx,(identifier,label,color) in enumerate(selections):
            path=np.array(by_id[identifier]['path']);ax.plot(*path.T,color=color,lw=1.8,ls=('-','--',':')[idx],label=label)
            local.append(ax.plot(*path[0],marker='o',color=color,markersize=7)[0])
        markers.append(local);ax.legend(fontsize=7,loc='lower left')
    axes[0].set_title('Moving truth: observer only');axes[1].set_title('Frozen GP: grey = unsupported; no new observations')
    title=fig.suptitle('')
    def position(route,distance):
        for a,b in zip(route,route[1:]):
            a,b=np.array(a),np.array(b);length=np.linalg.norm(b-a)
            if length>1e-9 and distance<=length:return a+(b-a)*distance/length
            distance-=length
        return np.array(route[-1])
    def update(index):
        image.set_data(truths[index]);offset=offsets[index]
        for column in markers:
            for marker,(identifier,_,_) in zip(column,selections):
                point=position(by_id[identifier]['path'],offset*cfg.aircraft.speed_mps)
                marker.set_data([point[0]],[point[1]])
        title.set_text(f'{name}: t={snapshot.time_seconds+.5+offset:.1f}s | FIXED routes, no replanning\n'
                       'Orange = posthoc truth reference, unavailable to LLM; this is NOT an online-policy result')
    update(0);fig.savefig(out/'figures'/f'{name}_replay.png',dpi=120)
    save_gif(fig,update,list(range(len(offsets))),[180]*(len(offsets)-1)+[1800],out/'replay'/f'{name}.gif');plt.close(fig)

def report(out):
    protocol=check(out);manifest=verify_candidates(out);rows=[];audits=[];sections=[]
    for name in protocol['cases']:
        data=read(out/'evaluations'/f'{name}.json');d=data['diagnosis'];by_id={r['id']:r for r in data['records']}
        if data['candidate_source_sha256']!=sha(out/'candidates'/f'{name}.json'):raise ValueError('evaluation source mismatch')
        plot_case(out,data)
        for role,identifier in [('原剩余路线','incumbent'),('当前GP J最低',d['best_current_j_id']),
                                ('事后最低暴露参考',d['posthoc_lowest_exposure_id'])]:
            r=by_id[identifier];e=r['evaluation'];m=r['metrics']
            rows.append(f"|{name}|{role}|{m['objective']:.3f}|{m['length_m']:.2f}|{e['future_excess_db_seconds']:.3f}|{e['supported_fraction_at_decision']:.1%}|")
        sections.append(f'<h2>{name}</h2><p>有限池搜索覆盖缺口：{d["finite_pool_search_coverage_gap"]}；'
                        f'当前J拒绝实质减暴露候选：{d["acceptance_task_tradeoff_evidence"]}。'
                        f'可行{d["feasible_count"]}条，长度参考内{d["eligible_count"]}条。</p>'
                        f'<img src="figures/{name}.png" width="100%"><p><a href="replay/{name}.gif">固定路线反事实GIF</a> · '
                        f'<a href="evaluations/{name}.json">全部指标/搜索记录</a></p>')
        searches=[r['report'] for r in data['records'] if r['family']=='rrt_search']
        audits.append({'case':name,'independent_runs':len(searches),'calls_zero':all(r['actual_advisor_calls']==0 for r in searches),
            'no_policy_errors':all(r['policy_errors']==0 for r in searches),
            'budgets_ok':all(r['resources']['edges']<=10000 and r['resources']['samples']<=2100 for r in searches),
            'all_endpoints_preserved':all(np.allclose(r['path'][0],by_id['incumbent']['path'][0]) and
                np.allclose(r['path'][-1],by_id['incumbent']['path'][-1]) for r in data['records'] if r['path'])})
    md='# 固定快照修复瓶颈诊断\n\n候选先保存并冻结，再实例化真值环境。0次模型调用，72次独立RRT搜索。'
    md+='几何池与多次搜索的总资源远高于一次决策，只是离线覆盖参考，不能冒充同预算方法。\n\n'
    md+='评价为固定路线开环反事实：从快照后0.5s以15m/s飞行，无新点测/重规划，不包含悬停暴露。'
    md+='积分步长≤0.25s。橙色路线按未来真值事后选择，不可给LLM；不是在线绕行成功结果。\n\n'
    md+='长度≤原路线1.25倍、积分至少降低10%只是预声明诊断阈值，不是硬安全/控制条件。\n\n'
    md+='|快照|选择|当前J|长度/m|未来超参考积分/dB·s|当前支持比例|\n|---|---|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n'
    (out/'reports/diagnostics.md').write_text(md)
    links=' · '.join(f'<a href="{n}.gif">{n}</a>' for n in protocol['cases'])
    (out/'replay/index.html').write_text('<meta charset="utf-8"><h1>固定路线反事实回放</h1><p>没有新观测与重规划。橙色为事后真值参考，不是LLM路线。</p>'+links)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>修复瓶颈诊断</title><style>body{font:17px/1.8 sans-serif;max-width:1200px;margin:30px auto;padding:20px}img{max-width:100%}</style>'
        '<h1>修复瓶颈诊断：搜索、当前GP目标与未来暴露</h1>'
        '<p>本轮没有新LLM试验或闭环策略收益结论。先用公共信息生成/冻结候选，再用隐藏真值事后评价。'
        'GIF是固定路线反事实，无补测/重规划；橙色事后参考不可供LLM使用。</p>'
        '<p><a href="reports/diagnostics.md">指标表</a> · <a href="replay/index.html">4个GIF</a> · '
        '<a href="reports/audit.json">冻结/预算审计</a> · <a href="protocol.json">预声明协议</a> · '
        '<a href="../../docs/repair_diagnostics.md">结论与最小算子接口</a> · '
        '<a href="../../docs/related_methods.md">相关方法对比</a></p>'+''.join(sections))
    audit={'source_hashes_match':True,'all_inputs_match':True,'candidate_freeze_match':True,
           'candidate_manifest_sha256':sha(out/'candidate_manifest.json'),'cases':audits,
           'replay_sha256':{p.name:sha(p) for p in sorted((out/'replay').glob('*.gif'))},
           'evaluation_sha256':{p.name:sha(p) for p in sorted((out/'evaluations').glob('*.json'))}}
    if not all(all(a[k] for k in ('calls_zero','no_policy_errors','budgets_ok','all_endpoints_preserved')) for a in audits):
        raise ValueError('audit failed')
    save_json(out/'reports/audit.json',audit);save_json(out/'status.json',{'status':'completed_numeric_diagnostic'})

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','propose','evaluate','report','literature','all'])
    parser.add_argument('--out',type=Path,default=DEFAULT);args=parser.parse_args()
    if args.phase=='all':
        for function in (prepare,propose,evaluate,report):function(args.out)
    else:globals()[args.phase](args.out)
if __name__=='__main__':main()
