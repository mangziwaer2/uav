"""C一致搜索原语：冻结16配对、8闭环、0模型/生成调用。"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json
from uav_em.aligned_search import AlignedOptimizer, OBJECTIVES, VERSION, run_primitive, PowerTimeSearchObjective
from uav_em.balanced_objective import PowerTimeSettings, choose, paired_seed, power_time_score
from uav_em.config import load_config
from uav_em.decision_log import render_decision_log
from uav_em.repair_process import canonical_path
from uav_em.repair_search import FixedObjective, WorkBudget
from run_repair_diagnostics import load_case
DEFAULT=ROOT/'outputs/interference_aligned_search_20261111'
PREVIOUS=ROOT/'outputs/interference_causal_feedback_20261111'
BALANCED=ROOT/'outputs/interference_balanced_objective_20261111'
SEEDS=(611,613,617,619)
PROFILES=('local_pad1','global_focus')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('cases','scenarios','pool_reports','paired','runs','evaluations','reports/runtime_source','replay','logs'):
        (out/folder).mkdir(parents=True)
    tasks=read(PREVIOUS/'protocol.json')['tasks'];external={}
    for folder in ('cases','pool_reports'):
        for p in sorted((PREVIOUS/folder).iterdir()):
            shutil.copy2(p,out/folder/p.name);external[str(p.relative_to(ROOT))]=sha(p)
    for seed in SEEDS:
        p=BALANCED/'scenarios'/f'dev{seed}.json';shutil.copy2(p,out/'scenarios'/p.name);external[str(p.relative_to(ROOT))]=sha(p)
    shutil.copy2(ROOT/'docs/aligned_search_v1.md',out/'reports/frozen_design.md')
    sources={}
    for p in [*(ROOT/'src/uav_em').glob('*.py'),Path(__file__).resolve(),ROOT/'tools/run_repair_diagnostics.py',ROOT/'tools/render_navigation_replays.py',ROOT/'tests/test_aligned_search.py']:
        rel=p.relative_to(ROOT);target=out/'reports/runtime_source'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);sources[str(rel)]=sha(p)
    save_json(out/'protocol.json',dict(study=VERSION,tasks=tasks,seeds=SEEDS,objectives=OBJECTIVES,profiles=PROFILES,
        main_profile='local_pad1',paired_tasks=16,new_flights=8,model_calls=0,new_generated_submissions=0,
        quota=700,samples=2100,edges=10000,workers=4,weight=1,validation_test_read=False,
        source_sha256=sources,external_sha256=external,design_sha256=sha(out/'reports/frozen_design.md'),
        input_sha256={str(p.relative_to(out)):sha(p) for f in ('cases','scenarios','pool_reports') for p in (out/f).iterdir()}))
    save_json(out/'status.json',dict(status='prepared'))

def check(out):
    p=read(out/'protocol.json')
    assert sha(out/'reports/frozen_design.md')==p['design_sha256']
    for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==sha(out/'reports/runtime_source'/rel)==h,rel
    for rel,h in p['input_sha256'].items():assert sha(out/rel)==h,rel
    for rel,h in p['external_sha256'].items():assert sha(ROOT/rel)==h,rel
    return p

def manifest(out,folder,target):
    save_json(out/'reports'/target,{str(p.relative_to(out)):sha(p) for p in sorted((out/folder).rglob('*')) if p.is_file()})

def verify(out,name,count=None):
    m=read(out/'reports'/name)
    if count is not None:assert len(m)==count
    for rel,h in m.items():assert sha(out/rel)==h,rel

def paired(out):
    p=check(out)
    if any((out/'paired').iterdir()):raise FileExistsError('paired exists')
    for task,name in p['tasks'].items():
        case,snapshot,cfg=load_case(out,out/'cases'/f'{name}.json');pool=read(out/'pool_reports'/f'{task}.json')
        for profile in PROFILES:
            pair=[]
            for objective in OBJECTIVES:
                r=run_primitive(snapshot,case['context'],pool,case['max_length_m'],case['seed'],objective,profile,PowerTimeSettings(**case['score_settings']))
                save_json(out/'paired'/f'{task}_{profile}_{objective}.json',dict(task=task,case=name,**r));pair.append(r)
                print(name,profile,objective,round(r['selected_balance']['score_seconds'],4),r['resources']['edges'],flush=True)
            assert pair[0]['initial_resources']==pair[1]['initial_resources']
            assert pair[0]['trace'][0]['proposal']==pair[1]['trace'][0]['proposal']
            assert pair[0]['searches'][0]['actual_seed']==pair[1]['searches'][0]['actual_seed']
    manifest(out,'paired','paired_manifest.json')

def flight_task(out,objective,seed):
    from uav_em.simulation import simulate
    folder=out/'runs'/f'{objective}_dev{seed}_seed{seed}';folder.mkdir()
    motions=[]
    def hook(stage,data):
        if stage=='flight_executed':motions.append(dict(end_time_seconds=data['time_seconds'],traversed_m=data['traversed_m']))
    try:
        cfg=load_config(out/'scenarios'/f'dev{seed}.json')
        settings=PowerTimeSettings(prior_mean_dbm=cfg.electromagnetic.background_dbm,prior_std_db=cfg.belief.unknown_error_db)
        ep=simulate(cfg,advisor_name=f'{objective}_numeric_no_model',route_optimizer=AlignedOptimizer(seed,objective,settings),debug_hook=hook)
        ep['assumptions'].update(aligned_search_study=True,runtime_model_calls=0,new_generated_submissions=0)
        for d in ep['decisions']:
            d['decision']['reason']=d['proposed_decision']['reason']=f'研究者编写的固定local_pad1，内部搜索{objective}，共同C接受，无模型'
            d['accepted_route_fixed_objective_meaning']='old_J_diagnostic_not_acceptance'
        save_json(folder/'episode.json',ep);save_json(folder/'execution.json',motions);save_json(folder/'metrics.json',ep['metrics'])
        (folder/'decision_log.txt').write_text(render_decision_log(ep))
        save_json(folder/'status.json',dict(status='completed',method=objective,scene=f'dev{seed}',seed=seed,**ep['metrics']))
        return dict(method=objective,seed=seed,success=ep['metrics']['success'],length=ep['metrics']['executed_length_m'],excess=ep['metrics']['interference_excess_integral_db_seconds'])
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        save_json(folder/'status.json',dict(status='failed',method=objective,scene=f'dev{seed}',seed=seed,error=error))
        return dict(method=objective,seed=seed,error=error)

def flights(out):
    check(out);verify(out,'paired_manifest.json',16)
    if any((out/'runs').iterdir()):raise FileExistsError('runs exist')
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(flight_task,out,o,s) for o in OBJECTIVES for s in SEEDS]):print(json.dumps(f.result()),flush=True)
    manifest(out,'runs','run_manifest.json')

def audit_searches(snapshot,report,speed,settings):
    checks=[]
    for s in report['searches']:
        assert s['actual_seed']==paired_seed(snapshot.seed if hasattr(snapshot,'seed') else report['_audit_seed'],snapshot.time_seconds,s['seed_stage'],s['seed_purpose'])
        if s.get('search_objective')=='power_time_C' and s['path']:
            expected=power_time_score(FixedObjective(snapshot,-80,WorkBudget()),s['path'],speed,settings)['score_seconds']
            assert abs(expected-s['cost'])<1e-6
            checks.append(expected)
    return len(checks)

def evaluate(out):
    p=check(out);verify(out,'paired_manifest.json',16);verify(out,'run_manifest.json')
    if (out/'reports/summary.json').exists():raise FileExistsError('evaluation exists')
    from uav_em.environment import ElectromagneticWorld
    from uav_em.risk_diagnostic_evaluation import evaluate_route
    from uav_em.balanced_evaluation import evaluate_motions
    from uav_em.diagnostic_evaluation import truth_at_times
    paired_rows=[];audit=[]
    for path in sorted((out/'paired').glob('*.json')):
        r=read(path);case,snapshot,cfg=load_case(out,out/'cases'/f"{r['case']}.json")
        selected=choose(r['candidates']);assert canonical_path(selected['path'])==canonical_path(r['path'])
        problem=FixedObjective(snapshot,-80,WorkBudget(2100,100000))
        settings=PowerTimeSettings(**case['score_settings'])
        for candidate in r['candidates']:
            if candidate['feasible']:
                c=power_time_score(problem,candidate['path'],cfg.aircraft.speed_mps,settings)
                assert abs(c['score_seconds']-candidate['balance']['score_seconds'])<1e-7
        assert selected['metrics']['length_m']<=case['max_length_m']+1e-7
        assert all(snapshot.geometry.edge_free(a,b) for a,b in zip(r['path'],r['path'][1:]))
        assert np.allclose(r['path'][0],case['context']['vehicle']['position_m']) and np.allclose(r['path'][-1],case['context']['vehicle']['goal_m'])
        r['_audit_seed']=case['seed'];n=audit_searches(snapshot,r,cfg.aircraft.speed_mps,settings)
        assert r['resources']['samples']<=2100 and r['resources']['edges']<=10000
        e=evaluate_route(ElectromagneticWorld(cfg),snapshot,r['path'])
        save_json(out/'evaluations'/path.name,dict(input_sha256=sha(path),evaluation=e))
        paired_rows.append(dict(case=r['case'],profile=r['profile'],objective=r['search_objective'],
            current_C_seconds=r['selected_balance']['score_seconds'],length_m=selected['metrics']['length_m'],
            **e,edges=r['resources']['edges'],samples=r['resources']['samples'],numeric_seconds=r['numeric_compute_seconds']))
        audit.append(dict(task=path.stem,candidate_scores_verified=True,internal_C_paths_verified=n,budgets_endpoints_geometry_verified=True))
    rows=[]
    for folder in sorted((out/'runs').iterdir()):
        status=read(folder/'status.json')
        if status['status']!='completed':rows.append(status);continue
        ep=read(folder/'episode.json');cfg=load_config(out/'scenarios'/f"{status['scene']}.json");world=ElectromagneticWorld(cfg)
        actual=evaluate_motions(world,read(folder/'execution.json'))
        assert abs(actual['reintegrated_excess_db_seconds']-ep['metrics']['interference_excess_integral_db_seconds'])<1e-7
        assert abs(actual['reintegrated_exposure_seconds']-ep['metrics']['high_interference_exposure_seconds'])<1e-7
        assert abs(actual['flight_seconds_from_trace']*cfg.aircraft.speed_mps-ep['metrics']['executed_length_m'])<1e-6
        cap=ep['decisions'][0]['repair_optimization']['mission_distance_cap_m'];hover_q=hover_t=0.
        for d in ep['decisions']:
            r=d['repair_optimization'];w=r['resources'];selected=choose(r['candidates'])
            assert w['samples']<=2100 and w['edges']<=10000
            assert r['actual_advisor_calls']==r['policy_errors']==0 and r['final_geometry_verified']
            assert r['mission_distance_cap_m']==cap
            flown=sum(e['distance_m'] for e in ep['events'] if e['event']=='flight' and e['time_seconds']<=d['time_seconds']+1e-8)
            assert abs(flown-r['executed_distance_m'])<1e-6
            assert selected and flown+selected['metrics']['length_m']<=cap+1e-6
            assert np.allclose(selected['path'][0],d['context']['vehicle']['position_m']) and np.allclose(selected['path'][-1],cfg.map.goal_m)
            assert all(world.geometry.edge_free(a,b) for a,b in zip(selected['path'],selected['path'][1:]))
            required=selected['metrics']['length_m']/cfg.aircraft.speed_mps*cfg.aircraft.flight_power_w/3600+cfg.simulation.decision_delay_seconds*cfg.aircraft.hover_power_w/3600
            assert required<=d['execution_checkpoint']['remaining_energy_wh']-cfg.aircraft.reserve_energy_wh+1e-7
            for search in r['searches']:
                assert search['actual_seed']==paired_seed(cfg.seed,d['time_seconds'],search['seed_stage'],search['seed_purpose'])
            if d['accepted_action'] is not None:
                dt=min(cfg.simulation.decision_delay_seconds,cfg.simulation.max_seconds-d['time_seconds'])
                n=max(1,int(np.ceil(dt/.25)));times=d['time_seconds']+(np.arange(n)+.5)*dt/n
                power=truth_at_times(world,np.tile(d['context']['vehicle']['position_m'],(n,1)),times)
                hover_q+=float(np.power(10.,(power+80)/10).sum()*dt/n);hover_t+=dt
        assert ep['metrics']['executed_length_m']<=cap+1e-6
        row=dict(scene=status['scene'],method=status['method'],**ep['metrics'],**actual,
            actual_joint_score_lambda1_seconds=actual['flight_seconds_from_trace']+actual['flight_power_integral_normalized_seconds'],
            actual_joint_with_hover_lambda1_seconds=actual['flight_seconds_from_trace']+actual['flight_power_integral_normalized_seconds']+hover_q+hover_t,
            hover_power_seconds=hover_q,hover_seconds=hover_t,
            numeric_seconds=sum(d['repair_optimization']['numeric_compute_seconds'] for d in ep['decisions']))
        rows.append(row);audit.append(dict(run=folder.name,decisions=len(ep['decisions']),geometry_energy_lifetime_caps_and_resources_verified=True,exposure_reintegrated=True))
    totals={}
    for objective in OBJECTIVES:
        group=[r for r in rows if r.get('method')==objective]
        totals[objective]=dict(all_arrive=len(group)==4 and all(r.get('success',False) for r in group))
        for key in ('executed_length_m','interference_excess_integral_db_seconds','flight_power_integral_normalized_seconds','actual_joint_score_lambda1_seconds','actual_joint_with_hover_lambda1_seconds','repair_edge_evaluations','numeric_seconds'):
            totals[objective][key]=sum(r.get(key,0) for r in group)
    save_json(out/'reports/summary.json',dict(paired=paired_rows,flights=rows,totals=totals,model_calls=0,new_generated_submissions=0))
    save_json(out/'reports/audit.json',audit)

def report(out):
    check(out);verify(out,'paired_manifest.json',16);verify(out,'run_manifest.json')
    summary=read(out/'reports/summary.json')
    from render_navigation_replays import render_replays
    render_replays(out,frame_stride=6,playback_speed=2.,decision_pause_seconds=.5)
    md='# C一致RRT*搜索原语\n\n研究者编写的结构改造；16同状态求解、8真实闭环，0模型/新生成提交。同一共同池、C接受器和总预算；仅池后一次固定local_pad1搜索改变内部代价。\n\n'
    md+='|场景|搜索|到达|航程/m|超参考/dB·s|真实T+Q/s|含悬停T+Q/s|边评价|数值墙钟/s|\n|---|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in summary['flights']:
        if 'error' in r:md+=f"|{r['scene']}|{r['method']}|失败|{r['error']}||||||\n";continue
        md+=f"|{r['scene']}|{r['method']}|{r['success']}|{r['executed_length_m']:.3f}|{r['interference_excess_integral_db_seconds']:.3f}|{r['actual_joint_score_lambda1_seconds']:.3f}|{r['actual_joint_with_hover_lambda1_seconds']:.3f}|{r['repair_edge_evaluations']}|{r['numeric_seconds']:.2f}|\n"
    md+='\n## 同状态固定路线评价\n\n|状态|操作|搜索|当前C/s|真实T+Q/s|超参考/dB·s|航程/m|边|\n|---|---|---|---:|---:|---:|---:|---:|\n'
    for r in summary['paired']:md+=f"|{r['case']}|{r['profile']}|{r['objective']}|{r['current_C_seconds']:.3f}|{r['future_joint_seconds']:.3f}|{r['future_excess_db_seconds']:.3f}|{r['length_m']:.3f}|{r['edges']}|\n"
    md+='\n同状态评价是开环反事实；GIF是实际闭环。未知仍未知，−80dBm仅为归一化单位/软参考；不是安全阈值。局部快照代理不预测抵达时真实场。结构改造不记作LLM收益，不根据真值改参数或选闭环profile。\n'
    (out/'RESULTS.md').write_text(md)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>C一致搜索原语</title><style>body{font:17px/1.8 sans-serif;max-width:1250px;margin:auto}a{margin-right:18px}</style><h1>C一致RRT*搜索原语</h1><p>16同状态求解、8实际闭环；研究者编写，0模型调用。共同池仍为旧J，仅池后固定local_pad1原语比较J/C。</p><p><a href="replay/index.html">8个闭环GIF</a><a href="RESULTS.md">全部结果</a><a href="reports/summary.json">结构化指标</a><a href="reports/audit.json">核验</a><a href="reports/frozen_design.md">前瞻协议</a></p>')
    save_json(out/'status.json',dict(status='completed',paired_tasks=16,new_flights=8,model_calls=0,new_generated_submissions=0))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','paired','flights','evaluate','report','all']);parser.add_argument('--out',type=Path,default=DEFAULT);a=parser.parse_args()
    for fn in (prepare,paired,flights,evaluate,report) if a.phase=='all' else (globals()[a.phase],):fn(a.out)
if __name__=='__main__':main()
