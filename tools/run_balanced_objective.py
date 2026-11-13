"""航程—连续功率共同评价：冻结36配对、24新飞行，0新模型调用。"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json
from uav_em.balanced_objective import BalancedOptimizer, PowerTimeSettings, choose, paired_seed
from uav_em.config import load_config
from uav_em.decision_log import render_decision_log
from uav_em.process_attribution import check_identical_executed_prefix
from uav_em.repair_process import canonical_path
from run_repair_diagnostics import load_case

PREVIOUS = ROOT/'outputs/interference_process_attribution_20261111'
GENERATION = ROOT/'outputs/interference_offline_generation_20261111'
DEFAULT = ROOT/'outputs/interference_balanced_objective_20261111'
SEEDS = (611,613,617,619)
MODES = ('paired_prefix_pool','balanced_pool','balanced_local','balanced_full','balanced_pool_low','balanced_pool_high')
WEIGHTS = {'balanced_pool_low':.25,'balanced_pool_high':4.}
read = lambda p:json.loads(p.read_text())
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def optimizer(out, mode, cfg):
    program = read(out/'programs'/('local.json' if mode=='balanced_local' else 'full.json')) if mode in ('balanced_local','balanced_full') else None
    settings = PowerTimeSettings(weight=WEIGHTS.get(mode,1.), prior_mean_dbm=cfg.electromagnetic.background_dbm,
                                 prior_std_db=cfg.belief.unknown_error_db)
    return BalancedOptimizer(cfg.seed,settings,program,'prefix' if mode=='paired_prefix_pool' else 'balanced')


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('cases','scenarios','programs','paired','runs','reports/runtime_source','logs','evaluations','replay','figures'):
        (out/folder).mkdir(parents=True)
    shutil.copy2(ROOT/'docs/balanced_objective_v3.md',out/'reports/frozen_design.md')
    external={}
    for p in sorted((PREVIOUS/'cases').glob('*')):
        shutil.copy2(p,out/'cases'/p.name);external[str(p.relative_to(ROOT))]=sha(p)
    for seed in SEEDS:
        p=GENERATION/'scenarios'/f'dev{seed}.json'
        shutil.copy2(p,out/'scenarios'/p.name);external[str(p.relative_to(ROOT))]=sha(p)
    for name,parent in (('local','local_prefix'),('full','full_process')):
        p=PREVIOUS/'programs'/f'{parent}.json'
        shutil.copy2(p,out/'programs'/f'{name}.json');external[str(p.relative_to(ROOT))]=sha(p)
    sources={}
    paths=[*(ROOT/'src/uav_em').glob('*.py'),Path(__file__).resolve(),ROOT/'tools/run_repair_diagnostics.py',
           ROOT/'tools/render_navigation_replays.py',ROOT/'tests/test_balanced_objective.py']
    for p in paths:
        rel=p.relative_to(ROOT);target=out/'reports/runtime_source'/rel
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);sources[str(rel)]=sha(p)
    inputs={str(p.relative_to(out)):sha(p) for folder in ('cases','scenarios','programs') for p in sorted((out/folder).iterdir())}
    save_json(out/'protocol.json',{'study':'balanced_power_time_v3_development_check','seeds':list(SEEDS),'modes':list(MODES),
        'weights':{m:WEIGHTS.get(m,1.) for m in MODES},'main_weight':1.,'paired_tasks':36,'new_flights':24,
        'new_model_calls':0,'new_generated_submissions':0,'workers':4,
        'cases':sorted(p.stem for p in (out/'cases').glob('*.json')),
        'source_sha256':sources,'input_sha256':inputs,'external_source_sha256':external,
        'design_sha256':sha(out/'reports/frozen_design.md'),
        'main_comparison':['paired_prefix_pool','balanced_pool'],'sensitivity_not_winner_selection':True,
        'random_stream':'scene seed + time us + repeat0 + stage/node + purpose; no cumulative plan count',
        'unknown':'explicit Gaussian-dBm prior expected-power cost; no unsupported numerical grid values read',
        'all_routes_current_snapshot_proxy_not_arrival_prediction':True,'validation_test_generated_or_read':False})
    save_json(out/'status.json',{'status':'prepared'})


def check(out):
    p=read(out/'protocol.json')
    assert sha(out/'reports/frozen_design.md')==p['design_sha256']
    for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==sha(out/'reports/runtime_source'/rel)==h,rel
    for rel,h in p['input_sha256'].items():assert sha(out/rel)==h,rel
    for rel,h in p['external_source_sha256'].items():assert sha(ROOT/rel)==h,rel
    return p


def paired(out):
    protocol=check(out)
    if any((out/'paired').iterdir()):raise FileExistsError('paired already exists')
    audits=[]
    for name in protocol['cases']:
        case,snapshot,cfg=load_case(out,out/'cases'/f'{name}.json')
        reports={}
        for mode in MODES:
            opt=optimizer(out,mode,cfg);opt.base.reference_length_m=case['reference_length_m']
            interval=case['context'].get('execution_feedback',{}).get('interval')
            opt.base.flown_m=case['flown_m']-(interval['flown_m'] if interval else 0.)
            plan,r=opt.optimize(snapshot,deepcopy(case['context']),case['route'],case['max_length_m'],case['seed'],0)
            assert plan.path and r['policy_errors']==0
            assert r['resources']['samples']<=2100 and r['resources']['edges']<=10000
            assert plan.length_m+case['flown_m']<=r['mission_distance_cap_m']+1e-6
            assert plan.length_m<=case['max_length_m']+1e-6
            save_json(out/'paired'/f'{name}_{mode}.json',{'case':name,'mode':mode,'path':plan.path,'report':r})
            reports[mode]=r
            print(name,mode,'length',round(plan.length_m,3),'score',round(r['selected_balance']['score_seconds'],3),flush=True)
        n=len(reports['balanced_pool']['candidates'])
        fields=('id','path','metrics','prefix','feasible','failure')
        common=lambda r:[{k:c[k] for k in fields} for c in r['candidates'][:n]]
        assert all(common(r)==common(reports['balanced_pool']) for r in reports.values())
        assert all(r['common_pool_resources']==reports['balanced_pool']['common_pool_resources'] for r in reports.values())
        prefix=check_identical_executed_prefix(reports['balanced_full'],reports['balanced_local'])
        audits.append({'case':name,'common_initial_pool_and_work_identical':True,'local_full_executed_prefix':prefix})
    save_json(out/'reports/paired_manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted((out/'paired').iterdir())})
    save_json(out/'reports/paired_audit.json',audits)


def verify_paired(out):
    check(out);manifest=read(out/'reports/paired_manifest.json');assert len(manifest)==36
    for rel,h in manifest.items():assert sha(out/rel)==h,rel


def flight_task(out,mode,seed):
    from uav_em.simulation import simulate
    folder=out/'runs'/f'{mode}_dev{seed}_seed{seed}';folder.mkdir()
    motions=[]
    def hook(stage,data):
        if stage=='flight_executed':
            motions.append({'end_time_seconds':data['time_seconds'],'traversed_m':data['traversed_m']})
    try:
        cfg=load_config(out/'scenarios'/f'dev{seed}.json')
        ep=simulate(cfg,advisor_name=f'{mode}_numeric_no_model',route_optimizer=optimizer(out,mode,cfg),debug_hook=hook)
        ep['assumptions'].update(balanced_objective_development=True,runtime_model_calls=0,new_generated_submissions=0,
                                 paired_search_streams=True,all_route_scores_use_current_snapshot=True)
        for d in ep['decisions']:
            d['decision']['reason']=d['proposed_decision']['reason']='航程—功率数值开发对照；旧J搜索，接受规则单独记录，无新模型调用'
            d['accepted_route_fixed_objective_meaning']='old J search diagnostic, not balanced acceptance score'
        save_json(folder/'episode.json',ep);save_json(folder/'metrics.json',ep['metrics'])
        save_json(folder/'execution.json',motions)
        (folder/'decision_log.txt').write_text('连续功率与航时/旧R10数值对照；详细共同评分与随机种子见episode。\n'+render_decision_log(ep))
        save_json(folder/'status.json',{'status':'completed','method':mode,'scene':f'dev{seed}','seed':seed,**ep['metrics']})
        return {'mode':mode,'seed':seed,'success':ep['metrics']['success'],'length_m':ep['metrics']['executed_length_m'],
                'excess_db_seconds':ep['metrics']['interference_excess_integral_db_seconds']}
    except Exception as exc:
        save_json(folder/'status.json',{'status':'failed','method':mode,'scene':f'dev{seed}','seed':seed,'error':f'{type(exc).__name__}: {exc}'})
        return {'mode':mode,'seed':seed,'error':f'{type(exc).__name__}: {exc}'}


def flights(out):
    verify_paired(out)
    if any((out/'runs').iterdir()):raise FileExistsError('flights already exists')
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(flight_task,out,m,s) for m in MODES for s in SEEDS]):print(json.dumps(f.result()),flush=True)
    save_json(out/'reports/run_manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted((out/'runs').rglob('*')) if p.is_file()})


def verify_runs(out):
    verify_paired(out)
    for rel,h in read(out/'reports/run_manifest.json').items():assert sha(out/rel)==h,rel


def evaluate(out):
    verify_runs(out)
    if (out/'reports/summary.json').exists():raise FileExistsError('evaluation already exists')
    from uav_em.balanced_evaluation import evaluate_motions
    from uav_em.diagnostic_evaluation import truth_at_times
    from uav_em.environment import ElectromagneticWorld
    rows=[];audits=[]
    for folder in sorted((out/'runs').iterdir()):
        status=read(folder/'status.json')
        if status['status']!='completed':
            rows.append({'scene':status['scene'],'method':status['method'],'success':False,'error':status['error']});continue
        ep=read(folder/'episode.json');cfg=load_config(out/'scenarios'/f"{status['scene']}.json");world=ElectromagneticWorld(cfg)
        motions=read(folder/'execution.json')
        actual=evaluate_motions(world,motions)
        assert abs(actual['reintegrated_excess_db_seconds']-ep['metrics']['interference_excess_integral_db_seconds'])<1e-7
        assert abs(actual['reintegrated_exposure_seconds']-ep['metrics']['high_interference_exposure_seconds'])<1e-7
        assert abs(actual['flight_seconds_from_trace']*cfg.aircraft.speed_mps-ep['metrics']['executed_length_m'])<1e-6
        cap=ep['decisions'][0]['repair_optimization']['mission_distance_cap_m'];checks=[];hover_q=hover_e=hover_t=0.
        for d in ep['decisions']:
            r=d['repair_optimization'];w=r['resources'];chosen=choose(r['candidates'],r['selection_rule'])
            assert r['policy_errors']==r['actual_advisor_calls']==0
            assert w['samples']<=2100 and w['edges']<=10000
            assert r['mission_distance_cap_m']==cap
            actual_flown=sum(e['distance_m'] for e in ep['events'] if e['event']=='flight' and e['time_seconds']<=d['time_seconds']+1e-8)
            assert abs(actual_flown-r['executed_distance_m'])<1e-6
            if chosen:
                assert chosen['id']==r['selected_candidate_id']
                assert actual_flown+chosen['metrics']['length_m']<=cap+1e-6
                assert np.allclose(chosen['path'][0],d['context']['vehicle']['position_m'],atol=1e-8,rtol=0)
                assert np.allclose(chosen['path'][-1],d['context']['vehicle']['goal_m'],atol=1e-8,rtol=0)
                assert all(world.geometry.edge_free(a,b) for a,b in zip(chosen['path'],chosen['path'][1:]))
                required=chosen['metrics']['length_m']/cfg.aircraft.speed_mps*cfg.aircraft.flight_power_w/3600+cfg.simulation.decision_delay_seconds*cfg.aircraft.hover_power_w/3600
                assert required<=d['execution_checkpoint']['remaining_energy_wh']-cfg.aircraft.reserve_energy_wh+1e-7
            for search in r['searches']:
                assert search['actual_seed']==paired_seed(cfg.seed,d['time_seconds'],search['seed_stage'],search['seed_purpose'])
            if d['accepted_action'] is not None:
                dt=min(cfg.simulation.decision_delay_seconds,cfg.simulation.max_seconds-d['time_seconds'])
                count=max(1,int(np.ceil(dt/.25)));times=d['time_seconds']+(np.arange(count)+.5)*dt/count
                points=np.tile(d['context']['vehicle']['position_m'],(count,1));power=truth_at_times(world,points,times)
                hover_q+=float(np.power(10.,(power+80)/10).sum()*dt/count)
                hover_e+=float(np.maximum(power-cfg.electromagnetic.interference_reference_dbm,0).sum()*dt/count);hover_t+=dt
            checks.append({'time_seconds':d['time_seconds'],'chosen_geometry_energy_cap_and_budget_valid':bool(chosen),'search_stream_keys_valid':True})
        assert ep['metrics']['executed_length_m']<=cap+1e-6
        row={'scene':status['scene'],'method':status['method'],**ep['metrics'],**actual,
             'decision_hover_power_normalized_seconds':hover_q,'decision_hover_excess_db_seconds':hover_e,
             'actual_joint_score_lambda1_seconds':actual['flight_seconds_from_trace']+actual['flight_power_integral_normalized_seconds'],
             'actual_joint_with_hover_lambda1_seconds':actual['flight_seconds_from_trace']+hover_t+actual['flight_power_integral_normalized_seconds']+hover_q,
             'score_field_nodes':sum(d['repair_optimization']['score_field_nodes'] for d in ep['decisions']),
             'mission_distance_cap_m':cap,'initial_selected_id':ep['decisions'][0]['repair_optimization']['selected_candidate_id']}
        rows.append(row);audits.append({'run':folder.name,'decisions':checks})
    totals={}
    for mode in MODES:
        group=[r for r in rows if r['method']==mode];valid=all(r['success'] and 'actual_joint_score_lambda1_seconds' in r for r in group)
        totals[mode]={'all_arrive':valid,'flights':len(group)}
        if valid:
            for key in ('executed_length_m','interference_excess_integral_db_seconds','flight_power_integral_normalized_seconds','actual_joint_score_lambda1_seconds','actual_joint_with_hover_lambda1_seconds','repair_edge_evaluations'):
                totals[mode][key]=sum(r[key] for r in group)
    base,new=totals['paired_prefix_pool'],totals['balanced_pool']
    main={'all_arrive':base['all_arrive'] and new['all_arrive'],'total_actual_joint_score_improves':False,
          'not_significance_safety_or_LLM_benefit':True,'sensitivity_not_used_to_select_main_weight':True}
    if main['all_arrive']:main['total_actual_joint_score_improves']=new['actual_joint_score_lambda1_seconds']<base['actual_joint_score_lambda1_seconds']
    save_json(out/'reports/summary.json',{'rows':rows,'totals':totals,'main_comparison':main,'model_calls':0,'new_generated_submissions':0})
    save_json(out/'reports/audit.json',audits)


def report(out):
    verify_runs(out);summary=read(out/'reports/summary.json')
    from render_navigation_replays import render_replays
    render_replays(out,frame_stride=6,playback_speed=2.,decision_pause_seconds=.5)
    p=out/'replay/index.html';p.write_text(p.read_text().replace('动画默认按仿真1倍速','本轮动画按仿真2倍速'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    fields=('executed_length_m','interference_excess_integral_db_seconds','actual_joint_score_lambda1_seconds')
    for i,mode in enumerate(MODES):
        group=sorted([r for r in summary['rows'] if r['method']==mode],key=lambda r:r['scene'])
        for ax,key in zip(axes,fields):ax.bar(np.arange(4)+(i-2.5)*.13,[r.get(key,np.nan) for r in group],width=.13,label=mode)
    for ax,title in zip(axes,('Actual flight length (m)','Actual excess (dB s)','Actual time + normalized power (s)')):
        ax.set_xticks(np.arange(4),[f'dev{s}' for s in SEEDS]);ax.set_title(title)
    axes[-1].legend(fontsize=7);fig.suptitle('Paired random streams; fixed main weight 1; development only')
    fig.savefig(out/'figures/comparison.png',dpi=140);plt.close(fig)
    md='# 航程—连续功率共同评价\n\n36同状态配对、24新闭环，0模型/新生成提交。主lambda=1，0.25/4只作敏感性，不据结果选权重。\n\n'
    md+='|场景|方法|到达|航程/m|超参考/dB·s|连续功率积分/归一化s|实际联合分/s|前50m向goal进度/m|边评价|\n|---|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in summary['rows']:
        if 'error' in r:md+=f"|{r['scene']}|{r['method']}|False|失败|失败|失败|失败|失败|失败|\n";continue
        md+=f"|{r['scene']}|{r['method']}|{r['success']}|{r['executed_length_m']:.3f}|{r['interference_excess_integral_db_seconds']:.3f}|{r['flight_power_integral_normalized_seconds']:.3f}|{r['actual_joint_score_lambda1_seconds']:.3f}|{r['first50_goal_distance_progress_m']:.3f}|{r['repair_edge_evaluations']}|\n"
    md+='\n[完整结果](reports/summary.json) · [配对核验](reports/paired_audit.json) · [实际飞行GIF](replay/index.html) · [冻结设计](reports/frozen_design.md)。\n\n未知代价是显式先验假设，不是测量/上界。全路线评分冻结当前GP，不预测抵达时场。原J仍用于搜索，本轮不是搜索目标消融或在线LLM实验。连续功率积分不是物理吸收剂量；−80dBm为归一化单位，不是容忍阈值。\n'
    (out/'RESULTS.md').write_text(md)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>航程与连续功率共同评价</title><style>body{font:17px/1.8 sans-serif;max-width:1450px;margin:25px auto;padding:15px}img{max-width:100%}</style><h1>航程—连续功率共同评价与配对搜索流</h1><p>36同状态配对、24新飞行，0模型/新生成提交。固定主权重1，0.25/4只做敏感性；不是在线LLM收益或未见验证。</p><p><a href="replay/index.html">24个实际飞行GIF</a> · <a href="RESULTS.md">完整数值表</a> · <a href="reports/summary.json">结果与主比较</a> · <a href="reports/paired_audit.json">同状态核验</a> · <a href="../../docs/balanced_objective_v3.md">方法与限制</a></p><img src="figures/comparison.png"><p>全路线当前快照代理；未知仍未知，其代价来自声明先验。−80dBm为功率归一化单位，不是设备安全阈值。主目标变更不归因于模型。</p>')
    save_json(out/'status.json',{'status':'completed','paired_tasks':36,'new_flights':24,'new_model_calls':0,'new_generated_submissions':0})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','paired','flights','evaluate','report','all'])
    parser.add_argument('--out',type=Path,default=DEFAULT);a=parser.parse_args()
    if a.phase=='all':
        for fn in (prepare,paired,flights,evaluate,report):fn(a.out)
    else:globals()[a.phase](a.out)


if __name__=='__main__':main()
