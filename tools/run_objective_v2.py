"""共同候选下旧J与短段风险v2数值闭环；无LLM/新传感/补测。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json
from uav_em.config import load_config
from uav_em.decision_objective import VERSION,ObjectiveOptimizer,ObjectiveSettings,select_candidate
from uav_em.decision_log import render_decision_log
from uav_em.planning import path_length

DEFAULT=ROOT/'outputs/interference_objective_v2_20261111'
SOURCE=ROOT/'outputs/interference_task_readiness_20261111'
METHODS=('pool_j','prefix_v2')
SEEDS=(611,613,617,619)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('scenarios','runs','reports/runtime_source','logs','replay','figures'):(out/folder).mkdir(parents=True)
    shutil.copy2(ROOT/'docs/decision_objective_v2.md',out/'reports/frozen_design.md')
    inputs={};sources={}
    for seed in SEEDS:
        path=SOURCE/'scenarios'/f'dev{seed}.json';dest=out/'scenarios'/path.name
        shutil.copy2(path,dest);inputs[str(dest.relative_to(out))]=sha(dest)
    for p in [*(ROOT/'src/uav_em').glob('*.py'),Path(__file__).resolve(),ROOT/'tools/render_navigation_replays.py',ROOT/'tests/test_decision_objective.py']:
        relative=p.relative_to(ROOT);dest=out/'reports/runtime_source'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);sources[str(relative)]=sha(p)
    save_json(out/'protocol.json',{'study':VERSION,'methods':list(METHODS),'seeds':list(SEEDS),
        'settings':asdict(ObjectiveSettings()),'source_study':str(SOURCE.relative_to(ROOT)),
        'input_sha256':inputs,'source_sha256':sources,'design_sha256':sha(out/'reports/frozen_design.md'),
        'validation_test_generated_or_read':False,'model_calls':0,'no_parameter_search':True,
        'gate':{'all_arrive':True,'budgets_and_mission_caps_respected':True,'total_flight_excess_ratio_max':.9,
            'mean_length_ratio_max':1.25,'per_scene_allowed_increase_absolute_db_seconds':1.,
            'per_scene_allowed_increase_fraction':.1},
        'truth_boundary':'world used by unchanged simulator only; controller receives current GP/context; evaluation after all runs frozen',
        'scope':'development changes on previously inspected scenes; not unseen validation or LLM benefit'})
    save_json(out/'status.json',{'status':'prepared'})


def check(out):
    protocol=read(out/'protocol.json')
    assert sha(out/'reports/frozen_design.md')==protocol['design_sha256']
    for relative,h in protocol['input_sha256'].items():assert sha(out/relative)==h
    for relative,h in protocol['source_sha256'].items():
        assert sha(ROOT/relative)==h and sha(out/'reports/runtime_source'/relative)==h,relative
    return protocol


def flights(out):
    check(out)
    if any((out/'runs').iterdir()):raise FileExistsError('runs exist')
    from uav_em.simulation import simulate
    for seed in SEEDS:
        scene=f'dev{seed}';cfg=load_config(out/'scenarios'/f'{scene}.json')
        for method in METHODS:
            optimizer=ObjectiveOptimizer(method)
            episode=simulate(cfg,advisor_name=f'{method}_no_model',route_optimizer=optimizer)
            episode['assumptions'].update(route_optimizer=VERSION,same_snapshot_feedback_rounds=False,
                control_objective=method,mission_distance_ratio=optimizer.settings.mission_length_ratio,
                active_probe_enabled=False,current_j_cost_is_diagnostic_for_v2=True)
            for decision in episode['decisions']:
                decision['decision']['reason']='共同候选数值选择；目标版本与实际风险见repair_optimization'
                decision['proposed_decision']['reason']=decision['decision']['reason']
                decision['accepted_route_fixed_objective_meaning']='reported original J, not v2 acceptance criterion'
                decision['accepted_prefix_risk']=decision['repair_optimization']['selected_prefix']
            run=out/'runs'/f'{method}_{scene}_seed{seed}';run.mkdir()
            save_json(run/'episode.json',episode);save_json(run/'metrics.json',episode['metrics'])
            log=render_decision_log(episode).replace('【区段修复：固定目标、共享预算、每轮真实提议/反馈及接受结果】',
                '【共同候选数值控制：原J或短段风险v2；完整候选、选路、全程航程与共享预算】')
            (run/'decision_log.txt').write_text(log)
            save_json(run/'status.json',{'status':'completed','method':method,'scene':scene,'seed':seed,**episode['metrics']})
            print(method,scene,'arrived',episode['metrics']['success'],'length',round(episode['metrics']['executed_length_m'],3),
                  'excess',round(episode['metrics']['interference_excess_integral_db_seconds'],3),flush=True)
    files={str(p.relative_to(out)):sha(p) for p in sorted((out/'runs').rglob('*')) if p.is_file()}
    save_json(out/'run_manifest.json',{'sha256':files,'all_runs_frozen_before_derived_evaluation':True})
    save_json(out/'status.json',{'status':'flights_frozen'})


def verify_runs(out):
    check(out)
    for p,h in read(out/'run_manifest.json')['sha256'].items():assert sha(out/p)==h,p


def evaluate(out):
    verify_runs(out)
    if (out/'reports/summary.json').exists():raise FileExistsError('evaluation exists')
    from uav_em.environment import ElectromagneticWorld
    rows=[];audit=[]
    for path in sorted((out/'runs').glob('*/episode.json')):
        ep=read(path);status=read(path.parent/'status.json');cfg=load_config(out/'scenarios'/f"{status['scene']}.json")
        world=ElectromagneticWorld(cfg);hover=0.;checks=[]
        for d in ep['decisions']:
            r=d['repair_optimization'];resources=r['resources']
            assert resources['samples']<=2100 and resources['edges']<=10000
            assert r['actual_advisor_calls']==r['advisor_failures']==r['policy_errors']==0
            actual_flown=sum(e['distance_m'] for e in ep['events'] if e['event']=='flight' and e['time_seconds']<=d['time_seconds']+1e-8)
            assert abs(actual_flown-r['executed_distance_m'])<1e-6
            chosen=select_candidate(r['candidates'],status['method'])
            assert (chosen['id'] if chosen else None)==r['selected_candidate_id']
            if chosen:
                assert abs(path_length(chosen['path'])-chosen['metrics']['length_m'])<1e-6
                assert np.allclose(chosen['path'][0],d['context']['vehicle']['position_m'],rtol=0,atol=1e-8)
                assert np.allclose(chosen['path'][-1],d['context']['vehicle']['goal_m'],rtol=0,atol=1e-8)
                assert all(world.geometry.edge_free(a,b) for a,b in zip(chosen['path'],chosen['path'][1:]))
                assert actual_flown+chosen['metrics']['length_m']<=r['mission_distance_cap_m']+1e-6
                assert chosen['metrics']['length_m']/cfg.aircraft.speed_mps*cfg.aircraft.flight_power_w/3600+cfg.simulation.decision_delay_seconds*cfg.aircraft.hover_power_w/3600<=d['context']['vehicle']['remaining_energy_wh']-cfg.aircraft.reserve_energy_wh+1e-7
            if d['accepted_action'] is not None:
                duration=min(cfg.simulation.decision_delay_seconds,cfg.simulation.max_seconds-d['time_seconds'])
                point=np.asarray(d['context']['vehicle']['position_m']);count=max(1,int(np.ceil(duration/.25)))
                hover+=sum(max(0,float(world.truth(point,d['time_seconds']+(i+.5)*duration/count))-cfg.electromagnetic.interference_reference_dbm)*duration/count for i in range(count))
            checks.append({'time_seconds':d['time_seconds'],'actual_execution_distance_matches':True,
                'selection_recomputed':True,'endpoints_geometry_energy_mission_cap_valid':bool(chosen),
                'resource_limits_respected':True})
        metrics=ep['metrics'];initial=ep['decisions'][0]['repair_optimization']
        cap=initial['mission_distance_cap_m']
        assert all(d['repair_optimization']['mission_distance_cap_m']==cap for d in ep['decisions'])
        if cap is not None:assert metrics['executed_length_m']<=cap+1e-6
        rows.append({'scene':status['scene'],'method':status['method'],**metrics,
            'mission_reference_length_m':initial['mission_reference_length_m'],'mission_cap_m':cap,
            'decision_hover_excess_db_seconds':hover,
            'flight_plus_decision_hover_excess_db_seconds':metrics['interference_excess_integral_db_seconds']+hover,
            'risk_field_nodes':sum(d['repair_optimization']['risk_field_nodes'] for d in ep['decisions'])})
        audit.append({'run':path.parent.name,'all_decision_checks':checks})
    assert len(rows)==8
    baseline=[r for r in rows if r['method']=='pool_j'];new=[r for r in rows if r['method']=='prefix_v2']
    base_excess=sum(r['interference_excess_integral_db_seconds'] for r in baseline)
    new_excess=sum(r['interference_excess_integral_db_seconds'] for r in new)
    length_ratio=sum(r['executed_length_m'] for r in new)/sum(r['executed_length_m'] for r in baseline)
    pairs=[]
    for seed in SEEDS:
        b=next(r for r in baseline if r['scene']==f'dev{seed}');n=next(r for r in new if r['scene']==f'dev{seed}')
        be=b['interference_excess_integral_db_seconds'];ne=n['interference_excess_integral_db_seconds']
        pairs.append({'scene':b['scene'],'baseline_excess_db_seconds':be,'v2_excess_db_seconds':ne,
            'no_material_regression':ne<=be+max(1.,.1*be)+1e-8})
    components={'all_arrive':all(r['success'] for r in rows),'budgets_and_caps_respected':True,
        'total_flight_excess_reduces_at_least_10percent':new_excess<=.9*base_excess+1e-8,
        'mean_length_ratio_at_most_1_25':length_ratio<=1.25+1e-8,
        'no_scene_material_exposure_regression':all(p['no_material_regression'] for p in pairs)}
    summary={'rows':rows,'pairs':pairs,'gate':{'passes':all(components.values()),'components':components},
        'total_flight_excess_db_seconds':{'pool_j':base_excess,'prefix_v2':new_excess},'mean_length_ratio':length_ratio,
        'total_flight_plus_hover_db_seconds':{m:sum(r['flight_plus_decision_hover_excess_db_seconds'] for r in rows if r['method']==m) for m in METHODS},
        'model_calls':0,'scope':'4 previously inspected development scenes; no independent statistical or LLM benefit claim'}
    save_json(out/'reports/summary.json',summary)
    save_json(out/'reports/audit.json',{'runtime_input_run_hashes_match':True,'all_decisions':audit,'model_calls':0,
        'all_failures_retained':True,'validation_test_unread':True,'no_probe_no_generated_operator':True})
    verify_runs(out);save_json(out/'status.json',{'status':'evaluated','gate_passed':summary['gate']['passes']})


def report(out):
    verify_runs(out);summary=read(out/'reports/summary.json')
    from render_navigation_replays import render_replays
    render_replays(out,frame_stride=6,playback_speed=2.,decision_pause_seconds=.5)
    p=out/'replay/index.html';p.write_text(p.read_text().replace('动画默认按仿真1倍速','本轮动画按仿真2倍速'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for i,method in enumerate(METHODS):
        group=sorted([r for r in summary['rows'] if r['method']==method],key=lambda r:r['scene'])
        x=np.arange(len(SEEDS))+(i-.5)*.35
        axes[0].bar(x,[r['interference_excess_integral_db_seconds'] for r in group],width=.35,label=method)
        axes[1].bar(x,[r['executed_length_m'] for r in group],width=.35,label=method)
    for ax in axes:ax.set_xticks(np.arange(len(SEEDS)),[f'dev{s}' for s in SEEDS]);ax.legend()
    axes[0].set_ylabel('Actual flight excess (dB s)');axes[1].set_ylabel('Actual flight length (m)')
    fig.suptitle('Development numeric controllers; 0 LLM calls; all scenes retained')
    fig.savefig(out/'figures/paired_metrics.png',dpi=140);plt.close(fig)
    md='# 共同目标v2数值闭环\n\n本轮无LLM调用/新生成算子，4城市为已用开发场景。两组使用同一候选生成器/预算，目标不同；没有与历史库对照偷换归因。\n\n'
    md+=f"预声明门槛：{summary['gate']['passes']}；各条件：{summary['gate']['components']}。\n\n"
    md+='|场景|方法|到达|航程/m|飞行积分/dB·s|决策悬停积分/dB·s|边评价|数值耗时/s|\n|---|---|---|---:|---:|---:|---:|---:|\n'
    for r in summary['rows']:
        md+=f"|{r['scene']}|{r['method']}|{r['success']}|{r['executed_length_m']:.3f}|{r['interference_excess_integral_db_seconds']:.3f}|{r['decision_hover_excess_db_seconds']:.3f}|{r['repair_edge_evaluations']}|{r['planner_seconds']:.3f}|\n"
    md+='\n总飞行积分：'+str(summary['total_flight_excess_db_seconds'])+'；平均航程比：'+str(summary['mean_length_ratio'])+'；飞行+决策悬停积分：'+str(summary['total_flight_plus_hover_db_seconds'])+'。\n'
    md+='\n未知区域10dB厌恶系数是研究偏好，不是测得的功率或安全上界。GP未校准且不预测到达场；短段风险代理不能当真实暴露保证。全部指标/分项见reports/summary.json，实验前设计见reports/frozen_design.md。真实计算不推进仿真，不宣称实时部署。\n'
    (out/'RESULTS.md').write_text(md)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>共同目标v2数值验证</title><style>body{font:17px/1.8 sans-serif;max-width:1250px;margin:30px auto;padding:15px}img{max-width:100%}</style><h1>共同目标v2：短段风险与全程绕行上限</h1>'
        '<p>4个已用开发城市、两种人工数值方法、0模型调用；不是LLM收益或未见泛化。场景/传播/GP未改，目标与接受器另立版本；没有主动补测。</p>'
        '<p><a href="replay/index.html">8个真实闭环GIF</a> · <a href="RESULTS.md">全部指标</a> · <a href="reports/audit.json">审计</a> · <a href="reports/frozen_design.md">实验前设计</a> · <a href="../../docs/decision_objective_v2.md">解释与下一步</a></p>'
        f"<p>预声明门槛：{summary['gate']['passes']}；飞行积分合计：{summary['total_flight_excess_db_seconds']}；平均航程比：{summary['mean_length_ratio']:.3f}。</p>"
        '<img src="figures/paired_metrics.png"><p>到达不等于避强有效。未知惩罚是排序偏好，未校准GP不是安全概率；飞行与悬停指标分别报告。</p>')
    save_json(out/'status.json',{'status':'completed','gate_passed':summary['gate']['passes'],'model_calls':0})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','flights','evaluate','report','all'])
    parser.add_argument('--out',type=Path,default=DEFAULT);args=parser.parse_args()
    if args.phase=='all':
        for fn in (prepare,flights,evaluate,report):fn(args.out)
    else:globals()[args.phase](args.out)
if __name__=='__main__':main()
