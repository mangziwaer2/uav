"""公开风险证据与绕行额度：四人工过程、四开发城市，无LLM/补测。"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
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
from uav_em.decision_objective import ObjectiveSettings,select_candidate
from uav_em.evidence_budget import VERSION,METHODS,EvidenceBudgetOptimizer,choose_process
from uav_em.decision_log import render_decision_log
from uav_em.planning import path_length

DEFAULT=ROOT/'outputs/interference_evidence_budget_20261111'
SOURCE=ROOT/'outputs/interference_objective_v2_20261111'
SEEDS=(611,613,617,619)
WORKERS=4
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('scenarios','runs','reports/runtime_source','logs','replay','figures'):(out/folder).mkdir(parents=True)
    shutil.copy2(ROOT/'docs/evidence_budget_process.md',out/'reports/frozen_design.md')
    inputs={};sources={}
    for seed in SEEDS:
        p=SOURCE/'scenarios'/f'dev{seed}.json';dest=out/'scenarios'/p.name
        shutil.copy2(p,dest);inputs[str(dest.relative_to(out))]=sha(dest)
    for p in [*(ROOT/'src/uav_em').glob('*.py'),Path(__file__).resolve(),ROOT/'tools/render_navigation_replays.py',ROOT/'tests/test_evidence_budget.py']:
        relative=p.relative_to(ROOT);dest=out/'reports/runtime_source'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);sources[str(relative)]=sha(p)
    save_json(out/'protocol.json',{'study':VERSION,'methods':list(METHODS),'seeds':list(SEEDS),'workers':WORKERS,
        'settings':asdict(ObjectiveSettings()),'source_study':str(SOURCE.relative_to(ROOT)),
        'input_sha256':inputs,'source_sha256':sources,'design_sha256':sha(out/'reports/frozen_design.md'),
        'validation_test_generated_or_read':False,'model_calls':0,'parameter_search':False,
        'process_parameters':{'evidence_window_seconds':10.,'above_reference':'strictly_greater',
            'maneuver_margin_fraction':.5,'supported_risk_tolerance_db_seconds':1.},
        'gate':{'all_arrive':True,'all_resource_mission_caps_valid':True,'total_flight_excess_ratio_max':.9,
            'mean_length_ratio_max':1.25,'per_scene_increase_max':'max(1 dB s, .1 * pool_j)',
            'zero_baseline_exposure_scene_length_ratio_max':1.05},
        'scope':'16 new real numeric flights on previously inspected development scenes; 0 LLM or generated operators',
        'parallel_timing':'4 independent processes; numeric wall times not deployment/controlled CPU comparisons',
        'truth_boundary':'only unchanged simulator generates actual measurements/execution metrics; controllers receive causal context and snapshot'})
    save_json(out/'status.json',{'status':'prepared'})


def check(out):
    protocol=read(out/'protocol.json')
    assert sha(out/'reports/frozen_design.md')==protocol['design_sha256']
    for relative,h in protocol['input_sha256'].items():assert sha(out/relative)==h
    for relative,h in protocol['source_sha256'].items():
        assert sha(ROOT/relative)==h and sha(out/'reports/runtime_source'/relative)==h,relative
    return protocol


def flight_task(out,seed,method):
    from uav_em.simulation import simulate
    scene=f'dev{seed}';run=out/'runs'/f'{method}_{scene}_seed{seed}';run.mkdir()
    try:
        cfg=load_config(out/'scenarios'/f'{scene}.json');optimizer=EvidenceBudgetOptimizer(method)
        episode=simulate(cfg,advisor_name=f'{method}_numeric_no_model',route_optimizer=optimizer)
        episode['assumptions'].update(route_optimizer=VERSION,same_snapshot_feedback_rounds=False,
            control_method=method,active_probe_enabled=False,mission_distance_ratio=1.25,
            current_j_cost_is_diagnostic_for_process=True,concurrent_study_workers=WORKERS)
        for d in episode['decisions']:
            reason='共同候选数值过程；证据/额度/接受规则见repair_optimization.process_selection'
            d['decision']['reason']=d['proposed_decision']['reason']=reason
            d['accepted_route_fixed_objective_meaning']='original current GP J diagnostic; process rules logged separately'
            d['accepted_prefix_risk']=d['repair_optimization']['selected_prefix']
        save_json(run/'episode.json',episode);save_json(run/'metrics.json',episode['metrics'])
        log=render_decision_log(episode).replace('【区段修复：固定目标、共享预算、每轮真实提议/反馈及接受结果】',
            '【人工数值过程：公开风险证据/剩余绕行额度/支持风险条件及预算】')
        (run/'decision_log.txt').write_text(log)
        save_json(run/'status.json',{'status':'completed','method':method,'scene':scene,'seed':seed,**episode['metrics']})
        return {'method':method,'scene':scene,'success':episode['metrics']['success'],
            'length_m':episode['metrics']['executed_length_m'],'excess_db_seconds':episode['metrics']['interference_excess_integral_db_seconds']}
    except Exception as exc:
        save_json(run/'status.json',{'status':'error','method':method,'scene':scene,'seed':seed,'error':repr(exc)})
        raise


def flights(out):
    protocol=check(out)
    if any((out/'runs').iterdir()):raise FileExistsError('runs exist; will not overwrite')
    with ProcessPoolExecutor(max_workers=protocol['workers']) as executor:
        tasks=[executor.submit(flight_task,out,seed,method) for seed in SEEDS for method in METHODS]
        for future in as_completed(tasks):print(json.dumps(future.result(),ensure_ascii=False),flush=True)
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
    rows=[];audits=[]
    for path in sorted((out/'runs').glob('*/episode.json')):
        ep=read(path);status=read(path.parent/'status.json');cfg=load_config(out/'scenarios'/f"{status['scene']}.json")
        world=ElectromagneticWorld(cfg);hover=0.;checks=[]
        for d in ep['decisions']:
            r=d['repair_optimization'];work=r['resources']
            assert work['samples']<=2100 and work['edges']<=10000
            assert work['total_edges_left']==work['edge_limit']-work['edges']
            assert r['actual_advisor_calls']==r['advisor_failures']==r['policy_errors']==0
            actual_flown=sum(e['distance_m'] for e in ep['events'] if e['event']=='flight' and e['time_seconds']<=d['time_seconds']+1e-8)
            assert abs(actual_flown-r['executed_distance_m'])<1e-6
            chosen,state=choose_process(r['candidates'],status['method'],d['context'],actual_flown,r['mission_distance_cap_m'])
            if r['selection_resource_fallback']:
                state['final_verification_fallback']='preserve_preverified_original_j_with_no_new_work'
                chosen=select_candidate(r['candidates'],'pool_j')
            state['selected_candidate_id']=chosen['id'] if chosen else None
            assert state==r['process_selection'],(path,d['time_seconds'])
            assert (chosen['id'] if chosen else None)==r['selected_candidate_id']
            if chosen:
                assert abs(path_length(chosen['path'])-chosen['metrics']['length_m'])<1e-6
                assert np.allclose(chosen['path'][0],d['context']['vehicle']['position_m'],rtol=0,atol=1e-8)
                assert np.allclose(chosen['path'][-1],d['context']['vehicle']['goal_m'],rtol=0,atol=1e-8)
                assert all(world.geometry.edge_free(a,b) for a,b in zip(chosen['path'],chosen['path'][1:]))
                assert actual_flown+chosen['metrics']['length_m']<=r['mission_distance_cap_m']+1e-6
                energy=chosen['metrics']['length_m']/cfg.aircraft.speed_mps*cfg.aircraft.flight_power_w/3600+cfg.simulation.decision_delay_seconds*cfg.aircraft.hover_power_w/3600
                assert energy<=d['context']['vehicle']['remaining_energy_wh']-cfg.aircraft.reserve_energy_wh+1e-7
            if d['accepted_action'] is not None:
                duration=min(cfg.simulation.decision_delay_seconds,cfg.simulation.max_seconds-d['time_seconds'])
                point=np.asarray(d['context']['vehicle']['position_m']);n=max(1,int(np.ceil(duration/.25)))
                hover+=sum(max(0,float(world.truth(point,d['time_seconds']+(i+.5)*duration/n))-cfg.electromagnetic.interference_reference_dbm)*duration/n for i in range(n))
            checks.append({'time_seconds':d['time_seconds'],'selection_and_state_reproduced':True,
                'causal_evidence_and_actual_distance_checked':True,'geometry_endpoints_energy_cap_valid':bool(chosen),
                'resource_caps_valid':True,'resource_fallback':r['selection_resource_fallback']})
        metrics=ep['metrics'];initial=ep['decisions'][0]['repair_optimization'];cap=initial['mission_distance_cap_m']
        assert all(d['repair_optimization']['mission_distance_cap_m']==cap for d in ep['decisions'])
        if cap is not None:assert metrics['executed_length_m']<=cap+1e-6
        rows.append({'scene':status['scene'],'method':status['method'],**metrics,
            'mission_reference_length_m':initial['mission_reference_length_m'],'mission_cap_m':cap,
            'decision_hover_excess_db_seconds':hover,
            'flight_plus_decision_hover_excess_db_seconds':metrics['interference_excess_integral_db_seconds']+hover,
            'risk_field_nodes':sum(d['repair_optimization']['risk_field_nodes'] for d in ep['decisions']),
            'risk_evidence_active_decisions':sum(d['repair_optimization']['process_selection']['evidence']['active'] for d in ep['decisions']),
            'selection_resource_fallbacks':sum(d['repair_optimization']['selection_resource_fallback'] for d in ep['decisions'])})
        audits.append({'run':path.parent.name,'decisions':checks})
    assert len(rows)==16
    group=lambda method:[r for r in rows if r['method']==method]
    totals={m:{'flight_excess_db_seconds':sum(r['interference_excess_integral_db_seconds'] for r in group(m)),
        'flight_plus_hover_db_seconds':sum(r['flight_plus_decision_hover_excess_db_seconds'] for r in group(m)),
        'mean_length_m':float(np.mean([r['executed_length_m'] for r in group(m)])),
        'total_edges':sum(r['repair_edge_evaluations'] for r in group(m))} for m in METHODS}
    pairs=[]
    for seed in SEEDS:
        baseline=next(r for r in group('pool_j') if r['scene']==f'dev{seed}')
        proposed=next(r for r in group('evidence_budget') if r['scene']==f'dev{seed}')
        base=baseline['interference_excess_integral_db_seconds'];new=proposed['interference_excess_integral_db_seconds']
        pairs.append({'scene':f'dev{seed}','baseline_excess_db_seconds':base,'proposed_excess_db_seconds':new,
            'no_material_exposure_regression':new<=base+max(1.,base*.1)+1e-8,
            'zero_exposure_baseline':base<=1e-8,
            'zero_exposure_scene_length_condition':base>1e-8 or proposed['executed_length_m']<=baseline['executed_length_m']*1.05+1e-8})
    components={'all_16_arrive':all(r['success'] for r in rows),'resource_mission_caps_valid':True,
        'total_flight_excess_reduces_at_least_10percent':totals['evidence_budget']['flight_excess_db_seconds']<=.9*totals['pool_j']['flight_excess_db_seconds']+1e-8,
        'mean_length_ratio_at_most_1_25':totals['evidence_budget']['mean_length_m']<=1.25*totals['pool_j']['mean_length_m']+1e-8,
        'no_scene_material_exposure_regression':all(r['no_material_exposure_regression'] for r in pairs),
        'no_low_risk_unnecessary_detour_over_5percent':all(r['zero_exposure_scene_length_condition'] for r in pairs)}
    save_json(out/'reports/summary.json',{'rows':rows,'totals':totals,'main_pairs':pairs,
        'gate':{'passes':all(components.values()),'components':components},'model_calls':0,
        'scope':'previously inspected development scenes, finite numeric process and bundled ablation; no unseen/LLM claim'})
    save_json(out/'reports/audit.json',{'all_runtime_input_run_hashes_match':True,'model_calls':0,
        'validation_test_unread':True,'all_failures_retained':True,'decisions':audits})
    verify_runs(out);save_json(out/'status.json',{'status':'evaluated','gate_passed':all(components.values())})


def report(out):
    verify_runs(out);summary=read(out/'reports/summary.json')
    from render_navigation_replays import render_replays
    render_replays(out,frame_stride=6,playback_speed=2.,decision_pause_seconds=.5)
    p=out/'replay/index.html';p.write_text(p.read_text().replace('动画默认按仿真1倍速','本轮动画按仿真2倍速'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for i,method in enumerate(METHODS):
        rows=sorted([r for r in summary['rows'] if r['method']==method],key=lambda r:r['scene']);x=np.arange(4)+(i-1.5)*.2
        axes[0].bar(x,[r['interference_excess_integral_db_seconds'] for r in rows],width=.2,label=method)
        axes[1].bar(x,[r['executed_length_m'] for r in rows],width=.2,label=method)
    for ax in axes:ax.set_xticks(np.arange(4),[f'dev{s}' for s in SEEDS]);ax.legend(fontsize=8)
    axes[0].set_ylabel('Actual flight excess (dB s)');axes[1].set_ylabel('Actual flight length (m)')
    fig.suptitle('Development numeric processes; 0 LLM; all 16 real flights retained')
    fig.savefig(out/'figures/comparison.png',dpi=140);plt.close(fig)
    md='# 证据与绕行额度数值过程\n\n4个已看过的开发城市、16真实飞行、0模型/补测/生成算子；非未见验证。\n\n'
    md+=f"预声明完整门槛：{summary['gate']['passes']}；条件：{summary['gate']['components']}。\n\n"
    md+='|场景|方法|到达|航程/m|飞行积分/dB·s|决策悬停/dB·s|边评价|数值墙钟/s|\n|---|---|---|---:|---:|---:|---:|---:|\n'
    for r in sorted(summary['rows'],key=lambda r:(r['scene'],METHODS.index(r['method']))):
        md+=f"|{r['scene']}|{r['method']}|{r['success']}|{r['executed_length_m']:.3f}|{r['interference_excess_integral_db_seconds']:.3f}|{r['decision_hover_excess_db_seconds']:.3f}|{r['repair_edge_evaluations']}|{r['planner_seconds']:.3f}|\n"
    md+='\n|方法|合计飞行积分/dB·s|飞行+悬停/dB·s|平均航程/m|总边评价|\n|---|---:|---:|---:|---:|\n'
    for method,t in summary['totals'].items():md+=f"|{method}|{t['flight_excess_db_seconds']:.3f}|{t['flight_plus_hover_db_seconds']:.3f}|{t['mean_length_m']:.3f}|{t['total_edges']}|\n"
    md+='\n四进程并行运行，墙钟不能作为受控CPU或部署时延比较。当前点测越参考是启发式证据条件，非统计显著性；GP及未知惩罚不是安全保证。evidence_budget相对evidence_only同时增加额度和支持风险条件，本轮不单独识别二者贡献。全部结果保留。\n'
    (out/'RESULTS.md').write_text(md)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>证据与绕行额度过程</title><style>body{font:17px/1.8 sans-serif;max-width:1300px;margin:30px auto;padding:15px}img{max-width:100%}</style><h1>风险证据与剩余绕行额度：人工数值过程</h1>'
        '<p>16个真实闭环、4个已用开发城市、0模型/补测/生成算子；场景/GP/传播/原v2不变，四组共用候选与预算上限。</p>'
        '<p><a href="replay/index.html">16个回放GIF</a> · <a href="RESULTS.md">全部数值</a> · <a href="reports/audit.json">审计</a> · <a href="reports/frozen_design.md">实验前设计</a> · <a href="../../docs/evidence_budget_process.md">结果解释与下一步</a></p>'
        f"<p>完整预声明门槛：{summary['gate']['passes']}。主方法预先指定为evidence_budget，不按结果换组。</p>"
        '<img src="figures/comparison.png"><p>evidence_only区分风险证据条件；evidence_budget进一步加入半余量与支持风险条件，两条件的独立贡献尚未分开。不是LLM推理优化收益或泛化证明。</p>')
    save_json(out/'status.json',{'status':'completed','gate_passed':summary['gate']['passes'],'model_calls':0})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','flights','evaluate','report','all'])
    parser.add_argument('--out',type=Path,default=DEFAULT);args=parser.parse_args()
    if args.phase=='all':
        for fn in (prepare,flights,evaluate,report):fn(args.out)
    else:globals()[args.phase](args.out)
if __name__=='__main__':main()
