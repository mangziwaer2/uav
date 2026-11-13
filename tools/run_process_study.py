"""有限修复过程的数值基线：同快照配对、事后评价与完整闭环分开留证。"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json
from uav_em.config import load_config
from uav_em.repair_process import ProcessRepairOptimizer,baseline_process
from uav_em.route_repair import RepairSettings
from run_repair_diagnostics import load_case

SOURCE=ROOT/'outputs/interference_repair_diagnostics_20261110'
DEFAULT=ROOT/'outputs/interference_process_baselines_20261110'
METHODS=('library_numeric','fixed_process','adaptive_process')
SETTINGS=RepairSettings()

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('cases','scenarios','programs','paired','evaluations','runs','reports/runtime_source','logs','replay','figures'):
        (out/folder).mkdir(parents=True)
    inputs={}
    for path in [*(SOURCE/'cases').glob('*'),*(SOURCE/'scenarios').glob('*.json')]:
        relative=Path(path.parent.name)/path.name;shutil.copy2(path,out/relative);inputs[str(relative)]=sha(out/relative)
    if len(list((out/'cases').glob('*.npz')))!=4:raise ValueError('exactly four causal snapshots required')
    for name,adaptive in [('fixed_process',False),('adaptive_process',True)]:
        save_json(out/'programs'/f'{name}.json',baseline_process(adaptive))
    shutil.copy2(ROOT/'docs/repair_process_v1.md',out/'reports/frozen_contract.md')
    sources={}
    paths=[*(ROOT/'src/uav_em').glob('*.py'),Path(__file__).resolve(),ROOT/'tools/run_repair_diagnostics.py',
           ROOT/'tools/render_navigation_replays.py',ROOT/'tests/test_repair_process.py']
    for path in paths:
        relative=path.relative_to(ROOT);target=out/'reports/runtime_source'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);sources[str(relative)]=sha(path)
    save_json(out/'protocol.json',{'study':'finite_repair_process_numeric_baselines_development_pilot',
        'methods':list(METHODS),'settings':SETTINGS.__dict__,'max_process_actions':6,
        'cases':sorted(p.stem for p in (out/'cases').glob('*.json')),'seed_offsets':[0,101,202],
        'closed_loop_scenes':['s01','s02'],'closed_loop_seeds':'unaltered copied scenario seeds311/313',
        'paired_evaluation':'freeze ALL paired paths before evaluating fixed-route future exposure, no new observations',
        'search_objective':'unchanged current GP balanced J; not future exposure',
        'constraints':'same known geometry, endpoints, energy; same GP;2100 samples/10000 edges across entire process',
        'task_improvement_gate':{'all_tasks_success':True,'both_mean_length_and_mean_excess_nonincreasing':True,
                                 'at_least_one_mean_reduction_fraction':.05},
        'information':'same public scalar measurements/GP summaries for selectors; full current GP only in numeric search/verification',
        'scope':'all researcher-authored numeric rules,0 model calls, no new code generation/GP/objective/probe/tree reuse',
        'development_data':'all existing four snapshots and both scenes; previously examined, not held out',
        'prospective_split':{'new_city_dev_seeds':[611,613,617,619],'new_city_validation_seeds':[701,709],
            'new_city_test_seeds':[1601,1607,1609,1613,1619,1621],
            'status':'reserved IDs only, not generated/read this stage; generator and source distribution must freeze before stage3'},
        'source_sha256':sources,'input_sha256':inputs,
        'program_sha256':{p.name:sha(p) for p in (out/'programs').glob('*.json')},
        'contract_sha256':sha(out/'reports/frozen_contract.md')})
    save_json(out/'status.json',{'status':'prepared'})

def check(out):
    p=read(out/'protocol.json')
    for relative,h in p['source_sha256'].items():
        if sha(ROOT/relative)!=h or sha(out/'reports/runtime_source'/relative)!=h:raise ValueError('source changed: '+relative)
    for relative,h in p['input_sha256'].items():
        if sha(out/relative)!=h:raise ValueError('input changed')
    for name,h in p['program_sha256'].items():
        if sha(out/'programs'/name)!=h:raise ValueError('program changed')
    if sha(out/'reports/frozen_contract.md')!=p['contract_sha256']:raise ValueError('frozen contract changed')
    return p

def optimizer(out,method):
    program=read(out/'programs'/f'{method}.json') if method!='library_numeric' else None
    return ProcessRepairOptimizer(method,program,SETTINGS)

def paired(out):
    protocol=check(out)
    if any((out/'paired').iterdir()):raise FileExistsError('refuse to overwrite paired tasks')
    for name in protocol['cases']:
        case,snapshot,_=load_case(out,out/'cases'/f'{name}.json')
        for offset in protocol['seed_offsets']:
            for method in METHODS:
                plan,report=optimizer(out,method).optimize(snapshot,case['context'],case['route'],case['max_length_m'],case['seed']+offset,0)
                filename=f'{name}_offset{offset}_{method}.json'
                save_json(out/'paired'/filename,{'case':name,'seed_offset':offset,'method':method,'path':plan.path,'report':report})
                print('paired',filename,'J',report['initial_metrics']['objective'],'->',report['best_metrics']['objective'],
                      'edges',report['resources']['edges'],'errors',report['policy_errors'],flush=True)
    save_json(out/'paired_manifest.json',{'all_solver_results_frozen_before_evaluation':True,
        'sha256':{p.name:sha(p) for p in sorted((out/'paired').glob('*.json'))}})
    save_json(out/'status.json',{'status':'paired_frozen'})

def verify_pairs(out):
    p=check(out);m=read(out/'paired_manifest.json');expected=len(p['cases'])*len(p['seed_offsets'])*len(METHODS)
    if len(m['sha256'])!=expected:raise ValueError('incomplete paired results')
    for name,h in m['sha256'].items():
        if sha(out/'paired'/name)!=h:raise ValueError('paired result changed')
    return m

def evaluate(out):
    verify_pairs(out)
    if any((out/'evaluations').iterdir()):raise FileExistsError('refuse to overwrite evaluations')
    from uav_em.environment import ElectromagneticWorld
    from uav_em.diagnostic_evaluation import evaluate_fixed_route
    for case_path in sorted((out/'cases').glob('*.json')):
        case,snapshot,cfg=load_case(out,case_path);world=ElectromagneticWorld(cfg)
        for path in sorted((out/'paired').glob(case['case']+'_offset*.json')):
            row=read(path)
            e=evaluate_fixed_route(world,snapshot,row['path']) if row['path'] else None
            save_json(out/'evaluations'/path.name,{'source_sha256':sha(path),'fixed_route_evaluation':e})
    verify_pairs(out);save_json(out/'status.json',{'status':'paired_evaluated'})

def flights(out):
    protocol=check(out)
    if any((out/'runs').iterdir()):raise FileExistsError('refuse to overwrite flights')
    from uav_em.simulation import simulate
    from uav_em.decision_log import render_decision_log
    for scene in protocol['closed_loop_scenes']:
        cfg=load_config(out/'scenarios'/f'{scene}.json')
        for method in METHODS:
            directory=out/'runs'/f'{method}_{scene}_seed{cfg.seed}';directory.mkdir()
            episode=simulate(cfg,advisor_name=method+'_no_model',route_optimizer=optimizer(out,method))
            # The unchanged simulator's generic route-optimizer label is refined by the actual implementation.
            episode['assumptions']['route_optimizer']='finite_numeric_process_v1'
            episode['assumptions']['maximum_process_actions']=6
            save_json(directory/'episode.json',episode);save_json(directory/'metrics.json',episode['metrics'])
            (directory/'decision_log.txt').write_text(render_decision_log(episode))
            save_json(directory/'status.json',{'status':'completed','method':method,'scene':scene,'seed':cfg.seed,**episode['metrics']})
            print('flight',scene,method,json.dumps(episode['metrics'],ensure_ascii=False),flush=True)
    save_json(out/'status.json',{'status':'flights_completed'})

def task_gate(rows,method,baseline='library_numeric'):
    a=[r['metrics'] for r in rows if r['method']==method];b=[r['metrics'] for r in rows if r['method']==baseline]
    if not a or len(a)!=len(b):raise ValueError('incomplete task comparison')
    fields=('executed_length_m','interference_excess_integral_db_seconds')
    ratios={}
    for key in fields:
        current=float(np.mean([m[key] for m in a]));reference=float(np.mean([m[key] for m in b]))
        ratios[key]=current/reference if reference>1e-12 else (1. if current<=1e-12 else None)
    success=all(m['success'] for m in a+b)
    passed=success and all(v is not None and v<=1.+1e-8 for v in ratios.values()) and min(ratios.values())<=.95
    return {'method':method,'baseline':baseline,'all_success':success,'mean_ratios':ratios,'passes_development_task_gate':passed,
            'not_statistical_or_hardware_safety_claim':True}

def report(out):
    protocol=check(out);verify_pairs(out)
    from uav_em.repair_search import FixedObjective,WorkBudget
    from uav_em.environment import ElectromagneticWorld
    from uav_em.diagnostic_evaluation import evaluate_fixed_route
    from render_navigation_replays import render_replays
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pairs=[];audits=[]
    for name in protocol['cases']:
        case,snapshot,cfg=load_case(out,out/'cases'/f'{name}.json')
        problem=FixedObjective(snapshot,-80,WorkBudget(2100,100000));first_states=[]
        for offset in protocol['seed_offsets']:
            states=[]
            for method in METHODS:
                path=out/'paired'/f'{name}_offset{offset}_{method}.json';row=read(path);r=row['report']
                evaluation=read(out/'evaluations'/path.name)
                if evaluation['source_sha256']!=sha(path):raise ValueError('evaluation source mismatch')
                metrics=problem.route_metrics(row['path']);valid=bool(metrics and metrics['length_m']<=case['max_length_m']+1e-8)
                error=max(abs(metrics[k]-r['best_metrics'][k]) for k in metrics) if metrics else None
                if not valid or error>1e-7 or r['policy_errors'] or r['resources']['edges']>10000 or r['resources']['samples']>2100:
                    raise ValueError('paired audit failed: '+path.name)
                states.append(r['process_trace'][0]['state'])
                pairs.append({'case':name,'seed_offset':offset,'method':method,'initial_j':r['initial_metrics']['objective'],
                    'final_j':r['best_metrics']['objective'],'length_m':metrics['length_m'],'edges':r['resources']['edges'],
                    'samples':r['resources']['samples'],'actions':len(r['process_trace']),'numeric_compute_seconds':r['numeric_compute_seconds'],
                    'future_excess_db_seconds':evaluation['fixed_route_evaluation']['future_excess_db_seconds']})
                audits.append({'task':path.stem,'geometry_energy_ok':valid,'metric_recompute_error':error,
                    'no_model_calls':r['actual_advisor_calls']==0,'budgets_ok':True,'policy_errors':r['policy_errors']})
            if any(state!=states[0] for state in states):raise ValueError('initial public states differ')
        # Plot one predeclared seed (offset0), no best-run selection.
        selected=[p for p in pairs if p['case']==name and p['seed_offset']==0]
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for index,method in enumerate(METHODS):
            group=[p for p in pairs if p['case']==name and p['method']==method]
            axes[0].scatter([p['edges'] for p in group],[p['final_j'] for p in group],label=method)
            row=read(out/'paired'/f'{name}_offset0_{method}.json');route=np.array(row['path'])
            axes[1].plot(*route.T,label=method,ls=('-','--',':')[index])
        for building in snapshot.geometry.blocking_buildings:
            from matplotlib.patches import Polygon
            axes[1].add_patch(Polygon(building.footprint_m,color='#888888',alpha=.5))
        axes[0].set(xlabel='Total edge evaluations',ylabel='Final current GP J',title='Three fixed seeds per method')
        axes[1].set(xlim=(0,snapshot.geometry.high[0]),ylim=(0,snapshot.geometry.high[1]),aspect='equal',title='Offset0 retained routes (no new observations)')
        for ax in axes:ax.legend(fontsize=7)
        fig.suptitle(name);fig.savefig(out/'figures'/f'{name}.png',dpi=130);plt.close(fig)
    flights_rows=[];flight_audit=[]
    for scene in protocol['closed_loop_scenes']:
        for method in METHODS:
            cfg=load_config(out/'scenarios'/f'{scene}.json');directory=out/'runs'/f'{method}_{scene}_seed{cfg.seed}'
            ep=read(directory/'episode.json');m=ep['metrics'];reports=[d['repair_optimization'] for d in ep['decisions']]
            if m['advisor_calls']!=0 or any(r['policy_errors'] or r['resources']['edges']>10000 or r['resources']['samples']>2100 for r in reports):
                raise ValueError('flight budget/policy audit failed')
            flights_rows.append({'scene':scene,'method':method,'metrics':m,'episode':str((directory/'episode.json').relative_to(out))})
            flight_audit.append({'scene':scene,'method':method,'sessions':len(reports),'all_budgets_ok':True,'all_policy_errors_zero':True,
                                 'model_calls':m['advisor_calls'],'success':m['success']})
    gates=[task_gate(flights_rows,m) for m in METHODS[1:]]
    save_json(out/'reports/summary.json',{'paired':pairs,'closed_loop':flights_rows,'development_task_gates':gates})
    save_json(out/'reports/audit.json',{'frozen_sources_inputs_programs_contract_match':True,'paired_manifest_match':True,
        'same_initial_public_state_all_triplets':True,'paired':audits,'flights':flight_audit})
    render_replays(out,frame_stride=5,playback_speed=2.,decision_pause_seconds=.5)
    (out/'reports/example_process.json').write_text((out/'paired'/'s02_after30_offset0_adaptive_process.json').read_text())
    md='# 有限修复过程：数值基线开发试验\n\n36配对求解、6闭环飞行，0模型调用。固定GP/J和每决策2100样本/10000边；三组最多6步。\n\n'
    md+='配对最终路线的未来积分是无新测量的事后开环评价；闭环指标另列。全部是已有开发场景，不是独立测试或LLM收益。\n\n'
    md+='|快照|方法|最终J均值|边评价均值|未来积分均值/dB·s|\n|---|---|---:|---:|---:|\n'
    for name in protocol['cases']:
        for method in METHODS:
            group=[p for p in pairs if p['case']==name and p['method']==method]
            md+=f"|{name}|{method}|{np.mean([p['final_j'] for p in group]):.3f}|{np.mean([p['edges'] for p in group]):.1f}|{np.mean([p['future_excess_db_seconds'] for p in group]):.3f}|\n"
    md+='\n|场景|方法|到达|航程/m|暴露/s|积分/dB·s|边评价|数值计算/s|\n|---|---|---|---:|---:|---:|---:|---:|\n'
    for row in flights_rows:
        m=row['metrics'];md+=f"|{row['scene']}|{row['method']}|{m['success']}|{m['executed_length_m']:.3f}|{m['high_interference_exposure_seconds']:.3f}|{m['interference_excess_integral_db_seconds']:.3f}|{m['repair_edge_evaluations']}|{m['planner_seconds']:.3f}|\n"
    md+='\n开发任务收益门槛（相对library_numeric）：\n\n'
    for gate in gates:md+=f"- {gate['method']}：{gate['passes_development_task_gate']}，均值比值{gate['mean_ratios']}。\n"
    md+='\n门槛只作开发验收，不是统计显著性或硬件安全保证；不通过就不能称已改善任务。\n'
    (out/'RESULTS.md').write_text(md)
    body=''.join(f'<h2>{name}</h2><img src="figures/{name}.png" width="100%">' for name in protocol['cases'])
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>有限修复过程数值基线</title><style>body{font:17px/1.8 sans-serif;max-width:1200px;margin:30px auto}img{max-width:100%}</style>'
        '<h1>有限修复过程：合同、反馈分流与数值对照</h1><p>0模型调用；36配对求解、6完整闭环。固定GP/J与共享预算，当前是开发基线，不是离线生成或在线LLM方法收益。</p>'
        '<p><a href="RESULTS.md">全部指标表</a> · <a href="replay/index.html">6个闭环GIF</a> · '
        '<a href="reports/audit.json">同信息/预算审计</a> · <a href="reports/example_process.json">实际过程执行记录</a> · '
        '<a href="reports/frozen_contract.md">实验前冻结合同</a> · <a href="../../docs/repair_process_v1.md">方法与结果解释</a></p>'
        '<p>任务收益门槛：'+html.escape(json.dumps(gates,ensure_ascii=False))+'</p>'+body)
    save_json(out/'status.json',{'status':'completed_numeric_baseline_study','model_calls':0,'paired_tasks':36,'closed_loop_tasks':6})

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','paired','evaluate','flights','report','all'])
    parser.add_argument('--out',type=Path,default=DEFAULT);args=parser.parse_args()
    if args.phase=='all':
        for function in (prepare,paired,evaluate,flights,report):function(args.out)
    else:globals()[args.phase](args.out)
if __name__=='__main__':main()
