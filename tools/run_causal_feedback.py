"""因果可靠性反馈 + 在线有限算子选择的16项同快照配对试验。"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from uav_em.agent_navigation import save_json
from uav_em.balanced_objective import choose, paired_seed, power_time_score, PowerTimeSettings
from uav_em.causal_feedback import FileFeedbackPolicy, MODES, VERSION, prequential_evidence, run_selection, validate_response
from uav_em.repair_process import canonical_path
from uav_em.repair_search import FixedObjective, WorkBudget
from run_repair_diagnostics import load_case

SOURCE = ROOT/'outputs/interference_risk_diagnostics_20261111'
POOL_SOURCE = ROOT/'outputs/interference_balanced_objective_20261111'
DEFAULT = ROOT/'outputs/interference_causal_feedback_20261111'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

CONTRACT = '''You are a decision LLM in one independent frozen-state route-repair task.
Read ONLY this task's public contract.md, control.json, requests/*.json and your own responses/*.json.
Do not read any parent directory, source, repository instructions/handoff, other experiments/tasks,
scene configurations, episode, private snapshots, true field, hidden sources, or future data. Do not
spawn agents, access network, or run your own planning/numerical scripts. The shared filesystem
allowlist is procedural, not OS isolation. Use file read/write tools only for the above workflow.
Use only this task's current request, feedback and same-task memory. Poll <=20s at a time if needed.
Remain until control status is completed/failed, or the selection limit has been reached.

The UAV knows building geometry and self-location and receives noisy scalar interference power
ONLY at its current location. GP extrapolations are uncertain estimates; grey/unsupported is NOT
known low power. No bearing/source/true side is given. Left/right below are geometric orientations
relative to current route, not measured source directions. Along-track changes may be spatial or
temporal. Historical prequential checks predict each reading BEFORE incorporating it; these are
correlated sparse checks, not calibrated risk confidence. Straight flight cannot identify lateral field.

Task utility is flight time + received power integral normalized at -80dBm, weight 1. The common
acceptor C uses whole remaining CURRENT GP field plus a declared prior for unknown time; this is
not the true field at arrival. -80 is a power unit, NOT a device safety threshold. Old J in route segment
metrics is a search diagnostic only. Never claim a smaller proxy C proves smaller actual exposure.
You CANNOT modify GP, weights, thresholds, energy/distance limits, or the common acceptor.

Choose one profile or stop. The numerical layer computes parameters from the same public evidence:
global_uniform: whole-route RRT*, uniform sampling.
global_focus: whole-route RRT*, mixture of global and wide route-corridor sampling.
local_pad1: RRT* reconnect around the supported-interference-selected segment, padding 1.
local_pad3: same with padding 3 and local corridor sampling.
via_left_near / via_right_near: same selected segment/padding 1, near geometric left/right waypoint,
two RRT* legs sharing the same quota; a waypoint is NOT a measured-safe point.
stop: retain current best route; no further search.
Current route anchor IDs contain only the remaining route. Past measurement locations are not repair
anchors. Uniform sampling always retains its existing minimum global mixture. No arbitrary code.

At most 2 selections in this task; each selected search <=700 proposals and shared mission-decision
ledger <=2100 samples/10000 edges INCLUDING the already-paid common pool. All feasible candidates
remain available; fixed acceptor chooses global lowest C, then length. An invalid response retains the
best route and ends this process, logs an error, and is not repaired into model success. Duplicate
profile consumes a selection and produces feedback, without another search. Final geometry/energy/
fixed lifetime flown+remaining<=1.25D0 verification is compulsory. No new measurement during this task.
Your role is to decide which available computation is justified by evidence and solver feedback;
stopping is allowed. You cannot force a route the fixed acceptor rejects.

Write EXACTLY one JSON object atomically to the response_path in each request, with no Markdown:
{"request_id":"select_01","profile":"one of the seven names","reason":"short evidence-based reason <=600 chars",
 "memory":"same-task memory <=1200 chars"}
Use the actual request_id. Do not output numeric search parameters or a generated program. Return
a brief final acknowledgement when control says completed, not a proposal to work later.
'''


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('cases', 'scenarios', 'pool_reports', 'reliability', 'results', 'evaluations',
                   'reports/runtime_source', 'logs', 'figures', 'replay'):
        (out/folder).mkdir(parents=True)
    case_paths = sorted((SOURCE/'cases').glob('*.json'))
    assert len(case_paths) == 4
    tasks, inputs = {}, {}
    episode_path = POOL_SOURCE/'runs/balanced_pool_dev617_seed617/episode.json'
    episode = read(episode_path)
    for i, path in enumerate(case_paths, 1):
        task = f'task{i:02d}'
        case = read(path)
        tasks[task] = case['case']
        for src, dest in [(path, out/'cases'/path.name), (path.with_suffix('.npz'), out/'cases'/path.with_suffix('.npz').name)]:
            shutil.copy2(src, dest)
            inputs[str(dest.relative_to(out))] = sha(dest)
        report = episode['decisions'][case['source_decision_index']]['repair_optimization']
        assert report['process_program'] is None and report['snapshot_time_seconds'] == case['time_seconds']
        assert report['candidates'] == [{k: v for k, v in r.items() if k != 'family'} for r in case['original_candidates']]
        dest = out/'pool_reports'/f'{task}.json'
        save_json(dest, report)
        inputs[str(dest.relative_to(out))] = sha(dest)
        public = out/'public'/task
        (public/'requests').mkdir(parents=True)
        (public/'responses').mkdir()
        (public/'contract.md').write_text(CONTRACT)
        save_json(public/'control.json', {'status': 'prepared', 'selection_limit': 2})
    for src, dest in [(SOURCE/'scenarios/dev617.json', out/'scenarios/dev617.json'),
                      (ROOT/'docs/causal_feedback_v1.md', out/'reports/frozen_design.md')]:
        shutil.copy2(src, dest)
        inputs[str(dest.relative_to(out))] = sha(dest)
    sources = {}
    for path in [*(ROOT/'src/uav_em').glob('*.py'), Path(__file__).resolve(),
                 ROOT/'tools/run_repair_diagnostics.py', ROOT/'tests/test_causal_feedback.py']:
        relative = path.relative_to(ROOT)
        dest = out/'reports/runtime_source'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        sources[str(relative)] = sha(path)
    save_json(out/'protocol.json', {'study': VERSION, 'tasks': tasks, 'modes': list(MODES),
        'input_sha256': inputs, 'source_sha256': sources,
        'public_contract_sha256': {t: sha(out/'public'/t/'contract.md') for t in tasks},
        'source_episode_sha256': sha(episode_path), 'same_original_pool_only': True,
        'source_extended_diagnostic_pool_used': False, 'new_flights': 0, 'new_generated_submissions': 0,
        'max_independent_model_sessions': 4, 'max_actual_model_responses': 8,
        'choices_per_task': 2, 'samples_per_search': 700, 'shared_samples': 2100, 'shared_edges': 10000,
        'weight': 1, 'GP_acceptor_sensor_changes': False, 'validation_test_read_or_generated': False,
        'numeric_stop_rule': 'after local_pad3 stop if current C gain >=1%; otherwise global_focus',
        'numeric_fixed_rule': 'local_pad3 then global_focus', 'outcome_screening_or_retry': False})
    save_json(out/'status.json', {'status': 'prepared'})


def check(out):
    p = read(out/'protocol.json')
    for relative, expected in p['input_sha256'].items():
        if sha(out/relative) != expected:
            raise ValueError('frozen input changed: '+relative)
    for relative, expected in p['source_sha256'].items():
        if sha(ROOT/relative) != expected or sha(out/'reports/runtime_source'/relative) != expected:
            raise ValueError('source changed: '+relative)
    for task, expected in p['public_contract_sha256'].items():
        if sha(out/'public'/task/'contract.md') != expected:
            raise ValueError('public contract changed')
    return p


def task_inputs(out, task):
    p = check(out)
    case, snapshot, cfg = load_case(out, out/'cases'/f"{p['tasks'][task]}.json")
    reliability = prequential_evidence(cfg, snapshot.geometry, case['received_point_measurements'], case['time_seconds'])
    return case, snapshot, read(out/'pool_reports'/f'{task}.json'), reliability


def numeric(out):
    p = check(out)
    if any((out/'results').glob('*_pool_only.json')):
        raise FileExistsError('numeric tasks already started')
    for task in p['tasks']:
        case, snapshot, pool, reliability = task_inputs(out, task)
        save_json(out/'reliability'/f'{task}.json', reliability)
        for mode in MODES[:-1]:
            path = out/'results'/f'{task}_{mode}.json'
            if path.exists():
                raise FileExistsError('result already exists')
            result = run_selection(snapshot, case, pool, reliability, mode)
            save_json(path, {'task': task, 'case': case['case'], **result})
            print(task, mode, 'C', result['selected_balance']['score_seconds'], 'edges', result['resources']['edges'], flush=True)


def model(out, task):
    case, snapshot, pool, reliability = task_inputs(out, task)
    public = out/'public'/task
    dest = out/'results'/f'{task}_llm_feedback.json'
    if dest.exists() or any((public/'requests').iterdir()):
        raise FileExistsError('model task already started')
    save_json(public/'control.json', {'status': 'running', 'selection_limit': 2})
    try:
        result = run_selection(snapshot, case, pool, reliability, 'llm_feedback', FileFeedbackPolicy(public))
        save_json(dest, {'task': task, 'case': case['case'], **result})
        save_json(public/'control.json', {'status': 'completed', 'model_calls': result['model_calls'],
                                         'model_errors': result['model_errors']})
        print(task, 'llm_feedback', 'calls', result['model_calls'], 'errors', result['model_errors'], flush=True)
    except Exception as exc:
        save_json(public/'control.json', {'status': 'failed', 'error': repr(exc)})
        raise


def freeze(out):
    p = check(out)
    if (out/'causal_manifest.json').exists():
        raise FileExistsError('causal outputs already frozen')
    for task in p['tasks']:
        for mode in MODES:
            assert (out/'results'/f'{task}_{mode}.json').exists()
    assert len(list((out/'results').glob('*.json'))) == 16
    files = {str(path.relative_to(out)): sha(path) for folder in ('results', 'reliability', 'public')
             for path in sorted((out/folder).rglob('*')) if path.is_file()}
    assert sum(read(out/'results'/f'{t}_llm_feedback.json')['model_calls'] for t in p['tasks']) <= 8
    save_json(out/'causal_manifest.json', {'sha256': files, 'all_results_and_choices_before_truth_evaluation': True})
    save_json(out/'status.json', {'status': 'all_causal_results_frozen'})


def verify_causal(out):
    check(out)
    manifest = read(out/'causal_manifest.json')
    for relative, expected in manifest['sha256'].items():
        if sha(out/relative) != expected:
            raise ValueError('causal output changed: '+relative)


def evaluate(out):
    verify_causal(out)
    if any((out/'evaluations').iterdir()):
        raise FileExistsError('evaluations already exist')
    from uav_em.environment import ElectromagneticWorld
    from uav_em.risk_diagnostic_evaluation import evaluate_route
    for path in sorted((out/'results').glob('*.json')):
        result = read(path)
        case, snapshot, cfg = load_case(out, out/'cases'/f"{result['case']}.json")
        evaluation = evaluate_route(ElectromagneticWorld(cfg), snapshot, result['path'])
        save_json(out/'evaluations'/path.name, {'task': result['task'], 'mode': result['mode'],
            'evaluation': evaluation, 'result_sha256': sha(path)})
        print(path.stem, 'T+Q', evaluation['future_joint_seconds'], 'excess', evaluation['future_excess_db_seconds'], flush=True)


def render_task(out, task, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from uav_em.environment import ElectromagneticWorld
    from uav_em.decision_objective import cut_route
    from uav_em.visualization import save_gif
    case, snapshot, cfg = load_case(out, out/'cases'/f'{name}.json')
    routes = {m: read(out/'results'/f'{task}_{m}.json')['path'] for m in MODES}
    colors = ('#111111', '#2485c0', '#4d9b50', '#b32d9d')
    world = ElectromagneticWorld(cfg)
    duration = max(sum(np.linalg.norm(np.asarray(b)-a) for a, b in zip(r,r[1:]))/cfg.aircraft.speed_mps for r in routes.values())
    offsets = np.linspace(0, duration, 20)
    fields = [world.truth(snapshot.geometry.grid(), case['time_seconds']+.5+t) for t in offsets]
    high = snapshot.geometry.high
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout='constrained')
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('#dddddd')
    image = axes[0].imshow(fields[0], origin='lower', extent=[0,high[0],0,high[1]], vmin=-100,vmax=-60,cmap=cmap)
    axes[1].imshow(np.ma.masked_where(~snapshot.supported, snapshot.estimate_dbm), origin='lower',
                   extent=[0,high[0],0,high[1]], vmin=-100,vmax=-60,cmap=cmap)
    fig.colorbar(image, ax=list(axes), label='Interference power (dBm)', shrink=.7)
    markers=[]
    for ax in axes:
        for building in snapshot.geometry.buildings:
            ax.add_patch(Polygon(building.footprint_m, facecolor='#444444' if building in snapshot.geometry.blocking_buildings else '#aaaaaa', alpha=.55))
        group=[]
        for i,(m,c) in enumerate(zip(MODES,colors)):
            r=np.asarray(routes[m]);ax.plot(*r.T,color=c,lw=2,ls=('-','--',':','-.')[i],label=m)
            group.append(ax.plot(*r[0],marker='o',ms=7,color=c)[0])
        markers.append(group)
        ax.set(xlim=(0,high[0]),ylim=(0,high[1]),aspect='equal',xlabel='x (m)',ylabel='y (m)')
        ax.legend(fontsize=7,loc='upper right')
    axes[0].set_title('Truth: evaluation only; no oracle route')
    axes[1].set_title('Frozen GP; grey=unknown; no new observations')
    title=fig.suptitle('')
    def update(i):
        image.set_data(fields[i])
        for group in markers:
            for marker,m in zip(group,MODES):
                front,_=cut_route(routes[m],float(offsets[i])*cfg.aircraft.speed_mps)
                marker.set_data([front[-1][0]],[front[-1][1]])
        title.set_text(f'{task} ({name}) t={case["time_seconds"]+.5+offsets[i]:.1f}s | selected routes, OPEN LOOP\nNo new sensing/replanning during replay; magenta = real LLM selected computation')
    update(0)
    fig.savefig(out/'figures'/f'{task}.png',dpi=115)
    save_gif(fig,update,list(range(20)),[160]*19+[1600],out/'replay'/f'{task}.gif')
    plt.close(fig)


def report(out):
    verify_causal(out)
    p=check(out)
    rows,audits,comparisons=[],[],[]
    for task,name in p['tasks'].items():
        case,snapshot,cfg=load_case(out,out/'cases'/f'{name}.json')
        initial_states=[]
        task_rows={}
        for mode in MODES:
            path=out/'results'/f'{task}_{mode}.json';r=read(path);e=read(out/'evaluations'/path.name)
            assert e['result_sha256']==sha(path)
            assert r['resources']['samples']<=2100 and r['resources']['edges']<=10000
            assert r['new_measurements']==r['fallback_searches']==0
            initial_states.append(r['trace'][0]['state'])
            chosen=choose(r['candidates'])
            assert chosen['id']==r['selected_id'] and chosen['balance']==r['selected_balance']
            assert canonical_path(chosen['path'])==canonical_path(r['path'])
            problem=FixedObjective(snapshot,-80,WorkBudget(2100,100000))
            metrics=problem.route_metrics(r['path']);assert metrics is not None
            assert metrics['length_m']<=case['max_length_m']+1e-7
            score=power_time_score(problem,r['path'],cfg.aircraft.speed_mps,PowerTimeSettings(**case['score_settings']))
            assert abs(score['score_seconds']-r['selected_balance']['score_seconds'])<1e-7
            for search in r['searches']:
                assert search['actual_seed']==paired_seed(case['seed'],case['time_seconds'],VERSION,search['seed_purpose'])
            if mode=='llm_feedback':
                requests=sorted((out/'public'/task/'requests').glob('*.json'))
                responses=sorted((out/'public'/task/'responses').glob('*.json'))
                assert len(requests)==r['model_calls'] and len(responses)<=len(requests)
                for trace in r['trace']:
                    call=trace['model_exchange']
                    if not call:continue
                    req=out/'public'/task/'requests'/f"{call['request_id']}.json"
                    assert sha(req)==call['request_sha256']
                    if call['status']=='accepted':
                        raw=(out/'public'/task/'responses'/f"{call['request_id']}.json").read_text()
                        assert raw==call['raw_response'] and validate_response(raw,call['request_id'])==call['parsed_response']
                        assert call['parsed_response']['profile']==trace['selected_profile']
            row={'task':task,'case':name,'method':mode,'current_C_seconds':score['score_seconds'],
                'length_m':metrics['length_m'],**e['evaluation'],'samples':r['resources']['samples'],
                'edges':r['resources']['edges'],'additional_samples':r['resources']['samples']-r['starting_resources_already_paid']['samples'],
                'additional_edges':r['resources']['edges']-r['starting_resources_already_paid']['edges'],
                'profiles':[t['selected_profile'] for t in r['trace']], 'model_calls':r['model_calls'],
                'model_errors':r['model_errors'],'numeric_errors':r['numeric_errors'],
                'model_wall_seconds':r['model_wall_seconds'],'numeric_wall_seconds':r['numeric_wall_seconds']}
            rows.append(row);task_rows[mode]=row
        assert all(s==initial_states[0] for s in initial_states)
        a,b=task_rows['numeric_feedback'],task_rows['llm_feedback']
        comparisons.append({'task':task,'case':name,
            'LLM_minus_numeric_current_C_seconds':b['current_C_seconds']-a['current_C_seconds'],
            'LLM_minus_numeric_true_joint_seconds':b['future_joint_seconds']-a['future_joint_seconds'],
            'LLM_minus_numeric_excess_db_seconds':b['future_excess_db_seconds']-a['future_excess_db_seconds'],
            'LLM_minus_numeric_edges':b['edges']-a['edges'],
            'description_only_not_significance_or_closed_loop_benefit':True})
        audits.append({'task':task,'common_initial_public_state_identical':True,'caps_endpoints_scores_and_seeds_verified':True})
        render_task(out,task,name)
    save_json(out/'reports/summary.json',{'rows':rows,'paired_comparisons':comparisons,
        'actual_model_calls':sum(r['model_calls'] for r in rows),'model_errors':sum(r['model_errors'] for r in rows),
        'numeric_errors':sum(r['numeric_errors'] for r in rows),'new_flights':0,'new_generated_submissions':0})
    save_json(out/'reports/audit.json',audits)
    md='# 因果反馈与在线选择：开发快照配对\n\n16项同状态求解，全部路线先冻结后评价。无新飞行/测量，不改变GP/共同C。LLM选计算过程，不能强制接受路径。\n\n'
    md+='|状态|方法|当前C/s|长度/m|真实T+Q/s|超参考/dB·s|总边|额外边|实际模型调用|选择|\n|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
    for r in rows:
        md+=f"|{r['case']}|{r['method']}|{r['current_C_seconds']:.3f}|{r['length_m']:.2f}|{r['future_joint_seconds']:.3f}|{r['future_excess_db_seconds']:.3f}|{r['edges']}|{r['additional_edges']}|{r['model_calls']}|{'→'.join(r['profiles'])}|\n"
    md+='\nGIF为所选路线的开环反事实：无新点测/后续重规划、无真值最优参考。不同任务是同一飞行的相关快照，不做统计/泛化或全程安全结论。新可靠性字段未做去字段消融，不归因于它。模型墙钟单列，样本/边节约不等于总推理时间改善。\n'
    (out/'RESULTS.md').write_text(md)
    gallery=''.join(f'<h2>{t} / {n}</h2><img src="{t}.gif">' for t,n in p['tasks'].items())
    (out/'replay/index.html').write_text('<!doctype html><meta charset="utf-8"><title>在线选择的固定路线回放</title><style>body{font:17px/1.8 sans-serif;max-width:1250px;margin:auto}img{max-width:100%}</style><h1>实际选择，固定路线开环回放</h1><p>紫色为真实LLM选择；各路线不再补测/重规划，没有真值最优参考。<a href="../index.html">完整报告</a></p>'+gallery)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>因果反馈与在线算子选择</title><style>body{font:17px/1.8 sans-serif;max-width:1250px;margin:auto}img{max-width:100%}</style><h1>因果可靠性反馈与在线算子选择</h1><p>16同快照任务；真实模型请求/原始响应/解析/执行分别留证，无新飞行、无目标或GP修改。</p><p><a href="RESULTS.md">全部数值</a> · <a href="reports/summary.json">配对结果</a> · <a href="replay/index.html">4个开环GIF</a> · <a href="reports/audit.json">核验</a> · <a href="reports/frozen_design.md">冻结设计</a></p>'+''.join(f'<h2>{t}/{n}</h2><img src="figures/{t}.png">' for t,n in p['tasks'].items()))
    save_json(out/'status.json',{'status':'completed','paired_tasks':16,'actual_model_calls':sum(r['model_calls'] for r in rows),
        'model_errors':sum(r['model_errors'] for r in rows),'new_flights':0,'new_generated_submissions':0})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','numeric','model','freeze','evaluate','report'])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--task',choices=['task01','task02','task03','task04'])
    a=parser.parse_args()
    if a.phase=='model':
        if not a.task:parser.error('--task required for model')
        model(a.out,a.task)
    else:globals()[a.phase](a.out)


if __name__=='__main__':main()
