"""生成过程的固定状态/阶段截断诊断：30配对、10闭环、0新模型调用。"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from uav_em.agent_navigation import save_json
from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.decision_log import render_decision_log
from uav_em.decision_objective import cut_route, prefix_risk, select_candidate
from uav_em.environment import Geometry, Measurement
from uav_em.offline_process import OfflineProcessOptimizer
from uav_em.process_attribution import diagnostic_prefix, check_identical_executed_prefix, selection_comparison
from uav_em.repair_process import canonical_path
from uav_em.repair_search import FixedObjective, WorkBudget
from run_repair_diagnostics import load_case

SOURCE = ROOT / 'outputs/interference_offline_generation_20261111'
DEFAULT = ROOT / 'outputs/interference_process_attribution_20261111'
SEEDS = (613, 619)
MODES = ('pool_only', 'local_prefix', 'via_prefix', 'global_prefix', 'full_process')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def make_optimizer(out, mode):
    if mode == 'pool_only':
        return OfflineProcessOptimizer('pool_only')
    return OfflineProcessOptimizer('submitted_process', read(out / 'programs' / f'{mode}.json'))


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('cases', 'scenarios', 'programs', 'source_pools', 'paired', 'runs', 'evaluations',
                   'reports/runtime_source', 'logs', 'replay', 'figures'):
        (out / folder).mkdir(parents=True)
    shutil.copy2(ROOT / 'docs/process_attribution_v1.md', out / 'reports/frozen_design.md')
    parent = read(SOURCE / 'submissions/draft_02/program.json')
    save_json(out / 'programs/parent.json', parent)
    for mode in MODES[1:]:
        save_json(out / 'programs' / f'{mode}.json', diagnostic_prefix(parent, mode))
    external = {str((SOURCE / 'submissions/draft_02/program.json').relative_to(ROOT)):
                sha(SOURCE / 'submissions/draft_02/program.json')}
    source_rows = []
    for seed in SEEDS:
        cfgpath = SOURCE / 'scenarios' / f'dev{seed}.json'
        shutil.copy2(cfgpath, out / 'scenarios' / cfgpath.name)
        cfg = load_config(cfgpath)
        for old_method in ('pool_only', 'draft_02'):
            path = SOURCE / 'runs' / f'{old_method}_dev{seed}_seed{seed}' / 'episode.json'
            external[str(path.relative_to(ROOT))] = sha(path)
        ep = read(SOURCE / 'runs' / f'draft_02_dev{seed}_seed{seed}' / 'episode.json')
        geometry = Geometry(cfg.map, cfg.aircraft.clearance_m, cfg.aircraft.altitude_m, cfg.aircraft.vertical_clearance_m)
        belief = ElectromagneticBelief(cfg, geometry)
        points = [e for e in ep['events'] if e['event'] == 'observation']
        chosen_decisions = {next(i for i, d in enumerate(ep['decisions']) if d['time_seconds'] >= threshold)
                            for threshold in (0, 20, 40)}
        index = 0
        for i, d in enumerate(ep['decisions']):
            r = d['repair_optimization']
            name = f'dev{seed}_decision{i}'
            save_json(out / 'source_pools' / f'{name}.json', {'scene': f'dev{seed}', 'time_seconds': d['time_seconds'],
                      'records': r['candidates'], 'selected_candidate_id': r['selected_candidate_id']})
            source_rows.append(name)
            if i not in chosen_decisions:
                continue
            now = d['time_seconds']
            while index < len(points) and points[index]['time_seconds'] <= now + 1e-9:
                p = points[index]
                belief.update(Measurement(tuple(p['position_m']), p['time_seconds'], p['interference_dbm']))
                index += 1
            snapshot = belief.snapshot(now)
            selected = next(c for c in r['candidates'] if c['id'] == r['selected_candidate_id'])
            score = prefix_risk(FixedObjective(snapshot, -80, WorkBudget()), selected['path'], cfg.aircraft.speed_mps)
            for key in score:
                if isinstance(score[key], float): assert abs(score[key] - selected['prefix'][key]) < 1e-8
                else: assert score[key] == selected['prefix'][key]
            energy = d['execution_checkpoint']['remaining_energy_wh']
            maximum = max(0., (energy - cfg.aircraft.reserve_energy_wh -
                              cfg.simulation.decision_delay_seconds * cfg.aircraft.hover_power_w / 3600) *
                          3600 * cfg.aircraft.speed_mps / cfg.aircraft.flight_power_w)
            route = [] if now == 0 else next(c['path'] for c in r['candidates'] if c['id'] == 'incumbent')
            case = {'scene': f'dev{seed}', 'time_seconds': now, 'decision_index': i, 'context': d['context'],
                    'route': route, 'max_length_m': maximum, 'seed': cfg.seed + sum(p['time_seconds'] < now - 1e-9 for p in ep['plans']),
                    'reference_length_m': r['mission_reference_length_m'], 'flown_m': r['executed_distance_m'],
                    'observation_count': snapshot.observation_count, 'estimation_details': snapshot.estimation_details,
                    'source_selected_path': selected['path'], 'source_selected_metrics': selected['metrics'],
                    'source_resources': r['resources']}
            save_json(out / 'cases' / f'{name}.json', case)
            np.savez_compressed(out / 'cases' / f'{name}.npz', **{key: getattr(snapshot, key) for key in
                ('estimate_dbm', 'uncertainty_db', 'supported', 'age_seconds', 'nearest_support_distance_m', 'local_support_count')})
    inputs = {str(p.relative_to(out)): sha(p) for folder in ('cases', 'scenarios', 'programs', 'source_pools')
              for p in sorted((out / folder).iterdir())}
    sources = {}
    for path in [*(ROOT / 'src/uav_em').glob('*.py'), Path(__file__).resolve(),
                 ROOT / 'tools/run_repair_diagnostics.py', ROOT / 'tools/render_navigation_replays.py',
                 ROOT / 'tests/test_process_attribution.py']:
        relative = path.relative_to(ROOT)
        target = out / 'reports/runtime_source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[str(relative)] = sha(path)
    save_json(out / 'protocol.json', {'study': 'fixed_generated_process_prefix_attribution_v1',
        'seeds': list(SEEDS), 'modes': list(MODES), 'cases': sorted(p.stem for p in (out / 'cases').glob('*.json')),
        'source_pools': source_rows, 'paired_tasks': 30, 'new_flights': 10, 'new_model_calls': 0,
        'input_sha256': inputs, 'source_sha256': sources, 'external_source_sha256': external,
        'design_sha256': sha(out / 'reports/frozen_design.md'), 'workers': 4,
        'state_selection': 'first parent decision at or after 0/20/40s in each of two declared scenes',
        'intervention': 'retain all six actions and allocation suffix counts; only cut later transitions',
        'not_a_fifth_generated_submission': True, 'validation_test_generated_or_read': False,
        'ranking_evaluation': 'freeze ALL paired candidates first; only then evaluate selected fixed 10s prefixes',
        'not_search_objective_ablation': True, 'not_unseen_or_method_gate_evaluation': True})
    save_json(out / 'status.json', {'status': 'prepared'})


def check(out):
    p = read(out / 'protocol.json')
    assert sha(out / 'reports/frozen_design.md') == p['design_sha256']
    for rel, h in p['input_sha256'].items(): assert sha(out / rel) == h, rel
    for rel, h in p['source_sha256'].items():
        assert sha(ROOT / rel) == sha(out / 'reports/runtime_source' / rel) == h, rel
    for rel, h in p['external_source_sha256'].items(): assert sha(ROOT / rel) == h, rel
    return p


def paired(out):
    protocol = check(out)
    if any((out / 'paired').iterdir()): raise FileExistsError('paired tasks already exist')
    audits = []
    for name in protocol['cases']:
        case, snapshot, _ = load_case(out, out / 'cases' / f'{name}.json')
        reports = {}
        for mode in MODES:
            optimizer = make_optimizer(out, mode)
            optimizer.base.reference_length_m = case['reference_length_m']
            interval = case['context'].get('execution_feedback', {}).get('interval')
            optimizer.base.flown_m = case['flown_m'] - (interval['flown_m'] if interval else 0)
            plan, report = optimizer.optimize(snapshot, deepcopy(case['context']), case['route'], case['max_length_m'], case['seed'], 0)
            assert plan.path and report['policy_errors'] == 0
            assert report['resources']['samples'] <= 2100 and report['resources']['edges'] <= 10000
            assert report['executed_distance_m'] == case['flown_m']
            assert plan.length_m + case['flown_m'] <= report['mission_distance_cap_m'] + 1e-6
            assert plan.length_m <= case['max_length_m'] + 1e-6
            save_json(out / 'paired' / f'{name}_{mode}.json', {'case': name, 'mode': mode, 'path': plan.path, 'report': report})
            reports[mode] = report
            print(name, mode, 'nodes', len(report['process_trace']), 'edges', report['resources']['edges'], flush=True)
        full = reports['full_process']
        full_choice = select_candidate(full['candidates'], 'prefix_v2')
        assert canonical_path(full_choice['path']) == canonical_path(case['source_selected_path'])
        assert full['resources'] == case['source_resources']
        assert full['best_metrics'] == case['source_selected_metrics']
        comparisons = {mode: check_identical_executed_prefix(full, reports[mode]) for mode in MODES[1:]}
        n = len(reports['pool_only']['candidates'])
        assert all(r['candidates'][:n] == reports['pool_only']['candidates'] for r in reports.values())
        audits.append({'case': name, 'original_parent_path_metrics_and_work_reproduce': True,
                       'same_initial_candidate_pool': True, 'prefix_checks': comparisons})
    save_json(out / 'reports/paired_manifest.json', {str(p.relative_to(out)): sha(p) for p in sorted((out / 'paired').iterdir())})
    save_json(out / 'reports/paired_audit.json', audits)


def verify_paired(out):
    check(out)
    manifest = read(out / 'reports/paired_manifest.json')
    assert len(manifest) == 30
    for rel, h in manifest.items(): assert sha(out / rel) == h, rel


def flight_task(out, mode, seed):
    from uav_em.simulation import simulate
    run = out / 'runs' / f'{mode}_dev{seed}_seed{seed}'
    run.mkdir()
    try:
        cfg = load_config(out / 'scenarios' / f'dev{seed}.json')
        episode = simulate(cfg, advisor_name=f'{mode}_diagnostic_no_model', route_optimizer=make_optimizer(out, mode))
        episode['assumptions'].update(process_prefix_attribution=True, mode=mode, new_generated_artifact=False,
                                      original_parent='offline_generation/draft_02', runtime_model_calls=0,
                                      unused_prefix_budget_redistributed=False)
        for d in episode['decisions']:
            d['decision']['reason'] = d['proposed_decision']['reason'] = '原生成过程阶段截断诊断；共同R10接受/旧J搜索，非新LLM生成'
            d['accepted_route_fixed_objective_meaning'] = 'old J diagnostic, not selection criterion'
        save_json(run / 'episode.json', episode)
        save_json(run / 'metrics.json', episode['metrics'])
        (run / 'decision_log.txt').write_text('阶段截断诊断，0模型调用；old J仅作搜索/诊断，共同R10接受。\n' + render_decision_log(episode))
        save_json(run / 'status.json', {'status': 'completed', 'method': mode, 'scene': f'dev{seed}', 'seed': seed, **episode['metrics']})
        return {'mode': mode, 'seed': seed, **episode['metrics']}
    except Exception as exc:
        save_json(run / 'status.json', {'status': 'failed', 'method': mode, 'scene': f'dev{seed}', 'seed': seed,
                                       'error': f'{type(exc).__name__}: {exc}'})
        return {'mode': mode, 'seed': seed, 'error': f'{type(exc).__name__}: {exc}'}


def flights(out):
    verify_paired(out)
    if any((out / 'runs').iterdir()): raise FileExistsError('flights already exist')
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(flight_task, out, mode, seed) for mode in MODES for seed in SEEDS]
        for f in as_completed(futures): print(json.dumps(f.result(), ensure_ascii=False), flush=True)
    save_json(out / 'reports/run_manifest.json', {str(p.relative_to(out)): sha(p) for p in sorted((out / 'runs').rglob('*')) if p.is_file()})


def evaluate(out):
    verify_paired(out)
    for rel, h in read(out / 'reports/run_manifest.json').items(): assert sha(out / rel) == h, rel
    if (out / 'reports/summary.json').exists(): raise FileExistsError('evaluation already exists')
    from uav_em.environment import ElectromagneticWorld
    from uav_em.diagnostic_evaluation import evaluate_fixed_route
    from uav_em.process_attribution_evaluation import evaluate_frozen_prefix
    fixed_rows = []
    for name in read(out / 'protocol.json')['cases']:
        case, snapshot, cfg = load_case(out, out / 'cases' / f'{name}.json')
        world = ElectromagneticWorld(cfg)
        for mode in MODES:
            task = read(out / 'paired' / f'{name}_{mode}.json')
            front, _ = cut_route(task['path'], cfg.aircraft.speed_mps * 10)
            evaluation = evaluate_fixed_route(world, snapshot, front)
            fixed_rows.append({'case': name, 'scene': case['scene'], 'time_seconds': case['time_seconds'], 'mode': mode,
                'selected_id': task['report']['selected_candidate_id'], 'prefix': task['report']['selected_prefix'],
                'metrics': task['report']['best_metrics'], 'resources': task['report']['resources'], 'evaluation': evaluation})
    save_json(out / 'evaluations/fixed_state_10s.json', fixed_rows)
    ranking_rows = []
    for path in sorted((out / 'source_pools').glob('*.json')):
        data = read(path)
        cfg = load_config(out / 'scenarios' / f"{data['scene']}.json")
        world = ElectromagneticWorld(cfg)
        comparison = selection_comparison(data['records'])
        # These are evaluation-only selected frozen routes. No estimator or planner
        # is updated, and no true value is returned to any controller.
        truth = {}
        for rule in ('risk_choice', 'old_j_choice'):
            truth[rule] = evaluate_frozen_prefix(world, comparison[rule]['path'], data['time_seconds'])
        ranking_rows.append({'source_pool': path.stem, 'scene': data['scene'], 'time_seconds': data['time_seconds'],
                             **comparison, 'posthoc_10s_truth': truth})
    save_json(out / 'evaluations/frozen_pool_J_vs_R10.json', ranking_rows)
    rows, audits, reproductions = [], [], []
    for seed in SEEDS:
        cfg = load_config(out / 'scenarios' / f'dev{seed}.json')
        world = ElectromagneticWorld(cfg)
        for mode in MODES:
            folder = out / 'runs' / f'{mode}_dev{seed}_seed{seed}'
            status = read(folder / 'status.json')
            if status['status'] != 'completed':
                rows.append({'scene': f'dev{seed}', 'mode': mode, 'success': False, 'error': status['error']})
                continue
            episode = read(folder / 'episode.json')
            hover, checked = 0., []
            cap = episode['decisions'][0]['repair_optimization']['mission_distance_cap_m']
            for d in episode['decisions']:
                r = d['repair_optimization']; w = r['resources']
                assert w['samples'] <= 2100 and w['edges'] <= 10000 and r['policy_errors'] == 0
                assert r['mission_distance_cap_m'] == cap
                flown = sum(e['distance_m'] for e in episode['events'] if e['event']=='flight' and e['time_seconds'] <= d['time_seconds']+1e-8)
                assert abs(flown-r['executed_distance_m'])<1e-6
                selected = select_candidate(r['candidates'], 'prefix_v2')
                assert selected and selected['id'] == r['selected_candidate_id']
                assert flown + selected['metrics']['length_m'] <= cap + 1e-6
                assert all(world.geometry.edge_free(a,b) for a,b in zip(selected['path'],selected['path'][1:]))
                assert np.allclose(selected['path'][0],d['context']['vehicle']['position_m'],atol=1e-8,rtol=0)
                assert np.allclose(selected['path'][-1],d['context']['vehicle']['goal_m'],atol=1e-8,rtol=0)
                duration = min(cfg.simulation.decision_delay_seconds,cfg.simulation.max_seconds-d['time_seconds'])
                required = selected['metrics']['length_m']/cfg.aircraft.speed_mps*cfg.aircraft.flight_power_w/3600+duration*cfg.aircraft.hover_power_w/3600
                assert required <= d['execution_checkpoint']['remaining_energy_wh']-cfg.aircraft.reserve_energy_wh+1e-7
                if d['accepted_action'] is not None:
                    count = max(1,int(np.ceil(duration/.25)));point=np.asarray(d['context']['vehicle']['position_m'])
                    hover += sum(max(0.,float(world.truth(point,d['time_seconds']+(i+.5)*duration/count))-cfg.electromagnetic.interference_reference_dbm)*duration/count for i in range(count))
                checked.append({'time_seconds':d['time_seconds'],'geometry_energy_cap_and_budget_valid':True})
            assert episode['metrics']['executed_length_m'] <= cap+1e-6
            assert episode['metrics']['advisor_calls']==episode['metrics']['probes']==0
            rows.append({'scene':f'dev{seed}','mode':mode,**episode['metrics'],'decision_hover_excess_db_seconds':hover,
                         'flight_plus_hover_excess_db_seconds':episode['metrics']['interference_excess_integral_db_seconds']+hover})
            audits.append({'scene':f'dev{seed}','mode':mode,'decisions':checked})
            if mode in ('pool_only','full_process'):
                original = 'pool_only' if mode=='pool_only' else 'draft_02'
                old = read(SOURCE/'runs'/f'{original}_dev{seed}_seed{seed}'/'episode.json')
                for field in ('executed_length_m','interference_excess_integral_db_seconds','simulation_seconds'):
                    assert abs(old['metrics'][field]-episode['metrics'][field])<1e-8
                np.testing.assert_allclose([f['position_m'] for f in episode['frames']],[f['position_m'] for f in old['frames']],atol=1e-8,rtol=0)
                reproductions.append({'scene':f'dev{seed}','mode':mode,'old_positions_and_metrics_reproduce':True})
    save_json(out/'reports/summary.json',{'closed_loop_rows':rows,'same_state_10s_rows':fixed_rows,
              'frozen_pool_comparisons':ranking_rows,'control_reproductions':reproductions,
              'new_model_calls':0,'new_generated_submissions':0,'method_benefit_gate_evaluated':False,
              'scope':'two inspected development scenes; prefix diagnostics, not new method selection'})
    save_json(out/'reports/audit.json',{'closed_loop':audits,'control_reproductions':reproductions})


def report(out):
    check(out)
    summary=read(out/'reports/summary.json')
    from render_navigation_replays import render_replays
    render_replays(out,frame_stride=6,playback_speed=2.,decision_pause_seconds=.5)
    p=out/'replay/index.html';p.write_text(p.read_text().replace('动画默认按仿真1倍速','本轮动画按仿真2倍速'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for i,mode in enumerate(MODES):
        group=sorted([r for r in summary['closed_loop_rows'] if r['mode']==mode],key=lambda r:r['scene'])
        x=np.arange(2)+(i-2)*.16
        axes[0].bar(x,[r.get('interference_excess_integral_db_seconds',0) for r in group],width=.16,label=mode)
        axes[1].bar(x,[r.get('executed_length_m',0) for r in group],width=.16,label=mode)
    for ax in axes:ax.set_xticks(np.arange(2),['dev613','dev619']);ax.legend(fontsize=8)
    axes[0].set_ylabel('Actual flight excess (dB s)');axes[1].set_ylabel('Actual flight length (m)')
    fig.suptitle('Diagnostic prefix interventions; unchanged allocation suffix; no new generated method')
    fig.savefig(out/'figures/prefix_comparison.png',dpi=140);plt.close(fig)
    md='# 原生成过程的阶段截断诊断\n\n30同状态求解、10新闭环，0模型调用/新生成产物。只用两个已观察开发场景，不选新方法。\n\n'
    md+='|场景|截断模式|到达|航程/m|飞行积分/dB·s|决策悬停/dB·s|总边评价|\n|---|---|---|---:|---:|---:|---:|\n'
    for r in summary['closed_loop_rows']:
        md+=f"|{r['scene']}|{r['mode']}|{r['success']}|{r.get('executed_length_m',0):.3f}|{r.get('interference_excess_integral_db_seconds',0):.3f}|{r.get('decision_hover_excess_db_seconds',0):.3f}|{r.get('repair_edge_evaluations',0)}|\n"
    md+='\n[完整结果](reports/summary.json) · [同状态前缀核验](reports/paired_audit.json) · [10秒反事实](evaluations/fixed_state_10s.json) · [固定池J/R10比较](evaluations/frozen_pool_J_vs_R10.json) · [10个实际飞行GIF](replay/index.html)。\n\n保留全部动作和额度后缀，仅改变停止位置；闭环后观测/种子流可能分岔，不把完整任务差值归因于单次操作。真值仅事后评价，固定10秒反事实不是实际飞行。\n'
    (out/'RESULTS.md').write_text(md)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>生成过程归因诊断</title><style>body{font:17px/1.8 sans-serif;max-width:1300px;margin:25px auto;padding:15px}img{max-width:100%}</style><h1>生成过程阶段截断与同状态诊断</h1><p>30同状态求解、10新闭环，0模型/新生成产物。完整参考与共同池均重跑核验，不把两个开发城的消融当新方法验收。</p><p><a href="replay/index.html">10个实际飞行GIF</a> · <a href="RESULTS.md">数值表</a> · <a href="reports/summary.json">全部结果</a> · <a href="reports/paired_audit.json">停止前逐位一致核验</a> · <a href="../../docs/process_attribution_v1.md">方法与局限</a></p><img src="figures/prefix_comparison.png"><p>未知不等于安全；旧J与R10重排只比较冻结候选，不是搜索目标消融。dB·s为仿真惩罚而非物理剂量。</p>')
    save_json(out/'status.json',{'status':'completed','new_model_calls':0,'new_generated_submissions':0,
                               'paired_tasks':30,'new_flights':10,'method_benefit_gate_evaluated':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','paired','flights','evaluate','report','all'])
    p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args()
    if a.phase=='all':
        for fn in (prepare,paired,flights,evaluate,report):fn(a.out)
    else:globals()[a.phase](a.out)


if __name__=='__main__':main()
