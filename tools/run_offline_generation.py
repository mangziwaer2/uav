"""小预算离线LLM有限过程生成；独立会话通过public文件提交，不自动伪造响应。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from uav_em.agent_navigation import save_json, strict_json
from uav_em.config import load_config
from uav_em.decision_log import render_decision_log
from uav_em.decision_objective import prefix_risk, select_candidate
from uav_em.offline_process import OfflineProcessOptimizer, generation_contract, task_evaluation, validate_program
from uav_em.repair_process import canonical_path
from uav_em.repair_search import FixedObjective, WorkBudget

SOURCE = ROOT / 'outputs/interference_offline_contract_20261111'
DEFAULT = ROOT / 'outputs/interference_offline_generation_20261111'
SEEDS = (611, 613, 617, 619)
REFERENCES = ('pool_only', 'fixed_process', 'adaptive_process')
DRAFTS = ('draft_01', 'draft_02', 'revision_01', 'revision_02')
WORKERS = 4
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
now = lambda: datetime.now(timezone.utc).isoformat()


def parse_response(raw, request_id):
    response = strict_json(raw)
    if not isinstance(response, dict) or set(response) != {'request_id', 'program', 'rationale'}:
        raise ValueError('response requires exactly request_id/program/rationale')
    if response['request_id'] != request_id:
        raise ValueError('response request_id mismatch')
    if not isinstance(response['rationale'], str) or not 1 <= len(response['rationale']) <= 2400:
        raise ValueError('rationale must contain 1..2400 characters')
    return validate_program(response['program'])


def check(out):
    protocol = read(out / 'protocol.json')
    for relative, h in protocol['source_sha256'].items():
        assert sha(ROOT / relative) == sha(out / 'reports/runtime_source' / relative) == h, relative
    for relative, h in protocol['input_sha256'].items():
        assert sha(out / relative) == h, relative
    assert sha(out / 'reports/frozen_design.md') == protocol['design_sha256']
    return protocol


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('scenarios', 'public', 'submissions', 'runs', 'reports/runtime_source', 'logs', 'replay', 'figures'):
        (out / folder).mkdir(parents=True)
    shutil.copy2(ROOT / 'docs/offline_generation_v1.md', out / 'reports/frozen_design.md')
    assert generation_contract() == read(SOURCE / 'public/contract.json')
    inputs = {}
    for seed in SEEDS:
        target = out / 'scenarios' / f'dev{seed}.json'
        shutil.copy2(SOURCE / 'scenarios' / target.name, target)
        inputs[str(target.relative_to(out))] = sha(target)
    for draft in DRAFTS:
        public = out / 'public' / draft
        public.mkdir()
        for path in sorted((SOURCE / 'public').iterdir()):
            shutil.copy2(path, public / path.name)
            inputs[str((public / path.name).relative_to(out))] = sha(public / path.name)
    sources = {}
    for path in [*(ROOT / 'src/uav_em').glob('*.py'), Path(__file__).resolve(),
                 ROOT / 'tools/render_navigation_replays.py', ROOT / 'tests/test_offline_generation.py']:
        relative = path.relative_to(ROOT)
        target = out / 'reports/runtime_source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[str(relative)] = sha(path)
    save_json(out / 'protocol.json', {'study': 'offline_generated_finite_process_development_v1',
        'source_sha256': sources, 'input_sha256': inputs, 'design_sha256': sha(out / 'reports/frozen_design.md'),
        'seeds': list(SEEDS), 'references': list(REFERENCES), 'allowed_submissions': list(DRAFTS),
        'initial_drafts': 2, 'revision_drafts': 2, 'maximum_submissions': 4,
        'max_generated_flights': 16, 'reference_flights': 12, 'workers': WORKERS,
        'validation_test_generated_or_read': False, 'baseline_metrics_reused': False,
        'revision_parent_rule': 'revision_01 revises draft_01, revision_02 revises draft_02; invalid initial draft may be revised',
        'generation_and_runtime_separation': 'actual offline agent calls recorded separately; no runtime model consultations',
        'timing': '4 independent flight processes, not a controlled deployment latency benchmark',
        'selection': 'only passing artifacts eligible; minimum total flight excess, then length, then edge count; no passing means no winner'})
    save_json(out / 'status.json', {'status': 'prepared'})
    for draft in DRAFTS[:2]:
        request(out, draft)


def request(out, draft):
    check(out)
    if draft not in DRAFTS:
        raise ValueError('unknown draft id')
    public = out / 'public' / draft
    if (public / 'request.json').exists():
        raise FileExistsError('request already exists')
    revision = draft.startswith('revision')
    if revision:
        if not (out / 'reports/initial_summary.json').exists():
            raise ValueError('both initial submissions and all initial evaluations must be frozen first')
        parent = 'draft_' + draft[-2:]
        parent_status = read(out / 'submissions' / parent / 'status.json')
        shutil.copy2(out / 'public' / parent / 'response.json', public / 'parent_response.json')
        summary = read(out / 'reports/initial_summary.json')
        # Only own task metrics and the three numeric reference metrics. No paths,
        # source parameters, truth maps or other generated artifacts are provided.
        rows = [{k: row[k] for k in ('scene', 'method', 'success', 'executed_length_m',
                                      'interference_excess_integral_db_seconds', 'repair_edge_evaluations',
                                      'decision_hover_excess_db_seconds', 'policy_errors')}
                for row in summary['rows'] if row['method'] in REFERENCES + (parent,)]
        feedback = {'rows': rows, 'parent_parse_accepted': parent_status['accepted'],
                    'parent_parse_error': parent_status.get('error'), 'gate': summary['gates'].get(parent),
                    'development_feedback_not_unseen_validation': True}
        save_json(public / 'feedback.json', feedback)
    else:
        parent = None
    design_focus = ('resource-efficient local repair and early stopping' if draft.endswith('01') else
                    'complementary geometric constructions and feedback ordering under the same resource cap')
    instructions = '''You are an independent offline UAV replanning process designer.
Read ONLY the files in this request's public directory, including contract.json,
request.json, reference JSON and the six *_state.json causal examples. Do not read
project AGENTS/docs/source/configs, other experiments, parent work, hidden sources,
truth maps or future scripts. This public-directory allowlist is procedural, not OS isolation.
Do not run simulations, search source files, contact other agents or write outside
this directory. You may use local Python solely to validate the JSON you author.
The output is a finite process assembled from fixed operators, not Python source.
Respond exactly with {"request_id": <from request>, "program": {...}, "rationale": "..."}.
Save that exact raw response as response.json (refuse overwrite). Your final answer
must repeat the same JSON. Do not optimize against examples' scene IDs, coordinates
or timestamps; the program has only enum selectors and numeric-library construction.
No evaluation is available for initial drafts. Revision requests receive only task
metrics and parse errors from their own parent and numeric references. Never invent
measurements, bearing, safety guarantees, metric results or model-token counts.

Implementation semantics (public contract): every decision first builds the same
pool of geometric candidates and a 700-sample old-J RRT*. Only resources remaining
inside 2100 samples/10000 edges are available to your nodes. The incumbent is already
the shared R10-window/shortest choice, so another whole shortcut may be a duplicate.
RRT* still searches old-J; all complete candidates, even if J is larger, compete under
the common R10 selector. E/unknown/support/age are soft diagnostics, not true future risk.
shortcut joins selected remaining-route anchors. local_rrt searches between selected
anchors. via uses a geometric side waypoint and two RRT* legs. global_rrt searches
current position to goal. keep stops. Side is geometric, never measured source bearing.
evidence selector uses supported interference density if nonzero, else detour, else
unknown_front. supported_interference uses supported-weighted penalty density (if all
zero, falls back to unknown_front); unknown_front favors nearby unsupported segments;
stale ranks supported fraction times support age; detour ranks excess chord length.
padding expands in remaining anchor indices, scale sets fixed geometric corridor size.
fixed allocates at most 350 samples per node; feedback uses remaining edges and recent
edge/sample work, split across planned remaining search nodes, at most 700 per node.
improved means the shared selector changed retained route, not guaranteed lower true
exposure. duplicate may occur after construction, search work is still charged.
Every next object requires default, optional outcome keys, and only strictly later
node indices (0-based) or "stop". Stop on resource_stop, stopped or policy_error.
You may use fewer than six nodes, stop earlier or choose different selector/geometry/
allocation sequences. You cannot change scoring, source information, limits or sensors.
'''
    (public / 'instructions.md').write_text(instructions)
    payload = {'request_id': draft, 'kind': 'revision' if revision else 'initial', 'parent': parent,
               'design_focus': design_focus, 'objective': 'improve flight-excess/length/task-completion tradeoff versus numeric references',
               'gate': 'all arrive and hard audit; >=10% total excess reduction vs pool_only; mean length<=125%; '
                       'per-scene excess increase<=max(1,.1 baseline); zero-baseline excess scene length<=105%',
               'response_fields': ['request_id', 'program', 'rationale'], 'created_utc': now()}
    save_json(public / 'request.json', payload)
    save_json(public / 'request_manifest.json', {p.name: sha(p) for p in public.iterdir()
                                               if p.name not in ('response.json', 'request_manifest.json')})


def submit(out, draft, agent_id):
    check(out)
    if draft not in DRAFTS:
        raise ValueError('unknown draft id')
    target = out / 'submissions' / draft
    target.mkdir()  # Invalid submissions also occupy their frozen slot.
    public = out / 'public' / draft
    request_manifest = read(public / 'request_manifest.json')
    for name, h in request_manifest.items():
        assert sha(public / name) == h, name
    raw = (public / 'response.json').read_text()
    (target / 'raw_response.json').write_text(raw)
    accepted, error, program = False, None, None
    try:
        program = parse_response(raw, draft)
        save_json(target / 'program.json', program)
        accepted = True
    except (ValueError, TypeError, KeyError) as exc:
        error = f'{type(exc).__name__}: {exc}'
    metadata = {'draft': draft, 'accepted': accepted, 'error': error, 'agent_id': agent_id,
                'actual_offline_design_sessions': 1, 'submitted_utc': now(),
                'raw_response_sha256': sha(target / 'raw_response.json'),
                'request_manifest_sha256': sha(public / 'request_manifest.json'),
                'program_sha256': sha(target / 'program.json') if accepted else None,
                'model_tokens': None, 'model_tokens_note': 'unavailable from collaboration tool; not inferred',
                'independent_session': True, 'filesystem_boundary': 'procedural public allowlist, not OS isolation'}
    save_json(target / 'status.json', metadata)
    return metadata


def flight_task(out, method, seed):
    from uav_em.simulation import simulate
    run = out / 'runs' / f'{method}_dev{seed}_seed{seed}'
    run.mkdir()
    started = time.perf_counter()
    try:
        generated = method not in REFERENCES
        program = read(out / 'submissions' / method / 'program.json') if generated else None
        optimizer = OfflineProcessOptimizer('submitted_process' if generated else method, program)
        cfg = load_config(out / 'scenarios' / f'dev{seed}.json')
        episode = simulate(cfg, advisor_name=f'{method}_offline_process_no_runtime_llm', route_optimizer=optimizer)
        episode['assumptions'].update(offline_process_version='offline_finite_process_v1',
            method=method, offline_generated_artifact=generated, runtime_model_consultations=0,
            common_R10_selection_old_J_search=True, active_probe_enabled=False, concurrent_study_workers=WORKERS,
            artifact_sha256=sha(out / 'submissions' / method / 'program.json') if generated else None)
        for decision in episode['decisions']:
            reason = '离线有限过程；共同R10接受、旧J搜索/诊断；逐节点反馈见repair_optimization.process_trace'
            decision['decision']['reason'] = decision['proposed_decision']['reason'] = reason
            decision['accepted_route_fixed_objective_meaning'] = 'old current-GP J diagnostic, not acceptance or true exposure'
        save_json(run / 'episode.json', episode)
        save_json(run / 'metrics.json', episode['metrics'])
        (run / 'decision_log.txt').write_text(render_decision_log(episode) +
            '\n本轮为离线固定过程执行，运行时0模型调用；old J不是统一接受目标，R10/过程反馈在episode中。\n')
        save_json(run / 'status.json', {'status': 'completed', 'method': method, 'scene': f'dev{seed}',
                  'seed': seed, 'wall_seconds': time.perf_counter() - started, **episode['metrics']})
        return {'method': method, 'seed': seed, **episode['metrics']}
    except Exception as exc:
        save_json(run / 'status.json', {'status': 'failed', 'method': method, 'scene': f'dev{seed}', 'seed': seed,
                  'error': f'{type(exc).__name__}: {exc}', 'wall_seconds': time.perf_counter() - started})
        return {'method': method, 'seed': seed, 'error': f'{type(exc).__name__}: {exc}'}


def flights(out, phase):
    check(out)
    if phase == 'references':
        methods = REFERENCES
    elif phase == 'initial':
        methods = DRAFTS[:2]
    elif phase == 'revisions':
        methods = DRAFTS[2:]
    else:
        raise ValueError('unknown flight phase')
    frozen = out / 'reports' / f'{phase}_runs_manifest.json'
    if frozen.exists():
        raise FileExistsError('flight phase already frozen')
    if phase != 'references':
        for method in methods:
            if not (out / 'submissions' / method / 'status.json').exists():
                raise ValueError('all phase submissions required before flights')
        methods = tuple(m for m in methods if read(out / 'submissions' / m / 'status.json')['accepted'])
    for method in methods:
        if any((out / 'runs').glob(f'{method}_*')):
            raise FileExistsError('refuse to rerun existing artifact')
        if method not in REFERENCES:
            submission = read(out / 'submissions' / method / 'status.json')
            assert sha(out / 'submissions' / method / 'program.json') == submission['program_sha256']
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(flight_task, out, method, seed) for method in methods for seed in SEEDS]
        for f in as_completed(futures):
            print(json.dumps(f.result(), ensure_ascii=False), flush=True)
    manifest = {str(p.relative_to(out)): sha(p) for method in methods
                for folder in sorted((out / 'runs').glob(f'{method}_*')) for p in sorted(folder.iterdir()) if p.is_file()}
    save_json(frozen, {'methods': list(methods), 'sha256': manifest, 'all_flights_frozen_before_feedback': True})


def summarize(out, initial=False):
    check(out)
    phases = ('references', 'initial') if initial else ('references', 'initial', 'revisions')
    for phase in phases:
        manifest = read(out / 'reports' / f'{phase}_runs_manifest.json')
        for rel, h in manifest['sha256'].items():
            assert sha(out / rel) == h, rel
    methods = list(REFERENCES) + list(DRAFTS[:2] if initial else DRAFTS)
    rows, audits = [], []
    from uav_em.environment import ElectromagneticWorld
    for method in methods:
        if method not in REFERENCES and not read(out / 'submissions' / method / 'status.json')['accepted']:
            continue
        for seed in SEEDS:
            run = out / 'runs' / f'{method}_dev{seed}_seed{seed}'
            status = read(run / 'status.json')
            if status['status'] != 'completed':
                rows.append({'scene': f'dev{seed}', 'method': method, 'success': False, 'executed_length_m': 0.,
                             'interference_excess_integral_db_seconds': 0., 'repair_edge_evaluations': 0,
                             'decision_hover_excess_db_seconds': 0., 'policy_errors': 1, 'error': status['error']})
                continue
            episode = read(run / 'episode.json')
            cfg = load_config(out / 'scenarios' / f'dev{seed}.json')
            world = ElectromagneticWorld(cfg)
            hover = 0.
            decisions = []
            cap = episode['decisions'][0]['repair_optimization']['mission_distance_cap_m']
            for d in episode['decisions']:
                r = d['repair_optimization']
                w = r['resources']
                assert w['samples'] <= 2100 and w['edges'] <= 10000
                assert r['mission_distance_cap_m'] == cap
                actual_flown = sum(e['distance_m'] for e in episode['events'] if e['event'] == 'flight' and e['time_seconds'] <= d['time_seconds'] + 1e-8)
                assert abs(actual_flown - r['executed_distance_m']) < 1e-6
                selected = select_candidate(r['candidates'], 'prefix_v2')
                assert selected and selected['id'] == r['selected_candidate_id']
                path = selected['path']
                assert actual_flown + selected['metrics']['length_m'] <= cap + 1e-6
                assert all(world.geometry.edge_free(a, b) for a, b in zip(path, path[1:]))
                assert np.allclose(path[0], d['context']['vehicle']['position_m'], rtol=0, atol=1e-8)
                assert np.allclose(path[-1], d['context']['vehicle']['goal_m'], rtol=0, atol=1e-8)
                energy = selected['metrics']['length_m'] / cfg.aircraft.speed_mps * cfg.aircraft.flight_power_w / 3600 + cfg.simulation.decision_delay_seconds * cfg.aircraft.hover_power_w / 3600
                precise_energy = d['execution_checkpoint']['remaining_energy_wh']
                assert energy <= precise_energy - cfg.aircraft.reserve_energy_wh + 1e-7
                assert r['actual_advisor_calls'] == r['advisor_failures'] == 0
                duration = min(cfg.simulation.decision_delay_seconds, cfg.simulation.max_seconds - d['time_seconds'])
                if d['accepted_action'] is not None:
                    count = max(1, int(np.ceil(duration / .25)))
                    position = np.asarray(d['context']['vehicle']['position_m'])
                    hover += sum(max(0., float(world.truth(position, d['time_seconds'] + (i + .5) * duration / count)) -
                                     cfg.electromagnetic.interference_reference_dbm) * duration / count for i in range(count))
                decisions.append({'time_seconds': d['time_seconds'], 'geometry_endpoints_energy_distance_valid': True,
                                  'common_selection_valid': True, 'sample_edge_caps_valid': True,
                                  'process_nodes': len(r['process_trace']), 'policy_errors': r['policy_errors']})
            assert episode['metrics']['executed_length_m'] <= cap + 1e-6
            rows.append({'scene': f'dev{seed}', 'method': method, **episode['metrics'],
                         'decision_hover_excess_db_seconds': hover,
                         'flight_plus_hover_excess_db_seconds': episode['metrics']['interference_excess_integral_db_seconds'] + hover,
                         'policy_errors': sum(d['repair_optimization']['policy_errors'] for d in episode['decisions']),
                         'risk_field_nodes': sum(d['repair_optimization']['risk_field_nodes'] for d in episode['decisions'])})
            audits.append({'method': method, 'scene': f'dev{seed}', 'decisions': decisions})
    baseline = {r['scene']: r for r in rows if r['method'] == 'pool_only'}
    gates, totals = {}, {}
    for method in methods:
        group = [r for r in rows if r['method'] == method]
        if not group:
            gates[method] = {'passes_task_gate': False, 'reason': 'invalid submission'}
            continue
        assert len(group) == 4
        paired = [{'scene': r['scene'], 'baseline': baseline[r['scene']], 'candidate': r} for r in group]
        gate = task_evaluation(paired)
        gate['hard_and_execution_audit_passed'] = all(r['policy_errors'] == 0 and 'error' not in r for r in group)
        gate['passes_task_gate'] &= gate['hard_and_execution_audit_passed']
        gates[method] = gate
        totals[method] = {'total_flight_excess_db_seconds': sum(r['interference_excess_integral_db_seconds'] for r in group),
                          'mean_length_m': sum(r['executed_length_m'] for r in group) / 4,
                          'total_edges': sum(r['repair_edge_evaluations'] for r in group),
                          'all_arrive': all(r['success'] for r in group)}
    passing = [m for m in DRAFTS if m in gates and gates[m]['passes_task_gate']]
    winner = min(passing, key=lambda m: (totals[m]['total_flight_excess_db_seconds'], totals[m]['mean_length_m'], totals[m]['total_edges'])) if passing else None
    result = {'rows': rows, 'totals': totals, 'gates': gates, 'selected_passing_generated_artifact': winner,
              'actual_offline_design_sessions': len(list((out / 'submissions').glob('*/status.json'))),
              'runtime_model_calls': sum(r.get('advisor_calls', 0) for r in rows), 'unseen_validation': False,
              'all_failures_retained': True, 'new_generated_primitives': 0}
    dest = out / 'reports' / ('initial_summary.json' if initial else 'summary.json')
    if dest.exists():
        raise FileExistsError('summary already frozen')
    save_json(dest, result)
    save_json(out / 'reports' / ('initial_audit.json' if initial else 'audit.json'), audits)
    return result


def report(out):
    check(out)
    summary = read(out / 'reports/summary.json')
    from render_navigation_replays import render_replays
    render_replays(out, frame_stride=6, playback_speed=2., decision_pause_seconds=.5)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = list(summary['totals'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    width = .8 / len(methods)
    for i, method in enumerate(methods):
        group = sorted((r for r in summary['rows'] if r['method'] == method), key=lambda r: r['scene'])
        x = np.arange(4) + (i - (len(methods) - 1) / 2) * width
        axes[0].bar(x, [r['interference_excess_integral_db_seconds'] for r in group], width, label=method)
        axes[1].bar(x, [r['executed_length_m'] for r in group], width, label=method)
    for ax in axes:
        ax.set_xticks(np.arange(4), [f'dev{s}' for s in SEEDS])
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Actual flight excess (dB s)')
    axes[1].set_ylabel('Actual flight length (m)')
    fig.suptitle('Offline generated finite processes; inspected development scenes; 0 runtime LLM')
    fig.savefig(out / 'figures/comparison.png', dpi=140)
    plt.close(fig)
    md = '# 小预算离线过程生成：开发结果\n\n最多4份实际LLM产物、3共同数值参考；每组4个已看过的开发城市，全部重新运行。运行时0模型咨询；无新底层算子。\n\n'
    md += f"通过预声明门槛的选定产物：{summary['selected_passing_generated_artifact']}。\n\n"
    md += '|方法|全部到达|合计飞行积分/dB·s|平均航程/m|总边评价|完整门槛|\n|---|---|---:|---:|---:|---|\n'
    for method, total in summary['totals'].items():
        md += f"|{method}|{total['all_arrive']}|{total['total_flight_excess_db_seconds']:.3f}|{total['mean_length_m']:.3f}|{total['total_edges']}|{summary['gates'][method]['passes_task_gate']}|\n"
    md += '\n[全部指标与门槛](reports/summary.json) · [审计](reports/audit.json) · [集中GIF](replay/index.html)。\n\n'
    md += 'dB·s为仿真惩罚而非物理剂量，−80dBm是软参考。研究为离线LLM设计算法过程；不是在线LLM辅助决策收益或未见泛化。4进程墙钟不可作为部署时延。所有失败与原始响应保留，修订看过开发反馈。\n'
    (out / 'RESULTS.md').write_text(md)
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>离线有限过程生成</title>'
        '<style>body{font:17px/1.8 sans-serif;max-width:1300px;margin:30px auto;padding:15px}img{max-width:100%}</style>'
        '<h1>小预算离线LLM过程生成</h1><p>独立LLM仅接公开合同与因果示例；生成现有算子的有限过程。'
        '运行时0模型咨询，全部共同数值参考重新运行，开发集结果不代表泛化。</p>'
        '<p><a href="replay/index.html">集中回放GIF</a> · <a href="RESULTS.md">指标表</a> · '
        '<a href="reports/summary.json">完整指标与门槛</a> · <a href="reports/audit.json">约束审计</a> · '
        '<a href="../../docs/offline_generation_v1.md">方法与实际边界</a></p>'
        f'<p>通过门槛的选定产物：{html.escape(str(summary["selected_passing_generated_artifact"]))}</p>'
        '<img src="figures/comparison.png"><p>没有通过则无胜者；到达、J下降、合法程序不能替代暴露/航程权衡收益。</p>')
    save_json(out / 'status.json', {'status': 'completed', 'actual_offline_design_sessions': summary['actual_offline_design_sessions'],
              'runtime_model_calls': summary['runtime_model_calls'],
              'selected_passing_generated_artifact': summary['selected_passing_generated_artifact']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'references', 'submit', 'initial', 'initial_summary', 'request', 'revisions', 'evaluate', 'report'])
    p.add_argument('--out', type=Path, default=DEFAULT)
    p.add_argument('--draft', choices=DRAFTS)
    p.add_argument('--agent-id')
    a = p.parse_args()
    if a.phase == 'prepare': prepare(a.out)
    elif a.phase in ('references', 'initial', 'revisions'): flights(a.out, a.phase)
    elif a.phase == 'submit':
        if not a.draft or not a.agent_id: p.error('submit requires --draft and --agent-id')
        print(json.dumps(submit(a.out, a.draft, a.agent_id), ensure_ascii=False))
    elif a.phase == 'request':
        if not a.draft: p.error('request requires --draft')
        request(a.out, a.draft)
    elif a.phase == 'initial_summary': summarize(a.out, initial=True)
    elif a.phase == 'evaluate': summarize(a.out)
    elif a.phase == 'report': report(a.out)


if __name__ == '__main__': main()
