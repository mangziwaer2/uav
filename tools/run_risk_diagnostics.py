"""dev617风险快照诊断：先冻结全部候选，再事后评价；无新飞行/模型。"""
from __future__ import annotations
import argparse
from copy import deepcopy
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
from uav_em.agent_navigation import save_json
from uav_em.balanced_objective import PowerTimeSettings, paired_seed
from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.environment import Geometry, Measurement
from uav_em.repair_diagnostics import SEARCH_PROFILES, geometric_candidates, search_candidate
from uav_em.risk_diagnostics import received_before_decision, score_pool, select_decisions, summarize
from run_repair_diagnostics import load_case

SOURCE = ROOT / 'outputs/interference_balanced_objective_20261111'
DEFAULT = ROOT / 'outputs/interference_risk_diagnostics_20261111'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('inputs', 'scenarios', 'cases', 'candidates', 'evaluations', 'reports/runtime_source', 'figures', 'logs', 'replay'):
        (out / folder).mkdir(parents=True)
    inputs = {}
    for src, dst in [(SOURCE / 'scenarios/dev617.json', out / 'scenarios/dev617.json'),
                     (SOURCE / 'runs/balanced_pool_dev617_seed617/episode.json', out / 'inputs/episode.json'),
                     (ROOT / 'docs/risk_repair_diagnostics_v3.md', out / 'reports/frozen_design.md')]:
        shutil.copy2(src, dst)
        inputs[str(dst.relative_to(out))] = sha(dst)
    sources = {}
    paths = [*(ROOT / 'src/uav_em').glob('*.py'), Path(__file__).resolve(), ROOT / 'tools/run_repair_diagnostics.py',
             ROOT / 'tests/test_risk_diagnostics.py']
    for path in paths:
        target = out / 'reports/runtime_source' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[str(path.relative_to(ROOT))] = sha(path)
    save_json(out / 'protocol.json', {'study': 'v3_risk_state_candidate_and_proxy_diagnosis',
        'source_study': str(SOURCE.relative_to(ROOT)), 'input_sha256': inputs, 'source_sha256': sources,
        'selection': ['first decision >=20s', 'first reading >=-80dBm', 'first decision >=30s', 'first decision >=40s'],
        'profiles': list(SEARCH_PROFILES), 'repeat_ids': [0, 1], 'maximum_states': 4, 'maximum_searches': 48,
        'independent_search_limits': {'sample_quota': 700, 'sample_cap': 2100, 'edge_cap': 10000},
        'aggregate_pool_is_not_one_online_budget': True, 'weight': 1., 'scale_dbm': -80.,
        'truth_joint_reduction': 'strict', 'excess_reduction_fraction': .10, 'material_proxy_reduction_fraction': .01,
        'new_model_calls': 0, 'new_generated_submissions': 0, 'new_flights': 0,
        'validation_test_read_or_generated': False, 'no_truth_adaptive_candidate_expansion': True})
    save_json(out / 'status.json', {'status': 'prepared'})


def check(out):
    protocol = read(out / 'protocol.json')
    for relative, expected in protocol['input_sha256'].items():
        if sha(out / relative) != expected:
            raise ValueError('frozen input changed: ' + relative)
    for relative, expected in protocol['source_sha256'].items():
        if sha(ROOT / relative) != expected or sha(out / 'reports/runtime_source' / relative) != expected:
            raise ValueError('runtime source changed: ' + relative)
    return protocol


def verify_manifest(out, name):
    manifest = read(out / name)
    for relative, expected in manifest['sha256'].items():
        if sha(out / relative) != expected:
            raise ValueError('frozen output changed: ' + relative)
    return manifest


def capture(out):
    check(out)
    if any((out / 'cases').iterdir()):
        raise FileExistsError('cases already exist')
    episode = read(out / 'inputs/episode.json')
    cfg = load_config(out / 'scenarios/dev617.json')
    geometry = Geometry(cfg.map, cfg.aircraft.clearance_m, cfg.aircraft.altitude_m, cfg.aircraft.vertical_clearance_m)
    selected, missing = select_decisions([d['context'] for d in episode['decisions']], cfg.electromagnetic.interference_reference_dbm)
    audit = []
    for index, labels in selected.items():
        decision = episode['decisions'][index]
        context, old = decision['context'], decision['repair_optimization']
        observations = received_before_decision(episode['events'], index)
        belief = ElectromagneticBelief(cfg, geometry)
        for point in observations:
            belief.update(Measurement(tuple(point['position_m']), point['time_seconds'], point['interference_dbm']))
        snapshot = belief.snapshot(decision['time_seconds'])
        assert snapshot.source_hypothesis is None and not snapshot.directional_observations
        assert snapshot.observation_count == context['electromagnetic_estimate']['observation_count']
        assert observations[-1]['time_seconds'] == context['latest_observation']['time_seconds']
        assert observations[-1]['interference_dbm'] == context['latest_observation']['interference_dbm']
        # Check every originally feasible candidate, not only the selected route.
        original = deepcopy(old['candidates'])
        for r in original:
            r['family'] = 'incumbent' if r['id'] == 'incumbent' else 'original_pool'
        energy = context['vehicle']['remaining_energy_wh']
        maximum = max(0., (energy - cfg.aircraft.reserve_energy_wh -
                          cfg.simulation.decision_delay_seconds * cfg.aircraft.hover_power_w / 3600) *
                      3600 * cfg.aircraft.speed_mps / cfg.aircraft.flight_power_w)
        mission = old['mission_distance_cap_m'] - old['executed_distance_m']
        records, selection, _ = score_pool(snapshot, context, original, maximum, mission, PowerTimeSettings(**old['power_time_settings']))
        checks = 0
        for before, after in zip(original, records):
            assert before['feasible'] == after['feasible']
            if not before['feasible']:
                continue
            for group in ('metrics', 'prefix', 'balance'):
                for key, expected in before[group].items():
                    actual = after[group][key]
                    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                        assert abs(expected - actual) < 1e-7, (index, before['id'], group, key, actual, expected)
                    else:
                        assert expected == actual
                    checks += 1
        assert selection['expanded_best_id'] == old['selected_candidate_id']
        route = next(r['path'] for r in original if r['id'] == 'incumbent')
        name = f"dev617_t{snapshot.time_seconds:g}"
        case = {'case': name, 'scene': 'dev617', 'time_seconds': snapshot.time_seconds,
                'context': context, 'route': route, 'max_length_m': min(maximum, mission),
                'energy_limit_m': maximum, 'mission_remaining_m': mission,
                'mission_total_cap_m': old['mission_distance_cap_m'], 'flown_m': old['executed_distance_m'],
                'seed': cfg.seed, 'observation_count': snapshot.observation_count, 'estimation_details': snapshot.estimation_details,
                'selection_labels': labels, 'source_decision_index': index, 'received_point_measurements': observations,
                'original_candidates': original, 'score_settings': old['power_time_settings'],
                'reconstructed_score_scalar_checks': checks}
        save_json(out / 'cases' / f'{name}.json', case)
        np.savez_compressed(out / 'cases' / f'{name}.npz', **{key: getattr(snapshot, key) for key in
            ('estimate_dbm', 'uncertainty_db', 'supported', 'age_seconds', 'nearest_support_distance_m', 'local_support_count')})
        audit.append({'case': name, 'observations': len(observations), 'original_feasible_candidates': sum(r['feasible'] for r in original),
                      'scalar_checks': checks, 'all_match': True})
    save_json(out / 'reports/capture_audit.json', {'states': audit, 'missing_labels': missing})
    save_json(out / 'case_manifest.json', {'sha256': {str(p.relative_to(out)): sha(p) for p in sorted((out / 'cases').iterdir())}})
    save_json(out / 'status.json', {'status': 'causal_states_reconstructed', 'states': len(selected)})


def propose(out):
    protocol = check(out)
    verify_manifest(out, 'case_manifest.json')
    if any((out / 'candidates').iterdir()) or (out / 'candidate_manifest.json').exists():
        raise FileExistsError('candidate generation is not repeatable into an existing output')
    cases = sorted((out / 'cases').glob('*.json'))
    assert len(cases) <= protocol['maximum_states']
    for path in cases:
        case, snapshot, _ = load_case(out, path)
        started = time.perf_counter()
        geometric, public, work = geometric_candidates(snapshot, case['context'], case['route'], case['max_length_m'])
        records = deepcopy(case['original_candidates'])
        for record in geometric:
            if record['id'] != 'incumbent':
                record['id'] = 'diagnostic_' + record['id']
                records.append(record)
        searches = []
        for profile in protocol['profiles']:
            for repeat in protocol['repeat_ids']:
                seed = paired_seed(case['seed'], case['time_seconds'], 'risk_diagnostic', profile, repeat)
                record = search_candidate(snapshot, case['context'], case['route'], case['max_length_m'], profile, seed)
                record['old_j_search_accepted'] = record.pop('accepted')
                record['stream_key'] = {'stage': 'risk_diagnostic', 'purpose': profile, 'repeat': repeat}
                records.append(record)
                searches.append(record['report'])
            print(case['case'], profile, 'done', flush=True)
        records, selection, costs = score_pool(snapshot, case['context'], records, case['energy_limit_m'], case['mission_remaining_m'],
                                               PowerTimeSettings(**case['score_settings']))
        save_json(out / 'candidates' / path.name, {'case': case['case'], 'records': records, 'selection': selection,
            'public_context': public, 'geometric_pool_resources': work, 'rescoring_resources': costs,
            'elapsed_seconds': time.perf_counter() - started,
            'aggregate_additional_search_resources': {'independent_attempts': len(searches),
                'actual_rrt_trees': sum(len(r['searches']) for r in searches),
                'samples': sum(r['resources']['samples'] for r in searches), 'edges': sum(r['resources']['edges'] for r in searches),
                'numeric_compute_seconds': sum(r['numeric_compute_seconds'] for r in searches),
                'model_calls': sum(r['actual_advisor_calls'] for r in searches), 'policy_errors': sum(r['policy_errors'] for r in searches)},
            'not_one_online_search_budget': True})
    files = {str(p.relative_to(out)): sha(p) for folder in ('cases', 'candidates') for p in sorted((out / folder).iterdir())}
    save_json(out / 'candidate_manifest.json', {'sha256': files, 'all_candidates_and_selections_frozen_before_truth': True})
    save_json(out / 'status.json', {'status': 'all_candidates_frozen'})


def evaluate(out):
    check(out)
    verify_manifest(out, 'candidate_manifest.json')
    if any((out / 'evaluations').iterdir()):
        raise FileExistsError('evaluations already exist')
    # Hidden source configuration is used only after all causal candidates freeze.
    from uav_em.environment import ElectromagneticWorld
    from uav_em.risk_diagnostic_evaluation import evaluate_route
    for path in sorted((out / 'candidates').glob('*.json')):
        case, snapshot, cfg = load_case(out, out / 'cases' / path.name)
        world = ElectromagneticWorld(cfg)
        data = read(path)
        for record in data['records']:
            if record['feasible']:
                record['evaluation'] = evaluate_route(world, snapshot, record['path'])
        data.update(diagnosis=summarize(data['records'], data['selection']), candidate_sha256=sha(path),
                    time_seconds=case['time_seconds'])
        save_json(out / 'evaluations' / path.name, data)
        print(case['case'], json.dumps(data['diagnosis'], ensure_ascii=False), flush=True)
    verify_manifest(out, 'candidate_manifest.json')
    save_json(out / 'status.json', {'status': 'evaluated'})


def render(out, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from uav_em.environment import ElectromagneticWorld
    from uav_em.decision_objective import cut_route
    from uav_em.visualization import save_gif
    case, snapshot, cfg = load_case(out, out / 'cases' / (data['case'] + '.json'))
    by = {r['id']: r for r in data['records']}
    choices = [('incumbent', 'Original remaining route', '#111111'),
               (data['diagnosis']['expanded_best_id'], 'Lowest current v3 C (expanded pool)', '#2080c0'),
               (data['diagnosis']['posthoc_true_joint_best_id'], 'Posthoc true T+Q reference ONLY', '#e47916')]
    high = snapshot.geometry.high
    def geometry(ax):
        for building in snapshot.geometry.buildings:
            ax.add_patch(Polygon(building.footprint_m, facecolor='#444444' if building in snapshot.geometry.blocking_buildings else '#aaaaaa', alpha=.55))
        ax.set(xlim=(0, high[0]), ylim=(0, high[1]), aspect='equal', xlabel='x (m)', ylabel='y (m)')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for family, marker in [('incumbent', '*'), ('original_pool', '.'), ('geometric', 'o'), ('rrt_search', '^')]:
        rs = [r for r in data['records'] if r['feasible'] and r['family'] == family]
        axes[0].scatter([r['balance']['score_seconds'] for r in rs], [r['evaluation']['future_joint_seconds'] for r in rs],
                        marker=marker, alpha=.6, label=family)
    geometry(axes[1])
    for i, (identifier, label, color) in enumerate(choices):
        r = by[identifier]
        axes[0].scatter(r['balance']['score_seconds'], r['evaluation']['future_joint_seconds'], s=140, facecolors='none', edgecolors=color)
        axes[1].plot(*np.array(r['path']).T, color=color, ls=('-', '--', ':')[i], lw=2, label=label)
    axes[0].set(xlabel='Current GP v3 C (s)', ylabel='Future true T+Q (s)', title='Finite diagnostic pool; fixed weight 1')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"{data['case']}: open-loop counterfactual, NOT closed-loop / LLM result")
    fig.savefig(out / 'figures' / (data['case'] + '.png'), dpi=135)
    plt.close(fig)
    world = ElectromagneticWorld(cfg)
    duration = max(by[k]['evaluation']['future_flight_seconds'] for k, _, _ in choices)
    offsets = np.linspace(0, duration, 30)
    fields = [world.truth(snapshot.geometry.grid(), snapshot.time_seconds + .5 + t) for t in offsets]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout='constrained')
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('#dddddd')
    image = axes[0].imshow(fields[0], origin='lower', extent=[0, high[0], 0, high[1]], vmin=-100, vmax=-60, cmap=cmap)
    axes[1].imshow(np.ma.masked_where(~snapshot.supported, snapshot.estimate_dbm), origin='lower',
                   extent=[0, high[0], 0, high[1]], vmin=-100, vmax=-60, cmap=cmap)
    fig.colorbar(image, ax=list(axes), label='Interference power (dBm)', shrink=.7)
    markers = []
    for ax in axes:
        geometry(ax)
        group = []
        for i, (identifier, label, color) in enumerate(choices):
            path = np.array(by[identifier]['path'])
            ax.plot(*path.T, color=color, lw=2, ls=('-', '--', ':')[i], label=label)
            group.append(ax.plot(*path[0], marker='o', color=color, ms=7)[0])
        markers.append(group)
        ax.legend(fontsize=6.5, loc='upper right')
    axes[0].set_title('Moving truth: evaluation only')
    axes[1].set_title('Frozen GP; grey = unknown; no new sensing')
    title = fig.suptitle('')
    def update(i):
        image.set_data(fields[i])
        for group in markers:
            for marker, (identifier, _, _) in zip(group, choices):
                prefix, _ = cut_route(by[identifier]['path'], float(offsets[i]) * cfg.aircraft.speed_mps)
                point = prefix[-1]
                marker.set_data([point[0]], [point[1]])
        title.set_text(f"{data['case']} | t={snapshot.time_seconds+.5+offsets[i]:.1f}s | FIXED routes; no replanning\nOrange uses future truth: unavailable online; this is NOT an LLM success")
    update(0)
    fig.savefig(out / 'figures' / (data['case'] + '_replay.png'), dpi=110)
    save_gif(fig, update, list(range(len(offsets))), [150] * (len(offsets)-1) + [1500], out / 'replay' / (data['case'] + '.gif'))
    plt.close(fig)


def report(out):
    protocol = check(out)
    verify_manifest(out, 'candidate_manifest.json')
    paths = sorted((out / 'evaluations').glob('*.json'))
    assert len(paths) == len(list((out / 'cases').glob('*.json')))
    summaries, rows, sections, audits = [], [], [], []
    for path in paths:
        data = read(path)
        assert data['candidate_sha256'] == sha(out / 'candidates' / path.name)
        assert summarize(data['records'], data['selection']) == data['diagnosis']
        case, snapshot, cfg = load_case(out, out / 'cases' / path.name)
        by = {r['id']: r for r in data['records']}
        search_records = [r for r in data['records'] if r['family'] == 'rrt_search']
        for r in search_records:
            assert r['seed'] == paired_seed(case['seed'], case['time_seconds'], 'risk_diagnostic', r['profile'], r['stream_key']['repeat'])
            report = r['report']
            assert report['resources']['samples'] <= 700 and report['resources']['edges'] <= 10000
            assert report['actual_advisor_calls'] == 0 and report['policy_errors'] == 0
        for r in data['records']:
            if r['feasible']:
                assert r['metrics']['length_m'] <= min(case['energy_limit_m'], case['mission_remaining_m']) + 1e-7
                assert all(snapshot.geometry.edge_free(a, b) for a, b in zip(r['path'], r['path'][1:]))
                assert np.allclose(r['path'][0], case['context']['vehicle']['position_m'], atol=1e-8, rtol=0)
                assert np.allclose(r['path'][-1], cfg.map.goal_m, atol=1e-8, rtol=0)
        d = data['diagnosis']
        summary = {'case': data['case'], 'time_seconds': data['time_seconds'], **d,
                   'aggregate_search_resources': data['aggregate_additional_search_resources'], 'choices': {}}
        for role, identifier in [('incumbent', 'incumbent'), ('original_v3', d['original_best_id']),
                                 ('expanded_v3', d['expanded_best_id']), ('posthoc_true_joint', d['posthoc_true_joint_best_id'])]:
            r = by[identifier]
            summary['choices'][role] = {'id': identifier, 'length_m': r['metrics']['length_m'], 'balance': r['balance'], 'evaluation': r['evaluation']}
            e = r['evaluation']
            rows.append(f"|{data['case']}|{role}|{r['metrics']['length_m']:.2f}|{r['balance']['score_seconds']:.3f}|{e['future_power_seconds']:.3f}|{e['future_joint_seconds']:.3f}|{e['future_excess_db_seconds']:.3f}|{e['prefix10']['excess_db_seconds']:.3f}|")
        summaries.append(summary)
        audits.append({'case': data['case'], 'geometry_caps_endpoints_verified': True, 'seeds_and_search_limits_verified': True,
                       'model_calls': 0, 'policy_errors': 0, 'candidate_sha256': data['candidate_sha256']})
        render(out, data)
        sections.append(f'<h2>{html.escape(data["case"])}</h2><p>候选{d["candidate_count"]}，可行{d["feasible_count"]}；'
                        f'真实联合分/暴露共同改善候选{len(d["truth_opportunity_ids"])}，其中当前v3可接受{len(d["v3_acceptable_truth_opportunity_ids"])}。'
                        f'扩大池降低当前C：{d["expanded_pool_lowers_current_score"]}；其选择改善实际联合分：{d["expanded_selection_improves_true_joint"]}。</p>'
                        f'<p><a href="replay/{data["case"]}.gif">开环反事实GIF</a> · <a href="evaluations/{path.name}">全候选证据</a></p>'
                        f'<img src="figures/{data["case"]}.png" width="100%"><img src="figures/{data["case"]}_replay.png" width="100%">')
    totals = {'states': len(summaries), 'candidates': sum(s['candidate_count'] for s in summaries),
              'feasible_candidates': sum(s['feasible_count'] for s in summaries),
              'independent_search_attempts': sum(s['aggregate_search_resources']['independent_attempts'] for s in summaries),
              'new_model_calls': 0, 'new_flights': 0, 'new_generated_submissions': 0}
    for key in ('samples', 'edges', 'numeric_compute_seconds', 'actual_rrt_trees'):
        totals[key] = sum(s['aggregate_search_resources'][key] for s in summaries)
    save_json(out / 'reports/summary.json', {'totals': totals, 'states': summaries})
    save_json(out / 'reports/audit.json', {'states': audits, 'core_sources_unchanged_since_prepare': True,
                                         'candidate_manifest_verified': True, 'protocol_sha256': sha(out / 'protocol.json')})
    md = '# dev617风险证据状态诊断\n\n4个既有因果状态；无新飞行/模型。扩大候选池成本高于一次在线决策，只诊断覆盖与评分偏差。\n\n'
    md += '固定路线开环反事实：决策后0.5s出发，≤0.25s积分，无新点测/重规划，不含悬停。橙色为事后真值联合分参考，不能给LLM。\n\n'
    md += '|状态|角色|长度/m|当前C/s|真实Q/s|真实T+Q/s|超参考/dB·s|前10s超参考/dB·s|\n|---|---|---:|---:|---:|---:|---:|---:|\n' + '\n'.join(rows) + '\n'
    (out / 'RESULTS.md').write_text(md)
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>风险状态修复诊断</title><style>body{font:17px/1.8 sans-serif;max-width:1250px;margin:25px auto;padding:15px}img{max-width:100%}</style><h1>dev617风险状态：候选与评分诊断</h1><p>无新模型/飞行；固定主权重1，GP与传播不变。累计离线诊断池不是一次在线预算，不宣称LLM改善。</p><p><a href="RESULTS.md">数值表</a> · <a href="reports/summary.json">完整总结</a> · <a href="replay/index.html">4个开环反事实GIF</a> · <a href="reports/frozen_design.md">实验前设计</a></p><p><b>橙色依据未来真值事后选择，不可给模型，也不是在线策略。GIF不更新观测、不重规划。</b></p>' + ''.join(sections))
    (out / 'replay/index.html').write_text('<!doctype html><meta charset="utf-8"><title>开环诊断回放</title><style>body{font:17px/1.8 sans-serif;max-width:1250px;margin:25px auto}img{max-width:100%}</style><h1>固定路线反事实：无新点测/重规划</h1><p>橙色用未来真值选路，不是LLM结果。黑色原路线，蓝色扩大池当前v3最佳。<a href="../index.html">完整报告</a></p>' + ''.join(f'<h2>{s["case"]}</h2><img src="{s["case"]}.gif">' for s in summaries))
    save_json(out / 'status.json', {'status': 'completed', **totals})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'capture', 'propose', 'evaluate', 'report', 'all'])
    parser.add_argument('--out', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.phase == 'all':
        for fn in (prepare, capture, propose, evaluate, report):
            fn(args.out)
    else:
        globals()[args.phase](args.out)


if __name__ == '__main__':
    main()
