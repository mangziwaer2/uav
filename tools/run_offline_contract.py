"""冻结离线生成合同，验证六个因果快照的共同执行器；无新飞行/模型。"""
from __future__ import annotations

import argparse
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
from uav_em.decision_objective import prefix_risk, select_candidate
from uav_em.environment import Geometry, Measurement
from uav_em.offline_process import OfflineProcessOptimizer, generation_contract, reference_program
from uav_em.repair_process import canonical_path
from uav_em.repair_search import FixedObjective, WorkBudget
from run_repair_diagnostics import load_case

SOURCE = ROOT / 'outputs/interference_evidence_budget_20261111'
DEFAULT = ROOT / 'outputs/interference_offline_contract_20261111'
METHODS = ('pool_only', 'fixed_process', 'adaptive_process', 'submitted_process')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('cases', 'scenarios', 'paired', 'public', 'reports/runtime_source', 'logs'):
        (out / folder).mkdir(parents=True)
    save_json(out / 'public/contract.json', generation_contract())
    save_json(out / 'public/reference_fixed.json', reference_program(False))
    save_json(out / 'public/reference_adaptive.json', reference_program(True))
    shutil.copy2(ROOT / 'docs/offline_process_contract.md', out / 'reports/frozen_design.md')
    inputs = {}
    for seed in (611, 613, 617, 619):
        src = SOURCE / 'scenarios' / f'dev{seed}.json'
        dest = out / 'scenarios' / src.name
        shutil.copy2(src, dest)
        inputs[str(dest.relative_to(out))] = sha(dest)
        cfg = load_config(dest)
        geometry = Geometry(cfg.map, cfg.aircraft.clearance_m, cfg.aircraft.altitude_m, cfg.aircraft.vertical_clearance_m)
        belief = ElectromagneticBelief(cfg, geometry)
        ep_path = SOURCE / 'runs' / f'pool_j_dev{seed}_seed{seed}' / 'episode.json'
        episode = read(ep_path)
        selected = [episode['decisions'][0]]
        if seed in (617, 619):
            selected.append(next(d for d in episode['decisions']
                                 if d['repair_optimization']['process_selection']['evidence']['active']))
        points = [e for e in episode['events'] if e['event'] == 'observation']
        index = 0
        for decision in selected:
            now = decision['time_seconds']
            while index < len(points) and points[index]['time_seconds'] <= now + 1e-9:
                p = points[index]
                belief.update(Measurement(tuple(p['position_m']), p['time_seconds'], p['interference_dbm']))
                index += 1
            snapshot = belief.snapshot(now)
            old = decision['repair_optimization']
            old_choice = next(r for r in old['candidates'] if r['id'] == old['selected_candidate_id'])
            score = prefix_risk(FixedObjective(snapshot, -80, WorkBudget()), old_choice['path'], cfg.aircraft.speed_mps)
            for key in score:
                assert score[key] == old_choice['prefix'][key] or isinstance(score[key], float) and abs(score[key] - old_choice['prefix'][key]) < 1e-8
            route = [] if now == 0 else next(r['path'] for r in old['candidates'] if r['id'] == 'incumbent')
            energy = decision['execution_checkpoint']['remaining_energy_wh']
            maximum = max(0., (energy - cfg.aircraft.reserve_energy_wh -
                              cfg.simulation.decision_delay_seconds * cfg.aircraft.hover_power_w / 3600) *
                          3600 * cfg.aircraft.speed_mps / cfg.aircraft.flight_power_w)
            case = {'scene': f'dev{seed}', 'time_seconds': now, 'context': decision['context'], 'route': route,
                    'seed': cfg.seed + sum(p['time_seconds'] < now - 1e-9 for p in episode['plans']),
                    'max_length_m': maximum, 'observation_count': snapshot.observation_count,
                    'estimation_details': snapshot.estimation_details,
                    'reference_length_m': old['mission_reference_length_m'], 'flown_m': old['executed_distance_m'],
                    'source_episode': str(ep_path.relative_to(ROOT)), 'source_episode_sha256': sha(ep_path),
                    'causal_reconstruction_matches_original_selected_risk': True}
            name = f'dev{seed}_t{now:g}'
            save_json(out / 'cases' / f'{name}.json', case)
            arrays = {key: getattr(snapshot, key) for key in
                      ('estimate_dbm', 'uncertainty_db', 'supported', 'age_seconds',
                       'nearest_support_distance_m', 'local_support_count')}
            np.savez_compressed(out / 'cases' / f'{name}.npz', **arrays)
    for p in sorted((out / 'cases').iterdir()):
        inputs[str(p.relative_to(out))] = sha(p)
    sources = {}
    for p in [*(ROOT / 'src/uav_em').glob('*.py'), Path(__file__).resolve(),
              ROOT / 'tools/run_repair_diagnostics.py', ROOT / 'tests/test_offline_process.py']:
        rel = p.relative_to(ROOT)
        dest = out / 'reports/runtime_source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        sources[str(rel)] = sha(p)
    save_json(out / 'protocol.json', {'scope': 'six causal states x four methods; interface verification only',
        'input_sha256': inputs, 'source_sha256': sources, 'model_calls': 0, 'new_flights': 0,
        'public_sha256': {p.name: sha(p) for p in (out / 'public').iterdir()},
        'design_sha256': sha(out / 'reports/frozen_design.md'),
        'cases': sorted(p.stem for p in (out / 'cases').glob('*.json')),
        'methods': list(METHODS), 'generated_artifacts': 0,
        'validation_test_generated_or_read': False,
        'note': 'submitted_process is the human adaptive reference for identity testing, not an LLM artifact'})


def check(out):
    protocol = read(out / 'protocol.json')
    assert sha(out / 'reports/frozen_design.md') == protocol['design_sha256']
    for rel, h in protocol['input_sha256'].items():
        assert sha(out / rel) == h, rel
    for rel, h in protocol['source_sha256'].items():
        assert sha(ROOT / rel) == sha(out / 'reports/runtime_source' / rel) == h, rel
    for name, h in protocol['public_sha256'].items():
        assert sha(out / 'public' / name) == h, name
    return protocol


def verify(out):
    protocol = check(out)
    if any((out / 'paired').iterdir()):
        raise FileExistsError('refuse to overwrite paired verification')
    checks = []
    for name in protocol['cases']:
        case, snapshot, _ = load_case(out, out / 'cases' / f'{name}.json')
        results = []
        for method in METHODS:
            program = reference_program() if method == 'submitted_process' else None
            optimizer = OfflineProcessOptimizer(method, program)
            optimizer.base.reference_length_m = case['reference_length_m']
            interval = case['context'].get('execution_feedback', {}).get('interval')
            optimizer.base.flown_m = case['flown_m'] - (interval['flown_m'] if interval else 0)
            plan, report = optimizer.optimize(snapshot, deepcopy(case['context']), case['route'], case['max_length_m'], case['seed'], 0)
            assert report['executed_distance_m'] == case['flown_m']
            assert report['resources']['samples'] <= 2100 and report['resources']['edges'] <= 10000
            assert plan.path and report['final_geometry_verified'] and not report['policy_errors']
            assert plan.length_m + case['flown_m'] <= report['mission_distance_cap_m'] + 1e-7
            assert plan.length_m <= case['max_length_m'] + 1e-7
            assert all(snapshot.geometry.edge_free(a, b) for a, b in zip(plan.path, plan.path[1:]))
            assert np.allclose(plan.path[0], case['context']['vehicle']['position_m'])
            assert np.allclose(plan.path[-1], case['context']['vehicle']['goal_m'])
            selected = select_candidate(report['candidates'], 'prefix_v2')
            assert canonical_path(selected['path']) == canonical_path(plan.path)
            recomputed = prefix_risk(FixedObjective(snapshot, -80, WorkBudget()), plan.path,
                                     case['context']['vehicle']['speed_mps'])
            for key in recomputed:
                assert recomputed[key] == selected['prefix'][key] or isinstance(recomputed[key], float) and abs(recomputed[key] - selected['prefix'][key]) < 1e-8
            save_json(out / 'paired' / f'{name}_{method}.json', {'case': name, 'method': method,
                      'path': plan.path, 'report': report})
            results.append((plan, report))
            print(name, method, 'samples', report['resources']['samples'], 'edges', report['resources']['edges'], flush=True)
        initial = results[0][1]
        n = len(initial['candidates'])
        assert all(r['candidates'][:n] == initial['candidates'] for _, r in results)
        assert all(r['common_pool_resources'] == initial['common_pool_resources'] for _, r in results)
        assert results[2][0].path == results[3][0].path
        assert results[2][1]['resources'] == results[3][1]['resources']
        traces = results[2][1]['process_trace']
        if traces:
            save_json(out / 'public' / f'{name}_state.json', traces[0]['state'])
        checks.append({'case': name, 'identical_initial_pool_and_work': True,
                       'submitted_reference_reproduces_numeric': True, 'causal_risk_recomputed': True,
                       'geometry_energy_lifetime_distance_and_shared_budgets_pass': True})
    manifest = {str(p.relative_to(out)): sha(p) for p in sorted((out / 'paired').iterdir())}
    save_json(out / 'reports/paired_manifest.json', manifest)
    save_json(out / 'reports/verification.json', {'checks': checks, 'paired_tasks': len(manifest),
              'model_calls': 0, 'generated_artifacts': 0, 'new_flights': 0, 'new_gifs': 0,
              'benefit_evaluated': False, 'generation_contract_frozen': True})
    save_json(out / 'status.json', {'status': 'interface_verified', 'model_calls': 0, 'new_flights': 0})
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>离线过程生成接口核验</title>'
        '<h1>离线有限过程接口：24项同状态核验</h1><p>6个真实点测重建的因果快照×4方法；'
        'submitted_process使用人工参考检查接口等价，不是LLM生成结果。0新飞行、0模型调用，无新GIF。</p>'
        '<p><a href="public/contract.json">冻结生成合同</a> · <a href="reports/verification.json">核验结果</a> · '
        '<a href="../../docs/offline_process_contract.md">接口、评价与生成预算说明</a></p>'
        '<p>四方法初态候选及工作一致，提交的人工参考过程重现数值自适应过程；几何、能量、固定全程航程、'
        '共享搜索预算与因果风险复算通过。本轮不评价任务收益。下一阶段最多4份离线过程、16次生成过程开发飞行。</p>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'verify', 'all'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.action in ('prepare', 'all'):
        prepare(args.output)
    if args.action in ('verify', 'all'):
        verify(args.output)


if __name__ == '__main__':
    main()
