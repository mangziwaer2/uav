"""D2-M0 source-only mechanism pilot; sole entry: run_experiment.py source-only."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .config import load_config
from .environment import ElectromagneticWorld
from .event_demo import read, save, sha
from .event_inference import digest
from .equivalent_field import normalized_expected_power
from .joint_experiment import (audit_before_truth, check, display_predictions,
                               numerical_samples)
from .joint_prediction import JointSettings
from .point_gp import FrozenGP, training_points
from .scenario_loop import run as fly
from .scenario_planning import ScenarioForecast, score_route
from .source_revision import RevisionSettings

ROOT = Path(__file__).resolve().parents[2]
D1 = ROOT / 'outputs/interference_joint_gp_d1_20261119_0940'
METHODS = ('source_early', 'source_revise')


def original_check(protocol):
    for rel, expected in protocol['original_d1_sha256'].items():
        if sha(D1 / rel) != expected:
            raise ValueError('D1 original changed: '+rel)
    for rel, expected in protocol['original_c1_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('C1 original changed: '+rel)


def prepare(out):
    out.mkdir(exist_ok=False)
    for name in ('runs', 'public', 'reports/frozen_source', 'predictions', 'evaluation', 'replay'):
        (out / name).mkdir(parents=True)
    shutil.copy2(ROOT / 'configs/local_power.json', out / 'scenario.json')
    sources = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'tests').rglob('*.py'))
    sources += [ROOT / 'requirements.txt', ROOT / 'docs/source_only_pilot_v1.md']
    hashes = {}
    for source in sources:
        rel = source.relative_to(ROOT)
        target = out / 'reports/frozen_source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(rel)] = sha(source)
    c1 = ROOT / 'outputs/interference_scenario_loop_20261118'
    protocol = dict(version='D2-M0', methods=list(METHODS), source_sha256=hashes,
        scenario_sha256=sha(out / 'scenario.json'), public_contract_sha256={},
        joint_settings=asdict(JointSettings()),
        revision_settings={m: asdict(RevisionSettings(enable_mismatch=m == 'source_revise')) for m in METHODS},
        max_flights=2, max_events_per_flight=4, max_actual_model_calls=0, max_bridge_requests=0,
        max_sensor_points=204, max_jobs=24, max_solver_calls=48, max_residual_calls=1920,
        max_samples=7000, max_edges=100000, max_monitor_receiver_queries=4000,
        max_soft_gp_queries=360, max_prequential_queries=202, max_planner_receiver_queries=40000000,
        max_evaluation_points=1000, max_replay_truth_points=21504,
        max_display_receiver_queries=21504, max_shadow_receiver_queries=1200,
        max_frames_per_flight=32, display_shape=[16, 21],
        original_d1_sha256=read(D1 / 'reports/completion_manifest_20261120.json'),
        original_c1_sha256={str(p.relative_to(ROOT)): sha(p) for p in c1.rglob('*') if p.is_file()},
        interpretation='two numerical mechanism flights; no new decision LLM; no causal gain claim against D1',
        timing='source initialization + supported warning; optional causal mismatch revision; fixed 0.5s hover')
    original_check(protocol)
    save(out / 'protocol.json', protocol)
    cache = ROOT / 'docs/references/hypothesis_refinement_20261119'
    save(out / 'reports/literature_reading.json', dict(
        type='selected existing full-text passages; not a new web retrieval or complete review',
        sources=[dict(title='Bayesian Online Changepoint Detection', url='https://arxiv.org/abs/0710.3742',
            original=str((cache / 'changepoint_html.raw').relative_to(ROOT)),
            original_sha256=sha(cache / 'changepoint_html.raw'),
            text_sha256=sha(cache / 'changepoint_html.txt'),
            scope='Abstract, introduction, sections 2 and 2.1; selected text only',
            use='causal predictive monitoring; no BOCPD implementation or claimed calibrated posterior',
            limitation='piecewise iid/hazard assumptions do not directly model a moving radio receiver'),
        dict(title='Literature Meets Data', url='https://arxiv.org/abs/2410.17309',
            original_sha256=sha(cache / 'hypothesis_literature_data.html'),
            text_sha256=sha(cache / 'hypothesis_literature_data.txt'),
            scope='sections 2.2 and 2.3 selected passages on wrongly predicted samples and refinement',
            use='feed prediction failures into bounded source revision',
            limitation='classification task; its rewards and measured gains do not establish UAV performance')]))
    print('PREPARED', out, flush=True)


def shadow(out, protocol):
    """Matched D1 prefixes: same parameters, conditional source weights, no truth."""
    episode = read(D1 / 'runs/joint_N/episode.json')
    cfg = load_config(out / 'scenario.json')
    world = ElectromagneticWorld(cfg)
    records, receiver_queries = [], 0
    xy = np.array([[x, y] for y in np.linspace(0, 600, 7) for x in np.linspace(0, 800, 9)])
    for t in (23.5, 27.5):
        frame = next(f for f in episode['frames'] if f['time_seconds'] == t and f['phase'] in ('after_decision', 'before_decision'))
        observed = frame['observation_time_seconds']
        prefix = [p for p in episode['observations'] if p['time_seconds'] <= observed]
        source = deepcopy([m for m in frame['hypotheses'] if m['bases']])
        mass = sum(m['weight'] for m in source)
        assert mass > 0
        for m in source:
            m['weight'] /= mass
            m['log_weight'] = math.log(m['weight']) if m['weight'] else -1e300
        gp = FrozenGP(training_points(prefix, world.geometry, observed, cfg.belief), world.geometry,
                      observed, cfg.belief, cfg.electromagnetic.background_dbm)
        record = dict(frame_time_seconds=t, observation_time_seconds=observed,
                      source_mass_before_conditioning=mass, variants={})
        for name, models in (('D1_saved_mixed_bank', frame['hypotheses']), ('source_conditional_bank', source)):
            f = ScenarioForecast(prefix, models, observed, gp, gp_version=digest(prefix), joint_settings=JointSettings())
            prediction = f.predict(xy, np.full(len(xy), t), 'joint')
            powers = prediction['weights'] @ prediction['expected_normalized_power']
            score = score_route(f, frame['route_m'], cfg.aircraft.speed_mps, method='joint')
            record['variants'][name] = dict(forecast_sha256=f.fingerprint,
                expected_power_dbm=(-80+10*np.log10(powers)).tolist(),
                supported=prediction['supported'].tolist(), route_score=score,
                relative_source_weights={m['id']: m['weight'] for m in models if m['bases']})
            receiver_queries += f.counts['receiver_queries']
        records.append(record)
    assert receiver_queries <= protocol['max_shadow_receiver_queries']
    save(out / 'reports/same_prefix_shadow.json', dict(records=records, receiver_queries=receiver_queries,
        position_m=xy.tolist(), new_fits=0, truth_queries=0,
        limitation='conditional representation and predicted score only; not evaluated route benefit'))


def run_one(out, method):
    protocol = check(out)
    original_check(protocol)
    if method not in protocol['methods']:
        raise ValueError('undeclared method')
    folder = out / 'runs' / method
    folder.mkdir(exist_ok=False)
    def emit(row):
        with (folder / 'decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        print('DECISION', method, row['index'], row['time_seconds'],
              row['public']['consultation_event']['reasons'], row['selected_id'], flush=True)
    started = time.monotonic()
    try:
        ep = fly(load_config(out / 'scenario.json'), 'mixture_numeric', timing='source_revision',
            forecast_method='joint', joint_settings=JointSettings(**protocol['joint_settings']),
            soft_fit=True, revision_settings=RevisionSettings(**protocol['revision_settings'][method]), emit=emit)
        ep['study_method'], ep['wall_seconds'] = method, time.monotonic()-started
        save(folder / 'episode.json', ep)
        save(folder / 'status.json', dict(status='completed', episode_sha256=sha(folder / 'episode.json')))
        print('COMPLETE', method, ep['metrics'], flush=True)
    except BaseException as exc:
        save(folder / 'failure.json', dict(error=repr(exc)))
        raise


def audit_sources(episodes, cfg):
    from .event_inference import HypothesisBank
    from .source_revision import SourceOnlyBank
    checks = []
    for method, ep in episodes.items():
        bank = SourceOnlyBank(ep['observations'][0], cfg.electromagnetic.background_dbm,
                              cfg.belief.unknown_error_db)
        births = {e['time_seconds']: e for e in ep['inferences']}
        rows = iter(ep['prediction_records'])
        for point in ep['observations']:
            if point is not ep['observations'][0]:
                issued = bank.issue(point['position_m'], point['time_seconds'])
                old = next(rows)
                assert issued == old['issued']
                result = bank.observe(point)
                assert result == old
            if point['time_seconds'] in births:
                event = births[point['time_seconds']]
                assert bank.snapshot() == event['bank_before_birth']
                bank.birth(event['committed_models'])
                assert bank.snapshot() == event['bank_after_birth']
        assert bank.snapshot() == ep['final_bank']
        for frame in ep['frames']:
            assert not any(m['family'] == 'persistence' for m in frame['hypotheses'])
            assert all(m['bases'] for m in frame['hypotheses'] if m['family'] != 'unknown_prior')
            assert not (any(m['bases'] for m in frame['hypotheses']) and
                        any(m['family'] == 'unknown_prior' for m in frame['hypotheses']))
        for event, decision in zip(ep['inferences'], ep['decisions']):
            assert event['time_seconds'] == decision['time_seconds']
            assert event['bank_after_birth'] == decision['public']['request']['models']
        checks.append(dict(method=method, observations=len(ep['observations']),
            reissued_predictions_match=True, no_spatial_persistence=True,
            every_fit_has_same_event_route_reassessment=True,
            source_weight_meaning='conditional relative evidence, not model adequacy'))
    return checks


def finalize(out):
    protocol = check(out)
    original_check(protocol)
    if any((out / 'evaluation').iterdir()) or any((out / 'predictions').iterdir()):
        raise ValueError('refuse duplicate scientific evaluation')
    episodes = {m: read(out / 'runs' / m / 'episode.json') for m in protocol['methods']}
    cfg = load_config(out / 'scenario.json')
    audit = audit_before_truth(out, protocol, episodes)
    audit['source_only_checks'] = audit_sources(episodes, cfg)
    save(out / 'reports/pre_truth_audit.json', audit)
    save(out / 'reports/flight_freeze.json', {str(p.relative_to(out)): sha(p) for p in (out / 'runs').rglob('*') if p.is_file()})
    shadow(out, protocol)
    world = ElectromagneticWorld(cfg)
    displays, count = {}, 0
    for method, ep in episodes.items():
        xy, arrays, metadata, queries = display_predictions(ep, cfg, world.geometry, JointSettings(**protocol['joint_settings']))
        # Unknown remains scored with its declared broad prior, but cannot be
        # displayed as an observed low/high-power spatial map.
        for k, index in enumerate(arrays['indices']):
            if not any(m['bases'] for m in ep['frames'][int(index)]['hypotheses']):
                arrays['actual_dbm'][k] = np.where(arrays['supported'][k], arrays['actual_dbm'][k], np.nan)
                metadata[k]['unsupported'] = 'unknown; explicit broad prior used in route cost, not observed spatial power'
        np.savez_compressed(out / 'predictions' / f'{method}.npz', **arrays)
        save(out / 'predictions' / f'{method}.json', metadata)
        displays[method] = (xy, arrays)
        count += queries
    assert count <= protocol['max_display_receiver_queries']
    save(out / 'reports/prediction_freeze.json', {str(p.relative_to(out)): sha(p) for p in (out / 'predictions').iterdir()})
    # First post-flight truth access occurs only after every prediction is frozen.
    results, truth_count, grid_count = [], 0, 0
    for method, ep in episodes.items():
        samples = numerical_samples(ep, cfg.aircraft.speed_mps)
        for s in samples:
            s['truth_dbm'] = float(world.truth(s['position_m'], s['time_seconds']))
        truth_count += len(samples)
        assert truth_count <= protocol['max_evaluation_points']
        q = math.fsum(s['dt_seconds'] * 10**((s['truth_dbm']+80)/10) for s in samples)
        rows = ep['joint_prediction_records']
        errors = [r['residual_db'] for r in rows]
        source_errors = [r['squared_error'] for r in ep['prediction_records']
                         if any(p['id'] != 'uninitialized_unknown' for p in r['issued']['predictions'])]
        result = dict(method=method, **ep['metrics'], Q=q,
            mission_T_plus_Q=ep['metrics']['simulation_seconds']+q,
            joint_prequential_rmse_db=math.sqrt(float(np.mean(np.square(errors)))),
            source_prequential_rmse_db=math.sqrt(float(np.mean(source_errors))) if source_errors else None,
            events=[dict(time_seconds=d['time_seconds'], reasons=d['public']['consultation_event']['reasons'],
                         selected_id=d['selected_id']) for d in ep['decisions']],
            mismatch_points=sum(r['all_source_mismatch'] for r in ep['trigger_records']),
            mismatch_budget_blocked=sum(r['all_source_mismatch'] and r['blocker'] == 'request_budget_exhausted'
                                        for r in ep['trigger_records']),
            actual_llm_calls=0)
        save(out / 'evaluation' / f'{method}.json', dict(metrics=result, samples=samples))
        xy, arrays = displays[method]
        truth = []
        for index in arrays['indices']:
            truth.append(world.truth(xy, ep['frames'][int(index)]['time_seconds']).reshape(16, 21))
            grid_count += len(xy)
        assert grid_count <= protocol['max_replay_truth_points']
        np.savez_compressed(out / 'evaluation' / f'{method}_grids.npz', truth_dbm=np.asarray(truth))
        results.append(result)
    save(out / 'reports/results.json', dict(rows=results, evaluation_truth_points=truth_count,
         replay_truth_points=grid_count, display_receiver_queries=count,
         interpretation=protocol['interpretation']))
    render(out, cfg, episodes, results)
    original_check(protocol)
    save(out / 'reports/completion_manifest.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


def render(out, cfg, episodes, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    from PIL import Image
    high = [(cfg.map.columns-1)*cfg.map.resolution_m, (cfg.map.rows-1)*cfg.map.resolution_m]
    for result in results:
        method, images = result['method'], []
        ep = episodes[method]
        arrays = np.load(out / 'predictions' / f'{method}.npz')
        truth = np.load(out / 'evaluation' / f'{method}_grids.npz')['truth_dbm']
        aliases = {m['id']: f'H{i+1}' for i, m in enumerate(ep['final_bank']) if m['bases']}
        for k, index in enumerate(arrays['indices']):
            f = ep['frames'][int(index)]
            now = f['time_seconds']
            fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=100)
            fig.subplots_adjust(top=.82, bottom=.30, left=.055, right=.90, wspace=.13)
            cmap = plt.get_cmap('YlOrRd').copy(); cmap.set_bad('#b9bdc2')
            trail = np.asarray([p['position_m'] for p in ep['observations'] if p['time_seconds'] <= now])
            for side, (ax, field) in enumerate(zip(axes, (truth[k], arrays['actual_dbm'][k]))):
                im = ax.imshow(np.ma.masked_invalid(field), origin='lower', extent=(0, high[0], 0, high[1]),
                               vmin=-100, vmax=-55, cmap=cmap)
                for b in cfg.map.buildings:
                    ax.add_patch(Polygon(b['footprint_m'], color='#303030' if b['height_m']+cfg.aircraft.vertical_clearance_m >= cfg.aircraft.altitude_m else '#abc4cf', zorder=3))
                ax.plot(*trail.T, color='#125ec9', lw=2, zorder=4)
                if f['route_m']:
                    ax.plot(*np.asarray(f['route_m']).T, '--', color='#007c91', lw=1.5, zorder=4)
                ax.plot(*f['position_m'], 'ko', ms=5, zorder=7)
                ax.plot(*cfg.map.goal_m, 'b*', ms=11, zorder=7)
                for d in ep['decisions']:
                    if d['time_seconds'] <= now:
                        ax.plot(*d['public']['request']['state']['position_m'], 'o', mfc='none', mec='#a000b0', ms=8, zorder=6)
                ax.set(xlim=(0, high[0]), ylim=(0, high[1]), xlabel='x (m)', ylabel='y (m)')
                ax.set_title('True field + active true sources (evaluation only)' if side == 0 else 'Actual planner field + ONE source hypothesis', fontsize=11)
                if side == 0:
                    for number, field_source in enumerate(cfg.electromagnetic.fields, 1):
                        if field_source['start_seconds'] <= now < field_source['end_seconds']:
                            center = np.asarray(field_source['center_m'])+(now-field_source['start_seconds'])*np.asarray(field_source['velocity_mps'])
                            ax.plot(*center, '^', mfc='black', mec='white', ms=10, zorder=8)
                            ax.annotate(f'T{number}', center, xytext=(7, -14), textcoords='offset points', zorder=9,
                                        bbox=dict(fc='white', ec='none', alpha=.85, pad=1))
                elif np.any(arrays['supported'][k]) and not np.all(arrays['supported'][k]):
                    ax.contour(np.linspace(0, high[0], 21), np.linspace(0, high[1], 16), arrays['supported'][k], levels=[.5], colors=['cyan'], linewidths=1)
            sources = sorted([m for m in f['hypotheses'] if m['bases']], key=lambda m: (-m['weight'], m['id']))
            selected = sources[0] if sources else None
            info = 'No source model yet. Gray = unsupported unknown, not measured low power.'
            detail = ''
            if selected:
                h, dt = aliases[selected['id']], now-selected['reference_time_seconds']
                info = f"{h} = {selected['id']} | {selected['family']} | conditional weight={selected['weight']:.4e} | scored={selected['scored_points']}"
                descriptions = []
                for number, (b, peak) in enumerate(zip(selected['bases'], selected['peaks_dbm']), 1):
                    center = np.asarray(b['center_m'])+dt*np.asarray(b['velocity_mps'])
                    axes[1].add_patch(Circle(center, b['scale_m'], fill=False, ec='#a000ab', ls=':', lw=1.2, zorder=6))
                    axes[1].plot(*center, 'X', color='#a000ab', mec='white', ms=10, zorder=8)
                    axes[1].annotate(f'{h}/B{number}', center, xytext=(7, 8+13*(number-1)), textcoords='offset points',
                        zorder=9, color='#790482', bbox=dict(fc='white', ec='none', alpha=.85, pad=1), annotation_clip=True)
                    descriptions.append(f"B{number}: ({center[0]:.1f},{center[1]:.1f})m, scale={b['scale_m']:g}m, peak={peak+dt*b['rate_db_per_second']:.1f}dBm")
                detail = ' | '.join(descriptions)
            last = [d for d in ep['decisions'] if d['time_seconds'] <= now]
            event = f"last event={last[-1]['time_seconds']:g}s: {', '.join(last[-1]['public']['consultation_event']['reasons'])}" if last else 'before source initialization'
            fig.text(.055, .95, f"D2-M0 | {method} | t={now:.2f}s | {f['phase']} | numerical organization, 0 LLM calls", fontsize=13)
            fig.text(.055, .91, event, fontsize=11)
            fig.text(.055, .87, f"True whole-flight T+Q={result['mission_T_plus_Q']:.2f} | length={result['executed_length_m']:.1f}m | source bank has NO power persistence", fontsize=11)
            fig.text(.055, .235, 'Blue: flown; teal dashed: committed route; purple rings: consultations; cyan: GP support boundary.', fontsize=10)
            fig.text(.055, .20, 'Purple X: one alternative world; dotted radius: Gaussian scale. True sources are never provided to inference.', fontsize=10)
            fig.text(.055, .165, info, fontsize=10, color='#7b1879')
            fig.text(.055, .13, detail, fontsize=9, color='#7b1879')
            weights = ', '.join(f"{aliases[m['id']]}:{m['weight']:.3e}" for m in sources[:5])
            fig.text(.055, .095, 'Conditional relative weights (not absolute trust): '+weights, fontsize=10)
            fig.text(.055, .05, 'Single seen development scene. Synchronous callbacks; fixed 0.5s hover. Removing persistence does not establish model correctness.', fontsize=9)
            fig.colorbar(im, cax=fig.add_axes([.925, .34, .012, .45]), label='Power (dBm; higher = stronger)')
            fig.canvas.draw()
            images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
            if f['phase'] == 'after_decision':
                fig.savefig(out / 'replay' / f'{method}_event{len(last)}.png', dpi=100)
            plt.close(fig)
        images[0].save(out / 'replay' / f'{method}.gif', save_all=True, append_images=images[1:], duration=400, loop=0)
        with Image.open(out / 'replay' / f'{method}.gif') as image:
            assert image.n_frames == len(arrays['indices'])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>D2-M0：移除全图功率延续</title>
<style>body{font:17px/1.7 system-ui;max-width:1450px;margin:25px auto;padding:15px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:7px}</style>
<h1>D2-M0：移除全图功率延续，检查源假设与失配修订</h1>
<p>两条新的数值机制飞行，0次决策LLM调用。源池不再包含最近点测的全图延续；8秒开始拟合。source_revise额外允许全部源版本在新点测上均失配时提前修订，并在该次事件复核未飞路线。共享预算最多4事件，包括初始化。</p>
<p>左图真场和活动真源仅供事后评价；右图为该版本实际规划聚合场，紫叉只表示一个替代源解释，虚圆是高斯尺度。权重是当前源集合内的相对证据，不代表集合正确。初始灰色为未知，规划代价使用声明的宽先验，未当零暴露。</p>
<p><a href="../protocol.json">冻结合同与预算</a> · <a href="../reports/results.json">结果</a> · <a href="../reports/pre_truth_audit.json">因果/预算核验</a> · <a href="../reports/same_prefix_shadow.json">D1同前缀表示检查</a> · <a href="../../../docs/source_only_pilot_v1.md">方法合同</a></p>
<table><tr><th>方法</th><th>到达</th><th>航程m</th><th>T</th><th>真实Q</th><th>T+Q</th><th>源一步RMSE</th><th>事件数</th></tr>'''
    for r in results:
        page += f"<tr><td>{r['method']}</td><td>{r['success']}</td><td>{r['executed_length_m']:.2f}</td><td>{r['simulation_seconds']:.2f}</td><td>{r['Q']:.2f}</td><td>{r['mission_T_plus_Q']:.2f}</td><td>{r['source_prequential_rmse_db']:.2f}</td><td>{r['decision_count']}</td></tr>"
    page += '</table><p>单个已见开发场景；数值搜索仍为原型，不是论文强基线。D1仅作历史描述，不把这轮变化称为LLM收益，也不以热点更清楚证明预测更准。</p>'
    for r in results:
        method = r['method']
        page += f'<h2>{method}</h2><p><a href="{method}.gif">打开GIF</a> · <a href="../runs/{method}/episode.json">原始飞行记录</a></p>'
        page += '<p>'+'；'.join(f"{e['time_seconds']:g}s：{', '.join(e['reasons'])} → {e['selected_id']}" for e in r['events'])+'</p>'
        page += f'<img src="{method}.gif" loading="lazy">'
    (out / 'replay/index.html').write_text(page+'</html>')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><a href="replay/index.html">D2-M0新飞行与源假设回放</a>')
