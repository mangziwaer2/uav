"""Render existing navigation episodes into one replay gallery; never rerun simulation."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

LABELS = {'numeric_rrt': '数值RRT*', 'numeric_astar': '数值A*', 'online_agent': '在线决策agent', 'repair_numeric': '数值修复',
          'repair_single': '单轮LLM修复', 'repair_feedback': '多轮反馈LLM修复'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def gallery(output, entries):
    text = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>导航回放</title>
<style>body{font:16px/1.7 sans-serif;max-width:1500px;margin:25px auto;padding:0 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}
article{border:1px solid #ddd;padding:12px;min-width:0}img{width:100%}a{color:#175c99}</style>
<h1>导航回放</h1><p><a href="../index.html">返回实验报告</a> · <a href="manifest.json">文件及原始episode清单</a></p>
<p>GIF均来自已保存的episode，没有重跑仿真。左侧为同一时刻完整真值（仅观察者可见），右侧为无人机局部估计；未知区域为灰色。点击“原始GIF”可单独打开。动画默认按仿真1倍速，并在决策帧增加阅读停顿，停顿不是真实推理耗时。未到达任务的低积分不能视为更优，须先检查到达状态和终止原因。</p>'''
    group = lambda row: row.get('case') or row['scene']
    for case in sorted({group(r) for r in entries}):
        text += f'<h2>{html.escape(str(case))}</h2><div class="grid">'
        for row in sorted((r for r in entries if group(r) == case), key=lambda r: (r['seed'], r['method'])):
            metrics = row['metrics']; label = html.escape(LABELS.get(row['method'], row['method']))
            outcome = '到达' if metrics['success'] else '未到达'
            budget = f'，预算{html.escape(str(row["budget"]))}' if row.get('budget') is not None else ''
            termination = metrics.get('termination')
            cause = f'，终止原因：{html.escape(str(termination))}' if termination else ''
            text += (f'<article><h3>{label} / seed{row["seed"]}</h3>'
                     f'<p><strong>{outcome}</strong>{budget}{cause}；航程{metrics["executed_length_m"]:.1f}m，积分{metrics["interference_excess_integral_db_seconds"]:.3f}dB·s</p>'
                     f'<p><a href="{html.escape(row["gif_file"], quote=True)}">原始GIF</a> · '
                     f'<a href="../{html.escape(row["episode_path"], quote=True)}">原始episode</a> · '
                     f'<a href="../{html.escape(row["decision_log_path"], quote=True)}">决策日志</a></p>'
                     f'<img loading="lazy" src="{html.escape(row["gif_file"], quote=True)}"></article>')
        text += '</div>'
    (output / 'replay/index.html').write_text(text + '</html>', encoding='utf-8')


def render_replays(output, methods=None, seeds=None, *, render_missing=True,
                   frame_stride=4, playback_speed=1., decision_pause_seconds=1.):
    """Use status metadata, keep all existing gallery entries, and move legacy GIFs unchanged."""
    output = Path(output).resolve()
    replay = output / 'replay'
    replay.mkdir(exist_ok=True)
    manifest_path = replay / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = {r['episode_path']: r for r in previous.get('entries', [])}
    for status_path in sorted((output / 'runs').rglob('status.json')):
        status = json.loads(status_path.read_text())
        episode_path = status_path.parent / 'episode.json'
        if status.get('status') != 'completed' or not episode_path.is_file():
            continue
        if not all(key in status for key in ('method', 'scene', 'seed')):
            continue
        method, scene, seed = status['method'], status['scene'], status['seed']
        case = status.get('case')
        identity = case or scene
        if methods is not None and method not in methods or seeds is not None and seed not in seeds:
            continue
        # These fields become a filename; refuse traversal or accidental nested output.
        if any('/' in str(v) or '\\' in str(v) or str(v) in ('.', '..') for v in (method, identity, seed)):
            raise ValueError('invalid replay identity')
        destination = replay / f'{method}_{identity}_seed{seed}.gif'
        source = status_path.parent / 'replay.gif'
        moved_from = None
        episode_identity = digest(episode_path)
        if source.exists():
            source_identity = digest(source)
            if destination.exists():
                if digest(destination) != source_identity:
                    raise FileExistsError(f'conflicting GIFs: {source} and {destination}')
                source.unlink()  # Identical duplicate only; keep a single canonical file.
            else:
                source.rename(destination)
            assert digest(destination) == source_identity
            moved_from = str(source.relative_to(output))
        if not destination.exists() and render_missing:
            from uav_em.visualization import load_result, render
            render(load_result(episode_path), destination, animate=True, frame_stride=frame_stride,
                   playback_speed=playback_speed, decision_pause_seconds=decision_pause_seconds)
        if not destination.exists():
            continue
        assert digest(episode_path) == episode_identity, 'rendering changed episode'
        key = str(episode_path.relative_to(output))
        old = entries.get(key, {})
        entries[key] = {'method': method, 'scene': scene, 'case': case,
                        'budget': status.get('budget'), 'seed': seed,
                        'episode_path': key, 'episode_sha256': episode_identity,
                        'decision_log_path': str((status_path.parent / 'decision_log.txt').relative_to(output)),
                        'gif_file': destination.name, 'gif_sha256': digest(destination),
                        'legacy_gif_moved_from': moved_from or old.get('legacy_gif_moved_from'),
                        'render_settings': old.get('render_settings', {
                            'frame_stride': frame_stride, 'playback_speed': playback_speed,
                            'decision_pause_seconds': decision_pause_seconds,
                            'legacy_settings_assumed_from_original_study': bool(moved_from)}),
                        'metrics': {key: status[key] for key in ('success', 'executed_length_m',
                                   'interference_excess_integral_db_seconds', 'termination') if key in status}}
        print('REPLAY', destination.relative_to(output), flush=True)
    values = sorted(entries.values(), key=lambda r: (r.get('case') or r['scene'], r['seed'], r['method']))
    write_json(manifest_path, {'schema_version': 1, 'source': 'saved_episodes_only_no_simulation_rerun',
                              'count': len(values), 'entries': values})
    gallery(output, values)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='实验根目录，内有runs/；输出固定为replay/')
    parser.add_argument('--method', action='append', help='方法名；可重复，省略则全部')
    parser.add_argument('--seed', action='append', type=int, help='任务随机种子；可重复，省略则全部')
    parser.add_argument('--stride', type=int, default=4, help='抽帧间隔，正整数；保留决策帧和末帧')
    parser.add_argument('--speed', type=float, default=1., help='仿真回放倍速，必须为正')
    parser.add_argument('--decision-pause', type=float, default=1., help='决策阅读停顿秒数，非负；不是推理耗时')
    args = parser.parse_args()
    render_replays(args.root, args.method, args.seed, frame_stride=args.stride,
                   playback_speed=args.speed, decision_pause_seconds=args.decision_pause)


if __name__ == '__main__':
    main()
