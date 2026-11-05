"""将原始机载地图与因果时序题直接可视化，不需要checkpoint或隐藏真值。"""

import csv
import json
import math
from pathlib import Path

import numpy as np

from .data import CHANNELS, NORMALIZATION, CollectionSettings, file_hash, json_write, temporal_tasks
from .quality import checked_path


def choose_task(arrays, public, collection):
    tasks = list(temporal_tasks(arrays, collection, public['flight_seed']))
    if not tasks:
        raise ValueError('episode has no temporal task to preview')

    def score(task):
        target = task['target']
        changed = sum(abs(value) for row in target['power_delta_db'] for value in row if value is not None)
        return changed + np.sum(target['newly_supported']) + np.sum(target['lost_support'])

    return max(tasks, key=score)


def preview_episode(dataset, episode_id, output, animate=True, stride=2, speed=1.0):
    if type(stride) is not int or stride < 1 or isinstance(speed, bool) or not math.isfinite(speed) or speed <= 0:
        raise ValueError('stride must be a positive integer and speed finite and positive')
    root = Path(dataset)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('channels') != list(CHANNELS) or manifest.get('normalization') != NORMALIZATION:
        raise ValueError('unsupported map dataset protocol')
    entry = next((item for item in manifest['episodes'] if item['id'] == episode_id), None)
    if entry is None:
        raise ValueError('unknown episode id')
    path = checked_path(root, entry['path'])
    public_path = path.with_suffix('.json')
    if file_hash(path) != entry['sha256'] or file_hash(public_path) != entry['metadata_sha256']:
        raise ValueError('episode data or metadata hash mismatch')
    public = json.loads(public_path.read_text(encoding='utf-8'))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    collection = CollectionSettings(**manifest['collection_settings'])
    task = choose_task(arrays, public, collection)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    json_write(destination / 'temporal_task.json', {'episode_id': episode_id, 'split': entry['split'], **task})
    json_write(destination / 'sample.json', {
        'episode_id': episode_id, 'split': entry['split'], 'policy': entry['policy'],
        'dataset': str(root.resolve()), 'manifest_sha256': file_hash(root / 'manifest.json'),
        'episode_sha256': entry['sha256'], 'shape': list(arrays['maps'].shape), 'channels': list(CHANNELS),
        'times_seconds': arrays['times'].tolist(), 'positions_m': arrays['positions'].tolist(),
        'selected_anchor': task['anchor'], 'roi_grid_xyxy': task['roi_grid_xyxy'],
        'uses_evaluation_only': False, 'is_model_reconstruction': False,
        'selection': 'largest visible ROI change for illustration, not a representative evaluation sample',
        'animation_speed': speed, 'animation_stride': stride})
    current = arrays['maps'][task['anchor']]
    columns = ('column', 'row', 'x_m', 'y_m', 'supported', 'power_dbm',
               'uncertainty_db', 'age_seconds', 'blocked', 'building_height_m')
    with (destination / 'frame_grid.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in range(current.shape[1]):
            for column in range(current.shape[2]):
                values = current[:, row, column]
                supported = values[1] > 0.5
                writer.writerow((column, row, column * float(arrays['resolution_m']), row * float(arrays['resolution_m']),
                                 int(supported), float(values[0] * 90 - 120) if supported else '',
                                 float(values[2] * 30), float(values[3] * 35) if supported else '',
                                 int(values[4]), float(values[5] * 80)))
    with (destination / 'observations.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('time_seconds', 'x_m', 'y_m', 'interference_dbm'))
        for observation in public['observations']:
            writer.writerow((observation['time_seconds'], *observation['position_m'], observation['interference_dbm']))
    render_preview(arrays, public, task, destination, animate, stride, speed)
    text = f'''# 数据样本怎么看

样本：{episode_id}；分组：{entry['split']}；行为：{entry['policy']}。
原始张量形状：{list(arrays['maps'].shape)}，顺序为[时间,通道,行,列]。

1. 先打开channels.png：选定时刻的6个通道，已经换回dBm、dB、秒和米；support与blocked是0/1。
2. 再看window.png：同一坐标系下4帧历史，洋红框始终是同一个8格或配置大小的ROI；不是未来帧。
3. temporal_task.png：同一区域较早／当前的支持功率、共支持格的功率差、新增支持和失效支持；空白表示无对应支持，不是0dBm。
4. replay.gif（如启用）：左为当时机载功率估计，中为支持mask，右为有效信息年龄；默认按仿真时间1倍播放，末帧额外停2秒。
5. sample.json是可读的形状、通道、时间、位置和选帧说明；temporal_task.json是一道完整自动监督题，含ROI和每格标签。
6. frame_grid.csv可用Excel打开，逐行显示选定帧的每格坐标和6通道真实数值，未知功率／年龄留空；observations.csv为整条轨迹各时刻的原始点功率测量，不是外推栅格。

功率／年龄图浅灰为未知／无支持，深灰为当前高度不可飞区域；二值mask图紫色为0、黄色为1。蓝线只画截至该帧真实执行的历史折线，星号为当前UAV。
功率越大（越不负）干扰越强。当前power来自有限点测的局部外推，support表示有效观测支持，绝不是圆盘内逐点真实测量。
年龄只在有支持处显示，未知处不填0秒；不确定性是启发式误差尺度，不是概率。楼高来自已知公开几何。
这些图直接显示数据，不是编码器重建，也不是真值地图。没有读取evaluation_only或显示隐藏源坐标。
选帧只依据可见ROI变化，以便看清内容；不能据此评价模型效果，也不能把估计变化直接解释为源移动。
'''
    (destination / 'README.md').write_text(text, encoding='utf-8')
    return destination


def render_preview(arrays, public, task, destination, animate, stride, speed):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    from PIL import Image

    maps, times = arrays['maps'], arrays['times']
    resolution = float(arrays['resolution_m'])
    rows, columns = maps.shape[-2:]
    extent = [-resolution / 2, (columns - 0.5) * resolution, -resolution / 2, (rows - 0.5) * resolution]
    anchor = task['anchor']
    power_cmap = pyplot.get_cmap('YlOrRd').with_extremes(bad='#dedede')

    def draw_map(axis, frame, channel=0, roi=False):
        current = maps[frame]
        supported = current[1] > 0.5
        scale, offset = ((90, -120), (1, 0), (30, 0), (35, 0), (1, 0), (80, 0))[channel]
        image = current[channel] * scale + offset
        if channel in (0, 3):
            image = np.where(supported, image, np.nan)
        limits = ((-120, -30), (0, 1), (0, 30), (0, 35), (0, 1), (0, 80))[channel]
        cmap = power_cmap if channel == 0 else pyplot.get_cmap('viridis').with_extremes(bad='#dedede')
        artist = axis.imshow(image, origin='lower', extent=extent, cmap=cmap,
                             vmin=limits[0], vmax=limits[1], interpolation='nearest')
        if channel not in (4, 5):
            blocked = np.ma.masked_where(current[4] < 0.5, np.ones((rows, columns)))
            axis.imshow(blocked, origin='lower', extent=extent, cmap=ListedColormap(['#30343b']), interpolation='nearest')
        route = [public['observations'][0]['position_m']]
        for transition in public['transitions'][:frame]:
            route.extend(transition['traversed_m'])
        route = np.asarray(route)
        axis.plot(route[:, 0], route[:, 1], color='#1475bd', linewidth=1)
        position = arrays['positions'][frame]
        axis.scatter(*position, marker='*', s=90, color='black', edgecolors='white', linewidths=0.5, zorder=5)
        if roi:
            left, top, right, bottom = task['roi_grid_xyxy']
            axis.add_patch(Rectangle(((left - 0.5) * resolution, (top - 0.5) * resolution),
                                    (right - left) * resolution, (bottom - top) * resolution,
                                    fill=False, edgecolor='magenta', linewidth=2))
        axis.set(xlim=(0, (columns - 1) * resolution), ylim=(0, (rows - 1) * resolution), xlabel='x (m)', ylabel='y (m)')
        return artist

    figure, axes = pyplot.subplots(2, 3, figsize=(14, 8), layout='constrained')
    labels = ('Supported power (dBm)', 'Support mask', 'Uncertainty (dB)',
              'Supported age (s)', 'Blocked at flight altitude', 'Public building height (m)')
    for channel, axis in enumerate(axes.flat):
        artist = draw_map(axis, anchor, channel)
        axis.set_title(labels[channel])
        figure.colorbar(artist, ax=axis, shrink=0.75)
    figure.suptitle(f'{public["scenario_id"]} / {public["policy"]} | t={times[anchor]:.1f}s | onboard data, NOT truth')
    figure.savefig(destination / 'channels.png', dpi=130)
    pyplot.close(figure)

    figure, axes = pyplot.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for axis, frame in zip(axes.flat, task['window']['indices']):
        if frame < 0:
            axis.set_title('Missing past frame (invalid)')
            axis.set_axis_off()
        else:
            artist = draw_map(axis, frame, roi=True)
            axis.set_title(f'frame {frame} | t={times[frame]:.1f}s')
            figure.colorbar(artist, ax=axis, shrink=0.7, label='dBm')
    figure.suptitle('Causal history: SAME magenta ROI; unknown remains gray')
    figure.savefig(destination / 'window.png', dpi=130)
    pyplot.close(figure)

    left, top, right, bottom = task['roi_grid_xyxy']
    older, newer = (maps[index, :, top:bottom, left:right] for index in (task['older_index'], anchor))
    target = task['target']
    common = (older[1] > 0.5) & (newer[1] > 0.5)
    panels = (np.where(older[1] > 0.5, older[0] * 90 - 120, np.nan),
              np.where(newer[1] > 0.5, newer[0] * 90 - 120, np.nan),
              np.where(common, (newer[0] - older[0]) * 90, np.nan),
              np.asarray(target['newly_supported']), np.asarray(target['lost_support']),
              np.where(newer[1] > 0.5, newer[3] * 35, np.nan))
    titles = ('Earlier supported power (dBm)', 'Current supported power (dBm)', 'Delta: common support ONLY (dB)',
              'New support (0/1)', 'Lost support (0/1)', 'Current supported age (s)')
    ranges = ((-120, -30), (-120, -30), (-20, 20), (0, 1), (0, 1), (0, 35))
    figure, axes = pyplot.subplots(2, 3, figsize=(13, 8), layout='constrained')
    for index, (axis, image, title, limits) in enumerate(zip(axes.flat, panels, titles, ranges)):
        cmap = power_cmap if index < 2 else pyplot.get_cmap('coolwarm' if index == 2 else 'viridis').with_extremes(bad='#dedede')
        artist = axis.imshow(image, origin='lower', vmin=limits[0], vmax=limits[1], cmap=cmap, interpolation='nearest')
        axis.set(title=title, xlabel='local column', ylabel='local row')
        figure.colorbar(artist, ax=axis, shrink=0.7)
    figure.suptitle(f'Task B | global ROI [{left},{top},{right},{bottom}] | belief change, NOT source motion')
    figure.savefig(destination / 'temporal_task.png', dpi=130)
    pyplot.close(figure)

    if not animate:
        return
    indexes = sorted(set(range(0, len(times), stride)) | {len(times) - 1})
    images = []
    figure, axes = pyplot.subplots(1, 3, figsize=(12, 4), layout='constrained', dpi=90)
    for frame in indexes:
        for axis, channel, title in zip(axes, (0, 1, 3), ('Power (dBm): -120 to -30', 'Support mask: 0 to 1', 'Age (s): 0 to 35')):
            axis.clear()
            draw_map(axis, frame, channel)
            axis.set_title(title)
        figure.suptitle(f'{public["policy"]} | t={times[frame]:.1f}s | frame {frame}/{len(times)-1} | onboard only')
        figure.canvas.draw()
        images.append(Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()))
    durations = [max(20, round((times[second] - times[first]) * 100 / speed) * 10)
                 for first, second in zip(indexes, indexes[1:])] + [2000]
    images[0].save(destination / 'replay.gif', save_all=True, append_images=images[1:],
                   duration=durations, loop=0, disposal=2)
    for image in images:
        image.close()
    pyplot.close(figure)
