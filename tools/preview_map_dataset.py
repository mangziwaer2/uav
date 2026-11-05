"""把一条机载地图轨迹保存为6通道PNG、时序题和慢速GIF；不读取隐藏真值。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from uav_em.map_learning.cli import fresh_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--episode', required=True, help='manifest中的episode id，不含npz后缀')
    parser.add_argument('--output', type=Path, help='新目录，拒绝覆盖')
    parser.add_argument('--no-gif', action='store_true')
    parser.add_argument('--stride', type=int, default=2, help='GIF抽帧间隔，正整数；保留末帧')
    parser.add_argument('--speed', type=float, default=1.0, help='仿真播放倍速，正数；默认1倍')
    args = parser.parse_args()
    from uav_em.map_learning.preview import preview_episode
    output = preview_episode(args.dataset, args.episode, args.output or fresh_output('map_preview'),
                             not args.no_gif, args.stride, args.speed)
    print(f'完成：{output.resolve()}；先打开README.md、channels.png和window.png。')


if __name__ == '__main__':
    main()
