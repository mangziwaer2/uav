"""导出因果时序地图tokens及重建图，供检查和后续LLM适配。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from uav_em.map_learning.cli import fresh_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episode', type=Path, required=True, help='episodes中的npz文件')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--anchor', type=int, default=-1, help='帧号，-1表示最后一帧')
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--spacing', type=float, default=5.0, help='历史帧目标间隔s')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    from uav_em.map_learning.training import export_window
    output = export_window(args.checkpoint, args.episode, args.output or fresh_output('map_inspect'),
                           args.anchor, args.frames, args.spacing, args.device)
    print(f'完成：{output.resolve()}；先看reconstruction.png及window.json。')


if __name__ == '__main__':
    main()
