"""固定地图编解码器的同区域时序保真核查；默认仅validation，不训练模型。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from uav_em.map_learning.cli import fresh_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--change-db', type=float, default=3.0, help='功率变化诊断阈值dB，有限正值，不是安全阈值')
    args = parser.parse_args()
    from uav_em.map_learning.temporal import audit_temporal_codec
    output = args.output or fresh_output('map_temporal_codec')
    report = audit_temporal_codec(args.dataset, args.checkpoint, output, args.split,
                                  args.device, args.batch_size, args.change_db)
    print(f'完成：{report["overall"]["tasks"]}道任务；结果：{output}')


if __name__ == '__main__':
    main()
