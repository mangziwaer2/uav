"""按源数量分层划分整场景并另存完整数据集，不重采集、不修改原目录。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from uav_em.map_learning.cli import fresh_output
from uav_em.map_learning.partition import repartition_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='新目录，默认自动时间戳')
    parser.add_argument('--seed', type=int, default=7, help='离线划分种子，非负整数')
    args = parser.parse_args()
    output = args.output or fresh_output('map_data_stratified')
    manifest = repartition_dataset(args.dataset, output, args.seed)
    print(f'完成：{output.resolve()}，{len(manifest["episodes"])}条轨迹；正式使用前仍需完整审核。')


if __name__ == '__main__':
    main()
