"""从已有机载观测派生独立测向证据，不重训编码器，不读取隐藏源或test。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='已有地图对齐数据')
    parser.add_argument('--source', type=Path, required=True, help='原始机载地图数据')
    parser.add_argument('--output', type=Path, required=True, help='不存在的新证据目录')
    args = parser.parse_args()
    from uav_em.map_learning.sensor_evidence import build_sensor_evidence
    report = build_sensor_evidence(args.dataset, args.source, args.output)
    print(f'完成：{args.output}；{report["counts"]}；不包含源假设。')


if __name__ == '__main__':
    main()
