"""核查编码对齐数据的因果窗口、哈希、场景划分和原始机载监督，不读隐藏源。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='新报告文件，默认数据目录quality_report.json，不覆盖')
    args = parser.parse_args()
    from uav_em.map_learning.alignment_data import audit_alignment_dataset
    report = audit_alignment_dataset(args.dataset, args.source, args.output or args.dataset / 'quality_report.json')
    print(f'审核通过：{report["questions_checked"]}，标签已从原始机载帧重算。')


if __name__ == '__main__':
    main()
