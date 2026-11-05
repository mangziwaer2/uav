"""核验地图数据、全部观测重放、因果时序题和分组覆盖，不调用LLM。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from uav_em.map_learning.quality import audit_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='新报告文件，默认数据集内quality_report.json')
    parser.add_argument('--require-coverage', action='store_true', help='要求每个split具有0—3源、移动、三行为和时序变化')
    args = parser.parse_args()
    result = audit_dataset(args.dataset, args.output or args.dataset / 'quality_report.json', args.require_coverage)
    print(f'审核passed={result["passed"]}，轨迹={result["episodes_checked"]}，错误={len(result["errors"])}')
    if not result['passed']:
        for error in result['errors'][:20]:
            print(error)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
