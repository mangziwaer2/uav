"""只用本地权重检查连续地图嵌入的LLM forward，不训练、不下载、不评价策略。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from uav_em.map_learning.cli import fresh_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='编码对齐数据目录，不是原始机载数据')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-positions', type=int, default=2048)
    args = parser.parse_args()
    from uav_em.map_learning.llm_bridge import smoke_local_llm
    output = args.output or fresh_output('map_llm_bridge')
    report = smoke_local_llm(args.dataset, args.model_path, output, args.sample_index,
                             args.device, args.seed, args.max_positions)
    print(f'完成：{output}；{report["status"]}，没有训练或地图理解结论。')


if __name__ == '__main__':
    main()
