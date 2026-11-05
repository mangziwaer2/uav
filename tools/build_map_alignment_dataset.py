"""冻结地图编码器，构建train/validation区域问答与因果tokens缓存，不训练LLM。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='不存在的新数据目录')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    from uav_em.map_learning.alignment_data import build_alignment_dataset
    result = build_alignment_dataset(args.dataset, args.checkpoint, args.output, args.device, args.batch_size)
    print(f'完成：{args.output}；{result["counts"]}，未训练LLM，未读取test。')


if __name__ == '__main__':
    main()
