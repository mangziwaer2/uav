"""以已有训练组时序题微调地图编解码器，仅在validation选模，不训练LLM或检查test。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from uav_em.map_learning.cli import fresh_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='匹配数据manifest的boundary_v2权重')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/map_encoding_temporal.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--require-quality', action='store_true')
    args = parser.parse_args()
    from uav_em.map_learning.temporal_training import TemporalSettings, train_temporal_autoencoder
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    if args.epochs is not None:
        settings['epochs'] = args.epochs
    output = args.output or fresh_output('map_encoder_temporal')
    result = train_temporal_autoencoder(args.dataset, args.checkpoint, output, TemporalSettings(**settings),
                                       args.device, args.require_quality)
    print(f'完成：{output}；best_epoch={result["best_epoch"]}；未检查test，不是导航收益。')


if __name__ == '__main__':
    main()
