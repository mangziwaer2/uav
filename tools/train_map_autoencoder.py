"""独立训练空间地图编解码器，不下载模型，不运行LLM或RL。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from uav_em.map_learning.cli import load_settings, fresh_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/map_encoding.json')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='新目录；已有目录拒绝覆盖')
    parser.add_argument('--device', default='auto', help='auto、cpu、cuda或cuda:0')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--require-quality', action='store_true', help='正式训练要求匹配manifest的完整覆盖审核已通过')
    args = parser.parse_args()
    from uav_em.map_learning.model import EncoderSettings
    from uav_em.map_learning.training import train_autoencoder
    settings = load_settings(args.config)
    training = dict(settings['training'])
    if args.epochs is not None:
        training['epochs'] = args.epochs
    output = args.output or fresh_output('map_encoder')
    metrics = train_autoencoder(args.dataset, output, EncoderSettings(**settings['encoder']), device=args.device,
                                require_quality=args.require_quality, **training)
    print(f'完成：{output.resolve()}，best_epoch={metrics["best_epoch"]}。这不是导航效果验收。')


if __name__ == '__main__':
    main()
