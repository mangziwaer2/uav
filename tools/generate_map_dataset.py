"""生成随机电磁场景和多行为机载地图数据，不调用LLM。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from uav_em.map_learning.cli import load_settings, fresh_output
from uav_em.map_learning.data import CollectionSettings, generate_dataset
from uav_em.map_learning.scenarios import ScenarioSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/map_encoding.json')
    parser.add_argument('--output', type=Path, help='新目录；已有目录拒绝覆盖')
    parser.add_argument('--scenes', type=int, help='场景数，至少3，按场景分组划分数据集')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, default=1, help='独立场景进程数，1—32，不改变种子或split')
    args = parser.parse_args()
    settings = load_settings(args.config)
    output = args.output or fresh_output('map_data')
    manifest = generate_dataset(output, args.scenes if args.scenes is not None else settings['scene_count'],
                                args.seed if args.seed is not None else settings['seed'],
                                ScenarioSettings(**settings['scenario']), CollectionSettings(**settings['collection']), args.workers)
    print(f'完成：{output.resolve()}，{len(manifest["episodes"])}条轨迹。隐藏配置仅在evaluation_only。')


if __name__ == '__main__':
    main()
