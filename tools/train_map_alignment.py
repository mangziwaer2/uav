"""冻结LLM与地图编码器，只训练投影／时间层；显存不足保留现场并停止。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--evidence-dataset', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--budget-only', action='store_true')
    args = parser.parse_args()
    import json
    from uav_em.map_learning.alignment_training import AlignmentTrainingSettings, train_map_alignment
    from uav_em.map_learning.cli import fresh_output
    settings = AlignmentTrainingSettings(**json.loads(args.config.read_text(encoding='utf-8')))
    output = args.output or fresh_output('map_alignment_training')
    result = train_map_alignment(args.dataset, args.evidence_dataset, args.model_path, output,
                                  settings, args.device, args.resume, args.budget_only)
    print(f'结果：{output}；{result["status"]}，已完成{result["optimizer_steps"]}个优化器步。')
    if result['status'] not in ('completed', 'step_limit_reached', 'budget_only_passed'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
