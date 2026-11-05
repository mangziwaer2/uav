"""离线工具共用配置与新输出路径；不加载训练后端。"""

from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_settings(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate pipeline setting: {key}')
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)
    expected = {'seed', 'scene_count', 'scenario', 'collection', 'encoder', 'training'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('pipeline settings require seed/scene_count/scenario/collection/encoder/training')
    if any(not isinstance(value[key], dict) for key in ('scenario', 'collection', 'encoder', 'training')):
        raise ValueError('pipeline sections must be objects')
    return value


def fresh_output(kind):
    return ROOT / 'outputs' / f'interference_{kind}_{datetime.now():%Y%m%d_%H%M%S_%f}'
