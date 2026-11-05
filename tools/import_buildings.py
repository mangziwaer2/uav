"""离线导入建筑；模板定义局部地图和飞行条件，输出自包含2.5D配置。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uav_em.config import load_config
from uav_em.map_import import import_buildings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--template", type=Path, required=True, help="buildings和obstacles_m均为空的配置")
    parser.add_argument("--output", type=Path, required=True, help="新配置路径，禁止覆盖")
    parser.add_argument("--coordinates", choices=("local_m", "lonlat"), required=True)
    parser.add_argument("--origin-lonlat", nargs=2, type=float)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True, dest="license_name")
    parser.add_argument("--default-height", type=float, help="缺失楼高的显式默认楼顶高度m，不默默推断")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = import_buildings(args.input, load_config(args.template), args.coordinates, args.source,
                              args.license_name, args.origin_lonlat, args.default_height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(config.to_dict(), stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Imported {len(config.map.buildings)} buildings: {args.output}")


if __name__ == "__main__":
    main()
