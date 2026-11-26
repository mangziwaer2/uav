"""Freeze and execute the three independent RP1 development pairs."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from random import Random
from secrets import randbits
from time import monotonic

from .config import MapConfig, ProjectConfig
from .environment import ElectromagneticWorld, Geometry
from .passive_repair_checkpoint import _json, _profile_record, _write, minimal_profile
from .passive_repair_flight import fly_repair
from .passive_repair_geometry import OnlineRepairGeometry
from .passive_repair_pair import publish_tool_pair
from .passive_repair_receiver import isolated_stdio_transport
from .passive_repair_tools import INSTRUCTIONS
from .passive_route_exchange import _sha256


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/passive_repair_RP1_20261126"
CONTRACT = ROOT / "docs/passive_repair_RP1_contract.md"
FAMILIES = ("single_side", "crossing", "staggered")
ARMS = ("numerical", "llm")
TIMEOUT_S, TOTAL_TIMEOUT_S = 180, 1200


def _scenario(family, layout_seed, noise_seed):
    rng = Random(layout_seed)
    start = (rng.randint(35, 50), rng.randint(96, 104))
    goal = (rng.randint(240, 270), start[1] + rng.randint(-8, 8))
    sign = rng.choice((-1, 1))
    lateral = rng.randint(33, 42)
    east = rng.randint(143, 156)
    width = rng.randint(23, 32)

    def building(index, west, offset, span):
        south, north = sorted((start[1] + sign * offset, start[1] + sign * (offset + span)))
        return dict(id=f"B{index}", height_m=rng.randint(42, 59),
                    footprint_m=((west, south), (west + width, south),
                                 (west + width, north), (west, north)))

    if family == "single_side":
        buildings = (building(1, east, lateral, 24),)
    elif family == "crossing":
        buildings = (building(1, east, -rng.randint(8, 15), rng.randint(34, 42)),)
    else:
        buildings = (building(1, east - 32, rng.randint(16, 24), 25),
                     building(2, east + 37, -rng.randint(45, 56), 32))

    def hotspot(center, sigma, peak, start_s):
        return dict(center_m=list(center), velocity_mps=[rng.uniform(-0.9, 0.9), rng.uniform(-0.65, 0.65)],
                    sigma_m=sigma, peak_dbm=peak, start_seconds=start_s, end_seconds=90.0)

    hotspots = (hotspot((start[0] + rng.randint(70, 94), start[1] + sign * rng.randint(-10, 10)),
                        rng.uniform(13, 19), rng.uniform(-70, -65), 0.0),)
    if family == "staggered":
        hotspots += (hotspot((east + rng.randint(49, 67), start[1] - sign * rng.randint(12, 24)),
                             rng.uniform(10, 16), rng.uniform(-80, -75), rng.uniform(3, 6)),)
    base = ProjectConfig(seed=noise_seed)
    return replace(base, map=MapConfig(columns=33, rows=23, resolution_m=10.0, start_m=start, goal_m=goal,
                                       obstacles_m=(), geometry_mode="fixed_altitude", buildings=buildings),
                   electromagnetic=replace(base.electromagnetic, fields=hotspots, background_dbm=-100.0,
                                           measurement_noise_db=1.0))


def _record(family, layout_seed, noise_seed):
    scene = _scenario(family, layout_seed, noise_seed)
    scene.validate()
    aircraft = scene.aircraft
    geometry = Geometry(scene.map, aircraft.clearance_m, aircraft.altitude_m, aircraft.vertical_clearance_m)
    corners = len(OnlineRepairGeometry(geometry, aircraft).nodes) - 1
    attempts = corners + 9
    return dict(family=family, layout_seed=layout_seed, noise_seed=noise_seed,
                scenario_sha256=_sha256(_json(scene.to_dict())), corners=corners,
                public=dict(map=asdict(scene.map), aircraft=asdict(aircraft), belief=asdict(scene.belief)),
                resources=dict(geometry=59 * attempts, query_points=300 + 59 * 13 * (attempts + 1),
                               display_points=61 * scene.map.columns * scene.map.rows, points=61, fits=61))


def _manifest(executable, profile, seeds):
    files = (*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("*.js"),
             *Path(__file__).parent.glob("*.html"), ROOT / "requirements.txt")
    return dict(contract_sha256=_sha256(CONTRACT.read_bytes()), prompt_sha256=_sha256(INSTRUCTIONS.encode()),
                implementation={str(path.relative_to(ROOT)): _sha256(path.read_bytes()) for path in sorted(files)},
                receiver_sha256=_sha256(executable.read_bytes()), profile=_profile_record(profile),
                timeout_s=TIMEOUT_S, total_timeout_s=TOTAL_TIMEOUT_S, arms=list(ARMS),
                scenes={f"RP1-{index:02d}": _record(family, *seed_pair)
                        for index, (family, seed_pair) in enumerate(zip(FAMILIES, seeds, strict=True), 1)})


def freeze(executable, profile):
    seeds = [(randbits(64) | (1 << 63), randbits(64) | (1 << 63)) for _ in FAMILIES]
    manifest = _manifest(executable, profile, seeds)
    OUTPUT.mkdir()
    _write(OUTPUT / "contract.md", CONTRACT.read_bytes())
    raw = _json(manifest)
    _write(OUTPUT / "manifest.json", raw)
    print(_sha256(raw), flush=True)


def run(scene_id, digest, executable, profile):
    raw = (OUTPUT / "manifest.json").read_bytes()
    if _sha256(raw) != digest:
        raise ValueError("expected the frozen RP1 manifest SHA-256")
    manifest = json.loads(raw)
    seeds = [(scene["layout_seed"], scene["noise_seed"]) for scene in manifest["scenes"].values()]
    if (manifest != json.loads(_json(_manifest(executable, profile, seeds)))
            or _sha256((OUTPUT / "contract.md").read_bytes()) != manifest["contract_sha256"]):
        raise ValueError("frozen RP1 implementation or inputs changed")
    record = manifest["scenes"][scene_id]
    scene = _scenario(record["family"], record["layout_seed"], record["noise_seed"])
    destination = OUTPUT / scene_id
    destination.mkdir()
    _write(destination / "started.json", _json(dict(scene_id=scene_id, manifest_sha256=digest)))
    logs, seconds = {}, {}
    for arm in ARMS:
        world = ElectromagneticWorld(scene)
        adapter = {}
        if arm == "llm":
            adapter = dict(request_prefix=scene_id, transport=isolated_stdio_transport(
                destination / "rounds", executable, profile, timeout_s=TIMEOUT_S, total_timeout_s=TOTAL_TIMEOUT_S))
        started = monotonic()
        log = fly_repair(world.geometry, scene.aircraft, scene.belief, world.observe, arm=arm, separation_m=20,
                         min_post_s=0.5, tool_geometry_limit=record["resources"]["geometry"],
                         tool_query_limit=record["resources"]["query_points"], **adapter)
        seconds[arm] = monotonic() - started
        logs[arm] = json.loads(_json(log))
        _write(destination / f"{arm}.json", _json(logs[arm]))
        print(f"{scene_id} {arm}: {log['stop']} at {log['frames'][-1]['t']:.3f}s", flush=True)
    _write(destination / "wall_clock.json", _json(seconds))
    report = publish_tool_pair(destination / "flight", world, logs, synthetic=False)
    print(f"{scene_id} replay: {destination / 'flight/replay.html'}, delta={report['paired_delta_tq']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    execution = commands.add_parser("run")
    execution.add_argument("scene", choices=[f"RP1-{index:02d}" for index in range(1, 4)])
    execution.add_argument("digest")
    args = parser.parse_args()
    profile = minimal_profile(args.profile)
    profile[0]["model_reasoning_effort"] = "medium"
    if args.command == "freeze":
        freeze(args.executable, profile)
    else:
        run(args.scene, args.digest, args.executable, profile)


if __name__ == "__main__":
    main()
