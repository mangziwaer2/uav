"""离线生成Sionna RT干扰真值；引擎只用于环境，不参与LLM决策。"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uav_em.config import load_config
from uav_em.rt_export import load_rt_settings, rectangular_building, validate_cell_centers, write_rt_bundle


ROOT = Path(__file__).resolve().parent.parent


def check_free_space(rt, frequency_hz):
    """用独立自由空间公式核对引擎RSS单位及单元平均值，非实测标定。"""
    scene = rt.load_scene()
    scene.frequency = frequency_hz
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                    horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                    horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.add(rt.Transmitter(name="sanity_only", position=[0, 0, 5], power_dbm=0))
    radio_map = rt.RadioMapSolver()(scene, center=[50, 0, 30], orientation=[0, 0, 0],
        size=[30, 30], cell_size=[10, 10], samples_per_tx=1000000, max_depth=0, seed=7,
        los=True, specular_reflection=False, diffuse_reflection=False, refraction=False, diffraction=False)
    actual_dbm = 10 * np.log10(float(np.asarray(radio_map.rss)[0, 1, 1]) * 1000)
    offsets = -5 + (np.arange(100) + 0.5) / 10
    horizontal, vertical = np.meshgrid(50 + offsets, offsets)
    distance_squared = horizontal**2 + vertical**2 + 25**2
    expected_gain = np.mean((299792458 / frequency_hz / (4 * np.pi))**2 / distance_squared)
    expected_dbm = 10 * np.log10(expected_gain)
    error_db = float(actual_dbm - expected_dbm)
    if not np.isfinite(actual_dbm) or abs(error_db) > 1:
        raise ValueError(f"free-space RSS sanity check failed: error={error_db:.3f} dB")
    print(f"自由空间功率核查通过：误差{error_db:.3f}dB（容限1dB，非实测）", flush=True)
    return {"actual_dbm": actual_dbm, "expected_cell_mean_dbm": float(expected_dbm),
            "error_db": error_db, "tolerance_db": 1.0, "samples_per_tx": 1000000,
            "reference": "isotropic Friis gain averaged over the central 10m cell"}


def generate(config_path, settings_path, output_dir, variant="cuda_ad_mono_polarized", samples_per_tx=None):
    if sys.version_info < (3, 11):
        raise RuntimeError("Sionna RT 2.2.0 imports typing.Self: use a separate Python >=3.11 environment (3.12 recommended)")
    config = load_config(config_path)
    settings = load_rt_settings(settings_path)
    if config.map.geometry_mode != "fixed_altitude" or config.map.obstacles_m or config.map.buildings:
        raise ValueError("RT template requires fixed_altitude and empty geometry; geometry comes from the engine")
    if Path(output_dir).exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if samples_per_tx is not None:
        if type(samples_per_tx) is not int or not 1 <= samples_per_tx <= 100000000:
            raise ValueError("samples_per_tx must be an integer in [1,100000000]")
        settings["samples_per_tx"] = samples_per_tx
    os.environ["MI_DEFAULT_VARIANT"] = variant
    cache = ROOT / "temp" / "sionna_mpl_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import mitsuba as mi
    mi.set_variant(variant)
    from sionna import rt

    if version("sionna-rt") != "2.2.0":
        raise RuntimeError("this exporter is verified against sionna-rt==2.2.0; use requirements-rt.txt")
    free_space_check = check_free_space(rt, config.electromagnetic.carrier_frequency_hz)
    scene_path = Path(rt.scene.simple_street_canyon)
    print(f"加载官方街谷场景，engine={version('sionna-rt')} variant={mi.variant()}", flush=True)
    scene = rt.load_scene(str(scene_path), merge_shapes=False)
    scene.frequency = config.electromagnetic.carrier_frequency_hz
    scene.bandwidth = settings["bandwidth_hz"]
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                    horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                    horizontal_spacing=0.5, pattern="iso", polarization="V")
    origin = np.asarray(settings["origin_xy_m"], dtype=float)
    buildings = []
    scene_hash = hashlib.sha256(scene_path.read_bytes())
    mesh_records = []
    mesh_vertices = {}
    for shape in scene.mi_scene.shapes():
        parameters = mi.traverse(shape)
        vertices = np.asarray(parameters["vertex_positions"]).reshape(-1, 3)
        mesh_vertices[shape.id()] = vertices.copy()
        scene_hash.update(shape.id().encode("utf-8"))
        scene_hash.update(vertices.tobytes())
        scene_hash.update(np.asarray(parameters["faces"]).tobytes())
        mesh_records.append({"name": shape.id(), "minimum_m": vertices.min(axis=0).tolist(),
                             "maximum_m": vertices.max(axis=0).tolist()})
    floor = mesh_vertices.get("mesh-floor")
    if floor is None or np.ptp(floor[:, 2]) > 1e-5:
        raise ValueError("expected a flat official scene floor for the local altitude datum")
    ground_z = float(floor[0, 2])
    for identifier, vertices in mesh_vertices.items():
        if identifier == "mesh-floor":
            continue
        local_vertices = vertices.copy()
        local_vertices[:, 2] -= ground_z
        buildings.append(rectangular_building(identifier, local_vertices, origin))
    if not buildings:
        raise ValueError("no verified buildings found; scene geometry must not be silently dropped")
    for source in settings["transmitters"]:
        local = np.asarray(source["position_m"], dtype=float)
        maximum = (np.array([config.map.columns, config.map.rows]) - 1) * config.map.resolution_m
        if np.any(local[:2] < 0) or np.any(local[:2] > maximum):
            raise ValueError("transmitter must be inside the declared project domain")
        for building in buildings:
            lower = np.min(building["footprint_m"], axis=0)
            upper = np.max(building["footprint_m"], axis=0)
            if (np.all(local[:2] >= lower) and np.all(local[:2] <= upper)
                    and building["base_m"] <= local[2] <= building["base_m"] + building["height_m"]):
                raise ValueError("transmitter is inside an opaque building")
        scene_position = np.r_[local[:2] + origin, local[2] + ground_z].tolist()
        scene.add(rt.Transmitter(name=source["name"], position=scene_position,
                                 power_dbm=source["power_dbm"]))
    spacing = config.map.resolution_m
    horizontal = np.arange(config.map.columns) * spacing
    vertical = np.arange(config.map.rows) * spacing
    center = [float(origin[0] + horizontal[-1] / 2), float(origin[1] + vertical[-1] / 2),
              config.aircraft.altitude_m + ground_z]
    print(f"计算{len(settings['transmitters'])}源功率，单源射线={settings['samples_per_tx']}，高度={center[2]}m", flush=True)
    started = time.perf_counter()
    radio_map = rt.RadioMapSolver()(scene, center=center, orientation=[0, 0, 0],
        size=[config.map.columns * spacing, config.map.rows * spacing], cell_size=[spacing, spacing],
        samples_per_tx=settings["samples_per_tx"], max_depth=settings["max_depth"], seed=settings["seed"],
        los=True, specular_reflection=True, diffuse_reflection=False, refraction=False, diffraction=False)
    centers = np.asarray(radio_map.cell_centers)
    validate_cell_centers(centers, horizontal, vertical, config.aircraft.altitude_m + ground_z, origin)
    received = np.asarray(radio_map.rss, dtype=float)
    elapsed = time.perf_counter() - started
    if received.shape != (len(settings["transmitters"]), config.map.rows, config.map.columns):
        raise ValueError(f"unexpected RSS map shape {received.shape}")
    provenance = {"engine_version": version("sionna-rt"), "mitsuba_version": version("mitsuba"),
        "drjit_version": version("drjit"), "numpy_version": version("numpy"), "variant": mi.variant(),
        "scipy_version": version("scipy"), "matplotlib_version": version("matplotlib"),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "solver_seconds": elapsed,
        "free_space_check": free_space_check,
        "scene_name": settings["scene_name"], "scene_sha256": scene_hash.hexdigest(), "meshes": mesh_records,
        "scene_ground_z_m": ground_z,
        "license": "Sionna RT distribution, Apache-2.0; retain upstream notices",
        "settings": settings, "template_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "effective_settings_sha256": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256((ROOT / "src/uav_em/rt_export.py").read_bytes()).hexdigest(),
        "propagation": {"los": True, "specular_reflection": True, "diffuse_reflection": False,
                        "refraction": False, "diffraction": False, "antenna": "single isotropic V-polarized",
                        "power_dbm_definition": "per-source total transmit power within the declared observation band",
                        "bandwidth_model": "single-frequency flat-band approximation, no spectral integration"}}
    path = write_rt_bundle(output_dir, config, received, buildings, provenance)
    print(f"已生成离线干扰真值：{path}，射线计算{elapsed:.3f}s", flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/optional/sionna_rt_demo.json")
    parser.add_argument("--sources", type=Path, default=ROOT / "configs/optional/sionna_rt_sources.json")
    parser.add_argument("--output-dir", type=Path, help="必须为不存在的新目录，默认outputs/interference_rt_时间戳")
    parser.add_argument("--variant", choices=["cuda_ad_mono_polarized", "llvm_ad_mono_polarized"],
                        default="cuda_ad_mono_polarized", help="显式选择GPU或LLVM CPU；不偷偷回退")
    parser.add_argument("--samples-per-tx", type=int, help="覆盖单源射线数，必须为1至1亿整数；小值仅用于烟雾检查")
    args = parser.parse_args(argv)
    directory = args.output_dir or ROOT / "outputs" / f"interference_rt_{datetime.now():%Y%m%d_%H%M%S_%f}"
    generate(args.config, args.sources, directory, args.variant, args.samples_per_tx)


if __name__ == "__main__":
    main()
