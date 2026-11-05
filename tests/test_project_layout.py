import json
from pathlib import Path
import subprocess
import sys

import pytest

from uav_em.config import load_config
from uav_em.rt_export import load_rt_settings


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", ["import_buildings.py", "generate_interference_rt.py"])
def test_offline_tool_help_works_without_optional_engines_or_root_working_directory(entry):
    result = subprocess.run([sys.executable, str(ROOT / "tools" / entry), "--help"],
                            cwd=ROOT / "tools", capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--help" in result.stdout


@pytest.mark.parametrize("name", ["import_25d_template.json", "sionna_rt_demo.json"])
def test_optional_templates_are_loadable_from_their_new_directory(name):
    config = load_config(ROOT / "configs" / "optional" / name)
    assert config.map.geometry_mode == "fixed_altitude"
    assert not config.map.obstacles_m
    assert not config.map.buildings


def test_offline_source_config_is_still_a_separate_generation_input():
    settings = load_rt_settings(ROOT / "configs/optional/sionna_rt_sources.json")
    assert len(settings["transmitters"]) == 2


def test_moved_power_grid_example_preserves_its_dataset_target():
    path = ROOT / "configs/optional/demo_power_grid.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    target = (path.parent / value["electromagnetic"]["power_grid"]["path"]).resolve()
    assert target == ROOT / "datasets/example_power_grid.json"
    assert value["sensor"]["mode"] == "power_only"


def test_current_launch_and_handoff_assets_exist():
    names = ("README.md", "AGENTS.md", "requirements.txt", "requirements-llm.txt", "requirements-rt.txt",
             "pytest.ini", "docs/AGENT_HANDOFF.md", "docs/CLOUD_HANDOFF.md", "docs/current_roadmap.md",
             "docs/first_version.md", "configs/README.md", "configs/optional/README.md", "tools/README.md",
             "datasets/README.md", "outputs/README.md", "temp/debug_full_pipeline.py",
             "temp/debug_full_pipeline.md")
    assert all((ROOT / name).is_file() for name in names)
