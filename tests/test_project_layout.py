"""Current entrypoints work without GPU packages or external datasets."""
from pathlib import Path
import subprocess
import sys

import pytest
from uav_em.config import load_config
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('entry', ['tools/run_local_power_demo.py','tools/run_point_agent_study.py',
                                  'tools/render_navigation_replays.py','src/run_experiment.py',
                                  'src/replay_experiment.py','temp/debug_full_pipeline.py'])
def test_current_entry_help_works_outside_root(entry):
    result=subprocess.run([sys.executable,str(ROOT/entry),'--help'],cwd=ROOT/'tools',
                          capture_output=True,text=True,timeout=30,
                          env=__import__('os').environ | {'PYTHONPATH':str(ROOT/'src')})
    assert result.returncode==0,result.stderr
    assert '--help' in result.stdout


def test_current_config_uses_only_point_power_and_local_gp():
    config=load_config(ROOT/'configs/local_power.json')
    assert config.sensor.mode=='power_only'
    assert config.belief.estimation_method=='local_gp'
    assert config.electromagnetic.propagation_mode=='single_reflection'


def test_current_launch_and_handoff_assets_exist():
    names=('README.md','AGENTS.md','requirements.txt','pytest.ini','docs/AGENT_HANDOFF.md',
           'docs/current_roadmap.md','docs/first_version.md','docs/local_power_environment.md',
           'docs/llm_information_contract.md','outputs/README.md','temp/debug_full_pipeline.py')
    assert all((ROOT/name).is_file() for name in names)
