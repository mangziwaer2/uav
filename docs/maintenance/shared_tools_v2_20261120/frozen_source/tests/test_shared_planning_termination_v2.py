"""One synthetic no-motion loop check for controlled planning failure."""
from uav_em.config import AircraftConfig,ElectromagneticConfig,MapConfig,ProjectConfig
from uav_em.scenario_loop import run


def test_no_feasible_common_route_returns_failure_with_paid_work_and_no_execution():
    config=ProjectConfig(
        aircraft=AircraftConfig(speed_mps=10.,flight_power_w=100.,hover_power_w=50.,
            initial_energy_wh=1.0001,reserve_energy_wh=1.),
        map=MapConfig(columns=11,rows=11,resolution_m=20.,start_m=(20.,20.),
            goal_m=(100.,20.),obstacles_m=(),buildings=()),
        electromagnetic=ElectromagneticConfig(fields=()))
    calls=[]
    def advisor(*args):
        calls.append(args)
        raise AssertionError('do not request an unexecutable proposal')
    episode=run(config,'mixture_llm',advisor=advisor)
    assert episode['metrics']['termination']=='no_feasible_route'
    assert not episode['metrics']['success']
    assert episode['motions']==episode['decisions']==calls==[]
    assert episode['metrics']['simulation_seconds']==episode['metrics']['executed_length_m']==0.
    assert episode['metrics']['energy_used_wh']==episode['metrics']['actual_model_calls']==0
    assert len(episode['planning_failures'])==1
    failure=episode['planning_failures'][0]
    assert {r['id'] for r in failure['candidates']}>={'incumbent','goal'}
    assert not any(r['feasible'] for r in failure['candidates'])
    assert episode['metrics']['total_edges']==failure['resources']['edges']>0
    assert episode['metrics']['total_samples']==failure['resources']['samples']==0
    assert episode['frames'][-1]['phase']=='planning_failed'
