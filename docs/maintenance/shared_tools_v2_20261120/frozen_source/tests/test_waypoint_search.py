from dataclasses import replace

import numpy as np
import pytest

from uav_em.config import load_config
from uav_em.environment import Geometry
from uav_em.feasible_scenario import best,evaluate,search_waypoints
from uav_em.repair_search import WorkBudget
from uav_em.scenario_planning import ScenarioForecast


def setup():
    cfg=load_config('configs/local_power.json')
    cfg=replace(cfg,map=replace(cfg.map,buildings=(),obstacles_m=()))
    geometry=Geometry(cfg.map,5.,30.)
    initial=dict(position_m=[40.,300.],time_seconds=0.,interference_dbm=-90.)
    model=dict(id='source',family='static',std_db=0.,weight=1.,born_index=0,
        reference_time_seconds=0.,bases=[dict(center_m=[400.,300.],scale_m=60.,
        velocity_mps=[0.,0.],rate_db_per_second=0.)],peaks_dbm=[-60.])
    forecast=ScenarioForecast([initial],[model],0.)
    state=dict(position_m=[40.,300.],goal_m=[760.,300.],route_m=[[40.,300.],[760.,300.]],
        speed_mps=15.,spent_length_m=0.,maximum_length_m=900.,remaining_energy_wh=100.,
        flight_power_w=350.,hover_power_w=300.,reserve_energy_wh=15.,delay_seconds=.5)
    return geometry,forecast,state


def test_lateral_search_finds_noncollinear_lower_cost_under_one_frozen_field():
    g,f,state=setup();work=WorkBudget(700,10000)
    incumbent=evaluate(state['route_m'],state,f,g,work,'mixture','incumbent')
    result=search_waypoints(state,f,g,work,'mixture')
    chosen=best([incumbent]+result['candidates'])
    assert chosen['id'].startswith('lattice_')
    assert chosen['score']['score_seconds']<incumbent['score']['score_seconds']
    assert abs(chosen['path'][1][1]-300)>0
    assert result['proposals']==50 and work.samples==50
    assert {c['score']['forecast_sha256'] for c in result['candidates'] if c['feasible']}=={f.fingerprint}
    assert all(min(c['score']['query_time_seconds'])>f.now for c in result['candidates'] if c['feasible'])


def test_waypoints_do_not_bypass_whole_mission_length_or_energy_constraints():
    g,f,state=setup();state['spent_length_m']=180.
    # Exactly 720m remain in the mission allowance: any lateral detour is invalid.
    work=WorkBudget(700,10000);result=search_waypoints(state,f,g,work,'mixture')
    assert all(not c['feasible'] and c['reason']=='length_energy_endpoints' for c in result['candidates'])
    assert f.counts['receiver_queries']==0
    state['spent_length_m']=0.;state['remaining_energy_wh']=state['reserve_energy_wh']
    result=search_waypoints(state,f,g,work,'mixture')
    assert not any(c['feasible'] for c in result['candidates'])


def test_failed_samples_and_geometry_share_existing_budget_and_stop_cleanly():
    g,f,state=setup();work=WorkBudget(3,10000)
    result=search_waypoints(state,f,g,work,'mixture')
    assert result['proposals']==work.samples==3
    assert result['stop_reason']=='sample_budget_exhausted'
    work=WorkBudget(700,1)
    result=search_waypoints(state,f,g,work,'mixture')
    assert work.edges==1 and result['stop_reason']=='edge_budget_exhausted'
    assert not any(c['feasible'] for c in result['candidates'])


def test_blocked_straight_legs_are_rejected_without_hidden_rrt():
    g,f,state=setup()
    cfg=replace(g.config,obstacles_m=((300.,0.,500.,600.),))
    g=Geometry(cfg,5.,30.)
    result=search_waypoints(state,f,g,WorkBudget(700,10000),'mixture')
    assert not any(c['feasible'] for c in result['candidates'])
    assert f.counts['receiver_queries']==0


@pytest.mark.parametrize('limit',[0,51,True,1.5])
def test_unbounded_or_untyped_proposal_limits_are_rejected(limit):
    g,f,state=setup()
    with pytest.raises(ValueError):search_waypoints(state,f,g,WorkBudget(700,10000),'mixture',limit)
