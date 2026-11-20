"""Shared-tool regressions on explicit synthetic geometry; no flight or world queries."""
from copy import deepcopy

import numpy as np
import pytest

from uav_em.config import MapConfig
from uav_em.environment import Geometry
from uav_em.feasible_scenario import best, execute, pool, resource_feasible, shortcut
from uav_em.planning import path_length
from uav_em.repair_search import WorkBudget
from uav_em.scenario_planning import ScenarioForecast, state_request


def public_state(start, goal, route, maximum=1000.):
    return dict(position_m=start,goal_m=goal,route_m=route,speed_mps=10.,
        spent_length_m=0.,maximum_length_m=maximum,remaining_energy_wh=100.,
        flight_power_w=100.,hover_power_w=50.,reserve_energy_wh=1.,delay_seconds=.5)


def constant_forecast(position):
    return ScenarioForecast([dict(position_m=position,time_seconds=0.,interference_dbm=-100.)],
        [dict(id='synthetic',family='persistence',std_db=0.,born_index=0,weight=1.)],0.)


def two_leg_problem(edge_limit=10000, quotas=(350,350)):
    geometry=Geometry(MapConfig(columns=41,rows=31,resolution_m=20.,
        obstacles_m=((50.,80.,70.,120.),(130.,80.,150.,120.)),buildings=()))
    state=public_state([20.,100.],[180.,100.],
        [[20.,100.],[40.,125.],[160.,125.],[180.,100.]],maximum=200.)
    forecast=constant_forecast(state['position_m']);work=WorkBudget(700,edge_limit)
    request=state_request(state,forecast,geometry,work)
    plan=dict(request_id=request['request_id'],action='via',waypoint_m=[100.,100.],
        sample_quotas=list(quotas),sampling='corridor',reason='synthetic two-obstacle construction')
    return geometry,state,forecast,work,request,plan


def shortcut_problem():
    geometry=Geometry(MapConfig(columns=41,rows=31,resolution_m=20.,
        obstacles_m=((50.,0.,70.,120.),(130.,80.,150.,600.)),buildings=()))
    route=[[20.,60.],[20.,120.],[40.,140.],[90.,140.],[110.,60.]]
    route += [[float(x),60.] for x in np.linspace(151.,600.,20)]
    assert all(geometry.edge_free(a,b) for a,b in zip(route,route[1:]))
    return geometry,route


def test_infeasible_incumbent_does_not_block_lower_energy_recovery():
    geometry=Geometry(MapConfig(columns=41,rows=41,resolution_m=5.,obstacles_m=(),buildings=()))
    state=public_state([20.,60.],[180.,60.],[[20.,60.],[100.,95.],[180.,60.]],maximum=200.)
    state['remaining_energy_wh']=state['reserve_energy_wh']+path_length(state['route_m'])/10.*100./3600.
    assert not resource_feasible(state['route_m'],state)
    rows=pool(state,constant_forecast(state['position_m']),geometry,WorkBudget(700,10000),'mixture')
    assert rows[0]['id']=='incumbent' and not rows[0]['feasible']
    assert rows[0]['reason']=='length_energy_endpoints'
    chosen=best(rows)
    assert chosen['id']=='goal' and path_length(chosen['path'])==160.
    assert resource_feasible(chosen['path'],state)


def test_all_infeasible_routes_fail_only_after_alternatives_are_recorded():
    geometry=Geometry(MapConfig(columns=41,rows=41,resolution_m=5.,obstacles_m=(),buildings=()))
    state=public_state([20.,60.],[180.,60.],[[20.,60.],[100.,95.],[180.,60.]],maximum=150.)
    rows=pool(state,constant_forecast(state['position_m']),geometry,WorkBudget(700,10000),'mixture')
    assert {row['id'] for row in rows}>={'incumbent','goal'}
    assert all(not row['feasible'] for row in rows)
    with pytest.raises(ValueError,match='no feasible retained route'):best(rows)


def test_incumbent_verification_budget_failure_is_recorded_without_uncaught_exception():
    geometry=Geometry(MapConfig(columns=41,rows=41,resolution_m=5.,obstacles_m=(),buildings=()))
    state=public_state([20.,60.],[180.,60.],[[20.,60.],[100.,95.],[180.,60.]],maximum=200.)
    work=WorkBudget(700,1)
    rows=pool(state,constant_forecast(state['position_m']),geometry,work,'mixture')
    assert rows[0]['id']=='incumbent' and rows[0]['reason']=='edge_budget_exhausted'
    assert {row['id'] for row in rows}>={'incumbent','goal'}
    assert not any(row['feasible'] for row in rows)
    assert work.edges==work.edge_limit==1


@pytest.mark.parametrize('edge_limit,quotas',[(10000,(350,350)),(744,(40,40))])
def test_paid_two_leg_search_retains_a_complete_feasible_proposal(edge_limit,quotas):
    geometry,state,forecast,work,request,plan=two_leg_problem(edge_limit,quotas)
    result=execute(request,plan,state,forecast,geometry,work,'mixture',7)
    assert 'failure' not in result
    chosen=best(result['candidates'])
    assert resource_feasible(chosen['path'],state)
    assert all(geometry.edge_free(a,b) for a,b in zip(chosen['path'],chosen['path'][1:]))
    assert chosen['score']['forecast_sha256']==forecast.fingerprint
    assert len(result['searches'])==2
    assert all(row['first_solution_iteration']<=row['iterations'] for row in result['searches'])
    assert all(row['reason'].startswith('solved') for row in result['searches'])
    assert all(row['construction_policy']=='balanced_leg_search_v2' for row in result['searches'])
    assert 0<work.samples<=sum(quotas)<=work.sample_limit
    assert 0<work.edges<=edge_limit and work.reserved_edges==0
    assert work.verification_edges>=len(chosen['path'])-1


def test_shared_search_budget_failure_does_not_claim_geometric_infeasibility():
    geometry,state,forecast,work,request,plan=two_leg_problem(1,(40,40))
    result=execute(request,plan,state,forecast,geometry,work,'mixture',7)
    assert result['failure']=='search_failed' and not result['candidates']
    assert result['searches'][0]['reason']=='edge_budget_exhausted'
    assert result['searches'][0]['search_edge_allowance']==0
    assert work.edges==1 and work.samples<=1 and work.reserved_edges==0


@pytest.mark.parametrize('edge_limit',[10000,40])
def test_shortcut_preserves_certified_prefix_at_check_or_edge_limit(edge_limit):
    geometry,route=shortcut_problem();before=deepcopy(route);work=WorkBudget(700,edge_limit)
    result,trace=shortcut(route,geometry,work)
    assert route==before and not trace['completed']
    assert result[0]==route[0] and result[-1]==route[-1]
    assert path_length(result)<path_length(route)
    assert all(geometry.edge_free(a,b) for a,b in zip(result,result[1:]))
    assert trace['checks']==work.edges==40
    assert trace['stop_reason']=='shortcut_check_limit'


def test_shortcut_preserves_progress_when_edge_ledger_exhausts_before_check_cap():
    geometry,route=shortcut_problem();work=WorkBudget(700,39)
    result,trace=shortcut(route,geometry,work)
    assert not trace['completed'] and trace['stop_reason']=='edge_budget_exhausted'
    assert trace['checks']==work.edges==39
    assert path_length(result)<path_length(route)
    assert all(geometry.edge_free(a,b) for a,b in zip(result,result[1:]))


