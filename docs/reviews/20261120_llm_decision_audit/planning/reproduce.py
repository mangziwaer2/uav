"""Pure synthetic planning component audit; no world/config files/truth/flight."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from uav_em.config import MapConfig
from uav_em.environment import Geometry
from uav_em.feasible_scenario import evaluate, execute, pool, resource_feasible, shortcut
from uav_em.planning import path_length
from uav_em.repair_search import WorkBudget
from uav_em.scenario_planning import ScenarioForecast, state_request


def forecast(position):
    return ScenarioForecast([dict(position_m=position, time_seconds=0., interference_dbm=-100.)],
        [dict(id='synthetic',family='persistence',std_db=0.,born_index=0,weight=1.)],0.)


def state(start, goal, route, maximum=1000.):
    return dict(position_m=start,goal_m=goal,route_m=route,speed_mps=10.,
        spent_length_m=0.,maximum_length_m=maximum,remaining_energy_wh=100.,
        flight_power_w=100.,hover_power_w=50.,reserve_energy_wh=1.,delay_seconds=.5)


def test_incumbent_requires_second_hover():
    g=Geometry(MapConfig(columns=41,rows=41,resolution_m=5.,obstacles_m=(),buildings=()))
    s=state([20.,60.],[180.,60.],[[20.,60.],[100.,95.],[180.,60.]],maximum=200.)
    # At this energy the incumbent can finish, but cannot pay one more hover.
    s['remaining_energy_wh']=s['reserve_energy_wh']+path_length(s['route_m'])/s['speed_mps']*s['flight_power_w']/3600
    f=forecast(s['position_m']); direct=[s['position_m'],s['goal_m']]
    assert not resource_feasible(s['route_m'],s)
    assert resource_feasible(direct,s)
    paid=deepcopy(s);paid['delay_seconds']=0.
    assert resource_feasible(s['route_m'],paid)
    fresh=evaluate(direct,s,f,g,WorkBudget(700,10000),'mixture','new_direct')
    try:
        pool(s,f,g,WorkBudget(700,10000),'mixture')
    except ValueError as exc:
        error=str(exc)
    else:
        raise AssertionError('expected current incumbent guard')
    return dict(error=error,incumbent_length_m=path_length(s['route_m']),
        direct_length_m=path_length(direct),new_direct_feasible=fresh['feasible'],
        old_route_without_new_hover_feasible=True,new_hover_seconds=.5)


def test_partial_shortcut_lost():
    g=Geometry(MapConfig(columns=41,rows=31,resolution_m=20.,
        obstacles_m=((50.,0.,70.,120.),(130.,80.,150.,600.)),buildings=()))
    route=[[20.,60.],[20.,120.],[40.,140.],[90.,140.],[110.,60.]]
    route += [[float(x),60.] for x in np.linspace(151.,600.,20)]
    assert all(g.edge_free(a,b) for a,b in zip(route,route[1:]))
    # The first completed shortening certifies 0->2. Subsequent LOS checks
    # hit the unchanged default cap of 40; the tool returns the old input.
    certified_partial=[route[0]]+route[2:]
    assert all(g.edge_free(a,b) for a,b in zip(certified_partial,certified_partial[1:]))
    out,trace=shortcut(route,g,WorkBudget(700,10000))
    assert not trace['completed'] and out==route
    assert path_length(certified_partial)<path_length(out)
    maximum=1.25*float(np.linalg.norm(np.asarray(route[-1])-route[0]))
    assert path_length(route)<maximum
    return dict(trace=trace,input_length_m=path_length(route),returned_length_m=path_length(out),
        certified_partial_length_m=path_length(certified_partial),
        maximum_length_m=maximum,discarded_certified_partial=certified_partial)


def test_two_leg_budget(seed=7, sample_quotas=(350,350), edge_limit=10000):
    g=Geometry(MapConfig(columns=41,rows=31,resolution_m=20.,
        obstacles_m=((50.,80.,70.,120.),(130.,80.,150.,120.)),buildings=()))
    s=state([20.,100.],[180.,100.],[[20.,100.],[40.,125.],[160.,125.],[180.,100.]],maximum=200.)
    f=forecast(s['position_m']);w=WorkBudget(700,edge_limit)
    request=state_request(s,f,g,w)
    proposal=dict(request_id=request['request_id'],action='via',waypoint_m=[100.,100.],
        sample_quotas=list(sample_quotas),sampling='corridor',reason='synthetic two-obstacle via')
    result=execute(request,proposal,s,f,g,w,'mixture',seed)
    return dict(seed=seed,quotas=list(sample_quotas),failure=result.get('failure'),
        resources=w.summary(),candidate_count=len(result['candidates']),
        candidates=[dict(id=c['id'],feasible=c['feasible'],length_m=path_length(c['path']),
            score_seconds=c.get('score',{}).get('score_seconds')) for c in result['candidates']],
        searches=[dict(leg=r['leg'],reason=r['reason'],iterations=r['iterations'],
            first_solution_iteration=r['first_solution_iteration'],path=r['path'],length_m=r['length_m'])
            for r in result['searches']])


def test_holes_missing_from_request():
    building=dict(id='synthetic_courtyard',footprint_m=[[20.,20.],[180.,20.],[180.,180.],[20.,180.]],
        holes_m=[[[40.,40.],[160.,40.],[160.,160.],[40.,160.]]],height_m=10.)
    g=Geometry(MapConfig(columns=41,rows=41,resolution_m=5.,obstacles_m=(),buildings=(building,)))
    s=state([60.,100.],[140.,100.],[[60.,100.],[140.,100.]])
    assert g.edge_free(*s['route_m'])
    req=state_request(s,forecast(s['position_m']),g,WorkBudget(700,10000))
    b=req['known_geometry']['blocking_buildings'][0]
    assert 'holes_m' not in b
    return dict(actual_courtyard_route_is_free=True,public_building=b,holes_omitted=True)


if __name__=='__main__':
    results=dict(incumbent_hover=test_incumbent_requires_second_hover(),
        partial_shortcut=test_partial_shortcut_lost(),
        holes=test_holes_missing_from_request(),
        two_leg_default=test_two_leg_budget(),
        two_leg_smaller=test_two_leg_budget(sample_quotas=(40,40)),
        fixed_reserve_small_budget=test_two_leg_budget(sample_quotas=(40,40),edge_limit=744))
    assert results['two_leg_default']['failure']=='search_failed'
    assert not results['two_leg_smaller']['failure']
    assert any(c['feasible'] for c in results['two_leg_smaller']['candidates'])
    output=Path(__file__).with_name('evidence.json')
    output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2))
