"""Bounded geometric construction/shortcut candidates, same arrival-time acceptor."""
from copy import deepcopy
from dataclasses import asdict
import math
import numpy as np
from .planning import path_length
from .repair_search import BudgetExhausted,repair_rrt_star
from .scenario_planning import FrozenSearchObjective,score_route,state_request,validate_search_plan


def checked_edge(geometry,work,a,b):
    work.take_edge()
    return geometry.edge_free(a,b)


def resource_feasible(route,state):
    if (len(route)<2 or not np.allclose(route[0],state['position_m'],rtol=0,atol=1e-8)
            or not np.allclose(route[-1],state['goal_m'],rtol=0,atol=1e-8)):
        return False
    length=path_length(route)
    return bool(state['spent_length_m']+length<=state['maximum_length_m']+1e-8 and
        state['remaining_energy_wh']-(length/state['speed_mps']*state['flight_power_w']+
        state['delay_seconds']*state['hover_power_w'])/3600>=state['reserve_energy_wh'])


def evaluate(route,state,forecast,geometry,work,method,identifier):
    if not resource_feasible(route,state):return dict(id=identifier,path=deepcopy(route),feasible=False,reason='length_energy_endpoints')
    for a,b in zip(route,route[1:]):
        ok=checked_edge(geometry,work,a,b);work.verification_edges+=1
        if not ok:return dict(id=identifier,path=deepcopy(route),feasible=False,reason='geometry')
    return dict(id=identifier,path=deepcopy(route),feasible=True,reason=None,
                score=score_route(forecast,route,state['speed_mps'],state['delay_seconds'],method))


def shortcut(route,geometry,work,max_checks=40):
    """Bounded greedy LOS shortening; not a power improvement guarantee."""
    route=deepcopy(route);out=[route[0]];i=0;checks=0
    while i<len(route)-1:
        chosen=i+1
        for j in range(len(route)-1,i+1,-1):
            if checks>=max_checks:return route,dict(checks=checks,completed=False)
            checks+=1
            if checked_edge(geometry,work,route[i],route[j]):chosen=j;break
        out.append(route[chosen]);i=chosen
    return out,dict(checks=checks,completed=True)


def best(records):
    valid=[r for r in records if r['feasible']]
    if not valid:raise ValueError('no feasible retained route')
    return min(valid,key=lambda r:(r['score']['score_seconds'],path_length(r['path']),r['id']))


def pool(state,forecast,geometry,work,method):
    rows=[evaluate(state['route_m'],state,forecast,geometry,work,method,'incumbent')]
    if not rows[0]['feasible']:raise ValueError('incumbent must remain feasible')
    position=np.asarray(state['position_m']);goal=np.asarray(state['goal_m']);delta=goal-position;distance=np.linalg.norm(delta)
    if distance<1e-8:return rows
    forward=delta/distance;side=np.array([-forward[1],forward[0]])
    candidates=[('goal',None)]
    for angle in (-60.,-35.,35.,60.):
        a=math.radians(angle);point=position+180.*(math.cos(a)*forward+math.sin(a)*side)
        if geometry.free(point):candidates.append((f'bearing_{angle:g}',point.tolist()))
    for name,waypoint in candidates:
        route=[position.tolist()]+([waypoint] if waypoint is not None else [])+[goal.tolist()]
        if not all(checked_edge(geometry,work,a,b) for a,b in zip(route,route[1:])):continue
        rows.append(evaluate(route,state,forecast,geometry,work,method,name))
    return rows


def feedback_plan(request,selected):
    state=request['state'];position=np.asarray(state['position_m']);goal=np.asarray(state['goal_m'])
    if len(selected['path'])==3:
        proposed=position+1.25*(np.asarray(selected['path'][1])-position)
    else:
        delta=goal-position;distance=np.linalg.norm(delta)
        proposed=position+min(225.,distance)*delta/max(distance,1e-20)
    if (np.any(proposed<0) or np.any(proposed>request['bounds_m'][1])
            or np.linalg.norm(goal-proposed)>=np.linalg.norm(goal-position)-1e-8
            or np.linalg.norm(proposed-goal)<1e-8):
        return dict(request_id=request['request_id'],action='keep',waypoint_m=None,sample_quotas=[],sampling='uniform',reason='no valid new forward extension')
    left=request['limits']['max_samples'];a=left//2
    return dict(request_id=request['request_id'],action='via',waypoint_m=proposed.tolist(),sample_quotas=[a,left-a],sampling='corridor',reason='extend best geometric heading by fixed factor 1.25')


def execute(request,raw,live_state,forecast,geometry,work,method,seed):
    if state_request(live_state,forecast,geometry,work)['request_id']!=request['request_id']:
        raise ValueError('state changed before proposal execution')
    p=validate_search_plan(request,raw);rows=[];searches=[];short=None
    if p['action']=='keep':return dict(plan=p,candidates=[],searches=[],shortcut=None,resources=work.summary())
    endpoints=[live_state['position_m']]+([p['waypoint_m']] if p['action']=='via' else [])+[live_state['goal_m']]
    candidate=[];objective=FrozenSearchObjective(geometry,forecast,live_state['speed_mps'],work,method)
    try:
        for leg,(a,b,quota) in enumerate(zip(endpoints,endpoints[1:],p['sample_quotas'])):
            if checked_edge(geometry,work,a,b):path=[a,b]
            else:
                # Both legs share ledger; reserve room for candidate geometry checks.
                prior=work.reserved_edges;work.reserved_edges=min(744,max(0,work.edge_limit-work.edges))
                if p['sampling']=='uniform':sampling=dict(focus_probability=0.,regions=[])
                else:
                    low=np.maximum(0.,np.minimum(a,b)-40.);high=np.minimum(geometry.high,np.maximum(a,b)+40.)
                    sampling=dict(focus_probability=.6,regions=[dict(low_m=low.tolist(),high_m=high.tolist(),weight=1.)])
                try:solved,stats=repair_rrt_star(objective,a,b,quota,seed+leg,sampling)
                finally:work.reserved_edges=prior
                searches.append(dict(leg=leg,seed=seed+leg,sampling=sampling,sampling_counts=stats,**asdict(solved)));path=solved.path
                if not path:return dict(plan=p,candidates=[],searches=searches,shortcut=None,resources=work.summary(),failure='search_failed')
            candidate.extend(path if not candidate else path[1:])
        rows.append(evaluate(candidate,live_state,forecast,geometry,work,method,'proposal_raw'))
        simplified,short=shortcut(candidate,geometry,work)
        if simplified!=candidate:rows.append(evaluate(simplified,live_state,forecast,geometry,work,method,'proposal_shortcut'))
    except BudgetExhausted as exc:
        return dict(plan=p,candidates=rows,searches=searches,shortcut=short,resources=work.summary(),failure=str(exc))
    return dict(plan=p,candidates=rows,searches=searches,shortcut=short,resources=work.summary())
