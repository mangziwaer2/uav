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
    def partial(reason):
        return out+route[i+1:],dict(checks=checks,completed=False,stop_reason=reason)
    while i<len(route)-1:
        chosen=i+1
        for j in range(len(route)-1,i+1,-1):
            if checks>=max_checks:return partial('shortcut_check_limit')
            try:free=checked_edge(geometry,work,route[i],route[j])
            except BudgetExhausted as exc:return partial(str(exc))
            checks+=1
            if free:chosen=j;break
        out.append(route[chosen]);i=chosen
    return out,dict(checks=checks,completed=True)


def best(records):
    valid=[r for r in records if r['feasible']]
    if not valid:raise ValueError('no feasible retained route')
    return min(valid,key=lambda r:(r['score']['score_seconds'],path_length(r['path']),r['id']))


def pool(state,forecast,geometry,work,method):
    try:rows=[evaluate(state['route_m'],state,forecast,geometry,work,method,'incumbent')]
    except BudgetExhausted as exc:
        rows=[dict(id='incumbent',path=deepcopy(state['route_m']),feasible=False,reason=str(exc))]
    position=np.asarray(state['position_m']);goal=np.asarray(state['goal_m']);delta=goal-position;distance=np.linalg.norm(delta)
    if distance<1e-8:return rows
    forward=delta/distance;side=np.array([-forward[1],forward[0]])
    candidates=[('goal',None)]
    for angle in (-60.,-35.,35.,60.):
        a=math.radians(angle);point=position+180.*(math.cos(a)*forward+math.sin(a)*side)
        if geometry.free(point):candidates.append((f'bearing_{angle:g}',point.tolist()))
    for name,waypoint in candidates:
        route=[position.tolist()]+([waypoint] if waypoint is not None else [])+[goal.tolist()]
        try:
            if not all(checked_edge(geometry,work,a,b) for a,b in zip(route,route[1:])):continue
            rows.append(evaluate(route,state,forecast,geometry,work,method,name))
        except BudgetExhausted as exc:
            rows.append(dict(id=name,path=route,feasible=False,reason=str(exc)));break
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


def search_waypoints(state,forecast,geometry,work,method,max_proposals=50):
    """Bounded task-geometric lateral proposals; no truth or model refitting.

    Each attempted waypoint consumes the same proposal ledger used by RRT.
    Blocked two-segment routes are rejected; no unbudgeted repair is hidden here.
    """
    if type(max_proposals) is not int or not 1<=max_proposals<=50:
        raise ValueError('waypoint proposal limit must be in [1,50]')
    before=work.summary();rows=[];attempts=0;stop=None
    position=np.asarray(state['position_m'],float);goal=np.asarray(state['goal_m'],float)
    delta=goal-position;distance=float(np.linalg.norm(delta))
    if distance<=1e-8:
        return dict(candidates=rows,proposals=0,resources_before=before,resources_after=work.summary(),stop_reason='at_goal')
    forward=delta/distance;side=np.array([-forward[1],forward[0]])
    try:
        for fraction in (.25,.5,.75,.1,.9):
            for offset in (20.,-20.,40.,-40.,80.,-80.,120.,-120.,160.,-160.):
                if attempts>=max_proposals:
                    stop='proposal_limit';break
                work.take_sample();attempts+=1
                waypoint=position+fraction*delta+offset*side
                identifier=f'lattice_{attempts:02d}'
                path=[position.tolist(),waypoint.tolist(),goal.tolist()]
                if np.any(waypoint<0) or np.any(waypoint>geometry.high):
                    row=dict(id=identifier,path=path,feasible=False,reason='waypoint_outside_map')
                else:
                    row=evaluate(path,state,forecast,geometry,work,method,identifier)
                row['proposal']=dict(kind='deterministic_lateral_waypoint',fraction=fraction,offset_m=offset)
                rows.append(row)
            if stop:break
    except BudgetExhausted as exc:
        stop=str(exc)
    return dict(candidates=rows,proposals=attempts,resources_before=before,
                resources_after=work.summary(),stop_reason=stop,
                forecast_sha256=forecast.fingerprint)


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
                # Protect future legs and raw-route verification within the same ledger.
                # A quota-q RRT path has at most q+1 edges; solved prefixes use actual size.
                prior=work.reserved_edges;available=work.edge_limit-work.edges-prior
                retained_edges=max(0,len(candidate)-1)
                verification=retained_edges+sum(q+1 for q in p['sample_quotas'][leg:])
                legs_left=len(endpoints)-1-leg
                search_edges=max(0,available-verification)//legs_left
                work.reserved_edges=prior+available-search_edges
                if p['sampling']=='uniform':sampling=dict(focus_probability=0.,regions=[])
                else:
                    low=np.maximum(0.,np.minimum(a,b)-40.);high=np.minimum(geometry.high,np.maximum(a,b)+40.)
                    sampling=dict(focus_probability=.6,regions=[dict(low_m=low.tolist(),high_m=high.tolist(),weight=1.)])
                try:solved,stats=repair_rrt_star(objective,a,b,quota,seed+leg,sampling)
                finally:work.reserved_edges=prior
                searches.append(dict(leg=leg,seed=seed+leg,sampling=sampling,sampling_counts=stats,
                    search_edge_allowance=search_edges,verification_edge_allowance=verification,
                    construction_policy='balanced_leg_search_v2',**asdict(solved)));path=solved.path
                if not path:return dict(plan=p,candidates=[],searches=searches,shortcut=None,resources=work.summary(),failure='search_failed')
            candidate.extend(path if not candidate else path[1:])
        rows.append(evaluate(candidate,live_state,forecast,geometry,work,method,'proposal_raw'))
        # A shortened path has no more edges than raw; keep enough to verify it.
        prior=work.reserved_edges
        work.reserved_edges=prior+min(len(candidate)-1,max(0,work.edge_limit-work.edges-prior))
        try:simplified,short=shortcut(candidate,geometry,work)
        finally:work.reserved_edges=prior
        if simplified!=candidate:rows.append(evaluate(simplified,live_state,forecast,geometry,work,method,'proposal_shortcut'))
    except BudgetExhausted as exc:
        return dict(plan=p,candidates=rows,searches=searches,shortcut=short,resources=work.summary(),failure=str(exc))
    return dict(plan=p,candidates=rows,searches=searches,shortcut=short,resources=work.summary())
