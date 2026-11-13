"""只用冻结GP与公开几何构造诊断候选；不接收world或未来信息。"""
from __future__ import annotations
from copy import deepcopy
import json
import numpy as np

from .repair_program import OperatorLibrary
from .repair_search import FixedObjective, WorkBudget
from .route_repair import RepairSession, RepairSettings, RouteRepairOptimizer, anchors


SEARCH_PROFILES = ('global_uniform', 'global_focus', 'local_pad1', 'local_pad3',
                   'via_left_near', 'via_right_near')


def splice_interval(route, i, j, local):
    """保留区间外所有拐点；删除零长相邻点，保留整体端点。"""
    points = list(route[:i]) + list(local) + list(route[j+1:])
    result = []
    for point in points:
        point = [float(v) for v in point]
        if not result or np.linalg.norm(np.array(point)-result[-1]) > 1e-9:
            result.append(point)
    return result


def candidate_record(identifier, family, path, problem, max_length_m, **metadata):
    metrics = problem.route_metrics(path)
    reason = None
    if metrics is None:
        reason = 'geometry_invalid'
    elif metrics['length_m'] > max_length_m + 1e-8:
        reason = 'energy_budget_exceeded'
    return {'id':identifier, 'family':family, 'path':deepcopy(path),
            'metrics':metrics, 'feasible':reason is None, 'failure':reason, **metadata}


def geometric_candidates(snapshot, context, route, max_length_m, settings=RepairSettings()):
    """固定枚举几何候选，供覆盖诊断；累计池评价不冒充单次在线搜索预算。"""
    session = RepairSession(snapshot, context, route, max_length_m, settings, 0)
    if session.best is None:
        raise ValueError('diagnostics require a feasible recorded incumbent')
    ctx = session.request(1, 1);lib = OperatorLibrary(ctx);base = session.best
    work = WorkBudget(settings.sample_limit, settings.edge_limit)
    objective = FixedObjective(snapshot, session.problem.reference, work)
    records = [candidate_record('incumbent', 'incumbent', base, objective, max_length_m)]
    intervals = {'whole':lib.whole(), 'front_pad1':lib.select_interval('front',1)}
    for padding in (0,1,3):
        intervals[f'risk_pad{padding}'] = lib.select_interval('interference',padding)
    seen_intervals = set()
    high = np.array(ctx['map']['bounds_m'][1],float)
    radius = ctx['electromagnetic_estimate']['estimator']['local_prediction_radius_m']
    for name,interval in intervals.items():
        if tuple(interval) in seen_intervals:
            continue
        seen_intervals.add(tuple(interval))
        i,j = (int(k[1:]) for k in interval)
        a,b = np.array(base[i]),np.array(base[j]);delta=b-a;length=np.linalg.norm(delta)
        if length < 1e-8:
            continue
        normal = np.array([-delta[1],delta[0]])/length
        choices = [('chord',[a,b],{})]
        for scale in (.5,1.,1.5,2.):
            for side,sign in (('left',1),('right',-1)):
                offset=normal*radius*scale*sign
                midpoint=np.clip((a+b)/2+offset,0,high)
                first,second=np.clip(a+offset,0,high),np.clip(b+offset,0,high)
                params={'side':side,'offset_m':radius*scale}
                choices.append((f'via_{side}_{scale:g}',[a,midpoint,b],params))
                choices.append((f'parallel_{side}_{scale:g}',[a,first,second,b],params))
        for suffix,local,params in choices:
            path=splice_interval(base,i,j,local)
            records.append(candidate_record(name+'_'+suffix,'geometric',path,objective,max_length_m,
                interval=interval,construction=suffix,parameters=params))
    # Pool construction never sees truth; record invalid/overlength candidates too.
    return records,ctx,work.summary()


def search_proposal(ctx, profile, sample_budget=700):
    lib=OperatorLibrary(ctx)
    if profile=='global_uniform':
        return lib.global_rrt(lib.uniform(),sample_budget)
    if profile=='global_focus':
        distribution=lib.sampling(lib.corridors(lib.whole(),side='both',scale='wide'))
        return lib.global_rrt(distribution,sample_budget)
    interval=lib.select_interval('interference',3 if profile=='local_pad3' else 1)
    if profile in ('local_pad1','local_pad3'):
        distribution=lib.sampling(lib.corridors(interval,side='both',scale='local'))
        return lib.local_rrt(interval,distribution,sample_budget)
    if profile in ('via_left_near','via_right_near'):
        side='left' if profile=='via_left_near' else 'right'
        distribution=lib.sampling(lib.corridors(interval,side=side,scale='near'))
        return lib.via(interval,lib.waypoint(interval,side=side,scale='near'),distribution,sample_budget)
    raise ValueError('unknown diagnostic profile')


def search_candidate(snapshot,context,route,max_length_m,profile,seed,settings=RepairSettings(max_rounds=1)):
    """一次独立的同上限搜索；原始被拒绝候选与实际保留路线分开记录。"""
    class FixedProposal:
        last_call={}
        def decide(self,ctx):
            return json.dumps(search_proposal(ctx,profile,min(700,settings.sample_limit)),ensure_ascii=False)
    optimizer=RouteRepairOptimizer(settings=settings,numeric_policy=FixedProposal())
    result,report=optimizer.optimize(snapshot,context,route,max_length_m,seed,0)
    trace=report['trace'][0];proposal=trace['proposal'];candidate=[]
    if proposal is not None:
        searches=report['searches'];op=proposal['operator']
        if op=='global_rrt':
            candidate=searches[0]['path'] if searches else []
        else:
            base=[a['position_m'] for a in trace['context']['route_repair']['anchors']]
            i,j=(int(proposal[k][1:]) for k in ('start_id','end_id'))
            if op=='local_rrt':
                local=searches[0]['path'] if searches else []
            else:
                local=(searches[0]['path']+searches[1]['path'][1:]
                       if len(searches)==2 and all(s['path'] for s in searches) else [])
            candidate=splice_interval(base,i,j,local) if local else []
    candidate=anchors(candidate,settings.anchor_spacing_m) if candidate else []
    feedback=trace['feedback']
    return {'id':f'{profile}_seed{seed}', 'family':'rrt_search','profile':profile,'seed':seed,
        'path':candidate,'metrics':feedback.get('candidate_metrics'),
        'feasible':bool(candidate and feedback.get('candidate_metrics') is not None
                        and feedback['candidate_metrics']['length_m']<=max_length_m+1e-8),
        'failure':feedback.get('reason') if (not candidate or feedback.get('candidate_metrics') is None
                    or feedback['candidate_metrics']['length_m']>max_length_m+1e-8) else None,
        'accepted':bool(feedback['accepted']), 'actual_retained_path':result.path,
        'report':report}
