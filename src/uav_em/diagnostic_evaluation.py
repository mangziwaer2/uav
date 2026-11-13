"""仅离线事后评价冻结候选；本模块不向在线策略提供真值或未来。"""
from __future__ import annotations
import math
import numpy as np
from .propagation import source_received_dbm


def truth_at_times(world, points, times):
    """逐点不同时间的批量真值，使用原传播函数；不产生额外传感读数。"""
    points=np.asarray(points,float);times=np.asarray(times,float)
    if points.ndim!=2 or points.shape[1]!=2 or times.shape!=(len(points),):
        raise ValueError('points Nx2 and times N required')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(times)):
        raise ValueError('finite points/times required')
    cfg=world.config
    power=np.full(len(points),10**(cfg.electromagnetic.background_dbm/10))
    for source in cfg.electromagnetic.fields:
        active=(times>=source['start_seconds'])&(times<source['end_seconds'])
        if not active.any():continue
        centers=np.array(source['center_m'])+(times[active]-source['start_seconds'])[:,None]*np.array(source['velocity_mps'])
        power[active]+=10**(source_received_dbm(points[active],centers,source,cfg,world.propagation)/10)
    return 10*np.log10(power)


def route_samples(route,speed_mps,step_seconds=.25):
    """按边中点时间积分；返回点、相对飞行时刻、每点代表时长。"""
    if not math.isfinite(speed_mps) or speed_mps<=0 or not math.isfinite(step_seconds) or step_seconds<=0:
        raise ValueError('positive speed/step required')
    points=[];times=[];weights=[];elapsed=0.
    for a,b in zip(route,route[1:]):
        a,b=np.array(a,float),np.array(b,float);duration=float(np.linalg.norm(b-a))/speed_mps
        if duration<=1e-12:continue
        count=max(1,int(math.ceil(duration/step_seconds)))
        fractions=(np.arange(count)+.5)/count
        points.extend((a+fractions[:,None]*(b-a)).tolist())
        times.extend((elapsed+fractions*duration).tolist());weights.extend([duration/count]*count)
        elapsed+=duration
    return np.asarray(points,float).reshape(-1,2),np.asarray(times),np.asarray(weights)


def evaluate_fixed_route(world,snapshot,route,departure_delay_seconds=0.5,step_seconds=.25):
    """无补测/重规划的固定路线反事实；真实评价不回传候选生成器。"""
    cfg=world.config
    points,offsets,weights=route_samples(route,cfg.aircraft.speed_mps,step_seconds)
    if not len(points):raise ValueError('nonempty flight route required')
    t=snapshot.time_seconds;reference=cfg.electromagnetic.interference_reference_dbm
    current=np.asarray(world.truth(points,t))
    future=truth_at_times(world,points,t+departure_delay_seconds+offsets)
    columns,rows=np.rint(points/snapshot.geometry.config.resolution_m).astype(int).T
    mean=snapshot.estimate_dbm[rows,columns];supported=snapshot.supported[rows,columns]
    std=snapshot.uncertainty_db[rows,columns]
    excess=np.maximum(0,future-reference);total=float(weights@excess)
    supported_duration=float(weights@supported)
    return {'evaluation_kind':'fixed_route_open_loop_counterfactual_not_closed_loop_policy',
        'departure_time_seconds':t+departure_delay_seconds,'flight_seconds':float(weights.sum()),
        'length_m':float(weights.sum()*cfg.aircraft.speed_mps),
        'future_exposure_seconds':float(weights@(future>reference)),
        'future_excess_db_seconds':total,
        'frozen_true_field_excess_db_seconds':float(weights@np.maximum(0,current-reference)),
        'frozen_gp_excess_db_seconds':float(weights@np.maximum(0,mean-reference)),
        'future_excess_on_currently_unsupported_db_seconds':float(weights@np.where(supported,0,excess)),
        'supported_fraction_at_decision':supported_duration/float(weights.sum()),
        'supported_mae_now_db':float(weights[supported]@np.abs(mean[supported]-current[supported])/supported_duration) if supported_duration else None,
        'supported_mean_std_db':float(weights[supported]@std[supported]/supported_duration) if supported_duration else None,
        'maximum_sample_step_seconds':step_seconds,'decision_hover_exposure_included':False,
        'new_observations':0}


def diagnose_candidates(records, length_ratio_limit=1.25, material_reduction_fraction=.1):
    """有限池的描述性诊断，不声称在线可选、全局最优或因果唯一归因。"""
    incumbent=next(r for r in records if r['id']=='incumbent')
    feasible=[r for r in records if r['feasible'] and r.get('evaluation')]
    cap=incumbent['metrics']['length_m']*length_ratio_limit
    eligible=[r for r in feasible if r['metrics']['length_m']<=cap+1e-8]
    best_j=min(feasible,key=lambda r:(r['metrics']['objective'],r['id']))
    best_exposure=min(eligible,key=lambda r:(r['evaluation']['future_excess_db_seconds'],r['metrics']['objective'],r['id']))
    base_j=incumbent['metrics']['objective'];base_e=incumbent['evaluation']['future_excess_db_seconds']
    material=[r for r in eligible if base_e>1e-8 and
              r['evaluation']['future_excess_db_seconds']<=base_e*(1-material_reduction_fraction)]
    rejected=[r for r in material if r['metrics']['objective']>=base_j-1e-7 and r['id']!='incumbent']
    searches=[r for r in feasible if r['family']=='rrt_search']
    best_search_j=min([base_j]+[r['metrics']['objective'] for r in searches])
    return {'incumbent_id':'incumbent','best_current_j_id':best_j['id'],
        'posthoc_lowest_exposure_id':best_exposure['id'],
        'length_ratio_limit':length_ratio_limit,'length_cap_m':cap,
        'material_reduction_fraction':material_reduction_fraction,
        'feasible_count':len(feasible),'eligible_count':len(eligible),
        'material_exposure_improvement_ids':[r['id'] for r in material],
        'material_improvements_rejected_by_current_j_ids':[r['id'] for r in rejected],
        'finite_pool_search_coverage_gap':best_j['metrics']['objective']<best_search_j-1e-7,
        'best_independent_search_or_incumbent_j':best_search_j,
        'acceptance_task_tradeoff_evidence':bool(rejected),
        'current_j_choice_future_excess_db_seconds':best_j['evaluation']['future_excess_db_seconds'],
        'incumbent_future_excess_db_seconds':base_e,
        'posthoc_lowest_excess_db_seconds':best_exposure['evaluation']['future_excess_db_seconds'],
        'interpretation':'finite frozen candidate pool; future truth is evaluation-only; no closed-loop or optimality claim'}
