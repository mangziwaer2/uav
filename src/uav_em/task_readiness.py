"""仅当前GP的开发任务诊断与预声明候选评分；不读取真值或改变控制目标。"""
from __future__ import annotations
import math
import numpy as np
from .repair_search import FixedObjective, WorkBudget

SCORE_NAMES=('current_j','gaussian_expected_j','one_std_envelope_j')


def gaussian_positive_part(mean, std, threshold):
    """E[(X-threshold)+]，X为假设的高斯dBm变量；不是实测风险保证。"""
    mean,std=np.broadcast_arrays(np.asarray(mean,float),np.asarray(std,float))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<0).any() or not math.isfinite(threshold):
        raise ValueError('finite means/nonnegative std and finite threshold required')
    safe=np.where(std>0,std,1.)
    z=(mean-threshold)/safe
    cdf=np.fromiter((.5*math.erfc(-float(x)/math.sqrt(2)) for x in z.flat),float,z.size).reshape(z.shape)
    density=np.exp(-.5*z*z)/math.sqrt(2*math.pi)
    value=(mean-threshold)*cdf+std*density
    return np.where(std>0,np.maximum(0,value),np.maximum(0,mean-threshold))


def expected_clipped_excess(mean,std,reference):
    """E[clip((X-reference)/10,0,4)]，通过两个正部期望之差精确计算。"""
    value=(gaussian_positive_part(mean,std,reference)-gaussian_positive_part(mean,std,reference+40))/10
    return np.clip(value,0,4)


def score_route(snapshot,route,reference_dbm=-80.,objective=None):
    """沿最近邻格精确积分，三种评分共用几何/权重/路线，只有干扰项不同。"""
    problem=objective or FixedObjective(snapshot,reference_dbm,WorkBudget(2100,100000))
    metrics=problem.route_metrics(route)
    if metrics is None:return None
    expected=envelope=known=age_sum=length=0.
    for a,b in zip(route,route[1:]):
        weights,nodes=problem.quadrature(a,b);rows,cols=nodes[:,1],nodes[:,0]
        mean=snapshot.estimate_dbm[rows,cols];std=snapshot.uncertainty_db[rows,cols]
        mask=snapshot.supported[rows,cols];age=snapshot.age_seconds[rows,cols]
        expected+=float(weights@expected_clipped_excess(mean,std,reference_dbm))*2
        envelope+=float(weights@np.clip((mean+std-reference_dbm)/10,0,4))*2
        known+=float(weights@mask);length+=float(weights.sum())
        age_sum+=float(weights[mask]@age[mask])
    common=metrics['length_m']+metrics['uncertainty_cost']
    return {'current_j':metrics['objective'],'gaussian_expected_j':common+expected,
            'one_std_envelope_j':common+envelope,'length_m':metrics['length_m'],
            'interference_cost_current':metrics['interference_cost'],
            'interference_cost_expected':expected,'interference_cost_envelope':envelope,
            'uncertainty_cost':metrics['uncertainty_cost'],
            'supported_fraction':known/max(length,1e-12),
            'mean_supported_age_seconds':age_sum/known if known>0 else None,
            'uncertainty_is_calibrated':False,'field_at_arrival_times_predicted':False}


def public_risk(context, route_metrics=None):
    reference=context['electromagnetic_estimate']['interference_reference_dbm']
    points=context.get('measurement_evidence',{}).get('recent_points',[])
    now=context['time_seconds']
    if any(p['time_seconds']>now+1e-3 for p in points):raise ValueError('future point in public evidence')
    latest=max(points,key=lambda p:p['time_seconds']) if points else None
    # A previously strong reading is a historical fact, not a claim the power is still high here.
    return {'latest_measurement_dbm':latest['interference_dbm'] if latest else None,
            'latest_measurement_above_reference':bool(latest and latest['interference_dbm']>=reference),
            'recent_observed_above_reference':any(p['interference_dbm']>=reference for p in points),
            'supported_remaining_route_penalty_positive':bool(route_metrics and route_metrics['interference_cost']>1e-7),
            'not_source_location_or_direction':True}


def choose_from_pool(records,length_ratio_limit=1.25):
    """当前评分选路，真值字段即使存在也不读取；保留同分incumbent。"""
    base=next(r for r in records if r['id']=='incumbent')
    cap=base['metrics']['length_m']*length_ratio_limit
    eligible=[r for r in records if r['feasible'] and r['metrics']['length_m']<=cap+1e-8]
    choices={}
    for score in SCORE_NAMES:
        best=base
        for r in eligible:
            if r['scores'][score]<best['scores'][score]-1e-7:best=r
        choices[score]=best['id']
    return {'choices':choices,'length_ratio_limit':length_ratio_limit,'length_cap_m':cap,
            'eligible_ids':[r['id'] for r in eligible],
            'selection_inputs':'frozen_current_gp_scores_geometry_energy_length_only_no_truth',
            'pool_is_offline_diagnostic_not_budget_matched_online_algorithm':True}
