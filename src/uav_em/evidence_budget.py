"""公开风险证据与有限机动余量的人工过程；无真值/测向/新传感。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import math
import time
import numpy as np
from .decision_objective import ObjectiveOptimizer,ObjectiveSettings,select_candidate
from .planning import path_length
from .repair_process import canonical_path

VERSION='evidence_budget_process_v1'
METHODS=('pool_j','immediate_v2','evidence_only','evidence_budget')


def risk_evidence(context,window_seconds=10.):
    """只读显式点测字段；历史越参考不等于已确认当前位置持续强干扰。"""
    now=float(context['time_seconds'])
    reference=float(context['electromagnetic_estimate']['interference_reference_dbm'])
    if not math.isfinite(now) or not math.isfinite(reference) or not math.isfinite(window_seconds) or window_seconds<=0:
        raise ValueError('finite time/reference and positive evidence window required')
    points=list(context.get('measurement_evidence',{}).get('recent_points',[]))
    latest=context.get('latest_observation')
    if latest is not None:points.append(latest)
    public=[]
    for point in points:
        timestamp=float(point['time_seconds']);power=float(point['interference_dbm'])
        if not math.isfinite(timestamp) or not math.isfinite(power) or timestamp>now+1e-8:
            raise ValueError('invalid or future point measurement')
        if timestamp>=now-window_seconds-1e-8:
            public.append({'time_seconds':timestamp,'interference_dbm':power})
    # The latest observation may already be present in the summary.
    unique={tuple(p.values()):p for p in public};public=sorted(unique.values(),key=lambda p:p['time_seconds'])
    strong=[p for p in public if p['interference_dbm']>reference]
    return {'window_seconds':window_seconds,'reference_dbm':reference,'point_count':len(public),
        'above_reference_point_count':len(strong),'active':bool(strong),
        'latest_public_point':public[-1] if public else None,
        'latest_above_reference_time_seconds':strong[-1]['time_seconds'] if strong else None,
        'noise_significance_test':False,'source_location_or_bearing_inferred':False,
        'historical_measurement_is_current_field_guarantee':False}


def choose_process(records,method,context,flown_m,mission_cap_m,settings=ObjectiveSettings()):
    """纯函数有限过程，固定证据窗/半余量/支持风险条件，保存拒绝分解。"""
    if method not in METHODS:raise ValueError('unknown process method')
    evidence=risk_evidence(context,settings.horizon_seconds)
    baseline=select_candidate(records,'pool_j',settings)
    decision={'method':method,'evidence':evidence,'baseline_candidate_id':baseline['id'] if baseline else None,
              'executed_distance_m':flown_m,'mission_distance_cap_m':mission_cap_m,
              'new_observations':0,'future_truth_used':False,'candidate_checks':[]}
    if baseline is None:
        decision['branch']='no_feasible_candidate';return None,decision
    if not math.isfinite(flown_m) or flown_m<0 or mission_cap_m is None or not math.isfinite(mission_cap_m):
        raise ValueError('valid mission distance accounting required')
    valid=[r for r in records if r['feasible']]
    shortest=min(r['metrics']['length_m'] for r in valid)
    slack=max(0.,mission_cap_m-flown_m-shortest)
    local_cap=shortest+.5*slack
    decision.update(shortest_pool_remaining_m=shortest,remaining_maneuver_margin_m=slack,
        allocated_margin_fraction=.5,allocated_margin_m=.5*slack,local_length_cap_m=local_cap,
        baseline_retention_exception=baseline['metrics']['length_m']>local_cap+1e-8)
    if method=='pool_j' or method in ('evidence_only','evidence_budget') and not evidence['active']:
        decision['branch']='original_j' if method=='pool_j' else 'no_fresh_above_reference_evidence'
        return baseline,decision
    if method in ('immediate_v2','evidence_only'):
        decision['branch']='unrestricted_prefix_risk'
        return select_candidate(records,'prefix_v2',settings),decision
    allowed=[];tolerance=settings.risk_tolerance_db_seconds
    base_known=baseline['prefix']['supported_expected_excess_db_seconds'];base_length=baseline['metrics']['length_m']
    for r in valid:
        reasons=[];length=r['metrics']['length_m'];known=r['prefix']['supported_expected_excess_db_seconds']
        if r['id']!=baseline['id']:
            if length>local_cap+1e-8:reasons.append('preserve_half_remaining_margin')
            if known>base_known+tolerance+1e-8:reasons.append('supported_risk_proxy_regression')
            if length>base_length+1e-8 and known>base_known-tolerance+1e-8:
                reasons.append('extra_length_without_supported_risk_proxy_gain')
        if not reasons:allowed.append(r)
        decision['candidate_checks'].append({'id':r['id'],'allowed':not reasons,'reasons':reasons,
            'additional_length_vs_baseline_m':length-base_length,
            'supported_expected_excess_change_db_seconds':known-base_known,
            'unknown_seconds':r['prefix']['unknown_seconds']})
    decision.update(branch='risk_evidence_and_maneuver_margin',allowed_ids=[r['id'] for r in allowed])
    # Do not rewrite feasibility or the frozen underlying pool.
    return select_candidate(allowed,'prefix_v2',settings),decision


class EvidenceBudgetOptimizer:
    """组合原共同池后作有限选择；原模块/接受器保持原样。"""
    policy=None
    last_call={}

    def __init__(self,method,settings=ObjectiveSettings()):
        if method not in METHODS:raise ValueError('unknown process method')
        self.method,self.settings=method,settings
        self.base=ObjectiveOptimizer('pool_j',settings)

    def optimize(self,snapshot,context,route,max_length_m,seed,call_limit):
        started=time.perf_counter()
        baseline,report=self.base.optimize(snapshot,context,route,max_length_m,seed,call_limit)
        records=report['candidates']
        chosen,process=choose_process(records,self.method,context,report['executed_distance_m'],report['mission_distance_cap_m'],self.settings)
        resources=report['resources'];fallback=False
        if chosen:
            extra_edges=max(0,len(chosen['path'])-1)
            if resources['edges']+extra_edges>resources['edge_limit']:
                chosen=select_candidate(records,'pool_j',self.settings);fallback=True
                process['final_verification_fallback']='preserve_preverified_original_j_with_no_new_work'
            else:
                for a,b in zip(chosen['path'],chosen['path'][1:]):
                    if not snapshot.geometry.edge_free(a,b):raise RuntimeError('process final geometry invalid')
                resources['edges']+=extra_edges;resources['verification_edges']+=extra_edges
                resources['edges_left']=resources['total_edges_left']=resources['edge_limit']-resources['edges']
        path=deepcopy(chosen['path']) if chosen else []
        elapsed=time.perf_counter()-started
        baseline.path=path;baseline.length_m=path_length(path) if path else None
        baseline.cost=chosen['metrics']['objective'] if chosen else None
        baseline.reason='evidence_process_selected' if path else 'evidence_process_no_feasible_route'
        baseline.elapsed_seconds=elapsed
        process['selected_candidate_id']=chosen['id'] if chosen else None
        report.update(contract_version=VERSION,method=self.method,objective='fixed_v2_with_predeclared_process_restrictions',
            best_metrics=chosen['metrics'] if chosen else None,selected_prefix=chosen['prefix'] if chosen else None,
            selected_candidate_id=chosen['id'] if chosen else None,selection_settings=asdict(self.settings),
            numeric_compute_seconds=elapsed,has_final_route=bool(path),final_geometry_verified=bool(path),
            accepted_proposals=int(bool(path) and (not route or canonical_path(path)!=canonical_path(route))),
            process_selection=process,selection_resource_fallback=fallback,
            baseline_verification_work_retained=True,process_candidate_pool_is_unchanged=True)
        return baseline,report
