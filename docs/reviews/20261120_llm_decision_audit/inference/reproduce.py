"""Pure component review; invented measurements only, no world/flight/output access."""
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from uav_em.config import MapConfig, BeliefConfig
from uav_em.environment import Geometry
from uav_em.event_inference import Budget, fit_job, in_region
from uav_em.equivalent_field import kernel, power_and_jacobian
from uav_em.joint_prediction import JointSettings, local_prediction
from uav_em.point_gp import FrozenGP
from uav_em.source_geometry import GeometrySettings, constrained_refine_fit


def base(center, scale=40., velocity=(0., 0.), rate=0.):
    return dict(center_m=list(center), scale_m=scale,
                velocity_mps=list(velocity), rate_db_per_second=rate)


def support_counterexample():
    geometry = Geometry(MapConfig(columns=31, rows=31, resolution_m=20., obstacles_m=(), buildings=()))
    settings = replace(BeliefConfig(), support_radius_m=80.)
    history = [dict(position_m=[100., 100.], time_seconds=0., interference_dbm=-85.),
               dict(position_m=[140., 100.], time_seconds=30., interference_dbm=-75.)]
    query = [[100., 100.]]
    time = [36.]
    gp = FrozenGP(history, geometry, 30., settings, -100.)
    both = local_prediction(gp, query, time, JointSettings())
    recent_only = local_prediction(FrozenGP(history[1:], geometry, 30., settings, -100.), query, time, JointSettings())
    return dict(query=query[0], time_seconds=time[0],
        eligible_point_indices=[i for i,p in enumerate(history)
            if np.linalg.norm(np.array(query[0])-p['position_m'])<=settings.support_radius_m
            and time[0]-p['time_seconds']<=settings.max_age_seconds],
        with_old_and_recent={k: np.asarray(v).tolist() for k,v in both.items()},
        with_recent_only={k: np.asarray(v).tolist() for k,v in recent_only.items()})


def warm_counterexample():
    public = dict(goal_m=[500., 300.], map=dict(bounds_m=[[0., 0.], [600., 600.]]))
    # Both old and useful replacement centers are legal, and the replacement is
    # already present in the unmodified finite dictionary.
    xy = [[x, y] for y in (220., 260., 300.) for x in (160., 200., 240., 280.)]
    xy[-1] = [300., 300.]
    times = np.arange(len(xy), dtype=float)
    truth_base = base([220., 300.])
    values = power_and_jacobian(kernel([truth_base], xy, times, times[-1]), [-65.])[0]
    history = [dict(position_m=p, time_seconds=float(t), interference_dbm=float(y))
               for p,t,y in zip(xy,times,values)]
    old = dict(id='old', family='static', std_db=4., born_index=0, scored_points=11,
               reference_time_seconds=0., bases=[base([380., 300.])], peaks_dbm=[-65.])
    result = {}
    for capacity in (1, 2):
        old_version=deepcopy(old)
        if capacity==2:
            old_version['bases'].append(base([300.,380.]))
            old_version['peaks_dbm'].append(-70.)
        job = dict(family='static', region='task_disk', max_bases=capacity)
        for name, models in [('warm', [old_version]), ('cold', [])]:
            bank = SimpleNamespace(history=history, models=models, log_weights=np.zeros(len(models)))
            budget = Budget(); budget.reserve([job])
            fit = fit_job(bank, public, job, budget, name)
            result[f'{name}_capacity_{capacity}'] = dict(mse_db2=fit['model']['fit_mse'], bases=fit['model']['bases'],
                peaks_dbm=fit['model']['peaks_dbm'], warm_count=fit['warm_start_bases'],
                solver_calls=budget.solver_calls, reserved_solver_calls=budget.reserved_calls, trace=fit['trace'],
                replacement_in_dictionary=truth_base in fit['dictionary'])
    result['both_centers_legal'] = all(in_region(b['center_m'], public, xy[-1], 'task_disk')
                                    for b in [truth_base, *old_version['bases']])
    return result


def incomplete_solver_counterexample():
    public = dict(goal_m=[480., 280.], map=dict(bounds_m=[[0., 0.], [500., 500.]]))
    model = dict(id='seed', family='static', std_db=4., born_index=11, scored_points=0,
        reference_time_seconds=11., bases=[base([220., 240.], 80.)], peaks_dbm=[-72.], fit_mse=0.)
    target = deepcopy(model); target['bases'][0].update(center_m=[245., 225.], scale_m=70.)
    xy = [[x,y] for y in (180.,230.,280.) for x in (180.,220.,260.,300.)]
    times = np.arange(len(xy), dtype=float)
    values = power_and_jacobian(kernel(target['bases'],xy,times,11.),target['peaks_dbm'])[0]
    history=[dict(position_m=p,time_seconds=float(t),interference_dbm=float(y)) for p,t,y in zip(xy,times,values)]
    fit=dict(job=dict(family='static',region='task_disk',max_bases=1),model=model,window=history,
        bounds_dbm=[-140.,-60.],gp_soft_constraints=None,fit_objective=0.)
    rows = []
    for cap in (2,4,6,8,10,20,40):
        refined = constrained_refine_fit(fit,public,GeometrySettings(max_evaluations=cap))
        trace=refined['geometry_refinement']
        candidates=[]
        for row in trace['objective_trace']:
            p=np.array(row['parameters'])
            legal=bool(np.all(p>=trace['lower_bounds']) and np.all(p<=trace['upper_bounds'])
                and in_region(p[1:3],public,xy[-1],'task_disk'))
            if legal:candidates.append(row['objective'])
        rows.append(dict(cap=cap,accepted=trace['accepted'],error=trace.get('error'),
            solver_success=trace.get('solver_success'),previous_objective=trace.get('previous_objective'),
            best_evaluated_legal_objective=min(candidates) if candidates else None,
            candidate_objective=trace.get('candidate_objective'),seed_retained=refined['model']==model,
            resources=trace['resources']))
    return rows


def default_budget_counterexample():
    public=dict(goal_m=[480.,280.],map=dict(bounds_m=[[0.,0.],[500.,500.]]))
    model=dict(id='seed2',family='static',std_db=4.,born_index=11,scored_points=0,
        reference_time_seconds=11.,bases=[base([200.,240.],80.),base([280.,210.],70.)],
        peaks_dbm=[-72.,-76.],fit_mse=0.)
    target=deepcopy(model)
    target['bases'][0].update(center_m=[239.,220.],scale_m=57.)
    target['bases'][1].update(center_m=[250.,245.],scale_m=98.)
    xy=[[x,y] for y in (180.,230.,280.) for x in (180.,220.,260.,300.)]
    times=np.arange(len(xy),dtype=float)
    values=power_and_jacobian(kernel(target['bases'],xy,times,11.),target['peaks_dbm'])[0]
    history=[dict(position_m=p,time_seconds=float(t),interference_dbm=float(y)) for p,t,y in zip(xy,times,values)]
    fit=dict(job=dict(family='static',region='task_disk',max_bases=2),model=model,window=history,
        bounds_dbm=[-140.,-60.],gp_soft_constraints=None,fit_objective=0.)
    out=constrained_refine_fit(fit,public);trace=out['geometry_refinement']
    legal=[]
    for row in trace['objective_trace']:
        p=np.array(row['parameters'])
        if (np.all(p>=trace['lower_bounds']) and np.all(p<=trace['upper_bounds'])
            and all(in_region(q,public,xy[-1],'task_disk') for q in p[2:6].reshape(2,2))):
            legal.append(row['objective'])
    return dict(previous_objective=trace.get('previous_objective'),best_evaluated_legal_objective=min(legal),
                accepted=trace['accepted'],error=trace.get('error'),solver_success=trace.get('solver_success'),
                candidate_objective=trace.get('candidate_objective'),seed_retained=out['model']==model,
                resources=trace['resources'])


result=dict(support=support_counterexample(), warm_start=warm_counterexample(),
            convergence_gate=incomplete_solver_counterexample(),default_budget=default_budget_counterexample())
path=Path(__file__).with_name('evidence.json')
path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(result, indent=2, allow_nan=False))
