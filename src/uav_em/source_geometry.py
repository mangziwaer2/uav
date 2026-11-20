"""Bounded equivalent-source refinement; no simulator, GP fitting or LLM access."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import least_squares, minimize

from .equivalent_field import kernel, power_and_jacobian
from .event_inference import in_region


@dataclass(frozen=True)
class GeometrySettings:
    center_radius_m: float = 40.
    scale_ratio: float = math.sqrt(2.)
    minimum_scale_m: float = 20.
    maximum_scale_m: float = 240.
    max_evaluations: int = 40
    max_kernel_terms: int = 5000
    minimum_improvement: float = 1e-8

    def __post_init__(self):
        for name in ('center_radius_m','scale_ratio','minimum_scale_m','maximum_scale_m','minimum_improvement'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError('positive finite refinement settings required')
        if self.scale_ratio<=1 or self.minimum_scale_m>=self.maximum_scale_m:
            raise ValueError('nonempty scale bounds required')
        if type(self.max_evaluations) is not int or not 1<=self.max_evaluations<=40:
            raise ValueError('bounded evaluation limit required')
        if type(self.max_kernel_terms) is not int or not 1<=self.max_kernel_terms<=5000:
            raise ValueError('bounded kernel limit required')


def pack(model,mode):
    amplitudes=np.asarray(model['peaks_dbm'],float)
    if mode=='amplitude':return amplitudes
    if mode!='geometry':raise ValueError('unknown refinement mode')
    return np.r_[amplitudes,np.asarray([b['center_m'] for b in model['bases']]).ravel(),
                 np.log([b['scale_m'] for b in model['bases']])]


def unpack(model,parameters,mode):
    result=deepcopy(model);count=len(model['bases'])
    result['peaks_dbm']=np.asarray(parameters[:count]).tolist()
    if mode=='geometry':
        centers=np.asarray(parameters[count:3*count]).reshape(count,2)
        scales=np.exp(parameters[3*count:])
        for b,center,scale in zip(result['bases'],centers,scales):
            b['center_m']=center.tolist();b['scale_m']=float(scale)
    return result


def prediction_jacobian(model,parameters,xy,times,mode):
    """Derivatives of dBm prediction wrt peaks, x/y centers, log scales."""
    current=unpack(model,parameters,mode)
    matrix=kernel(current['bases'],xy,times,current['reference_time_seconds'])
    mean,amplitude_jac=power_and_jacobian(matrix,current['peaks_dbm'])
    if mode=='amplitude':return mean,amplitude_jac
    count=len(current['bases']);dt=np.asarray(times)-current['reference_time_seconds']
    centers=np.asarray([b['center_m'] for b in current['bases']])
    speeds=np.asarray([b['velocity_mps'] for b in current['bases']])
    scales=np.asarray([b['scale_m'] for b in current['bases']])
    delta=np.asarray(xy)[:,None,:]-centers[None,:,:]-dt[:,None,None]*speeds[None,:,:]
    factor=10/math.log(10)*amplitude_jac
    center_jac=(factor[:,:,None]*delta/scales[None,:,None]**2).reshape(len(times),2*count)
    scale_jac=factor*np.sum(delta**2,axis=2)/scales[None,:]**2
    return mean,np.column_stack((amplitude_jac,center_jac,scale_jac))


def _parameter_list(parameters):
    return [float(value) if math.isfinite(value) else None for value in np.asarray(parameters).ravel()]


class _EvaluatedCandidates:
    """Retain objective-evaluated legal candidates without another prediction."""
    def __init__(self,model,public,origin,region,mode,low,high,trace):
        self.model,self.public,self.origin,self.region=model,public,origin,region
        self.mode,self.low,self.high,self.trace=mode,low,high,trace
        self.best=None

    def eligibility(self,parameters):
        finite=bool(np.all(np.isfinite(parameters)))
        bounds=finite and bool(np.all(parameters>=self.low) and np.all(parameters<=self.high))
        if not finite:return False,False
        current=unpack(self.model,parameters,self.mode)
        region=all(in_region(b['center_m'],self.public,self.origin,self.region) for b in current['bases'])
        return bounds,region

    def consider(self,parameters,prediction,loss,record=True):
        bounds,region=self.eligibility(parameters)
        legal=bool(bounds and region and math.isfinite(loss) and np.all(np.isfinite(prediction)))
        if record:
            self.trace['objective_trace'].append(dict(parameters=_parameter_list(parameters),
                objective=loss if math.isfinite(loss) else None,
                bounds_eligible=bounds,region_eligible=region,eligible=legal))
        if legal and (self.best is None or loss<self.best[2]):
            self.best=(np.array(parameters,copy=True),np.array(prediction,copy=True),loss)

    def apply(self,result,labels,real_count,soft,minimum_improvement):
        if self.best is None:return
        parameters,prediction,loss=self.best
        accepted=bool(loss<self.trace['previous_objective']-minimum_improvement)
        self.trace.update(final_parameters=parameters.tolist(),candidate_objective=loss,
            region_eligible=True,bounds_eligible=True,accepted=accepted,
            acceptance_reason='best_evaluated_feasible_improvement' if accepted else 'no_feasible_improvement')
        if accepted:
            current=unpack(self.model,parameters,self.mode)
            current['fit_mse']=float(np.mean((prediction[:real_count]-labels[:real_count])**2))
            result['model']=current;result['fit_objective']=loss
            result['gp_shape_mse']=(float(np.sum(np.asarray(soft['weights'])*(prediction[real_count:]-labels[real_count:])**2))
                if soft and soft['position_m'] else None)


def _check_acceptance_policy(policy):
    if policy not in ('best_feasible_v2','converged_v1'):
        raise ValueError('unknown refinement acceptance policy')


def refine_fit(fit,public,mode,settings=None,*,acceptance_policy='best_feasible_v2'):
    _check_acceptance_policy(acceptance_policy)
    settings=settings or GeometrySettings()
    if mode not in ('amplitude','geometry'):raise ValueError('unknown refinement mode')
    result=deepcopy(fit);model=result['model'];count=len(model['bases'])
    if not count:return result
    if model['scored_points']!=0:raise ValueError('refinement must precede birth evidence')
    history=fit['window'];now=history[-1]['time_seconds']
    if model['reference_time_seconds']!=now:raise ValueError('refinement requires original training reference')
    xy=[p['position_m'] for p in history];times=[p['time_seconds'] for p in history]
    labels=np.asarray([p['interference_dbm'] for p in history]);real_count=len(labels)
    weights=np.full(real_count,1/math.sqrt(real_count));soft=fit.get('gp_soft_constraints')
    if soft and soft['position_m']:
        if soft['prefix_end_seconds']!=now or soft['independent_evidence_points']!=0:
            raise ValueError('soft constraints must share the original fit prefix')
        xy=xy+soft['position_m'];times=times+soft['time_seconds']
        labels=np.r_[labels,soft['mean_dbm']]
        weights=np.r_[weights,np.sqrt(soft['strength']*np.asarray(soft['weights']))]
    xy=np.asarray(xy,float);times=np.asarray(times,float)
    initial=pack(model,mode);low=np.full(count,fit['bounds_dbm'][0]);high=np.full(count,fit['bounds_dbm'][1])
    if mode=='geometry':
        centers=np.asarray([b['center_m'] for b in model['bases']]);scales=np.asarray([b['scale_m'] for b in model['bases']])
        bounds=np.asarray(public['map']['bounds_m'])
        low=np.r_[low,np.maximum(bounds[0],centers-settings.center_radius_m).ravel(),
                  np.log(np.maximum(settings.minimum_scale_m,scales/settings.scale_ratio))]
        high=np.r_[high,np.minimum(bounds[1],centers+settings.center_radius_m).ravel(),
                   np.log(np.minimum(settings.maximum_scale_m,scales*settings.scale_ratio))]
    work=dict(solver_calls=0,residual_calls=0,jacobian_calls=0,kernel_terms=0)
    trace=dict(mode=mode,settings=asdict(settings),seed_model=deepcopy(model),accepted=False,
        acceptance_policy=acceptance_policy,solver_success=False,objective_trace=[],
        training_history_is_future_evidence=False,prefix_end_seconds=now,
        lower_bounds=low.tolist(),upper_bounds=high.tolist(),initial_parameters=initial.tolist())
    result['geometry_refinement']=trace
    candidates=_EvaluatedCandidates(model,public,history[-1]['position_m'],fit['job']['region'],mode,low,high,trace)
    cached=None
    def evaluate(parameters):
        nonlocal cached
        if cached is None or not np.array_equal(parameters,cached[0]):
            terms=len(times)*count
            if work['kernel_terms']+terms>settings.max_kernel_terms:raise ValueError('refinement kernel budget exhausted')
            work['kernel_terms']+=terms
            mean,jac=prediction_jacobian(model,parameters,xy,times,mode)
            cached=(np.array(parameters),mean,jac)
        return cached[1:]
    def objective(parameters):
        if work['residual_calls']>=settings.max_evaluations:raise ValueError('refinement residual budget exhausted')
        work['residual_calls']+=1
        prediction=evaluate(parameters)[0];residual=(prediction-labels)*weights
        candidates.consider(parameters,prediction,float(np.sum(residual**2)))
        return residual
    def jacobian(parameters):
        if work['jacobian_calls']>=settings.max_evaluations:raise ValueError('refinement Jacobian budget exhausted')
        work['jacobian_calls']+=1
        return evaluate(parameters)[1]*weights[:,None]
    try:
        if np.any(low>=high) or not all(candidates.eligibility(initial)):raise ValueError('seed outside refinement bounds/region')
        baseline=evaluate(initial)[0];old_loss=float(np.sum(((baseline-labels)*weights)**2))
        trace['previous_objective']=old_loss
        candidates.consider(initial,baseline,old_loss,record=False)
        work['solver_calls']=1
        solved=least_squares(objective,initial,jac=jacobian,bounds=(low,high),x_scale='jac',
            max_nfev=settings.max_evaluations,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        bounds_ok,eligible=candidates.eligibility(solved.x)
        trace.update(solver_final_parameters=_parameter_list(solved.x),solver_success=bool(solved.success),solver_status=int(solved.status),
            solver_final_region_eligible=eligible,solver_final_bounds_eligible=bounds_ok,
            nfev=int(solved.nfev),njev=int(solved.njev),
            optimality=float(solved.optimality) if math.isfinite(solved.optimality) else None)
        if acceptance_policy=='converged_v1':
            prediction=evaluate(solved.x)[0];loss=float(np.sum(((prediction-labels)*weights)**2))
            accepted=bool(solved.success and eligible and bounds_ok and loss<old_loss-settings.minimum_improvement)
            trace.update(final_parameters=solved.x.tolist(),candidate_objective=loss,
                accepted=accepted,region_eligible=eligible,bounds_eligible=bounds_ok)
            if accepted:
                refined=unpack(model,solved.x,mode)
                refined['fit_mse']=float(np.mean((prediction[:real_count]-labels[:real_count])**2))
                result['model']=refined;result['fit_objective']=loss
                result['gp_shape_mse']=(float(np.sum(np.asarray(soft['weights'])*(prediction[real_count:]-labels[real_count:])**2))
                    if soft and soft['position_m'] else None)
    except (ValueError,FloatingPointError) as exc:
        trace['error']=str(exc)
    trace['stop_reason']=trace.get('error') or ('solver_converged' if trace['solver_success'] else 'solver_nonconverged')
    if acceptance_policy=='best_feasible_v2':
        candidates.apply(result,labels,real_count,soft,settings.minimum_improvement)
    trace['resources']=work
    return result


def region_values_jacobian(parameters,count,origin,goal,region,margin_m=1e-6):
    """Signed physical-distance constraints (>=0), with analytic parameter Jacobian."""
    if region not in ('task_disk','left_half','right_half'):raise ValueError('unknown source region')
    centers=np.asarray(parameters[count:3*count]).reshape(count,2)
    delta=centers-np.asarray(origin);direction=np.asarray(goal)-origin
    values=(160.**2-np.sum(delta**2,axis=1))/320.-margin_m
    jac=np.zeros((count,4*count))
    for i in range(count):jac[i,count+2*i:count+2*i+2]=-delta[i]/160.
    norm=float(np.linalg.norm(direction))
    if region!='task_disk' and norm>0:
        normal=np.array([-direction[1],direction[0]])/norm
        if region=='right_half':normal=-normal
        half_jac=np.zeros_like(jac)
        for i in range(count):half_jac[i,count+2*i:count+2*i+2]=normal
        values=np.r_[values,delta@normal-margin_m];jac=np.vstack((jac,half_jac))
    return values,jac


def constrained_refine_fit(fit,public,settings=None,*,acceptance_policy='best_feasible_v2'):
    """Explicit-region SLSQP with bounded, independently checked candidate retention."""
    _check_acceptance_policy(acceptance_policy)
    settings=settings or GeometrySettings();result=deepcopy(fit);model=result['model'];count=len(model['bases'])
    if not count:return result
    if model['scored_points']!=0:raise ValueError('refinement must precede birth evidence')
    history=fit['window'];now=history[-1]['time_seconds'];origin=np.asarray(history[-1]['position_m'])
    if model['reference_time_seconds']!=now:raise ValueError('refinement requires original training reference')
    xy=[p['position_m'] for p in history];times=[p['time_seconds'] for p in history]
    labels=np.asarray([p['interference_dbm'] for p in history]);real_count=len(labels)
    weights=np.full(real_count,1/math.sqrt(real_count));soft=fit.get('gp_soft_constraints')
    if soft and soft['position_m']:
        if soft['prefix_end_seconds']!=now or soft['independent_evidence_points']!=0:
            raise ValueError('soft constraints must share the original fit prefix')
        xy+=soft['position_m'];times+=soft['time_seconds'];labels=np.r_[labels,soft['mean_dbm']]
        weights=np.r_[weights,np.sqrt(soft['strength']*np.asarray(soft['weights']))]
    xy=np.asarray(xy,float);times=np.asarray(times,float);initial=pack(model,'geometry')
    centers=np.asarray([b['center_m'] for b in model['bases']]);scales=np.asarray([b['scale_m'] for b in model['bases']])
    bounds=np.asarray(public['map']['bounds_m'])
    low=np.r_[np.full(count,fit['bounds_dbm'][0]),np.maximum(bounds[0],centers-settings.center_radius_m).ravel(),
              np.log(np.maximum(settings.minimum_scale_m,scales/settings.scale_ratio))]
    high=np.r_[np.full(count,fit['bounds_dbm'][1]),np.minimum(bounds[1],centers+settings.center_radius_m).ravel(),
               np.log(np.minimum(settings.maximum_scale_m,scales*settings.scale_ratio))]
    units=np.r_[np.full(count,4.),np.full(2*count,40.),np.full(count,math.log(math.sqrt(2.)))]
    work=dict(solver_calls=0,residual_calls=0,jacobian_calls=0,kernel_terms=0)
    trace=dict(mode='geometry',algorithm='SLSQP_explicit_region_v1',settings=asdict(settings),seed_model=deepcopy(model),
        accepted=False,acceptance_policy=acceptance_policy,solver_success=False,
        training_history_is_future_evidence=False,prefix_end_seconds=now,
        lower_bounds=low.tolist(),upper_bounds=high.tolist(),initial_parameters=initial.tolist(),
        parameter_units=units.tolist(),interior_margin_m=1e-6,ftol=1e-8,
        objective_trace=[],gradient_trace=[],constraint_value_calls=0,constraint_jacobian_calls=0,prediction_cache_hits=0)
    result['geometry_refinement']=trace;cached=None
    candidates=_EvaluatedCandidates(model,public,origin,fit['job']['region'],'geometry',low,high,trace)
    def evaluate(parameters):
        nonlocal cached
        if cached is None or not np.array_equal(parameters,cached[0]):
            terms=len(times)*count
            if work['kernel_terms']+terms>settings.max_kernel_terms:raise ValueError('refinement kernel budget exhausted')
            work['kernel_terms']+=terms
            mean,jac=prediction_jacobian(model,parameters,xy,times,'geometry')
            cached=(np.array(parameters),mean,jac)
        else:trace['prediction_cache_hits']+=1
        return cached[1:]
    def objective(z):
        if work['residual_calls']>=settings.max_evaluations:raise ValueError('refinement objective budget exhausted')
        work['residual_calls']+=1;p=initial+units*z
        prediction=evaluate(p)[0];loss=float(np.sum(((prediction-labels)*weights)**2))
        candidates.consider(p,prediction,loss)
        return loss
    def gradient(z):
        if work['jacobian_calls']>=settings.max_evaluations:raise ValueError('refinement gradient budget exhausted')
        work['jacobian_calls']+=1;mean,jac=evaluate(initial+units*z)
        grad=2*(jac.T@((mean-labels)*weights**2))*units
        norm=float(np.linalg.norm(grad))
        trace['gradient_trace'].append(norm if math.isfinite(norm) else None)
        if not np.all(np.isfinite(grad)):
            raise FloatingPointError('nonfinite refinement gradient')
        return grad
    def constraint(z):
        trace['constraint_value_calls']+=1
        return region_values_jacobian(initial+units*z,count,origin,public['goal_m'],fit['job']['region'])[0]
    def constraint_jac(z):
        trace['constraint_jacobian_calls']+=1
        return region_values_jacobian(initial+units*z,count,origin,public['goal_m'],fit['job']['region'])[1]*units
    def describe(prediction,loss):
        result['fit_objective']=loss
        result['gp_shape_mse']=(float(np.sum(np.asarray(soft['weights'])*(prediction[real_count:]-labels[real_count:])**2))
            if soft and soft['position_m'] else None)
    try:
        if (np.any(low>=high) or not np.all(np.isfinite(initial)) or np.any(initial<low) or np.any(initial>high)
                or not all(in_region(b['center_m'],public,origin,fit['job']['region']) for b in model['bases'])):
            raise ValueError('seed outside original refinement region/bounds')
        baseline=evaluate(initial)[0];old_loss=float(np.sum(((baseline-labels)*weights)**2))
        describe(baseline,old_loss);trace['previous_objective']=old_loss;work['solver_calls']=1
        candidates.consider(initial,baseline,old_loss,record=False)
        solved=minimize(objective,np.zeros(len(initial)),jac=gradient,method='SLSQP',
            bounds=list(zip((low-initial)/units,(high-initial)/units)),
            constraints=[dict(type='ineq',fun=constraint,jac=constraint_jac)],
            options=dict(ftol=1e-8,maxiter=settings.max_evaluations))
        final=initial+units*solved.x
        refined=unpack(model,final,'geometry');delta=np.asarray([b['center_m'] for b in refined['bases']])-origin
        direction=np.asarray(public['goal_m'])-origin
        cross=direction[0]*delta[:,1]-direction[1]*delta[:,0]
        bound_ok,eligible=candidates.eligibility(final)
        trace.update(solver_final_parameters=_parameter_list(final),
            solver_success=bool(solved.success),solver_status=int(solved.status),solver_message=str(solved.message),
            solver_final_region_eligible=eligible,solver_final_bounds_eligible=bound_ok,
            nfev=int(solved.nfev),njev=int(solved.njev))
        if acceptance_policy=='converged_v1':
            prediction=evaluate(final)[0];loss=float(np.sum(((prediction-labels)*weights)**2))
            accepted=bool(solved.success and np.all(np.isfinite(final)) and eligible and bound_ok
                          and loss<old_loss-settings.minimum_improvement)
            trace.update(final_parameters=final.tolist(),candidate_objective=loss,accepted=accepted,
                region_eligible=eligible,bounds_eligible=bound_ok,
                radial_slack_m=(160.-np.linalg.norm(delta,axis=1)).tolist(),cross_to_goal_m2=cross.tolist())
            if accepted:
                refined['fit_mse']=float(np.mean((prediction[:real_count]-labels[:real_count])**2))
                result['model']=refined;describe(prediction,loss)
    except (ValueError,FloatingPointError) as exc:trace['error']=str(exc)
    trace['stop_reason']=trace.get('error') or ('solver_converged' if trace['solver_success'] else 'solver_nonconverged')
    if acceptance_policy=='best_feasible_v2':
        candidates.apply(result,labels,real_count,soft,settings.minimum_improvement)
        selected=result['model'];delta=np.asarray([b['center_m'] for b in selected['bases']])-origin
        direction=np.asarray(public['goal_m'])-origin
        trace.update(radial_slack_m=(160.-np.linalg.norm(delta,axis=1)).tolist(),
            cross_to_goal_m2=(direction[0]*delta[:,1]-direction[1]*delta[:,0]).tolist())
    trace['resources']=work
    return result
