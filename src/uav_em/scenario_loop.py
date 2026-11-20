"""Point-sensor loop with power-rise events or the historical C1 schedule.

Advisor callbacks are synchronous prototypes, not a wall-clock flight model.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import math
import numpy as np
from .environment import ElectromagneticWorld
from .event_inference import HypothesisBank,digest
from .event_llm_inference import run_event
from .feasible_scenario import pool,best,feedback_plan,execute,search_waypoints
from .repair_search import WorkBudget
from .scenario_planning import ScenarioForecast,state_request,validate_search_plan,score_route
from .point_gp import PointGP
from .planning import path_length
from .motion import advance_route
from .decision_timing import PowerRiseTrigger

EVENTS=(0.,8.,20.,32.)
METHODS=('gp_numeric','persistence_numeric','mixture_numeric','mixture_llm')


def public_map(geometry,goal):
    return dict(goal_m=list(goal),map=dict(bounds_m=[[0.,0.],geometry.high.tolist()]))


def recovery_plan(request):
    """Same bounded direct/repair tool when no common route can be retained."""
    left=request['limits']['max_samples']
    return dict(request_id=request['request_id'],action='direct' if left else 'keep',
        waypoint_m=None,sample_quotas=[left] if left else [],sampling='uniform',
        reason='bounded direct repair with remaining shared budget' if left else 'shared sample budget exhausted')


def run(config,method,advisor=None,emit=None,*,timing='fixed',inference_advisor=None,
        joint_settings=None,forecast_method=None,soft_fit=False,revision_settings=None,path_search='feedback',
        source_renewal=False,geometry_refinement=None):
    if method not in METHODS or config.sensor.mode!='power_only':raise ValueError('point-only declared method required')
    if timing not in ('fixed','power_rise','forward','source_revision','task_source'):raise ValueError('unknown consultation timing')
    task_source=timing=='task_source';source_timing=timing in ('source_revision','task_source')
    if timing in ('forward','source_revision','task_source') and joint_settings is None:raise ValueError('forward timing requires frozen joint settings')
    if timing=='source_revision' and (revision_settings is None or method!='mixture_numeric' or inference_advisor is not None):
        raise ValueError('source revision pilot requires explicit settings and numerical organization')
    if task_source and (revision_settings is None or method not in ('mixture_numeric','mixture_llm')):
        raise ValueError('task source mode requires explicit revision settings and a source method')
    if not source_timing and revision_settings is not None:raise ValueError('revision settings require source revision timing')
    if type(source_renewal) is not bool or (source_renewal and not source_timing):
        raise ValueError('source renewal requires explicit source-revision mode')
    if geometry_refinement not in (None,'amplitude','geometry') or (geometry_refinement is not None and timing!='source_revision'):
        raise ValueError('geometry refinement requires explicit source-revision mode')
    if path_search not in ('feedback','lattice'):raise ValueError('unknown path organization')
    if path_search=='lattice' and (timing!='source_revision' or method!='mixture_numeric'):
        raise ValueError('lateral-search pilot requires numerical source-revision mode')
    if method=='mixture_llm' and timing in ('power_rise','forward','task_source') and advisor is None:
        raise ValueError('event LLM flight requires an explicit advisor')
    if inference_advisor is not None and not (timing=='power_rise' and method=='mixture_llm' or
                                             timing in ('forward','task_source') and method in ('mixture_llm','mixture_numeric')):
        raise ValueError('inference advisor requires the event LLM method')
    predict_method=forecast_method or ('joint' if task_source else 'gp' if method=='gp_numeric' else 'persistence' if method=='persistence_numeric' else 'mixture')
    if predict_method not in ('gp','persistence','mixture','joint'):raise ValueError('unknown planner forecast')
    if task_source and predict_method!='joint':raise ValueError('task source mode requires the shared joint forecast')
    world=ElectromagneticWorld(config);geometry=world.geometry;a=config.aircraft
    estimator=PointGP(geometry,config.belief,config.electromagnetic.background_dbm)
    position=list(config.map.start_m);goal=list(config.map.goal_m);now=spent=0.;energy=a.initial_energy_wh
    maximum=1.25*float(np.linalg.norm(np.asarray(goal)-position));route=[position,goal]
    observations=[];frames=[];motions=[];decisions=[];inferences=[];rows=[];planning_failures=[];bank=None;termination='time_limit'
    initial_work=WorkBudget(700,10000);initial_search=[]
    if not geometry.edge_free(position,goal):
        from .planning import PlanResult
        from .repair_search import repair_rrt_star
        class Geometric:
            budget=initial_work
            def __init__(self):self.geometry=geometry
            def edge_cost(self,x,y):
                self.budget.take_edge()
                return float(np.linalg.norm(np.asarray(y)-x)) if geometry.edge_free(x,y) else math.inf
        solved,stats=repair_rrt_star(Geometric(),position,goal,700,config.seed,dict(focus_probability=0.,regions=[]))
        initial_search=[dict(**asdict(solved),sampling_counts=stats)];route=solved.path
    from .joint_prediction import ForwardWarningTrigger
    from .source_revision import SourceOnlyBank,SourceRenewalBank,SourceRevisionTrigger
    event_number=0;trigger=(SourceRevisionTrigger(revision_settings) if source_timing else
        ForwardWarningTrigger() if timing=='forward' else PowerRiseTrigger() if timing=='power_rise' else None)
    trigger_records=[];joint_predictions=[];previous_forecast=None
    while now<=100.+1e-8:
        if bank is not None:bank.issue(position,now)
        issued_joint=None
        if previous_forecast is not None:
            d=previous_forecast.predict([position],[now],'joint')
            issued_joint=dict(position_m=deepcopy(position),time_seconds=now,
                forecast_at_seconds=previous_forecast.now,forecast_sha256=previous_forecast.fingerprint,
                history_sha256=digest(previous_forecast.history),
                mean_dbm=float(d['weights']@d['means'][:,0]),
                source_mean_dbm=float(d['weights']@d['source_mean_dbm'][:,0]),
                gp_mean_dbm=float(d['gp_mean_dbm'][0]) if d['supported'][0] else None,
                supported=bool(d['supported'][0]),gp_weight=float(d['gp_weight'][0]),
                predicted_before_observation=True)
        measurement=world.observe(position,now)
        point={k:asdict(measurement)[k] for k in ('position_m','time_seconds','interference_dbm')};point['position_m']=list(point['position_m'])
        if issued_joint is not None:
            joint_predictions.append(dict(issued=issued_joint,observation=deepcopy(point),
                residual_db=point['interference_dbm']-issued_joint['mean_dbm']))
        observations.append(point);estimator.observe(measurement)
        if bank is None:
            bank_type=SourceRenewalBank if source_renewal else SourceOnlyBank
            bank=(bank_type(point,config.electromagnetic.background_dbm,config.belief.unknown_error_db,
                           **({'mismatch_db':revision_settings.mismatch_db} if source_renewal else {}))
                  if source_timing else HypothesisBank(point))
        else:
            rows.append(bank.observe(point))
        snapshot,gp=estimator.freeze(now)
        def frame(phase):
            return dict(time_seconds=now,observation_time_seconds=point['time_seconds'],phase=phase,
                position_m=deepcopy(position),route_m=deepcopy(route),
                estimate_dbm=snapshot.estimate_dbm.tolist(),uncertainty_db=snapshot.uncertainty_db.tolist(),supported=snapshot.supported.tolist(),
                observed_positions_m=[p['position_m'] for p in observations],
                hypotheses=bank.snapshot(),latest_power_dbm=point['interference_dbm'],planner_method=predict_method)
        frames.append(frame('before_decision'))
        if not route:termination='initial_plan_failed';break
        if np.linalg.norm(np.asarray(position)-goal)<1e-7:termination='goal_reached';break
        if now>=100.-1e-8:break
        gp_version=digest(dict(history=observations,now=point['time_seconds'],settings=asdict(config.belief)))
        if timing in ('forward','source_revision','task_source'):
            monitor=ScenarioForecast(bank.history,bank.snapshot(),now,gp,gp_version=gp_version,joint_settings=joint_settings)
            trigger_record=(trigger.observe(monitor,route,a.speed_mps,len(bank.history),rows[-1] if rows else None)
                            if source_timing else trigger.observe(monitor,route,a.speed_mps,len(bank.history)))
            trigger_record['receiver_work']=deepcopy(monitor.counts)
        else:
            trigger_record=trigger.observe(now,point['interference_dbm']) if trigger else None
        if trigger_record is not None:trigger_records.append(trigger_record)
        due=(trigger_record['requested'] if trigger else event_number<len(EVENTS) and abs(now-EVENTS[event_number])<1e-7)
        if due:
            state=dict(position_m=deepcopy(position),goal_m=goal,route_m=deepcopy(route),speed_mps=a.speed_mps,
                spent_length_m=spent,maximum_length_m=maximum,remaining_energy_wh=energy,flight_power_w=a.flight_power_w,
                hover_power_w=a.hover_power_w,reserve_energy_wh=a.reserve_energy_wh,delay_seconds=.5)
            work=WorkBudget(700,10000);task_context=None
            if task_source:
                from .task_context import build_task_context
                pre_forecast=ScenarioForecast(bank.history,bank.snapshot(),now,gp,gp_version=gp_version,joint_settings=joint_settings)
                task_context=build_task_context(state,pre_forecast,geometry,work,predict_method)
            # Pilot fitting withholds three known readings; never fit before four exist.
            fit_ready=(len(bank.history)>=4 and method not in ('gp_numeric','persistence_numeric')) if trigger else event_number>0
            if fit_ready:
                fit_options=dict(trigger_reason=trigger_record['reasons'][0] if trigger else None,
                    soft_context=dict(geometry=geometry,belief=config.belief,background=config.electromagnetic.background_dbm,
                                      settings=joint_settings) if soft_fit or task_source else None)
                if task_source:
                    from .task_decision import run_task_event
                    from .task_context import validate_task_context
                    def checked_task_advisor(request):
                        try:
                            return inference_advisor(request)
                        finally:
                            validate_task_context(task_context,history=bank.history,models=bank.snapshot(),
                                state=state,forecast=pre_forecast,geometry=geometry)
                    event=run_task_event(bank,public_map(geometry,goal),len(inferences),rows,
                        'llm' if inference_advisor is not None else 'feedback',
                        advisor=checked_task_advisor if inference_advisor is not None else None,
                        task_context=task_context,**fit_options)
                else:
                    event=run_event(bank,public_map(geometry,goal),len(inferences),rows,
                        'llm' if inference_advisor is not None else 'feedback',advisor=inference_advisor,
                        geometry_refinement=geometry_refinement,**fit_options)
                inferences.append(event)
            forecast=ScenarioForecast(bank.history,bank.snapshot(),now,gp,gp_version=gp_version,joint_settings=joint_settings)
            if task_source:
                from .task_context import rescore_task_candidates
                candidates=rescore_task_candidates(task_context,state,forecast,geometry,work,predict_method)
            else:candidates=pool(state,forecast,geometry,work,predict_method)
            has_common=any(candidate['feasible'] for candidate in candidates)
            if not has_common and not task_source:
                planning_failures.append(dict(index=event_number,time_seconds=now,
                    reason='no_feasible_retained_route',candidates=deepcopy(candidates),
                    resources=work.summary(),receiver_work=deepcopy(forecast.counts),
                    forecast_sha256=forecast.fingerprint))
                termination='no_feasible_route';frames.append(frame('planning_failed'));break
            chosen=best(candidates) if has_common else None
            # Same paid common-pool state, then one extra numeric or LLM proposal.
            if chosen is not None:state['route_m']=deepcopy(chosen['path'])
            request=state_request(state,forecast,geometry,work)
            def view(row):
                result={k:deepcopy(row[k]) for k in ('id','path','feasible','reason')}
                if row['feasible']:
                    result['score']={k:deepcopy(row['score'][k]) for k in ('score_seconds','remaining_flight_seconds','power_proxy_seconds','scenario_q_std_seconds','scenario_scores')}
                return result
            local=[];xy=[];tt=[]
            for horizon in (0.,8.,16.):
                for dx,dy in ((0.,0.),(60.,0.),(-60.,0.),(0.,60.),(0.,-60.)):
                    q=np.asarray(position)+[dx,dy]
                    if geometry.free(q):xy.append(q.tolist());tt.append(now+horizon)
            mus,sds,supported=gp.predict(np.asarray(xy),np.asarray(tt))
            for q,t,mu,sd,ok in zip(xy,tt,mus,sds,supported):
                local.append(dict(position_m=q,time_seconds=t,mean_dbm=float(mu) if ok else None,std_db=float(sd),supported=bool(ok)))
            request_info=dict(request=request,common_pool=[view(c) for c in candidates],common_best=chosen['id'] if chosen else None,
                gp_local_samples=local,additional_gp_receiver_queries=len(xy),
                planner_method=predict_method,prior_mission_feedback=[dict(time_seconds=d['time_seconds'],selected=d['selected_id'],reason=d['extra'].get('failure')) for d in decisions])
            request_info['path_search_policy']=path_search
            if task_source:
                request_info.update(task_context=deepcopy(task_context),
                    task_context_id=task_context['task_snapshot_id'],
                    task_context_role='pre_inference_diagnostic',common_feasible_route_available=has_common)
            if timing in ('forward','source_revision','task_source'):
                request_info['joint_settings']=asdict(joint_settings)
                request_info['forecast_identity']=forecast.fingerprint
            if trigger_record is not None:
                request_info['consultation_event']=deepcopy(trigger_record)
                request_info['inference_status']=(('kept_current_pool' if task_source and not event['committed_models'] else 'updated') if fit_ready else
                    'reference_does_not_use_equivalent_fit' if method in ('gp_numeric','persistence_numeric') else
                    'insufficient_history_kept_existing_pool')
                request_info['inference_feedback']=[]
                if fit_ready:
                    for rnd in event['rounds']:
                        request_info['inference_feedback'].append(dict(phase=rnd['request']['phase'],
                            parser_accepted=rnd['parser_accepted'],pilot_feedback=rnd['pilot_feedback'],
                            fits=[dict(job=f['job'],error=f.get('error'),model=f.get('model'),
                                training_error_is_predictive_evidence=False) for f in rnd['fits']],
                            resources_after=rnd['resources_after']))
                    if geometry_refinement is not None:
                        request_info['geometry_refinement']=dict(mode=geometry_refinement,
                            reports=[f.get('geometry_refinement') for rnd in event['rounds'] for f in rnd['fits']])
            raw=None;envelope=None;fallback=None
            if method=='mixture_llm' and advisor is not None:
                try:
                    envelope=advisor(deepcopy(request_info),event_number)
                except TimeoutError as exc:
                    envelope=dict(raw=None,timely=False,started=None,
                        invocation_accounting='unknown_after_advisor_exception',advisor_error=str(exc))
                if not isinstance(envelope,dict):
                    envelope=dict(raw=None,timely=None,started=None,original_envelope=deepcopy(envelope),
                        invocation_accounting='invalid_envelope_invocation_unknown')
                if type(envelope.get('started')) is not bool:
                    envelope.setdefault('invocation_accounting','not_reported_by_advisor')
                raw=envelope.get('raw')
                try:
                    if envelope.get('timely') is not True:raise ValueError('not timely')
                    proposal=validate_search_plan(request,raw)
                except (ValueError,TypeError) as exc:
                    fallback=str(exc);proposal=feedback_plan(request,chosen) if chosen else recovery_plan(request)
            else:proposal=feedback_plan(request,chosen) if chosen else recovery_plan(request)
            extra=execute(request,proposal,state,forecast,geometry,work,predict_method,config.seed+1000*(event_number+1))
            candidates+=extra['candidates']
            grid_search=search_waypoints(state,forecast,geometry,work,predict_method) if path_search=='lattice' else None
            if grid_search:candidates+=grid_search['candidates']
            if not any(candidate['feasible'] for candidate in candidates):
                failure=dict(index=event_number,time_seconds=now,reason='no_feasible_retained_route',
                    candidates=deepcopy(candidates),resources=work.summary(),receiver_work=deepcopy(forecast.counts),
                    forecast_sha256=forecast.fingerprint,envelope=deepcopy(envelope),raw_response=raw,
                    fallback=fallback,extra=extra,public=request_info,task_context_id=task_context['task_snapshot_id'],
                    charged_hover_seconds=.5)
                energy-=.5*a.hover_power_w/3600;now+=.5
                planning_failures.append(failure);termination='no_feasible_route'
                frames.append(frame('planning_failed'));break
            new=best(candidates)
            route=deepcopy(new['path'])
            entry=dict(index=event_number,time_seconds=now,public=request_info,envelope=envelope,raw_response=raw,fallback=fallback,
                extra=extra,grid_search=grid_search,candidates=candidates,selected_id=new['id'],route_after=deepcopy(route),receiver_work=forecast.counts,
                common_selected_id=chosen['id'] if chosen else None,novel_waypoint=bool(proposal['action']=='via' and all(len(x['path'])!=3 or not np.allclose(proposal['waypoint_m'],x['path'][1]) for x in candidates if not x['id'].startswith('proposal'))),
                resources=work.summary())
            decisions.append(entry)
            if emit:emit(entry)
            event_number+=1;energy-=.5*a.hover_power_w/3600;now+=.5
        frames.append(frame('after_decision' if due else 'continuing'))
        if joint_settings is not None:
            previous_forecast=ScenarioForecast(bank.history,bank.snapshot(),point['time_seconds'],gp,
                                              gp_version=gp_version,joint_settings=joint_settings)
        step=min(1.,100.-now,path_length(route)/a.speed_mps)
        if timing=='fixed' and event_number<len(EVENTS) and EVENTS[event_number]>now:step=min(step,EVENTS[event_number]-now)
        if step<=1e-9:break
        if energy-step*a.flight_power_w/3600<a.reserve_energy_wh:termination='energy_reserve';break
        remaining,traversed=advance_route(route,step*a.speed_mps);distance=path_length(traversed)
        if spent+distance>maximum+1e-7 or not all(geometry.edge_free(x,y) for x,y in zip(traversed,traversed[1:])):raise RuntimeError('invalid execution')
        motions.append(dict(start_time_seconds=now,end_time_seconds=now+step,traversed_m=traversed))
        now+=step;spent+=distance;energy-=step*a.flight_power_w/3600;route=remaining;position=list(route[0])
    inference_envelopes=[r['envelope'] for e in inferences for r in e['rounds']]
    path_envelopes=[d['envelope'] for d in decisions+planning_failures if d.get('envelope') is not None]
    inference_calls=sum(e.get('actual_llm_request') is True for e in inference_envelopes)
    path_calls=sum(e.get('started') is True for e in path_envelopes)
    unknown_inference=sum(type(e.get('actual_llm_request')) is not bool for e in inference_envelopes)
    unknown_path=sum(type(e.get('started')) is not bool for e in path_envelopes)
    return dict(method=method,timing=timing,trigger_records=trigger_records,joint_prediction_records=joint_predictions,
        planner_method=predict_method,joint_settings=asdict(joint_settings) if joint_settings is not None else None,gp_soft_fit=soft_fit or task_source,
        source_revision_settings=asdict(revision_settings) if revision_settings is not None else None,
        source_renewal=source_renewal,source_replacement_events=deepcopy(getattr(bank,'replacement_events',[])),
        geometry_refinement=geometry_refinement,
        path_search_policy=path_search,task_interface_version='task_source_v1' if task_source else None,
        timing_limitations=dict(synchronous_advisor=True,physical_state_frozen_during_callback=True,
                               model_latency_charged=False,post_decision_hover_seconds=.5),
        inference_organization='llm' if inference_advisor is not None else 'numeric_feedback',
        observations=observations,prediction_records=rows,final_bank=bank.snapshot(),frames=frames,motions=motions,
        decisions=decisions,inferences=inferences,planning_failures=planning_failures,
        initial_resources=initial_work.summary(),initial_searches=initial_search,
        metrics=dict(success=termination=='goal_reached',termination=termination,simulation_seconds=now,executed_length_m=spent,
            energy_used_wh=a.initial_energy_wh-energy,decision_count=len(decisions),fallbacks=sum(d.get('fallback') is not None for d in decisions+planning_failures),
            actual_model_calls=path_calls+inference_calls,path_model_calls=path_calls,inference_model_calls=inference_calls,
            unknown_model_call_receipts=unknown_inference+unknown_path,
            unknown_inference_call_receipts=unknown_inference,unknown_path_call_receipts=unknown_path,
            model_call_accounting_complete=not (unknown_inference+unknown_path),
            accepted_model_responses=sum(bool(d['envelope'] and d['envelope'].get('timely') and not d['fallback']) for d in decisions),
            total_samples=initial_work.samples+sum(d['resources']['samples'] for d in decisions+planning_failures),
            total_edges=initial_work.edges+sum(d['resources']['edges'] for d in decisions+planning_failures),
            inferred_jobs=sum(len(r['accepted_plan']['jobs']) for e in inferences for r in e['rounds']),
            solver_calls=sum(e['resources']['solver_calls'] for e in inferences),residual_calls=sum(e['resources']['residual_calls'] for e in inferences),
            grid_proposals=sum(d['grid_search']['proposals'] for d in decisions if d['grid_search']),
            common_grid_proposals=sum(d['public']['task_context']['resources_after']['samples']-
                d['public']['task_context']['resources_before']['samples']
                for d in decisions+planning_failures if d.get('public',{}).get('task_context')),
            task_snapshot_receiver_queries=sum(d['public']['task_context']['receiver_work']['receiver_queries']
                for d in decisions+planning_failures if d.get('public',{}).get('task_context')),
            grid_selected_events=sum(d['selected_id'].startswith('lattice_') for d in decisions)))
