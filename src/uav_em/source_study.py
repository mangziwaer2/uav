"""Bounded source-mechanism studies; sole entry: run_experiment.py."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .config import load_config
from .environment import ElectromagneticWorld
from .event_demo import read, save, sha
from .event_inference import digest
from .feasible_scenario import best, evaluate, search_waypoints
from .equivalent_field import normalized_expected_power
from .joint_experiment import (audit_before_truth, check, display_predictions,
                               numerical_samples)
from .joint_prediction import JointSettings
from .point_gp import FrozenGP, training_points
from .scenario_loop import run as fly
from .scenario_planning import ScenarioForecast, score_route
from .source_revision import RevisionSettings
from .repair_search import WorkBudget

ROOT = Path(__file__).resolve().parents[2]
D1 = ROOT / 'outputs/interference_joint_gp_d1_20261119_0940'
METHODS = ('source_early', 'source_revise')
SEARCH_METHODS = ('search_feedback', 'search_lattice')
RENEWAL_METHODS = ('renewal_keep', 'renewal_replace')
GEOMETRY_METHODS = ('geometry_amplitude','geometry_bounded')
M0 = ROOT / 'outputs/interference_source_only_d2_m0_20261120'
M1 = ROOT / 'outputs/interference_source_search_d2_m1_20261120'
M2 = ROOT / 'outputs/interference_source_renewal_d2_m2_20261120'


def original_check(protocol):
    for rel, expected in protocol['original_d1_sha256'].items():
        if sha(D1 / rel) != expected:
            raise ValueError('D1 original changed: '+rel)
    for rel, expected in protocol['original_c1_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('C1 original changed: '+rel)
    for rel, expected in protocol.get('original_m0_sha256', {}).items():
        if sha(ROOT / rel) != expected:
            raise ValueError('D2-M0 original changed: '+rel)
    for rel, expected in protocol.get('original_m1_sha256', {}).items():
        if sha(ROOT / rel) != expected:
            raise ValueError('D2-M1 original changed: '+rel)
    for rel, expected in protocol.get('original_m2_sha256', {}).items():
        if sha(ROOT / rel) != expected:
            raise ValueError('D2-M2 original changed: '+rel)


def prepare(out, *, search=False, renewal=False, geometry=False):
    if sum((search,renewal,geometry))>1:raise ValueError('one study treatment required')
    out.mkdir(exist_ok=False)
    for name in ('runs', 'public', 'reports/frozen_source', 'predictions', 'evaluation', 'replay'):
        (out / name).mkdir(parents=True)
    shutil.copy2(ROOT / 'configs/local_power.json', out / 'scenario.json')
    sources = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'tests').rglob('*.py'))
    contract=('source_geometry_pilot_v1.md' if geometry else 'source_renewal_pilot_v1.md' if renewal else
              'source_search_pilot_v1.md' if search else 'source_only_pilot_v1.md')
    sources += [ROOT / 'requirements.txt', ROOT / 'docs' / contract]
    hashes = {}
    for source in sources:
        rel = source.relative_to(ROOT)
        target = out / 'reports/frozen_source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(rel)] = sha(source)
    c1 = ROOT / 'outputs/interference_scenario_loop_20261118'
    methods=GEOMETRY_METHODS if geometry else RENEWAL_METHODS if renewal else SEARCH_METHODS if search else METHODS
    protocol = dict(version='D2-M3' if geometry else 'D2-M2' if renewal else 'D2-M1' if search else 'D2-M0', methods=list(methods), source_sha256=hashes,
        scenario_sha256=sha(out / 'scenario.json'), public_contract_sha256={},
        joint_settings=asdict(JointSettings()),
        revision_settings={m: asdict(RevisionSettings(enable_mismatch=geometry or renewal or search or m == 'source_revise')) for m in methods},
        path_search={m: 'lattice' if geometry or renewal or m=='search_lattice' else 'feedback' for m in methods},
        source_renewal={m:m=='renewal_replace' for m in methods},
        geometry_refinement={m:'geometry' if m=='geometry_bounded' else 'amplitude' if m=='geometry_amplitude' else None for m in methods},
        max_flights=2, max_events_per_flight=4, max_actual_model_calls=0, max_bridge_requests=0,
        max_sensor_points=204, max_jobs=24, max_solver_calls=72 if geometry else 48, max_residual_calls=2880 if geometry else 1920,
        max_refinement_calls=24 if geometry else 0,max_refinement_kernel_terms=120000 if geometry else 0,
        max_samples=7000, max_edges=100000, max_monitor_receiver_queries=4000,
        max_soft_gp_queries=360, max_prequential_queries=202, max_planner_receiver_queries=40000000,
        max_evaluation_points=2000 if search or renewal or geometry else 1000, max_flight_evaluation_points=1000,
        max_counterfactual_evaluation_points=1000 if search or renewal or geometry else 0, max_replay_truth_points=21504,
        max_display_receiver_queries=21504, max_shadow_receiver_queries=12000 if search or renewal or geometry else 1200,
        max_shadow_proposals=150 if search or renewal or geometry else 0,max_shadow_edges=30000 if renewal or geometry else 0,
        max_shadow_refinement_calls=9 if geometry else 0,max_shadow_refinement_residual_calls=360 if geometry else 0,
        max_shadow_refinement_kernel_terms=45000 if geometry else 0,
        max_frames_per_flight=32, display_shape=[16, 21],
        original_d1_sha256=read(D1 / 'reports/completion_manifest_20261120.json'),
        original_c1_sha256={str(p.relative_to(ROOT)): sha(p) for p in c1.rglob('*') if p.is_file()},
        original_m0_sha256={str(p.relative_to(ROOT)): sha(p) for p in M0.rglob('*') if p.is_file()} if search or renewal or geometry else {},
        original_m1_sha256={str(p.relative_to(ROOT)): sha(p) for p in M1.rglob('*') if p.is_file()} if renewal or geometry else {},
        original_m2_sha256={str(p.relative_to(ROOT)): sha(p) for p in M2.rglob('*') if p.is_file()} if geometry else {},
        interpretation='two numerical mechanism flights; no new decision LLM; classical search gain is not an LLM contribution',
        timing='source initialization + supported warning; optional causal mismatch revision; fixed 0.5s hover')
    original_check(protocol)
    save(out / 'protocol.json', protocol)
    if geometry:
        literature=ROOT/'docs/references/source_geometry_20261120/reading.json'
        if not literature.is_file():raise ValueError('geometry literature reading record required before freeze')
        save(out/'reports/literature_reading.json',read(literature))
        print('PREPARED',out,flush=True)
        return
    if renewal:
        literature=ROOT/'docs/references/source_renewal_20261120/reading.json'
        if not literature.is_file():raise ValueError('renewal literature reading record required before freeze')
        save(out/'reports/literature_reading.json',read(literature))
        print('PREPARED',out,flush=True)
        return
    if search:
        sources=[]
        for name,scope in (('IGV-RRT','II-C selected passages, local text lines 118-140'),
                           ('VLM-RRT','Method passages on sector sampling, local text lines 114-169')):
            cache=ROOT/'docs/references'/name
            sources.append(dict(name=name,original_sha256=sha(cache/'paper.html'),
                text_sha256=sha(cache/'paper.txt'),scope=scope,
                limitation='vision/semantic inputs differ; no sensing transfer or novelty claim for fixed waypoint enumeration'))
        save(out/'reports/literature_reading.json',dict(type='selected cached originals, not new web retrieval or full review',sources=sources))
        print('PREPARED',out,flush=True)
        return
    cache = ROOT / 'docs/references/hypothesis_refinement_20261119'
    save(out / 'reports/literature_reading.json', dict(
        type='selected existing full-text passages; not a new web retrieval or complete review',
        sources=[dict(title='Bayesian Online Changepoint Detection', url='https://arxiv.org/abs/0710.3742',
            original=str((cache / 'changepoint_html.raw').relative_to(ROOT)),
            original_sha256=sha(cache / 'changepoint_html.raw'),
            text_sha256=sha(cache / 'changepoint_html.txt'),
            scope='Abstract, introduction, sections 2 and 2.1; selected text only',
            use='causal predictive monitoring; no BOCPD implementation or claimed calibrated posterior',
            limitation='piecewise iid/hazard assumptions do not directly model a moving radio receiver'),
        dict(title='Literature Meets Data', url='https://arxiv.org/abs/2410.17309',
            original_sha256=sha(cache / 'hypothesis_literature_data.html'),
            text_sha256=sha(cache / 'hypothesis_literature_data.txt'),
            scope='sections 2.2 and 2.3 selected passages on wrongly predicted samples and refinement',
            use='feed prediction failures into bounded source revision',
            limitation='classification task; its rewards and measured gains do not establish UAV performance')]))
    print('PREPARED', out, flush=True)


def shadow(out, protocol):
    """Matched D1 prefixes: same parameters, conditional source weights, no truth."""
    episode = read(D1 / 'runs/joint_N/episode.json')
    cfg = load_config(out / 'scenario.json')
    world = ElectromagneticWorld(cfg)
    records, receiver_queries = [], 0
    xy = np.array([[x, y] for y in np.linspace(0, 600, 7) for x in np.linspace(0, 800, 9)])
    for t in (23.5, 27.5):
        frame = next(f for f in episode['frames'] if f['time_seconds'] == t and f['phase'] in ('after_decision', 'before_decision'))
        observed = frame['observation_time_seconds']
        prefix = [p for p in episode['observations'] if p['time_seconds'] <= observed]
        source = deepcopy([m for m in frame['hypotheses'] if m['bases']])
        mass = sum(m['weight'] for m in source)
        assert mass > 0
        for m in source:
            m['weight'] /= mass
            m['log_weight'] = math.log(m['weight']) if m['weight'] else -1e300
        gp = FrozenGP(training_points(prefix, world.geometry, observed, cfg.belief), world.geometry,
                      observed, cfg.belief, cfg.electromagnetic.background_dbm)
        record = dict(frame_time_seconds=t, observation_time_seconds=observed,
                      source_mass_before_conditioning=mass, variants={})
        for name, models in (('D1_saved_mixed_bank', frame['hypotheses']), ('source_conditional_bank', source)):
            f = ScenarioForecast(prefix, models, observed, gp, gp_version=digest(prefix), joint_settings=JointSettings())
            prediction = f.predict(xy, np.full(len(xy), t), 'joint')
            powers = prediction['weights'] @ prediction['expected_normalized_power']
            score = score_route(f, frame['route_m'], cfg.aircraft.speed_mps, method='joint')
            record['variants'][name] = dict(forecast_sha256=f.fingerprint,
                expected_power_dbm=(-80+10*np.log10(powers)).tolist(),
                supported=prediction['supported'].tolist(), route_score=score,
                relative_source_weights={m['id']: m['weight'] for m in models if m['bases']})
            receiver_queries += f.counts['receiver_queries']
        records.append(record)
    assert receiver_queries <= protocol['max_shadow_receiver_queries']
    save(out / 'reports/same_prefix_shadow.json', dict(records=records, receiver_queries=receiver_queries,
        position_m=xy.tolist(), new_fits=0, truth_queries=0,
        limitation='conditional representation and predicted score only; not evaluated route benefit'))


def run_one(out, method):
    protocol = check(out)
    original_check(protocol)
    if method not in protocol['methods']:
        raise ValueError('undeclared method')
    folder = out / 'runs' / method
    folder.mkdir(exist_ok=False)
    def emit(row):
        with (folder / 'decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        print('DECISION', method, row['index'], row['time_seconds'],
              row['public']['consultation_event']['reasons'], row['selected_id'], flush=True)
    started = time.monotonic()
    try:
        ep = fly(load_config(out / 'scenario.json'), 'mixture_numeric', timing='source_revision',
            forecast_method='joint', joint_settings=JointSettings(**protocol['joint_settings']),
            soft_fit=True, revision_settings=RevisionSettings(**protocol['revision_settings'][method]), emit=emit,
            path_search=protocol.get('path_search',{}).get(method,'feedback'),
            source_renewal=protocol.get('source_renewal',{}).get(method,False),
            geometry_refinement=protocol.get('geometry_refinement',{}).get(method))
        ep['study_method'], ep['wall_seconds'] = method, time.monotonic()-started
        save(folder / 'episode.json', ep)
        save(folder / 'status.json', dict(status='completed', episode_sha256=sha(folder / 'episode.json')))
        print('COMPLETE', method, ep['metrics'], flush=True)
    except BaseException as exc:
        save(folder / 'failure.json', dict(error=repr(exc)))
        raise


def audit_sources(episodes, cfg):
    from .event_inference import HypothesisBank
    from .source_revision import SourceOnlyBank, SourceRenewalBank
    checks = []
    for method, ep in episodes.items():
        bank_type=SourceRenewalBank if ep.get('source_renewal') else SourceOnlyBank
        bank = bank_type(ep['observations'][0], cfg.electromagnetic.background_dbm,
                         cfg.belief.unknown_error_db)
        births = {e['time_seconds']: e for e in ep['inferences']}
        rows = iter(ep['prediction_records'])
        for point in ep['observations']:
            if point is not ep['observations'][0]:
                issued = bank.issue(point['position_m'], point['time_seconds'])
                old = next(rows)
                assert issued == old['issued']
                result = bank.observe(point)
                assert result == old
            if point['time_seconds'] in births:
                event = births[point['time_seconds']]
                assert bank.snapshot() == event['bank_before_birth']
                bank.birth(event['committed_models'])
                assert bank.snapshot() == event['bank_after_birth']
        assert bank.snapshot() == ep['final_bank']
        assert getattr(bank,'replacement_events',[])==ep.get('source_replacement_events',[])
        for frame in ep['frames']:
            assert not any(m['family'] == 'persistence' for m in frame['hypotheses'])
            assert all(m['bases'] for m in frame['hypotheses'] if m['family'] != 'unknown_prior')
            assert not (any(m['bases'] for m in frame['hypotheses']) and
                        any(m['family'] == 'unknown_prior' for m in frame['hypotheses']))
        for event, decision in zip(ep['inferences'], ep['decisions']):
            assert event['time_seconds'] == decision['time_seconds']
            assert event['bank_after_birth'] == decision['public']['request']['models']
            grid=decision.get('grid_search')
            if grid:
                assert 0<=grid['proposals']<=50
                assert grid['resources_after']['samples']-grid['resources_before']['samples']==grid['proposals']
                assert all(c['score']['forecast_sha256']==decision['public']['request']['forecast_sha256']
                           for c in grid['candidates'] if c['feasible'])
        checks.append(dict(method=method, observations=len(ep['observations']),
            reissued_predictions_match=True, no_spatial_persistence=True,
            every_fit_has_same_event_route_reassessment=True,
            replacement_events=len(getattr(bank,'replacement_events',[])),
            source_weight_meaning='conditional relative evidence, not model adequacy'))
    return checks


def shadow_search(out,protocol,cfg,geometry):
    """Saved M0 prefixes only. Freeze all search outcomes before truth access."""
    episode=read(M0/'runs/source_revise/episode.json')
    records=[];queries=proposals=0
    for now in (20.5,31.,41.5):
        decision=next(d for d in episode['decisions'] if d['time_seconds']==now)
        request=decision['public']['request'];history=request['history']
        assert history[-1]['time_seconds']==now and all(p['time_seconds']<=now for p in history)
        gp=FrozenGP(training_points(history,geometry,now,cfg.belief),geometry,now,cfg.belief,cfg.electromagnetic.background_dbm)
        version=digest(dict(history=history,now=now,settings=asdict(cfg.belief)))
        forecast=ScenarioForecast(history,request['models'],now,gp,gp_version=version,joint_settings=JointSettings())
        assert forecast.fingerprint==request['forecast_sha256']
        state=deepcopy(request['state'])
        work=WorkBudget(**{k:decision['resources'][k] for k in WorkBudget.__dataclass_fields__})
        extra=search_waypoints(state,forecast,geometry,work,'joint')
        old=next(c for c in decision['candidates'] if c['id']==decision['selected_id'])
        chosen=best(decision['candidates']+extra['candidates'])
        records.append(dict(time_seconds=now,public_prefix_sha256=digest(history),
            forecast_sha256=forecast.fingerprint,source_models=deepcopy(request['models']),state=state,
            original_selected=deepcopy(old),search_selected=chosen,search=extra,
            predicted_gain=old['score']['score_seconds']-chosen['score']['score_seconds']))
        queries+=forecast.counts['receiver_queries'];proposals+=extra['proposals']
    assert queries<=protocol['max_shadow_receiver_queries'] and proposals<=150
    save(out/'reports/same_state_search.json',dict(records=records,receiver_queries=queries,proposals=proposals,
        new_fits=0,sensor_points=0,truth_queries_before_freeze=0,
        scope='same saved public prefix, models, weights, field and constraints; new deterministic waypoint search'))
    return records


def shadow_renewal(out,protocol,cfg,geometry):
    """Change only one saved birth's eligibility, with fixed fitted versions."""
    from .source_revision import SourceRenewalBank
    old=read(M0/'runs/source_revise/episode.json')
    references=read(M1/'reports/same_state_search.json')['records']
    records=[];queries=proposals=edges=0
    for now in (20.5,31.,41.5):
        decision=next(d for d in old['decisions'] if d['time_seconds']==now)
        event=next(e for e in old['inferences'] if e['time_seconds']==now)
        request=decision['public']['request'];history=request['history']
        latest=next(r for r in old['prediction_records'] if r['observation']['time_seconds']==now)
        # Reconstruct the saved pre-birth state; do not replay changed past births.
        bank=SourceRenewalBank(history[0]);bank.history=deepcopy(history)
        before=event['bank_before_birth']
        bank.models=[{k:deepcopy(v) for k,v in m.items() if k not in ('weight','log_weight')} for m in before]
        bank.log_weights=np.asarray([m['log_weight'] for m in before]);bank.latest_record=deepcopy(latest)
        bank.birth(event['committed_models'])
        assert len(bank.replacement_events)==1
        gp=FrozenGP(training_points(history,geometry,now,cfg.belief),geometry,now,cfg.belief,cfg.electromagnetic.background_dbm)
        version=digest(dict(history=history,now=now,settings=asdict(cfg.belief)))
        forecast=ScenarioForecast(history,bank.snapshot(),now,gp,gp_version=version,joint_settings=JointSettings())
        state=deepcopy(request['state']);work=WorkBudget(700,10000)
        candidates=[evaluate(c['path'],state,forecast,geometry,work,'joint',c['id']) for c in decision['candidates']]
        extra=search_waypoints(state,forecast,geometry,work,'joint')
        reference=next(r for r in references if r['time_seconds']==now)
        assert [c['path'] for c in extra['candidates']]==[c['path'] for c in reference['search']['candidates']]
        chosen=best(candidates+extra['candidates'])
        incumbent=next(c for c in candidates+extra['candidates'] if c['id']==reference['search_selected']['id'])
        records.append(dict(time_seconds=now,public_prefix_sha256=digest(history),
            reference_forecast_sha256=reference['forecast_sha256'],forecast_sha256=forecast.fingerprint,
            source_models=bank.snapshot(),state=state,source_replacement=bank.replacement_events[0],
            reference_original_prediction=reference['search_selected'],
            original_selected=incumbent,search_selected=chosen,
            reference_truth_cache='D2-M1 same_state_routes search_selected',
            candidates=candidates,search=extra,
            predicted_gain=incumbent['score']['score_seconds']-chosen['score']['score_seconds']))
        queries+=forecast.counts['receiver_queries'];proposals+=extra['proposals'];edges+=work.edges
    assert queries<=protocol['max_shadow_receiver_queries'] and proposals<=protocol['max_shadow_proposals']
    assert edges<=protocol['max_shadow_edges']
    save(out/'reports/same_state_search.json',dict(records=records,receiver_queries=queries,
        proposals=proposals,edges=edges,new_fits=0,sensor_points=0,truth_queries_before_freeze=0,
        scope='one saved birth replacement; fixed prefix, GP, fitted sources, geometry and candidate set; old route truth reused'))
    return records


def audit_refinements(episodes,protocol):
    totals=dict(calls=0,solver_calls=0,residual_calls=0,jacobian_calls=0,kernel_terms=0,accepted=0)
    for ep in episodes.values():
        for event in ep['inferences']:
            local={key:0 for key in totals if key!='accepted'}
            for rnd in event['rounds']:
                for fit in rnd['fits']:
                    refinement=fit.get('geometry_refinement')
                    if refinement is None:continue
                    work=refinement['resources'];local['calls']+=1
                    for key,value in work.items():local[key]+=value
                    assert work['solver_calls']<=1 and work['residual_calls']<=40
                    assert work['jacobian_calls']<=40 and work['kernel_terms']<=5000
                    assert refinement['prefix_end_seconds']==fit['window'][-1]['time_seconds']
                    if rnd['request']['phase']=='pilot':
                        assert refinement['prefix_end_seconds']==rnd['request']['pilot_training_cutoff_seconds']
                    if refinement['accepted']:
                        assert refinement['solver_success'] and refinement['region_eligible']
                        assert refinement['candidate_objective']<refinement['previous_objective']-1e-8
                        parameters=np.asarray(refinement['final_parameters'])
                        assert np.all(parameters>=refinement['lower_bounds']) and np.all(parameters<=refinement['upper_bounds'])
                        totals['accepted']+=1
            assert local==event['resources']['refinement']
            assert local['calls']<=3 and event['resources']['solver_calls']<=9 and event['resources']['residual_calls']<=360
            for key,value in local.items():totals[key]+=value
    assert totals['calls']<=protocol['max_refinement_calls']
    assert totals['kernel_terms']<=protocol['max_refinement_kernel_terms']
    return totals


def action_diagnosis(episodes):
    """Read previously scored candidates only; no forecast/sensor queries."""
    from .motion import advance_route
    records=[]
    for method,ep in episodes.items():
        for decision in ep['decisions']:
            candidates=[c for c in decision['candidates'] if c['feasible']]
            worlds=candidates[0]['score']['scenario_scores'];actions=[]
            def endpoint(path):
                remaining,_=advance_route(deepcopy(path),min(15.,sum(math.dist(a,b) for a,b in zip(path,path[1:]))))
                return remaining[0]
            for world in worlds:
                selected=min(candidates,key=lambda c:(next(s['score_seconds'] for s in c['score']['scenario_scores'] if s['id']==world['id']),
                    sum(math.dist(a,b) for a,b in zip(c['path'],c['path'][1:])),c['id']))
                actions.append(dict(id=world['id'],weight=world['weight'],selected=selected['id'],
                    next_15m_position=endpoint(selected['path'])))
            actual=next(c for c in candidates if c['id']==decision['selected_id'])
            actual_endpoint=endpoint(actual['path'])
            records.append(dict(method=method,time_seconds=decision['time_seconds'],actions=actions,
                mixture_selected=actual['id'],mixture_next_15m_position=actual_endpoint,
                distinct_world_actions=len({tuple(round(v,7) for v in a['next_15m_position']) for a in actions}),
                conditional_mass_disagreeing_with_mixture=sum(a['weight'] for a in actions if not np.allclose(a['next_15m_position'],actual_endpoint,rtol=0,atol=1e-7))))
    return dict(records=records,additional_field_queries=0,
        scope='per-world argmin on same saved feasible candidates; mutually exclusive alternatives, not simultaneous sources; diagnostics do not change decisions')


def shadow_geometry(out,protocol,cfg,geometry):
    from .source_geometry import refine_fit
    from .event_llm_inference import rebase_model
    old=read(M0/'runs/source_revise/episode.json')
    references=read(M1/'reports/same_state_search.json')['records']
    records=[];queries=proposals=edges=0
    fit_work=dict(calls=0,solver_calls=0,residual_calls=0,jacobian_calls=0,kernel_terms=0)
    for now in (20.5,31.,41.5):
        decision=next(d for d in old['decisions'] if d['time_seconds']==now)
        event=next(e for e in old['inferences'] if e['time_seconds']==now)
        request=decision['public']['request'];history=request['history']
        public=dict(goal_m=cfg.map.goal_m,map=dict(bounds_m=[[0.,0.],geometry.high.tolist()]))
        refined_models={};fits=[]
        for rnd in event['rounds']:
            for fit in rnd['fits']:
                if 'model' not in fit or not fit['model']['bases']:continue
                updated=refine_fit(fit,public,'geometry');report=updated['geometry_refinement']
                fit_work['calls']+=1
                for key,value in report['resources'].items():fit_work[key]+=value
                model=rebase_model(updated['model'],now,len(history)-1);refined_models[model['id']]=model
                fits.append(dict(model=model,geometry_refinement=report,window=fit['window'],
                    gp_soft_constraints=fit['gp_soft_constraints'],job=fit['job']))
        models=[dict(refined_models[m['id']],weight=m['weight'],log_weight=m['log_weight'])
                if m['id'] in refined_models else deepcopy(m) for m in request['models']]
        gp=FrozenGP(training_points(history,geometry,now,cfg.belief),geometry,now,cfg.belief,cfg.electromagnetic.background_dbm)
        version=digest(dict(history=history,now=now,settings=asdict(cfg.belief)))
        forecast=ScenarioForecast(history,models,now,gp,gp_version=version,joint_settings=JointSettings())
        state=deepcopy(request['state']);work=WorkBudget(700,10000)
        candidates=[evaluate(c['path'],state,forecast,geometry,work,'joint',c['id']) for c in decision['candidates']]
        extra=search_waypoints(state,forecast,geometry,work,'joint')
        reference=next(r for r in references if r['time_seconds']==now)
        assert [c['path'] for c in extra['candidates']]==[c['path'] for c in reference['search']['candidates']]
        chosen=best(candidates+extra['candidates'])
        incumbent=next(c for c in candidates+extra['candidates'] if c['id']==reference['search_selected']['id'])
        records.append(dict(time_seconds=now,public_prefix_sha256=digest(history),
            reference_forecast_sha256=reference['forecast_sha256'],forecast_sha256=forecast.fingerprint,
            source_models=models,state=state,geometry_refinements=fits,
            reference_original_prediction=reference['search_selected'],original_selected=incumbent,
            search_selected=chosen,reference_truth_cache='D2-M1 same_state_routes search_selected',
            candidates=candidates,search=extra,predicted_gain=incumbent['score']['score_seconds']-chosen['score']['score_seconds']))
        queries+=forecast.counts['receiver_queries'];proposals+=extra['proposals'];edges+=work.edges
    assert queries<=protocol['max_shadow_receiver_queries'] and proposals<=protocol['max_shadow_proposals'] and edges<=protocol['max_shadow_edges']
    assert fit_work['calls']<=9 and fit_work['solver_calls']<=9 and fit_work['residual_calls']<=360 and fit_work['kernel_terms']<=45000
    save(out/'reports/same_state_search.json',dict(records=records,receiver_queries=queries,proposals=proposals,
        edges=edges,refinement_resources=fit_work,new_dictionary_fits=0,sensor_points=0,truth_queries_before_freeze=0,
        scope='fixed saved birth/fitted bases; bounded geometry refinement; unchanged old models, weights, GP, candidate geometry and constraints'))
    return records


def evaluate_shadow_routes(out,world,cfg,records,protocol):
    result=[];queries=0
    for record in records:
        row=dict(time_seconds=record['time_seconds'],predicted_gain=record['predicted_gain'],routes={})
        for name in ('original_selected','search_selected'):
            candidate=record[name];route=candidate['path'];now=record['time_seconds']
            if name=='original_selected' and 'reference_truth_cache' in record:
                cached_rows=read(M1/'evaluation/same_state_routes.json')['rows']
                cached=deepcopy(next(r for r in cached_rows if r['time_seconds']==now)['routes']['search_selected'])
                assert cached['id']==candidate['id'] and cached['path']==route
                cached['reference_old_prediction']=cached['predicted']
                cached['predicted']=candidate['score']['score_seconds']
                cached['truth_source']='cached D2-M1 same-state frozen-route evaluation; no new truth queries'
                row['routes'][name]=cached
                continue
            synthetic=dict(motions=[dict(start_time_seconds=now+.5,traversed_m=route)],decisions=[])
            samples=numerical_samples(synthetic,cfg.aircraft.speed_mps)
            assert queries+len(samples)<=protocol['max_counterfactual_evaluation_points']
            for s in samples:s['truth_dbm']=float(world.truth(s['position_m'],s['time_seconds']))
            queries+=len(samples)
            q=math.fsum(s['dt_seconds']*10**((s['truth_dbm']+80.)/10.) for s in samples)
            duration=math.fsum(s['dt_seconds'] for s in samples)
            row['routes'][name]=dict(id=candidate['id'],path=route,predicted=candidate['score']['score_seconds'],
                true_flight_Q=q,true_flight_T=duration,true_flight_T_plus_Q=duration+q,samples=samples)
        row['true_frozen_route_gain']=row['routes']['original_selected']['true_flight_T_plus_Q']-row['routes']['search_selected']['true_flight_T_plus_Q']
        result.append(row)
    save(out/'evaluation/same_state_routes.json',dict(rows=result,truth_receiver_queries=queries,
        scope='frozen full-route counterfactual, not a realized closed-loop gain; common initial hover excluded'))
    return queries


def finalize(out):
    protocol = check(out)
    original_check(protocol)
    if any((out / 'evaluation').iterdir()) or any((out / 'predictions').iterdir()):
        raise ValueError('refuse duplicate scientific evaluation')
    episodes = {m: read(out / 'runs' / m / 'episode.json') for m in protocol['methods']}
    cfg = load_config(out / 'scenario.json')
    audit = audit_before_truth(out, protocol, episodes)
    audit['source_only_checks'] = audit_sources(episodes, cfg)
    if protocol['version']=='D2-M3':
        audit['refinement_resources']=audit_refinements(episodes,protocol)
        save(out/'reports/action_disagreement.json',action_diagnosis(episodes))
    save(out / 'reports/pre_truth_audit.json', audit)
    save(out / 'reports/flight_freeze.json', {str(p.relative_to(out)): sha(p) for p in (out / 'runs').rglob('*') if p.is_file()})
    world = ElectromagneticWorld(cfg)
    shadow_records=(shadow_geometry(out,protocol,cfg,world.geometry) if protocol['version']=='D2-M3' else
                    shadow_renewal(out,protocol,cfg,world.geometry) if protocol['version']=='D2-M2' else
                    shadow_search(out,protocol,cfg,world.geometry) if protocol['version']=='D2-M1' else None)
    if shadow_records is None:shadow(out, protocol)
    displays, count = {}, 0
    for method, ep in episodes.items():
        xy, arrays, metadata, queries = display_predictions(ep, cfg, world.geometry, JointSettings(**protocol['joint_settings']))
        # Unknown remains scored with its declared broad prior, but cannot be
        # displayed as an observed low/high-power spatial map.
        for k, index in enumerate(arrays['indices']):
            if not any(m['bases'] for m in ep['frames'][int(index)]['hypotheses']):
                arrays['actual_dbm'][k] = np.where(arrays['supported'][k], arrays['actual_dbm'][k], np.nan)
                metadata[k]['unsupported'] = 'unknown; explicit broad prior used in route cost, not observed spatial power'
        np.savez_compressed(out / 'predictions' / f'{method}.npz', **arrays)
        save(out / 'predictions' / f'{method}.json', metadata)
        displays[method] = (xy, arrays)
        count += queries
    assert count <= protocol['max_display_receiver_queries']
    save(out / 'reports/prediction_freeze.json', {str(p.relative_to(out)): sha(p) for p in (out / 'predictions').iterdir()})
    if shadow_records is not None:
        save(out/'reports/search_prediction_freeze.json',dict(path='reports/same_state_search.json',
             sha256=sha(out/'reports/same_state_search.json')))
    # First post-flight truth access occurs only after every prediction is frozen.
    results, truth_count, grid_count = [], 0, 0
    shadow_truth_count=evaluate_shadow_routes(out,world,cfg,shadow_records,protocol) if shadow_records is not None else 0
    for method, ep in episodes.items():
        samples = numerical_samples(ep, cfg.aircraft.speed_mps)
        assert truth_count+len(samples)<=protocol['max_flight_evaluation_points']
        for s in samples:
            s['truth_dbm'] = float(world.truth(s['position_m'], s['time_seconds']))
        truth_count += len(samples)
        assert truth_count+shadow_truth_count <= protocol['max_evaluation_points']
        q = math.fsum(s['dt_seconds'] * 10**((s['truth_dbm']+80)/10) for s in samples)
        rows = ep['joint_prediction_records']
        errors = [r['residual_db'] for r in rows]
        source_errors = [r['squared_error'] for r in ep['prediction_records']
                         if any(p['id'] != 'uninitialized_unknown' for p in r['issued']['predictions'])]
        result = dict(method=method, **ep['metrics'], Q=q,
            mission_T_plus_Q=ep['metrics']['simulation_seconds']+q,
            joint_prequential_rmse_db=math.sqrt(float(np.mean(np.square(errors)))),
            source_prequential_rmse_db=math.sqrt(float(np.mean(source_errors))) if source_errors else None,
            events=[dict(time_seconds=d['time_seconds'], reasons=d['public']['consultation_event']['reasons'],
                         selected_id=d['selected_id']) for d in ep['decisions']],
            mismatch_points=sum(r['all_source_mismatch'] for r in ep['trigger_records']),
            mismatch_budget_blocked=sum(r['all_source_mismatch'] and r['blocker'] == 'request_budget_exhausted'
                                        for r in ep['trigger_records']),
            actual_llm_calls=0)
        result['source_replacement_events']=len(ep.get('source_replacement_events',[]))
        result['retired_source_versions']=sum(len(r['retired']) for r in ep.get('source_replacement_events',[]))
        if protocol['version']=='D2-M3':
            result['refinement_calls']=sum(e['resources']['refinement']['calls'] for e in ep['inferences'])
            result['accepted_refinements']=sum(bool(f.get('geometry_refinement',{}).get('accepted')) for e in ep['inferences'] for r in e['rounds'] for f in r['fits'])
        save(out / 'evaluation' / f'{method}.json', dict(metrics=result, samples=samples))
        xy, arrays = displays[method]
        truth = []
        for index in arrays['indices']:
            truth.append(world.truth(xy, ep['frames'][int(index)]['time_seconds']).reshape(16, 21))
            grid_count += len(xy)
        assert grid_count <= protocol['max_replay_truth_points']
        np.savez_compressed(out / 'evaluation' / f'{method}_grids.npz', truth_dbm=np.asarray(truth))
        results.append(result)
    save(out / 'reports/results.json', dict(rows=results, evaluation_truth_points=truth_count,
         counterfactual_evaluation_truth_points=shadow_truth_count,
         replay_truth_points=grid_count, display_receiver_queries=count,
         interpretation=protocol['interpretation']))
    render(out, cfg, episodes, results)
    original_check(protocol)
    save(out / 'reports/completion_manifest.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


def render(out, cfg, episodes, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    from PIL import Image
    protocol=read(out/'protocol.json');version=protocol['version'];is_renewal=version=='D2-M2'
    is_geometry=version=='D2-M3';is_search=version in ('D2-M1','D2-M2','D2-M3')
    high = [(cfg.map.columns-1)*cfg.map.resolution_m, (cfg.map.rows-1)*cfg.map.resolution_m]
    for result in results:
        method, images = result['method'], []
        ep = episodes[method]
        arrays = np.load(out / 'predictions' / f'{method}.npz')
        truth = np.load(out / 'evaluation' / f'{method}_grids.npz')['truth_dbm']
        all_models={m['id']:m for f in ep['frames'] for m in f['hypotheses'] if m['bases']}
        aliases = {name:f'H{i+1}' for i,name in enumerate(all_models)}
        for k, index in enumerate(arrays['indices']):
            f = ep['frames'][int(index)]
            now = f['time_seconds']
            fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=100)
            fig.subplots_adjust(top=.82, bottom=.30, left=.055, right=.90, wspace=.13)
            cmap = plt.get_cmap('YlOrRd').copy(); cmap.set_bad('#b9bdc2')
            trail = np.asarray([p['position_m'] for p in ep['observations'] if p['time_seconds'] <= now])
            for side, (ax, field) in enumerate(zip(axes, (truth[k], arrays['actual_dbm'][k]))):
                im = ax.imshow(np.ma.masked_invalid(field), origin='lower', extent=(0, high[0], 0, high[1]),
                               vmin=-100, vmax=-55, cmap=cmap)
                for b in cfg.map.buildings:
                    ax.add_patch(Polygon(b['footprint_m'], color='#303030' if b['height_m']+cfg.aircraft.vertical_clearance_m >= cfg.aircraft.altitude_m else '#abc4cf', zorder=3))
                ax.plot(*trail.T, color='#125ec9', lw=2, zorder=4)
                if f['route_m']:
                    ax.plot(*np.asarray(f['route_m']).T, '--', color='#007c91', lw=1.5, zorder=4)
                ax.plot(*f['position_m'], 'ko', ms=5, zorder=7)
                ax.plot(*cfg.map.goal_m, 'b*', ms=11, zorder=7)
                for d in ep['decisions']:
                    if d['time_seconds'] <= now:
                        ax.plot(*d['public']['request']['state']['position_m'], 'o', mfc='none', mec='#a000b0', ms=8, zorder=6)
                ax.set(xlim=(0, high[0]), ylim=(0, high[1]), xlabel='x (m)', ylabel='y (m)')
                ax.set_title('True field + active true sources (evaluation only)' if side == 0 else 'Actual planner field + ONE source hypothesis', fontsize=11)
                if side == 0:
                    for number, field_source in enumerate(cfg.electromagnetic.fields, 1):
                        if field_source['start_seconds'] <= now < field_source['end_seconds']:
                            center = np.asarray(field_source['center_m'])+(now-field_source['start_seconds'])*np.asarray(field_source['velocity_mps'])
                            ax.plot(*center, '^', mfc='black', mec='white', ms=10, zorder=8)
                            ax.annotate(f'T{number}', center, xytext=(7, -14), textcoords='offset points', zorder=9,
                                        bbox=dict(fc='white', ec='none', alpha=.85, pad=1))
                elif np.any(arrays['supported'][k]) and not np.all(arrays['supported'][k]):
                    ax.contour(np.linspace(0, high[0], 21), np.linspace(0, high[1], 16), arrays['supported'][k], levels=[.5], colors=['cyan'], linewidths=1)
            sources = sorted([m for m in f['hypotheses'] if m['bases']], key=lambda m: (-m['weight'], m['id']))
            selected = sources[0] if sources else None
            info = 'No source model yet. Gray = unsupported unknown, not measured low power.'
            detail = ''
            if selected:
                h, dt = aliases[selected['id']], now-selected['reference_time_seconds']
                info = f"{h} = {selected['id']} | {selected['family']} | conditional weight={selected['weight']:.4e} | scored={selected['scored_points']}"
                descriptions = []
                for number, (b, peak) in enumerate(zip(selected['bases'], selected['peaks_dbm']), 1):
                    center = np.asarray(b['center_m'])+dt*np.asarray(b['velocity_mps'])
                    axes[1].add_patch(Circle(center, b['scale_m'], fill=False, ec='#a000ab', ls=':', lw=1.2, zorder=6))
                    axes[1].plot(*center, 'X', color='#a000ab', mec='white', ms=10, zorder=8)
                    axes[1].annotate(f'{h}/B{number}', center, xytext=(7, 8+13*(number-1)), textcoords='offset points',
                        zorder=9, color='#790482', bbox=dict(fc='white', ec='none', alpha=.85, pad=1), annotation_clip=True)
                    descriptions.append(f"B{number}: ({center[0]:.1f},{center[1]:.1f})m, scale={b['scale_m']:g}m, peak={peak+dt*b['rate_db_per_second']:.1f}dBm")
                detail = ' | '.join(descriptions)
            last = [d for d in ep['decisions'] if d['time_seconds'] <= now]
            event = f"last event={last[-1]['time_seconds']:g}s: {', '.join(last[-1]['public']['consultation_event']['reasons'])}" if last else 'before source initialization'
            fig.text(.055, .95, f"{version} | {method} | t={now:.2f}s | {f['phase']} | numerical organization, 0 LLM calls", fontsize=13)
            fig.text(.055, .91, event, fontsize=11)
            search_info=f" | lateral proposals={sum(d['grid_search']['proposals'] for d in last if d.get('grid_search'))}" if is_search else ''
            if is_renewal:
                retired=sum(len(e['retired']) for e in ep.get('source_replacement_events',[]) if e['time_seconds']<=now)
                search_info+=f' | suspended sources={retired}'
            if is_geometry:
                accepted=sum(bool(f.get('geometry_refinement',{}).get('accepted')) for e in ep['inferences'] if e['time_seconds']<=now for r in e['rounds'] for f in r['fits'])
                search_info+=f' | accepted refinements={accepted}'
            fig.text(.055, .87, f"True whole-flight T+Q={result['mission_T_plus_Q']:.2f} | length={result['executed_length_m']:.1f}m | source bank has NO power persistence"+search_info, fontsize=10)
            fig.text(.055, .235, 'Blue: flown; teal dashed: committed route; purple rings: consultations; cyan: GP support boundary.', fontsize=10)
            fig.text(.055, .20, 'Purple X: one alternative world; dotted radius: Gaussian scale. True sources are never provided to inference.', fontsize=10)
            fig.text(.055, .165, info, fontsize=10, color='#7b1879')
            fig.text(.055, .13, detail, fontsize=9, color='#7b1879')
            weights = ', '.join(f"{aliases[m['id']]}:{m['weight']:.3e}" for m in sources[:5])
            fig.text(.055, .095, 'Conditional relative weights (not absolute trust): '+weights, fontsize=10)
            fig.text(.055, .05, 'Single seen development scene. Synchronous callbacks; fixed 0.5s hover. Removing persistence does not establish model correctness.', fontsize=9)
            fig.colorbar(im, cax=fig.add_axes([.925, .34, .012, .45]), label='Power (dBm; higher = stronger)')
            fig.canvas.draw()
            images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
            if f['phase'] == 'after_decision':
                fig.savefig(out / 'replay' / f'{method}_event{len(last)}.png', dpi=100)
            plt.close(fig)
        images[0].save(out / 'replay' / f'{method}.gif', save_all=True, append_images=images[1:], duration=400, loop=0)
        with Image.open(out / 'replay' / f'{method}.gif') as image:
            assert image.n_frames == len(arrays['indices'])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>D2-M0：移除全图功率延续</title>
<style>body{font:17px/1.7 system-ui;max-width:1450px;margin:25px auto;padding:15px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:7px}</style>
<h1>D2-M0：移除全图功率延续，检查源假设与失配修订</h1>
<p>两条新的数值机制飞行，0次决策LLM调用。源池不再包含最近点测的全图延续；8秒开始拟合。source_revise额外允许全部源版本在新点测上均失配时提前修订，并在该次事件复核未飞路线。共享预算最多4事件，包括初始化。</p>
<p>左图真场和活动真源仅供事后评价；右图为该版本实际规划聚合场，紫叉只表示一个替代源解释，虚圆是高斯尺度。权重是当前源集合内的相对证据，不代表集合正确。初始灰色为未知，规划代价使用声明的宽先验，未当零暴露。</p>
<p><a href="../protocol.json">冻结合同与预算</a> · <a href="../reports/results.json">结果</a> · <a href="../reports/pre_truth_audit.json">因果/预算核验</a> · <a href="../reports/same_prefix_shadow.json">D1同前缀表示检查</a> · <a href="../../../docs/source_only_pilot_v1.md">方法合同</a></p>
<table><tr><th>方法</th><th>到达</th><th>航程m</th><th>T</th><th>真实Q</th><th>T+Q</th><th>源一步RMSE</th><th>事件数</th></tr>'''
    for r in results:
        page += f"<tr><td>{r['method']}</td><td>{r['success']}</td><td>{r['executed_length_m']:.2f}</td><td>{r['simulation_seconds']:.2f}</td><td>{r['Q']:.2f}</td><td>{r['mission_T_plus_Q']:.2f}</td><td>{r['source_prequential_rmse_db']:.2f}</td><td>{r['decision_count']}</td></tr>"
    page += '</table><p>单个已见开发场景；数值搜索仍为原型，不是论文强基线。D1仅作历史描述，不把这轮变化称为LLM收益，也不以热点更清楚证明预测更准。</p>'
    if is_search:
        page=page.replace('<h1>D2-M0：移除全图功率延续，检查源假设与失配修订</h1>',
            '<h1>D2-M1：侧向中间点搜索与路线排序</h1>')
        page=page.replace('D2-M0：移除全图功率延续','D2-M1：同一源预测下的侧向路径搜索')
        page=page.replace('两条新的数值机制飞行，0次决策LLM调用。源池不再包含最近点测的全图延续；8秒开始拟合。source_revise额外允许全部源版本在新点测上均失配时提前修订，并在该次事件复核未飞路线。共享预算最多4事件，包括初始化。',
            '两条新的数值机制飞行，0次决策LLM调用。源推断、权重、GP、触发和目标完全相同；search_lattice在共享预算内额外尝试最多50个侧向中间点。比较新路线的预测改善和真实任务代价，不把经典数值搜索收益记作LLM收益。')
        page=page.replace('same_prefix_shadow.json','same_state_search.json').replace('D1同前缀表示检查','同状态路线搜索')
        page=page.replace('source_only_pilot_v1.md','source_search_pilot_v1.md')
        page+='<p><a href="../evaluation/same_state_routes.json">冻结路线反事实真值诊断</a>：只解释路线排序，不等于完整闭环收益。</p>'
    if is_renewal:
        page=page.replace('D2-M1：同一源预测下的侧向路径搜索','D2-M2：集体失配后的活动源版本替换')
        page=page.replace('D2-M1：侧向中间点搜索与路线排序','D2-M2：活动源替换与路线排序')
        page=page.replace('两条新的数值机制飞行，0次决策LLM调用。源推断、权重、GP、触发和目标完全相同；search_lattice在共享预算内额外尝试最多50个侧向中间点。比较新路线的预测改善和真实任务代价，不把经典数值搜索收益记作LLM收益。',
            '两条数值机制飞行，0次决策LLM调用。源字典、GP、拟合、触发、侧向搜索和目标相同；renewal_replace在所有旧源的预发预测误差均≥8dB且新拟合成功时，将旧源留档，活动池换为新版本。新源尚未被未来点测验证；单点失配可能导致过早替换，不保证改善。')
        page=page.replace('source_search_pilot_v1.md','source_renewal_pilot_v1.md')
    if is_geometry:
        page=page.replace('D2-M1：同一源预测下的侧向路径搜索','D2-M3：同预算的源中心/尺度细化')
        page=page.replace('D2-M1：侧向中间点搜索与路线排序','D2-M3：源几何细化与实际行动')
        page=page.replace('两条新的数值机制飞行，0次决策LLM调用。源推断、权重、GP、触发和目标完全相同；search_lattice在共享预算内额外尝试最多50个侧向中间点。比较新路线的预测改善和真实任务代价，不把经典数值搜索收益记作LLM收益。',
            '两条数值机制飞行，0次决策LLM调用。两组共享相同细化预算；geometry_amplitude只重拟合幅度，geometry_bounded还允许中心各轴±40米和尺度除以/乘以sqrt(2)。GP、出生权重、触发、侧向搜索和目标规则相同。训练拟合改善不保证前方准确或任务收益。')
        page=page.replace('source_search_pilot_v1.md','source_geometry_pilot_v1.md')
        page+='<p><a href="../reports/action_disagreement.json">逐假设下一15米行动分歧</a>：使用既有候选评分，不增加场查询。</p>'
    for r in results:
        method = r['method']
        page += f'<h2>{method}</h2><p><a href="{method}.gif">打开GIF</a> · <a href="../runs/{method}/episode.json">原始飞行记录</a></p>'
        page += '<p>'+'；'.join(f"{e['time_seconds']:g}s：{', '.join(e['reasons'])} → {e['selected_id']}" for e in r['events'])+'</p>'
        page += f'<img src="{method}.gif" loading="lazy">'
    (out / 'replay/index.html').write_text(page+'</html>')
    (out / 'index.html').write_text(f'<!doctype html><meta charset="utf-8"><a href="replay/index.html">{version}新飞行与源假设回放</a>')
