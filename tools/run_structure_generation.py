"""新预算离线LLM结构表达式生成；最多4提交/16飞行/16配对，参考显式复用。"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json,strict_json
from uav_em.balanced_objective import PowerTimeSettings,_BalancedSession,choose,power_time_score,paired_seed
from uav_em.structural_policy import VERSION,FEATURES,ExpressionPolicy,validate_policy
from uav_em.generated_structure import GeneratedOptimizer,run_generated,segments_public,build_options,arm_features
from uav_em.decision_objective import ObjectiveSettings
from uav_em.repair_search import WorkBudget,FixedObjective
from uav_em.repair_process import canonical_path
from uav_em.route_repair import anchors
from uav_em.config import load_config
from uav_em.decision_log import render_decision_log
from run_repair_diagnostics import load_case
DEFAULT=ROOT/'outputs/interference_structure_generation_20261111'
SOURCE=ROOT/'outputs/interference_reconnection_20261111'
DRAFTS=('draft_01','draft_02','revision_01','revision_02')
REFERENCES=('single_C','single_batched','multi_equal','multi_feedback')
SEEDS=(611,613,617,619)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.now(timezone.utc).isoformat()

CONTRACT='''You are an independent OFFLINE algorithm designer. Read ONLY files within your own public directory. Do not read project AGENTS/docs/source/configs, hidden scenes, episodes, truth maps, other studies or agents. Do not search outside your directory, simulate, call other agents, or edit other files. This is a procedural allowlist, not OS isolation. Local Python may ONLY validate the JSON/expressions you author. Save one exact raw JSON response to response.json, refuse overwrite, and repeat that JSON in your final answer.
Required response: {"request_id": <request id>, "program": <schema below>, "rationale": <1..3000 characters>}. One submission consumes your slot, including invalid responses. Do not invent evaluation results or token counts.
PROGRAM: exactly version="structural_expressions_v2", name (1..100 chars), max_arms (integer1..3), and eight expression STRINGS: entry_score, entry_offset, exit_score, priority, sample_quota, edge_quota, stop, operation.
Expressions are hand-interpreted AST, not arbitrary Python. Allowed: finite numeric/bool constants abs<=1e6; + - * /; unary +/-/not; comparisons < <= > >= == !=; and/or (boolean semantics); X if CONDITION else Y; scalar min/max(2..6args), abs/sqrt/log1p(1arg). No pow, indexing, attributes, imports, assignment, lists, comprehensions, loops, functions or I/O. <=1200 chars,160AST nodes,24 depth per expression. Numeric results finite abs<=1e15. Only stop returns bool; all others numbers. Do not hard-code scenario IDs/coordinates/timestamps; runtime has none of those fields.
The runner freezes the common C-selected remaining route and preserves all bends + <=80m collinear anchors. Per decision common old-J pool is unchanged. Final ALL-candidate acceptance is C = full remaining flight time + normalized expected linear power, weight1/P0=-80dBm. Supported cells use current GP lognormal expectation; unknown uses explicit N(-100dBm,15dB) prior expected cost, NOT measured/low/bounded interference. No arrival-time prediction. You cannot change C, GP, source/sensor information, mission cap or seed. Lower C does not guarantee lower future true exposure.
entry_score: MAX over remaining segments, stable first ties. Features: index (0-based), fraction=index/max(1,number_segments-1), supported (length fraction with local GP data support), power_density (C power proxy seconds / segment flight seconds INCLUDING unknown prior), uncertainty (legacy J uncertainty cost/m), age (mean supported age seconds,0 if none), seconds (segment flight duration), detour (next two edges length minus chord,m; last0).
entry_offset: integer -3..0; entry=max(0,selected_segment+offset). Features index=selected segment,count=number segments. Remaining route only; no already-flown segment exists in this namespace.
exit_score: MAX ranking of all distinct later anchors; select up to max_arms. Features end_index, fraction=end_index/(anchors-1), span_seconds (original entry->exit flight time), chord_seconds (Euclidean time), chord_free (known geometry collision check), chord_C (WHOLE path C after replacing span by chord, even if blocked), base_C (whole original C), supported (supported duration fraction on original span), unknown_seconds (unsupported duration on original span). Chord geometry tests cost actual edges. A clear chord is not evidence of low actual interference.
priority: MAX per retained entry/exit arm EACH batch. Features attempts (#past actions for arm), successes (#past feasible actions incl duplicates), best_candidate_C (best past feasible WHOLE route C,1e12 if none), best_C (current shared best whole-route C), initial_C (original common selection), gain=max(0,initial_C-best_candidate_C), last_gain (global C improvement last time this arm tried), edges_spent/samples_spent (RRT-only costs for arm), chord_free/chord_C/chord_seconds/span_seconds/supported/unknown_seconds as above, batch (0..5), arms (#retained options). No requirement to explore all first; design your own schedule.
operation: numeric0=try direct chord shortcut,1=run C-RRT*. Features chord_free,chord_C,best_C,attempts,successes,batch. Shortcut still gets actual geometry/energy/whole-route acceptance checks. Duplicate attempts consume a step. Collision/failure never forces path acceptance.
sample_quota,edge_quota: nonnegative numbers; floor and cap to shared remaining work. Features samples_left (min of remaining post-pool700 and shared remaining), edges_left (currently available AFTER common scoring reserve), batches_left (including current of max6), batch, attempts (chosen arm), edges_per_sample (past global RRT edges/samples,min1; initial64). Zero either stops if RRT selected. Shortcut requires no RRT samples but checks cost edges. Quotas are evaluated for shortcuts too and must remain valid. No extra budgets after an error.
stop: bool before each action. Features batch,arms,best_C,initial_C,gain=max(0,initial_C-best_C),samples_left,edges_left,batches_left. Max6 actions total. New trees/empty search caches per RRT batch; no persistent tree reuse. Same geometric both-side local sampling library as references, at least30% global sampling. Seeds paired by base route/entry/exit/repeat, not selectable by you.
All candidates are complete splices into FROZEN original route; best retained route can change but indices do not drift. Total post-pool search <=700 samples, full2100 samples/10000 edges including common pool/feature geometry/candidate checks/final verification. Full mission <=1.25 initial reference length and energy unchanged. Feature integration and interpreter CPU recorded separately. Runtime errors stop generated process and retain existing best; they count as failures, no free recovery.
Objective: improve actual time+power tradeoff under partial observability/dynamics. Gate: all four development flights arrive, audits pass,0policy errors; total flight T+Q >=1% lower vs single_C, total including fixed hover lower, total excess dB*s <=single_C, each scene flight T+Q<=1.05*single_C. These are research selection criteria, not receiver safety thresholds. All four numeric references reported. Fixed GP/C may make goal difficult; do not invent measurements or circumvent acceptor.
Initial requests have public current-GP feature examples and numeric reference task metrics only. Revision requests additionally have own parent response and own per-scene scalar task metrics/error feedback. Future evaluation scalars are OFFLINE development fitness; no truth map/path/source is exposed. Runtime never sees these scalars. No validation/test access. Unseen generalization and online LLM necessity remain unproven.
'''

def check(out):
 p=read(out/'protocol.json')
 for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==sha(out/'reports/runtime_source'/rel)==h,rel
 for rel,h in p['input_sha256'].items():assert sha(out/rel)==h,rel
 for rel,h in p['external_sha256'].items():assert sha(ROOT/rel)==h,rel
 assert sha(out/'reports/frozen_design.md')==p['design_sha256']
 return p

def diagnostic_program():
 return dict(version=VERSION,name='feature_example_builder_NOT_model_artifact',max_arms=3,entry_score='supported * power_density',entry_offset='-1',exit_score='span_seconds - chord_seconds',priority='-attempts',sample_quota='samples_left / max(1,batches_left)',edge_quota='edges_left / max(1,batches_left)',stop='False',operation='1')

def examples(out):
 rows=[]
 for task,name in read(SOURCE/'protocol.json')['tasks'].items():
  case,s,cfg=load_case(out,out/'cases'/f'{name}.json');pool=read(out/'pool_reports'/f'{task}.json');selected=choose(pool['candidates'])
  session=_BalancedSession(s,case['context'],anchors(selected['path'],80.),case['max_length_m'],ObjectiveSettings(),case['seed'],pool,PowerTimeSettings(**case['score_settings']),'balanced')
  ctx=session.request(1,1);policy=ExpressionPolicy(diagnostic_program());arms,build=build_options(session,ctx,policy);initial_C=selected['balance']['score_seconds']
  af=arm_features(arms[0],[],0,initial_C,initial_C,len(arms));q=dict(samples_left=700,edges_left=session.work.summary()['edges_left']-128,batches_left=6,batch=0,attempts=0,edges_per_sample=64.)
  rows.append(dict(example=f'causal_example{len(rows)+1}',entry_score=build['segments'],entry_offset=[dict(index=build['selected_segment'],count=len(build['segments']))],exit_score=[a['features'] for a in arms],priority=[arm_features(a,[],0,initial_C,initial_C,len(arms)) for a in arms],sample_quota=[q],edge_quota=[q],stop=[dict(batch=0,arms=len(arms),best_C=initial_C,initial_C=initial_C,gain=0.,samples_left=700,edges_left=q['edges_left'],batches_left=6)],operation=[{k:af[k] for k in FEATURES['operation']}]))
 return rows

def prepare(out):
 out.mkdir(parents=True,exist_ok=False)
 for f in ('cases','scenarios','pool_reports','reference_paired','paired','public','submissions','runs','reports/runtime_source','evaluations','logs','replay','figures'):(out/f).mkdir(parents=True)
 external={}
 for f in ('cases','scenarios','pool_reports'):
  for p in (SOURCE/f).iterdir():shutil.copy2(p,out/f/p.name);external[str(p.relative_to(ROOT))]=sha(p)
 for p in (SOURCE/'paired').iterdir():shutil.copy2(p,out/'reference_paired'/p.name);external[str(p.relative_to(ROOT))]=sha(p)
 for name in ('protocol.json','reports/summary.json','reports/audit.json','reports/run_manifest.json','reports/completion_manifest.json'):
  p=SOURCE/name;external[str(p.relative_to(ROOT))]=sha(p)
 for rel,h in read(SOURCE/'reports/run_manifest.json').items():
  assert sha(SOURCE/rel)==h;external[str((SOURCE/rel).relative_to(ROOT))]=h
 old=read(SOURCE/'protocol.json')['source_sha256']
 for rel,h in old.items():
  if rel.startswith('src/uav_em/'):assert sha(ROOT/rel)==h
 shutil.copy2(SOURCE/'reports/summary.json',out/'reports/references_summary.json')
 shutil.copy2(ROOT/'docs/offline_structure_generation_v2.md',out/'reports/frozen_design.md')
 ex=examples(out);reference_rows=[dict(scene=f"task_{SEEDS.index(int(r['scene'][3:]))+1}",method=r['method'],**{k:r[k] for k in ('success','executed_length_m','interference_excess_integral_db_seconds','actual_joint_score_lambda1_seconds','actual_joint_with_hover_lambda1_seconds','repair_edge_evaluations')}) for r in read(SOURCE/'reports/summary.json')['flights']]
 for method in DRAFTS:
  public=out/'public'/method;public.mkdir();(public/'contract.md').write_text(CONTRACT)
  save_json(public/'feature_examples.json',ex);save_json(public/'numeric_reference_metrics.json',reference_rows)
 sources={}
 for p in [*(ROOT/'src/uav_em').glob('*.py'),Path(__file__).resolve(),ROOT/'tools/run_repair_diagnostics.py',ROOT/'tools/render_navigation_replays.py',ROOT/'tests/test_structural_generation.py']:
  rel=p.relative_to(ROOT);target=out/'reports/runtime_source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);sources[str(rel)]=sha(p)
 save_json(out/'protocol.json',dict(study='offline_structure_generation_v2',seeds=SEEDS,allowed_submissions=DRAFTS,max_design_sessions=4,max_new_flights=16,max_new_paired=16,reference_flights_reused=16,reference_paired_reused=16,references=REFERENCES,source_sha256=sources,external_sha256=external,design_sha256=sha(out/'reports/frozen_design.md'),input_sha256={str(p.relative_to(out)):sha(p) for f in ('cases','scenarios','pool_reports','reference_paired') for p in (out/f).iterdir()},validation_test_read=False,workers=4,replay_frame_stride=10))
 save_json(out/'status.json',dict(status='prepared'))
 for method in DRAFTS[:2]:request(out,method)

def request(out,method):
 check(out)
 if method not in DRAFTS:raise ValueError('unknown submission')
 public=out/'public'/method
 if (public/'request.json').exists():raise FileExistsError('request exists')
 revision=method.startswith('revision');parent='draft_'+method[-2:] if revision else None
 if revision:
  summary=read(out/'reports/initial_summary.json');status=read(out/'submissions'/parent/'status.json')
  shutil.copy2(out/'submissions'/parent/'raw_response.json',public/'parent_response.json')
  rows=[dict(scene=f"task_{SEEDS.index(int(r['scene'][3:]))+1}",**{k:r.get(k) for k in ('success','executed_length_m','interference_excess_integral_db_seconds','actual_joint_score_lambda1_seconds','actual_joint_with_hover_lambda1_seconds','repair_edge_evaluations','policy_errors','error')}) for r in summary['generated_rows'] if r['method']==parent]
  save_json(public/'own_feedback.json',dict(parent_accepted=status['accepted'],parse_error=status['error'],rows=rows,gate=summary['gates'][parent],own_execution_errors=summary.get('errors',{}).get(parent,[]),offline_development_fitness_only=True))
 payload=dict(request_id=method,parent=parent,created_utc=now(),design_focus='derive evidence-conditioned repair boundaries and stop/resource rules' if method.endswith('01') else 'derive complementary reconnections and scheduling using cost and search feedback',response_path=str((public/'response.json').resolve()),new_budget_max4_submissions=True)
 save_json(public/'request.json',payload)
 save_json(public/'request_manifest.json',{p.name:sha(p) for p in public.iterdir() if p.name not in ('response.json','request_manifest.json')})

def submit(out,method,agent_id,started_utc=None):
 check(out)
 if method not in DRAFTS:raise ValueError('invalid id')
 folder=out/'submissions'/method;folder.mkdir();public=out/'public'/method
 for name,h in read(public/'request_manifest.json').items():assert sha(public/name)==h
 raw=(public/'response.json').read_text();(folder/'raw_response.json').write_text(raw);accepted=False;error=None;program=None
 try:
  value=strict_json(raw)
  if not isinstance(value,dict) or set(value)!={'request_id','program','rationale'} or value['request_id']!=method:raise ValueError('response identity/schema')
  if not isinstance(value['rationale'],str) or not 1<=len(value['rationale'])<=3000:raise ValueError('short rationale required')
  program=validate_policy(value['program']);policy=ExpressionPolicy(program)
  for example in read(public/'feature_examples.json'):
   for key in FEATURES:
    for features in example[key]:
     v=policy.value(key,features)
     if key=='entry_offset' and (v!=int(v) or not -3<=v<=0):raise ValueError('entry offset range')
     if key=='operation' and v not in (0,1):raise ValueError('operation range')
     if key in ('sample_quota','edge_quota') and v<0:raise ValueError('negative quota')
  save_json(folder/'program.json',program);accepted=True
 except (ValueError,TypeError,KeyError) as exc:error=f'{type(exc).__name__}: {exc}'
 metadata=dict(method=method,accepted=accepted,error=error,agent_id=agent_id,independent_fork_none=True,actual_offline_design_sessions=1,started_utc=started_utc,submitted_utc=now(),request_manifest_sha256=sha(public/'request_manifest.json'),raw_response_sha256=sha(folder/'raw_response.json'),program_sha256=sha(folder/'program.json') if accepted else None,tokens=None,tokens_unavailable=True,filesystem_boundary='procedural_public_allowlist_not_OS_sandbox')
 save_json(folder/'status.json',metadata);return metadata

def flight_task(out,method,seed):
 from uav_em.simulation import simulate
 folder=out/'runs'/f'{method}_dev{seed}_seed{seed}';folder.mkdir();motions=[]
 def hook(stage,data):
  if stage=='flight_executed':motions.append(dict(end_time_seconds=data['time_seconds'],traversed_m=data['traversed_m']))
 try:
  cfg=load_config(out/'scenarios'/f'dev{seed}.json');program=read(out/'submissions'/method/'program.json');settings=PowerTimeSettings(prior_mean_dbm=cfg.electromagnetic.background_dbm,prior_std_db=cfg.belief.unknown_error_db)
  ep=simulate(cfg,advisor_name=f'{method}_offline_generated_no_runtime_llm',route_optimizer=GeneratedOptimizer(seed,program,settings),debug_hook=hook)
  ep['assumptions'].update(offline_structure_generated=True,runtime_model_calls=0,artifact_sha256=sha(out/'submissions'/method/'program.json'))
  for d in ep['decisions']:
   d['decision']['reason']=d['proposed_decision']['reason']='离线LLM生成的结构表达式；共同C/GP/总预算不变，运行时无模型'
   d['accepted_route_fixed_objective_meaning']='old_J_diagnostic_not_acceptance'
  save_json(folder/'episode.json',ep);save_json(folder/'execution.json',motions);save_json(folder/'metrics.json',ep['metrics']);(folder/'decision_log.txt').write_text(render_decision_log(ep))
  save_json(folder/'status.json',dict(status='completed',method=method,scene=f'dev{seed}',seed=seed,**ep['metrics']))
  return dict(method=method,seed=seed,success=ep['metrics']['success'],length=ep['metrics']['executed_length_m'],excess=ep['metrics']['interference_excess_integral_db_seconds'])
 except Exception as exc:
  error=f'{type(exc).__name__}: {exc}';save_json(folder/'status.json',dict(status='failed',method=method,scene=f'dev{seed}',seed=seed,error=error));return dict(method=method,seed=seed,error=error)

def run_phase(out,phase):
 p=check(out);methods=DRAFTS[:2] if phase=='initial' else DRAFTS[2:];manifest=out/'reports'/f'{phase}_manifest.json'
 if manifest.exists():raise FileExistsError('phase already run')
 for m in methods:assert (out/'submissions'/m/'status.json').exists()
 accepted=[m for m in methods if read(out/'submissions'/m/'status.json')['accepted']]
 for m in accepted:
  status=read(out/'submissions'/m/'status.json');assert sha(out/'submissions'/m/'program.json')==status['program_sha256']
  if any((out/'runs').glob(m+'_*')) or any((out/'paired').glob(m+'_*')):raise FileExistsError('refuse rerun')
  for task,name in read(SOURCE/'protocol.json')['tasks'].items():
   case,s,cfg=load_case(out,out/'cases'/f'{name}.json');pool=read(out/'pool_reports'/f'{task}.json')
   r=run_generated(s,case['context'],pool,case['max_length_m'],case['seed'],read(out/'submissions'/m/'program.json'),PowerTimeSettings(**case['score_settings']))
   save_json(out/'paired'/f'{m}_{task}.json',dict(method=m,task=task,case=name,**r));print(m,name,'C',r['selected_balance']['score_seconds'],'errors',r['policy_errors'],flush=True)
 with ProcessPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(flight_task,out,m,s) for m in accepted for s in SEEDS]):print(json.dumps(f.result()),flush=True)
 paths=[p for m in accepted for folder in (out/'runs').glob(m+'_*') for p in folder.iterdir() if p.is_file()]+[p for m in accepted for p in (out/'paired').glob(m+'_*')]
 save_json(manifest,dict(methods=list(methods),accepted=accepted,sha256={str(p.relative_to(out)):sha(p) for p in paths},all_choices_frozen_before_derived_evaluation=True))

def verify_phase(out,phase):
 m=read(out/'reports'/f'{phase}_manifest.json')
 for rel,h in m['sha256'].items():assert sha(out/rel)==h
 return m

def audit_generated(r,s,cfg,maximum):
 policy=ExpressionPolicy(r['generated_program']);checks=0;problem=FixedObjective(s,-80,WorkBudget());settings=PowerTimeSettings(**r.get('power_time_settings',asdict(PowerTimeSettings())))
 for c in r['candidates']:
  if c['feasible']:
   expected=power_time_score(problem,c['path'],cfg.aircraft.speed_mps,settings)
   for key in ('score_seconds','power_proxy_seconds','remaining_flight_seconds','unknown_seconds'):assert abs(expected[key]-c['balance'][key])<1e-6
   checks+=1
 selected=choose(r['candidates']);assert selected['id']==r['selected_candidate_id'] and selected['balance']==r['selected_balance']
 assert canonical_path(r['path'])==canonical_path(selected['path']) and selected['metrics']['length_m']<=maximum+1e-6
 assert r['resources']['samples']<=2100 and r['resources']['edges']<=10000
 assert r['resources']['samples']-r['starting_resources_already_paid']['samples']<=700
 assert r['actual_advisor_calls']==0 and r['final_geometry_verified']
 assert all(s.geometry.edge_free(a,b) for a,b in zip(r['path'],r['path'][1:]))
 for t in r['expression_trace']:
  assert policy.value('stop',t['stop_features'])==t['stop']
  if t['stop']:continue
  assert [policy.value('priority',f) for f in t['priority_features']]==t['priorities']
  assert policy.value('operation',t['operation_features'])==t['operation']
  assert policy.value('sample_quota',t['quota_features'])==t['raw_sample_quota']
  assert policy.value('edge_quota',t['quota_features'])==t['raw_edge_quota']
 for arm in r['reconnection_options']:
  assert 0<=arm['start_index']<arm['end_index']<len(r['frozen_base_route'])
  assert policy.value('exit_score',arm['features'])==arm['exit_rank']
 for t in r['generated_trace']:
  assert t['search_edges']<=t['edge_quota'] and t['search_samples']<=t['sample_quota']
  assert t['best_C_seconds']<=t['previous_best_C_seconds']+1e-6
 tree_checks=0
 for search in r['searches']:
  assert search['actual_seed']==paired_seed(cfg.seed,s.time_seconds,search['seed_stage'],search['seed_purpose'])
  if search.get('search_objective')=='power_time_C' and search['path']:
   expected=power_time_score(problem,search['path'],cfg.aircraft.speed_mps,settings)['score_seconds'];assert abs(expected-search['cost'])<1e-6;tree_checks+=1
 return dict(candidate_C_checks=checks,internal_C_checks=tree_checks,expressions_recomputed=len(r['expression_trace']),policy_errors=r['policy_errors'])

def evaluate_phase(out,phase):
 check(out);manifest=verify_phase(out,phase);dest=out/'reports'/f'{phase}_summary.json'
 if dest.exists():raise FileExistsError('summary exists')
 from uav_em.environment import ElectromagneticWorld,Measurement
 from uav_em.belief import ElectromagneticBelief
 from uav_em.risk_diagnostics import received_before_decision
 from uav_em.balanced_evaluation import evaluate_motions
 from uav_em.diagnostic_evaluation import truth_at_times
 from uav_em.risk_diagnostic_evaluation import evaluate_route
 rows=[];paired=[];audits=[];errors={}
 for method in manifest['accepted']:
  errors[method]=[]
  for path in (out/'paired').glob(method+'_*'):
   r=read(path);case,s,cfg=load_case(out,out/'cases'/f"{r['case']}.json");r['power_time_settings']=case['score_settings']
   audits.append(dict(case=r['case'],method=method,**audit_generated(r,s,cfg,case['max_length_m'])))
   e=evaluate_route(ElectromagneticWorld(cfg),s,r['path']);paired.append(dict(method=method,case=r['case'],current_C=r['selected_balance']['score_seconds'],**e,policy_errors=r['policy_errors']))
  for seed in SEEDS:
   folder=out/'runs'/f'{method}_dev{seed}_seed{seed}';status=read(folder/'status.json')
   if status['status']!='completed':rows.append(dict(scene=f'dev{seed}',method=method,success=False,error=status['error'],policy_errors=1));errors[method].append(status['error']);continue
   ep=read(folder/'episode.json');cfg=load_config(out/'scenarios'/f'dev{seed}.json');world=ElectromagneticWorld(cfg);actual=evaluate_motions(world,read(folder/'execution.json'))
   assert abs(actual['reintegrated_excess_db_seconds']-ep['metrics']['interference_excess_integral_db_seconds'])<1e-7
   assert abs(actual['reintegrated_exposure_seconds']-ep['metrics']['high_interference_exposure_seconds'])<1e-7
   cap=ep['decisions'][0]['repair_optimization']['mission_distance_cap_m'];checks=[];hover_q=hover_t=0.
   for i,d in enumerate(ep['decisions']):
    r=d['repair_optimization'];belief=ElectromagneticBelief(cfg,world.geometry)
    for pt in received_before_decision(ep['events'],i):belief.update(Measurement(tuple(pt['position_m']),pt['time_seconds'],pt['interference_dbm']))
    s=belief.snapshot(d['time_seconds']);assert s.observation_count==d['context']['electromagnetic_estimate']['observation_count']
    flown=sum(e['distance_m'] for e in ep['events'] if e['event']=='flight' and e['time_seconds']<=d['time_seconds']+1e-8)
    assert abs(flown-r['executed_distance_m'])<1e-6 and r['mission_distance_cap_m']==cap
    assert flown+r['best_metrics']['length_m']<=cap+1e-6
    required=r['best_metrics']['length_m']/cfg.aircraft.speed_mps*cfg.aircraft.flight_power_w/3600+cfg.simulation.decision_delay_seconds*cfg.aircraft.hover_power_w/3600
    assert required<=d['execution_checkpoint']['remaining_energy_wh']-cfg.aircraft.reserve_energy_wh+1e-7
    assert np.allclose(r['path'][0],d['context']['vehicle']['position_m']) and np.allclose(r['path'][-1],cfg.map.goal_m)
    checks.append(dict(time=d['time_seconds'],**audit_generated(r,s,cfg,cap-flown)))
    errors[method]+=r['policy_error_details']
    if d['accepted_action'] is not None:
     dt=min(cfg.simulation.decision_delay_seconds,cfg.simulation.max_seconds-d['time_seconds']);n=max(1,int(np.ceil(dt/.25)))
     power=truth_at_times(world,np.tile(d['context']['vehicle']['position_m'],(n,1)),d['time_seconds']+(np.arange(n)+.5)*dt/n)
     hover_q+=float(np.power(10.,(power+80)/10).sum()*dt/n);hover_t+=dt
   assert ep['metrics']['executed_length_m']<=cap+1e-6
   row=dict(scene=f'dev{seed}',method=method,**ep['metrics'],**actual,policy_errors=sum(d['repair_optimization']['policy_errors'] for d in ep['decisions']),actual_joint_score_lambda1_seconds=actual['flight_seconds_from_trace']+actual['flight_power_integral_normalized_seconds'],actual_joint_with_hover_lambda1_seconds=actual['flight_seconds_from_trace']+actual['flight_power_integral_normalized_seconds']+hover_q+hover_t,numeric_seconds=sum(d['repair_optimization']['numeric_compute_seconds'] for d in ep['decisions']))
   rows.append(row);audits.append(dict(run=folder.name,decisions=checks,geometry_energy_mission_and_point_measurements_verified=True))
 references=read(out/'reports/references_summary.json')['flights'];baseline={r['scene']:r for r in references if r['method']=='single_C'};gates={};totals={}
 for method in manifest['methods']:
  group=[r for r in rows if r['method']==method];valid=len(group)==4 and all(r.get('success',False) and r.get('policy_errors',1)==0 and 'actual_joint_score_lambda1_seconds' in r for r in group)
  if group:
   totals[method]={k:sum(r.get(k,0) for r in group) for k in ('executed_length_m','interference_excess_integral_db_seconds','actual_joint_score_lambda1_seconds','actual_joint_with_hover_lambda1_seconds','repair_edge_evaluations','numeric_seconds')};totals[method]['all_arrive']=all(r.get('success',False) for r in group)
  base_sum=lambda k:sum(r[k] for r in baseline.values())
  gate=dict(all_arrive_hard_audit_zero_policy_error=valid,flight_joint_reduction_at_least1percent=False,hover_joint_improves=False,total_excess_not_increased=False,per_scene_joint_within105percent=False)
  if valid:
   t=totals[method];gate.update(flight_joint_reduction_at_least1percent=t['actual_joint_score_lambda1_seconds']<=.99*base_sum('actual_joint_score_lambda1_seconds'),hover_joint_improves=t['actual_joint_with_hover_lambda1_seconds']<base_sum('actual_joint_with_hover_lambda1_seconds'),total_excess_not_increased=t['interference_excess_integral_db_seconds']<=base_sum('interference_excess_integral_db_seconds')+1e-8,per_scene_joint_within105percent=all(r['actual_joint_score_lambda1_seconds']<=1.05*baseline[r['scene']]['actual_joint_score_lambda1_seconds'] for r in group))
  gate['passes']=all(gate.values());gates[method]=gate
 result=dict(generated_rows=rows,reference_rows=references,reference_results_reused=True,paired=paired,gates=gates,totals=totals,errors=errors,offline_design_sessions=len(manifest['methods']),runtime_model_calls=0)
 save_json(dest,result);save_json(out/'reports'/f'{phase}_audit.json',audits);return result

def final_summary(out):
 check(out);verify_phase(out,'initial');verify_phase(out,'revisions')
 if (out/'reports/summary.json').exists():raise FileExistsError('final summary exists')
 a=read(out/'reports/initial_summary.json');b=read(out/'reports/revisions_summary.json')
 summary=dict(generated_rows=a['generated_rows']+b['generated_rows'],reference_rows=a['reference_rows'],reference_results_reused=True,paired=a['paired']+b['paired'],gates={**a['gates'],**b['gates']},totals={**a['totals'],**b['totals']},runtime_model_calls=0,actual_offline_design_sessions=4,max_new_flights=16,new_generated_expression_artifacts=sum(read(out/'submissions'/m/'status.json')['accepted'] for m in DRAFTS),unseen_validation=False)
 passing=[m for m,g in summary['gates'].items() if g['passes']]
 summary['selected_passing_artifact']=min(passing,key=lambda m:(summary['totals'][m]['actual_joint_score_lambda1_seconds'],summary['totals'][m]['interference_excess_integral_db_seconds'],summary['totals'][m]['executed_length_m'])) if passing else None
 save_json(out/'reports/summary.json',summary)

def report(out):
 check(out);verify_phase(out,'initial');verify_phase(out,'revisions');summary=read(out/'reports/summary.json')
 from render_navigation_replays import render_replays
 render_replays(out,frame_stride=10,playback_speed=2.,decision_pause_seconds=.5)
 md='# 离线LLM生成结构表达式v2\n\n4真实独立设计会话，2初稿+2修订；运行时0模型。新飞行/配对最多各16；四数值组16飞行/16配对来自上一轮冻结结果，明确复用。\n\n'
 md+=f"通过预声明完整门槛的选定产物：{summary['selected_passing_artifact']}。\n\n|方法|来源|到达|合计飞行T+Q/s|含悬停T+Q/s|合计超参考/dB·s|合计航程/m|策略错误|\n|---|---|---|---:|---:|---:|---:|---:|\n"
 for method in REFERENCES+DRAFTS:
  rows=[r for r in summary['reference_rows' if method in REFERENCES else 'generated_rows'] if r['method']==method]
  if not rows:md+=f'|{method}|新提交被拒绝|失败||||||\n';continue
  total=lambda k:sum(r.get(k,0) for r in rows)
  md+=f"|{method}|{'复用参考' if method in REFERENCES else '本轮生成新飞行'}|{all(r.get('success',False) for r in rows)}|{total('actual_joint_score_lambda1_seconds'):.3f}|{total('actual_joint_with_hover_lambda1_seconds'):.3f}|{total('interference_excess_integral_db_seconds'):.3f}|{total('executed_length_m'):.3f}|{int(total('policy_errors'))}|\n"
 md+='\n[完整指标及门槛](reports/summary.json) · [真实提交](submissions/) · [新实际飞行GIF](replay/index.html) · [原四组参考GIF](../interference_reconnection_20261111/replay/index.html)。\n\n只生成有限评分/分配表达式，不执行任意Python，不修改GP/C/传感。修订看过自身开发反馈，不是未见泛化或在线LLM必要性证据。失败与原始响应全部保留。\n';(out/'RESULTS.md').write_text(md)
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>离线LLM结构生成v2</title><style>body{font:17px/1.8 sans-serif;max-width:1350px;margin:auto}a{margin-right:18px}</style><h1>离线LLM结构表达式生成v2</h1><p>4真实独立设计会话，运行时0模型；新飞行最多16，四数值参考16飞行显式复用。生成入口/接回/调度短表达式，保持共同C/GP/总预算。</p><p><a href="replay/index.html">新实际飞行GIF</a><a href="RESULTS.md">全部数值</a><a href="reports/summary.json">指标及门槛</a><a href="reports/frozen_design.md">前瞻协议</a><a href="../interference_reconnection_20261111/replay/index.html">复用参考GIF</a></p>'+f"<p>通过完整门槛的产物：{summary['selected_passing_artifact']}</p>")
 save_json(out/'status.json',dict(status='completed',actual_offline_design_sessions=4,runtime_model_calls=0,new_flights=len(summary['generated_rows']),reference_flights_reused=16,selected_passing_artifact=summary['selected_passing_artifact']))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','submit','initial','initial_summary','request','revisions','evaluate','report']);p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--draft',choices=DRAFTS);p.add_argument('--agent-id');p.add_argument('--started-utc');a=p.parse_args()
 if a.phase=='prepare':prepare(a.out)
 elif a.phase=='submit':
  if not a.draft or not a.agent_id:p.error('submit needs draft/agent-id')
  print(json.dumps(submit(a.out,a.draft,a.agent_id,a.started_utc)))
 elif a.phase in ('initial','revisions'):run_phase(a.out,a.phase)
 elif a.phase=='initial_summary':evaluate_phase(a.out,'initial')
 elif a.phase=='request':request(a.out,a.draft)
 elif a.phase=='evaluate':evaluate_phase(a.out,'revisions');final_summary(a.out)
 else:report(a.out)
if __name__=='__main__':main()
