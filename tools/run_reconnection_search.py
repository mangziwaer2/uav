"""多接回点/分批竞争：冻结16同状态求解、16真实闭环、0模型。"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import hashlib,json
from pathlib import Path
import shutil,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json
from uav_em.reconnection_search import MODES,VERSION,ReconnectionOptimizer,run_competition,select_arm
from uav_em.balanced_objective import PowerTimeSettings,choose,power_time_score,paired_seed
from uav_em.config import load_config
from uav_em.decision_log import render_decision_log
from uav_em.repair_search import FixedObjective,WorkBudget
from uav_em.repair_process import canonical_path
from run_repair_diagnostics import load_case
from run_aligned_search import check,manifest,verify
DEFAULT=ROOT/'outputs/interference_reconnection_20261111'
PREVIOUS=ROOT/'outputs/interference_aligned_search_20261111'
SEEDS=(611,613,617,619)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(out):
 out.mkdir(parents=True,exist_ok=False)
 for f in ('cases','scenarios','pool_reports','paired','runs','evaluations','reports/runtime_source','logs','replay'):(out/f).mkdir(parents=True)
 external={}
 for f in ('cases','scenarios','pool_reports'):
  for p in (PREVIOUS/f).iterdir():
   shutil.copy2(p,out/f/p.name);external[str(p.relative_to(ROOT))]=sha(p)
 shutil.copy2(ROOT/'docs/reconnection_search_v1.md',out/'reports/frozen_design.md')
 sources={}
 for p in [*(ROOT/'src/uav_em').glob('*.py'),Path(__file__).resolve(),ROOT/'tools/run_aligned_search.py',ROOT/'tools/run_repair_diagnostics.py',ROOT/'tools/render_navigation_replays.py',ROOT/'tests/test_reconnection_search.py']:
  rel=p.relative_to(ROOT);target=out/'reports/runtime_source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);sources[str(rel)]=sha(p)
 save_json(out/'protocol.json',dict(study=VERSION,tasks=read(PREVIOUS/'protocol.json')['tasks'],seeds=SEEDS,modes=MODES,
  paired_tasks=16,new_flights=16,model_calls=0,new_generated_submissions=0,workers=4,
  post_pool_search_sample_cap=700,batches=6,shared_samples=2100,shared_edges=10000,
  main_comparison=['single_C','multi_feedback'],validation_test_generated_or_read=False,
  source_sha256=sources,external_sha256=external,design_sha256=sha(out/'reports/frozen_design.md'),
  input_sha256={str(p.relative_to(out)):sha(p) for f in ('cases','scenarios','pool_reports') for p in (out/f).iterdir()}))
 save_json(out/'status.json',dict(status='prepared'))

def audit_report(r,snapshot,cfg,settings,seed,maximum):
 selected=choose(r['candidates']);assert selected['id']==r['selected_candidate_id']
 assert canonical_path(selected['path'])==canonical_path(r['path'])
 assert selected['balance']==r['selected_balance']
 assert selected['metrics']['length_m']<=maximum+1e-6
 assert r['resources']['samples']<=2100 and r['resources']['edges']<=10000
 assert r['resources']['samples']-r['starting_resources_already_paid']['samples']<=700
 assert r['policy_errors']==r['actual_advisor_calls']==r['batch_errors']==0
 problem=FixedObjective(snapshot,-80,WorkBudget())
 checks=0
 for c in r['candidates']:
  if c['feasible']:
   score=power_time_score(problem,c['path'],cfg.aircraft.speed_mps,settings)
   for k in ('score_seconds','remaining_flight_seconds','power_proxy_seconds','unknown_seconds'):
    assert abs(score[k]-c['balance'][k])<1e-6,(k,c['id'])
   checks+=1
 arms=r['reconnection_options'];base=r['frozen_base_route'];history=[];search_count=0
 extra_searches=[s for s in r['searches'] if s.get('seed_stage')==VERSION]
 assert len(extra_searches)==len(r['competition_trace'])<=6
 for t,s,c in zip(r['competition_trace'],extra_searches,r['competition_candidates']):
  assert select_arm(r['competition_mode'],arms,history,t['batch'])['id']==t['arm_id']
  assert 0<=t['start_index']<t['end_index']<len(base)
  assert t['search_edges']<=t['edge_quota'] and t['search_samples']<=t['sample_quota']
  assert t['best_C_seconds']<=t['previous_best_C_seconds']+1e-7
  assert t['frozen_base_fingerprint']==canonical_path(base)
  if s['path']:
   assert np.allclose(s['path'][0],base[t['start_index']]) and np.allclose(s['path'][-1],base[t['end_index']])
   score=power_time_score(problem,s['path'],cfg.aircraft.speed_mps,settings)
   assert abs(score['score_seconds']-s['cost'])<1e-6;search_count+=1
  if c['feasible']:
   assert c['balance'] is not None
   score=power_time_score(problem,c['path'],cfg.aircraft.speed_mps,settings)
   assert abs(score['score_seconds']-c['balance']['score_seconds'])<1e-6
  history.append(t)
 for s in r['searches']:assert s['actual_seed']==paired_seed(seed,snapshot.time_seconds,s['seed_stage'],s['seed_purpose'])
 assert all(snapshot.geometry.edge_free(a,b) for a,b in zip(r['path'],r['path'][1:]))
 return dict(candidate_score_checks=checks,internal_C_path_checks=search_count,batches=len(history))

def paired(out):
 p=check(out)
 if any((out/'paired').iterdir()):raise FileExistsError('paired exists')
 for task,name in p['tasks'].items():
  case,s,cfg=load_case(out,out/'cases'/f'{name}.json');pool=read(out/'pool_reports'/f'{task}.json');reports=[]
  for mode in MODES:
   r=run_competition(s,case['context'],pool,case['max_length_m'],case['seed'],mode,PowerTimeSettings(**case['score_settings']))
   save_json(out/'paired'/f'{task}_{mode}.json',dict(task=task,case=name,**r));reports.append(r)
   print(name,mode,round(r['selected_balance']['score_seconds'],4),r['resources']['edges'],len(r['competition_trace']),flush=True)
  assert all(r['initial_resources']==reports[0]['initial_resources'] for r in reports)
  assert all(r['frozen_base_route']==reports[0]['frozen_base_route'] for r in reports)
  assert len({r['searches'][0]['actual_seed'] for r in reports})==1
  n=len(reports[2]['reconnection_options'])
  for k in range(n):
   a,b=reports[2]['competition_trace'][k],reports[3]['competition_trace'][k]
   assert a==b # identical exploration, including numerical work; no wall fields
 manifest(out,'paired','paired_manifest.json')

def flight_task(out,mode,seed):
 from uav_em.simulation import simulate
 folder=out/'runs'/f'{mode}_dev{seed}_seed{seed}';folder.mkdir();motions=[]
 def hook(stage,data):
  if stage=='flight_executed':motions.append(dict(end_time_seconds=data['time_seconds'],traversed_m=data['traversed_m']))
 try:
  cfg=load_config(out/'scenarios'/f'dev{seed}.json');settings=PowerTimeSettings(prior_mean_dbm=cfg.electromagnetic.background_dbm,prior_std_db=cfg.belief.unknown_error_db)
  ep=simulate(cfg,advisor_name=f'{mode}_numeric_no_model',route_optimizer=ReconnectionOptimizer(seed,mode,settings),debug_hook=hook)
  ep['assumptions'].update(reconnection_competition_study=True,runtime_model_calls=0,new_generated_submissions=0)
  for d in ep['decisions']:
   d['decision']['reason']=d['proposed_decision']['reason']=f'研究者编写的接回点/分批方法{mode}，固定C，无模型'
   d['accepted_route_fixed_objective_meaning']='old_J_diagnostic_not_acceptance'
  save_json(folder/'episode.json',ep);save_json(folder/'execution.json',motions);save_json(folder/'metrics.json',ep['metrics'])
  (folder/'decision_log.txt').write_text(render_decision_log(ep))
  save_json(folder/'status.json',dict(status='completed',method=mode,scene=f'dev{seed}',seed=seed,**ep['metrics']))
  return dict(method=mode,seed=seed,success=ep['metrics']['success'],length=ep['metrics']['executed_length_m'],excess=ep['metrics']['interference_excess_integral_db_seconds'])
 except Exception as exc:
  error=f'{type(exc).__name__}: {exc}';save_json(folder/'status.json',dict(status='failed',method=mode,scene=f'dev{seed}',seed=seed,error=error));return dict(method=mode,seed=seed,error=error)

def flights(out):
 check(out);verify(out,'paired_manifest.json',16)
 if any((out/'runs').iterdir()):raise FileExistsError('runs exist')
 with ProcessPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(flight_task,out,m,s) for m in MODES for s in SEEDS]):print(json.dumps(f.result()),flush=True)
 manifest(out,'runs','run_manifest.json')

def evaluate(out):
 p=check(out);verify(out,'paired_manifest.json',16);verify(out,'run_manifest.json')
 if (out/'reports/summary.json').exists():raise FileExistsError('evaluation exists')
 from uav_em.environment import ElectromagneticWorld,Measurement
 from uav_em.belief import ElectromagneticBelief
 from uav_em.risk_diagnostics import received_before_decision
 from uav_em.risk_diagnostic_evaluation import evaluate_route
 from uav_em.balanced_evaluation import evaluate_motions
 from uav_em.diagnostic_evaluation import truth_at_times
 paired_rows=[];audits=[];rows=[]
 for path in sorted((out/'paired').glob('*.json')):
  r=read(path);case,s,cfg=load_case(out,out/'cases'/f"{r['case']}.json")
  audits.append(dict(case=r['case'],method=r['competition_mode'],**audit_report(r,s,cfg,PowerTimeSettings(**case['score_settings']),case['seed'],case['max_length_m'])))
  e=evaluate_route(ElectromagneticWorld(cfg),s,r['path']);save_json(out/'evaluations'/path.name,dict(input_sha256=sha(path),evaluation=e))
  paired_rows.append(dict(case=r['case'],method=r['competition_mode'],current_C_seconds=r['selected_balance']['score_seconds'],length_m=r['best_metrics']['length_m'],**e,
   edges=r['resources']['edges'],samples=r['resources']['samples'],batches=len(r['competition_trace']),arm_sequence=[t['arm_id'] for t in r['competition_trace']]))
 for folder in sorted((out/'runs').iterdir()):
  status=read(folder/'status.json')
  if status['status']!='completed':rows.append(status);continue
  ep=read(folder/'episode.json');cfg=load_config(out/'scenarios'/f"{status['scene']}.json");world=ElectromagneticWorld(cfg)
  actual=evaluate_motions(world,read(folder/'execution.json'))
  assert abs(actual['reintegrated_excess_db_seconds']-ep['metrics']['interference_excess_integral_db_seconds'])<1e-7
  assert abs(actual['reintegrated_exposure_seconds']-ep['metrics']['high_interference_exposure_seconds'])<1e-7
  assert abs(actual['flight_seconds_from_trace']*cfg.aircraft.speed_mps-ep['metrics']['executed_length_m'])<1e-6
  cap=ep['decisions'][0]['repair_optimization']['mission_distance_cap_m'];hover_t=hover_q=0.;checks=[]
  for i,d in enumerate(ep['decisions']):
   r=d['repair_optimization'];points=received_before_decision(ep['events'],i);belief=ElectromagneticBelief(cfg,world.geometry)
   for pt in points:belief.update(Measurement(tuple(pt['position_m']),pt['time_seconds'],pt['interference_dbm']))
   s=belief.snapshot(d['time_seconds']);assert s.observation_count==d['context']['electromagnetic_estimate']['observation_count']
   assert r['mission_distance_cap_m']==cap
   flown=sum(e['distance_m'] for e in ep['events'] if e['event']=='flight' and e['time_seconds']<=d['time_seconds']+1e-8)
   assert abs(flown-r['executed_distance_m'])<1e-6
   selected=choose(r['candidates']);assert selected and flown+selected['metrics']['length_m']<=cap+1e-6
   assert np.allclose(r['path'][0],d['context']['vehicle']['position_m']) and np.allclose(r['path'][-1],cfg.map.goal_m)
   required=selected['metrics']['length_m']/cfg.aircraft.speed_mps*cfg.aircraft.flight_power_w/3600+cfg.simulation.decision_delay_seconds*cfg.aircraft.hover_power_w/3600
   assert required<=d['execution_checkpoint']['remaining_energy_wh']-cfg.aircraft.reserve_energy_wh+1e-7
   settings=PowerTimeSettings(**r['power_time_settings']);maximum=min(cap-flown,(d['execution_checkpoint']['remaining_energy_wh']-cfg.aircraft.reserve_energy_wh-cfg.simulation.decision_delay_seconds*cfg.aircraft.hover_power_w/3600)*3600*cfg.aircraft.speed_mps/cfg.aircraft.flight_power_w)
   checks.append(dict(time=d['time_seconds'],**audit_report(r,s,cfg,settings,cfg.seed,maximum)))
   if d['accepted_action'] is not None:
    dt=min(cfg.simulation.decision_delay_seconds,cfg.simulation.max_seconds-d['time_seconds']);n=max(1,int(np.ceil(dt/.25)))
    power=truth_at_times(world,np.tile(d['context']['vehicle']['position_m'],(n,1)),d['time_seconds']+(np.arange(n)+.5)*dt/n)
    hover_q+=float(np.power(10.,(power+80)/10).sum()*dt/n);hover_t+=dt
  assert ep['metrics']['executed_length_m']<=cap+1e-6
  rows.append(dict(scene=status['scene'],method=status['method'],**ep['metrics'],**actual,
   actual_joint_score_lambda1_seconds=actual['flight_seconds_from_trace']+actual['flight_power_integral_normalized_seconds'],
   actual_joint_with_hover_lambda1_seconds=actual['flight_seconds_from_trace']+actual['flight_power_integral_normalized_seconds']+hover_q+hover_t,
   numeric_seconds=sum(d['repair_optimization']['numeric_compute_seconds'] for d in ep['decisions']),
   batches=sum(len(d['repair_optimization']['competition_trace']) for d in ep['decisions'])))
  audits.append(dict(run=folder.name,decisions=checks,geometry_energy_lifetime_caps_verified=True,flight_exposure_reintegrated=True))
 totals={}
 for mode in MODES:
  group=[r for r in rows if r.get('method')==mode];valid=len(group)==4 and all(r.get('success',False) for r in group)
  totals[mode]=dict(all_arrive=valid)
  for k in ('executed_length_m','interference_excess_integral_db_seconds','flight_power_integral_normalized_seconds','actual_joint_score_lambda1_seconds','actual_joint_with_hover_lambda1_seconds','repair_edge_evaluations','numeric_seconds','batches'):totals[mode][k]=sum(r.get(k,0) for r in group)
 save_json(out/'reports/summary.json',dict(paired=paired_rows,flights=rows,totals=totals,model_calls=0,new_generated_submissions=0))
 save_json(out/'reports/audit.json',audits)

def report(out):
 check(out);verify(out,'paired_manifest.json',16);verify(out,'run_manifest.json')
 summary=read(out/'reports/summary.json')
 from render_navigation_replays import render_replays
 render_replays(out,frame_stride=6,playback_speed=2.,decision_pause_seconds=.5)
 md='# 多接回点/分批竞争搜索\n\n16同状态求解/16实际闭环，0模型或新生成。四组同GP/共同池/C接受器及预算。本轮新配对流，旧研究数值不是本轮分母。\n\n|场景|方法|到达|航程/m|超参考/dB·s|真实T+Q/s|含悬停T+Q/s|边评价|批数|\n|---|---|---|---:|---:|---:|---:|---:|---:|\n'
 for r in summary['flights']:
  if 'error' in r:md+=f"|{r['scene']}|{r['method']}|失败|{r['error']}||||||\n";continue
  md+=f"|{r['scene']}|{r['method']}|{r['success']}|{r['executed_length_m']:.3f}|{r['interference_excess_integral_db_seconds']:.3f}|{r['actual_joint_score_lambda1_seconds']:.3f}|{r['actual_joint_with_hover_lambda1_seconds']:.3f}|{r['repair_edge_evaluations']}|{r['batches']}|\n"
 md+='\n## 同状态固定路线（开环反事实）\n\n|状态|方法|当前C/s|真实T+Q/s|超参考/dB·s|边|接回点顺序|\n|---|---|---:|---:|---:|---:|---|\n'
 for r in summary['paired']:md+=f"|{r['case']}|{r['method']}|{r['current_C_seconds']:.3f}|{r['future_joint_seconds']:.3f}|{r['future_excess_db_seconds']:.3f}|{r['edges']}|{'→'.join(r['arm_sequence'])}|\n"
 md+='\n[冻结设计](reports/frozen_design.md) · [全部结构化指标](reports/summary.json) · [GP重建/候选/预算核验](reports/audit.json) · [实际闭环GIF](replay/index.html)。未知仍未知，C是当前代理，不预测未来。没有LLM贡献或未见泛化结论。\n'
 (out/'RESULTS.md').write_text(md)
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>多接回点分批搜索</title><style>body{font:17px/1.8 sans-serif;max-width:1300px;margin:auto}a{margin-right:18px}</style><h1>多接回点与分批竞争搜索</h1><p>研究者编写的新结构；16同状态求解、16真实闭环，0模型调用。单次、同接回点分批、多接回点均分、多接回点数值反馈四组。</p><p><a href="replay/index.html">16个闭环GIF</a><a href="RESULTS.md">全部数值</a><a href="reports/summary.json">结构化结果</a><a href="reports/audit.json">逐状态核验</a><a href="reports/frozen_design.md">前瞻协议</a></p>')
 save_json(out/'status.json',dict(status='completed',paired_tasks=16,new_flights=16,model_calls=0,new_generated_submissions=0))

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','paired','flights','evaluate','report','all']);parser.add_argument('--out',type=Path,default=DEFAULT);a=parser.parse_args()
 for fn in (prepare,paired,flights,evaluate,report) if a.phase=='all' else (globals()[a.phase],):fn(a.out)
if __name__=='__main__':main()
