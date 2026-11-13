"""直接数值vs预设算子短程序：同快照配对 + 一个场景的飞行验证。"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json
from uav_em.belief import MapSnapshot
from uav_em.config import load_config
from uav_em.environment import Geometry
from uav_em.repair_program import API_DESCRIPTION, FileProgramPolicy, LibraryNumericalPolicy
from uav_em.route_repair import RepairSettings, RouteRepairOptimizer, FileRepairPolicy
from uav_em.simulation import simulate
from uav_em.decision_log import render_decision_log
from uav_em.visualization import render
from render_navigation_replays import render_replays
from run_repair_study import CONTRACT as DIRECT_CONTRACT, audit_exchange

SETTINGS=RepairSettings(max_rounds=3)
METHODS=('library_numeric','direct_values','operator_program')

COMMON='''
This is a NEW interface comparison. Read ONLY your own public directory's contract.md, control.json,
requests/*.json and responses/*.json. Do not read source, other experiments/arms, hidden maps/configs,
other agents, or parent folders. Do not spawn agents. Procedural allowlist, not OS isolation.
Only use local tools to read/poll your allowed requests and atomically publish matching responses.
Do not run private planning/numeric-parameter scripts yourself: numeric assistance in the program arm
is provided ONLY by the recorded preset-operator interpreter. Do not search outside the public folder.
There are four independent frozen-snapshot tasks, followed by ONE closed-loop flight task.
Fresh request memory per task. Keep task identities separate. The first request in each paired task
has exactly the same observation/geometry/route/GP/J/search seed/budgets for all arms. Later rounds
can differ after route acceptance. No flight or new measurement inside paired tasks.
Remain available until control says completed/failed; a temporary absence of requests is NOT completion.
Poll no longer than20s per call; all requests need a response, even if keeping the current route.
At most3 calls per decision, 2100 sample proposals/10000 edges shared. Current GP J is NOT future
true exposure. Only remaining route anchors may be used; past measurement positions are not anchors.
No unknown field is known safe. UAV senses ONLY scalar power at its position. No true source info.
No arbitrary online Python execution: the program arm only interprets listed preset library calls.
The same preset library is documented for both arms. It computes from public context ONLY.
Library numeric formulas/priors are frozen research heuristics, not calibrated safety estimates.
'''


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    for part in ('scenarios','runs','paired/cases','paired/results','reports','logs','replay',
                 'public/direct_values/requests','public/direct_values/responses',
                 'public/operator_program/requests','public/operator_program/responses'):
        (out/part).mkdir(parents=True,exist_ok=True)
    # Reuse declared scenes, without outcome-based filtering or source retuning.
    first=load_config(ROOT/'outputs/interference_route_repair_20261110/scenarios/r01.json')
    second=load_config(ROOT/'configs/local_power.json')
    for scene,cfg,seed in [('s01',first,311),('s02',second,313)]:
        cfg=replace(cfg,seed=seed,simulation=replace(cfg.simulation,max_seconds=140,max_advisor_calls=24))
        cfg.validate();save_json(out/'scenarios'/f'{scene}.json',cfg.to_dict())
    contract_direct=COMMON+'\nDirect-values response rules:\n'+DIRECT_CONTRACT+'\nPreset numeric library reference (NOT executed in direct-values arm):\n'+API_DESCRIPTION
    contract_program=COMMON+'\n'+API_DESCRIPTION+'''
Output exactly one JSON response matching request_id:
{"request_id":"...","decision":{"program":"...","reason":"short evidence-based justification"},
 "strategy":{"mode":"baseline"},"memory":"same-task memory <=1200chars"}
Use actual newline characters in the program string (JSON escapes them). No Markdown fences.
The program is a straight-line composition of preset functions; it must assign decision.
Examples (format only, not recommended actions):
decision = keep()
or:
part = select_interval(metric="interference", padding=1)
boxes = corridors(part, side="both", scale="local")
dist = sampling(boxes, weighting="cost", concentration="focused")
decision = local_rrt(part, dist, quota(mode="balanced"))
When there is no feasible incumbent, use global_rrt; interval functions then are unavailable.
shortcut(whole()) can shorten a route ONLY if collision-free and lower J; the verifier decides.
You can select different metrics, sides, scales, quota modes and compose stop_if_no_gain.
Do not put coordinates/floats into code. The fixed numerical library derives them from public data.
Both syntactic acceptance and numerical outcome are logged; errors are NOT silently repaired.
'''
    for method,contract in [('direct_values',contract_direct),('operator_program',contract_program)]:
        public=out/'public'/method;(public/'contract.md').write_text(contract)
        save_json(public/'control.json',{'status':'prepared','paired_tasks':4,'closed_loop_tasks':1})
    sources={}
    for p in [*(ROOT/'src/uav_em').glob('*.py'),Path(__file__).resolve(),ROOT/'tools/run_repair_study.py',ROOT/'tools/render_navigation_replays.py']:
        rel=p.relative_to(ROOT);dest=out/'reports/runtime_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest);sources[str(rel)]=sha(p)
    save_json(out/'protocol.json',{'study':'direct_values_vs_preset_operator_composition_development_pilot',
        'methods':list(METHODS),'scenes':['s01','s02'],'paired_selection':'first numeric-control decision at/after12s and30s per scene; no result filtering',
        'closed_loop_scene':'s02','same_settings':SETTINGS.__dict__,'sources_sha256':sources,
        'scenario_sha256':{s:sha(out/'scenarios'/f'{s}.json') for s in ('s01','s02')},
        'contract_sha256':{m:sha(out/'public'/m/'contract.md') for m in ('direct_values','operator_program')},
        'model_sessions':'two fresh independent fork-none sessions; each reads own public folder only',
        'scope':'preset operators are researcher-authored offline; model composes them online; no raw code execution, evolution, GP or objective changes',
        'limits':'small development pilot; format plus arithmetic assistance compared, not proof of intrinsic numeric token failure; no calibrated risk or statistical superiority'})
    save_json(out/'status.json',{'status':'prepared'})


def check_frozen(out):
    p=read(out/'protocol.json')
    for name,h in p['sources_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('source changed after freeze: '+name)
    for scene,h in p['scenario_sha256'].items():
        if sha(out/'scenarios'/f'{scene}.json')!=h:raise ValueError('scenario changed')
    return p


class CaptureOptimizer(RouteRepairOptimizer):
    def __init__(self,out,scene):
        super().__init__(settings=SETTINGS,numeric_policy=LibraryNumericalPolicy())
        self.out,self.scene,self.saved=out,scene,set()

    def optimize(self,snapshot,context,route,max_length_m,seed,call_limit):
        for threshold in (12,30):
            if threshold not in self.saved and snapshot.time_seconds>=threshold and route:
                name=f'{self.scene}_after{threshold}';base=self.out/'paired/cases'/name
                np.savez_compressed(base.with_suffix('.npz'),estimate_dbm=snapshot.estimate_dbm,
                    uncertainty_db=snapshot.uncertainty_db,supported=snapshot.supported,age_seconds=snapshot.age_seconds)
                save_json(base.with_suffix('.json'),{'case':name,'scene':self.scene,'time_seconds':snapshot.time_seconds,
                    'context':context,'route':route,'max_length_m':max_length_m,'seed':seed,
                    'observation_count':snapshot.observation_count,'estimation_details':snapshot.estimation_details})
                self.saved.add(threshold)
        return super().optimize(snapshot,context,route,max_length_m,seed,call_limit)


def store_episode(directory,ep,method,scene,seed):
    save_json(directory/'episode.json',ep);save_json(directory/'metrics.json',ep['metrics'])
    (directory/'decision_log.txt').write_text(render_decision_log(ep))
    save_json(directory/'status.json',dict(status='completed',method=method,scene=scene,seed=seed,**ep['metrics']))


def baseline(out):
    check_frozen(out)
    for scene in ('s01','s02'):
        cfg=load_config(out/'scenarios'/f'{scene}.json');directory=out/'runs'/f'library_numeric_{scene}_seed{cfg.seed}'
        directory.mkdir(exist_ok=False)
        optimizer=CaptureOptimizer(out,scene)
        ep=simulate(cfg,advisor_name='numeric_library_no_model',route_optimizer=optimizer)
        store_episode(directory,ep,'library_numeric',scene,cfg.seed)
        print('baseline',scene,ep['metrics']['success'],optimizer.saved,flush=True)
        if optimizer.saved!={12,30}:raise ValueError('declared paired snapshots unavailable; no silent replacement')
    for case in sorted((out/'paired/cases').glob('*.json')):
        optimize_case(out,case,'library_numeric')


def load_case(out,path):
    case=read(path);cfg=load_config(out/'scenarios'/f"{case['scene']}.json")
    g=Geometry(cfg.map,cfg.aircraft.clearance_m,cfg.aircraft.altitude_m,cfg.aircraft.vertical_clearance_m)
    with np.load(path.with_suffix('.npz')) as arrays:
        snap=MapSnapshot(case['time_seconds'],**{k:arrays[k].copy() for k in arrays.files},
                         observation_count=case['observation_count'],geometry=g,estimation_details=case['estimation_details'])
    return case,snap


def model_policy(out,method,mission):
    cls=FileRepairPolicy if method=='direct_values' else FileProgramPolicy
    return cls(out/'public'/method,mission,900)


def optimize_case(out,path,method):
    case,snap=load_case(out,path);directory=out/'paired/results'/method/case['case'];directory.mkdir(parents=True,exist_ok=False)
    policy=None if method=='library_numeric' else model_policy(out,method,'pair_'+case['case'])
    optimizer=RouteRepairOptimizer(policy,SETTINGS,numeric_policy=LibraryNumericalPolicy() if policy is None else None)
    plan,report=optimizer.optimize(snap,case['context'],case['route'],case['max_length_m'],case['seed'],3)
    save_json(directory/'result.json',{'case':case['case'],'method':method,'path':plan.path,'report':report})
    print('paired',method,case['case'],'J',report['initial_metrics']['objective'],'->',report['best_metrics']['objective'],flush=True)


def run_arm(out,method):
    p=check_frozen(out);public=out/'public'/method
    if sha(public/'contract.md')!=p['contract_sha256'][method]:raise ValueError('contract changed')
    save_json(public/'control.json',{'status':'running','paired_tasks':4,'closed_loop_tasks':1})
    try:
        cases=sorted((out/'paired/cases').glob('*.json'))
        if len(cases)!=4:raise ValueError('baseline must produce all four paired snapshots first')
        for path in cases:optimize_case(out,path,method)
        scene='s02';cfg=load_config(out/'scenarios'/f'{scene}.json');directory=out/'runs'/f'{method}_{scene}_seed{cfg.seed}'
        directory.mkdir(exist_ok=False)
        optimizer=RouteRepairOptimizer(model_policy(out,method,'flight_'+scene),SETTINGS)
        ep=simulate(cfg,advisor_name=method+'_subagent',route_optimizer=optimizer)
        store_episode(directory,ep,method,scene,cfg.seed)
        save_json(public/'control.json',{'status':'completed','paired_tasks':4,'closed_loop_tasks':1})
        print('flight',method,ep['metrics'],flush=True)
    except BaseException as exc:
        save_json(public/'control.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
        raise


def reports(out):
    protocol=read(out/'protocol.json');paired=[];flights=[];audits=[];calls=[]
    for path in sorted((out/'paired/results').glob('*/*/result.json')):
        item=read(path);paired.append(item)
        collect_audit(out,item['method'],item['report'],item['case'],audits,calls)
    if len(paired)!=12:raise ValueError('expected4 paired snapshots x3 methods')
    for status in sorted((out/'runs').glob('*/status.json')):
        s=read(status);ep=read(status.parent/'episode.json');render(ep,status.parent/'summary.png')
        flights.append({'method':s['method'],'scene':s['scene'],'seed':s['seed'],**ep['metrics']})
        for d in ep['decisions']:collect_audit(out,s['method'],d['repair_optimization'],'flight_'+s['scene'],audits,calls)
    # First-round public inputs and initial plans must be exactly paired. Later versions may differ.
    pair_checks=[]
    for case in sorted({r['case'] for r in paired}):
        items=[r for r in paired if r['case']==case];contexts=[r['report']['trace'][0]['context'] for r in items]
        pair_checks.append({'case':case,'same_initial_context':all(c==contexts[0] for c in contexts),
            'same_initial_J':all(r['report']['initial_metrics']==items[0]['report']['initial_metrics'] for r in items)})
    source_checks={'frozen_sources_intact':all(sha(out/'reports/runtime_source'/n)==h for n,h in protocol['sources_sha256'].items()),
        'current_matches_frozen':all(sha(ROOT/n)==h for n,h in protocol['sources_sha256'].items())}
    all_pass=all(all(v for v in a.values() if isinstance(v,bool)) for a in audits+pair_checks+[source_checks])
    save_json(out/'reports/audit.json',{'checks':audits,'pairing':pair_checks,**source_checks,'all_pass':all_pass,
        'filesystem_boundary':'procedural_allowlist_not_OS_sandbox','raw_python_execution':False})
    save_json(out/'results.json',{'paired':[{'case':r['case'],'method':r['method'],
        'initial':r['report']['initial_metrics'],'final':r['report']['best_metrics'],
        'resources':r['report']['resources'],'calls':r['report']['actual_advisor_calls'],
        'errors':r['report']['advisor_failures'],'recovery':r['report']['numeric_recovery']} for r in paired],
        'flights':flights})
    save_json(out/'reports/model_calls.json',calls)
    render_replays(out,frame_stride=5,playback_speed=2.,decision_pause_seconds=.6)
    page='''<!doctype html><meta charset="utf-8"><title>数值输出与算子程序对比</title>
<style>body{max-width:1300px;margin:30px auto;font:16px/1.8 sans-serif}td,th{border:1px solid #ccc;padding:7px}table{border-collapse:collapse}img{max-width:100%}pre{white-space:pre-wrap}</style>
<h1>直接数值输出 vs 预设算子短程序</h1><p><a href="../index.html">总入口</a> · <a href="replay/index.html">集中GIF</a> · <a href="results.json">原始指标</a> · <a href="reports/model_calls.json">真实响应/程序计算/执行反馈</a> · <a href="reports/audit.json">审计</a></p>
<p>先做4个同快照配对任务，再在s02做闭环飞行。固定点测/GP/目标/操作集合/总搜索上限。数值算子是研究者预先编写；模型只组合，不执行任意Python。比较包含数值工具辅助的差异，不单独证明代码语言形式更优。</p>
<h2>同快照配对</h2><p>评价当前GP目标J，不把它当作真实未来暴露；后续反馈轮的路线可不同。初始公共上下文相同。</p><table><tr><th>快照</th><th>方法</th><th>初始J</th><th>最终J</th><th>边评价</th><th>模型调用</th><th>错误</th></tr>'''
    for r in paired:
        report=r['report'];row=[r['case'],r['method'],round(report['initial_metrics']['objective'],3),round(report['best_metrics']['objective'],3),report['resources']['edges'],report['actual_advisor_calls'],report['advisor_failures']]
        page+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>'
    page+='</table><h2>闭环飞行</h2><p>s01只有数值轨迹，用于取得配对快照；闭环三方法对比只看s02。轨迹变化会导致后续观测不同。</p><table><tr><th>场景</th><th>方法</th><th>到达</th><th>航程/m</th><th>超参考积分/dB·s</th><th>边评价</th><th>模型调用</th><th>错误</th></tr>'
    for f in flights:
        row=[f['scene'],f['method'],f['success'],round(f['executed_length_m'],2),round(f['interference_excess_integral_db_seconds'],3),f['repair_edge_evaluations'],f['advisor_calls'],f['advisor_failures']]
        page+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>'
    page+='</table><p>模型及数值墙钟耗时单独记录，不推进仿真；不据此宣称实时部署。单种子开发验证不构成泛化、统计或论文新颖性证明。</p>'
    for f in flights:
        name=f"{f['method']}_{f['scene']}_seed{f['seed']}"
        page+=f'<h3>{html.escape(name)}</h3><p><a href="runs/{name}/decision_log.txt">决策日志</a></p><img src="runs/{name}/summary.png">'
    page+='<h2>解释与限制</h2><p>代码算出的数值可复算，但公式、启发式、GP和信息支持仍可能不准确。程序可运行不意味着规划更好。需同时检查目标改进、真实飞行暴露、调用开销和数值对照。</p>'
    (out/'index.html').write_text(page);save_json(out/'status.json',{'status':'completed','audit_pass':all_pass})
    if not all_pass:raise AssertionError('audit failed')


def collect_audit(out,method,report,task,audits,calls):
    w=report['resources'];audits.append({'method':method,'task':task,'samples_within_cap':w['samples']<=SETTINGS.sample_limit,
        'edges_within_cap':w['edges']<=SETTINGS.edge_limit,'final_geometry_verified':report['final_geometry_verified'],
        'objective_nonincrease':report['initial_metrics'] is None or report['best_metrics']['objective']<=report['initial_metrics']['objective']+1e-7})
    for t in report['trace']:
        runtime=t.get('model_runtime',{});ctx=t.get('context',{})
        if not runtime.get('request_id'):continue
        a=audit_exchange(out/'public'/method,runtime,ctx);a.pop('missing_response_is_failure_not_model_success')
        a.update(method=method,task=task,no_future_points=all(p['time_seconds']<=ctx['time_seconds']+1e-8 for p in ctx['measurement_evidence']['recent_points']))
        if method=='operator_program' and runtime['status']=='accepted':
            a['program_resolves_to_executed_proposal']=runtime['program_execution']['resolved_proposal']==t['proposal']
            a['no_arbitrary_python']=not runtime['program_execution']['arbitrary_python_executed']
        audits.append(a)
        calls.append({'method':method,'task':task,'context':ctx,'raw_response':runtime.get('raw_agent_response'),
            'proposal':t['proposal'],'program_execution':runtime.get('program_execution'),'feedback':t['feedback'],
            'error':t.get('error'),'runtime':runtime})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','baseline','direct_values','operator_program','report'),required=True)
    a=p.parse_args();out=a.output.resolve()
    if a.phase=='prepare':prepare(out)
    elif a.phase=='baseline':baseline(out)
    elif a.phase=='report':reports(out)
    else:run_arm(out,a.phase)
