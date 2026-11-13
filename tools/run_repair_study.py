"""第一版区段修复试验：相同目标/资源上限的数值、单轮LLM及多轮反馈；CPU可运行。"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json, append_json
from uav_em.config import load_config
from uav_em.route_repair import FileRepairPolicy, RouteRepairOptimizer, RepairSettings
from uav_em.simulation import simulate
from uav_em.decision_log import render_decision_log
from uav_em.visualization import render
from render_navigation_replays import render_replays

CONTRACT='''# UAV fixed-objective route repair v1
You are a real online decision LLM. Read ONLY THIS public directory's contract.md, control.json,
requests/*.json and your responses/*.json. Never read parents, source, hidden scenarios, other
methods, truth maps, or other agents. Do not spawn agents. This is procedural allowlisting, not OS isolation.
Poll until control status completed/failed. No requests yet: wait at most 20 seconds per tool call.
UAV has known buildings and pose, and ONLY noisy power point measurements at its own past/current
positions. Local GP estimates are inferred, unknown/null is not safe. There are no bearings or
ranges/source locations/counts/future scripts. Stronger dBm is worse. Model std is not calibrated.
Along-track changes conflate space/time; prior decay is not evidence of source disappearance.
Task: use the SAME fixed objective and budgets to improve remaining route. This objective scores
CURRENT GP, not future true exposure. No new measurements or flight occur within feedback rounds.
Do not change GP, cost weights, thresholds, or constraints. No generated code, probes, source hypotheses.
route_repair contains the authoritative CURRENT incumbent, version, anchors, segments, numerical
feedback and resources; anchors may change after acceptance. Use CURRENT ids and version only.
Operators:
- keep: stop optimizing this decision, requires incumbent_feasible. Fields only route_version,operator,reason.
- shortcut: replace interval by straight chord, preserve outside. Add start_id,end_id, NO sampling/budget.
- local_rrt: search only a replacement between selected ordered anchor ids; preserve outside.
- via: replace interval by two RRT* searches through waypoint_m:[x,y], ordered ids required; preserves outside.
- global_rrt: search current position to goal; NO anchor ids/waypoint. Only option if no incumbent.
RRT operators require sample_budget integer within resources.samples_left and sampling object:
{"focus_probability":0.6,"regions":[{"low_m":[x,y],"high_m":[x,y],"weight":1.0}]}
focus_probability in [0,0.6]. Max3 positive-area in-bounds boxes, weights in (0,100].
Sampling mix = 10% target endpoint + focus_probability boxes + remainder global UNIFORM; global
share at least30%. Boxes are search preferences, not safe regions; edges still scored on fixed J.
Uniform sampling uses {"focus_probability":0.0,"regions":[]}.
The total budget is shared over ALL rounds, both via legs and recovery. Edge evaluations often
exhaust before samples. Consider modest searches (e.g.100-350 samples) and feedback resources;
via divides its requested sample budget between legs, not one full budget each. Initial construction
may need more samples. Single-round missions may spend the full remaining resources.
Numerical feedback reports feasibility, cost decomposition, change from previous best, accept/reject
and remaining work. Only a lower fixed J with geometry/energy feasibility is accepted. If a candidate
is rejected, the old feasible route stays. Unknown fields remain uncertain; do not promise safety.
Positive progress does not mean leaving a weakly-supported route unexamined; weigh local improvement,
shortcut, alternative focused search and scarce remaining resources using actual evidence.
Output exactly one response JSON, written atomically to the provided response_path:
{"request_id":"matching id","decision":{"route_version":0,"operator":"...","reason":"short evidence-based reason",...},
"strategy":{"mode":"baseline"},"memory":"mission memory <=1200 characters"}
Do not edit a response after publication. Save only allowed fields, no code fences. This does not
request private chain of thought, only action and brief evidence. Process all requests until completed.
'''


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    for part in ('scenarios','runs','reports','logs','replay','public/repair_single','public/repair_feedback'):
        (out/part).mkdir(parents=True)
    cfg=load_config(ROOT/'configs/local_power.json')
    buildings=(
      dict(id='north',footprint_m=[[300,420],[400,420],[400,510],[300,510]],height_m=44,height_source='synthetic'),
      dict(id='south',footprint_m=[[490,110],[570,110],[570,210],[490,210]],height_m=42,height_source='synthetic'),
      dict(id='low',footprint_m=[[150,260],[215,260],[215,335],[150,335]],height_m=12,height_source='synthetic'))
    fields=(dict(center_m=[360,345],eirp_dbm=2.,height_m=1.5,start_seconds=0.,end_seconds=39.,velocity_mps=[0.,-.6]),
            dict(center_m=[560,315],eirp_dbm=0.,height_m=1.5,start_seconds=25.,end_seconds=180.,velocity_mps=[.3,0.]))
    cfg=replace(cfg,seed=311,map=replace(cfg.map,buildings=buildings),electromagnetic=replace(cfg.electromagnetic,fields=fields),
                simulation=replace(cfg.simulation,max_seconds=140.,max_advisor_calls=30))
    cfg.validate();save_json(out/'scenarios/r01.json',cfg.to_dict())
    for method in ('repair_single','repair_feedback'):
        public=out/'public'/method
        for name in ('requests','responses'):(public/name).mkdir()
        (public/'contract.md').write_text(CONTRACT)
        save_json(public/'control.json',{'status':'prepared','planned_missions':1})
    sources={}
    for p in list((ROOT/'src/uav_em').glob('*.py'))+[Path(__file__).resolve(),ROOT/'tools/render_navigation_replays.py']:
        rel=p.relative_to(ROOT);dest=out/'reports/runtime_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);sources[str(rel)]=sha(p)
    save_json(out/'protocol.json',dict(study='route_repair_v1_integration_pilot_not_statistical_advantage',
       methods=['repair_numeric','repair_single','repair_feedback'],scene='r01',seed=311,
       scenario_sha256=sha(out/'scenarios/r01.json'),sources_sha256=sources,
       settings=dict(sample_limit=2100,edge_limit=10000,anchor_spacing_m=80.),
       rounds={'repair_numeric':3,'repair_single':1,'repair_feedback':3},
       model_call_limit=30,model_sessions='separate_fresh_agents_per_arm_no_cross_arm_reading',
       contract_sha256={m:sha(out/'public'/m/'contract.md') for m in ('repair_single','repair_feedback')},
       objective='fixed weights 1/2/0.4 exact nearest-grid integral; no LLM weight changes',
       comparison='same caps/operation set/point sensor/GP; actual usage may differ; numeric policy is a simple development control',
       limitations='single constructed dynamic scene, no population/generalization claim; no probe action in repair v1; compute walltime not simulated'))
    save_json(out/'status.json',{'status':'prepared'})


def run(out,method):
    protocol=read(out/'protocol.json')
    for name,expected in protocol['sources_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('frozen source changed: '+name)
    if sha(out/'scenarios/r01.json')!=protocol['scenario_sha256']:raise ValueError('scenario changed')
    cfg=load_config(out/'scenarios/r01.json');directory=out/'runs'/f'{method}_r01_seed{cfg.seed}';directory.mkdir(exist_ok=False)
    public=out/'public'/method if method!='repair_numeric' else None
    if public:
        if sha(public/'contract.md')!=protocol['contract_sha256'][method]:raise ValueError('contract changed')
        save_json(public/'control.json',{'status':'running','planned_missions':1})
    save_json(directory/'status.json',dict(status='running',method=method,scene='r01',seed=cfg.seed))
    policy=FileRepairPolicy(public,'mission_r01',900) if public else None
    optimizer=RouteRepairOptimizer(policy,RepairSettings(max_rounds=protocol['rounds'][method],**protocol['settings']))
    def hook(stage,data):
        if stage in ('decision_executed','finished'):
            append_json(directory/'execution.jsonl',{'stage':stage,**data})
        if stage=='decision_executed':
            r=data['decision']['repair_optimization']
            print(method,'t',data['time_seconds'],'accepted',r['accepted_proposals'],'resources',r['resources'],flush=True)
    try:
        ep=simulate(cfg,advisor_name='codex_repair_subagent' if policy else 'numeric_repair',debug_hook=hook,route_optimizer=optimizer)
        save_json(directory/'episode.json',ep);save_json(directory/'metrics.json',ep['metrics'])
        (directory/'decision_log.txt').write_text(render_decision_log(ep))
        save_json(directory/'status.json',dict(status='completed',method=method,scene='r01',seed=cfg.seed,**ep['metrics']))
        if public:save_json(public/'control.json',dict(status='completed',completed_missions=1))
        print(json.dumps(ep['metrics'],indent=2),flush=True)
    except BaseException as exc:
        save_json(directory/'status.json',dict(status='failed',method=method,error=repr(exc),traceback=traceback.format_exc()))
        if public:save_json(public/'control.json',dict(status='failed',error=repr(exc)))
        raise


def audit_exchange(public,runtime,ctx):
    """Timeouts and rejected responses remain reportable outcomes, not missing-file errors."""
    identifier=runtime['request_id'];req=public/'requests'/f'{identifier}.json';res=public/'responses'/f'{identifier}.json'
    available=res.is_file();recorded='raw_agent_response' in runtime
    return {'request_id':identifier,'input_sha':req.is_file() and sha(req)==runtime.get('request_sha256'),
        'context_matches':req.is_file() and read(req)['visible_context']==ctx,
        'response_presence_consistent':available==recorded,
        'response_record_consistent':(available and sha(res)==runtime.get('response_sha256')
            and res.read_text()==runtime.get('raw_agent_response')) if available else runtime.get('status')=='error',
        'model_runtime_status':runtime.get('status'),
        'missing_response_is_failure_not_model_success':not available}


def summarize(out):
    protocol=read(out/'protocol.json');rows=[];checks=[];calls=[]
    for method in protocol['methods']:
        directory=out/'runs'/f'{method}_r01_seed311';status=read(directory/'status.json')
        if status['status']!='completed':raise ValueError('method incomplete '+method)
        ep=read(directory/'episode.json');rows.append({'method':method,**ep['metrics']});render(ep,directory/'summary.png')
        for d in ep['decisions']:
            r=d['repair_optimization'];trace=r['trace'];prior=r['initial_metrics']
            checks.append({'method':method,'time_seconds':d['time_seconds'],
                           'sample_cap':r['resources']['samples']<=2100,'edge_cap':r['resources']['edges']<=10000,
                           'geometry_verified':r['final_geometry_verified'],
                           'score_nonincrease':prior is None or r['best_metrics']['objective']<=prior['objective']+1e-7})
            for t in trace:
                if t.get('backend')=='numeric_recovery':
                    calls.append({'method':method,'time_seconds':d['time_seconds'],'round':None,
                                  'backend':'numeric_recovery','proposal':t['proposal'],'feedback':t['feedback']})
                    continue
                feedback=t['feedback'];runtime=t.get('model_runtime') or {};ctx=t['context']
                if runtime.get('request_id'):
                    exchange=audit_exchange(out/'public'/method,runtime,ctx)
                    # This informational boolean is not an assertion of model success.
                    exchange.pop('missing_response_is_failure_not_model_success')
                    checks.append({'method':method,**exchange,
                       'no_future_points':all(p['time_seconds']<=d['time_seconds']+1e-8 for p in ctx['measurement_evidence']['recent_points']),
                       'same_snapshot':ctx['route_repair']['snapshot_time_seconds']==d['time_seconds'],
                       'fixed_objective':ctx['route_repair']['fixed_objective']['interference_weight']==2.})
                calls.append({'method':method,'time_seconds':d['time_seconds'],'round':t['round'],'proposal':t.get('proposal'),
                              'feedback':feedback,'error':t.get('error'),'request_id':runtime.get('request_id')})
    save_json(out/'results.json',rows);save_json(out/'reports/repair_trace.json',calls)
    save_json(out/'reports/audit.json',{'checks':checks,'all_pass':all(all(v for v in c.values() if isinstance(v,bool)) for c in checks),
       'source_frozen_matches':all(sha(out/'reports/runtime_source'/p)==h for p,h in protocol['sources_sha256'].items()),
       'filesystem_boundary':'procedural_allowlist_not_os_sandbox',
       'current_matches_frozen_sources':all((ROOT/p).exists() and sha(ROOT/p)==h for p,h in protocol['sources_sha256'].items()),
       'frozen_planner_seconds_note':'initial frozen pilot only counts RRT time; full repair scoring/verification time was not preserved. Current code fixes the accounting; historical values not rewritten.'})
    render_replays(out,frame_stride=4,playback_speed=2.,decision_pause_seconds=.7)
    text='<!doctype html><meta charset="utf-8"><title>第一版区段修复</title><style>body{max-width:1250px;margin:30px auto;font:16px/1.8 sans-serif}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}img{max-width:100%}</style><h1>固定目标/总预算下的LLM区段修复与RRT*采样分配</h1><p><a href="../index.html">总入口</a> · <a href="replay/index.html">集中GIF</a> · <a href="reports/repair_trace.json">提议与反馈</a> · <a href="reports/audit.json">预算/因果/执行审计</a></p><p>单个预先固定动态场景的第一版集成验证，不是泛化或统计优势证明。三方法使用同一操作集合、固定目标和每决策总资源上限；实际用量、轮数、航迹和观测不同。数值方法只是简单开发对照。</p><table><tr><th>方法</th><th>到达</th><th>航程/m</th><th>暴露/s</th><th>积分/dB·s</th><th>调用</th><th>模型错误</th><th>数值恢复</th><th>边评价</th></tr>'
    report=['# 第一版区段修复集成验证','','单构造场景；不能据此宣称论文创新成立或某模型/策略全面更优。','',
       '| 方法 | 到达 | 航程/m | 暴露/s | 积分/dB·s | LLM调用 | 错误 | 数值恢复 | 边评价 |',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values=[r['method'],r['success'],round(r['executed_length_m'],2),round(r['high_interference_exposure_seconds'],3),round(r['interference_excess_integral_db_seconds'],3),r['advisor_calls'],r['advisor_failures'],r['repair_numeric_recoveries'],r['repair_edge_evaluations']]
        text+='<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in values)+'</tr>';report.append('| '+' | '.join(str(v) for v in values)+' |')
    text+='</table><p>参考功率是软代价，不是设备失效线；积分不含决策悬停。模型墙钟耗时不推进仿真。较低当前GP目标不保证较低真实未来暴露；两类指标分开评价。</p>'
    for r in rows:
        method=r['method'];text+=f'<h2>{method}</h2><p><a href="runs/{method}_r01_seed311/decision_log.txt">逐轮日志</a></p><img src="runs/{method}_r01_seed311/summary.png">'
    by_method={r['method']:r for r in rows}
    control=by_method['repair_numeric'];feedback=by_method['repair_feedback']
    same=(abs(control['executed_length_m']-feedback['executed_length_m'])<1e-6
          and abs(control['interference_excess_integral_db_seconds']-feedback['interference_excess_integral_db_seconds'])<1e-6)
    interpretation=('本例多轮LLM与简单数值路线长度/积分相同，未验证干扰规避提升。'
        if same else '本例方法指标存在差异；只有一个场景，尚不能归因成稳定方法优势。')
    interpretation+=' 数值策略反复全局搜索，当前节省不能代替强数值自适应/提前停止对照，也须另计模型开销。'
    report+=['',interpretation]
    text+='<p><strong>'+html.escape(interpretation)+'</strong></p>'
    if (out/'reports/reproduction.json').exists():
        replay=read(out/'reports/reproduction.json')
        text+='<p><a href="reports/reproduction.json">当前源码的记录提议重放</a>全部一致：'+str(replay['all_match'])+'；没有新模型调用。</p>'
        report+=['','计时/报告修正后记录提议重放全部一致：'+str(replay['all_match'])+'，不属于独立LLM重复试验。']
    final_tests=out/'logs/tests_final.log'
    if final_tests.exists():
        import re
        counts=re.findall(r'(\d+ passed[^\n]*)',final_tests.read_text())
        if counts:
            text+='<p>当前测试：'+html.escape(counts[-1])+'。</p>'
            report+=['','当前测试：'+counts[-1]+'。']
    text+='<p>冻结版planner_seconds仅计RRT搜索，遗漏修复评分/终检；不据此比较总规划时延。当前源码已修正并通过记录动作重放核验后另列，历史计时不补造。冻结源码与后续修正版本分别记录。</p>'
    (out/'index.html').write_text(text)
    report+=['','固定目标J=长度+2×超参考估计惩罚+0.4×不确定性；精确最近邻栅格线积分。每决策最多2100次样本、10000次未缓存边评价，终检也计入。单轮/多轮共享同一上限；报告实际用量。',
        'GP固定；LLM选择区间、操作、采样区域和预算。只接受可行且J下降的候选，不接受通过改变权重制造的收益。旧路保留不等于新提议有效，数值恢复单列。',
        '当前修复模式没有主动补测行动，原基础模式仍保留；树每次新建，只复用路线区间，不冒充RRTX/持久树修复。']
    report+=['','冻结运行后的审查发现planner_seconds仅计RRT、遗漏评分/终检。当前代码已修正，但本轮历史缺失耗时不补造，不据此比较完整规划耗时。accepted_proposals为全部被接受候选（可能含恢复），须结合backend归因；完整逐轮轨迹保留恢复。']
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n');save_json(out/'status.json',{'status':'completed','tasks':3})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','repair_numeric','repair_single','repair_feedback','report'),required=True)
    a=p.parse_args();out=a.output.resolve()
    if a.phase=='prepare':prepare(out)
    elif a.phase=='report':summarize(out)
    else:run(out,a.phase)
