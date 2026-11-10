"""Summarize saved causal point-agent runs and render GIFs; never rerun decisions."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json, visible_context
from uav_em.config import load_config
from uav_em.environment import ElectromagneticWorld
from uav_em.visualization import render
from render_navigation_replays import render_replays

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())

def summarize(out, gifs=True):
    if load(out/'status.json')['status']!='completed':raise ValueError('study is not completed')
    rows=load(out/'results.json'); protocol=load(out/'protocol.json')
    checks=[]; timelines={}; sources={}; endpoints=[]
    for name,sha in protocol['sources_sha256'].items():
        frozen=out/'reports/runtime_source'/name
        sources[name]={'frozen_matches':frozen.exists() and digest(frozen)==sha,
                       'current_matches':(ROOT/name).exists() and digest(ROOT/name)==sha}
    for row in rows:
        run=out/'runs'/f"{row['method']}_{row['scene']}_seed{row['seed']}";ep=load(run/'episode.json')
        render(ep,run/'summary.png')
        cfg=load_config(out/'scenarios'/f"{row['scene']}.json")
        end_power=float(ElectromagneticWorld(cfg).truth([cfg.map.goal_m],row['simulation_seconds'])[0])
        endpoints.append({'scene':row['scene'],'method':row['method'],'arrival_seconds':row['simulation_seconds'],
                          'goal_truth_dbm_at_arrival':end_power,
                          'above_reference':end_power>cfg.electromagnetic.interference_reference_dbm})
        entry={'scene':row['scene'],'method':row['method'],'episode_sha256':digest(run/'episode.json'),
               'zero_initial_points':ep['metrics'].get('initial_observations',0)==0,
               'point_only':ep['metrics']['directional_scans']==0,
               'raw_and_actual_execution_separated':True,'decisions':[]}
        lines=[]
        for d in ep['decisions']:
            ctx=d['context'];t=d['time_seconds'];runtime=d.get('model_runtime') or {}
            base={'time_seconds':t,'position_m':ctx['vehicle']['position_m'],
                  'point_dbm':ctx['latest_observation']['interference_dbm'],
                  'proposed':d['proposed_decision'],'accepted_action':d['accepted_action'],
                  'error':d['error'],'planning_failure':d['planning_failure']}
            lines.append(base)
            supported=ctx['electromagnetic_estimate']
            causal=all(p['time_seconds']<=t+1e-6 for p in ctx['measurement_evidence']['recent_points'])
            if row['method']=='online_agent':
                identifier=runtime['request_id'];req=out/'public/requests'/f'{identifier}.json';res=out/'public/responses'/f'{identifier}.json'
                request=load(req);response=load(res)
                check={'request_id':identifier,'request_sha_matches':digest(req)==runtime['request_sha256'],
                       'response_sha_matches':digest(res)==runtime['response_sha256'],
                       'raw_response_preserved':res.read_text()==runtime['raw_agent_response'],
                       'public_context_matches_recorded':request['visible_context'] in (ctx, visible_context(ctx)),
                       'new_measurement_evidence_present':'measurement_evidence' in request['visible_context'],
                       'GP_local_samples_present':'local_samples' in request['visible_context']['electromagnetic_estimate'],
                       'only_baseline_strategy':response['strategy']=={'mode':'baseline'},
                       'no_hidden_source_hypothesis':ctx.get('source_inference',{}).get('hypothesis') is None,
                       'no_future_points':causal,'parse_accepted':runtime.get('status')=='accepted',
                       'no_error_or_search_failure':d['error'] is None and d['planning_failure'] is None}
                entry['decisions'].append(check)
            else: entry['decisions'].append({'no_future_points':causal})
            assert supported['measurement_is_at_vehicle_only']
            assert not supported['field_at_arrival_times_predicted']
        checks.append(entry);timelines[f"{row['method']}_{row['scene']}"]=lines
    audit={'kind':'file_identity_and_causal_interface_checks_not_OS_access_proof',
           'filesystem_boundary':'procedural_allowlist_not_os_sandbox',
           'online_backend':'fresh_codex_subagent_not_local_Qwen',
           'model_internal_chat_not_exported':True,'actual_contract_requests_responses_preserved':True,
           'frozen_runtime_source':sources,'runs':checks,
           'all_frozen_sources_match':all(v['frozen_matches'] for v in sources.values()),
           'source_changed_during_cleanup':any(not v['current_matches'] for v in sources.values()),
           'no_counterfactual_oracle_information_given_to_agent':True}
    save_json(out/'reports/execution_audit.json',audit)
    save_json(out/'reports/endpoint_evaluation.json',{'retrospective_only_not_given_to_agent':True,'runs':endpoints})
    save_json(out/'reports/decision_timeline.json',timelines)
    if gifs:render_replays(out,frame_stride=4,playback_speed=2.,decision_pause_seconds=.7)
    report=['# 点测强LLM基础绕行验证','',
      '两个预先固定的构造机制场景，每个场景一个噪声/搜索种子；不是随机总体测试，不证明泛化或论文创新成立。',
      '无预飞行稀疏数据、无测向、无源位置或未来脚本。模型为新的在线Codex决策子agent，不是本地4B。',
      '','| 场景 | 方法 | 到达 | 航程/m | 暴露/s | 超参考积分/dB·s | LLM调用 | 失败/回退 | 搜索次数 |',
      '|---|---|---|---:|---:|---:|---:|---|---:|']
    for r in rows:
        report.append(f"| {r['scene']} | {r['method']} | {r['success']} | {r['executed_length_m']:.2f} | {r['high_interference_exposure_seconds']:.3f} | {r['interference_excess_integral_db_seconds']:.3f} | {r['advisor_calls']} | {r['advisor_failures']}/{r['fallbacks']} | {r['planning_calls']} |")
    report+=['','p01为静态源机制，p02为移动/开关源机制；这些机制说明只供事后分析，不是agent输入。',
      'LLM通过点测趋势、主动侧向补测、单中间点重规划和跨复核保留路线实现避让。点测无法消除方向歧义，试探期间仍会暴露；不宣称全程绕开所有强干扰。',
      '','基线和在线策略共享环境、点测接口、噪声设置和GP估计器，实际轨迹不同导致收到的读数不同，但在线可选择profile、补测和waypoint；一个waypoint执行两次标准RRT*。路线、目标权重与总搜索开销不同，不能归因成同目标同预算的推理优化优势。',
      '真值仅用于当前读数和事后飞行积分。−80dBm为软参考；dB·s不是物理剂量。积分不含决策悬停。模型/求解实际耗时未推进仿真，不声称实时性。',
      '','运行后按用户授权精简了代码，当前源码不与冻结源码冒充一致。reports/runtime_source为本轮必要运行证据；execution_audit明确两套哈希。可用reproduce_point_agent_study.py核验当前源码的记录动作重放；这不是新模型实验。',
      '','下一步实现证据条件化区段修复与固定目标/约束/总预算的数值反馈优化，再做同信息、同操作集、同资源预算的强数值和单轮/多轮LLM对照。']
    note='动态场景积分降低但暴露时长增加，不能称为全面避开强干扰。事后核查显示p02终点功率约−79.37dBm，本身高于−80dBm软参考；到达必有末段暴露，此真值从未给到agent。该事实不能解释全部探索/绕行开销，仍须改进策略。'
    report+=['',note]
    validation=''
    if (out/'reports/reproduction.json').exists():
        match=load(out/'reports/reproduction.json')['all_match']
        validation=f'当前源码的记录动作重放全部一致：{match}；未重新调用模型，详见reports/reproduction.json。'
    test_log=out/'logs/tests_current.log'
    if test_log.exists():
        import re
        passed=re.findall(r'(\d+ passed[^\n]*)',test_log.read_text())
        if passed:
            validation+=' 当前范围测试：'+passed[-1]+'。旧功能测试随能力删除，不冒充旧缺失夹具已修复。'
    if validation:report+=['',validation]
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>点测在线LLM绕行验证</title>
<style>body{font:16px/1.7 sans-serif;max-width:1300px;margin:25px auto;padding:0 16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}img{max-width:100%}pre{white-space:pre-wrap;background:#f7f8fa;padding:14px}a{color:#16599a}</style>
<h1>局部点测下的在线LLM绕行与重规划</h1><p><a href="../index.html">总入口</a> · <a href="replay/index.html">集中GIF回放</a> · <a href="RESULTS.md">结果与边界</a> · <a href="reports/execution_audit.json">执行/版本审计</a> · <a href="reports/cleanup_manifest.json">清理清单</a></p>
<p>仅使用机载当前位置带噪功率、局部时空GP与已知建筑。左侧真值仅研究者可见，右侧是策略估计；灰色未知，不代表安全。无测向、无初始电磁地图、无真实源信息。两个机制场景的基础可行性验证，尚不证明同预算优化优势。</p>
<table><tr><th>场景</th><th>方法</th><th>到达</th><th>航程/m</th><th>暴露/s</th><th>超参考积分/dB·s</th><th>LLM调用</th><th>失败/回退</th><th>搜索次数</th></tr>'''
    for r in rows:
        label='数值RRT*' if r['method']=='numeric_rrt' else '在线agent + RRT*'
        page+=f"<tr><td>{r['scene']}</td><td>{label}</td><td>{r['success']}</td><td>{r['executed_length_m']:.2f}</td><td>{r['high_interference_exposure_seconds']:.3f}</td><td>{r['interference_excess_integral_db_seconds']:.3f}</td><td>{r['advisor_calls']}</td><td>{r['advisor_failures']}/{r['fallbacks']}</td><td>{r['planning_calls']}</td></tr>"
    page+='</table><p>航程与总搜索量不同，在线agent也可选择不同profile；本轮用于确认基本能力，不是公平优化优越性试验。</p>'
    for scene in protocol['scenes']:
        page+=f"<h2>{scene['id']}</h2>"
        for method in protocol['methods']:
            identity=f"{method}_{scene['id']}_seed{scene['seed']}"
            page+=f'<p>{identity} · <a href="runs/{identity}/decision_log.txt">完整可读日志</a> · <a href="runs/{identity}/episode.json">原始episode（含评价真值）</a></p><img src="runs/{identity}/summary.png">'
        page+='<details><summary>在线agent的实际决策时间线</summary><table><tr><th>仿真秒</th><th>当前位置点测/dBm</th><th>实际行动</th><th>模型依据（原文）</th></tr>'
        for item in timelines['online_agent_'+scene['id']]:
            page+=f"<tr><td>{item['time_seconds']:.2f}</td><td>{item['point_dbm']:.2f}</td><td>{html.escape(item['accepted_action'])}</td><td>{html.escape(item['proposed']['reason'])}</td></tr>"
        page+='</table></details>'
    page+='<p>'+html.escape(note)+'</p><p>'+html.escape(validation)+'</p>'
    page+='<h2>方法定位</h2><p>当前演示展示点测→估计→LLM探索/绕行→数值复核→新点测闭环。下一阶段研究证据条件化的局部航路修复与固定评价目标下的数值反馈推理优化；尚未实现该完整算法。</p><p>审计为文件身份与接口核验，不是OS访问审计。实际响应和接受/执行分开保留；没有把传统数值恢复计作模型成功。清理后源码的动作重放见reports/reproduction.json。</p></html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--no-gif',action='store_true')
    a=p.parse_args();summarize(a.output.resolve(),not a.no_gif)
