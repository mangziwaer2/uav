"""新开发城市的点测任务/目标一致性诊断；备选评分只排序，不接入控制。"""
from __future__ import annotations
import argparse
from dataclasses import replace,asdict
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json
from uav_em.config import load_config
from uav_em.scenarios import generate_scenario,ScenarioSettings
from uav_em.repair_process import ProcessRepairOptimizer
from uav_em.repair_diagnostics import geometric_candidates
from uav_em.task_readiness import SCORE_NAMES,score_route,choose_from_pool,public_risk
from run_repair_diagnostics import load_case

DEFAULT=ROOT/'outputs/interference_task_readiness_20261111'
DEV_SEEDS=(611,613,617,619)
GENERATOR_SETTINGS=ScenarioSettings(columns=41,rows=31,resolution_m=20,building_count=6,
    min_sources=2,max_sources=3,horizon_seconds=120)

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def native_config(config):
    """现有生成器起终点可能含numpy整数；只转换存储类型，不改变数值。"""
    def scalar(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(f'unsupported configuration value: {type(value).__name__}')
    return json.loads(json.dumps(config.to_dict(),default=scalar,allow_nan=False))

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    for directory in ('scenarios','cases','candidates','evaluations','runs','reports/runtime_source','logs','replay','figures'):
        (out/directory).mkdir(parents=True)
    shutil.copy2(ROOT/'docs/task_readiness.md',out/'reports/frozen_design.md')
    shutil.copy2(ROOT/'configs/local_power.json',out/'reports/base_config.json')
    paths=[*(ROOT/'src/uav_em').glob('*.py'),Path(__file__).resolve(),ROOT/'tools/run_repair_diagnostics.py',
           ROOT/'tools/render_navigation_replays.py',ROOT/'tests/test_task_readiness.py']
    sources={}
    for path in paths:
        relative=path.relative_to(ROOT);target=out/'reports/runtime_source'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);sources[str(relative)]=sha(path)
    save_json(out/'protocol.json',{'study':'prospective_development_task_readiness_and_score_alignment',
        'dev_seeds':list(DEV_SEEDS),'generator_settings':asdict(GENERATOR_SETTINGS),
        'generator_rule':'existing generate_scenario; no retuning/source relocation/outcome screening',
        'config_merge':'main power_only/local_gp aircraft+sensor+belief+propagation constants; generated buildings and hidden sources',
        'validation_seeds_unread':[701,709],'test_seeds_unread':[1601,1607,1609,1613,1619,1621],
        'controller':'unchanged library_numeric in finite process v1; no score alternatives used for flight',
        'capture_rule':'first feasible incumbent decision at/after10,20,30s plus first such decision with supported remaining-route interference_cost>0; every missing label explicit',
        'candidate_pool':'fixed geometric pool from existing diagnostics; geometry/energy checks; no future input',
        'selection':{'scores':list(SCORE_NAMES),'length_ratio_limit':1.25,'ties_keep_incumbent':True,
            'weights':'same1/2/0.4; alternative interference only; uncertainty term retained',
            'expected_score':'exact Gaussian E[clip((X-reference)/10,0,4)], uncalibrated assumption',
            'envelope_score':'clip((mean+std-reference)/10,0,4);1std fixed before outcomes,not confidence guarantee'},
        'readiness_gate':{'min_scenes':2,'observable_risk':'supported current remaining-route penalty>0',
            'current_j_relative_improvement':.01,'future_excess_reduction_fraction':.10,
            'length_ratio_limit':1.25,'kind':'development opportunity gate_not_model_superiority'},
        'truth_boundary':'freeze all scenarios/cases/candidates/score_choices then evaluate future truth; no return to controller',
        'scope':'0model calls,no new sensor/no preflight sparse map/no GP or controller objective change',
        'source_sha256':sources,'frozen_design_sha256':sha(out/'reports/frozen_design.md'),
        'base_config_sha256':sha(out/'reports/base_config.json')})
    save_json(out/'status.json',{'status':'prepared'})

def check(out):
    protocol=read(out/'protocol.json')
    for relative,h in protocol['source_sha256'].items():
        if sha(ROOT/relative)!=h or sha(out/'reports/runtime_source'/relative)!=h:raise ValueError('source changed: '+relative)
    if sha(out/'reports/frozen_design.md')!=protocol['frozen_design_sha256']:raise ValueError('frozen design changed')
    if sha(out/'reports/base_config.json')!=protocol['base_config_sha256']:raise ValueError('base config changed')
    return protocol

def scenarios(out):
    protocol=check(out)
    if any((out/'scenarios').iterdir()):raise FileExistsError('scenarios already present')
    base=load_config(out/'reports/base_config.json');generation=[]
    # No validation/test scenes are generated, including their truth or geometry.
    for seed in protocol['dev_seeds']:
        generated,report=generate_scenario(seed,GENERATOR_SETTINGS)
        cfg=replace(base,seed=seed,map=generated.map,
            electromagnetic=replace(base.electromagnetic,fields=generated.electromagnetic.fields))
        cfg.validate();save_json(out/'scenarios'/f'dev{seed}.json',native_config(cfg))
        generation.append({'seed':seed,'generator_report':report,'merge_note':protocol['config_merge']})
        print('generated',seed,'attempt',report['attempt'],flush=True)
    save_json(out/'reports/generation.json',generation)
    save_json(out/'scenario_manifest.json',{p.name:sha(p) for p in sorted((out/'scenarios').glob('*.json'))})

class CaptureOptimizer(ProcessRepairOptimizer):
    def __init__(self,out,scene):
        super().__init__('library_numeric');self.out,self.scene,self.labels=out,scene,{}
        self.decisions=0

    def optimize(self,snapshot,context,route,max_length_m,seed,call_limit):
        plan,report=super().optimize(snapshot,context,route,max_length_m,seed,call_limit)
        labels=[];initial=report['initial_metrics'];self.decisions+=1
        if initial is not None:
            for threshold in (10,20,30):
                label=f'after{threshold}'
                if label not in self.labels and snapshot.time_seconds>=threshold:labels.append(label)
            if 'first_supported_risk' not in self.labels and initial['interference_cost']>1e-7:
                labels.append('first_supported_risk')
        if labels:
            name=f'{self.scene}_decision{self.decisions:02d}';base=self.out/'cases'/name
            fields=('estimate_dbm','uncertainty_db','supported','age_seconds')
            np.savez_compressed(base.with_suffix('.npz'),**{k:getattr(snapshot,k) for k in fields})
            save_json(base.with_suffix('.json'),{'case':name,'scene':self.scene,'time_seconds':snapshot.time_seconds,
                'context':context,'route':route,'max_length_m':max_length_m,'seed':seed,
                'observation_count':snapshot.observation_count,'estimation_details':snapshot.estimation_details,
                'selection_labels':labels,'public_risk':public_risk(context,initial),
                'captured_before_this_decision_actions':True})
            for label in labels:self.labels[label]=name
        return plan,report

def collect(out):
    protocol=check(out)
    if any((out/'runs').iterdir()) or any((out/'cases').iterdir()):raise FileExistsError('collection already present')
    from uav_em.simulation import simulate
    from uav_em.decision_log import render_decision_log
    manifest=read(out/'scenario_manifest.json');selection=[]
    for filename,h in manifest.items():
        if sha(out/'scenarios'/filename)!=h:raise ValueError('scenario changed')
        scene=Path(filename).stem;cfg=load_config(out/'scenarios'/filename);optimizer=CaptureOptimizer(out,scene)
        episode=simulate(cfg,advisor_name='library_numeric_no_model',route_optimizer=optimizer)
        episode['assumptions']['route_optimizer']='finite_numeric_process_v1'
        run=out/'runs'/f'library_numeric_{scene}_seed{cfg.seed}';run.mkdir()
        save_json(run/'episode.json',episode);save_json(run/'metrics.json',episode['metrics'])
        (run/'decision_log.txt').write_text(render_decision_log(episode))
        save_json(run/'status.json',{'status':'completed','method':'library_numeric','scene':scene,'seed':cfg.seed,**episode['metrics']})
        labels=('after10','after20','after30','first_supported_risk')
        selection.append({'scene':scene,'selected':optimizer.labels,
                          'missing_labels':[label for label in labels if label not in optimizer.labels]})
        print('collected',scene,'success',episode['metrics']['success'],'labels',optimizer.labels,flush=True)
    save_json(out/'reports/selection.json',selection)
    save_json(out/'case_manifest.json',{p.name:sha(p) for p in sorted((out/'cases').glob('*'))})

def propose(out):
    check(out)
    if any((out/'candidates').iterdir()):raise FileExistsError('candidate pool exists')
    for path in sorted((out/'cases').glob('*.json')):
        case,snapshot,cfg=load_case(out,path)
        records,context,work=geometric_candidates(snapshot,case['context'],case['route'],case['max_length_m'])
        for r in records:
            if r['feasible']:r['scores']=score_route(snapshot,r['path'],cfg.electromagnetic.interference_reference_dbm)
        selection=choose_from_pool(records)
        save_json(out/'candidates'/path.name,{'case':case['case'],'scene':case['scene'],
            'records':records,'selection':selection,'public_context':context,'resources':work})
        print('proposed',case['case'],'candidates',len(records),'choices',selection['choices'],flush=True)
    frozen={}
    for folder in ('scenarios','cases','candidates','runs'):
        for path in sorted((out/folder).rglob('*')):
            if path.is_file():frozen[str(path.relative_to(out))]=sha(path)
    save_json(out/'causal_manifest.json',{'sha256':frozen,'all_candidates_and_choices_frozen_before_evaluation':True})
    save_json(out/'status.json',{'status':'all_causal_outputs_frozen'})

def verify_causal(out):
    check(out);manifest=read(out/'causal_manifest.json')
    for relative,h in manifest['sha256'].items():
        if sha(out/relative)!=h:raise ValueError('frozen causal result changed: '+relative)
    return manifest

def evaluate(out):
    verify_causal(out)
    if any((out/'evaluations').iterdir()):raise FileExistsError('evaluation exists')
    from uav_em.environment import ElectromagneticWorld
    from uav_em.diagnostic_evaluation import evaluate_fixed_route
    from uav_em.repair_search import FixedObjective,WorkBudget
    worlds={}
    for path in sorted((out/'candidates').glob('*.json')):
        case,snapshot,cfg=load_case(out,out/'cases'/path.name);data=read(path)
        world=worlds.setdefault(case['scene'],ElectromagneticWorld(cfg))
        problem=FixedObjective(snapshot,cfg.electromagnetic.interference_reference_dbm,WorkBudget(2100,100000))
        for r in data['records']:
            if not r['feasible']:continue
            actual=problem.route_metrics(r['path'])
            if actual is None or any(abs(actual[k]-r['metrics'][k])>1e-7 for k in actual):raise ValueError('candidate metric mismatch')
            r['evaluation']=evaluate_fixed_route(world,snapshot,r['path'])
        eligible=[r for r in data['records'] if r['id'] in data['selection']['eligible_ids']]
        base=next(r for r in eligible if r['id']=='incumbent')
        best=min(eligible,key=lambda r:(r['evaluation']['future_excess_db_seconds'],r['id']))
        # This oracle result remains inside evaluator, never used to choose capture/candidates/scores.
        qualifying=[r['id'] for r in eligible if r['metrics']['objective']<=base['metrics']['objective']*.99
            and base['evaluation']['future_excess_db_seconds']>1e-8
            and r['evaluation']['future_excess_db_seconds']<=base['evaluation']['future_excess_db_seconds']*.9]
        readiness=bool(case['public_risk']['supported_remaining_route_penalty_positive'] and qualifying)
        save_json(out/'evaluations'/path.name,{'case':case['case'],'scene':case['scene'],'time_seconds':case['time_seconds'],
            'labels':case['selection_labels'],'public_risk':case['public_risk'],'records':data['records'],
            'selection':data['selection'],'posthoc_low_exposure_reference_id':best['id'],
            'current_j_and_true_exposure_improvement_ids':qualifying,'ready_case_under_predeclared_gate':readiness,
            'candidate_source_sha256':sha(path),'score_choices_recomputed_identically':choose_from_pool(data['records'])==data['selection']})
        print('evaluated',case['case'],'ready',readiness,'paired improvements',len(qualifying),flush=True)
    verify_causal(out);save_json(out/'status.json',{'status':'evaluated'})

def report(out):
    protocol=check(out);verify_causal(out)
    from render_navigation_replays import render_replays
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    data=[read(p) for p in sorted((out/'evaluations').glob('*.json'))]
    expected_cases=list((out/'cases').glob('*.json'))
    if len(data)!=len(expected_cases) or not data:raise ValueError('incomplete evaluations')
    rows=[];audits=[]
    for d in data:
        by={r['id']:r for r in d['records']};base=by['incumbent'];case=d['case']
        for role,identifier in [('incumbent','incumbent'),*d['selection']['choices'].items(),
                                ('posthoc_reference',d['posthoc_low_exposure_reference_id'])]:
            r=by[identifier];e=r['evaluation'];rows.append({'case':case,'scene':d['scene'],'role':role,'candidate_id':identifier,
                'current_j':r['metrics']['objective'],'length_m':r['metrics']['length_m'],
                'future_excess_db_seconds':e['future_excess_db_seconds'],
                'frozen_true_excess_db_seconds':e['frozen_true_field_excess_db_seconds'],
                'supported_fraction':e['supported_fraction_at_decision'],
                'base_future_excess_db_seconds':base['evaluation']['future_excess_db_seconds']})
        recomputed=choose_from_pool(d['records']);assert recomputed==d['selection']
        audits.append({'case':case,'score_choices_match_frozen_pretruth_selection':True,
            'candidate_source_hash_match':sha(out/'candidates'/f'{case}.json')==d['candidate_source_sha256'],
            'feasible_routes_independently_rechecked':sum(r['feasible'] for r in d['records'])})
    scene_summary=[];sections=[]
    for seed in protocol['dev_seeds']:
        scene=f'dev{seed}';group=[d for d in data if d['scene']==scene]
        run=out/'runs'/f'library_numeric_{scene}_seed{seed}';ep=read(run/'episode.json');metrics=ep['metrics']
        if metrics['advisor_calls']!=0 or any(d['repair_optimization']['policy_errors'] for d in ep['decisions']):
            raise ValueError('controller errors/model calls')
        ready=any(d['ready_case_under_predeclared_gate'] for d in group)
        scene_summary.append({'scene':scene,'metrics':metrics,'case_count':len(group),'ready_scene':ready,
            'observable_route_risk_cases':sum(d['public_risk']['supported_remaining_route_penalty_positive'] for d in group)})
        if not group:
            sections.append(f'<h2>{scene}</h2><p>未获得可行剩余路线快照；到达：{metrics["success"]}。保留失败，不补造状态。</p>')
            continue
        # One illustration per scene selected by PUBLIC label, never future performance.
        choices=[d for d in group if 'first_supported_risk' in d['labels']]
        chosen=choices[0] if choices else next((d for d in group if 'after20' in d['labels']),group[0])
        _,snapshot,cfg=load_case(out,out/'cases'/f"{chosen['case']}.json")
        by={r['id']:r for r in chosen['records']}
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
        valid=[r for r in chosen['records'] if r['id'] in chosen['selection']['eligible_ids']]
        axes[0].scatter([r['scores']['current_j'] for r in valid],[r['evaluation']['future_excess_db_seconds'] for r in valid],c='#aaaaaa',label='fixed pool')
        plotted=[('incumbent','incumbent','#222222'),('current_j',chosen['selection']['choices']['current_j'],'#1976c9'),
            ('gaussian_expected_j',chosen['selection']['choices']['gaussian_expected_j'],'#31934a'),
            ('one_std_envelope_j',chosen['selection']['choices']['one_std_envelope_j'],'#9962c4'),
            ('posthoc_reference',chosen['posthoc_low_exposure_reference_id'],'#d55b15')]
        for index,(label,identifier,color) in enumerate(plotted):
            r=by[identifier];axes[0].scatter(r['scores']['current_j'],r['evaluation']['future_excess_db_seconds'],
                s=130,facecolors='none',edgecolors=color,linewidths=2,label=label)
            route=np.array(r['path']);axes[1].plot(*route.T,color=color,label=label,lw=2,ls=('-','--',':','-.',':')[index])
        for building in snapshot.geometry.buildings:
            axes[1].add_patch(Polygon(building.footprint_m,color='#444444' if building in snapshot.geometry.blocking_buildings else '#aaaaaa',alpha=.5))
        axes[0].set(xlabel='Current fixed GP J',ylabel='Future fixed-route excess (dB s)',title='Scores frozen BEFORE truth evaluation')
        axes[1].set(xlim=(0,snapshot.geometry.high[0]),ylim=(0,snapshot.geometry.high[1]),aspect='equal',title='Orange is posthoc only; grey areas are buildings')
        for ax in axes:ax.legend(fontsize=7)
        fig.suptitle(f"{chosen['case']} @ {chosen['time_seconds']:.1f}s; no new measurements inside candidate comparison")
        fig.savefig(out/'figures'/f'{scene}_candidates.png',dpi=130);plt.close(fig)
        sections.append(f'<h2>{scene}</h2><p>闭环到达：{metrics["success"]}；航程{metrics["executed_length_m"]:.2f}m；积分{metrics["interference_excess_integral_db_seconds"]:.3f}dB·s；'
            f'捕获{len(group)}个状态，满足机会门槛：{ready}。图示按公开风险标签选取{chosen["case"]}，未按真值效果筛图。</p>'
            f'<img src="figures/{scene}_candidates.png" width="100%">')
    ready_count=sum(r['ready_scene'] for r in scene_summary);passed=ready_count>=protocol['readiness_gate']['min_scenes']
    score_summary=[]
    for score in SCORE_NAMES:
        choices=[r for r in rows if r['role']==score]
        score_summary.append({'score':score,'case_count':len(choices),
            'mean_future_excess_db_seconds':float(np.mean([r['future_excess_db_seconds'] for r in choices])),
            'worse_exposure_than_incumbent_cases':sum(r['future_excess_db_seconds']>r['base_future_excess_db_seconds']+max(.1,.01*r['base_future_excess_db_seconds']) for r in choices),
            'materially_lower_exposure_than_incumbent_cases':sum(r['base_future_excess_db_seconds']>1e-8 and r['future_excess_db_seconds']<.9*r['base_future_excess_db_seconds'] for r in choices),
            'scope':'equal weight per captured state; overlaps/correlation,not independent trials; not closed-loop alternative controller'})
    save_json(out/'reports/summary.json',{'scenes':scene_summary,'choices':rows,'score_summary':score_summary,
        'readiness_gate':{'passes':passed,'ready_scenes':ready_count,'required_scenes':2},
        'unique_cases':len(data),'candidate_records':sum(len(d['records']) for d in data),
        'feasible_candidates':sum(r['feasible'] for d in data for r in d['records'])})
    save_json(out/'reports/audit.json',{'frozen_source_inputs_results_match':True,'all_choices_before_truth':True,
        'model_calls':0,'validation_test_seeds_generated_or_read':False,'scenario_retuning_or_result_rejection':False,
        'controller_unchanged':True,'alternative_scores_not_used_for_flight':True,'cases':audits})
    render_replays(out,frame_stride=6,playback_speed=2.,decision_pause_seconds=.5)
    page=out/'replay/index.html';page.write_text(page.read_text().replace('动画默认按仿真1倍速','本轮动画按仿真2倍速'))
    md='# 新开发任务与候选评分一致性\n\n4个预留开发seed，未生成/读取验证测试。0模型调用。评分备选只做同候选池排序，不进入控制器。\n\n'
    md+=f'共{len(data)}个唯一因果快照、{sum(len(d["records"]) for d in data)}条候选；开发机会门槛：{passed}（{ready_count}/4场景，要求≥2）。\n\n'
    md+='|场景|到达|航程/m|闭环积分/dB·s|快照数|有当前J与真实暴露共同改进机会|\n|---|---|---:|---:|---:|---|\n'
    for s in scene_summary:
        m=s['metrics'];md+=f"|{s['scene']}|{m['success']}|{m['executed_length_m']:.3f}|{m['interference_excess_integral_db_seconds']:.3f}|{s['case_count']}|{s['ready_scene']}|\n"
    md+='\n|评分|固定路线未来积分均值/dB·s|比原路线更差状态数|比原路线至少低10%状态数|\n|---|---:|---:|---:|\n'
    for s in score_summary:md+=f"|{s['score']}|{s['mean_future_excess_db_seconds']:.3f}|{s['worse_exposure_than_incumbent_cases']}|{s['materially_lower_exposure_than_incumbent_cases']}|\n"
    md+='\n评分均值按捕获状态等权，状态相关且未来路线重叠，不是独立样本或闭环策略排名。\n\n'
    md+='|快照|选择|当前J|长度/m|未来积分/dB·s|当前支持比例|\n|---|---|---:|---:|---:|---:|\n'
    for r in rows:md+=f"|{r['case']}|{r['role']}|{r['current_j']:.3f}|{r['length_m']:.2f}|{r['future_excess_db_seconds']:.3f}|{r['supported_fraction']:.1%}|\n"
    md+='\n“posthoc_reference”使用真实未来评分选择，只供诊断，不可提供给LLM。GIF是原library_numeric控制器的真实闭环，备选评分没有被执行。\n'
    (out/'RESULTS.md').write_text(md)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>开发任务与目标一致性</title><style>body{font:17px/1.8 sans-serif;max-width:1250px;margin:30px auto;padding:15px}img{max-width:100%}</style>'
        '<h1>新开发任务：可观测风险与修复目标</h1><p>4个预留开发种子，0模型调用，未读取验证/测试。全部场景保留，不因效果重采样。'
        '备选评分只做离线排序诊断，GIF均为原控制器实际闭环。</p>'
        '<p><a href="RESULTS.md">全部指标</a> · <a href="replay/index.html">4个闭环GIF</a> · <a href="reports/audit.json">审计</a> · '
        '<a href="reports/frozen_design.md">实验前设计</a> · <a href="../../docs/task_readiness.md">结论与下一步</a></p>'
        f'<p>开发机会门槛：{passed}，满足{ready_count}/4场景（要求≥2）。</p>'+''.join(sections))
    save_json(out/'status.json',{'status':'completed_development_readiness','readiness_gate_passed':passed,'model_calls':0})

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','scenarios','collect','propose','evaluate','report','all'])
    parser.add_argument('--out',type=Path,default=DEFAULT);args=parser.parse_args()
    if args.phase=='all':
        for function in (prepare,scenarios,collect,propose,evaluate,report):function(args.out)
    else:globals()[args.phase](args.out)
if __name__=='__main__':main()
