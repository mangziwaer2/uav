"""Single entry: event-triggered numerical loop or historical fixed-time reference."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from uav_em.config import load_config
from uav_em.scenario_loop import run

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('describe', help='Report status without running a flight')
    ref = commands.add_parser('reference', help='New numerical flight with fixed C1 schedule; no LLM')
    ref.add_argument('--config', type=Path, default=ROOT/'configs/local_power.json')
    ref.add_argument('--method', choices=('gp_numeric', 'persistence_numeric', 'mixture_numeric'), default='mixture_numeric')
    ref.add_argument('--output', type=Path, required=True, help='New outputs/interference_* directory')
    event = commands.add_parser('event', help='Power-rise-triggered numerical loop; no LLM without an advisor integration')
    event.add_argument('--config', type=Path, default=ROOT/'configs/local_power.json')
    event.add_argument('--method', choices=('gp_numeric', 'persistence_numeric', 'mixture_numeric'), default='mixture_numeric')
    event.add_argument('--output', type=Path, required=True, help='New outputs/interference_* directory')
    demo = commands.add_parser('demo', help='Bounded real-LLM event demonstration and replay')
    demo.add_argument('stage', choices=('prepare','run','finalize'))
    demo.add_argument('--method', choices=('gp_numeric','mixture_numeric','mixture_llm'))
    demo.add_argument('--output', type=Path, required=True)
    joint=commands.add_parser('joint',help='Frozen GP/equivalent-field development experiment D1')
    joint.add_argument('stage',choices=('prepare','run','finalize'))
    joint.add_argument('--method',choices=('source_numeric','joint_point_numeric','joint_N','joint_I','joint_P','joint_J','gp_numeric','persistence_numeric'))
    joint.add_argument('--output',type=Path,required=True)
    source_only=commands.add_parser('source-only',help='D2-M0 bounded numerical source-only mechanism pilot')
    source_only.add_argument('stage',choices=('prepare','run','finalize'))
    source_only.add_argument('--method',choices=('source_early','source_revise'))
    source_only.add_argument('--output',type=Path,required=True)
    search=commands.add_parser('source-search',help='D2-M1 bounded lateral-search mechanism pilot')
    search.add_argument('stage',choices=('prepare','run','finalize'))
    search.add_argument('--method',choices=('search_feedback','search_lattice'))
    search.add_argument('--output',type=Path,required=True)
    renewal=commands.add_parser('source-renewal',help='D2-M2 bounded active-source replacement mechanism pilot')
    renewal.add_argument('stage',choices=('prepare','run','finalize'))
    renewal.add_argument('--method',choices=('renewal_keep','renewal_replace'))
    renewal.add_argument('--output',type=Path,required=True)
    geometry=commands.add_parser('source-geometry',help='D2-M3 bounded source geometry refinement pilot')
    geometry.add_argument('stage',choices=('prepare','run','finalize'))
    geometry.add_argument('--method',choices=('geometry_amplitude','geometry_bounded'))
    geometry.add_argument('--output',type=Path,required=True)
    allocation=commands.add_parser('task-allocation',help='D2-M4 fixed-state numerical/LLM computation allocation')
    allocation.add_argument('stage',choices=('prepare','run','finalize'))
    allocation.add_argument('--method',choices=('allocation_N','allocation_I'))
    allocation.add_argument('--output',type=Path,required=True)
    proposals=commands.add_parser('task-proposals',help='D2-M5 fixed-state two-round source mechanism proposals')
    proposals.add_argument('stage',choices=('prepare','run','finalize'))
    proposals.add_argument('--method',choices=('allocation_N','allocation_I'))
    proposals.add_argument('--output',type=Path,required=True)
    constrained=commands.add_parser('constraint-replay',help='D2-M6 fixed M5 proposal replay with explicit geometry constraints')
    constrained.add_argument('stage',choices=('prepare','run','finalize'))
    constrained.add_argument('--method',choices=('allocation_N','allocation_I'))
    constrained.add_argument('--output',type=Path,required=True)
    contrast=commands.add_parser('task-contrast',help='D2-M7 cached route contrasts and compatible rate witnesses')
    contrast.add_argument('stage',choices=('prepare','run','finalize'))
    contrast.add_argument('--method',choices=('contrast_N','contrast_I'))
    contrast.add_argument('--output',type=Path,required=True)
    overview=commands.add_parser('replay-overview',help='Display-only two-panel replay from an existing study')
    overview.add_argument('--study',type=Path,required=True)
    overview.add_argument('--scene',choices=('static','dynamic'),default='dynamic')
    overview.add_argument('--view',choices=('source','full-mixture'),default='source')
    args = parser.parse_args()
    if args.command == 'describe':
        print('研究：复杂动态电磁环境下，LLM辅助UAV提前避让高风险区域并到达目标，兼顾航程与暴露。')
        print('目标链：点测 -> GP辅助等效场 -> 前方预警 -> LLM组织检验/路径搜索 -> 数值复核 -> 短段执行与新点测更新。')
        print('当前实现：保留旧GP/mixture分支；D1新增冻结联合预测、GP软形状拟合及前方支持内阈值预警。')
        print('新主分支：源池不含全图功率延续；source-only检查失配修订，source-search检查共享预算内的侧向路线搜索。')
        print('source-renewal检验集体失配后活动源版本替换；数值机制实验不等于LLM收益。')
        print('source-geometry在相同细化预算下比较幅度与源中心/尺度优化，仍须验证未来预测和任务代价。')
        print('task-allocation在三个固定公开状态比较数值/独立LLM分配一次细化机会；不是新闭环飞行。')
        print('研究证据：旧C1数值推断+LLM路径未获优化收益；D1结果以独立目录的合同/原始记录为准，代码实现不代表性能已证明。')
        print('event_demo/timing_diagnostic旧目录缺失，历史叙述不能代替原件。')
        print('MLP/训练及进一步异步工程暂缓；通信受干扰是背景，不另建断联/恢复研究线。等待冻结物理时间，不称实机实时。')
        print('接手见docs/AGENT_HANDOFF.md；唯一待办顺序见docs/current_roadmap.md。本命令不运行飞行。')
        return
    if args.command=='replay-overview':
        from uav_em.replay_overview import render_overview
        study=args.study.resolve()
        if study.parent!=ROOT/'outputs' or not study.name.startswith('interference_'):
            parser.error('--study must be an existing direct outputs/interference_* directory')
        render_overview(study,args.scene,args.view)
        return
    out = args.output.resolve()
    if out.parent != ROOT/'outputs' or not out.name.startswith('interference_'):
        parser.error('--output must be a new direct outputs/interference_* directory')
    if args.command=='joint':
        from uav_em import joint_experiment
        if args.stage=='run':
            if args.method is None:parser.error('joint run requires --method')
            joint_experiment.run_one(out,args.method)
        else:{'prepare':joint_experiment.prepare,'finalize':joint_experiment.finalize}[args.stage](out)
        return
    if args.command=='source-only':
        from uav_em import source_study
        if args.stage=='run':
            if args.method is None:parser.error('source-only run requires --method')
            source_study.run_one(out,args.method)
        else:{'prepare':source_study.prepare,'finalize':source_study.finalize}[args.stage](out)
        return
    if args.command=='source-search':
        from uav_em import source_study
        if args.stage=='prepare':source_study.prepare(out,search=True)
        elif args.stage=='run':
            if args.method is None:parser.error('source-search run requires --method')
            source_study.run_one(out,args.method)
        else:source_study.finalize(out)
        return
    if args.command=='source-renewal':
        from uav_em import source_study
        if args.stage=='prepare':source_study.prepare(out,renewal=True)
        elif args.stage=='run':
            if args.method is None:parser.error('source-renewal run requires --method')
            source_study.run_one(out,args.method)
        else:source_study.finalize(out)
        return
    if args.command=='task-allocation':
        from uav_em import task_allocation
        if args.stage=='run':
            if args.method is None:parser.error('task-allocation run requires --method')
            task_allocation.run_one(out,args.method)
        else:{'prepare':task_allocation.prepare,'finalize':task_allocation.finalize}[args.stage](out)
        return
    if args.command=='task-proposals':
        from uav_em import task_proposals
        if args.stage=='run':
            if args.method is None:parser.error('task-proposals run requires --method')
            task_proposals.run_one(out,args.method)
        else:{'prepare':task_proposals.prepare,'finalize':task_proposals.finalize}[args.stage](out)
        return
    if args.command=='constraint-replay':
        from uav_em import constraint_replay
        if args.stage=='run':
            if args.method is None:parser.error('constraint-replay run requires --method')
            constraint_replay.run_one(out,args.method)
        else:{'prepare':constraint_replay.prepare,'finalize':constraint_replay.finalize}[args.stage](out)
        return
    if args.command=='task-contrast':
        from uav_em import task_contrast
        if args.stage=='run':
            if args.method is None:parser.error('task-contrast run requires --method')
            task_contrast.run_one(out,args.method)
        else:{'prepare':task_contrast.prepare,'finalize':task_contrast.finalize}[args.stage](out)
        return
    if args.command=='source-geometry':
        from uav_em import source_study
        if args.stage=='prepare':source_study.prepare(out,geometry=True)
        elif args.stage=='run':
            if args.method is None:parser.error('source-geometry run requires --method')
            source_study.run_one(out,args.method)
        else:source_study.finalize(out)
        return
    if args.command=='demo':
        from uav_em import event_demo
        if args.stage=='run':
            if args.method is None:parser.error('demo run requires --method')
            event_demo.run_one(out,args.method)
        else:
            {'prepare':event_demo.prepare,'finalize':event_demo.finalize}[args.stage](out)
        return
    config = load_config(args.config)
    out.mkdir(exist_ok=False)
    timing='power_rise' if args.command=='event' else 'fixed'
    metadata = dict(kind='power_rise_numerical_prototype' if timing=='power_rise' else 'fixed_time_numerical_reference',
                    timing=timing,asynchronous=False,actual_llm=False,method=args.method,config=asdict(config))
    (out/'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    try:
        episode = run(config, args.method, timing=timing)
        (out/'episode.json').write_text(json.dumps(episode, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc)), ensure_ascii=False)+'\n')
        raise
    print(json.dumps(episode['metrics'], ensure_ascii=False, indent=2))
    print('Raw flight only: no post-flight truth evaluation or GIF rendering was performed.')


if __name__ == '__main__':
    main()
