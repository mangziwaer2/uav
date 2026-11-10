"""Point-only, causal online-agent feasibility pilot; no initial survey or bearing sensor."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from uav_em.agent_navigation import FileDecisionAdvisor, save_json, append_json
from uav_em.config import load_config
from uav_em.decision_log import render_decision_log
from uav_em.simulation import simulate

CONTRACT = '''# Causal UAV decision experiment
You are the online decision LLM. Read ONLY this public directory's contract.md, control.json,
requests/*.json and your own responses/*.json. Never inspect parent directories, source code,
scenario files, logs, other agents, truth images or future source scripts. This allowlist is a
procedural information boundary, not an operating-system sandbox. Do not spawn agents.
Each request is an independent causal observation; previous_memory belongs to that mission only.
The UAV knows building geometry and its own pose. It measures aggregate interference power ONLY
at its current position with noise; it has NO bearings, range or source coordinates. Recent
point measurements and local spatiotemporal GP estimates are provided. Unknown/null is not low
interference. Supported cells are inferred, not directly measured; GP uncertainty is not calibrated.
Field estimates score the CURRENT snapshot, not future arrival-time power. Time decay toward
prior is not evidence of an emitter turning off. Along-track change mixes spatial and temporal
changes. Never invent direction measurements. You may infer tentative trends and test them by flight.
Goal: arrive, avoid strong interference where practical, while retaining progress and respecting
hard geometry/energy constraints. -60dBm is stronger than -90dBm; reference is a SOFT cost, not
an equipment failure threshold. Do not claim safety of unobserved detours. Use new measurements
to revise or retain a detour. A direction choice under ambiguity should be labelled exploratory.
Available actions come from visible_context.available_actions. Every decision has action,
planner="rrt_star", profile=balanced/avoid_interference/explore, reason (short evidence-based string).
continue keeps the CURRENT route, NOT the entire original waypoint; replan searches to the goal.
replan may add ONE waypoint_m:[x,y] inside bounds and outside known blocking buildings/clearance.
The solver performs TWO standard RRT* searches through that waypoint, then geometry/energy checks.
The waypoint is a search endpoint, not a known safe location. After a replan, continue retains its
remaining route. probe must add target_id from the current probe_candidates and no waypoint_m.
Probe flies to one point, not a disk scan. At probe completion you must plan to goal again.
No strategy modules, source hypotheses, executable code, threshold edits or future information.
For each request write exactly one JSON response to response_path, atomically (temporary+rename):
{"request_id":"matching id","decision":{"action":"...","planner":"rrt_star",
"profile":"...","reason":"..."},"strategy":{"mode":"baseline"},"memory":"..."}
Memory is at most 1200 characters. Reuse only mission-specific previous_memory; do not transfer
hidden assumptions between missions. No response editing after publication. Invalid output/fallback
is recorded separately from accepted model execution. Poll public/control.json until status=completed
or failed; wait at most 30 seconds per tool call. Do not finish after a single response.
'''

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for part in ('scenarios','runs','reports','logs','replay','public/requests','public/responses'):
        (out/part).mkdir(parents=True)
    (out/'public/contract.md').write_text(CONTRACT, encoding='utf-8')
    base = load_config(ROOT/'configs/local_power.json')
    # Mechanism diagnostics fixed before any flight; not a representative benchmark.
    buildings = (
        {'id':'west_low','footprint_m':[[130,250],[200,250],[200,330],[130,330]],'height_m':12,'height_source':'synthetic'},
        {'id':'north','footprint_m':[[290,440],[380,440],[380,520],[290,520]],'height_m':45,'height_source':'synthetic'},
        {'id':'south','footprint_m':[[510,100],[590,100],[590,190],[510,190]],'height_m':40,'height_source':'synthetic'})
    def source(x,y,power,start=0,end=200,velocity=(0,0)):
        return dict(center_m=[x,y],eirp_dbm=power,height_m=1.5,start_seconds=start,end_seconds=end,velocity_mps=list(velocity))
    specs = [('p01',221,'static_interference',[source(390,300,4)]),
             ('p02',222,'moving_and_appearing',[source(360,350,2,end=38,velocity=(0,-.8)),source(590,270,4,start=26)])]
    scenes=[]
    for name,seed,mechanism,fields in specs:
        cfg=replace(base,seed=seed,map=replace(base.map,buildings=buildings),
            simulation=replace(base.simulation,max_seconds=160,max_advisor_calls=22),
            electromagnetic=replace(base.electromagnetic,fields=tuple(fields)))
        cfg.validate(); path=out/'scenarios'/f'{name}.json'; save_json(path,cfg.to_dict())
        scenes.append(dict(id=name,seed=seed,mechanism_evaluation_only=mechanism,sha256=digest(path)))
    paths=list((ROOT/'src/uav_em').glob('*.py'))+[Path(__file__).resolve()]
    sources={}
    for p in paths:
        rel=p.relative_to(ROOT); target=out/'reports/runtime_source'/rel
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);sources[str(rel)]=digest(p)
    save_json(out/'protocol.json',dict(study='point_only_online_agent_feasibility_not_novelty_or_generalization',
        scenes=scenes,methods=['numeric_rrt','online_agent'],sources_sha256=sources,
        contract_sha256=digest(out/'public/contract.md'),
        information='known buildings + current/past noisy point power + local GP; no bearing/initial survey/truth',
        planning='standard RRT*, balanced baseline; agent may change profile, use one waypoint or probe',
        budget_note='700 proposals per search; waypoint uses two searches; not equal total compute or objective. No optimality attribution.',
        filesystem_isolation='procedural_allowlist_not_OS_sandbox', initial_measurements=0))
    save_json(out/'public/control.json',dict(status='prepared',planned_missions=2))
    save_json(out/'status.json',dict(status='prepared'))

def run(out):
    protocol=json.loads((out/'protocol.json').read_text())
    for name,sha in protocol['sources_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('source changed since preparation: '+name)
    if digest(out/'public/contract.md')!=protocol['contract_sha256']:raise ValueError('contract changed')
    rows=[];save_json(out/'public/control.json',dict(status='running',planned_missions=2))
    try:
        for scene in protocol['scenes']:
            cfg=load_config(out/'scenarios'/f"{scene['id']}.json")
            if digest(out/'scenarios'/f"{scene['id']}.json")!=scene['sha256']:raise ValueError('scene changed')
            for method in protocol['methods']:
                run=out/'runs'/f"{method}_{scene['id']}_seed{cfg.seed}";run.mkdir(exist_ok=False)
                meta=dict(method=method,scene=scene['id'],seed=cfg.seed)
                save_json(run/'status.json',dict(status='running',**meta))
                advisor=FileDecisionAdvisor(out/'public',f"mission_{scene['id']}",timeout_seconds=900) if method=='online_agent' else None
                def hook(stage,data):
                    if stage in ('advisor_started','advisor_returned','decision_executed','finished'):
                        append_json(run/'execution.jsonl',dict(stage=stage,**data))
                    if stage=='decision_executed':print(method,scene['id'],data['time_seconds'],data['decision'].get('accepted_action'),flush=True)
                ep=simulate(cfg,advisor,'rrt_star','codex_subagent' if advisor else 'none',hook,allow_waypoints=True)
                save_json(run/'episode.json',ep);save_json(run/'metrics.json',ep['metrics'])
                (run/'decision_log.txt').write_text(render_decision_log(ep),encoding='utf-8')
                row=dict(**meta,**ep['metrics']);rows.append(row)
                save_json(run/'status.json',dict(status='completed',**row));save_json(out/'results.json',rows)
                print('COMPLETED',json.dumps(row),flush=True)
        save_json(out/'public/control.json',dict(status='completed',completed_missions=2))
        save_json(out/'status.json',dict(status='completed',tasks=len(rows)))
    except BaseException as exc:
        save_json(out/'public/control.json',dict(status='failed',error=repr(exc)))
        save_json(out/'status.json',dict(status='failed',error=repr(exc)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.output.resolve()) if a.prepare else run(a.output.resolve())
