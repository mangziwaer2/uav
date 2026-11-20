"""Synthetic component checks only; stdout by default, exclusive JSON output if named.

Uses explicit synthetic fixtures from the archived regression tests. No world,
flight, scientific outputs, hidden configuration, or model service is accessed.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))

from uav_em.event_demo import FileAdvisor
from uav_em.feasible_scenario import execute,shortcut
from uav_em.joint_prediction import JointSettings,local_prediction
from uav_em.planning import path_length
from uav_em.repair_search import WorkBudget
from uav_em.source_geometry import constrained_refine_fit


def fixtures(name):
    return runpy.run_path(str(ROOT/'tests'/name))


def collect():
    basis=fixtures('test_shared_basis_v2.py');warm=[]
    for capacity in (1,2):
        bank,public,job=basis['problem'](capacity)
        result,work=basis['fit'](bank,public,job)
        warm.append(dict(capacity=capacity,fit_mse_db2=result['model']['fit_mse'],
            selected_branch=result['selected_branch'],resources=result['job_resources'],
            dictionary_kernel_terms=work.dictionary_kernel_terms))
    geo=fixtures('test_shared_geometry_v2.py')
    public,fit,_=geo['problem'](two_sources=True)
    refined=constrained_refine_fit(fit,public);trace=refined['geometry_refinement']
    geometry=dict(previous_objective=trace['previous_objective'],candidate_objective=trace['candidate_objective'],
        accepted=trace['accepted'],solver_success=trace['solver_success'],stop_reason=trace['stop_reason'],
        resources=trace['resources'],region_eligible=trace['region_eligible'],bounds_eligible=trace['bounds_eligible'])
    planning=fixtures('test_shared_planning_v2.py');paths=[]
    for limit,quotas in ((10000,(350,350)),(744,(40,40))):
        g,state,forecast,work,request,plan=planning['two_leg_problem'](limit,quotas)
        result=execute(request,plan,state,forecast,g,work,'mixture',7)
        paths.append(dict(edge_limit=limit,quotas=list(quotas),failure=result.get('failure'),
            resources=work.summary(),candidates=[dict(id=c['id'],length_m=path_length(c['path']),
                feasible=c['feasible'],reason=c['reason']) for c in result['candidates']]))
    g,route=planning['shortcut_problem']();work=WorkBudget(700,10000)
    partial,short_trace=shortcut(route,g,work)
    shortening=dict(original_length_m=path_length(route),retained_length_m=path_length(partial),
        trace=short_trace,resources=work.summary())
    support=fixtures('test_shared_support_v2.py')
    local=local_prediction(support['fixture'](),[[100.,100.]],[36.],JointSettings())
    local={key:value.tolist() for key,value in local.items()}
    with tempfile.TemporaryDirectory(prefix='shared_tools_v2_synthetic_') as folder:
        public=Path(folder)
        for name in ('requests','responses'):(public/name).mkdir()
        receipt=public/'responses/00.started.json';receipt.write_text('{"agent_id":')
        clock=[100.];stages=[0]
        def sleep(dt):
            clock[0]+=dt;stages[0]+=1
            if stages[0]==2:
                receipt.write_text(json.dumps(dict(agent_id='synthetic',request_id='bound',started_monotonic=100.15)))
                (public/'responses/00.raw.txt').write_text('{}');(public/'responses/00.ready').touch()
        advisor=FileAdvisor(public,'synthetic',lambda:clock[0],sleep)
        envelope=advisor.ask('inference',dict(request_id='bound',time_seconds=0.))
        mailbox=dict(simulated_worker_receipt=True,real_model_calls=0,
            timely=envelope['timely'],error=envelope['error'],receipt_json_retries=envelope['receipt_json_retries'],
            wall_seconds=envelope['wall_seconds'])
    return dict(kind='synthetic_shared_tool_diagnostics_v2',warm_replacement=warm,geometry_retention=geometry,
        two_leg_search=paths,partial_shortcut=shortening,gp_support=local,mailbox=mailbox,
        new_scientific_flights=0,new_sensor_points=0,truth_queries=0,real_experiment_model_calls=0,
        limitations='Component counterexamples only; no historical causation or LLM task benefit established.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=collect();payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:stream.write(payload)
    else:print(payload)
