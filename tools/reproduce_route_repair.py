"""当前源码重放第一版真实提议，核验数值行为；不调用新LLM，不覆盖原证据。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json
from uav_em.config import load_config
from uav_em.route_repair import RouteRepairOptimizer,RepairSettings
from uav_em.simulation import simulate


def read(p):return json.loads(p.read_text())

class RecordedPolicy:
    def __init__(self,episode):
        self.recordings=[t for d in episode['decisions'] for t in d['repair_optimization']['trace'] if t.get('backend')!='numeric_recovery']
        self.calls=0;self.last_call={}
    def decide(self,ctx):
        t=self.recordings[self.calls];self.calls+=1
        for key in ('route_repair','measurement_evidence','vehicle','time_seconds'):
            if ctx[key]!=t['context'][key]:raise ValueError('causal replay context mismatch: '+key)
        self.last_call={'backend':'recorded_repair_actions_no_new_model','original_request_id':t.get('model_runtime',{}).get('request_id')}
        return t['raw_response']


def reproduce(out):
    destination=out/'reports/reproduction.json'
    if destination.exists():raise FileExistsError('reproduction exists')
    protocol=read(out/'protocol.json');results=[]
    for method in protocol['methods']:
        old=read(out/'runs'/f'{method}_r01_seed311/episode.json')
        policy=RecordedPolicy(old) if method!='repair_numeric' else None
        optimizer=RouteRepairOptimizer(policy,RepairSettings(max_rounds=protocol['rounds'][method],**protocol['settings']))
        new=simulate(load_config(out/'scenarios/r01.json'),advisor_name='recorded_repair_actions' if policy else 'numeric_repair',route_optimizer=optimizer)
        checks={}
        for key in ('success','termination','executed_length_m','simulation_seconds','energy_used_wh',
            'high_interference_exposure_seconds','interference_excess_integral_db_seconds','decisions','advisor_calls',
            'advisor_failures','fallbacks','repair_edge_evaluations','repair_sample_proposals','repair_accepted_proposals'):
            a,b=old['metrics'][key],new['metrics'][key]
            checks[key]=(a==b if isinstance(a,(str,bool,int)) else bool(np.isclose(a,b,rtol=0,atol=1e-8)))
        before=np.array([f['position_m'] for f in old['frames']]);after=np.array([f['position_m'] for f in new['frames']])
        checks['trajectory']=bool(before.shape==after.shape and np.allclose(before,after,rtol=0,atol=1e-8))
        checks['all_recordings_consumed']=policy is None or policy.calls==len(policy.recordings)
        results.append({'method':method,'checks':checks,'all_match':all(checks.values()),'new_model_calls':0,
                        'current_replay_compute_seconds':new['metrics']['planner_seconds'],
                        'current_replay_rrt_seconds':new['metrics']['repair_rrt_seconds'],
                        'timing_note':'replay walltime excludes model inference; not the missing historical time'})
        print(method,results[-1]['all_match'],flush=True)
    save_json(destination,{'kind':'recorded_proposal_replay_after_timing_and_report_fixes_not_independent_LLM_trial',
        'runs':results,'all_match':all(r['all_match'] for r in results),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/uav_em').glob('*.py')}})
    if not all(r['all_match'] for r in results):raise AssertionError('replay mismatch')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    reproduce(p.parse_args().output.resolve())
