"""Replay recorded actions on current code; no new model calls and no result overwrite."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from uav_em.agent_navigation import save_json
from uav_em.config import load_config
from uav_em.simulation import simulate

def load(p):return json.loads(p.read_text())

class RecordedActions:
    def __init__(self,decisions):self.decisions=decisions;self.calls=0;self.last_call={}
    def decide(self,ctx):
        d=self.decisions[self.calls];self.calls+=1;old=d['context']
        # Rebuild causal evidence; do not return actions if changed measurements/pose
        # would make this a different decision problem. Removed metadata is not evidence.
        for key in ('time_seconds','vehicle','measurement_evidence','latest_observation','electromagnetic_estimate'):
            if ctx[key]!=old[key]:raise ValueError('recorded action context changed: '+key)
        self.last_call={'backend':'recorded_actions_not_new_model','actual_model_called':False,
                        'original_request_id':d['model_runtime']['request_id']}
        return d['raw_response']

def reproduce(out):
    report_path=out/'reports/reproduction.json'
    if report_path.exists():raise FileExistsError('reproduction already exists')
    results=[]
    keys=('success','termination','simulation_seconds','executed_length_m','energy_used_wh',
          'high_interference_exposure_seconds','interference_excess_integral_db_seconds',
          'observations','flight_observations','decisions','advisor_calls','advisor_failures','fallbacks',
          'planning_calls','rrt_sample_proposals')
    for row in load(out/'results.json'):
        original=load(out/'runs'/f"{row['method']}_{row['scene']}_seed{row['seed']}"/'episode.json')
        cfg=load_config(out/'scenarios'/f"{row['scene']}.json")
        advisor=RecordedActions(original['decisions']) if row['method']=='online_agent' else None
        replay=simulate(cfg,advisor,'rrt_star','recorded_actions_not_new_model' if advisor else 'none',allow_waypoints=True)
        checks={}
        for key in keys:
            a,b=original['metrics'][key],replay['metrics'][key]
            checks[key]=a==b if isinstance(a,(str,bool,int)) else bool(np.isclose(a,b,rtol=0,atol=1e-8))
        old_positions=np.asarray([f['position_m'] for f in original['frames']])
        new_positions=np.asarray([f['position_m'] for f in replay['frames']])
        positions=old_positions.shape==new_positions.shape and np.allclose(old_positions,new_positions,rtol=0,atol=1e-8)
        times=np.allclose([f['time_seconds'] for f in original['frames']],[f['time_seconds'] for f in replay['frames']],rtol=0,atol=1e-8) if positions else False
        results.append({'scene':row['scene'],'method':row['method'],'metrics_match':checks,
                        'trajectory_matches':bool(positions),'frame_times_match':bool(times),
                        'actual_model_calls':0,'replayed_actions':advisor.calls if advisor else 0,
                        'all_match':all(checks.values()) and bool(positions) and bool(times)})
        print(row['scene'],row['method'],results[-1]['all_match'],flush=True)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((ROOT/'src/uav_em').glob('*.py'))}
    report={'purpose':'deterministic_recorded_action_replay_after_cleanup_not_an_independent_LLM_trial',
            'fresh_model_calls':0,'current_source_sha256':sources,'runs':results,
            'all_match':all(r['all_match'] for r in results)}
    save_json(report_path,report)
    if not report['all_match']:raise AssertionError('reproduction mismatch')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    reproduce(p.parse_args().output.resolve())
