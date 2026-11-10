"""Causal file exchange and independent execution checks for agent experiments."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import time

import pytest

from uav_em.agent_navigation import (FileDecisionAdvisor, response_policy, save_json, strict_json)
from uav_em.config import ProjectConfig
from uav_em.planning import PlanResult, plan
from uav_em.simulation import simulate


def context():
    return {'time_seconds':0.,'vehicle':{'position_m':[20,80],'goal_m':[220,80]},
            'map':{'bounds_m':[[0,0],[240,160]]},'available_actions':['replan'],
            'probe_candidates':[], 'measurement_evidence':{'recent_points':[[20,80,0,-90]]},
            'information_sources':{'measurement_evidence':'point measurement'},
            'electromagnetic_estimate':{'local_samples':[[20,80,-90,1,True]],'estimator':{'method':'local_gp'}}, 'source_inference':{'measurements':[[20,80,0,-90]],
            'hidden_fields':[{'center_m':[123456,99999]}]},'hidden_truth':'DO_NOT_EXPOSE'}


def response(identifier):
    return {'request_id':identifier,'decision':{'action':'replan','planner':'astar','profile':'balanced','reason':'检查可见路线'},
            'strategy':{'mode':'baseline'},'memory':'没有测量的区域仍未知'}


def small_config():
    config=ProjectConfig()
    return replace(config,map=replace(config.map,columns=13,rows=9,start_m=(20,80),goal_m=(220,80),obstacles_m=()),
                   electromagnetic=replace(config.electromagnetic,fields=()),
                   simulation=replace(config.simulation,max_seconds=30,source_refinement_rounds=1))


def test_external_request_contains_causal_evidence_and_matching_response(tmp_path):
    (tmp_path/'contract.md').write_text('Visible-only decision protocol')
    advisor=FileDecisionAdvisor(tmp_path,'m1',timeout_seconds=2)
    def external_test_double():
        request=tmp_path/'requests/m1_001.json'
        while not request.exists():time.sleep(.01)
        data=json.loads(request.read_text())
        assert data['visible_context']['source_inference']['measurements']==[[20,80,0,-90]]
        assert data['visible_context']['measurement_evidence']['recent_points']==[[20,80,0,-90]]
        assert data['visible_context']['electromagnetic_estimate']['estimator']['method']=='local_gp'
        assert data['visible_context']['electromagnetic_estimate']['local_samples']==[[20,80,-90,1,True]]
        assert 'DO_NOT_EXPOSE' not in request.read_text()
        assert '123456' not in request.read_text()
        save_json(data['response_path'],response(data['request_id']))
    with ThreadPoolExecutor() as pool:
        future=pool.submit(external_test_double)
        result=advisor.decide(context())
        future.result()
    assert json.loads(result)['action']=='replan'
    assert advisor.last_call['backend']=='codex_subagent'
    assert advisor.last_call['status']=='accepted'
    assert advisor.strategy=={'mode':'baseline'}
    assert advisor.memory


@pytest.mark.parametrize('edit',[lambda x:x.update(request_id='stale'),
    lambda x:x['decision'].update(action='continue'),lambda x:x['decision'].update(source_hypothesis={}),
    lambda x:x.update(memory='x'*1201),lambda x:x.update(strategy={'mode':'baseline','risk_weight':8}),
    lambda x:x.update(unknown_code='print(1)')])
def test_invalid_external_outputs_are_rejected(edit):
    value=response('r1');edit(value)
    with pytest.raises(ValueError):response_policy(value,'r1',context())


def test_duplicate_json_or_nonfinite_output_is_rejected():
    with pytest.raises(ValueError):strict_json('{"action":1,"action":2}')
    with pytest.raises(ValueError):strict_json('{"weight":NaN}')


def test_timeout_is_not_repaired_into_a_model_decision(tmp_path):
    (tmp_path/'contract.md').write_text('test')
    advisor=FileDecisionAdvisor(tmp_path,'m1',timeout_seconds=.01)
    advisor.strategy={'mode':'bearing_risk'}
    with pytest.raises(TimeoutError):advisor.decide(context())
    assert advisor.strategy=={'mode':'baseline'}
    assert advisor.last_call['status']=='error'


def test_invalid_file_response_preserves_original_text_and_cleared_strategy(tmp_path):
    (tmp_path/'contract.md').write_text('Visible-only synthetic test')
    advisor = FileDecisionAdvisor(tmp_path, 'invalid', timeout_seconds=2)
    advisor.strategy = {'mode':'bearing_risk', 'risk_weight':12}
    def invalid_output():
        request_path = tmp_path/'requests/invalid_001.json'
        while not request_path.exists():
            time.sleep(.01)
        request = json.loads(request_path.read_text())
        value = response(request['request_id'])
        value['strategy'] = {'mode':'bearing_risk', 'risk_weight':31}
        save_json(request['response_path'], value)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(invalid_output)
        with pytest.raises(ValueError, match='strategy'):
            advisor.decide(context())
        future.result()
    original = (tmp_path/'responses/invalid_001.json').read_text()
    assert advisor.strategy == {'mode':'baseline'}
    assert advisor.last_call['raw_agent_response'] == original
    assert advisor.last_call['strategy_before_error'] == {'mode':'bearing_risk', 'risk_weight':12}
    assert advisor.last_call['status'] == 'error'
    assert json.loads((tmp_path/'exchange.jsonl').read_text())['raw_agent_response'] == original


def test_optional_backend_receives_snapshot_and_all_route_searches_use_it():
    seen=[]
    def backend(snapshot,start,goal,*args):
        assert not hasattr(snapshot,'electromagnetic') and not hasattr(snapshot,'world')
        seen.append(snapshot.time_seconds)
        return plan(snapshot,start,goal,*args)
    episode=simulate(small_config(),initial_planner='astar',planning_backend=backend)
    assert episode['metrics']['success']
    assert len(seen)==episode['metrics']['planning_calls']
    assert len(seen)>1


def test_external_backend_cannot_skip_endpoints_or_cross_geometry():
    def wrong(snapshot,start,goal,*args):
        return PlanResult('astar',path=[[100,80],list(goal)],cost=1,length_m=1)
    with pytest.raises(ValueError,match='invalid route'):
        simulate(small_config(),initial_planner='astar',planning_backend=wrong)
    config=small_config()
    config=replace(config,map=replace(config.map,obstacles_m=((100,60,140,100),)))
    def crosses(snapshot,start,goal,*args):
        return PlanResult('astar',path=[list(start),list(goal)],cost=1,length_m=1)
    with pytest.raises(ValueError,match='invalid route'):
        simulate(config,initial_planner='astar',planning_backend=crosses)


def test_codex_runtime_is_saved_without_pretending_it_is_local_qwen():
    class Agent:
        last_call={'backend':'codex_subagent','request_id':'test-only'}
        def decide(self,ctx):return json.dumps(response('r')['decision'])
    ep=simulate(small_config(),Agent(),'astar','codex_subagent')
    assert ep['decisions'][0]['backend']=='codex_subagent'
    assert ep['decisions'][0]['model_runtime']['request_id']=='test-only'
