"""Numeric composition must preserve public information and the existing acceptor."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import pytest

from uav_em.agent_navigation import save_json
from uav_em.belief import ElectromagneticBelief
from uav_em.config import load_config
from uav_em.context import build_context
from uav_em.environment import ElectromagneticWorld
from uav_em.repair_program import OperatorLibrary, compile_program, FileProgramPolicy, LibraryNumericalPolicy
from uav_em.route_repair import RepairSession, RepairSettings, parse_proposal, RouteRepairOptimizer
from uav_em.simulation import advance_route

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def state():
    cfg = load_config(ROOT/'configs/local_power.json')
    cfg = replace(cfg, map=replace(cfg.map, buildings=(), obstacles_m=()),
                  electromagnetic=replace(cfg.electromagnetic, fields=(), measurement_noise_db=0))
    world = ElectromagneticWorld(cfg);belief = ElectromagneticBelief(cfg, world.geometry)
    belief.update(world.observe(cfg.map.start_m, 0));snap = belief.snapshot(0)
    route = [[40,300], [180,400], [480,400], [760,300]]
    ctx = build_context(route[0], route[-1], route, snap, cfg, 100, [], [], None)
    session = RepairSession(snap, ctx, route, 3000, RepairSettings(), 41)
    return cfg, snap, ctx, session


def test_program_calculates_finite_parameters_and_same_normal_proposal(state):
    _,_,_,session = state;c = session.request(1, 3);before = deepcopy(c)
    source = '''part = select_interval(metric="interference", padding=1)
boxes = corridors(part, side="both", scale="local")
distribution = sampling(boxes, weighting="cost", concentration="focused")
decision = local_rrt(part, distribution, quota(mode="balanced"))'''
    p,t = compile_program(source,c,'test computed local repair')
    assert p == parse_proposal(p,c) and p['operator'] == 'local_rrt'
    assert 1 <= p['sample_budget'] <= 350 and p['sampling']['focus_probability'] == .6
    assert len(p['sampling']['regions']) == 2 and not t['arbitrary_python_executed']
    assert c == before and session.work.samples == 0


def test_future_route_only_no_modification_to_flown_prefix(state):
    _,snap,ctx,_ = state
    original = [[40,300], [180,300], [480,300], [760,300]]
    remaining,flown = advance_route(original, 200)
    assert flown[-1] == remaining[0] == [240,300]
    ctx['vehicle']['position_m'] = remaining[0]
    session = RepairSession(snap,ctx,remaining,3000,RepairSettings(),1)
    c = session.request(1,3)
    assert c['route_repair']['anchors'][0]['position_m'] == [240,300]
    assert all(a['position_m'][0] >= 240 for a in c['route_repair']['anchors'])
    p,t = compile_program('decision = shortcut(whole())',c,'only remaining route')
    assert session.apply(p)['preserves_route_outside_interval']
    assert original[0] == [40,300] and c['measurement_evidence']['origin'] == 'received_point_measurements'
    bad = dict(p,start_id='historical_anchor')
    with pytest.raises(ValueError):parse_proposal(bad,c)


@pytest.mark.parametrize('source',[
    'import os\ndecision = keep()',
    'decision = __import__("os")',
    'decision = keep.__globals__',
    'decision = open("secret")',
    'decision = [keep() for i in range(100)]',
    'while True:\n decision = keep()',
    'decision = keep()\ndecision = keep()',
    'decision = keep()["operator"]',
    'decision = 1 + 2',
    'decision = quota(mode="infinite")',
    'decision = select_interval(padding=99)',
    'foo = keep()',
])
def test_invalid_program_rejected_without_search_or_context_mutation(state,source):
    _,_,_,session = state;c = session.request(1,3);before = deepcopy(c)
    with pytest.raises((ValueError,TypeError,SyntaxError)):compile_program(source,c,'bad')
    assert session.work.samples == 0 and c == before


def test_program_result_still_subject_to_fixed_objective_acceptance(state):
    _,_,_,session = state;c = session.request(1,3)
    p,t = compile_program('decision = shortcut(whole())',c,'attempt shorten')
    f = session.apply(p)
    assert f['accepted'] and f['candidate_minus_previous_best'] < 0
    newer = session.request(2,3)
    p,t = compile_program('decision = shortcut(whole())',newer,'try unchanged route')
    assert not session.apply(p)['accepted']


def test_program_sampling_uses_only_published_samples_and_clips_map(state):
    _,_,_,session = state;c = session.request(1,3)
    lib = OperatorLibrary(c);boxes = lib.corridors(lib.whole(), 'both', 'wide')
    dist = lib.sampling(boxes)
    for b in dist['regions']:
        assert np.all(np.array(b['low_m']) >= 0)
        assert np.all(np.array(b['high_m']) <= c['map']['bounds_m'][1])
        assert 0 < b['weight'] <= 1
    c['evaluation_only_config'] = {'secret':'must never matter'}
    assert OperatorLibrary(c).sampling(boxes) == dist


def test_quota_responds_to_measured_search_cost_and_shared_resources(state):
    _,_,_,session = state;c = session.request(1,3)
    first = OperatorLibrary(c).quota()
    c['route_repair']['resources'].update(samples=100,edges=9000,edges_left=900,samples_left=2000)
    second = OperatorLibrary(c).quota()
    assert second < first and 1 <= second <= 900//90


def test_file_program_bridge_records_source_computation_and_resolved_proposal(state,tmp_path):
    _,_,_,session = state;c = session.request(1,3)
    (tmp_path/'contract.md').write_text('restricted library composition test')
    policy = FileProgramPolicy(tmp_path,'test',3)
    def respond():
        req_path = tmp_path/'requests/test_001.json'
        while not req_path.exists():time.sleep(.01)
        req = json.loads(req_path.read_text())
        save_json(req['response_path'],dict(request_id=req['request_id'],
            decision={'program':'decision = shortcut(whole())','reason':'shorten remaining route'},
            strategy={'mode':'baseline'},memory=''))
    with ThreadPoolExecutor() as pool:
        f=pool.submit(respond);raw=policy.decide(c);f.result()
    assert parse_proposal(raw,c)['operator'] == 'shortcut'
    execution = policy.last_call['program_execution']
    assert execution['program'] == 'decision = shortcut(whole())'
    assert execution['resolved_proposal'] == json.loads(raw)
    exchange=json.loads((tmp_path/'exchange.jsonl').read_text())
    assert exchange['program_execution'] == execution


def test_library_numeric_control_is_not_counted_as_model_or_recovery(state):
    _,snap,ctx,session=state
    opt=RouteRepairOptimizer(numeric_policy=LibraryNumericalPolicy())
    result,report=opt.optimize(snap,ctx,session.best,3000,41,0)
    assert report['actual_advisor_calls']==0 and report['advisor_failures']==0
    assert report['accepted_model_proposals']==0 and not report['numeric_recovery']
    assert result.path and report['trace'][0]['feedback']['accepted']
