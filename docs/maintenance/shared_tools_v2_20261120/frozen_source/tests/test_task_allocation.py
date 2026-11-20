from copy import deepcopy
import json
from pathlib import Path

import pytest

from uav_em import task_allocation as task
from uav_em.event_inference import digest


@pytest.fixture
def packet():
    # Existing unsealed development cache. Missing scientific fixtures must fail.
    episode=json.loads((task.M3/'runs/geometry_amplitude/episode.json').read_text())
    return task.make_packet(episode,25.,dict(bounds_m=[[0.,0.],[800.,600.]],buildings=[]))


def test_public_packet_ignores_future_truth_and_metrics_and_keeps_pilot_cutoff(packet):
    episode=json.loads((task.M3/'runs/geometry_amplitude/episode.json').read_text())
    original=deepcopy(episode)
    episode['metrics']={'hidden_truth':'do not reveal'}
    for row in episode['observations']:
        if row['time_seconds']>25:row['interference_dbm']=9999.
    for key in ('prediction_records','joint_prediction_records'):
        for row in episode[key]:
            if row['observation']['time_seconds']>25:row['observation']['interference_dbm']=9999.
    actual=task.make_packet(episode,25.,packet['known_map'])
    assert actual==packet
    assert packet['history']==[p for p in original['observations'] if p['time_seconds']<=25.]
    assert 'metrics' not in packet and 'hidden_truth' not in json.dumps(packet)
    pilot=next(f for f in packet['fits'] if f['phase']=='pilot')
    assert pilot['window'][-1]['time_seconds']<25
    assert all(p['time_seconds']<=25 for p in pilot['pilot_feedback']['tail_points'])
    assert pilot['gp_soft_constraints']['prefix_end_seconds']==pilot['window'][-1]['time_seconds']
    assert packet['history_sha256']==digest(packet['history'])


@pytest.mark.parametrize('change',[
    dict(request_id='wrong'),dict(action='change_weight'),dict(target_id='old_model'),
    dict(target_id=[]),dict(extra='disallowed'),dict(reason=''),
])
def test_response_cannot_rewrite_bindings_action_parameters_or_target(packet,change):
    raw=task.numerical_choice(packet);raw.update(change)
    with pytest.raises((ValueError,TypeError)):task.validate(packet,raw)


def test_keep_requires_no_target_and_packet_is_tamper_evident(packet):
    raw=dict(request_id=packet['request_id'],action='keep',target_id=None,reason='retain original')
    assert task.validate(packet,json.dumps(raw))==raw
    raw['target_id']=packet['allowed_target_ids'][0]
    with pytest.raises(ValueError,match='null'):task.validate(packet,raw)
    packet['state']['remaining_energy_wh']+=1
    with pytest.raises(ValueError,match='changed'):task.validate(packet,raw)


def test_numeric_rule_uses_task_regret_then_loss_and_stable_identifier(packet):
    ids=packet['allowed_target_ids']
    for w in packet['world_actions']:w['allocation_priority']=0.
    for f in packet['fits']:f['fit_objective']=1.
    next(w for w in packet['world_actions'] if w['id']==ids[1])['allocation_priority']=10.
    assert task.numerical_choice(packet)['target_id']==ids[1]
    for w in packet['world_actions']:w['allocation_priority']=0.
    packet['fits'][2]['fit_objective']=2.
    assert task.numerical_choice(packet)['target_id']==ids[2]


def test_single_target_refinement_preserves_other_models_weights_and_input(packet,monkeypatch):
    before=deepcopy(packet);calls=[];target=packet['allowed_target_ids'][1]
    def refine(fit,public,mode):
        calls.append((fit,public,mode));result=deepcopy(fit)
        result['model']['peaks_dbm'][0]-=1.
        result['geometry_refinement']=dict(accepted=True)
        return result
    monkeypatch.setattr(task,'refine_fit',refine)
    raw=dict(request_id=packet['request_id'],action='refine_geometry',target_id=target,reason='task-relevant explanation')
    models,fit=task.apply_choice(packet,raw)
    assert len(calls)==1 and calls[0][2]=='geometry' and packet==before
    for a,b in zip(packet['models'],models):
        assert a['weight']==b['weight'] and a['log_weight']==b['log_weight']
        if a['id']==target:
            assert b['peaks_dbm'][0]==pytest.approx(a['peaks_dbm'][0]-1)
            assert b['scored_points']==0 and b['reference_time_seconds']==25.
        else:assert a==b
    assert fit['geometry_refinement']['accepted']


def test_failed_refinement_and_keep_do_not_change_model_bank(packet,monkeypatch):
    calls=[]
    def refine(fit,public,mode):
        calls.append(1);return dict(deepcopy(fit),geometry_refinement=dict(accepted=False))
    monkeypatch.setattr(task,'refine_fit',refine)
    raw=task.numerical_choice(packet);models,_=task.apply_choice(packet,raw)
    assert models==packet['models'] and len(calls)==1
    raw.update(action='keep',target_id=None)
    models,fit=task.apply_choice(packet,raw)
    assert models==packet['models'] and fit is None and len(calls)==1


def test_freeze_check_detects_current_source_mutation_before_any_operation(tmp_path,monkeypatch):
    (tmp_path/'src').mkdir();(tmp_path/'src/x.py').write_text('original')
    out=tmp_path/'out';(out/'reports/frozen_source/src').mkdir(parents=True)
    (out/'reports/frozen_source/src/x.py').write_text('original')
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256={'src/x.py':task.sha(tmp_path/'src/x.py')})))
    monkeypatch.setattr(task,'ROOT',tmp_path)
    (tmp_path/'src/x.py').write_text('mutated')
    with pytest.raises(ValueError,match='source changed'):task.check(out)
