import json
import math
from pathlib import Path

import numpy as np
import pytest

from uav_em.event_demo import FileAdvisor, execution_samples, frozen_display


def mailbox(tmp_path):
    for name in ('requests','responses'):(tmp_path/name).mkdir()
    return dict(request_id='id',phase='pilot',time_seconds=3.)


def test_actual_call_requires_bound_start_receipt_and_retains_original(tmp_path):
    request=mailbox(tmp_path);clock=[0.]
    def sleep(_):
        clock[0]+=.15
        (tmp_path/'responses/00.started.json').write_text(json.dumps(dict(agent_id='agent',request_id='id',started_monotonic=clock[0])))
        (tmp_path/'responses/00.raw.txt').write_text('{"answer":1}')
        (tmp_path/'responses/00.ready').touch()
    advisor=FileAdvisor(tmp_path,'agent',clock=lambda:clock[0],sleeper=sleep)
    env=advisor.ask('inference',request)
    assert env['actual_llm_request'] and env['timely']
    assert env['raw']=='{"answer":1}' and env['request_id']=='id'
    assert json.loads((tmp_path/'requests/00.json').read_text())['request']==request


def test_forged_receipt_is_not_counted_as_actual_model(tmp_path):
    request=mailbox(tmp_path)
    (tmp_path/'responses/00.started.json').write_text(json.dumps(dict(agent_id='wrong',request_id='id',started_monotonic=0.)))
    env=FileAdvisor(tmp_path,'agent',clock=lambda:0.).ask('inference',request)
    assert not env['started'] and not env['actual_llm_request'] and not env['timely']


def test_unsolicited_raw_does_not_bypass_start_or_dispatch_deadline(tmp_path):
    request=mailbox(tmp_path);clock=[0.]
    (tmp_path/'responses/00.raw.txt').write_text('{}');(tmp_path/'responses/00.ready').touch()
    def sleep(_):clock[0]+=61.
    env=FileAdvisor(tmp_path,'agent',clock=lambda:clock[0],sleeper=sleep).ask('inference',request)
    assert env['error']=='undispatched timeout' and env['raw'] is None and not env['actual_llm_request']


def test_display_mixes_alternatives_in_linear_power_and_does_not_mutate_history():
    models=[dict(id='constant',family='persistence',weight=.25,std_db=0.,bases=[],scored_points=2),
        dict(id='source',family='static',weight=.75,std_db=0.,reference_time_seconds=0.,
             bases=[dict(center_m=[400.,280.],scale_m=40.,velocity_mps=[0.,0.],rate_db_per_second=0.)],
             peaks_dbm=[-60.],scored_points=0)]
    frame=dict(time_seconds=0.,hypotheses=models,latest_power_dbm=-80.)
    ep=dict(frames=[frame],decisions=[],metrics=dict(simulation_seconds=0.))
    before=json.dumps(ep,sort_keys=True)
    xy,arrays,meta,_=frozen_display(ep,'mixture_llm',[800.,600.])
    index=np.flatnonzero(np.all(xy==[400.,280.],axis=1))[0]
    expected=-80.+10.*math.log10(.25+.75*(.01+100.))
    assert math.isclose(arrays['actual_dbm'].ravel()[index],expected,abs_tol=1e-10)
    assert meta[0]['persistence_mass']==.25 and meta[0]['alternatives'][0]['scored_points']==0
    assert json.dumps(ep,sort_keys=True)==before


def test_evaluation_samples_include_real_corners_and_fixed_hover():
    ep=dict(motions=[dict(start_time_seconds=1.,end_time_seconds=3.,traversed_m=[[0.,0.],[10.,0.],[10.,10.]])],
        decisions=[dict(time_seconds=.5,public=dict(request=dict(state=dict(position_m=[0.,0.]))))])
    samples=execution_samples(ep,10.)
    assert [s['position_m'] for s in samples[:2]]==[[5.,0.],[10.,5.]]
    assert samples[-1]['kind']=='fixed_hover' and samples[-1]['dt_seconds']==.5
    assert math.isclose(sum(s['dt_seconds'] for s in samples),2.5)


def test_truth_evaluation_refuses_unfinished_flights(tmp_path,monkeypatch):
    import uav_em.event_demo as module
    (tmp_path/'evaluation').mkdir();(tmp_path/'predictions').mkdir()
    monkeypatch.setattr(module,'check',lambda _: {})
    monkeypatch.setattr(module,'ElectromagneticWorld',lambda *_:pytest.fail('truth before all flights frozen'))
    with pytest.raises(FileNotFoundError):module.finalize(tmp_path)
