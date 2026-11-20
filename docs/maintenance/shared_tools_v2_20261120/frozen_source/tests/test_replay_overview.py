import numpy as np

from uav_em.replay_overview import saved_state,source_maps


def test_display_keeps_trigger_before_and_after_birth_distinct():
    before=[dict(id='old')];after=[dict(id='new')]
    observation=dict(time_seconds=8.,position_m=[20.,30.],interference_dbm=-80.)
    event=dict(time_seconds=8.,bank_after_birth=after)
    decision=dict(time_seconds=8.)
    episode=dict(observations=[dict(time_seconds=0.),observation],
        prediction_records=[dict(posterior=before)],inferences=[event],decisions=[decision])
    models,_,last,pending=saved_state(episode,dict(time_seconds=8.))
    assert models==before and last is None and pending
    models,_,last,pending=saved_state(episode,dict(time_seconds=8.5))
    assert models==after and last==decision and not pending


def test_source_location_view_does_not_remove_constant_term_from_actual_mix():
    models=[dict(id='constant',family='persistence',weight=1.-1e-14,std_db=4.,bases=[]),
        dict(id='weak_source',family='static',weight=1e-14,std_db=4.,reference_time_seconds=0.,
            bases=[dict(center_m=[0.,0.],scale_m=10.,velocity_mps=[0.,0.],rate_db_per_second=0.)],peaks_dbm=[-60.])]
    actual,sources,top,_=source_maps(models,np.array([[0.,0.],[100.,0.]]),0.,-80.)
    assert np.ptp(actual)<1e-8
    assert sources[0][0]-sources[0][1]>30.
    assert top[0]['weight']==1e-14
