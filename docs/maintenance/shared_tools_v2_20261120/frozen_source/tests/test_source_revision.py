from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from uav_em.event_inference import digest
from uav_em.event_llm_inference import family_evidence
from uav_em.scenario_planning import ScenarioForecast
from uav_em.source_revision import RevisionSettings, SourceOnlyBank, SourceRenewalBank, SourceRevisionTrigger


def point(t, power=-90.):
    return dict(position_m=[40.+10*t, 100.], time_seconds=float(t), interference_dbm=power)


def source(name='s', time=1., index=1):
    return dict(id=name, family='static', std_db=4., born_index=index, scored_points=0,
                reference_time_seconds=time, bases=[dict(center_m=[160., 100.], scale_m=80.,
                velocity_mps=[0., 0.], rate_db_per_second=0.)], peaks_dbm=[-75.])


def advance(bank, t, value=-90.):
    q = point(t, value)
    issued = bank.issue(q['position_m'], q['time_seconds'])
    return issued, bank.observe(q)


def test_unknown_placeholder_does_not_extrapolate_latest_measured_power():
    predictions = []
    for power in (-100., -60.):
        bank = SourceOnlyBank(point(0, power))
        issued = bank.issue([700., 500.], 1.)
        predictions.append(issued['predictions'][0]['mean_dbm'])
        assert issued['predictions'][0]['std_db'] == 15.
        f = ScenarioForecast(bank.history, bank.snapshot(), 0.)
        d = f.predict([[700., 500.]], [0.])
        assert d['means'][0, 0] == -100.
    assert predictions == [-100., -100.]


def test_first_source_birth_removes_unknown_without_reusing_fitting_evidence():
    bank = SourceOnlyBank(point(0))
    advance(bank, 1)
    bank.birth([source('a'), source('b')])
    assert [m['id'] for m in bank.models] == ['a', 'b']
    assert [m['weight'] for m in bank.snapshot()] == [.5, .5]
    assert all(m['scored_points'] == 0 for m in bank.models)
    advance(bank, 2)
    assert all(m['scored_points'] == 1 for m in bank.models)
    bank.birth([source('c', 2., 2)])
    assert bank.snapshot()[-1]['weight'] == pytest.approx(.25)
    assert bank.models[-1]['scored_points'] == 0
    assert all(m['family'] != 'persistence' for m in bank.models)


def test_invalid_first_birth_is_atomic_and_does_not_remove_unknown():
    bank = SourceOnlyBank(point(0))
    advance(bank, 1)
    before = bank.snapshot()
    bad = source(); bad['scored_points'] = 10
    with pytest.raises(ValueError):
        bank.birth([source('good'), bad])
    assert bank.snapshot() == before
    bank.issue(point(2)['position_m'], 2.)
    with pytest.raises(ValueError):
        bank.birth([])


def test_retired_unknown_predictions_do_not_become_family_evidence():
    bank = SourceOnlyBank(point(0))
    _, record = advance(bank, 1)
    bank.birth([source()])
    result = family_evidence(bank, [record])
    assert all(row['points'] == 0 for row in result.values())


def forecast(t, *, source_exists=True, errors=None):
    history = [point(t-1), point(t)]
    f = SimpleNamespace(now=float(t), fingerprint=f'f{t}', history=history,
                        models=[source()] if source_exists else [dict(bases=[])])
    f.predict = lambda xy, tt, method: dict(weights=np.ones(1), means=np.full((1, len(tt)), -95.),
        supported=np.ones(len(tt), bool), gp_weight=np.full(len(tt), .5))
    issued = dict(time_seconds=float(t), history_sha256=digest(history[:-1]),
                  position_m=history[-1]['position_m'], predictions=[dict(id=f's{i}',
                  mean_dbm=history[-1]['interference_dbm']-e, std_db=4.) for i, e in enumerate(errors or [])])
    return f, dict(issued=issued, observation=history[-1])


@pytest.mark.parametrize('enabled, expected', [(False, False), (True, True)])
def test_all_source_mismatch_revises_before_eight_second_risk_repeat(enabled, expected):
    trigger = SourceRevisionTrigger(RevisionSettings(enable_mismatch=enabled))
    f, _ = forecast(8, source_exists=False)
    first = trigger.observe(f, [[120., 100.], [700., 100.]], 15., 9)
    assert first['requested'] and first['reasons'] == ['initial_source_fit']
    f, record = forecast(10.5, errors=[9., -10.])
    second = trigger.observe(f, [[145., 100.], [700., 100.]], 15., 11, record)
    assert second['all_source_mismatch'] and second['requested'] is expected
    assert not second['risk_detected']


def test_one_surviving_source_blocks_claim_that_all_models_failed():
    trigger = SourceRevisionTrigger(RevisionSettings(enable_mismatch=True))
    f, record = forecast(12, errors=[12., 3.])
    row = trigger.observe(f, [[160., 100.], [700., 100.]], 15., 13, record)
    assert row['min_abs_source_error_db'] == 3.
    assert not row['requested'] and not row['all_source_mismatch']


def test_revision_uses_one_budget_for_initialization_and_mismatch_and_honors_cooldown():
    trigger = SourceRevisionTrigger(RevisionSettings(enable_mismatch=True, max_requests=2))
    f, _ = forecast(8, source_exists=False)
    assert trigger.observe(f, [[120., 100.], [700., 100.]], 15., 9)['requested']
    f, record = forecast(9, errors=[10.])
    assert trigger.observe(f, [[130., 100.], [700., 100.]], 15., 10, record)['blocker'] == 'minimum_interval'
    f, record = forecast(10, errors=[10.])
    assert trigger.observe(f, [[140., 100.], [700., 100.]], 15., 11, record)['requested']
    f, record = forecast(13, errors=[10.])
    row = trigger.observe(f, [[170., 100.], [700., 100.]], 15., 14, record)
    assert row['blocker'] == 'request_budget_exhausted' and trigger.request_count == 2


def test_mismatch_rejects_incorrect_prefix_before_new_fit():
    trigger = SourceRevisionTrigger(RevisionSettings(enable_mismatch=True))
    f, record = forecast(10, errors=[20.])
    record['issued']['history_sha256'] = digest(f.history)
    with pytest.raises(ValueError, match='causal public prefix'):
        trigger.observe(f, [[140., 100.], [700., 100.]], 15., 11, record)


def initialized_renewal():
    bank=SourceRenewalBank(point(0))
    advance(bank,1)
    bank.birth([source('old')])
    return bank


def test_failed_predictions_remain_active_until_successful_replacement_then_start_fresh():
    bank=initialized_renewal()
    issued,record=advance(bank,2,-30.)
    assert bank.models[0]['id']=='old' and not bank.replacement_events
    old=bank.snapshot()
    bank.birth([source('new_a',2.,2),source('new_b',2.,2)])
    assert [m['id'] for m in bank.models]==['new_a','new_b']
    assert [m['weight'] for m in bank.snapshot()]==[.5,.5]
    assert all(m['scored_points']==0 for m in bank.models)
    archived=bank.replacement_events[0]
    assert archived['retired']==old and archived['failure_record']==record
    assert archived['failures'][0]['absolute_error_db']==abs(-30.-issued['predictions'][0]['mean_dbm'])
    assert archived['old_scores_inherited'] is False
    advance(bank,3)
    assert all(m['scored_points']==1 for m in bank.models)


def test_one_compatible_source_preserves_old_cohort_and_original_birth_mass():
    bank=initialized_renewal()
    issued=bank.issue(point(2)['position_m'],2.)
    bank.observe(point(2,issued['predictions'][0]['mean_dbm']))
    alternative=source('incompatible',2.,2);alternative['peaks_dbm']=[-40.]
    bank.birth([alternative])
    issued=bank.issue(point(3)['position_m'],3.)
    measurement=point(3,issued['predictions'][0]['mean_dbm'])
    bank.observe(measurement)
    bank.birth([source('new',3.,3)])
    assert not bank.replacement_events
    assert {m['id'] for m in bank.models}=={'old','incompatible','new'}
    assert bank.snapshot()[-1]['weight']==pytest.approx(.25)


def test_empty_or_invalid_replacement_cannot_discard_failed_cohort():
    bank=initialized_renewal();advance(bank,2,-30.);before=bank.snapshot()
    bank.birth([])
    assert bank.snapshot()==before and not bank.replacement_events
    bad=source('bad',2.,2);bad['scored_points']=4
    with pytest.raises(ValueError):bank.birth([source('good',2.,2),bad])
    assert bank.snapshot()==before and not bank.replacement_events


def test_replacement_cannot_change_pending_prediction_or_reuse_retired_identity():
    bank=initialized_renewal();advance(bank,2,-30.)
    bank.birth([source('new',2.,2)])
    before=bank.snapshot();bank.issue(point(3)['position_m'],3.)
    with pytest.raises(ValueError):bank.birth([source('other',2.,2)])
    assert bank.snapshot()==before
    bank.observe(point(3,-30.))
    with pytest.raises(ValueError,match='identifiers'):
        bank.birth([source('old',3.,3)])


def test_replacement_rejects_stale_failure_record():
    bank=initialized_renewal();advance(bank,2,-30.)
    bank.latest_record['issued']['history_sha256']=digest(bank.history)
    before=bank.snapshot()
    with pytest.raises(ValueError,match='causal prediction'):
        bank.birth([source('new',2.,2)])
    assert bank.snapshot()==before


@pytest.mark.parametrize('value',[0.,float('inf')])
def test_replacement_threshold_must_be_positive_finite(value):
    with pytest.raises(ValueError):SourceRenewalBank(point(0),mismatch_db=value)
