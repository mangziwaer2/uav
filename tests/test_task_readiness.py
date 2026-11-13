from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math
import numpy as np
import pytest
from uav_em.config import load_config
from uav_em.environment import ElectromagneticWorld
from uav_em.belief import ElectromagneticBelief
from uav_em.task_readiness import (gaussian_positive_part,expected_clipped_excess,score_route,
                                   choose_from_pool,public_risk,SCORE_NAMES)


def test_gaussian_positive_part_matches_closed_form_at_threshold_and_deterministic_limit():
    assert gaussian_positive_part(-80,10,-80)==pytest.approx(10/math.sqrt(2*math.pi))
    np.testing.assert_allclose(gaussian_positive_part([-100,-70],0,-80),[0,10])
    np.testing.assert_allclose(expected_clipped_excess([-100,-70,20],0,-80),[0,1,4])

def test_expected_clipped_excess_matches_independent_numerical_normal_integral():
    x=np.linspace(-10,10,100001);pdf=np.exp(-x*x/2)/np.sqrt(2*np.pi)
    for mean,std in [(-100,15),(-80,1),(-60,15),(-30,20)]:
        integrand=np.clip((mean+std*x+80)/10,0,4)*pdf
        numeric=np.sum((integrand[:-1]+integrand[1:])*.5*np.diff(x))
        assert expected_clipped_excess(mean,std,-80)==pytest.approx(numeric,abs=1e-7)

@pytest.mark.parametrize('mean,std',[(-80,-1),(float('nan'),1),(-80,float('inf'))])
def test_invalid_distribution_rejected(mean,std):
    with pytest.raises(ValueError):expected_clipped_excess(mean,std,-80)

def test_scoring_is_collinear_split_invariant_and_does_not_mutate_gp():
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/local_power.json')
    cfg=replace(cfg,map=replace(cfg.map,buildings=(),obstacles_m=()))
    world=ElectromagneticWorld(cfg);belief=ElectromagneticBelief(cfg,world.geometry)
    belief.update(world.observe([40,300],0));snapshot=belief.snapshot(0)
    original=snapshot.estimate_dbm.copy();route=[[40,300],[760,300]]
    a=score_route(snapshot,route);b=score_route(snapshot,[[40,300],[140,300],[760,300]])
    for key in a:
        if isinstance(a[key],float):assert a[key]==pytest.approx(b[key],abs=1e-8)
    assert all(a[k]>=a['length_m'] for k in SCORE_NAMES)
    np.testing.assert_array_equal(snapshot.estimate_dbm,original)

def test_pool_selector_is_blind_to_truth_and_excludes_overlong_or_invalid_routes():
    def rec(identifier,j,length,feasible=True):
        return {'id':identifier,'metrics':{'length_m':length},'scores':dict.fromkeys(SCORE_NAMES,j),'feasible':feasible}
    pool=[rec('incumbent',100,100),rec('tie',100,100),rec('short',90,110),
          rec('too_long',0,130),rec('collision',0,110,False)]
    choices=choose_from_pool(pool)['choices'];assert set(choices.values())=={'short'}
    before=deepcopy(pool)
    for r in pool:r['evaluation']={'future_exposure':-999 if r['id']=='too_long' else 999}
    assert choose_from_pool(pool)['choices']==choose_from_pool(before)['choices']
    assert set(choose_from_pool(pool[:2])['choices'].values())=={'incumbent'}

def test_risk_labels_use_latest_and_recent_measurement_separately():
    ctx={'time_seconds':10,'electromagnetic_estimate':{'interference_reference_dbm':-80},
         'measurement_evidence':{'recent_points':[{'time_seconds':0,'interference_dbm':-70},
                                                  {'time_seconds':10,'interference_dbm':-90}]}}
    r=public_risk(ctx,{'interference_cost':0})
    assert r['recent_observed_above_reference'] and not r['latest_measurement_above_reference']
    assert not r['supported_remaining_route_penalty_positive']
    ctx['measurement_evidence']['recent_points'][1]['time_seconds']=11
    with pytest.raises(ValueError):public_risk(ctx)

def test_generated_numpy_start_goal_are_saved_without_changing_config(monkeypatch,tmp_path):
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/'tools'))
    from run_task_readiness import native_config
    import json
    cfg=load_config(root/'configs/local_power.json')
    cfg=replace(cfg,map=replace(cfg.map,start_m=(np.int64(40),np.int64(300))))
    value=native_config(cfg);path=tmp_path/'scene.json';path.write_text(json.dumps(value))
    restored=load_config(path)
    np.testing.assert_array_equal(restored.map.start_m,cfg.map.start_m)
    assert native_config(restored)==value
