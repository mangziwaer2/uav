from dataclasses import replace
from pathlib import Path

import pytest

from uav_em.config import load_config
from uav_em.event_demo import save
from uav_em.joint_prediction import JointSettings
from uav_em.scenario_loop import run


def test_finalize_never_queries_truth_before_all_declared_flights_finish(tmp_path,monkeypatch):
    import uav_em.joint_experiment as m
    (tmp_path/'evaluation').mkdir();(tmp_path/'predictions').mkdir()
    monkeypatch.setattr(m,'check',lambda _:dict(methods=['joint_N']))
    monkeypatch.setattr(m,'ElectromagneticWorld',lambda *_:pytest.fail('truth provider before flight freeze'))
    with pytest.raises(FileNotFoundError):m.finalize(tmp_path)


def test_synthetic_study_freezes_predictions_audits_and_evaluates_scalar_times(tmp_path,monkeypatch):
    import uav_em.joint_experiment as m
    c=load_config('configs/local_power.json')
    c=replace(c,map=replace(c.map,buildings=(),obstacles_m=(),start_m=[40.,40.],goal_m=[100.,40.]),
              electromagnetic=replace(c.electromagnetic,fields=()))
    for tree in ('runs','public','reports','predictions','evaluation','replay'):(tmp_path/tree).mkdir()
    protocol=dict(methods=list(m.METHODS),joint_settings=c.belief and JointSettings().__dict__,
        max_flights=8,max_events_per_flight=4,max_actual_model_calls=24,max_bridge_requests=24,max_sensor_points=824,
        max_jobs=72,max_solver_calls=144,max_residual_calls=5760,max_samples=28000,max_edges=400000,
        max_evaluation_points=4000,max_replay_truth_points=86016,max_display_receiver_queries=86016,
        max_monitor_receiver_queries=9000,max_soft_gp_queries=960,max_prequential_queries=816,
        max_planner_receiver_queries=40000000,original_c1_sha256={})
    for method,(underlying,predictor,soft,_,path_llm) in m.METHODS.items():
        ep=run(c,underlying,advisor=(lambda *_:pytest.fail('no warning in constant field')) if path_llm else None,
               timing='forward',joint_settings=JointSettings(),forecast_method=predictor,soft_fit=soft)
        ep['study_method']=method;ep['wall_seconds']=0.
        folder=tmp_path/'runs'/method;folder.mkdir()
        save(folder/'episode.json',ep);save(folder/'status.json',dict(status='completed'))
    monkeypatch.setattr(m,'check',lambda _:protocol)
    monkeypatch.setattr(m,'load_config',lambda _:c)
    monkeypatch.setattr(m,'render',lambda _:None)
    OriginalWorld=m.ElectromagneticWorld
    class World(OriginalWorld):
        def truth(self,xy,t):
            assert (tmp_path/'reports/flight_freeze.json').exists()
            assert (tmp_path/'reports/prediction_freeze.json').exists()
            return super().truth(xy,t)
    monkeypatch.setattr(m,'ElectromagneticWorld',World)
    m.finalize(tmp_path)
    result=m.read(tmp_path/'reports/results.json')
    assert len(result['rows'])==8 and all(r['success'] for r in result['rows'])
    assert m.read(tmp_path/'reports/verification.json')['passed']
    assert all(r['mission_T_plus_Q']>=r['simulation_seconds'] for r in result['rows'])
