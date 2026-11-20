from dataclasses import replace
import json
import pytest
from uav_em.config import load_config
from uav_em.scenario_loop import run


def config():
    c=load_config('configs/local_power.json')
    return replace(c,map=replace(c.map,buildings=(),obstacles_m=(),start_m=[40.,40.],goal_m=[100.,40.]),electromagnetic=replace(c.electromagnetic,fields=()))


def test_point_prediction_precedes_sensor_and_short_flight_reaches_goal(monkeypatch):
    import uav_em.scenario_loop as module
    OriginalBank=module.HypothesisBank;OriginalWorld=module.ElectromagneticWorld;holder={}
    class Bank(OriginalBank):
        def __init__(self,p):super().__init__(p);holder['bank']=self
    class World(OriginalWorld):
        def observe(self,p,t):
            if holder.get('bank') is not None:assert holder['bank'].pending is not None
            return super().observe(p,t)
    monkeypatch.setattr(module,'HypothesisBank',Bank);monkeypatch.setattr(module,'ElectromagneticWorld',World)
    e=run(config(),'persistence_numeric')
    assert e['metrics']['success'] and e['metrics']['executed_length_m']==60.
    assert len(e['prediction_records'])==len(e['observations'])-1
    assert e['metrics']['actual_model_calls']==0
    assert all(d['resources']['samples']<=700 and d['resources']['edges']<=10000 for d in e['decisions'])


def test_illegal_llm_response_is_logged_and_uses_numeric_plan():
    seen=[]
    def advisor(info,i):
        seen.append(info)
        assert 'fields' not in json.dumps(info) and 'eirp' not in json.dumps(info)
        return dict(raw='bad',timely=True,started=True)
    e=run(config(),'mixture_llm',advisor)
    assert e['metrics']['success'] and len(seen)==1
    assert e['metrics']['fallbacks']==1 and e['decisions'][0]['raw_response']=='bad'


def test_power_rise_immediately_enters_inference_then_path_callback(monkeypatch):
    import uav_em.scenario_loop as module
    from uav_em.event_llm_inference import fixed_response
    OriginalWorld=module.ElectromagneticWorld
    calls=[]
    class World(OriginalWorld):
        def observe(self,p,t):
            # Declared synthetic sensor sequence, not a hidden-source benchmark.
            return replace(super().observe(p,t),interference_dbm=-90.+2*min(t,3.))
    monkeypatch.setattr(module,'ElectromagneticWorld',World)
    def inference_advisor(request):
        calls.append(('inference',request['phase']))
        assert request['history'][-1]['time_seconds']==3.
        assert request['trigger_reason']=='observed_power_rise'
        return dict(raw=fixed_response(request),timely=True,actual_llm_request=False)
    def advisor(info,i):
        calls.append(('path',i))
        assert info['consultation_event']['time_seconds']==3.
        assert info['inference_status']=='updated'
        request=info['request']
        return dict(raw=dict(request_id=request['request_id'],action='keep',waypoint_m=None,
                            sample_quotas=[],sampling='uniform',reason='synthetic test'),
                    timely=True,started=False)
    e=run(config(),'mixture_llm',advisor,timing='power_rise',inference_advisor=inference_advisor)
    assert e['metrics']['success'] and e['metrics']['actual_model_calls']==0
    assert [d['time_seconds'] for d in e['decisions']]==[3.]
    assert calls==[('inference','pilot'),('inference','refine'),('path',0)]
    assert len(e['inferences'])==1
    assert e['timing_limitations']['physical_state_frozen_during_callback']
    assert e['metrics']['fallbacks']==0


def test_stable_power_event_flight_makes_no_consultation(monkeypatch):
    import uav_em.scenario_loop as module
    OriginalWorld=module.ElectromagneticWorld
    class World(OriginalWorld):
        def observe(self,p,t):return replace(super().observe(p,t),interference_dbm=-90.)
    monkeypatch.setattr(module,'ElectromagneticWorld',World)
    e=run(config(),'mixture_llm',lambda *_:pytest.fail('no rise: no consultation'),timing='power_rise')
    assert e['metrics']['success'] and e['decisions']==[] and e['inferences']==[]


def test_event_llm_requires_an_advisor():
    with pytest.raises(ValueError,match='explicit advisor'):
        run(config(),'mixture_llm',timing='power_rise')

