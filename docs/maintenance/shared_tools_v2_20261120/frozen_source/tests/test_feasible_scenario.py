from copy import deepcopy
from dataclasses import replace
from uav_em.config import load_config
from uav_em.environment import Geometry
from uav_em.repair_search import WorkBudget
from uav_em.scenario_planning import ScenarioForecast,state_request
from uav_em.feasible_scenario import shortcut,evaluate,best,pool,execute,feedback_plan


def setup(obstacles=()):
    c=load_config('configs/local_power.json');g=Geometry(replace(c.map,buildings=(),obstacles_m=obstacles),0.,c.aircraft.altitude_m)
    h=[dict(position_m=[20.,60.],time_seconds=0.,interference_dbm=-85.)]
    m=dict(id='hot',family='static',std_db=0.,born_index=0,weight=1.,reference_time_seconds=0.,
        bases=[dict(center_m=[100.,60.],scale_m=18.,velocity_mps=[0.,0.],rate_db_per_second=0.)],peaks_dbm=[-65.])
    f=ScenarioForecast(h,[m],0.)
    s=dict(position_m=[20.,60.],goal_m=[180.,60.],route_m=[[20.,60.],[180.,60.]],speed_mps=10.,spent_length_m=0.,maximum_length_m=200.,remaining_energy_wh=100.,flight_power_w=100.,hover_power_w=50.,reserve_energy_wh=1.,delay_seconds=.5)
    return g,f,s


def test_shortcut_removes_excess_vertices_without_forcing_power_penalty():
    g,f,s=setup();w=WorkBudget(80,1200)
    path=[[20.,60.],[30.,80.],[100.,95.],[170.,80.],[180.,60.]]
    short,trace=shortcut(path,g,w)
    assert short==s['route_m'] and trace['checks']<=40
    a=evaluate(path,s,f,g,w,'mixture','raw');b=evaluate(short,s,f,g,w,'mixture','short')
    assert a['feasible'] and a['score']['score_seconds']<b['score']['score_seconds']
    assert best([a,b])['id']=='raw'


def test_shortcut_never_crosses_a_known_building():
    g,f,s=setup(((80.,40.,120.,80.),));w=WorkBudget(80,1200)
    path=[[20.,60.],[70.,90.],[130.,90.],[180.,60.]]
    short,_=shortcut(path,g,w)
    assert all(g.edge_free(a,b) for a,b in zip(short,short[1:]))
    assert len(short)>2


def test_unobstructed_via_avoids_unnecessary_rrt_and_respects_total_cap():
    g,f,s=setup();w=WorkBudget(80,1200);r=state_request(s,f,g,w)
    p=dict(request_id=r['request_id'],action='via',waypoint_m=[100.,95.],sample_quotas=[40,40],sampling='uniform',reason='synthetic')
    x=execute(r,p,s,f,g,w,'mixture',31)
    assert not x['searches'] and w.samples==0
    assert best(x['candidates'])['id']=='proposal_raw'
    s2=deepcopy(s);s2['spent_length_m']=50.;w2=WorkBudget(80,1200);r2=state_request(s2,f,g,w2);p['request_id']=r2['request_id']
    xx=execute(r2,p,s2,f,g,w2,'mixture',31)
    assert not any(c['feasible'] for c in xx['candidates'])


def test_geometric_pool_and_extra_plan_share_paid_ledger():
    g,f,s=setup();w=WorkBudget(700,10000);rows=pool(s,f,g,w,'mixture');paid=w.edges
    selected=best(rows);s['route_m']=selected['path'];r=state_request(s,f,g,w)
    p=feedback_plan(r,selected);x=execute(r,p,s,f,g,w,'mixture',31)
    assert paid>0 and x['resources']['edges']>=paid and w.samples<=700 and w.edges<=10000

