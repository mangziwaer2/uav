from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import pytest
from uav_em.structural_policy import VERSION,ExpressionPolicy,validate_policy,parse_expression,interpret
from uav_em.generated_structure import run_generated,GeneratedOptimizer
from uav_em.balanced_objective import BalancedOptimizer,choose
from uav_em.config import load_config
from uav_em.belief import ElectromagneticBelief
from uav_em.environment import ElectromagneticWorld
from uav_em.context import build_context
from uav_em.decision_objective import ObjectiveSettings

@pytest.fixture
def program():
 return dict(version=VERSION,name='synthetic_test_not_generated',max_arms=3,entry_score='supported * power_density',entry_offset='-1',
  exit_score='span_seconds - chord_seconds',priority='-attempts',sample_quota='samples_left / max(1, batches_left)',
  edge_quota='edges_left / max(1, batches_left)',stop='False',operation='0 if chord_free and chord_C < best_C and attempts == 0 else 1')

@pytest.fixture
def setup():
 cfg=load_config(Path(__file__).resolve().parents[1]/'configs/local_power.json')
 cfg=replace(cfg,map=replace(cfg.map,buildings=(),obstacles_m=()),electromagnetic=replace(cfg.electromagnetic,fields=()))
 world=ElectromagneticWorld(cfg);belief=ElectromagneticBelief(cfg,world.geometry);belief.update(world.observe(cfg.map.start_m,0));s=belief.snapshot(0)
 route=[list(cfg.map.start_m),list(cfg.map.goal_m)]
 return cfg,s,build_context(cfg.map.start_m,cfg.map.goal_m,route,s,cfg,100,[],[],None),route

@pytest.mark.parametrize('expr',['__import__("os")','x.real','x[0]','[x for x in (1,2)]','(lambda: 1)()','2 ** 20','"text"','open(1)','min(x)','sqrt(x,1)','True.__class__'])
def test_unsafe_or_unbounded_expressions_rejected(expr):
 with pytest.raises(ValueError):parse_expression(expr,{'x'})

def test_expression_arithmetic_lazy_branches_and_typed_finite_results():
 assert interpret(parse_expression('max(1, x) + sqrt(4) if x > 0 else 0',{'x'}),{'x':3})==5
 assert interpret(parse_expression('0 if x == 0 else 1 / x',{'x'}),{'x':0})==0
 with pytest.raises(ValueError):interpret(parse_expression('1 / x',{'x'}),{'x':0})
 with pytest.raises(ValueError):interpret(parse_expression('x * x',{'x'}),{'x':1e12})
 with pytest.raises(ValueError):parse_expression('+('*30+'1'+')'*30,set())

@pytest.mark.parametrize('change',[{'max_arms':4},{'max_arms':True},{'entry_score':'seed'},{'version':'wrong'},{'extra':1}])
def test_exact_program_schema_and_feature_allowlist(program,change):
 program.update(change)
 with pytest.raises(ValueError):validate_policy(program)

def test_stop_bool_required_and_missing_feature_rejected(program):
 program['stop']='1';p=ExpressionPolicy(program)
 features={k:1 for k in ('batch','arms','best_C','initial_C','gain','samples_left','edges_left','batches_left')}
 with pytest.raises(ValueError):p.value('stop',features)
 with pytest.raises(ValueError):p.value('priority',{'attempts':1})

def test_generated_options_budget_full_C_monotonicity_and_runtime_errors(setup,program):
 _,s,ctx,route=setup;limits=ObjectiveSettings(search_samples=3,sample_limit=100,edge_limit=2000)
 _,pool=BalancedOptimizer(17,limits=limits).optimize(s,deepcopy(ctx),route,2000,999,0)
 r=run_generated(s,ctx,pool,2000,17,program,limits=limits)
 assert r['policy_errors']==0 and r['actual_advisor_calls']==0 and not r['arbitrary_python_executed']
 assert r['resources']['samples']<=100 and r['resources']['edges']<=2000
 assert r['path'][0]==ctx['vehicle']['position_m'] and r['path'][-1]==ctx['vehicle']['goal_m']
 assert choose(r['candidates'])['balance']==r['selected_balance']
 assert all(0<=a['start_index']<a['end_index']<len(r['frozen_base_route']) for a in r['reconnection_options'])
 assert all(t['search_edges']<=t['edge_quota'] and t['best_C_seconds']<=t['previous_best_C_seconds']+1e-7 for t in r['generated_trace'])
 bad=deepcopy(program);bad['priority']='1 / attempts'
 failed=run_generated(s,ctx,pool,2000,17,bad,limits=limits)
 assert failed['policy_errors']==1 and failed['stop_reason']=='policy_error' and failed['final_geometry_verified']
 assert failed['selected_balance']==choose(pool['candidates'])['balance']

def test_optimizer_preserves_mission_constraint_and_generated_stop(setup,program):
 _,s,ctx,route=setup;program['stop']='True';limits=ObjectiveSettings(search_samples=3,sample_limit=50,edge_limit=1200)
 result,r=GeneratedOptimizer(17,program,limits=limits).optimize(s,ctx,route,2000,999,0)
 assert result.path and result.length_m<=r['mission_distance_cap_m']
 assert r['stop_reason']=='generated_stop' and not r['generated_trace'] and r['policy_errors']==0
 assert r['actual_advisor_calls']==0 and r['resources']['samples']==r['starting_resources_already_paid']['samples']
