"""离线生成的有限表达式算子；手动AST解释，不用eval/exec或任意Python。"""
from __future__ import annotations
import ast
from copy import deepcopy
import math

VERSION='structural_expressions_v2'
FEATURES={
 'entry_score':{'index','fraction','supported','power_density','uncertainty','age','seconds','detour'},
 'entry_offset':{'index','count'},
 'exit_score':{'end_index','fraction','span_seconds','chord_seconds','chord_free','chord_C','base_C','supported','unknown_seconds'},
 'priority':{'attempts','successes','best_candidate_C','best_C','initial_C','gain','last_gain','edges_spent','samples_spent','chord_free','chord_C','chord_seconds','span_seconds','supported','unknown_seconds','batch','arms'},
 'sample_quota':{'samples_left','edges_left','batches_left','batch','attempts','edges_per_sample'},
 'edge_quota':{'samples_left','edges_left','batches_left','batch','attempts','edges_per_sample'},
 'stop':{'batch','arms','best_C','initial_C','gain','samples_left','edges_left','batches_left'},
 'operation':{'chord_free','chord_C','best_C','attempts','successes','batch'},
}
FUNCS={'min':min,'max':max,'abs':abs,'sqrt':math.sqrt,'log1p':math.log1p}
BIN={ast.Add:lambda a,b:a+b,ast.Sub:lambda a,b:a-b,ast.Mult:lambda a,b:a*b,ast.Div:lambda a,b:a/b}
CMP={ast.Lt:lambda a,b:a<b,ast.LtE:lambda a,b:a<=b,ast.Gt:lambda a,b:a>b,ast.GtE:lambda a,b:a>=b,ast.Eq:lambda a,b:a==b,ast.NotEq:lambda a,b:a!=b}

def parse_expression(source,names):
 if not isinstance(source,str) or not 1<=len(source)<=1200:raise ValueError('short expression required')
 try:tree=ast.parse(source,mode='eval')
 except (SyntaxError,RecursionError) as exc:raise ValueError('invalid expression syntax') from exc
 nodes=list(ast.walk(tree))
 if len(nodes)>160:raise ValueError('expression node limit')
 def check(node,depth=0):
  if depth>24:raise ValueError('expression depth limit')
  if isinstance(node,ast.Constant):
   if type(node.value) not in (int,float,bool) or not math.isfinite(node.value) or abs(node.value)>1e6:raise ValueError('finite bounded numeric constants only')
  elif isinstance(node,ast.Name):
   if node.id not in names:raise ValueError('unknown feature '+node.id)
  elif isinstance(node,ast.BinOp) and type(node.op) in BIN:check(node.left,depth+1);check(node.right,depth+1)
  elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub,ast.Not)):check(node.operand,depth+1)
  elif isinstance(node,ast.BoolOp) and isinstance(node.op,(ast.And,ast.Or)):
   for child in node.values:check(child,depth+1)
  elif isinstance(node,ast.Compare) and all(type(op) in CMP for op in node.ops):
   check(node.left,depth+1)
   for child in node.comparators:check(child,depth+1)
  elif isinstance(node,ast.IfExp):
   for child in (node.test,node.body,node.orelse):check(child,depth+1)
  elif isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in FUNCS and not node.keywords and 1<=len(node.args)<=6:
   if node.func.id in ('abs','sqrt','log1p') and len(node.args)!=1:raise ValueError('unary function arity')
   if node.func.id in ('min','max') and len(node.args)<2:raise ValueError('min/max need at least2 scalar arguments')
   for child in node.args:check(child,depth+1)
  else:raise ValueError('unsupported expression node '+type(node).__name__)
 check(tree.body)
 return tree.body

def interpret(node,features):
 def evaluate(n):
  if isinstance(n,ast.Constant):return n.value
  if isinstance(n,ast.Name):return features[n.id]
  if isinstance(n,ast.BinOp):return BIN[type(n.op)](evaluate(n.left),evaluate(n.right))
  if isinstance(n,ast.UnaryOp):
   v=evaluate(n.operand)
   return not v if isinstance(n.op,ast.Not) else (-v if isinstance(n.op,ast.USub) else v)
  if isinstance(n,ast.IfExp):return evaluate(n.body) if evaluate(n.test) else evaluate(n.orelse)
  if isinstance(n,ast.BoolOp):
   if isinstance(n.op,ast.And):return all(evaluate(x) for x in n.values)
   return any(evaluate(x) for x in n.values)
  if isinstance(n,ast.Compare):
   left=evaluate(n.left)
   for op,right_node in zip(n.ops,n.comparators):
    right=evaluate(right_node)
    if not CMP[type(op)](left,right):return False
    left=right
   return True
  if isinstance(n,ast.Call):return FUNCS[n.func.id](*(evaluate(x) for x in n.args))
  raise ValueError('unreachable AST node')
 try:value=evaluate(node)
 except (ArithmeticError,ValueError,TypeError,KeyError,OverflowError) as exc:raise ValueError('expression arithmetic error: '+str(exc)) from exc
 if type(value) not in (int,float,bool) or not math.isfinite(value) or abs(value)>1e15:raise ValueError('expression result must be finite scalar')
 return value

def validate_policy(policy):
 if not isinstance(policy,dict) or set(policy)!={'version','name','max_arms',*FEATURES}:raise ValueError('policy fields invalid')
 if policy['version']!=VERSION or not isinstance(policy['name'],str) or not 1<=len(policy['name'])<=100:raise ValueError('invalid version/name')
 if type(policy['max_arms']) is not int or not 1<=policy['max_arms']<=3:raise ValueError('max_arms in1..3 required')
 for key,names in FEATURES.items():parse_expression(policy[key],names)
 return deepcopy(policy)

class ExpressionPolicy:
 def __init__(self,policy):
  self.program=validate_policy(policy);self.trees={k:parse_expression(self.program[k],names) for k,names in FEATURES.items()}
 def value(self,key,features):
  if set(features)!=FEATURES[key]:raise ValueError('exact public feature schema required: '+key)
  value=interpret(self.trees[key],features)
  if key=='stop':
   if type(value) is not bool:raise ValueError('stop must return bool')
  elif type(value) is bool:raise ValueError('numeric expression returned bool')
  return value
