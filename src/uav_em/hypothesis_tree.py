"""因果数值规则与证据假设树；不访问场景/路径代价或调用LLM。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from .branch_belief import gaussian_logpdf, logsumexp, normalized

BASES=('B0','L','M','G')
RECIPES=tuple(f'{b}:{op}' for b in BASES for op in ('none','level','trend'))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


class ResidualLibrary:
    """十二规则共同预序缓存。过去拟合仅用于下一条读数，非历史补分。"""
    def __init__(self):
        self.history=[]
        self.pending=None

    def forecast(self,event_id,time_seconds,position_m,bases):
        if self.pending is not None:raise ValueError('consume previous prediction first')
        if set(bases)!=set(BASES) or type(event_id) is not int or not np.all(np.isfinite([time_seconds,*position_m])) or len(position_m)!=2:
            raise ValueError('invalid base forecast/query')
        if self.history and (event_id<=self.history[-1]['event_id'] or time_seconds<self.history[-1]['time_seconds']):
            raise ValueError('unordered query')
        for d in bases.values():
            if set(d)!= {'mean_dbm','std_db'}:raise ValueError('only Gaussian base forecast accepted')
            gaussian_logpdf(d['mean_dbm'],d['mean_dbm'],d['std_db'])
        past=self.history[-8:];out={}
        for b in BASES:
            base=bases[b];res=np.array([r['value']-r['bases'][b]['mean_dbm'] for r in past])
            times=np.array([r['time_seconds'] for r in past]);n=len(past)
            bias=float(np.clip(res.mean(),-8,8)) if n>=2 else 0.
            level_var=float(np.var(res,ddof=1)/n) if n>=2 else 0.
            slope=slope_var=0.
            if n>=2:
                centered=times-times.mean();ss=float(centered@centered)
                if ss>1e-12:
                    slope=float(np.clip(centered@(res-res.mean())/ss,-1,1))
                    fit=res.mean()+slope*centered
                    slope_var=float(np.sum((res-fit)**2)/max(1,n-2)/ss)
            delta=min(8.,max(0.,time_seconds-times[-1])) if n else 0.
            fade=math.exp(-float(np.linalg.norm(np.array(position_m)-past[-1]['position_m']))/80.) if n else 1.
            for op in ('none','level','trend'):
                correction=0. if op=='none' else bias if op=='level' else (bias+slope*delta)*fade
                extra=0. if op=='none' else level_var if op=='level' else (level_var+delta**2*slope_var)*fade**2
                out[f'{b}:{op}']=dict(mean_dbm=float(base['mean_dbm']+correction),std_db=math.sqrt(base['std_db']**2+extra))
        out['U']=dict(mean_dbm=-100.,std_db=15.)
        self.pending=dict(event_id=event_id,time_seconds=float(time_seconds),position_m=list(position_m),bases=deepcopy(bases),predictions=deepcopy(out),past_count=len(self.history))
        return deepcopy(out)

    def observe(self,event_id,value):
        if self.pending is None or self.pending['event_id']!=event_id or not math.isfinite(value):raise ValueError('matching finite observation required')
        row=deepcopy(self.pending);row['value']=float(value)
        row['nll']={r:-gaussian_logpdf(value,d['mean_dbm'],d['std_db']) for r,d in row.pop('predictions').items()}
        self.history.append(row);self.pending=None

    def summary(self):
        if not self.history:return []
        return [dict(recipe=r,count=min(8,len(self.history)),past_prequential_nll=float(np.mean([h['nll'][r] for h in self.history[-8:]]))) for r in RECIPES]


class HypothesisTree:
    """只在叶上维护质量；每次split均守恒，祖先质量由后代求和。"""
    def __init__(self,recipes,born_count):
        if len(recipes)!=3 or len(set(recipes))!=3 or any(r not in RECIPES for r in recipes):raise ValueError('three unique known recipes required')
        if type(born_count) is not int or born_count<0:raise ValueError('invalid birth count')
        self.nodes={'A':dict(id='A',parent_id=None,children=['B','C','D','U'],recipe=None,born_count=born_count,scored_count=0)}
        for node,recipe in zip(('B','C','D','U'),[*recipes,'U']):
            self.nodes[node]=dict(id=node,parent_id='A',children=[],recipe=recipe,born_count=born_count,scored_count=0)
        self.log_weights={b:-math.log(4) for b in ('B','C','D','U')}
        self.log_reference=dict(self.log_weights);self.count=born_count;self.last_time=None;self.last_event=-1
        self.pending=None;self.scored_components=0
        self._check()

    def _check(self):
        leaves={k for k,n in self.nodes.items() if not n['children']}
        if leaves!=set(self.log_weights) or leaves!=set(self.log_reference) or 'U' not in leaves:raise ValueError('leaf mass mismatch')
        if abs(logsumexp(list(self.log_weights.values())))>1e-9 or abs(logsumexp(list(self.log_reference.values())))>1e-9:raise ValueError('mass not normalized')
        recipes=[self.nodes[k]['recipe'] for k in leaves]
        if len(recipes)!=len(set(recipes)) or any(r not in (*RECIPES,'U') for r in recipes):raise ValueError('duplicate/unknown leaf recipe')
        for key,n in self.nodes.items():
            if key!=n['id'] or (key!='A' and (n['parent_id'] not in self.nodes or key not in self.nodes[n['parent_id']]['children'])):raise ValueError('invalid ancestry')
            if any(c not in self.nodes or self.nodes[c]['parent_id']!=key for c in n['children']):raise ValueError('invalid child')
        if len(self.nodes)>7 or len(leaves)>5:raise ValueError('tree budget exceeded')

    def active(self,log_weights=None):
        weights=self.log_weights if log_weights is None else log_weights
        return sorted((k for k in weights if k!='U'),key=lambda k:(-weights[k],k))[:2]

    def split(self,parent,new_recipe):
        if self.pending is not None:raise ValueError('cannot change issued prediction')
        if parent not in self.log_weights or parent=='U' or new_recipe not in RECIPES:raise ValueError('valid concrete leaf and recipe required')
        if len(self.nodes)+2>7 or new_recipe in [self.nodes[k]['recipe'] for k in self.log_weights]:raise ValueError('budget or repeated recipe')
        before=self.snapshot();old=self.nodes[parent]['recipe'];children=[parent+'.0',parent+'.1']
        self.nodes[parent]['children']=children
        for child,recipe in zip(children,(old,new_recipe)):
            self.nodes[child]=dict(id=child,parent_id=parent,children=[],recipe=recipe,born_count=self.count,scored_count=0)
        for mass in (self.log_weights,self.log_reference):
            value=mass.pop(parent)
            mass.update({c:value-math.log(2) for c in children})
        self._check()
        return dict(parent_id=parent,children=children,before=before,after=self.snapshot(),conditional=[.5,.5],old_scores_not_inherited=True)

    def issue(self,event_id,time_seconds,predictions):
        if self.pending is not None or type(event_id) is not int or event_id<=self.last_event or not math.isfinite(time_seconds) or time_seconds<0 or (self.last_time is not None and time_seconds<self.last_time):raise ValueError('unordered/unconsumed event')
        if set(predictions)!=set(self.log_weights):raise ValueError('exact leaf predictions required')
        for d in predictions.values():
            if set(d)!= {'mean_dbm','std_db'}:raise ValueError('only Gaussian predictions')
            gaussian_logpdf(d['mean_dbm'],d['mean_dbm'],d['std_db'])
        keys=list(self.log_weights);prior=np.array([self.log_weights[k] for k in keys]);dt=0 if self.last_time is None else time_seconds-self.last_time
        if dt>0:
            alpha=-math.expm1(-dt/20)
            ref=np.array([self.log_reference[k] for k in keys])
            prior=np.logaddexp(math.log1p(-alpha)+prior,math.log(alpha)+ref) if alpha<1 else ref
        prior=normalized(prior);prior=dict(zip(keys,map(float,prior)))
        self.pending=dict(event_id=event_id,time_seconds=float(time_seconds),past_count=self.count,predictions=deepcopy(predictions),
            log_prior=prior,prior_weights={k:math.exp(v) for k,v in prior.items()},active=self.active(prior),
            leaf_recipes={k:self.nodes[k]['recipe'] for k in keys},node_count=len(self.nodes))
        return deepcopy(self.pending)

    def observe(self,event_id,value):
        if self.pending is None or event_id!=self.pending['event_id'] or not math.isfinite(value):raise ValueError('matching finite reading required')
        x=self.pending;keys=list(x['log_prior']);ll={k:gaussian_logpdf(value,d['mean_dbm'],d['std_db']) for k,d in x['predictions'].items()}
        logits=[x['log_prior'][k]+ll[k] for k in keys];nll=-logsumexp(logits)
        self.log_weights=dict(zip(keys,map(float,normalized(logits))))
        self.count+=1;self.scored_components+=len(keys);self.last_time=x['time_seconds'];self.last_event=event_id;self.pending=None
        for k in keys:self.nodes[k]['scored_count']+=1
        self._check()
        return dict(event_id=event_id,value=float(value),issued_sha256=digest(x),nll=nll,branch_nll={k:-v for k,v in ll.items()},tree=self.snapshot())

    def snapshot(self):
        def descendants(key):
            if key in self.log_weights:return [key]
            return [leaf for child in self.nodes[key]['children'] for leaf in descendants(child)]
        nodes=[]
        for key,node in self.nodes.items():
            logmass=logsumexp([self.log_weights[k] for k in descendants(key)])
            nodes.append(dict(**deepcopy(node),log_mass=logmass,mass=math.exp(logmass),is_leaf=key in self.log_weights))
        return dict(nodes=nodes,active=self.active(),leaf_log_weights=deepcopy(self.log_weights),
            leaf_log_reference=deepcopy(self.log_reference),received_count=self.count,last_time=self.last_time,
            last_event=self.last_event,scored_components=self.scored_components,only_leaf_mass_normalized=True)

    @classmethod
    def restore(cls,state):
        obj=cls.__new__(cls);obj.nodes={n['id']:{k:deepcopy(v) for k,v in n.items() if k not in ('mass','log_mass','is_leaf')} for n in state['nodes']}
        obj.log_weights=deepcopy(state['leaf_log_weights']);obj.log_reference=deepcopy(state['leaf_log_reference'])
        obj.count=state['received_count'];obj.last_time=state['last_time'];obj.last_event=state['last_event'];obj.scored_components=state['scored_components'];obj.pending=None
        obj._check();return obj


def parse_proposal(value,request_id,phase,tree=None):
    common={'request_id','action','rationale'}
    expected=common|({'recipes'} if phase=='initialize' else {'parent_id','new_recipe'})
    if not isinstance(value,dict) or set(value)!=expected or value['request_id']!=request_id or not isinstance(value['rationale'],str) or not 1<=len(value['rationale'])<=3000:raise ValueError('response schema/identity')
    if phase=='initialize':
        if value['action']!='initialize':raise ValueError('initialize required')
        HypothesisTree(value['recipes'],12)
    else:
        if value['action']=='keep':
            if value['parent_id'] is not None or value['new_recipe'] is not None:raise ValueError('keep has null operands')
        elif value['action']=='split':
            candidate=HypothesisTree.restore(tree.snapshot());candidate.split(value['parent_id'],value['new_recipe'])
        else:raise ValueError('split or keep required')
    return deepcopy(value)
