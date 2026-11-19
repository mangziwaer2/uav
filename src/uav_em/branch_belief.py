"""观测证据驱动的分支库；无地图真值、路径代价或模型调用。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np


def logsumexp(values):
    x=np.asarray(values,float)
    if np.any(np.isnan(x)) or np.any(np.isposinf(x)):raise ValueError('invalid log mass')
    peak=float(x.max())
    if peak==-math.inf:return peak
    return peak+math.log(float(np.exp(x-peak).sum()))


def normalized(log_weights):
    z=logsumexp(log_weights)
    if not math.isfinite(z):raise ValueError('all branches have zero mass')
    return np.asarray(log_weights,float)-z


def gaussian_logpdf(value,mean,std):
    if not np.all(np.isfinite([value,mean,std])) or std<=0:raise ValueError('finite mean/value and positive std required')
    z=(value-mean)/std
    return -.5*z*z-math.log(std)-.5*math.log(2*math.pi)


@dataclass(frozen=True)
class BranchSpec:
    id: str
    parent_id: str
    model: str
    born_event: int = -1


def split_log_mass(masses,parent,children,conditional):
    """将父质量分给子、移除父，不能用多生孩子增加概率；无历史拟合奖励。"""
    if parent not in masses or len(children)<2 or len(set(children))!=len(children):raise ValueError('valid unique split required')
    if any(c in masses for c in children):raise ValueError('child already exists')
    p=np.asarray(conditional,float)
    if p.shape!=(len(children),) or not np.all(np.isfinite(p)) or np.any(p<=0) or not np.isclose(p.sum(),1.,rtol=0,atol=1e-12):
        raise ValueError('positive conditional probabilities summing to one required')
    out=dict(masses);old=out.pop(parent)
    out.update({c:old+math.log(v) for c,v in zip(children,p)})
    assert abs(logsumexp(list(masses.values()))-logsumexp(list(out.values())))<1e-10
    return out


class BranchBank:
    """每事件先发预测，再接读数；活动集只供下游，不限制全池维护。"""
    def __init__(self,specs,mode='switching_bank',unknown_id='U'):
        self.specs=tuple(specs);specs=self.specs;self.ids=tuple(s.id for s in specs);self.unknown_id=unknown_id
        if mode not in ('full_bayes','switching_bank','irreversible_top2'):raise ValueError('unknown bank mode')
        if len(set(self.ids))!=len(self.ids) or unknown_id not in self.ids or len(specs)<3:raise ValueError('unique branch IDs and unknown required')
        if any(not s.id or not s.parent_id or s.id==s.parent_id or s.born_event!=-1 for s in specs):
            raise ValueError('this fixed-bank contract only accepts preexisting leaf hypotheses')
        self.mode=mode;self.log_weights=np.full(len(specs),-math.log(len(specs)))
        self.initial_log_prior=self.log_weights.copy();self.alive=np.ones(len(specs),bool)
        self.last_time=None;self.last_event=-1;self.pending=None;self.count=0;self.scored_components=0

    def active(self,log_weights):
        # A stable ID order breaks equal-weight ties; the unknown remains separately visible.
        valid=[i for i,b in enumerate(self.ids) if b!=self.unknown_id and self.alive[i]]
        chosen=sorted(valid,key=lambda i:(-log_weights[i],i))[:2]
        return [self.ids[i] for i in chosen]

    def issue(self,event_id,time_seconds,predictions):
        if self.pending is not None:raise ValueError('must consume pending observation first')
        if type(event_id) is not int or event_id<=self.last_event or not math.isfinite(time_seconds) or time_seconds<0 or (self.last_time is not None and time_seconds<self.last_time):
            raise ValueError('duplicate, unordered or invalid event')
        if set(predictions)!=set(self.ids):raise ValueError('same complete branch predictions required')
        for p in predictions.values():
            if set(p)!= {'mean_dbm','std_db'}:raise ValueError('only predictive Gaussian parameters accepted, no path score')
            gaussian_logpdf(p['mean_dbm'],p['mean_dbm'],p['std_db'])
        prior=self.log_weights.copy();dt=0. if self.last_time is None else time_seconds-self.last_time
        if self.mode!='full_bayes' and dt>0:
            alpha=-math.expm1(-dt/20.)
            reset=normalized(np.where(self.alive,self.initial_log_prior,-math.inf))
            prior=np.logaddexp(math.log1p(-alpha)+prior,math.log(alpha)+reset) if alpha<1 else reset
        prior=normalized(np.where(self.alive,prior,-math.inf))
        self.pending=dict(event_id=event_id,time_seconds=float(time_seconds),predictions=deepcopy(predictions),
            log_prior=prior.tolist(),prior_weights=np.exp(prior).tolist(),planning_active=self.active(prior),
            unknown_id=self.unknown_id,alive_before=[b for b,a in zip(self.ids,self.alive) if a],
            past_observations=self.count,meaning='relative_evidence_under_declared_model_set_not_truth_probability')
        return deepcopy(self.pending)

    def observe(self,event_id,value):
        if self.pending is None or event_id!=self.pending['event_id']:raise ValueError('observation must match issued event exactly once')
        if not math.isfinite(value):raise ValueError('finite observed power required')
        issued=self.pending
        likelihood=np.array([gaussian_logpdf(value,issued['predictions'][b]['mean_dbm'],issued['predictions'][b]['std_db']) if a else 0.
                             for b,a in zip(self.ids,self.alive)])
        prior=np.array(issued['log_prior']);predictive=logsumexp(prior+likelihood)
        posterior=normalized(prior+likelihood);all_evidence_posterior=posterior.copy()
        self.count+=1;removed=[]
        if self.mode=='irreversible_top2' and self.count%8==0:
            keep=set(self.active(posterior)+[self.unknown_id])
            for i,b in enumerate(self.ids):
                if self.alive[i] and b not in keep:self.alive[i]=False;removed.append(b)
            posterior=normalized(np.where(self.alive,posterior,-math.inf))
        self.log_weights=posterior;self.last_time=issued['time_seconds'];self.last_event=event_id
        self.scored_components+=int(np.sum(np.isfinite(prior)))
        self.pending=None
        return dict(event_id=event_id,time_seconds=self.last_time,observed_dbm=float(value),
            issued_sha256=hashlib.sha256(json.dumps(serializable_issue(issued),sort_keys=True,allow_nan=False).encode()).hexdigest(),
            nll=float(-predictive),branch_nll={b:float(-p) if math.isfinite(prior[i]) else None for i,(b,p) in enumerate(zip(self.ids,likelihood))},
            posterior_weights=dict(zip(self.ids,np.exp(posterior).tolist())),
            all_evidence_posterior_weights=dict(zip(self.ids,np.exp(all_evidence_posterior).tolist())),
            # JSON stores impossible masses as null; finite tiny logs remain finite and recoverable.
            posterior_log_weights={b:float(v) if math.isfinite(v) else None for b,v in zip(self.ids,posterior)},
            planning_active_after=self.active(posterior),permanently_removed=removed,
            alive_after=[b for b,a in zip(self.ids,self.alive) if a],scored_components=self.scored_components,
            top_branch=self.ids[int(np.argmax(posterior))])


def serializable_issue(issued):
    """−inf只用于负对照永久删除；数据文件用null表示，避免非法JSON。"""
    out=deepcopy(issued)
    out['log_prior']=[v if math.isfinite(v) else None for v in out['log_prior']]
    return out
