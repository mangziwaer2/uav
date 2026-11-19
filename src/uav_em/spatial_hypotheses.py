"""仅用公开传播先验与已收标量点测构造有限空间场景。无world/真源接口。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .branch_belief import logsumexp, normalized
from .propagation import PropagationModel, propagation_received_dbm

DB_TO_LOG=math.log(10)/10
MODES={'stationary':(0.,0.,0.),'east':(3.,0.,0.),'west':(-3.,0.,0.),
       'north':(0.,3.,0.),'south':(0.,-3.,0.),'rising_power':(0.,0.,.5),'falling_power':(0.,0.,-.5)}
SIGMA_DB=math.sqrt(17.)


@dataclass(frozen=True)
class SpatialContext:
    propagation: PropagationModel
    width_m: float
    height_m: float
    background_dbm: float = -100.

    def __post_init__(self):
        if not np.all(np.isfinite([self.width_m,self.height_m,self.background_dbm])) or min(self.width_m,self.height_m)<=40:
            raise ValueError('finite public context, both map dimensions > 40m required')


def validate_points(points,now):
    if not math.isfinite(now) or now<0 or not points:raise ValueError('nonempty causal prefix required')
    last=-1.
    for p in points:
        if set(p)!= {'position_m','time_seconds','interference_dbm'} or len(p['position_m'])!=2 or not np.all(np.isfinite([*p['position_m'],p['time_seconds'],p['interference_dbm']])) or not last<=p['time_seconds']<=now:
            raise ValueError('finite ordered received scalar observations only')
        last=p['time_seconds']


def total_dbm(background,gain_dbm,eirp_dbm):
    return np.logaddexp(np.asarray(background)*DB_TO_LOG,(np.asarray(gain_dbm)+eirp_dbm)*DB_TO_LOG)/DB_TO_LOG


def source_centers(source,times,birth_time):
    return np.asarray(source['anchor_m'])[None,:]+(np.asarray(times)-birth_time)[:,None]*np.asarray(source['velocity_mps'])[None,:]


def unit_gains(context,sources,xy,times,birth_time):
    """Q×S dBm transfers at 0dBm EIRP, with the declared power slope included."""
    xy=np.asarray(xy,float);times=np.asarray(times,float);out=np.zeros((len(xy),len(sources)))
    for height in sorted({s['height_m'] for s in sources}):
        indices=[i for i,s in enumerate(sources) if s['height_m']==height]
        centers=np.stack([source_centers(sources[i],times,birth_time) for i in indices],axis=1)
        receivers=np.broadcast_to(xy[:,None,:],centers.shape)
        value=propagation_received_dbm(receivers,centers,dict(height_m=height,eirp_dbm=0.),context.propagation)
        slopes=np.array([sources[i]['power_slope_db_s'] for i in indices])
        out[:,indices]=value+(times-birth_time)[:,None]*slopes[None,:]
    return out


def optimize_power(gain,y,background):
    """14 fixed vectorized golden-section steps, independent of unknown future data."""
    gain=np.asarray(gain,float);y=np.asarray(y,float)[:,None]
    lo=np.zeros(gain.shape[1]);hi=np.full(gain.shape[1],50.);ratio=(math.sqrt(5)-1)/2
    def loss(power):return np.mean((total_dbm(background,gain,power[None,:])-y)**2,axis=0)
    for _ in range(14):
        a=hi-ratio*(hi-lo);b=lo+ratio*(hi-lo);left=loss(a)<=loss(b)
        hi=np.where(left,b,hi);lo=np.where(left,lo,a)
    power=(lo+hi)/2
    return power,loss(power)


def valid_motion(context,source,start_time,birth_time):
    centers=source_centers(source,[start_time,birth_time+10],birth_time)
    if np.any(centers<0) or np.any(centers[:,0]>context.width_m) or np.any(centers[:,1]>context.height_m):return False
    if not np.any(np.asarray(source['velocity_mps'])!=0):return True
    for building in context.propagation.propagation_buildings:
        if building.base_m<=source['height_m']<=building.base_m+building.height_m and bool(building.segment_blocked(centers[0],centers[1])):return False
    return True


def _region(context,anchor):
    col=min(2,int(anchor[0]/context.width_m*3));row=min(1,int(anchor[1]/context.height_m*2))
    return f'R{row}{col}'


class SpatialHypothesisBank:
    def __init__(self,context,state):
        self.context=context;self.state=deepcopy(state)
        if not self.state['components'] or len(self.state['components'])!=len(self.state['log_weights']):raise ValueError('invalid bank')
        normalized(self.state['log_weights'])

    @classmethod
    def fit(cls,context,points,now):
        validate_points(points,now)
        for p in points:
            x,y=p['position_m']
            if not 0<=x<=context.width_m or not 0<=y<=context.height_m:raise ValueError('measurement outside map')
        window=points[-16:];xy=np.array([p['position_m'] for p in window]);times=np.array([p['time_seconds'] for p in window]);y=np.array([p['interference_dbm'] for p in window])
        sources=[];modes=[]
        for col,x in enumerate(np.linspace(20,context.width_m-20,9)):
            for row,z in enumerate(np.linspace(20,context.height_m-20,7)):
                for height in (2.,20.):
                    for mode,(vx,vy,slope) in MODES.items():
                        source=dict(anchor_m=[float(x),float(z)],height_m=height,velocity_mps=[vx,vy],power_slope_db_s=slope)
                        if valid_motion(context,source,times[0],now):sources.append(source);modes.append(mode)
        gains=unit_gains(context,sources,xy,times,now)
        powers,mse=optimize_power(gains,y,context.background_dbm)
        components=[dict(id='background',group_id='background',source_count=0,sources=[],fit_rmse_db=float(np.sqrt(np.mean((y-context.background_dbm)**2))),fitted_amplitudes=0)]
        for i,(source,mode,power,error) in enumerate(zip(sources,modes,powers,mse)):
            source=dict(**source,eirp_dbm=float(power),mode=mode,region_id=_region(context,source['anchor_m']))
            components.append(dict(id=f'single_{i:04}',group_id='S_'+source['region_id']+'_'+mode,source_count=1,
                sources=[source],fit_rmse_db=math.sqrt(float(error)),fitted_amplitudes=1))
        chosen=[]
        for index in sorted(range(len(sources)),key=lambda i:(mse[i],i)):
            if all(np.linalg.norm(np.array(sources[index]['anchor_m'])-sources[j]['anchor_m'])>=60 for j in chosen):chosen.append(index)
            if len(chosen)==8:break
        pair_indices=[(a,b) for j,a in enumerate(chosen) for b in chosen[j+1:]]
        for j,(a,b) in enumerate(pair_indices):
            p1,p2=float(powers[a]),float(powers[b]);g1,g2=gains[:,a:a+1],gains[:,b:b+1]
            for _ in range(4):
                p1=float(optimize_power(g1,y,total_dbm(context.background_dbm,g2,p2))[0][0])
                p2=float(optimize_power(g2,y,total_dbm(context.background_dbm,g1,p1))[0][0])
            fitted=total_dbm(total_dbm(context.background_dbm,g1[:,0],p1),g2[:,0],p2)
            ss=[deepcopy(components[a+1]['sources'][0]),deepcopy(components[b+1]['sources'][0])]
            ss[0]['eirp_dbm']=p1;ss[1]['eirp_dbm']=p2
            components.append(dict(id=f'pair_{j:02}',group_id=f'P_{j:02}',source_count=2,sources=ss,
                fit_rmse_db=math.sqrt(float(np.mean((fitted-y)**2))),fitted_amplitudes=2))
        counts={k:sum(c['source_count']==k for c in components) for k in (0,1,2)};families=sum(v>0 for v in counts.values())
        logs=[-len(window)*c['fit_rmse_db']**2/(2*SIGMA_DB**2)-.5*c['fitted_amplitudes']*math.log(len(window))
              -math.log(families)-math.log(counts[c['source_count']]) for c in components]
        state=dict(birth_time_seconds=float(now),received_count=len(points),fitted_count=len(window),horizon_seconds=10.,
            components=components,log_weights=normalized(logs).tolist(),sigma_db=SIGMA_DB,
            candidate_counts=counts,power_optimization_steps=14,pair_coordinate_rounds=4,
            relative_evidence_not_calibrated_world_probabilities=True,fit_history_not_predictive_score=True)
        return cls(context,state)

    def _queries(self,xy,times):
        xy=np.asarray(xy,float);times=np.asarray(times,float)
        birth=self.state['birth_time_seconds']
        if xy.ndim!=2 or xy.shape[1]!=2 or times.shape!=(len(xy),) or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(times)) or np.any(times<birth) or np.any(times>birth+10):raise ValueError('finite current/future queries within 10s required')
        if np.any(xy<0) or np.any(xy[:,0]>self.context.width_m) or np.any(xy[:,1]>self.context.height_m):raise ValueError('query outside map')
        return xy,times

    def predict_components(self,xy,times,indices=None):
        xy,times=self._queries(xy,times);components=self.state['components']
        indices=list(range(len(components))) if indices is None else list(indices)
        if not indices or any(type(i) is not int or not 0<=i<len(components) for i in indices):raise ValueError('valid component indices required')
        power=np.full((len(xy),len(indices)),10**(self.context.background_dbm/10))
        for slot in (0,1):
            columns=[j for j,i in enumerate(indices) if len(components[i]['sources'])>slot]
            if not columns:continue
            sources=[components[indices[j]]['sources'][slot] for j in columns]
            gains=unit_gains(self.context,sources,xy,times,self.state['birth_time_seconds'])
            eirp=np.array([s['eirp_dbm'] for s in sources])
            power[:,columns]+=10**((gains+eirp[None,:])/10)
        return 10*np.log10(power)

    def groups(self):
        result=[];weights=np.exp(self.state['log_weights'])
        for group in sorted({c['group_id'] for c in self.state['components']}):
            indices=[i for i,c in enumerate(self.state['components']) if c['group_id']==group]
            log_mass=logsumexp([self.state['log_weights'][i] for i in indices]);w=np.exp(normalized([self.state['log_weights'][i] for i in indices]))
            components=[self.state['components'][i] for i in indices];source_summary=[]
            for slot in range(components[0]['source_count']):
                anchors=np.array([c['sources'][slot]['anchor_m'] for c in components]);mu=w@anchors;centered=anchors-mu
                source_summary.append(dict(region_ids=sorted({c['sources'][slot]['region_id'] for c in components}),
                    modes=sorted({c['sources'][slot]['mode'] for c in components}),anchor_mean_m=mu.tolist(),
                    anchor_covariance_m2=((centered*w[:,None]).T@centered).tolist(),
                    mean_is_hypothesis_summary_not_measured_location=True))
            result.append(dict(group_id=group,source_count=components[0]['source_count'],component_indices=indices,
                component_count=len(indices),relative_mass=math.exp(log_mass),log_relative_mass=log_mass,
                mean_fit_rmse_db=float(w@np.array([c['fit_rmse_db'] for c in components])),sources=source_summary))
        return sorted(result,key=lambda r:(-r['log_relative_mass'],r['group_id']))

    def compile(self,declaration,xy,times):
        if not isinstance(declaration,dict) or set(declaration)!= {'group_id'}:raise ValueError('only a declared group ID accepted; no arbitrary code')
        groups={g['group_id']:g for g in self.groups()}
        if declaration['group_id'] not in groups:raise ValueError('unknown hypothesis group')
        group=groups[declaration['group_id']];indices=group['component_indices']
        means=self.predict_components(xy,times,indices)
        weights=np.exp(normalized([self.state['log_weights'][i] for i in indices]))
        mean=means@weights;variance=(means*means)@weights-mean*mean+self.state['sigma_db']**2
        return dict(group_id=group['group_id'],component_indices=indices,component_mean_dbm=means.tolist(),
            component_std_db=self.state['sigma_db'],weights=weights.tolist(),mean_dbm=mean.tolist(),
            std_db=np.sqrt(np.maximum(variance,0)).tolist(),hypothesis_not_measurement=True)
