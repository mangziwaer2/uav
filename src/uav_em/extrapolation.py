"""因果点测记忆和独立外推；只接收公开几何，不导入传播真值或规划控制。"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .local_estimation import space_time_kernel

BASE_METHODS = ('baseline_gp', 'local_level_gp', 'memory_level_gp', 'geometry_memory_gp')
METHODS = BASE_METHODS + ('causal_mixture', 'calibrated_mixture')


def validate_points(points, geometry, now):
    if not math.isfinite(now) or now < 0:
        raise ValueError('nonnegative finite decision time required')
    last = -1.
    result = []
    for p in points:
        xy, t, y = p['position_m'], p['time_seconds'], p['interference_dbm']
        if len(xy) != 2 or not np.all(np.isfinite([*xy, t, y])) or not 0 <= t <= now or t < last:
            raise ValueError('future, unordered or nonfinite point')
        if not geometry.free(xy):
            raise ValueError('point outside free geometry')
        result.append(dict(position_m=list(xy), time_seconds=float(t), interference_dbm=float(y)))
        last = t
    return result


def training_points(points, geometry, now, settings, memory=False):
    """按原始时序选点；重复同网格同时刻覆盖，跨时重访保留最多2条。"""
    points = validate_points(points, geometry, now)
    if not memory:
        cached = {}
        for p in points:
            if now-p['time_seconds'] <= settings.max_age_seconds:
                cached[geometry.node(p['position_m'])] = p
        return sorted(cached.values(), key=lambda p:p['time_seconds'])[-settings.gp_max_points:]
    by_node = {}
    for p in points:
        if now-p['time_seconds'] > 120.: continue
        key = geometry.node(p['position_m'])
        entries = by_node.setdefault(key, [])
        if entries and entries[-1]['time_seconds'] == p['time_seconds']: entries[-1] = p
        else: entries.append(p)
        by_node[key] = entries[-2:]
    active = sorted((p for group in by_node.values() for p in group), key=lambda p:p['time_seconds'])
    if len(active) <= 128: return active
    recent, older = active[-64:], active[:-64]
    indices = np.unique(np.linspace(0, len(older)-1, 64).astype(int))
    return [older[i] for i in indices] + recent


def geometry_features(xy, geometry):
    """已知楼体包围框侧向嵌入；非源相对遮挡或真实阴影。"""
    xy = np.asarray(xy, float)
    bounds = [(np.min(b.footprint_m, axis=0), np.max(b.footprint_m, axis=0)) for b in geometry.buildings]
    bounds += [(np.asarray(b[:2]), np.asarray(b[2:])) for b in geometry.config.obstacles_m]
    if not bounds: return np.zeros((len(xy), 0))
    cols = []
    for low, high in bounds:
        # Features transition around each bounding wall, not a guessed source.
        for axis in (0, 1):
            for wall in (low[axis], high[axis]):
                cols.append(np.tanh((xy[:, axis]-wall)/geometry.config.resolution_m))
    return np.column_stack(cols)/math.sqrt(len(bounds))


def covariance(x, t, z, u, settings, geometry, mode):
    base = space_time_kernel(x, t, z, u, settings)
    if mode in ('memory_level_gp', 'geometry_memory_gp'):
        distance = np.linalg.norm(np.asarray(x)[:, None, :]-np.asarray(z)[None, :, :], axis=-1)
        r = np.sqrt(3.)*distance/120.
        slow = settings.unknown_error_db**2*(1+r)*np.exp(-r-np.abs(np.asarray(t)[:, None]-np.asarray(u)[None, :])/120.)
        base = .75*base+.25*slow
    if mode == 'geometry_memory_gp':
        first, second = geometry_features(x, geometry), geometry_features(z, geometry)
        d2 = np.sum(first**2, axis=1)[:, None]+np.sum(second**2, axis=1)[None, :]-2*first@second.T
        base *= np.exp(-.5*np.maximum(d2, 0.))
    return base


class LocalPredictor:
    def __init__(self, points, geometry, now, settings, mode, background=-100.):
        if mode not in BASE_METHODS: raise ValueError('unknown predictor')
        self.geometry, self.now, self.settings, self.mode, self.background = geometry, now, settings, mode, background
        self.points = training_points(points, geometry, now, settings, mode in BASE_METHODS[2:])
        self.xy = np.array([p['position_m'] for p in self.points],float).reshape(-1,2)
        self.times = np.array([p['time_seconds'] for p in self.points])
        self.mean_level = background
        if not self.points: return
        values = np.array([p['interference_dbm'] for p in self.points])
        K = covariance(self.xy, self.times, self.xy, self.times, settings, geometry, mode)
        K.flat[::len(K)+1] += settings.measured_error_db**2+1e-8
        self.inverse = np.linalg.solve(K, np.eye(len(K)))
        self.inverse_ones = self.inverse@np.ones(len(K))
        self.level_precision = float(self.inverse_ones.sum())
        if mode != 'baseline_gp':
            self.mean_level = float(self.inverse_ones@values/self.level_precision)
        self.alpha = self.inverse@(values-self.mean_level)

    def predict(self, xy):
        xy = np.asarray(xy,float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.all(np.isfinite(xy)) or np.any(xy<0) or np.any(xy>self.geometry.high):
            raise ValueError('finite Nx2 queries within map required')
        if not self.points:
            return dict(mean=np.full(len(xy),self.background), std=np.full(len(xy),self.settings.unknown_error_db),
                        nearest_m=np.full(len(xy),-1.), age_seconds=np.full(len(xy),-1.))
        out = {k:[] for k in ('mean','std','nearest_m','age_seconds')}
        for start in range(0,len(xy),2048):
            q=xy[start:start+2048]
            cross=covariance(q,np.full(len(q),self.now),self.xy,self.times,self.settings,self.geometry,self.mode)
            mean=self.mean_level+cross@self.alpha
            variance=self.settings.unknown_error_db**2-np.sum((cross@self.inverse)*cross,axis=1)
            if self.mode!='baseline_gp':
                variance+=(1.-cross@self.inverse_ones)**2/self.level_precision
            variance=np.maximum(0.,variance)+self.settings.measured_error_db**2
            distance=np.linalg.norm(q[:,None,:]-self.xy[None,:,:],axis=-1)
            nearest=np.argmin(distance,axis=1)
            for key,value in dict(mean=mean,std=np.sqrt(variance),nearest_m=distance[np.arange(len(q)),nearest],
                                  age_seconds=self.now-self.times[nearest]).items():out[key].append(value)
        return {k:np.concatenate(v) if v else np.array([]) for k,v in out.items()}


def mixture_logpdf(y, means, stds, weights):
    means,stds,weights=map(lambda a:np.asarray(a,float),(means,stds,weights))
    if np.any(stds<=0) or np.any(weights<0) or not np.isclose(weights.sum(),1):raise ValueError('valid distribution required')
    logs=np.log(np.maximum(weights,1e-300))-.5*((y-means)/stds)**2-np.log(stds)-.5*math.log(2*math.pi)
    peak=float(logs.max())
    return peak+math.log(float(np.exp(logs-peak).sum()))


def mixture_quantiles(means,stds,weights,probabilities=(.025,.975)):
    means,stds,weights=map(lambda a:np.asarray(a,float),(means,stds,weights))
    out=[]
    for probability in probabilities:
        low,high=float(np.min(means-12*stds)),float(np.max(means+12*stds))
        for _ in range(64):
            mid=(low+high)/2
            cdf=sum(w*.5*(1+math.erf((mid-m)/(s*math.sqrt(2)))) for m,s,w in zip(means,stds,weights))
            if cdf<probability:low=mid
            else:high=mid
        out.append((low+high)/2)
    return out


class CausalExtrapolator:
    """预测→收读数→更新；模型从不接收episode/未来目标功率。"""
    def __init__(self,geometry,settings,background=-100.):
        self.geometry,self.settings,self.background=geometry,settings,background
        self.points=[];self.losses=np.zeros(4);self.bias=0.;self.residual_second=0.;self.last_time=None

    def state(self):
        return dict(observation_count=len(self.points),last_received_time_seconds=self.last_time,
                    prefix_sha256=hashlib.sha256(json.dumps(self.points,sort_keys=True).encode()).hexdigest(),
                    loss_ema=self.losses.tolist(),bias_db=self.bias,residual_second_db2=self.residual_second,
                    calibration_uses_only_preobservation_errors=True)

    def freeze(self,now):
        validate_points(self.points,self.geometry,now)
        loss=self.losses-self.losses.min();weights=np.exp(-loss);weights/=weights.sum()
        return FrozenExtrapolator(self.points,self.geometry,now,self.settings,self.background,weights,
                                 self.bias,max(0.,self.residual_second-self.bias**2),self.state())

    def observe(self,point):
        now=float(point['time_seconds'])
        validate_points(self.points+[point],self.geometry,now)
        # An observation can update only after every version's prediction is frozen.
        frozen=self.freeze(now);prediction=frozen.predict([point['position_m']])
        y=float(point['interference_dbm'])
        losses=[]
        for method in BASE_METHODS:
            d=prediction[method];losses.append(-mixture_logpdf(y,d['component_mean_dbm'][0],d['component_std_db'][0],d['weights']))
        mix=prediction['causal_mixture'];residual=y-float(mix['mean_dbm'][0])
        dt=1. if self.last_time is None else now-self.last_time
        rate=-math.expm1(-dt/20.)
        self.losses=(1-rate)*self.losses+rate*np.array(losses)
        self.bias=(1-rate)*self.bias+rate*residual
        self.residual_second=(1-rate)*self.residual_second+rate*residual**2
        before=self.state()
        self.points.append(dict(position_m=list(point['position_m']),time_seconds=now,interference_dbm=y))
        self.last_time=now
        return dict(time_seconds=now,position_m=list(point['position_m']),prediction=prediction,
                    residual_db=residual,update_rate=rate,used_prefix=frozen.state,post_error_update=before)


class FrozenExtrapolator:
    def __init__(self,points,geometry,now,settings,background,weights,bias,extra_variance,state):
        self.models=[LocalPredictor(points,geometry,now,settings,m,background) for m in BASE_METHODS]
        self.weights,self.bias,self.extra_variance=np.array(weights),bias,extra_variance
        self.now,self.settings,self.background,self.state=now,settings,background,state

    def predict(self,xy):
        raw=[m.predict(xy) for m in self.models]
        nearest=np.array(raw[0]['nearest_m']);common=(nearest>=0)&(nearest<=self.settings.support_radius_m)
        output={}
        for i,method in enumerate(METHODS):
            indices=[i] if i<4 else list(range(4))
            means=np.column_stack([raw[j]['mean'] for j in indices])
            stds=np.column_stack([raw[j]['std'] for j in indices])
            distances=np.column_stack([raw[j]['nearest_m'] for j in indices])
            own_distance=np.min(np.where(distances>=0,distances,np.inf),axis=1)
            own_distance=np.where(np.isfinite(own_distance),own_distance,-1.)
            domain=(own_distance>=0)&(own_distance<=160.)
            if method=='baseline_gp':domain=common.copy()
            if i==5:means+=self.bias;stds=np.sqrt(stds**2+self.extra_variance)
            means[~domain]=self.background;stds[~domain]=self.settings.unknown_error_db
            weights=np.array([1.]) if i<4 else self.weights
            mean=means@weights
            variance=np.maximum(0.,(stds**2+means**2)@weights-mean**2)
            output[method]=dict(mean_dbm=mean.tolist(),std_db=np.sqrt(variance).tolist(),
                component_mean_dbm=means.tolist(),component_std_db=stds.tolist(),weights=weights.tolist(),
                common_80m=common.tolist(),baseline_nearest_m=nearest.tolist(),baseline_nearest_age_seconds=raw[0]['age_seconds'].tolist(),
                model_nearest_m=own_distance.tolist(),model_domain_160m=domain.tolist(),
                uncertainty_calibrated=False,extended_values_are_model_only=True)
        return output

    def metadata(self):
        return dict(now_seconds=self.now,state=self.state,mixture_weights=self.weights.tolist(),bias_db=self.bias,
                    extra_variance_db2=self.extra_variance,training={m.mode:dict(count=len(m.points),
                    maximum_time_seconds=max((p['time_seconds'] for p in m.points),default=None),
                    mean_level_dbm=m.mean_level,selected_points=m.points) for m in self.models})


def route_score(distribution,weights,settings,scope='common80'):
    """各分量先转线性功率再混合；80m共同域外保持旧未知先验。"""
    from .balanced_objective import normalized_expected_power
    if scope not in ('common80','model160'):raise ValueError('unknown scope')
    weights=np.asarray(weights,float)
    mu=np.asarray(distribution['component_mean_dbm']);sd=np.asarray(distribution['component_std_db'])
    mixture=np.asarray(distribution['weights'])
    cost=normalized_expected_power(mu,sd,settings.scale_dbm)@mixture
    mask=np.asarray(distribution['common_80m'] if scope=='common80' else distribution['model_domain_160m'],bool)
    prior=float(normalized_expected_power(settings.prior_mean_dbm,settings.prior_std_db,settings.scale_dbm))
    power=float(weights@np.where(mask,cost,prior));flight=float(weights.sum())
    return dict(score_seconds=flight+settings.weight*power,flight_seconds=flight,power_seconds=power,
                predicted_seconds=float(weights@mask),unknown_seconds=float(weights@~mask),scope=scope,
                expectation_computed_before_mixture=True,uncertainty_calibrated=False)
