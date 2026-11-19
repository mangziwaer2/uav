"""仅按观测前候选质量匹配活动分量预算；无功率真值或路径接口。"""
import math
import numpy as np


def select(log_weights,indices,method):
    logs=np.asarray(log_weights,float)
    if logs.ndim!=1 or not len(logs) or not np.all(np.isfinite(logs)):raise ValueError('finite log weights required')
    peak=float(logs.max());z=peak+math.log(float(np.exp(logs-peak).sum()))
    if abs(z)>1e-8:raise ValueError('normalized source evidence required')
    if not isinstance(indices,list) or not indices or len(indices)!=len(set(indices)) or any(type(i) is not int or not 0<=i<len(logs) for i in indices):raise ValueError('unique actual component indices required')
    if method not in ('replay','count_top','count_mass_top','all'):raise ValueError('unknown fixed compression rule')
    weights=np.exp(logs);weights/=weights.sum();original=sorted(indices);target=float(weights[original].sum())
    chosen=list(range(len(logs))) if method=='all' else original if method=='replay' else sorted(sorted(range(len(logs)),key=lambda i:(-logs[i],i))[:len(indices)])
    available=float(weights[chosen].sum());alpha=target/available if method=='count_mass_top' and available>0 else 1.
    if alpha>1+1e-12:raise ValueError('topK must have at least the original mass')
    alpha=min(1.,alpha);kept=weights[chosen]*alpha;mass=float(kept.sum())
    return dict(indices=chosen,kept_weights=kept.tolist(),retention_coefficient=alpha,retained_mass=min(1.,mass),
        component_count=len(chosen),target_count=len(original),target_mass=target,
        partial_retention=method=='count_mass_top',computed_without_current_readout=True)


def distribution(selection,means,sigma,gp):
    means=np.asarray(means,float)
    if not np.all(np.isfinite(means)) or not math.isfinite(sigma) or sigma<=0:raise ValueError('valid fixed component predictions required')
    indices=selection['indices'];weights=list(selection['kept_weights']);mu=means[indices].tolist();std=[sigma]*len(indices)
    remainder=max(0.,1-selection['retained_mass'])
    if remainder>0:
        if not math.isfinite(gp['mean_dbm']) or not math.isfinite(gp['std_db']) or gp['std_db']<=0:raise ValueError('valid common GP required')
        mu.append(gp['mean_dbm']);std.append(gp['std_db']);weights.append(remainder)
    weights=np.array(weights,float);weights/=weights.sum()
    return dict(means=mu,stds=std,weights=weights.tolist())


def conditional_signature(selection,signatures):
    signatures=np.asarray(signatures,float);w=np.array(selection['kept_weights'])
    if signatures.ndim!=2 or not np.all(np.isfinite(signatures)):raise ValueError('finite Q x components model signature required')
    if w.sum()==0:return None
    return (signatures[:,selection['indices']]@(w/w.sum())).tolist()
