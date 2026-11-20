"""Log-domain observation likelihood primitives."""
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

