"""Feasible positive net/beta targets, using only decision-known inputs."""
import numpy as np

def unit_market(beta,mask):
    ix=np.flatnonzero(mask);out=np.zeros(len(mask))
    if len(ix)<3 or not np.min(beta[ix])<=1<=np.max(beta[ix]):raise ValueError('No feasible unit-beta long basket')
    active=ix.copy()
    for _ in range(len(ix)):
        c=np.vstack([np.ones(len(active)),beta[active]])
        seed=np.ones(len(active))/len(ix)
        w=seed+c.T@np.linalg.pinv(c@c.T)@(np.ones(2)-c@seed)
        if np.min(w)>=-1e-12:
            out[active]=w;assert abs(out.sum()-1)<1e-9 and abs(out@np.nan_to_num(beta)-1)<1e-9
            return out
        active=active[w>0]
    raise ValueError('Long-basket active set failed')

def neutral(values,beta,mask):
    out=np.zeros(len(mask));ix=np.flatnonzero(mask)
    if len(ix)<3:return out
    c=np.vstack([np.ones(len(ix)),beta[ix]]);_,s,v=np.linalg.svd(c,full_matrices=False)
    rank=np.count_nonzero(s>max(c.shape)*np.finfo(float).eps*s[0]);a=np.asarray(values)[ix].copy()
    for _ in range(2):a-=v[:rank].T@(v[:rank]@a)
    out[ix]=a;return out

def cap_alpha(bias,alpha,gross_limit,cap=.15):
    assert abs(bias).sum()<=gross_limit+1e-9 and abs(bias).max(initial=0)<=cap+1e-9
    def valid(t):
        w=bias+t*alpha;return abs(w).sum()<=gross_limit and abs(w).max(initial=0)<=cap
    if valid(1):return 1.
    lo,hi=0.,1.
    for _ in range(50):
        mid=(lo+hi)/2
        if valid(mid):lo=mid
        else:hi=mid
    return lo*(1-1e-12)

def project_biased(values,beta,mask,gross_limit,cap,target_net):
    """Repair affine constraints; guard projection contracts, never adds alpha."""
    bias=unit_market(beta,mask)*target_net
    alpha=neutral(np.asarray(values)-bias,beta,mask)
    return bias+cap_alpha(bias,alpha,gross_limit,cap)*alpha

def annual_variance(w,centered,indices):
    a=w[indices];x=centered
    return 252*(.5*np.sum((x@a)**2)/62+.5*np.sum(np.sum(x*x,axis=0)/62*a*a))

def target(alpha,beta,mask,centered,indices,vol,bias_size,gate):
    """Combined budget: overlay never sits outside the total risk/gross caps."""
    gross=3*min(1,mask.sum()/500);a=neutral(alpha,beta,mask)
    va=annual_variance(a,centered,indices)
    if va>1e-20:
        a*=min(vol/np.sqrt(va),gross/max(abs(a).sum(),1e-15),.15/max(abs(a).max(),1e-15))*(1-1e-12)
    else:a*=0
    a*=gate
    if bias_size==0:return a,dict(requested_bias=0,effective_bias=0,alpha_scale=1)
    unit=unit_market(beta,mask);vu=annual_variance(unit,centered,indices)
    delta=min(bias_size,vol/np.sqrt(vu),gross/abs(unit).sum(),.15/abs(unit).max())*(1-1e-12)
    b=unit*delta;va=annual_variance(a,centered,indices);vb=annual_variance(b,centered,indices)
    cov=(annual_variance(a+b,centered,indices)-va-vb)/2
    scale=1. if va<1e-20 else min(1.,max(0.,(-cov+np.sqrt(max(0.,cov*cov+va*(vol*vol-vb))))/va))
    scale=min(scale,cap_alpha(b,a,gross));w=b+scale*a
    assert annual_variance(w,centered,indices)<=vol*vol+1e-10
    assert abs(w.sum()-delta)<1e-9 and abs(w@np.nan_to_num(beta)-delta)<1e-9
    return w,dict(requested_bias=bias_size,effective_bias=float(delta),alpha_scale=float(scale))
