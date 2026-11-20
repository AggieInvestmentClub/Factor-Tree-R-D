"""Train the same RFF policy under signed or nonnegative allocation constraints."""
from model import *
import time

def tensors(d,ix):
    x={k:torch.from_numpy(np.array(d[k][ix],copy=True)).to(torch.float64) for k in ('x','centered','diag','mask','beta')}
    x['returns']=torch.from_numpy(np.nan_to_num(np.array(d['returns'][ix],copy=True)))
    x['breaks']=np.r_[True,np.diff(ix)!=1];x['end_segment']=np.r_[np.diff(ix)!=1,True]
    return x

def weights(scores,x,mode):
    n=x['mask'].sum(-1).clamp(min=1);w=3*scores*x['mask']/n[:,None]
    if mode=='long_only':w=torch.relu(w)
    z=torch.bmm(x['centered'],w.unsqueeze(-1)).squeeze(-1)
    v=252*(.5*z.square().sum(-1)/62+.5*(x['diag']*w.square()).sum(-1))
    cap=gross_cap(mode)*(n/500).clamp(max=1)
    scale=torch.ones_like(n)
    for bound in (.12/v.clamp(min=1e-30).sqrt(),cap/w.abs().sum(-1).clamp(min=1e-15),
      .15/w.abs().amax(-1).clamp(min=1e-15),.99/w.sum(-1).clamp(min=1e-15),1/(-w.sum(-1)).clamp(min=1e-15),1/(w*x['beta']).sum(-1).abs().clamp(min=1e-15)):
        scale=torch.minimum(scale,bound)
    return w*(scale*(1-1e-12))[:,None]

def replay(w,x,mode,model=None,penalty=0.,details=False):
    actual=torch.zeros_like(w[0]);rewards=[];cap=gross_cap(mode);bounds=limits(mode)
    for t in range(len(w)):
        if x['breaks'][t]:actual=torch.zeros_like(actual)
        target=w[t];distance=(target-actual).abs().sum();net=actual.sum();beta=(actual*x['beta'][t]).sum()
        mandatory=(actual.abs().sum()>1.1*cap)|(net<bounds['min_net']-.05)|(net>bounds['max_net']+.05)|(beta.abs()>1.05)|((actual.abs()*(1-x['mask'][t])).sum()>1e-12)
        if bool((distance>=.2)|mandatory):
            fraction=torch.ones((),dtype=w.dtype)
            for _ in range(10):fraction=1-.0002*(fraction*target-actual).abs().sum()
            held=target
        else:fraction=torch.ones((),dtype=w.dtype);held=actual
        growth=1+(held*x['returns'][t]).sum();assert float(growth.detach())>0
        actual=held*(1+x['returns'][t])/growth
        terminal=.0002*actual.abs().sum() if x['end_segment'][t] else 0.
        rewards.append(fraction*growth*(1-terminal)-1)
    r=torch.stack(rewards);utility=12*torch.log1p(r).mean()
    if details:return dict(annual_log_growth=float(utility.detach()),annual_mean=float(12*r.mean().detach()),mean_target_gross=float(w.abs().sum(-1).mean().detach()),mean_target_net=float(w.sum(-1).mean().detach()))
    regularizer=sum(q.square().sum() for q in model.parameters()) if model is not None else 0.
    return -utility+penalty/2*regularizer

def train(d,ix,mode,penalty):
    assert d['complete'][ix].all() and np.isfinite(d['returns'][ix][d['mask'][ix]]).all()
    x=tensors(d,ix);model=p.Policy('replay_random');opt=torch.optim.Adam(model.parameters(),lr=.01);trace=[]
    for epoch in range(100):
        opt.zero_grad();w=weights(model(x['x']),x,mode);loss=replay(w,x,mode,model,penalty)
        assert torch.isfinite(loss);loss.backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if epoch%20==0 or epoch==99:trace.append(dict(epoch=epoch,loss=float(loss.detach()),gradient_norm=float(norm)))
    with torch.no_grad():summary=replay(weights(model(x['x']),x,mode),x,mode,details=True)
    summary['trace']=trace;return model,summary
