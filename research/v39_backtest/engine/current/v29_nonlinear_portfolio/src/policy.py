from common import *
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

class Policy(torch.nn.Module):
    def __init__(self,kind):
        super().__init__();torch.manual_seed(20261111);self.kind=kind
        self.linear=torch.nn.Linear(58,1,bias=False)
        torch.nn.init.normal_(self.linear.weight,std=.03)
        if kind=='replay_random':
            self.register_buffer('frequency',torch.randn(58,32)/np.sqrt(58));self.register_buffer('phase',torch.rand(32)*2*np.pi)
            self.nonlinear=torch.nn.Linear(32,1,bias=False);torch.nn.init.normal_(self.nonlinear.weight,std=.01)
        if kind=='replay_mlp':
            self.hidden=torch.nn.Linear(58,8);self.output=torch.nn.Linear(8,1,bias=False)
            torch.nn.init.normal_(self.output.weight,std=.01)
    def forward(self,x):
        s=self.linear(x)
        if self.kind=='replay_random':s=s+self.nonlinear(torch.cos(x@self.frequency+self.phase))
        if self.kind=='replay_mlp':s=s+self.output(torch.tanh(self.hidden(x)))
        return s.squeeze(-1)

def tensors(d,ix):
    out={k:torch.from_numpy(np.array(d[k][ix],copy=True)).to(torch.float64) for k in ('x','centered','diag','orth','mask','beta','gate')}
    out['returns']=torch.from_numpy(np.nan_to_num(np.array(d['returns'][ix],copy=True)))
    out['breaks']=np.r_[True,np.diff(ix)!=1];out['end_segment']=np.r_[np.diff(ix)!=1,True]
    return out

def variance(w,d):
    z=torch.bmm(d['centered'],w.unsqueeze(-1)).squeeze(-1)
    return 252*(.5*(z*z).sum(-1)/62+.5*(d['diag']*w*w).sum(-1))

def weights(model,d):
    s=model(d['x'])*d['mask'];u=d['orth']
    for _ in range(2):s=s-torch.bmm(u.transpose(1,2),torch.bmm(u,s.unsqueeze(-1))).squeeze(-1)
    vol=torch.sqrt(torch.clamp(variance(s,d),min=1e-20));gross=s.abs().sum(-1);peak=s.abs().amax(-1)
    scale=torch.minimum(.12/vol,torch.minimum(3*torch.clamp(d['mask'].sum(-1)/500,max=1)/gross.clamp(min=1e-15),.15/peak.clamp(min=1e-15)))
    return s*(scale*d['gate']*(1-1e-12)).unsqueeze(-1)

def replay(w,d,penalty=0,model=None,details=False):
    actual=torch.zeros_like(w[0]);rewards=[];risks=[];turnovers=[];trades=[]
    for t in range(len(w)):
        if d['breaks'][t]:actual=torch.zeros_like(actual)
        # Scheduled projection of carried targets onto current eligibility/risk.
        target=w[t];distance=(target-actual).abs().sum()
        mandatory=(actual.abs().sum()>3.3)|((actual*d['beta'][t]).sum().abs()>.05)|((actual.abs()*(1-d['mask'][t])).sum()>1e-12)
        trade=bool((distance>=.2)|mandatory)
        if trade:
            fraction=torch.ones((),dtype=w.dtype)
            for _ in range(10):fraction=1-.0002*(fraction*target-actual).abs().sum()
            held=target;turn=(fraction*target-actual).abs().sum()
        else:fraction=torch.ones((),dtype=w.dtype);held=actual;turn=torch.zeros((),dtype=w.dtype)
        rr=d['returns'][t];pnl=(held*rr).sum();growth=1+pnl
        if float(growth.detach())<=0:raise ValueError('Surrogate insolvency')
        actual=held*(1+rr)/growth
        terminal=.0002*actual.abs().sum() if d['end_segment'][t] else 0.
        rewards.append(fraction*growth*(1-terminal)-1)
        # Ex-ante risk of actual post-trade holdings, not only desired aim.
        x=d['centered'][t]@held
        risks.append(252*(.5*(x*x).sum()/62+.5*(d['diag'][t]*held*held).sum()))
        turnovers.append(turn+(actual.abs().sum() if d['end_segment'][t] else 0.));trades.append(trade)
    rewards=torch.stack(rewards);risks=torch.stack(risks)
    utility=12*rewards.mean()-2.5*risks.mean()
    regularizer=sum((p*p).sum() for p in model.parameters()) if model is not None else 0.
    loss=-utility+penalty/2*regularizer
    if details:return dict(utility=float(utility.detach()),mean_return=float((12*rewards.mean()).detach()),risk=float(risks.mean().detach()),turnover=float((12*torch.stack(turnovers).mean()).detach()),trades=sum(trades),periods=len(w))
    return loss

def train(kind,d,ix,penalty):
    assert d['complete'][ix].all()
    assert np.isfinite(d['returns'][ix][d['mask'][ix]]).all(), 'Known training outcomes cannot be zero imputed'
    data=tensors(d,ix);model=Policy(kind);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    trace=[]
    for epoch in range(EPOCHS):
        optimizer.zero_grad();w=weights(model,data);loss=replay(w,data,penalty,model)
        if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if epoch%20==0 or epoch==EPOCHS-1:trace.append(dict(epoch=epoch,loss=float(loss.detach())))
    with torch.no_grad():stats=replay(weights(model,data),data,details=True)
    stats['trace']=trace;return model,stats

def predict(model,d,ix):
    with torch.no_grad():return weights(model,tensors(d,ix)).numpy()
