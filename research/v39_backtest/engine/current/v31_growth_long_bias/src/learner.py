from settings import *
from policy import Policy,tensors,weights,torch
from runner import inputs,validation
import gc,time,argparse

def loss(w,d,objective,model=None,penalty=0.,details=False):
    actual=torch.zeros_like(w[0]);rewards=[];risks=[]
    for t in range(len(w)):
        if d['breaks'][t]:actual=torch.zeros_like(actual)
        target=w[t];distance=(target-actual).abs().sum()
        mandatory=(actual.abs().sum()>3.3)|((actual*d['beta'][t]).sum().abs()>.05)|((actual.abs()*(1-d['mask'][t])).sum()>1e-12)
        if bool((distance>=.2)|mandatory):
            fraction=torch.ones((),dtype=w.dtype)
            for _ in range(10):fraction=1-.0002*(fraction*target-actual).abs().sum()
            held=target
        else:fraction=torch.ones((),dtype=w.dtype);held=actual
        growth=1+(held*d['returns'][t]).sum()
        if float(growth.detach())<=0:raise ValueError('Surrogate insolvency')
        actual=held*(1+d['returns'][t])/growth
        terminal=.0002*actual.abs().sum() if d['end_segment'][t] else 0.
        rewards.append(fraction*growth*(1-terminal)-1)
        x=d['centered'][t]@held;risks.append(252*(.5*(x*x).sum()/62+.5*(d['diag'][t]*held*held).sum()))
    r=torch.stack(rewards);v=torch.stack(risks);mu=12*r.mean();risk=v.mean();growth=12*torch.log1p(r).mean()
    utility={'utility5':mu-2.5*risk,'utility1':mu-.5*risk,'net_return':mu,'log_growth':growth}[objective]
    reg=sum((p*p).sum() for p in model.parameters()) if model is not None else 0.
    if details:return dict(utility=float(utility),annual_mean=float(mu),annual_log_growth=float(growth),annual_variance=float(risk))
    return -utility+penalty/2*reg
def train(d,ix,objective,penalty):
    assert d['complete'][ix].all() and np.isfinite(d['returns'][ix][d['mask'][ix]]).all()
    x=tensors(d,ix);model=Policy('replay_random');opt=torch.optim.Adam(model.parameters(),lr=.01);trace=[]
    for epoch in range(EPOCHS):
        opt.zero_grad();l=loss(weights(model,x),x,objective,model,penalty);assert torch.isfinite(l);l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if epoch%20==0 or epoch==EPOCHS-1:trace.append(dict(epoch=epoch,loss=float(l.detach())))
    with torch.no_grad():stats=loss(weights(model,x),x,objective,details=True)
    stats['trace']=trace;return model,stats
def predict(model,d,ix):
    with torch.no_grad():return weights(model,tensors(d,ix)).numpy()
def fit(years):
    seal=protocol();d=load();loaded=None
    for year in years:
        cut=pd.Timestamp(year,1,1);start=cut-pd.DateOffset(years=10);split=cut-pd.DateOffset(years=2)
        full=selected(d,cut,start);inner=selected(d,split,start);val=np.flatnonzero((d['dates']>=split)&(d['dates']<cut));query=np.flatnonzero(d['dates'].year==year)
        fallback=year<2009 or len(inner)<20
        for objective in OBJECTIVES:
            done=ROOT/f'models/{objective}_{year}.json'
            if done.exists():assert read(done)['protocol_sha256']==seal;continue
            began=time.monotonic();choices=[]
            if not fallback:
                if loaded is None:loaded=inputs()
                for penalty in PENALTIES:
                    model,stats=train(d,inner,objective,penalty);target=pd.DataFrame(predict(model,d,val),index=d['dates'][val],columns=d['names'])
                    v=validation(loaded,target,loaded['clock'][loaded['clock']<cut][-1],objective);choices.append(dict(penalty=penalty,validation=v,training=stats))
                    print('INNER',year,objective,penalty,round(v['utility'],5),flush=True)
                penalty=max(choices,key=lambda c:c['validation']['utility'])['penalty']
            else:penalty=PENALTIES[0]
            model,stats=train(d,full,objective,penalty);file=ROOT/f'models/{objective}_{year}.pt';file.parent.mkdir(exist_ok=True);torch.save(model.state_dict(),file)
            target=ROOT/f'account_inputs/{objective}_{year}.npy';target.parent.mkdir(exist_ok=True);np.save(target,predict(model,d,query))
            write(done,dict(objective=objective,year=year,protocol_sha256=seal,weights_sha256=sha(file),targets_sha256=sha(target),training_periods=len(full),
                training_query_sha256=hashlib.sha256(d['dates'][full].asi8.tobytes()).hexdigest(),max_training_label_end=str(d['ends'][full].max()),refit_boundary=str(cut),
                inner_max_training_label_end=str(d['ends'][inner].max()),validation_start=str(split),selection=choices,selected_penalty=penalty,fallback=fallback,training=stats,seconds=time.monotonic()-began))
            print('FIT_DONE',year,objective,round(time.monotonic()-began,1),flush=True);gc.collect()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--years',nargs='+',type=int,default=list(range(2007,2024)));a=p.parse_args();fit(a.years)
