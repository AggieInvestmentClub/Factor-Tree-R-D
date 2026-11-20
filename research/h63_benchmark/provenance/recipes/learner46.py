"""Train the same RFF policy under signed or nonnegative allocation constraints."""
from study46 import *
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

def verify():
    restrict();freeze();d=data();ix=np.array([140,190,240,270,len(d['dates'])-1]);x=tensors(d,ix)
    rng=np.random.default_rng(46);checks=[]
    for mode in MODES:
        for magnitude in (0.1,1.,10.):
            score=rng.normal(size=x['mask'].shape)*magnitude
            with torch.no_grad():w=weights(torch.tensor(score),x,mode).numpy()
            ref=targets_from_scores(d,ix,score,mode).to_numpy();error=float(abs(w-ref).max());assert error<1e-10
            checks.append(dict(mode=mode,score_magnitude=magnitude,max_weight_error=error))
        score=torch.tensor(rng.normal(size=x['mask'].shape),requires_grad=True);direction=torch.tensor(rng.normal(size=score.shape));probe=torch.tensor(rng.normal(size=score.shape))
        derivative=torch.autograd.grad((weights(score,x,mode)*probe).sum(),score)[0];eps=1e-7
        with torch.no_grad():finite=float(((weights(score+eps*direction,x,mode)-weights(score-eps*direction,x,mode))*probe).sum()/(2*eps))
        automatic=float((derivative*direction).sum());assert abs(finite-automatic)<1e-6*max(1,abs(finite))
        model=p.Policy('replay_random');loss=replay(weights(model(x['x']),x,mode),x,mode,model,0.);loss.backward()
        utility_gradient=sum(float(v.grad.norm()) for v in model.parameters() if v.grad is not None)
        # Broad initial score variations must permit learning before regularization.
        assert np.isfinite(utility_gradient) and utility_gradient>0
        checks.append(dict(mode=mode,finite_derivative=finite,autodiff_derivative=automatic,unpenalized_utility_gradient=utility_gradient))
    from engine46 import build
    engine,_=build();beta=np.array([.5,1.,1.5]);eligible=np.ones(3,bool)
    for mode in MODES:
        for raw in (np.array([2.,-2.,1.]),np.array([-2.,-1.,-1.]),np.array([2.,2.,2.])):
            w=engine._directional_contract(raw,beta,eligible,gross_cap(mode),.15,limits(mode))
            assert abs(w).sum()<=gross_cap(mode)+1e-12 and abs(w).max()<=.15+1e-12 and abs(w@beta)<=1+1e-12 and -1<=w.sum()<=1
            if mode=='long_only':assert w.min()>=0
    write(HERE/'reports/model_qa.json',dict(passed=True,checks=checks,mandate_repair_checked=True))
    print('MODEL_QA_PASSED',flush=True)

def fit(mode,year):
    restrict();freeze();assert read(HERE/'reports/model_qa.json')['passed'];done=HERE/f'models/{mode}_{year}.json'
    if done.exists():assert read(done)['protocol_sha256']==sha(HERE/'protocol.json');return
    began=time.monotonic();d=data();cut=pd.Timestamp(year,1,1);start=cut-pd.DateOffset(years=10);split=cut-pd.DateOffset(years=2)
    full=p.selected(d,cut,start);inner=p.selected(d,split,start);val=np.flatnonzero((d['dates']>=split)&(d['dates']<cut))
    original=read(SOURCE/f'models/log_growth_{year}.json')
    assert len(full)==original['training_periods'] and hashlib.sha256(d['dates'][full].asi8.tobytes()).hexdigest()==original['training_query_sha256']
    assert d['ends'][full].max()<cut and d['ends'][inner].max()<split
    clock=pd.DatetimeIndex(d['calendar']);begin=clock[clock.get_loc(d['dates'][val[0]])+1];end=clock[clock<cut][-1]
    account=Account(str(begin.date()),str(end.date()),str(cut.date()))
    ledger,_,_,audit,_=account.run(broad_targets(d,val),'broad_long',strict=False)
    usable=audit['v46_selection_qualified'];reference_growth=float(252*np.log1p(ledger.daily_return).mean()) if usable else None;choices=[]
    for penalty in ((.1,.01) if usable else ()):
        model,training=train(d,inner,mode,penalty);score=score_model(model,d,val);targets=targets_from_scores(d,val,score,mode)
        ledger,_,_,check,_=account.run(targets,mode,strict=False);qualified=check['v46_selection_qualified']
        growth=float(252*np.log1p(ledger.daily_return).mean()) if qualified else None
        choices.append(dict(penalty=penalty,training=training,selection_qualified=qualified,
          validation_log_growth=growth,control_log_growth=reference_growth,incremental_log_growth=growth-reference_growth if qualified else None,issues=check['issues']))
        print('INNER_DONE',mode,year,penalty,choices[-1]['incremental_log_growth'],flush=True);del model;gc.collect()
    fallback=not usable or not all(v['selection_qualified'] for v in choices)
    penalty=.1 if fallback else max(choices,key=lambda q:q['incremental_log_growth'])['penalty']
    del account;gc.collect();model,training=train(d,full,mode,penalty);torch.save(model.state_dict(),done.with_suffix('.pt'))
    ix=np.flatnonzero(d['dates'].year==year);scores=score_model(model,d,ix);np.save(HERE/f'account_inputs/{mode}_scores_{year}.npy',scores)
    if year==2024:
        j=np.flatnonzero(d['dates']<pd.Timestamp('2024-01-01'))[-1]
        assert d['ends'][full].max()<pd.Timestamp('2023-12-29') and end<=pd.Timestamp('2023-12-29')
        initial=targets_from_scores(d,np.array([j]),score_model(model,d,np.array([j])),mode)
        initial.iloc[0]=numpy_contract(initial.iloc[0].to_numpy(),initial_view(d,j),j,mode)
        initial.index=pd.DatetimeIndex(['2023-12-29'])
        initial.to_pickle(HERE/f'account_inputs/{mode}_initial.pkl')
    write(done,dict(mode=mode,year=year,protocol_sha256=sha(HERE/'protocol.json'),weights_sha256=sha(done.with_suffix('.pt')),
      training_periods=len(full),max_training_label_end=str(d['ends'][full].max()),inner_max_training_label_end=str(d['ends'][inner].max()),
      selected_penalty=penalty,validation_fallback=fallback,validation_reference_audit=audit,validation_start=str(begin.date()),validation_end=str(end.date()),
      training=training,selection=choices,seconds=time.monotonic()-began))
    print('FIT_DONE',mode,year,round(time.monotonic()-began,1),flush=True)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['verify','fit']);a.add_argument('--mode',choices=MODES);a.add_argument('--year',type=int);args=a.parse_args()
    verify() if args.stage=='verify' else fit(args.mode,args.year)
