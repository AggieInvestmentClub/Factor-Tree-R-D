"""Causal volatility-targeted beta, with a shared alpha budget and daily controls."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];V31=HERE.parent/'v31_growth_long_bias'
sys.path.insert(0,str(V31/'src'))
from settings import read,write,sha,SOURCE,V29,V23,V28
import runner as prior
from allocation import unit_market,cap_alpha,annual_variance,target
import numpy as np,pandas as pd,json,datetime,time,gc,argparse,importlib.util,difflib
ARMS=('daily_neutral','daily_static25','daily_static50','vol_target','trend_vol_long','trend_vol_signed')
PAIRS=[['daily_static50','monthly_static50'],['vol_target','daily_static50'],['trend_vol_long','vol_target'],['trend_vol_signed','trend_vol_long'],
       ['vol_target','daily_static25'],['trend_vol_long','daily_static25'],['trend_vol_signed','daily_static25'],['daily_static50','daily_neutral'],['trend_vol_long','daily_neutral'],['trend_vol_signed','daily_neutral']]

def build_engine():
    source=V31/'src/research_account.py';text=source.read_text()
    old="desired.sum(axis=1).lt(-1e-9).any()";assert text.count(old)==1;text=text.replace(old,"desired.sum(axis=1).lt(-.25-1e-9).any()")
    old="Net bias must lie in[0,.5]";assert text.count(old)==1;text=text.replace(old,"Net bias must lie in[-.25,.5]")
    out=HERE/'src/signed_account.py';out.write_text(text)
    (HERE/'engine_changes.diff').write_text(''.join(difflib.unified_diff(source.read_text().splitlines(True),text.splitlines(True),fromfile='v31_account.py',tofile='signed_account.py')))
    return sha(out)

def protocol():
    p=HERE/'protocol.json';assert sha(p)==p.with_suffix('.sha256').read_text().strip();m=read(p)
    for name,h in m['code'].items():assert sha(HERE/'src'/name)==h,name
    for name,h in m['sources'].items():assert sha(name)==h,name
    return sha(p)

def freeze():
    p=HERE/'protocol.json'
    if p.exists():return protocol()
    build_engine()
    m=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='Frozen before v33 accounts;2022 drawdown already known and explicitly requested; no thresholds tuned to2022',arms=ARMS,pairs=PAIRS,
        alpha='Immutable v31 log_growth21-session forecasts, forward-held between their original dates. No new forecast fitting or architecture selection.',
        schedule='Daily decisions from original2007 anchor to penultimate2023 session; execution next close; only risk/beta policy updates daily. Daily neutral,25%,50% controls use identical schedule and alpha reference.',
        regime='SPY qualified adjusted close. sigma=max(21,63session sample standard deviation)*sqrt252, known at current close. f=min(1,.15/sigma). Bull:close>SMA200 and63session return>0. Bear:both negative. Mixed:otherwise. No direction inferred from volatility alone.',
        bias={'daily_neutral':0,'daily_static25':.25,'daily_static50':.5,'vol_target':'.5*f, targeting at most7.5% annual standalone market-risk contribution','trend_vol_long':'.5*f times1/bull,.5/mixed,0/bear','trend_vol_signed':'.5*f times1/bull,.5/mixed,-.5/bear;short cap-.25'},
        joint_budget='12% total forecast volatility, gross3 active/500, name.15. Construct one daily static50 reference under these limits, then fix its neutral alpha component for all six arms. Replacing beta can only contract that common alpha further; removed market risk is never refilled into alpha. The signed risk projection respects the same total budget.',
        engine='v31 repaired engine, with only declared desired-net lower-bound validation extended from0 to-.25. Same actual net and estimated beta deviation guards.05, grossguard3.3,buffer.2,2bps/side,next-close quote failures,corporate events,claims,margin.',
        inference=dict(primary='full2007-2023',secondary='2012-2023',growth_family=10,blocks=[20,60],replications=2000,endpoint='Annual log growth difference',risk='Drawdown,volatility,expected shortfall,turnover and exposure descriptive; no promise of safety or drawdown bounds',stress='2008,2020,2022 calendar years;2021-2023 path around known2022 drawdown; no model selection on these windows'),
        controls='monthly_static50 and monthly_neutral imported unchanged from v31. Daily static50 scheduled-date targets must match original monthly50 within1e-9. Covariance factors must match original source caches on all204 original query dates.',
        causal_checks='Qualified contiguous return histories only;63session diag50 covariance;numeric data bounded pre2024. Poison future SPY values and verify past regime features unchanged. Decision dates strictly precede their execution dates.',
        promotion=False,financing='Original zero borrow,financing,cash yield retained;100/300bp illustrative fixed-holdings borrow sensitivities',
        sources={str(f):sha(f) for f in [V31/'protocol.json',V31/'src/research_account.py',V31/'src/allocation.py',V31/'src/runner.py',V31/'account_inputs/log_growth_desired.pkl',V31/'account_inputs/growth_long50_v12_desired.pkl',SOURCE/'src/lazy_covariances.py',SOURCE/'src/risk_engine.py',SOURCE/'data/assembled/approval.json',SOURCE/'account_inputs/betas.pkl',SOURCE/'account_inputs/eligible.pkl']},
        code={f.name:sha(f) for f in (HERE/'src').glob('*.py') if f.name in ('controller.py','signed_account.py','test_controller.py')},
        limitations=['Adaptive research on reused history;2022 explicitly selected after viewing its drawdown','Rule-based state proxies, not inferred true market states','Daily risk management changes trades; matched daily controls are essential','Same historical universe and conditional source valuations','Volatility is not a forecast of return direction; short exposure can lose during rebounds'])
    write(p,m);p.with_suffix('.sha256').write_text(sha(p));return sha(p)

def qualified_returns(prices,names,calendar):
    returns=np.full((len(calendar),len(names)),np.nan)
    for j,name in enumerate(names):
        frame=prices[name].loc[lambda f:(f.index<=calendar[-1])&(f.index<pd.Timestamp('2024-01-01'))].reindex(calendar)
        if 'q' in frame:q=frame.q.eq(1)
        elif {'source_available','tr_chain_available'}.issubset(frame):q=frame.source_available.eq(1)&frame.tr_chain_available.eq(1)
        elif 'tr_qualified' in frame:q=frame.tr_qualified.eq(1)
        elif 'price_qualified' in frame:q=frame.price_qualified.eq(1)
        else:raise ValueError('Explicit qualification required')
        close=pd.to_numeric(frame.adj_close,errors='raise');segment=frame.tr_segment_id if 'tr_segment_id' in frame else pd.Series(1.,index=calendar)
        good=q&np.isfinite(close)&close.gt(0)&segment.notna()
        returns[:,j]=(close/close.shift(1)-1).where(good&good.shift(1,fill_value=False)&segment.eq(segment.shift(1))).to_numpy()
    return returns

def regime(spy):
    close=spy.adj_close.where(spy.q.eq(1));same=spy.tr_segment_id.eq(spy.tr_segment_id.shift()) if 'tr_segment_id' in spy else pd.Series(True,index=spy.index)
    r=close.pct_change(fill_method=None).where(same);v21=r.rolling(21,min_periods=21).std(ddof=1)*np.sqrt(252);v63=r.rolling(63,min_periods=63).std(ddof=1)*np.sqrt(252)
    sigma=pd.concat([v21,v63],axis=1).max(axis=1,skipna=False);sma=close.rolling(200,min_periods=200).mean();ret63=close/close.shift(63)-1
    bull=(close>sma)&ret63.gt(0);bear=(close<sma)&ret63.lt(0);state=np.where(bull,1,np.where(bear,-1,0));factor=np.minimum(1,.15/sigma)
    return pd.DataFrame(dict(sigma21=v21,sigma63=v63,sigma=sigma,sma200=sma,close=close,return63=ret63,state=state,factor=factor),index=spy.index)

def requested(row):
    f=float(row.factor);state=int(row.state)
    return dict(daily_neutral=0.,daily_static25=.25,daily_static50=.5,vol_target=.5*f,
        trend_vol_long=.5*f*(1 if state==1 else 0 if state==-1 else .5),trend_vol_signed=.5*f*(1 if state==1 else -.5 if state==-1 else .5))

def combine(alpha,unit,centered,indices,beta_request,gross_limit,vol=.12):
    variance_unit=annual_variance(unit,centered,indices)
    delta=np.sign(beta_request)*min(abs(beta_request),vol/np.sqrt(variance_unit),gross_limit/abs(unit).sum(),.15/abs(unit).max())*(1-1e-12)
    bias=unit*delta;va=annual_variance(alpha,centered,indices);vb=annual_variance(bias,centered,indices)
    cov=(annual_variance(alpha+bias,centered,indices)-va-vb)/2
    scale=1. if va<1e-20 else min(1.,max(0.,(-cov+np.sqrt(max(0.,cov*cov+va*(vol*vol-vb))))/va))
    scale=min(scale,cap_alpha(bias,alpha,gross_limit));w=bias+scale*alpha
    assert annual_variance(w,centered,indices)<=vol*vol+1e-10
    return w,float(delta),float(scale)

def prepare():
    seal=protocol();loaded=prior.inputs();monthly=pd.read_pickle(V31/'account_inputs/log_growth_desired.pkl');names=list(monthly.columns)
    clock=loaded['clock'];dates=clock[(clock>=monthly.index[0])&(clock<clock[-1])];calendar=loaded['prices']['SPY'].index;calendar=calendar[(calendar<=dates[-1])&(calendar<pd.Timestamp('2024-01-01'))]
    returns=qualified_returns(loaded['prices'],names,calendar);spy=loaded['prices']['SPY'].loc[calendar];reg=regime(spy);assert np.isfinite(reg.loc[dates].to_numpy()).all()
    # A future-value poison cannot affect any earlier indicator.
    poison=spy.copy();cut=pd.Timestamp('2021-12-31');poison.loc[poison.index>cut,'adj_close']*=np.linspace(3,9,(poison.index>cut).sum());pd.testing.assert_frame_equal(reg.loc[:cut],regime(poison).loc[:cut])
    beta=loaded['beta'].reindex(index=dates,columns=names).to_numpy();eligible=loaded['eligible'].reindex(index=dates,columns=names).to_numpy(bool)
    daily={a:np.empty((len(dates),len(names))) for a in ARMS};rows=[];parity=[];out=HERE/'account_inputs';out.mkdir(exist_ok=True);reports=HERE/'reports';reports.mkdir(exist_ok=True)
    original50=pd.read_pickle(V31/'account_inputs/growth_long50_v12_desired.pkl');max_cov_error=0.;max_target_error=0.
    for i,date in enumerate(dates):
        end=calendar.get_loc(date)+1;x=returns[end-63:end];valid=np.isfinite(x).all(0);indices=np.flatnonzero(valid);work=x[:,indices];centered=work-work.mean(0)
        mask=eligible[i]&np.isfinite(beta[i])&valid;assert mask.sum()>=3;b=np.nan_to_num(beta[i]);gross=3*min(1,mask.sum()/500)
        source_date=monthly.index[monthly.index.searchsorted(date,side='right')-1];a=monthly.loc[source_date].to_numpy()
        reference,info=target(a,b,mask,centered,indices,.12,.5,1.);unit=unit_market(b,mask);alpha=reference-unit*info['effective_bias']
        if date in original50.index:
            with np.load(V28/f'data/covariances/{date:%Y-%m-%d}.npz') as z:
                np.testing.assert_array_equal(indices,z['indices']);error=float(abs(centered-z['centered']).max(initial=0));assert error<1e-12;max_cov_error=max(max_cov_error,error)
            error=float(abs(reference-original50.loc[date].to_numpy()).max());assert error<1e-9;max_target_error=max(max_target_error,error);parity.append(str(date.date()))
        for arm,value in requested(reg.loc[date]).items():
            w,delta,scale=combine(alpha,unit,centered,indices,value,gross)
            if arm=='daily_static50':np.testing.assert_allclose(w,reference,atol=1e-10)
            assert abs(w[~mask]).max(initial=0)<1e-10 and abs(w.sum()-delta)<1e-9 and abs(w@b-delta)<1e-9
            assert abs(w).sum()<=gross+1e-9 and abs(w).max()<=.15+1e-9 and scale<=1+1e-12
            daily[arm][i]=w;rows.append(dict(arm=arm,date=str(date.date()),alpha_source_date=str(source_date.date()),requested_beta=value,effective_beta=delta,alpha_contraction=scale,reference_alpha_gross=float(abs(alpha).sum()),gross=float(abs(w).sum()),forecast_vol=float(np.sqrt(annual_variance(w,centered,indices)))))
        if i%500==0:print('DAILY_TARGETS',i,len(dates),flush=True)
    assert len(parity)==204
    for arm,w in daily.items():
        f=out/f'{arm}_desired.pkl';pd.DataFrame(w,index=dates,columns=names).to_pickle(f);write(out/f'{arm}.json',dict(protocol_sha256=seal,target_sha256=sha(f),rows=len(w)))
    reg.loc[dates].to_csv(reports/'regime.csv');pd.DataFrame(rows).to_csv(reports/'target_diagnostics.csv.gz',index=False)
    np.save(out/'qualified_returns.npy',returns);write(out/'return_clock.json',dict(dates=[str(x.date()) for x in calendar],names=names,source_price_sha256=sha(SOURCE/'data/assembled/prices_candidate.pkl')))
    write(reports/'preparation_qa.json',dict(passed=True,protocol_sha256=seal,decisions=len(dates),arms=len(ARMS),target_vectors=len(dates)*len(ARMS),source_covariance_parity_dates=len(parity),max_covariance_error=max_cov_error,max_scheduled_target_error=max_target_error,future_indicator_poison_passed=True,no_freed_risk_refill=True,pre2024_only=True))
    print('PREPARED',len(dates),len(ARMS),flush=True)

def account_engine():
    s=importlib.util.spec_from_file_location('v33_signed_engine',HERE/'src/signed_account.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def simulate(loaded,targets,positions=False,sink=None):
    clock=loaded['clock'];start=clock[clock.get_loc(targets.index[0])+1];end=clock[-1]
    claims=[r for r in loaded['claims'] if start<=pd.Timestamp(r['ex_date'])<=end];assert all(pd.Timestamp(r['settlement_date'])<=end for r in claims)
    return account_engine().simulate_stock_portfolio(loaded['prices'],targets,loaded['events'],start=start,end=end,cost_bps=2,buffer_l1=.2,risk_betas=loaded['beta'],eligibility=loaded['eligible'],
        neutral=False,dollar_neutral=False,beta_neutral=False,controlled_long_bias=True,repair_stale_risk=True,position_cap=.15,gross_cap=3,gross_guard=3.3,data_cutoff='2024-01-01',record_positions=positions,position_sink=sink,distribution_claims=claims)

def accounts(arms):
    seal=protocol();loaded=prior.inputs();out=HERE/'accounts';out.mkdir(exist_ok=True)
    for arm in arms:
        done=out/f'{arm}_complete.json'
        if done.exists():assert read(done)['ledger_sha256']==sha(out/f'{arm}_ledger.csv.gz');continue
        f=HERE/f'account_inputs/{arm}_desired.pkl';m=read(f.with_name(f'{arm}.json'));assert m['protocol_sha256']==seal and m['target_sha256']==sha(f)
        audit=prior.bridge.NoncashExposureAudit(pd.read_csv(SOURCE/'data/distribution_audit/spin_census_dispositions.csv'));count=0
        def sink(frame):
            nonlocal count
            audit.consume(frame);count+=len(frame)
        start=time.monotonic();print('ACCOUNT_BEGIN',arm,flush=True);ledger,pos,actual,decisions,status=simulate(loaded,pd.read_pickle(f),True,sink)
        assert pos.empty and status['streamed_position_rows']==count and status['arithmetic_passed'];status['noncash_exposure_audit']=audit.report()
        if audit.rows:status['issues'].extend(audit.rows);status['validated_performance']=False
        for suffix,frame,index in [('ledger',ledger,True),('targets',actual,True),('decisions',decisions,False)]:frame.to_csv(out/f'{arm}_{suffix}.csv.gz',index=index)
        write(done,dict(protocol_sha256=seal,arm=arm,target_sha256=sha(f),ledger_sha256=sha(out/f'{arm}_ledger.csv.gz'),audit=status,seconds=time.monotonic()-start));print('ACCOUNT_DONE',arm,round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','freeze','prepare','accounts']);p.add_argument('--arms',nargs='+',default=list(ARMS));a=p.parse_args()
    if a.stage=='build':print(build_engine())
    elif a.stage=='freeze':print(freeze())
    elif a.stage=='prepare':prepare()
    else:accounts(a.arms)
