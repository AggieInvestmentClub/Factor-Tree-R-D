from controller import *
from analyze_state_v20 import load_ledger,performance,stats
from evaluate_horizons import paired_comparisons
s=importlib.util.spec_from_file_location('v31_evaluation',V31/'src/evaluate.py');evaluation=importlib.util.module_from_spec(s);s.loader.exec_module(evaluation)
PERIODS=[('full','2007-01-01','2023-12-31'),('secondary_2012_2023','2012-01-01','2023-12-31'),('early','2007-01-01','2011-12-31'),
    ('stress_2008','2008-01-01','2008-12-31'),('stress_2020','2020-01-01','2020-12-31'),('stress_2022','2022-01-01','2022-12-31'),('drawdown_2021_2023','2021-01-01','2023-12-31')]

def drawdown_dates(r):
    nav=(1+r).cumprod();peak=nav.cummax().clip(lower=1);dd=nav/peak-1;trough=dd.idxmin();before=nav.loc[:trough];level=max(1.,before.max());peakdate=str(before.idxmax().date()) if before.max()>=1 else 'starting capital'
    recovery=nav.loc[nav.index>trough];recovery=recovery.loc[recovery>=level]
    return dict(drawdown_peak=peakdate,drawdown_trough=str(trough.date()),drawdown_recovery=str(recovery.index[0].date()) if len(recovery) else 'unrecovered at period end')

def validate():
    seal=protocol();prep=read(HERE/'reports/preparation_qa.json');assert prep['passed'] and prep['protocol_sha256']==seal
    targets={a:pd.read_pickle(HERE/f'account_inputs/{a}_desired.pkl') for a in ARMS};first=targets[ARMS[0]];dates=first.index;names=list(first.columns);cube=np.stack([targets[a].to_numpy() for a in ARMS])
    clockmeta=read(HERE/'account_inputs/return_clock.json');calendar=pd.DatetimeIndex(clockmeta['dates']);assert names==clockmeta['names'];assert calendar.max()<pd.Timestamp('2024-01-01')
    assert clockmeta['source_price_sha256']==read(SOURCE/'data/assembled/approval.json')['prices_sha256']
    returns=np.load(HERE/'account_inputs/qualified_returns.npy',mmap_mode='r');beta=pd.read_pickle(SOURCE/'account_inputs/betas.pkl').reindex(index=dates,columns=names).to_numpy();elig=pd.read_pickle(SOURCE/'account_inputs/eligible.pkl').reindex(index=dates,columns=names).to_numpy(bool)
    td=pd.read_csv(HERE/'reports/target_diagnostics.csv.gz',parse_dates=['date','alpha_source_date']);assert td.alpha_source_date.le(td.date).all() and td.date.max()<pd.Timestamp('2024-01-01')
    records={a:td.loc[td.arm.eq(a)].set_index('date').loc[dates] for a in ARMS};max_error=0.;max_vol=0.;projection_error=0.
    for i,date in enumerate(dates):
        end=calendar.get_loc(date)+1;x=returns[end-63:end];valid=np.isfinite(x).all(0);indices=np.flatnonzero(valid);c=x[:,indices]-x[:,indices].mean(0);b=np.nan_to_num(beta[i]);mask=elig[i]&np.isfinite(beta[i])&valid;gross=3*min(1,mask.sum()/500);w=cube[:,i,:]
        unit=unit_market(b,mask);reference=records['daily_static50'].iloc[i];alpha=w[2]-unit*reference.effective_beta
        vol=np.sqrt(252*(.5*np.sum((c@w[:,indices].T)**2,axis=0)/62+.5*np.sum(np.sum(c*c,axis=0)/62*w[:,indices]**2,axis=1)))
        assert np.max(vol)<=.12+1e-9 and abs(w[:,~mask]).max(initial=0)<1e-10 and abs(w).max()<=.15+1e-9 and abs(w).sum(1).max()<=gross+1e-9
        max_vol=max(max_vol,float(vol.max()))
        for k,a in enumerate(ARMS):
            row=records[a].iloc[i];assert -.25-1e-9<=row.effective_beta<=.5+1e-9 and 0<=row.alpha_contraction<=1+1e-12 and abs(row.effective_beta)<=abs(row.requested_beta)+1e-9
            error=max(abs(w[k].sum()-row.effective_beta),abs(w[k]@b-row.effective_beta),abs(vol[k]-row.forecast_vol));assert error<1e-9;max_error=max(max_error,error)
            error=float(abs(w[k]-unit*row.effective_beta-row.alpha_contraction*alpha).max());assert error<1e-9;projection_error=max(projection_error,error)
    monthly=pd.read_pickle(V31/'account_inputs/growth_long50_v12_desired.pkl');np.testing.assert_allclose(targets['daily_static50'].loc[monthly.index],monthly,atol=1e-9)
    states={}
    for a in ARMS:
        m=read(HERE/f'accounts/{a}_complete.json');assert m['protocol_sha256']==seal and m['ledger_sha256']==sha(HERE/f'accounts/{a}_ledger.csv.gz') and m['target_sha256']==sha(HERE/f'account_inputs/{a}_desired.pkl')
        assert m['audit']['arithmetic_passed'] and m['audit']['noncash_exposure_audit']['passed'];f=load_ledger(HERE/f'accounts/{a}_ledger.csv.gz',a);assert f.cash.min()>-1e-9
        d=pd.read_csv(HERE/f'accounts/{a}_decisions.csv.gz',parse_dates=['decision_date','execution_date']);known=d.execution_date.notna();assert d.loc[known,'execution_date'].gt(d.loc[known,'decision_date']).all()
        states[a]={k:m['audit'][k] for k in ('provisional_sessions','validated_performance')};states[a]['margin_calls']=int(f.margin_calls.sum())
    q=dict(passed=True,protocol_sha256=seal,decisions=len(dates),target_vectors=len(dates)*len(ARMS),max_target_constraint_error=max_error,max_forecast_volatility=max_vol,no_refill_decomposition_error=projection_error,
        monthly_target_parity=True,source_covariance_parity=prep['max_covariance_error'],source_parity_dates=204,future_indicator_poison_passed=True,strict_next_close_execution=True,accounts=states,source_performance_is_conditional=True)
    write(HERE/'reports/qa.json',q);return q

def main():
    q=validate();out=HERE/'reports';ledgers={a:load_ledger(HERE/f'accounts/{a}_ledger.csv.gz',a) for a in ARMS}
    for a,old in [('monthly_static50','growth_long50_v12'),('monthly_neutral','log_growth'),('h21_original','h21_legacy')]:ledgers[a]=load_ledger(V31/f'accounts/{old}_ledger.csv.gz',a)
    clock=next(iter(ledgers.values())).index;assert all(f.index.equals(clock) for f in ledgers.values());spy=evaluation.market_returns();rows=[];annual=[];funding=[];diagnostics=[]
    td=pd.read_csv(out/'target_diagnostics.csv.gz',parse_dates=['date']);reg=pd.read_csv(out/'regime.csv',index_col=0,parse_dates=True)
    for a,f in ledgers.items():
        previous=f.nav.shift(1,fill_value=1)
        for period,left,right in PERIODS:
            p=f.loc[left:right];r=performance(a,p,previous.loc[p.index],period,spy);r.update(drawdown_dates(p.daily_return));r['annual_log_growth']=252*np.log1p(p.daily_return).mean();r['flat_fraction']=p.gross_exposure.lt(1e-8).mean();r['daily_expected_shortfall5']=p.loc[p.daily_return.le(p.daily_return.quantile(.05)),'daily_return'].mean();rows.append(r)
        for year in range(2007,2024):
            p=f.loc[str(year)];r=performance(a,p,previous.loc[p.index],str(year),spy);r['year']=year;annual.append(r)
        if a in ARMS:
            d=pd.read_csv(HERE/f'accounts/{a}_decisions.csv.gz',parse_dates=['decision_date','execution_date']);targets=td.loc[td.arm.eq(a)]
            for period,left,right in PERIODS:
                p=f.loc[left:right];dec=d.loc[d.decision_date.between(left,right)];t=targets.loc[targets.date.between(left,right)];market=reg.loc[left:right]
                diagnostics.append(dict(arm=a,period=period,mean_requested_beta=t.requested_beta.mean(),mean_effective_beta=t.effective_beta.mean(),mean_actual_estimated_beta=dec.actual_estimated_beta.mean(),min_actual_estimated_beta=dec.actual_estimated_beta.min(),max_actual_estimated_beta=dec.actual_estimated_beta.max(),average_gross=p.gross_exposure.mean(),
                    mean_alpha_contraction=t.alpha_contraction.mean(),extra_alpha_contraction_fraction=t.alpha_contraction.lt(1-1e-8).mean(),submissions=int(dec.submitted.sum()),abort_batches=int(p.order_batch_aborted.sum()),margin_calls=int(p.margin_calls.sum()),provisional_sessions=int((~p.pnl_validated.astype(bool)).sum()),
                    fraction_short_bias=t.effective_beta.lt(-1e-8).mean(),fraction_bear=market.state.eq(-1).mean(),mean_market_volatility=market.sigma.mean()))
        short=f.short_notional.shift(1,fill_value=0)/previous
        for bps in (0,100,300):
            adjusted=f.daily_return-short*bps/10000/252
            for period,left,right in PERIODS[:2]:funding.append(dict(arm=a,period=period,annual_borrow_bps=bps,**stats(adjusted.loc[left:right].to_numpy()),scope='Fixed-holdings return-path charge; no funding/guard feedback, stock-specific borrow or collateral yield'))
    returns=pd.DataFrame({a:f.daily_return for a,f in ledgers.items()});growth=pd.concat([evaluation.paired_growth(returns,PAIRS,b) for b in (20,60)],ignore_index=True);growth['period']='full2007_2023'
    secondary=pd.concat([paired_comparisons(returns,PAIRS,b,family_size=10) for b in (20,60)],ignore_index=True);secondary['period']='full2007_2023'
    positive=[]
    for a,b in PAIRS:
        x=growth.loc[growth.arm.eq(a)&growth.baseline.eq(b)]
        if x.delta_annual_log_growth.gt(0).all() and x.p_holm.lt(.05).all():positive.append([a,b])
    for file,frame in [('performance.csv',pd.DataFrame(rows)),('annual.csv',pd.DataFrame(annual)),('paired_growth_full.csv',growth),('paired_sharpe_full_secondary.csv',secondary),('borrow_sensitivity.csv',pd.DataFrame(funding)),('account_diagnostics.csv',pd.DataFrame(diagnostics))]:frame.to_csv(out/file,index=False)
    returns.to_csv(out/'daily_returns.csv.gz');write(out/'status.json',dict(completed=True,protocol_sha256=protocol(),positive_growth_holm_both_blocks=positive,promotion=False,primary='full2007_2023',accounts=len(ledgers),sessions=len(returns),stress_window_is_post_hoc_known=True))
    p=pd.DataFrame(rows);print(p.loc[p.period.isin(['full','stress_2022']),['arm','period','cagr','sharpe','volatility','max_drawdown','drawdown_peak','drawdown_trough']].to_string(index=False));print('GROWTH_HOLM',positive)
if __name__=='__main__':main()
