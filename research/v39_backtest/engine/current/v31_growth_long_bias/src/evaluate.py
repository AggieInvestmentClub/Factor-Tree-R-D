"""Fixed-family evaluation; no outcome-dependent portfolio or parameter choices."""
from settings import *
from analyze_state_v20 import load_ledger,performance,stats,holm
from evaluate_horizons import PERIODS,paired_comparisons
import gc

def paired_growth(returns,pairs,block):
    names=list(dict.fromkeys(n for pair in pairs for n in pair));x=np.log1p(returns[names].to_numpy());n=len(x)
    assert np.isfinite(x).all();observed=252*x.mean(0);rng=np.random.default_rng(20261111);samples=np.empty((2000,len(names)))
    for start in range(0,2000,50):
        ix=((rng.integers(0,n,(50,int(np.ceil(n/block))))[:,:,None]+np.arange(block))%n).reshape(50,-1)[:,:n]
        samples[start:start+50]=252*x[ix].mean(1)
    rows=[]
    for a,b in pairs:
        j,k=names.index(a),names.index(b);delta=observed[j]-observed[k];dist=samples[:,j]-samples[:,k]
        rows.append(dict(arm=a,baseline=b,block_length=block,delta_annual_log_growth=delta,ci_low=np.quantile(dist,.025),ci_high=np.quantile(dist,.975),
            p_one_sided=(1+np.sum(dist-delta>=delta))/2001,replications=2000,sessions=n,seed=20261111,family_size=len(pairs),interval_scope='Pointwise percentile95; not simultaneous',inference_scope='Exploratory reused history; family correction excludes prior research'))
    f=pd.DataFrame(rows);f['p_holm']=holm(f.p_one_sided.to_numpy());return f

def market_returns():
    p=ROOT/'reports/spy_returns.csv'
    source=SOURCE/'data/assembled/prices_candidate.pkl'
    if p.exists():
        assert read(p.with_suffix('.json'))['prices_sha256']==sha(source)
        return pd.read_csv(p,index_col=0,parse_dates=True).iloc[:,0]
    prices=pd.read_pickle(source);spy=prices['SPY'].loc[:'2023-12-31'].copy();del prices;gc.collect()
    assert spy.index.is_unique and spy.adj_close.gt(0).all()
    r=spy.adj_close.pct_change(fill_method=None);r.name='SPY';r.to_csv(p)
    write(p.with_suffix('.json'),dict(prices_sha256=sha(source),return_method='adj_close total-return series, descriptive post-hoc market attribution only; no future regressors used in portfolio decisions'))
    return r

def main():
    seal=protocol();spec=read(ROOT/'protocol.json');out=ROOT/'reports';out.mkdir(exist_ok=True)
    ledgers={n:load_ledger(ROOT/f'accounts/{n}_ledger.csv.gz',n) for n in arm_specs()}
    ledgers['old_rff']=load_ledger(V29/'accounts/replay_random_ledger.csv.gz','old_rff')
    first=next(iter(ledgers.values()));assert all(f.index.equals(first.index) for f in ledgers.values())
    returns=pd.DataFrame({n:f.daily_return for n,f in ledgers.items()});spy=market_returns();rows=[];annual=[];diagnostics=[];funding=[]
    gate=pd.read_csv(V23/'account_inputs/original_huber_gate_cadence21.csv',index_col=0,parse_dates=True).multiplier
    g=gate.reindex(first.index.union(gate.index)).sort_index().ffill().reindex(first.index).shift(1)
    td=pd.read_csv(out/'target_diagnostics.csv',parse_dates=['date'])
    for name,f in ledgers.items():
        prior=f.nav.shift(1,fill_value=1)
        for period,left,right in PERIODS:
            part=f.loc[left:right];r=performance(name,part,prior.loc[part.index],period,spy);r['annual_log_growth']=252*np.log1p(part.daily_return).mean()
            r['flat_fraction']=part.gross_exposure.lt(1e-8).mean();r['flat_gate_on_fraction']=(part.gross_exposure.lt(1e-8)&g.loc[part.index].gt(0)).mean();rows.append(r)
        for year in range(2007,2024):
            part=f.loc[str(year)];r=performance(name,part,prior.loc[part.index],str(year),spy);r['year']=year;annual.append(r)
        if name!='old_rff':
            m=read(ROOT/f'accounts/{name}_complete.json');assert m['protocol_sha256']==seal and m['ledger_sha256']==sha(ROOT/f'accounts/{name}_ledger.csv.gz')
            d=pd.read_csv(ROOT/f'accounts/{name}_decisions.csv.gz',parse_dates=['decision_date','execution_date']);d=d.loc[d.decision_date.between('2012-01-01','2023-12-31')]
            t=td.loc[td.arm.eq(name)&td.date.between('2012-01-01','2023-12-31')];p=f.loc['2012':'2023']
            diagnostics.append(dict(arm=name,repairs_full=m['audit']['stale_risk_repairs'],repaired_decisions_primary=int(d.stale_risk_repaired.sum()),missing_risk_cash_targets=int(d.risk_missing_target.sum()),
                mandatory_decisions=int(d.mandatory_guard.sum()),submissions=int(d.submitted.sum()),abort_batches=int(p.order_batch_aborted.sum()),margin_calls=int(p.margin_calls.sum()),
                provisional_sessions_full=m['audit']['provisional_sessions'],provisional_sessions_primary=int((~p.pnl_validated.astype(bool)).sum()),arithmetic_passed=m['audit']['arithmetic_passed'],validated_performance=m['audit']['validated_performance'],
                noncash_audit_passed=m['audit']['noncash_exposure_audit']['passed'],estimated_beta_mean=d.actual_estimated_beta.mean(),estimated_beta_min=d.actual_estimated_beta.min(),estimated_beta_max=d.actual_estimated_beta.max(),
                mean_target_vol=t.forecast_vol.mean(),gate_on_mean_target_vol=t.loc[t.gate.gt(0),'forecast_vol'].mean(),mean_target_bias=t.effective_bias.mean(),min_target_bias=t.effective_bias.min(),
                mean_alpha_scale=t.alpha_scale.mean(),max_realized_gross=p.gross_exposure.max(),min_cash=p.cash.min(),primary_flat_fraction=p.gross_exposure.lt(1e-8).mean()))
        # Holding paths are kept fixed: sensitivity is not a full funded-account rerun.
        short=f.short_notional.shift(1,fill_value=0)/prior
        for bps in (0,100,300):
            adjusted=f.daily_return-short*bps/10000/252
            for period,left,right in PERIODS[:3]:
                funding.append(dict(arm=name,period=period,annual_borrow_bps=bps,**stats(adjusted.loc[left:right].to_numpy()),method='Approximate return-path charge on prior close short notional; fixed holdings, no NAV/guard feedback or stock-specific availability; zero cash and rebate interest'))
    decomposition=[]
    for old,new in [('old_rff','rff_repaired'),('h21_legacy','h21_repaired')]:
        o=ledgers[old].loc['2012':'2023'];n=ledgers[new].loc[o.index];gate_on=g.loc[o.index].gt(0);flat=o.gross_exposure.lt(1e-8)
        for label,mask in [('gate_off',~gate_on),('old_flat_gate_on',flat&gate_on),('old_invested_gate_on',~flat&gate_on)]:
            old_contribution=252*o.loc[mask,'daily_return'].sum()/len(o);new_contribution=252*n.loc[mask,'daily_return'].sum()/len(o)
            decomposition.append(dict(original=old,repaired=new,category=label,sessions=int(mask.sum()),annual_old_mean_contribution=old_contribution,annual_repaired_mean_contribution=new_contribution,delta_annual_mean_contribution=new_contribution-old_contribution,scope='Additive arithmetic decomposition of paired return paths, using original closing cash state; path changes propagate beyond restored days, so not a causal missed-opportunity estimate'))
    pd.DataFrame(decomposition).to_csv(out/'exposure_decomposition.csv',index=False)
    perf=pd.DataFrame(rows);growth=pd.concat([paired_growth(returns.loc['2012':'2023'],spec['pairs'],b) for b in (20,60)],ignore_index=True)
    secondary=pd.concat([paired_comparisons(returns.loc['2012':'2023'],spec['pairs'],b,family_size=len(spec['pairs'])) for b in (20,60)],ignore_index=True)
    for file,f in [('performance.csv',perf),('annual.csv',pd.DataFrame(annual)),('account_diagnostics.csv',pd.DataFrame(diagnostics)),('borrow_sensitivity.csv',pd.DataFrame(funding)),('paired_growth.csv',growth),('paired_sharpe_secondary.csv',secondary)]:f.to_csv(out/file,index=False)
    returns.to_csv(out/'daily_returns.csv.gz');positive=[]
    for a,b in spec['pairs']:
        q=growth.loc[growth.arm.eq(a)&growth.baseline.eq(b)]
        if q.delta_annual_log_growth.gt(0).all() and q.p_holm.lt(.05).all():positive.append([a,b])
    write(out/'status.json',dict(completed=True,protocol_sha256=seal,accounts=len(ledgers),primary_sessions=len(returns.loc['2012':'2023']),positive_growth_holm_both_blocks=positive,promotion=False,primary_endpoint='Annual log growth; CAGR is its exponential minus1',borrow_sensitivity_is_approximate=True,zero_cash_rate_sharpe=True,conditional_source_valuations_retained=True))
    print(perf.loc[perf.period.eq('primary'),['arm','cagr','sharpe','volatility','max_drawdown','spy_beta','flat_fraction']].to_string(index=False))
    print('GROWTH_HOLM_BOTH_BLOCKS',positive)
if __name__=='__main__':main()
