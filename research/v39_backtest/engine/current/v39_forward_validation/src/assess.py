"""One frozen paired comparison, with source uncertainty kept visible."""
from portfolio import c,annual_variance,unit_market,data,selected
import numpy as np,pandas as pd,json,importlib.util,hashlib
c.sys.path.insert(0,str(c.SOURCE/'src'))
from analyze_state_v20 import stats,performance
c.sys.path.insert(0,str(c.PROJECT/'work'))
from research_reporting import table,document,plt

def ledger(arm):
    f=pd.read_csv(c.HERE/f'accounts/{arm}_ledger.csv.gz',index_col=0,parse_dates=True)
    assert f.index.min()==pd.Timestamp(c.START) and f.index.max()==pd.Timestamp(c.END) and f.index.is_unique and f.index.is_monotonic_increasing
    assert np.isfinite(f[['nav','daily_return','costs','turnover','long_net_pnl','short_net_pnl']]).all().all()
    assert f.nav.gt(0).all() and f.daily_return.gt(-1).all() and f.cash.min()>-1e-9
    np.testing.assert_allclose((1+f.daily_return).cumprod(),f.nav,atol=1e-10,rtol=0)
    np.testing.assert_allclose((f.long_net_pnl+f.short_net_pnl)/f.nav.shift(1,fill_value=1),f.daily_return,atol=1e-9,rtol=0)
    return f

def validate():
    design=c.read(c.HERE/'design.json');seal=c.sha(c.HERE/'design.json');assert seal=='203d5d63df27df26c0ca3d36b25ec9d4df58c8f21b4dc0ee3c57824e5ab469cc'
    for path,h in design['source_hashes'].items():assert c.sha(path)==h,path
    for name in ('feature','label','gate','array'):assert c.read(c.HERE/f'data/{name}_qa.json')['passed']
    assert c.read(c.HERE/'data/gate_numerical_qa.json')['passed']
    source=c.read(c.HERE/'data/assembled/approval.json');assert source['source_qualification_passed']
    training=data()
    for year in (2024,2025,2026):
        m=c.read(c.HERE/f'models/log_growth_{year}.json');cut=pd.Timestamp(year,1,1)
        assert m['design_sha256']==seal and c.sha(c.HERE/f'models/log_growth_{year}.pt')==m['weights_sha256']
        chosen=selected(training,cut,cut-pd.DateOffset(years=10));assert len(chosen)==m['training_periods'];assert hashlib.sha256(training['dates'][chosen].asi8.tobytes()).hexdigest()==m['training_query_sha256']
        assert pd.Timestamp(m['max_training_label_end'])<cut and pd.Timestamp(m['inner_max_training_label_end'])<cut-pd.DateOffset(years=2)
        h=c.read(c.HERE/f'forecasts/huber_{year}.csv.json');assert pd.Timestamp(h['max_training_label_end'])<cut
    desired=pd.read_pickle(c.HERE/'account_inputs/desired.pkl');names=list(desired.columns);meta=c.read(c.HERE/'data/model/metadata.json');clock=pd.DatetimeIndex(meta['calendar']);sdates=pd.DatetimeIndex(meta['dates'])
    returns=np.load(c.HERE/'data/model/qualified_returns.npy',mmap_mode='r');loads=np.load(c.HERE/'data/model/factor_loadings.npy',mmap_mode='r');beta=pd.read_pickle(c.HERE/'data/model/daily_beta.pkl');eligible=pd.read_pickle(c.HERE/'data/model/daily_eligible.pkl')
    vector=lambda text:pd.Series(json.loads(text),dtype=float).reindex(names,fill_value=0).to_numpy()
    accounts={};count=0;maxrisk=0.;maxfactor=0.;noncash=[]
    for arm in ('benchmark','vol2sigma'):
        m=c.read(c.HERE/f'accounts/{arm}_complete.json');a=m['audit'];assert m['design_sha256']==seal and m['ledger_sha256']==c.sha(c.HERE/f'accounts/{arm}_ledger.csv.gz') and a['arithmetic_passed']
        assert not a['unresolved_terminal_assets'];assert all(i['kind'] in ('unresolved_conditional_corporate_value','unresolved_distribution_claim_mark') for i in a['issues']),a['issues']
        f=ledger(arm);assert not f[['unpriced_held_open','unpriced_held_close','held_segment_breaks']].to_numpy().any()
        assert not f.margin_calls.any() and not f.order_batch_aborted.any()
        d=pd.read_csv(c.HERE/f'accounts/{arm}_decisions.csv.gz',parse_dates=['decision_date','execution_date'])
        assert d.execution_date.gt(d.decision_date).all()
        assert all(clock[clock.searchsorted(day,side='right')]==execution for day,execution in zip(d.decision_date,d.execution_date))
        path=c.HERE/f'accounts/{arm}_noncash_positions.csv.gz';p=pd.read_csv(path,parse_dates=['Date']);census=pd.read_csv(c.HERE/'data/assembled/noncash_census.csv',parse_dates=['date']);assert len(p)==len(census)
        for row in p.itertuples():
            assert row.close_quote_valid and not row.mark_provisional and not row.segment_break
            assert abs(row.distribution_reinvestment_cost-.0002*row.distribution_reinvestment_notional)<1e-12
            noncash.append(dict(arm=arm,date=row.Date,ticker=row.ticker,previous_closing_shares=row.previous_closing_shares,cost=row.distribution_reinvestment_cost))
        assert abs(p.distribution_reinvestment_cost.sum()-f.distribution_reinvestment_cost.sum())<1e-10
        accounts[arm]=dict(validated_performance=a['validated_performance'],provisional_sessions=a['provisional_sessions'],conditional_rights=[i for i in a['issues'] if i['kind']=='unresolved_conditional_corporate_value'],conditional_claim_marks=[i for i in a['issues'] if i['kind']=='unresolved_distribution_claim_mark'],delayed_distribution_events=a['delayed_distribution_events'],noncash_events=len(p),unmodeled_later_deletions_exposure_zero=True,source_quote_issues=0,aborted_order_batches=0,margin_calls=0)
        if arm=='benchmark':continue
        t=pd.read_csv(c.HERE/f'accounts/{arm}_triggers.csv.gz',parse_dates=['date']);assert len(t)==m['trigger_count'] and not t.duplicated(['date','ticker']).any()
        np.testing.assert_allclose(t.threshold,np.clip(2*t.entry_sigma21,.03,.20),atol=1e-12,rtol=0);assert t.pnl.le(-t.threshold+1e-12).all()
        groups={date:list(z.ticker) for date,z in t.groupby('date')};blocked=set()
        for row in d.itertuples():
            day=row.decision_date
            if day in desired.index:blocked.clear()
            new=groups.get(day,[]);assert not blocked.intersection(new);blocked.update(new);detail=json.loads(row.position_exit_diagnostics_json)
            assert detail['blocked']==len(blocked) and detail['new_exits']==len(new)
            if detail['force_exit']:assert row.submitted and row.mandatory_guard
            w=vector(row.base_weights_json);blocked_mask=np.array([n in blocked for n in names]);assert abs(w[blocked_mask]).max(initial=0)<1e-10 and abs(w).max()<=.15+1e-9 and -1e-9<=w.sum()<=.5+1e-9
            if pd.notna(row.execution_target_weights_json):assert abs(vector(row.execution_target_weights_json)[blocked_mask]).max(initial=0)<1e-10
            if not blocked or not abs(w).sum():continue
            end=clock.get_loc(day)+1;x=returns[end-63:end];valid=np.isfinite(x).all(0);ix=np.flatnonzero(valid);centered=x[:,ix]-x[:,ix].mean(0);b=beta.loc[day].to_numpy();mask=eligible.loc[day].to_numpy(bool)&np.isfinite(b)&valid&~blocked_mask
            assert abs(w[~mask]).max(initial=0)<1e-10;vol=np.sqrt(annual_variance(w,centered,ix));assert vol<=.12+1e-9 and abs(w).sum()<=detail['gross_budget']+1e-9
            assert abs(w[np.isfinite(b)]@b[np.isfinite(b)]-w.sum())<1e-9;si=sdates.searchsorted(day,side='right')-1;error=float(abs(loads[si]@(w-unit_market(b,mask)*w.sum())).max());assert error<1e-8
            maxrisk=max(maxrisk,vol);maxfactor=max(maxfactor,error);count+=1
    pd.DataFrame(noncash).to_csv(c.HERE/'reports/noncash_account_exposure.csv',index=False)
    q=dict(passed=True,design_sha256=seal,accounts=accounts,prior_sources_immutable=True,annual_training_strictly_purged=True,strict_next_close=True,exact_volatility_thresholds=True,blocked_names_zero=True,post_exit_risk_checks=count,maximum_forecast_risk=maxrisk,maximum_alpha_factor_error=maxfactor,conditional_source_values_disclosed=True)
    c.write(c.HERE/'reports/qa.json',q);return q

def bootstrap(returns,block):
    # Identical paired circular-block procedure and seed as the prior experiments.
    x=np.log1p(returns[['vol2sigma','benchmark']].to_numpy());n=len(x);delta=252*(x[:,0]-x[:,1]).mean();rng=np.random.default_rng(20261111);samples=[]
    for _ in range(40):
        ix=((rng.integers(0,n,(50,int(np.ceil(n/block))))[:,:,None]+np.arange(block))%n).reshape(50,-1)[:,:n];samples.extend(252*(x[ix,0]-x[ix,1]).mean(1))
    dist=np.asarray(samples)
    return dict(arm='vol2sigma',baseline='benchmark',block_length=block,delta_annual_log_growth=delta,ci_low=np.quantile(dist,.025),ci_high=np.quantile(dist,.975),p_one_sided=(1+sum(dist-delta>=delta))/2001,replications=2000,sessions=n,family_size=1,seed=20261111,scope='Frozen later-period walk-forward comparison; historical project previously examined this era; intervals conditional on source conventions')

def main():
    out=c.HERE/'reports';out.mkdir(exist_ok=True);q=validate();ledgers={a:ledger(a) for a in ('benchmark','vol2sigma')};returns=pd.DataFrame({a:f.daily_return for a,f in ledgers.items()});spy=pd.read_csv(c.HERE/'data/model/market_returns.csv',index_col=0,parse_dates=True).iloc[:,0];rows=[];annual=[];diagnostics=[];borrow=[];rights=[]
    for arm,f in ledgers.items():
        prior=f.nav.shift(1,fill_value=1);d=pd.read_csv(c.HERE/f'accounts/{arm}_decisions.csv.gz',parse_dates=['decision_date']);t=pd.read_csv(c.HERE/f'accounts/{arm}_triggers.csv.gz',parse_dates=['date']);t['date']=pd.to_datetime(t.date)
        for period,left,right in [('full',c.START,c.END),('2024','2024-01-01','2024-12-31'),('2025','2025-01-01','2025-12-31'),('2026 partial','2026-01-01',c.END)]:
            z=f.loc[left:right];row=performance(arm,z,prior.loc[z.index],period,spy);row['annual_log_growth']=252*np.log1p(z.daily_return).mean();(rows if period=='full' else annual).append(row);tr=t[t.date.between(left,right)];dec=d[d.decision_date.between(left,right)]
            diagnostics.append(dict(arm=arm,period=period,triggers=len(tr),long_exits=int(tr.side.gt(0).sum()),short_exits=int(tr.side.lt(0).sum()),mean_gross=z.gross_exposure.mean(),mean_net=z.net_exposure.mean(),mean_cash_fraction=(z.cash/z.nav).mean(),flat_fraction=z.gross_exposure.lt(1e-8).mean(),mean_estimated_beta=dec.actual_estimated_beta.mean(),provisional_sessions=int((~z.pnl_validated.astype(bool)).sum()),submissions=int(dec.submitted.sum()),terminal_cost=z.terminal_cost.sum()))
        short=f.short_notional.shift(1,fill_value=0)/prior
        for bps in (0,100,300):borrow.append(dict(arm=arm,annual_borrow_bps=bps,**stats((f.daily_return-short*bps/10000/252).to_numpy()),method='Approximate fixed-holdings path charge; no NAV/guard feedback, stock-specific availability or cash interest'))
        low=0.;high=0.
        for issue in q['accounts'][arm]['conditional_rights']:
            value=issue['raw_share_equivalent']*issue['cvr_upper'];low+=min(0,value);high+=max(0,value);rights.append(dict(arm=arm,**issue,signed_cap_cash=value))
        q['accounts'][arm]['rights_fixed_holdings_terminal_nav_bounds']=[float(f.nav.iloc[-1]+low),float(f.nav.iloc[-1]+high)]
    growth=pd.DataFrame([bootstrap(returns,b) for b in (20,60)])
    for name,z in [('performance',rows),('annual',annual),('diagnostics',diagnostics),('borrow_sensitivity',borrow),('conditional_rights',rights),('paired_growth',growth)]:pd.DataFrame(z).to_csv(out/f'{name}.csv',index=False)
    returns.to_csv(out/'daily_returns.csv.gz');c.write(out/'qa.json',q)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,constrained_layout=True)
    for arm,color,label in [('benchmark','#245879','Benchmark'),('vol2sigma','#d07832','Volatility stop')]:
        nav=ledgers[arm].nav;dd=nav/nav.cummax().clip(lower=1)-1;axes[0].plot(nav.index,nav,label=label,color=color);axes[1].plot(dd.index,100*dd,color=color)
    axes[0].set(title='Frozen forward comparison: January 2024–November 11, 2026',ylabel='Growth of $1');axes[0].legend();axes[1].set(ylabel='Drawdown (%)')
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'forward_comparison.png',dpi=160);plt.close(fig)
    perf=pd.DataFrame(rows);ann=pd.DataFrame(annual);diag=pd.DataFrame(diagnostics);a=perf.set_index('arm');delta=a.loc['vol2sigma','cagr']-a.loc['benchmark','cagr'];supported=bool(growth.delta_annual_log_growth.gt(0).all() and growth.p_one_sided.lt(.05).all())
    text=f'<p>The volatility stop changed CAGR by {delta*100:+.2f} percentage points against the frozen benchmark. Evidence for higher log growth is {"positive at both declared block lengths" if supported else "not conclusive at both declared block lengths"}. No parameters were changed in response to these results.</p>'
    text+='<p class="note">Later-period walk-forward validation, not a pristine holdout: an older 110-stock project previously examined 2024 onward. The current models and stop settings were fixed using pre-2024 work. Models refit annually using only matured prior data. Both accounts start in cash on January 2, 2024. November 11, 2026 is the last completed session; 2026 is partial.</p>'
    text+=table(perf[['arm','cagr','net_return','sharpe','volatility','max_drawdown','spy_beta','turnover_per_year','cost_per_year']],percent=['cagr','net_return','volatility','max_drawdown','cost_per_year'])+'<img src="forward_comparison.png" alt="Forward account equity and drawdown paths">'
    text+='<h2>Calendar years</h2>'+table(ann[['arm','period','sessions','net_return','sharpe','max_drawdown']],percent=['net_return','max_drawdown'])
    text+='<h2>Paired uncertainty</h2><p>One declared stop-versus-benchmark comparison; 2,000 paired circular-block bootstrap samples with 20- and 60-session blocks. Confidence intervals are pointwise, conditional on these data and accounting conventions; they do not correct for the broader history of model development. Drawdown differences are descriptive.</p>'+table(growth[['block_length','delta_annual_log_growth','ci_low','ci_high','p_one_sided']],percent=['delta_annual_log_growth','ci_low','ci_high'])
    text+='<h2>Trading and exposure</h2>'+table(diag,percent=['mean_net','mean_cash_fraction','flat_fraction'])
    text+='<p>Same 21-session forecast schedule, annual 10-year refits, three correlation-factor constraints on alpha, maximum +0.50 target beta/net exposure, 12% forecast risk, 15% target name cap and unchanged gross/cash guards. Stop threshold is twice entry-time 63-session volatility scaled by sqrt(21), clipped to 3–20%. A close trigger executes at the next close; prices can pass the threshold. Stopped names remain excluded until the next original forecast. Surviving positions are reprojected under the original risk and exposure limits.</p>'
    text+='<h2>Source and funding limits</h2><p>Daily observations and corporate-action terms are historical reconstructions captured now, not a full vintage vendor archive. Original pre-2024 price, feature and risk prefixes are preserved and checked. Spin-off wealth is reconciled from separately dated raw prices and issuer terms, using the inherited synthetic reinvestment convention and two trading legs; broker settlement delays and fractional-share cash are not modeled. Merger stock is liquidated at the recipient regular-session open. The final session uses inherited account liquidation and its fees. The backtest assumes zero stock-borrow, financing and cash interest; Sharpe uses a zero cash rate.</p>'
    text+=table(pd.DataFrame(borrow)[['arm','annual_borrow_bps','cagr','sharpe','max_drawdown']],percent=['cagr','max_drawdown'])
    text+='<p>Borrow sensitivity holds the trading path fixed; it is not a funded account rerun. Nontransferable contingent rights, if held, use cash-only base NAV and explicit 0-to-contractual-cap bounds. Those bounds hold positions fixed and do not model subsequent trading feedback or imply a market value.</p>'
    if rights:text+=table(pd.DataFrame(rights)[['arm','date','ticker','raw_share_equivalent','cvr_upper','signed_cap_cash']])
    else:text+='<p>Neither account held a contingent-right event at its conversion date.</p>'
    claims=any(a['conditional_claim_marks'] for a in q['accounts'].values())
    if claims:text+='<p class="note">The accounts include a Lennar/Millrose entitlement whose traded value was unavailable at its January 21, 2025 ex-date. The inherited claim book carries a conditional zero mark until liquidation at the first regular-session open on February 7. Interim NAV, stops, Sharpe and drawdown are therefore scenario results, not fully observed wealth. The claim survives sale of the parent; no settlement price is backfilled into an earlier decision.</p>'
    text+='<h2>Audit</h2><pre>'+json.dumps(q,indent=2,default=str)+'</pre><p>Machine-readable tables, account ledgers, source terms, timestamp checks and the frozen design are retained beside this report. No model has been promoted automatically.</p>'
    document('Benchmark versus volatility stop: forward validation',text,out/'report.html')
    c.write(out/'status.json',dict(completed=True,design_sha256=c.sha(c.HERE/'design.json'),start=c.START,end=c.END,sessions=len(returns),paired_growth_positive_both_blocks=supported,promotion=False,pristine_holdout=False,qa_passed=True,conditional_source_values=bool(rights) or claims,report_sha256=c.sha(out/'report.html')))
    print(perf[['arm','cagr','net_return','sharpe','max_drawdown']].to_string(index=False),flush=True)
if __name__=='__main__':main()
