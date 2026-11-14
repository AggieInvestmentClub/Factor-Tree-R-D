from exits import *
import json
from analyze_state_v20 import load_ledger,performance,stats
sys.path.insert(0,str(HERE.parent/'v35_long_bias_drawdown/src'))
s=importlib.util.spec_from_file_location('v35_metrics',HERE.parent/'v35_long_bias_drawdown/src/assess.py');shared=importlib.util.module_from_spec(s);s.loader.exec_module(shared)

def validate():
    label_audit=read(V23/'reports/labels/label_audit.json');assert label_audit['status']=='complete' and label_audit['label_pickle_sha256']==sha(V23/'data/horizon_labels.pkl') and label_audit['source_hashes']['prices']==sha(SOURCE/'data/assembled/prices_candidate.pkl')
    seal=protocol();desired=pd.read_pickle(V36/'account_inputs/fixed_corr_long50_desired.pkl');names=list(desired.columns);meta=read(V33/'account_inputs/return_clock.json');clock=pd.DatetimeIndex(meta['dates']);assert names==meta['names'];assert meta['source_price_sha256']==sha(SOURCE/'data/assembled/prices_candidate.pkl')==read(SOURCE/'data/assembled/approval.json')['prices_sha256'];returns=np.load(V33/'account_inputs/qualified_returns.npy',mmap_mode='r');beta=pd.read_pickle(SOURCE/'account_inputs/betas.pkl').reindex(columns=names);eligible=pd.read_pickle(SOURCE/'account_inputs/eligible.pkl').reindex(columns=names);loads=np.load(V34/'data/factor_loadings.npy');sdates=pd.DatetimeIndex(read(V34/'data_manifest.json')['dates']);accounts={};maxrisk=0.;maxfactor=0.;count=0;trigger_count=0
    def vector(text):return pd.Series(json.loads(text),dtype=float).reindex(names,fill_value=0).to_numpy()
    for arm in ARMS:
        m=read(HERE/f'accounts/{arm}_complete.json');assert m['protocol_sha256']==seal and m['ledger_sha256']==sha(HERE/f'accounts/{arm}_ledger.csv.gz') and m['audit']['arithmetic_passed'] and m['audit']['noncash_exposure_audit']['passed'];f=load_ledger(HERE/f'accounts/{arm}_ledger.csv.gz',arm);assert f.cash.min()>-1e-9 and f.index.max()<pd.Timestamp('2024-01-01');d=pd.read_csv(HERE/f'accounts/{arm}_decisions.csv.gz',parse_dates=['decision_date','execution_date']);assert d.execution_date.gt(d.decision_date).all();assert all(clock[clock.searchsorted(date,side='right')]==execution if execution<=clock[-1] else execution==pd.Timestamp('2023-12-29') for date,execution in zip(d.decision_date,d.execution_date));accounts[arm]={k:m['audit'][k] for k in ('provisional_sessions','validated_performance')}
        if arm=='none':
            assert m['baseline_parity_error']<1e-10;ref=load_ledger(V36/'accounts/fixed_corr_long50_ledger.csv.gz','ref');np.testing.assert_allclose(f.nav,ref.nav,atol=1e-12);continue
        t=pd.read_csv(HERE/f'accounts/{arm}_triggers.csv.gz',parse_dates=['date']);trigger_count+=len(t);assert len(t)==m['trigger_count'] and not t.duplicated(['date','ticker']).any();groups={date:list(z.ticker) for date,z in t.groupby('date')};blocked=set()
        if arm=='fixed5':assert t.pnl.le(-.05+1e-12).all()
        elif arm=='trailing8':assert t.trail.ge(.08-1e-12).all()
        elif arm=='bracket5_10':assert (t.pnl.le(-.05+1e-12)|t.pnl.ge(.10-1e-12)).all()
        elif arm=='vol2sigma':assert t.pnl.le(-.03+1e-12).all()
        else:
            before=t.date.dt.year.lt(2012);assert t.loc[before,'pnl'].le(-.05+1e-12).all();assert t.loc[~before,'age'].ge(5).all() and t.loc[~before,'predicted_hold_advantage'].lt(-.002).all()
        for row in d.itertuples():
            date=row.decision_date
            if date in desired.index:blocked.clear()
            new=groups.get(date,[]);assert not blocked.intersection(new);blocked.update(new);detail=json.loads(row.position_exit_diagnostics_json);assert detail['blocked']==len(blocked) and detail['new_exits']==len(new)
            if detail['force_exit']:assert row.submitted and row.mandatory_guard
            w=vector(row.base_weights_json);maskblocked=np.array([n in blocked for n in names]);assert abs(w[maskblocked]).max(initial=0)<1e-10;assert abs(w).max()<=.15+1e-9;assert -1e-9<=w.sum()<=.5+1e-9
            if pd.notna(row.execution_target_weights_json):assert abs(vector(row.execution_target_weights_json)[maskblocked]).max(initial=0)<1e-10
            if not blocked or not abs(w).sum():continue
            end=clock.get_loc(date)+1;x=returns[end-63:end];valid=np.isfinite(x).all(0);ix=np.flatnonzero(valid);c=x[:,ix]-x[:,ix].mean(0);b=beta.loc[date].to_numpy();mask=eligible.loc[date].to_numpy(bool)&np.isfinite(b)&valid&~maskblocked;assert abs(w[~mask]).max(initial=0)<1e-10;v=np.sqrt(annual_variance(w,c,ix));assert v<=.12+1e-9 and abs(w).sum()<=detail['gross_budget']+1e-9;assert abs(w[np.isfinite(b)]@b[np.isfinite(b)]-w.sum())<1e-9;unit=unit_market(b,mask);si=sdates.searchsorted(date,side='right')-1;residual=float(abs(loads[si]@(w-unit*w.sum())).max());assert residual<1e-8;maxrisk=max(maxrisk,v);maxfactor=max(maxfactor,residual);count+=1
    training=read(HERE/'reports/tree_training.json');assert len(training['models'])==12
    for m in training['models']:
        cut=pd.Timestamp(m['year'],1,1);assert pd.Timestamp(m['max_label_end'])<cut and pd.Timestamp(m['training_end'])<cut and pd.Timestamp(m['training_start'])>=cut-pd.DateOffset(years=5);assert m['depth']<=2 and m['rows']>=10000 and m['features']==FEATURES;assert sha(HERE/f"models/tree_{m['year']}.pkl")==m['model_sha256'] and m['protocol_sha256']==seal
    q=dict(passed=True,protocol_sha256=seal,accounts=accounts,baseline_exact_parity=True,strict_next_close=True,stop_trigger_records=trigger_count,blocked_target_vectors_checked=count,maximum_post_exit_forecast_vol=maxrisk,maximum_post_exit_alpha_factor_error=maxfactor,blocked_names_zero_before_and_after_execution=True,tree_models=12,strict_mature_labels=True,source_hashes_unchanged=True,execution='Close signal,next close fill; not intraday stop guarantee');write(HERE/'reports/qa.json',q);return q

def main():
    q=read(HERE/'reports/qa.json') if '--metrics-only' in sys.argv else validate();assert q['passed'] and q['protocol_sha256']==protocol();out=HERE/'reports';ledgers={a:load_ledger(HERE/f'accounts/{a}_ledger.csv.gz',a) for a in ARMS};spy=shared.evaluation.market_returns();rows=[];annual=[];diagnostics=[];borrow=[];triggerannual=[]
    for a,f in ledgers.items():
        previous=f.nav.shift(1,fill_value=1);d=pd.read_csv(HERE/f'accounts/{a}_decisions.csv.gz',parse_dates=['decision_date']);triggers=pd.read_csv(HERE/f'accounts/{a}_triggers.csv.gz',parse_dates=['date']);triggers['date']=pd.to_datetime(triggers.date)
        for period,left,right in shared.PERIODS:
            z=f.loc[left:right];r=performance(a,z,previous.loc[z.index],period,spy);r.update(shared.underwater(z.daily_return));r['annual_log_growth']=252*np.log1p(z.daily_return).mean();rows.append(r);dec=d.loc[d.decision_date.between(left,right)];t=triggers.loc[triggers.date.between(left,right)];diagnostics.append(dict(arm=a,period=period,triggers=len(t),long_exits=int(t.side.gt(0).sum()),short_exits=int(t.side.lt(0).sum()),submissions=int(dec.submitted.sum()),mean_target_beta=dec.reference_beta.mean(),mean_actual_beta=dec.actual_estimated_beta.mean(),mean_gross=z.gross_exposure.mean(),mean_net=z.net_exposure.mean(),mean_cash_fraction=(z.cash/z.nav).mean(),provisional_sessions=int((~z.pnl_validated.astype(bool)).sum()),margin_calls=int(z.margin_calls.sum()),aborted_batches=int(z.order_batch_aborted.sum())))
        for year in range(2007,2024):
            z=f.loc[str(year)];r=performance(a,z,previous.loc[z.index],str(year),spy);r['year']=year;annual.append(r);t=triggers.loc[triggers.date.dt.year.eq(year)];triggerannual.append(dict(arm=a,year=year,triggers=len(t),long_exits=int(t.side.gt(0).sum()),short_exits=int(t.side.lt(0).sum())))
        short=f.short_notional.shift(1,fill_value=0)/previous
        for bps in (0,100,300):
            adjusted=f.daily_return-short*bps/10000/252
            for period,left,right in shared.PERIODS[:2]:borrow.append(dict(arm=a,period=period,annual_borrow_bps=bps,**stats(adjusted.loc[left:right].to_numpy())))
    labels=pd.read_pickle(V23/'data/horizon_labels.pkl')[['decision_date','ticker','future_return_h21','label_end_h21']];opportunity=[]
    for arm in ARMS[1:]:
        t=pd.read_csv(HERE/f'accounts/{arm}_triggers.csv.gz',parse_dates=['date']).rename(columns={'date':'decision_date'});t=t.merge(labels,on=['decision_date','ticker'],how='left',validate='one_to_one');t['signed_future_h21']=t.side*t.future_return_h21
        for period,left,right in shared.PERIODS:
            for side in (-1,1):
                z=t.loc[t.decision_date.between(left,right)&t.side.eq(side)];known=z.signed_future_h21.dropna();opportunity.append(dict(arm=arm,period=period,side=side,triggers=len(z),mature_labels=len(known),coverage=len(known)/max(1,len(z)),mean_signed_h21_after_exit=known.mean(),median_signed_h21_after_exit=known.median(),positive_fraction=known.gt(0).mean(),interpretation='Unweighted underlying signed return from next-close exit to21sessions later; descriptive opportunity cost, not realized portfolio PnL or a training feature.'))
    pd.DataFrame(opportunity).to_csv(out/'exit_opportunity.csv',index=False)
    returns=pd.DataFrame({a:f.daily_return for a,f in ledgers.items()});growth=pd.concat([shared.evaluation.paired_growth(returns.loc[left:right],PAIRS,b).assign(period=period) for period,left,right in shared.PERIODS[:2] for b in (20,60)],ignore_index=True);passing={}
    for period in ('primary','full'):passing[period]=[[a,b] for a,b in PAIRS if (lambda z:z.delta_annual_log_growth.gt(0).all() and z.p_holm.lt(.05).all())(growth.loc[growth.period.eq(period)&growth.arm.eq(a)&growth.baseline.eq(b)])]
    for name,frame in [('performance.csv',rows),('annual.csv',annual),('account_diagnostics.csv',diagnostics),('borrow_sensitivity.csv',borrow),('trigger_annual.csv',triggerannual),('paired_growth.csv',growth)]:pd.DataFrame(frame).to_csv(out/name,index=False)
    returns.to_csv(out/'daily_returns.csv.gz');write(out/'status.json',dict(completed=True,protocol_sha256=protocol(),promotion=False,primary='2012-2023',full='2007-2023',growth_holm_both_blocks=passing,accounts=len(ARMS),reused_history=True,drawdown_descriptive=True,not_intraday_stops=True,not_globally_optimal_tree=True));p=pd.DataFrame(rows);print(p.loc[p.period.isin(['primary','full','stress_2022']),['arm','period','cagr','net_return','sharpe','max_drawdown']].to_string(index=False));print('GROWTH_HOLM',passing)

if __name__=='__main__':main()
