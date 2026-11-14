from risk_learning import *
from analyze_state_v20 import load_ledger,performance,stats
from evaluate_horizons import paired_comparisons
import importlib.util
s=importlib.util.spec_from_file_location('v31_evaluator',V31/'src/evaluate.py');evaluation=importlib.util.module_from_spec(s);s.loader.exec_module(evaluation)
PERIODS=[('full','2007-01-01','2023-12-31'),('secondary_2012_2023','2012-01-01','2023-12-31'),('early','2007-01-01','2011-12-31'),('stress_2022','2022-01-01','2022-12-31'),('drawdown_2021_2023','2021-01-01','2023-12-31')]

def validate():
    seal=protocol();manifest=read(HERE/'data_manifest.json');assert manifest['protocol_sha256']==seal
    for name,h in manifest['files'].items():assert sha(HERE/'data'/name)==h,name
    source=pd.read_csv(HERE/'data/source_audit.csv',parse_dates=['date','max_input_date','window_start']);assert source.max_input_date.le(source.date).all() and source.date.max()<pd.Timestamp('2024-01-01')
    assert sha(SOURCE/'data/assembled/prices_candidate.pkl')==manifest['price_source_sha256']==read(SOURCE/'data/assembled/approval.json')['prices_sha256']
    for row in source.itertuples():
        with np.load(HERE.parent/f'v28_direct_portfolio/data/covariances/{row.date:%Y-%m-%d}.npz') as z:metadata=json.loads(str(z['metadata']))
        assert pd.Timestamp(metadata['window_start'])==row.window_start and pd.Timestamp(metadata['window_end'])==row.date and metadata['window_sessions']==63
    count=0;inner=0;mx=0.;rows=[];states={};target_checks=[]
    for arm,a in ARMS.items():
        d=data(a['kind']);ix=np.flatnonzero(d['dates']>=pd.Timestamp(d['anchor']));file=HERE/f'account_inputs/{arm}_desired.pkl';w=pd.read_pickle(file);expected=[]
        for year in range(2007,2024):
            folder=HERE if a['mode']=='retrained' else V31;stem=a['kind'] if a['mode']=='retrained' else 'log_growth';m=read(folder/f'models/{stem}_{year}.json');file_model=folder/f'models/{stem}_{year}.pt';assert sha(file_model)==m['weights_sha256'];model=Policy('replay_random');model.load_state_dict(torch.load(file_model,weights_only=True))
            cut=pd.Timestamp(year,1,1);split=cut-pd.DateOffset(years=2);sel=base.selected(d,cut,cut-pd.DateOffset(years=10));assert pd.Timestamp(m['max_training_label_end'])<cut and pd.Timestamp(m['inner_max_training_label_end'])<split
            assert m['training_query_sha256']==hashlib.sha256(d['dates'][sel].asi8.tobytes()).hexdigest();pred=predict(model,d,np.flatnonzero(d['dates'].year==year));expected.append(pred)
            if a['mode']=='retrained':
                count+=1;assert m['protocol_sha256']==seal and m['data_manifest_sha256']==sha(HERE/'data_manifest.json');f=HERE/f'account_inputs/{a["kind"]}_{year}.npy';assert sha(f)==m['targets_sha256'];saved=np.load(f);np.testing.assert_allclose(pred,saved,atol=1e-12);mx=max(mx,float(abs(pred-saved).max()))
                for c in m['selection']:
                    v=c['validation'];assert pd.Timestamp(v['end'])<cut and v['exact_repaired_engine'] and abs(v['utility']-v['annual_log_growth'])<1e-12
                    if year==2012:assert v['claim_boundary_shortened'] and v['end'].startswith('2011-12-13')
                if m['selection']:assert m['selected_penalty']==max(m['selection'],key=lambda c:c['validation']['utility'])['penalty']
                else:assert m['fallback'] and m['selected_penalty']==.1
                inner+=len(m['selection']);rows.append(dict(kind=a['kind'],year=year,penalty=m['selected_penalty'],training_periods=m['training_periods'],loss_start=m['training']['trace'][0]['loss'],loss_end=m['training']['trace'][-1]['loss']))
        expected=np.concatenate(expected);assert w.index.equals(d['dates'][ix]) and list(w.columns)==d['names'];np.testing.assert_allclose(w,expected,atol=1e-12);w=w.to_numpy()
        residual=float(abs(np.einsum('tkn,tn->tk',d['orth'][ix],w)).max());assert residual<1e-9 and abs(w.sum(1)).max()<1e-9 and abs((w*d['beta'][ix]).sum(1)).max()<1e-9 and abs(w[~d['mask'][ix]]).max(initial=0)<1e-10
        vol=np.sqrt(252*(.5*np.sum(np.einsum('tkn,tn->tk',d['centered'][ix],w)**2,axis=1)/62+.5*np.sum(d['diag'][ix]*w*w,axis=1)));assert np.max(vol-.12*d['gate'][ix])<1e-9 and abs(w).max()<=.15+1e-9 and np.max(abs(w).sum(1)-3*np.minimum(1,d['mask'][ix].sum(1)/500))<1e-9
        active=d['gate'][ix]>0;gross=abs(w).sum(1);cap=3*np.minimum(1,d['mask'][ix].sum(1)/500)
        target_checks.append(dict(arm=arm,dates=len(w),constraint_residual=residual,mean_forecast_vol=float(vol.mean()),gate_on_forecast_vol=float(vol[active].mean()),max_forecast_vol=float(vol.max()),gate_on_mean_gross=float(gross[active].mean()),gate_on_gross_cap_binding=float(np.mean(abs(gross[active]-cap[active])<1e-8)),gate_on_vol_limit_binding=float(np.mean(abs(vol[active]-.12)<1e-8))))
        m=read(HERE/f'accounts/{arm}_complete.json');assert m['protocol_sha256']==seal and m['target_sha256']==sha(file) and m['ledger_sha256']==sha(HERE/f'accounts/{arm}_ledger.csv.gz')
        assert m['audit']['arithmetic_passed'] and m['audit']['noncash_exposure_audit']['passed'];ledger=load_ledger(HERE/f'accounts/{arm}_ledger.csv.gz',arm);assert ledger.cash.min()>-1e-9;states[arm]={k:m['audit'][k] for k in ('provisional_sessions','validated_performance')}
    assert count==51 and inner==90
    q=dict(passed=True,protocol_sha256=seal,models=count,inner_validations=inner,recomputed_target_max_error=mx,target_checks=target_checks,accounts=states,strict_label_purge=True,source_hashes_unchanged=True,market_stock_window_alignment_dates=len(source),approved_price_source_hash_verified=True,additional_constraints_are_scheduled_not_daily=True)
    out=HERE/'reports';out.mkdir(exist_ok=True);write(out/'qa.json',q);pd.DataFrame(rows).to_csv(out/'fit_diagnostics.csv',index=False);return q

def main():
    q=validate();out=HERE/'reports';ledgers={a:load_ledger(HERE/f'accounts/{a}_ledger.csv.gz',a) for a in ARMS};ledgers['baseline']=load_ledger(V31/'accounts/log_growth_ledger.csv.gz','baseline');ledgers['h21_original']=load_ledger(V31/'accounts/h21_legacy_ledger.csv.gz','h21_original')
    clock=next(iter(ledgers.values())).index;assert all(f.index.equals(clock) for f in ledgers.values());spy=evaluation.market_returns();rows=[];annual=[];funding=[];diagnostics=[]
    for a,f in ledgers.items():
        previous=f.nav.shift(1,fill_value=1)
        for period,left,right in PERIODS:
            p=f.loc[left:right];r=performance(a,p,previous.loc[p.index],period,spy);r['annual_log_growth']=252*np.log1p(p.daily_return).mean();r['flat_fraction']=p.gross_exposure.lt(1e-8).mean();rows.append(r)
        for year in range(2007,2024):
            p=f.loc[str(year)];r=performance(a,p,previous.loc[p.index],str(year),spy);r['year']=year;annual.append(r)
        if a in ARMS:
            exposure=pd.read_csv(HERE/f'accounts/{a}_sector_exposure.csv.gz',index_col=0,parse_dates=True).reindex(f.index).fillna(0);np.testing.assert_allclose(exposure.sum(axis=1),f.net_exposure,atol=1e-9)
            for period,left,right in PERIODS:
                p=f.loc[left:right];e=exposure.loc[left:right];diagnostics.append(dict(arm=a,period=period,max_abs_sector_exposure=float(abs(e).max().max()),median_largest_sector_exposure=float(abs(e).max(axis=1).median()),p95_largest_sector_exposure=float(abs(e).max(axis=1).quantile(.95)),tech_mean_net=float(e.XLK.mean()),energy_mean_net=float(e.XLE.mean()),margin_calls=int(p.margin_calls.sum()),abort_batches=int(p.order_batch_aborted.sum()),provisional_sessions=int((~p.pnl_validated.astype(bool)).sum())))
        short=f.short_notional.shift(1,fill_value=0)/previous
        for bps in (0,100,300):
            adjusted=f.daily_return-short*bps/10000/252
            for period,left,right in PERIODS[:2]:funding.append(dict(arm=a,period=period,annual_borrow_bps=bps,**stats(adjusted.loc[left:right].to_numpy()),scope='Fixed-holdings return-path charge, not a funded-account replay'))
    returns=pd.DataFrame({a:f.daily_return for a,f in ledgers.items()});growth=pd.concat([evaluation.paired_growth(returns,PAIRS,b) for b in (20,60)],ignore_index=True);growth['period']='full2007_2023';secondary=pd.concat([paired_comparisons(returns,PAIRS,b,family_size=9) for b in (20,60)],ignore_index=True);secondary['period']='full2007_2023'
    positive=[]
    for a,b in PAIRS:
        z=growth.loc[growth.arm.eq(a)&growth.baseline.eq(b)]
        if z.delta_annual_log_growth.gt(0).all() and z.p_holm.lt(.05).all():positive.append([a,b])
    for name,frame in [('performance.csv',pd.DataFrame(rows)),('annual.csv',pd.DataFrame(annual)),('paired_growth_full.csv',growth),('paired_sharpe_full_secondary.csv',secondary),('borrow_sensitivity.csv',pd.DataFrame(funding)),('actual_sector_exposure.csv',pd.DataFrame(diagnostics))]:frame.to_csv(out/name,index=False)
    returns.to_csv(out/'daily_returns.csv.gz');write(out/'status.json',dict(completed=True,protocol_sha256=protocol(),primary='full2007_2023',positive_growth_holm_both_blocks=positive,promotion=False,known_2022_diagnostic=True,accounts=len(ledgers)))
    f=pd.DataFrame(rows);print(f.loc[f.period.isin(['full','stress_2022']),['arm','period','cagr','net_return','sharpe','volatility','max_drawdown']].to_string(index=False));print('GROWTH_HOLM',positive)
if __name__=='__main__':main()
