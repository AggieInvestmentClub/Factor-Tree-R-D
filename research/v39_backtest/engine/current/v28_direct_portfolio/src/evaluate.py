"""Frozen six-contrast evaluation and independent data/model/account checks."""
from core import *
import pandas as pd
from accounts import contract,source_gate
from evaluate_horizons import paired_comparisons,PERIODS
from analyze_state_v20 import load_ledger,performance

def validate():
    from fit_policy import load_data,choose_dates
    data=load_data();spec=read(ROOT/'protocol.json');source=contract();records=[]
    for year in range(2007,2024):
        cut=pd.Timestamp(year,1,1);selected=choose_dates(data,cut,cut-pd.DateOffset(years=10))
        expected=hashlib.sha256(data['dates'][selected].asi8.tobytes()).hexdigest()
        for name in (*MODELS,'sample_aligned_h21'):
            m=read(ROOT/f'models/{name}_{year}.json')
            assert m['protocol_sha256']==protocol() and m['data_manifest_sha256']==sha(ROOT/'data_manifest.json')
            assert m['training_query_sha256']==expected and pd.Timestamp(m['max_training_label_end'])<cut
            assert m['training_periods']==len(selected)
            if name in MODELS:
                assert sha(ROOT/f'models/{name}_{year}.npy')==m['weights_sha256']
                assert sha(ROOT/f'account_inputs/{name}_{year}.npy')==m['targets_sha256']
                assert pd.Timestamp(m['inner_max_label_end'])<pd.Timestamp(m['validation_start'])
                assert m['solver']['success'] and m['solver']['gap']<=2e-5
                theta=np.load(ROOT/f'models/{name}_{year}.npy');assert abs(theta).sum()<=1+1e-10
                records.append(dict(arm=name,year=year,penalty=m['selected_penalty'],coefficient_l1=float(abs(theta).sum()),
                    gap=m['solver']['gap'],training_periods=len(selected),validation_periods=m['validation_dates']))
            else:
                assert sha(ROOT/f'models/{name}_{year}.pkl')==m['model_sha256']
                f=ROOT/f'forecast_runs/{name}_{year}.csv.gz';assert sha(f)==m['forecast_sha256']
                forecast=pd.read_csv(f,parse_dates=['decision_date','refit_boundary','max_training_label_end'])
                assert forecast.decision_date.dt.year.eq(year).all() and np.isfinite(forecast.score).all()
                assert forecast.max_training_label_end.lt(forecast.refit_boundary).all()
    pd.DataFrame(records).to_csv(ROOT/'reports/annual_model_diagnostics.csv',index=False)
    counts={};gate=source_gate();ix=np.flatnonzero(data['dates']>=pd.Timestamp(data['anchor']))
    for arm in ARMS:
        m=read(ROOT/f'accounts/{arm}_complete.json');p=read(ROOT/f'account_inputs/{arm}_prepared.json')
        for k,v in source.items():assert m[k]==p[k]==v,k
        f=ROOT/f'account_inputs/{arm}_desired.pkl';assert sha(f)==m['target_sha256']==p['target_sha256']
        target=pd.read_pickle(f);assert target.index.equals(gate.index) and list(target.columns)==data['names']
        assert np.isfinite(target.to_numpy()).all()
        for row,j in enumerate(ix):
            w=target.iloc[row].to_numpy();mask=data['feasible'][j];beta=data['beta'][j]
            assert np.max(abs(w[~mask]),initial=0)<1e-10
            assert abs(w.sum())<1e-9 and abs(w[mask]@beta[mask])<1e-9
            assert abs(w).sum()<=3*min(1,mask.sum()/500)+1e-9 and abs(w).max()<=.15+1e-9
            if gate.multiplier.iloc[row]==0:assert np.max(abs(w),initial=0)==0
        assert sha(ROOT/f'accounts/{arm}_ledger.csv.gz')==m['ledger_sha256']
        assert m['audit']['arithmetic_passed'] and m['audit']['noncash_exposure_audit']['passed']
        counts[arm]=dict(provisional_sessions=m['audit']['provisional_sessions'],validated_performance=m['audit']['validated_performance'])
    replay=read(ROOT/'accounts/h21_replay_complete.json')['reference_ledger_parity']
    assert replay and max(replay.values())<1e-10
    result=dict(passed=True,annual_models_verified=68,accounts_verified=6,model_queries_identical=True,
        label_purge_passed=True,inner_validation_purge_passed=True,source_risk_basis_parity=data['risk_parity'],
        target_constraints_passed=True,unchanged_challenger_max_ledger_error=max(replay.values()),accounts=counts,
        proxy_not_exact_account_objective=True,protocol_sha256=protocol(),data_manifest_sha256=sha(ROOT/'data_manifest.json'))
    write(ROOT/'reports/validation.json',result)
    return result

def main():
    validation=validate();spec=read(ROOT/'protocol.json');ledgers={}
    for name in ARMS:
        if name=='h21_replay':name_out=CHALLENGER
        else:name_out=name
        ledgers[name_out]=load_ledger(ROOT/f'accounts/{name}_ledger.csv.gz',name_out)
    for name,directory in [('huber_ridge_10y',SOURCE),('h5_sector_risk_weekly',V23)]:
        m=read(directory/f'accounts/{name}_complete.json');f=directory/f'accounts/{name}_ledger.csv.gz'
        assert sha(f)==m['ledger_sha256'];ledgers[name]=load_ledger(f,name)
    clock=ledgers[CHALLENGER].index;assert all(f.index.equals(clock) for f in ledgers.values())
    returns=pd.DataFrame({n:f.daily_return for n,f in ledgers.items()});rows=[];annual=[]
    for name,f in ledgers.items():
        prior=f.nav.shift(1,fill_value=1)
        for period,left,right in PERIODS:
            part=f.loc[left:right];rows.append(performance(name,part,prior.reindex(part.index),period,None))
        for year in range(2007,2024):
            part=f.loc[str(year)];a=performance(name,part,prior.reindex(part.index),f'year_{year}',None);a['year']=year;annual.append(a)
    perf=pd.DataFrame(rows);pairs=spec['primary_pairs'];primary=returns.loc['2012':'2023']
    inference=pd.concat([paired_comparisons(primary,pairs,b,family_size=6) for b in (20,60)],ignore_index=True)
    out=ROOT/'reports/account_comparison';out.mkdir(exist_ok=True)
    for name,f in [('performance.csv',perf),('annual.csv',pd.DataFrame(annual)),('paired_primary.csv',inference)]:f.to_csv(out/name,index=False)
    returns.to_csv(out/'common_daily_returns.csv.gz',index_label='date')
    positive=[]
    for a,b in pairs:
        q=inference.loc[inference.arm.eq(a)&inference.baseline.eq(b)]
        if q.delta_sharpe.gt(0).all() and q.p_holm.lt(.05).all():positive.append([a,b])
    write(out/'analysis_status.json',dict(completed=True,protocol_sha256=protocol(),data_manifest_sha256=sha(ROOT/'data_manifest.json'),
        validation=validation,accounts=list(ledgers),primary_pairs=pairs,positive_holm_both_blocks=positive,
        common_sessions=len(clock),primary_sessions=len(primary),promotion=False,adaptive_research=True,
        interpretation='Economic portfolio-policy objective compared with unchanged H21 challenger, matched-sample Huber, cost/persistence ablations and original current model',
        caveats=spec['limitations']+spec['objective']['proxy_limits']))
    print(perf.loc[perf.period.isin(['primary','early','full']),['arm','period','sharpe','cagr','volatility','max_drawdown']].to_string(index=False))

if __name__=='__main__':main()
