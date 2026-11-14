"""Frozen horizon x target factorial with identical training census and weights.

Every horizon/target shares the 21-session-mature whole-date training census.
Prediction eligibility remains the original current-known universe, never labels.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,datetime,hashlib,json,pickle,time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(str(Path(__file__).resolve().parents[3] / 'prior/v17_long_history'))
sys.path.insert(0,str(SOURCE/'src'))
from model_core_long import known,mature_training
from fit_long_history import prepare

HORIZONS=(5,10,21)
TARGETS=('sector_risk','projected_scaled','projected_rank')
MODELS=[f'h{h}_{t}' for h in HORIZONS for t in TARGETS]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def freeze():
    file=ROOT/'protocol.json'
    if file.exists():
        assert sha(file)==file.with_suffix('.sha256').read_text().strip();return
    audit=read(ROOT/'reports/labels/label_audit.json')
    config=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='Declared before new horizon/target forecast or account outcomes',
        purpose='Separate forecast-horizon and learning-target interaction from sample attrition and rebalance cadence',
        horizons=list(HORIZONS),targets=list(TARGETS),models=MODELS,
        primary_reference='h5_sector_risk_weekly',external_reference='huber_ridge_10y',
        training=dict(window_years=10,first_fit=2007,last_fit=2023,refit='January1 annually',features=36,
            common_census='Every known member in a date has complete H5,H10,H21 labels; max H21 label_end strictly before annual refit',
            common_weights='Equal date influence times H21 interval uniqueness, same normalized row weights across all nine models',
            prediction_eligibility='Original current-known model_eligible; no future-label filtering',
            target_sector_risk='Original sector-beta residual divided by original decision-known idio_vol21*sqrt(H), clipped[-8,8]',
            target_projected_scaled='Forward raw return projected off a constant and decision-known SPY beta across each complete date; divide by common median idio_vol21*sqrt(H), floor .005*sqrt(H/5), clip[-8,8]',
            target_projected_rank='Centered within-date percentile rank of the same dollar/beta-projected return; exact mean0',
            estimator='Huber IRLS Ridge alpha100,10iterations,delta1.345,residual-scale floor.1; unchanged global36 features',
            preprocessing='Common train-only winsorization/scaling for all9 models; no hyperparameter search'),
        allocation=dict(weekly='Every5observed sessions for all9',matched='EveryH sessions for H10/H21, all6 additional cases, no selection',
            common='Cold daily EWMA.25, next-close execution, original risk/account engines,12%target,gross3,cap.15,buffer.2,cost2bps',
            gate='Original Huber H5 efficacy gate computed at each schedule, identical across models for that schedule; not a horizon-specific efficacy gate',
            borrow=False,financing=False,actual_holding='Daily original risk guards may trade between scheduled rebalances'),
        inference=dict(primary='2012-2023',early='2007-2011',full='2007-2023',replications=2000,blocks=[20,60],
            main_family='All14 alternatives versus h5_sector_risk_weekly, Holm14 separately by block/endpoint',
            cadence_family='Six paired matched-minus-weekly contrasts, Holm6 separately by block/endpoint',
            external_reference='Original Huber is descriptive; its training census/weights differ from the common-sample H5 control',
            interpretation='Exploratory reused development; not independent confirmation; no automated promotion'),
        boundary_exclusive='2024-01-01',
        label_sha256=sha(ROOT/'data/horizon_labels.pkl'),label_audit_sha256=sha(ROOT/'reports/labels/label_audit.json'),
        source_panel_sha256=sha(SOURCE/'data/candidate_features/panel.pkl'),
        source_approval_sha256=sha(SOURCE/'data/assembled/approval.json'),
        fit_code_sha256=sha(__file__),source_code_hashes={n:sha(SOURCE/'src'/n) for n in ['model_core_long.py','fit_long_history.py']},
        limits=['Original provisional historical valuations retained','Some common complete dates excluded due source gaps, reported separately',
                'Reserved2024+ not used here; earlier project viewed it','Longer forecasts and longer scheduled rebalances are separate dimensions'])
    file.write_text(json.dumps(config,indent=2),encoding='utf-8');file.with_suffix('.sha256').write_text(sha(file),encoding='ascii')


def common_training(panel,labels,cutoff,clock):
    window=panel.loc[panel.decision_date.ge(cutoff-pd.DateOffset(years=10))&panel.decision_date.lt(cutoff)].copy()
    window['source_index']=window.index
    label=labels.loc[window.index]
    fields=[f'{c}_h{h}' for h in HORIZONS for c in ['future_return','future_parent_return','future_spy','target_risk_resid']]
    valid=np.isfinite(label[fields].to_numpy(float)).all(axis=1)
    # Reuse the verified primitive with H21 endpoints/response for maturity and
    # overlap weights; actual training responses are constructed separately below.
    window['target_risk_resid_h5']=label.target_risk_resid_h21.where(valid).to_numpy()
    window['label_end_h5']=pd.to_datetime(label.label_end_h21).to_numpy()
    result=mature_training(window,cutoff,clock)
    assert result.label_end_h5.lt(cutoff).all()
    return result


def projected_response(frame,returns):
    """OLS projection off intercept and known beta; handles constant beta."""
    date=frame.decision_date.to_numpy();beta=frame.beta_spy63.to_numpy(float)
    r=pd.Series(np.asarray(returns,float));b=pd.Series(beta)
    rc=r-r.groupby(date).transform('mean');bc=b-b.groupby(date).transform('mean')
    numerator=(rc*bc).groupby(date).transform('sum').to_numpy()
    denominator=(bc*bc).groupby(date).transform('sum').to_numpy()
    slope=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-12)
    return rc.to_numpy()-slope*bc.to_numpy()


def responses(train,labels):
    label=labels.loc[train.source_index.to_numpy()].reset_index(drop=True)
    out={};date=train.decision_date.to_numpy()
    for h in HORIZONS:
        assert pd.to_datetime(label[f'label_end_h{h}']).lt(train.label_end_h5.max()+pd.Timedelta(days=1)).all()
        projected=projected_response(train,label[f'future_return_h{h}'].to_numpy(float))
        scale=(train.idio_vol21*np.sqrt(h)).groupby(train.decision_date).transform('median').clip(lower=.005*np.sqrt(h/5)).to_numpy(float)
        ranked=pd.Series(projected).groupby(date).rank(method='average')
        count=pd.Series(projected).groupby(date).transform('size')
        out[f'h{h}_sector_risk']=np.clip(label[f'target_risk_resid_h{h}'].to_numpy(float),-8,8)
        out[f'h{h}_projected_scaled']=np.clip(projected/scale,-8,8)
        out[f'h{h}_projected_rank']=(2*(ranked-.5)/count-1).to_numpy(float)
    assert all(np.isfinite(y).all() for y in out.values())
    return out


def fit_huber(x,y,w):
    model=Ridge(alpha=100);rw=w.copy()
    for _ in range(10):
        model.fit(x,y,sample_weight=rw);resid=y-model.predict(x)
        scale=max(.1,1.4826*np.median(abs(resid-np.median(resid))))
        rw=w*np.minimum(1,1.345*scale/np.maximum(abs(resid),1e-12));rw*=w.sum()/rw.sum()
    return model,scale


def main(years):
    freeze();protocol=read(ROOT/'protocol.json');digest=sha(ROOT/'protocol.json')
    assert sha(__file__)==protocol['fit_code_sha256']
    assert sha(ROOT/'data/horizon_labels.pkl')==protocol['label_sha256']
    assert sha(SOURCE/'data/candidate_features/panel.pkl')==protocol['source_panel_sha256']
    assert protocol['source_code_hashes']=={n:sha(SOURCE/'src'/n) for n in protocol['source_code_hashes']}
    panel=pd.read_pickle(SOURCE/'data/candidate_features/panel.pkl');labels=pd.read_pickle(ROOT/'data/horizon_labels.pkl')
    assert labels.index.equals(panel.index) and labels[['decision_date','ticker']].equals(panel[['decision_date','ticker']])
    assert panel.decision_date.max()<pd.Timestamp('2024-01-01')
    clock=pd.DatetimeIndex(pd.read_csv(SOURCE/'data/assembled/calendar.csv',parse_dates=['Date']).Date)
    columns=read(SOURCE/'data/candidate_features/columns.json')
    for d in ['forecast_runs','forecast_models']:(ROOT/d).mkdir(exist_ok=True)
    for year in years:
        assert 2007<=year<=2023;start=time.time();cutoff=pd.Timestamp(year,1,1)
        if all((ROOT/f'forecast_models/{name}_{year}.json').exists() for name in MODELS):
            for name in MODELS:
                m=read(ROOT/f'forecast_models/{name}_{year}.json');assert m['protocol_sha256']==digest
                assert sha(ROOT/f'forecast_runs/{name}_{year}.csv.gz')==m['forecast_sha256']
                assert sha(ROOT/f'forecast_models/{name}_{year}.pkl')==m['model_sha256']
            print('HORIZON_CACHED',year,flush=True);continue
        train=common_training(panel,labels,cutoff,clock)
        query=panel.loc[known(panel)&panel.decision_date.dt.year.eq(year)].sort_values(['decision_date','ticker']).reset_index(drop=True)
        (x,q),pre=prepare(train,[query],columns,cutoff)
        targets=responses(train,labels);w=train.sample_weight.to_numpy(float)
        census_hash=hashlib.sha256(train.source_index.to_numpy().tobytes()+w.tobytes()).hexdigest()
        date_counts=train.groupby('decision_date').size()
        for name,y in targets.items():
            t=time.time();model,scale=fit_huber(x,y,w);score=np.clip(model.predict(q),-8,8)
            assert np.isfinite(score).all()
            frame=query[['decision_date','ticker']].copy();frame['score']=score
            frame['refit_boundary']=cutoff;frame['max_training_label_end']=train.label_end_h5.max()
            file=ROOT/f'forecast_runs/{name}_{year}.csv.gz';tmp=file.with_suffix('.partial')
            frame.to_csv(tmp,index=False,compression='gzip');tmp.replace(file)
            model_file=ROOT/f'forecast_models/{name}_{year}.pkl'
            with model_file.open('wb') as stream:pickle.dump(dict(model=model,preprocessing=pre,columns=columns),stream,protocol=5)
            record=dict(model=name,year=year,protocol_sha256=digest,forecast_sha256=sha(file),model_sha256=sha(model_file),
                source_panel_sha256=protocol['source_panel_sha256'],label_sha256=protocol['label_sha256'],
                training_rows=len(train),training_dates=len(date_counts),training_start=str(train.decision_date.min()),
                max_training_label_end=str(train.label_end_h5.max()),refit_boundary=str(cutoff),
                common_training_census_weight_sha256=census_hash,overlap_weight_horizon=21,
                sample_weight_sum=float(w.sum()),effective_row_weights=float(w.sum()**2/(w@w)),
                query_rows=len(query),prediction_eligibility_uses_labels=False,
                residual_scale=float(scale),target_mean=float(np.average(y,weights=w)),
                training_weighted_mse=float(np.average((y-model.predict(x))**2,weights=w)),
                standardized_coefficients=dict(zip(columns,model.coef_.astype(float))),seconds=time.time()-t)
            (ROOT/f'forecast_models/{name}_{year}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print('HORIZON_FIT',year,'rows',len(train),'dates',len(date_counts),'models9','seconds',round(time.time()-start,1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--years',type=int,nargs='+',default=list(range(2007,2024)));p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    if args.freeze_only:freeze()
    else:main(args.years)
