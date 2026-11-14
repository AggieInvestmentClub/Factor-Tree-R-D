"""Same matured Pearson-IC gate, advancing a maturity queue once per query.

Numeric outcomes are parsed only after their complete date query matures.
This avoids repeatedly scanning every old label date in a 17-year account.
"""
import bisect
import numpy as np,pandas as pd
from signal_engine import _boundary,_clock,_queries

def compute_gate(predictions,panel,calendar,query_dates,*,end_before='2024-01-01'):
    boundary=_boundary(end_before);queries=_clock(query_dates)
    if not len(queries) or (queries>=boundary).any():raise ValueError('Nonempty bounded gate queries required')
    f,p,clock,origin=_queries(predictions,panel,calendar,boundary,queries[-1],provenance=True)
    if not queries.isin(f.decision_date).all() or not queries.isin(clock).all():raise ValueError('Gate query missing complete OOS forecasts or observed clock')
    spy=next((c for c in ('future_spy_h5','v8_future_spy_h5') if c in p),None)
    if spy is None:raise ValueError('Independent H5 SPY label required')
    cols=['decision_date','ticker','label_end_h5','future_return_h5','beta_spy63','idio_vol21',spy]
    combined=p[cols].merge(f[['decision_date','ticker','score']],on=['decision_date','ticker'],how='left',validate='one_to_one')
    combined['label_end_h5']=pd.to_datetime(combined.label_end_h5,errors='raise')
    waiting=[]
    for d,g in combined.groupby('decision_date',sort=True):
        end=g.label_end_h5
        if len(g)<10 or end.isna().any() or not end.gt(d).all():continue
        waiting.append((end.max(),pd.Timestamp(d),g))
    waiting.sort(key=lambda x:(x[0],x[1]));cursor=0;accepted=[];cache={}
    positions=pd.Series(np.arange(len(clock)),index=clock);records=[]
    for date in queries:
        while cursor<len(waiting) and waiting[cursor][0]<date:
            label_end,history_date,g=waiting[cursor];cursor+=1
            assert history_date<date
            y=pd.to_numeric(g.future_return_h5,errors='raise').to_numpy(dtype=float)
            market=pd.to_numeric(g[spy],errors='raise').to_numpy(dtype=float)
            if not np.isfinite(y).all() or not np.isfinite(market).all():continue
            score=pd.to_numeric(g.score,errors='raise').to_numpy(dtype=float)
            beta=pd.to_numeric(g.beta_spy63,errors='raise').to_numpy(dtype=float)
            sigma=pd.to_numeric(g.idio_vol21,errors='raise').to_numpy(dtype=float)*np.sqrt(5.)
            if not np.isfinite(score).all() or not np.isfinite(beta).all() or not np.isfinite(sigma).all() or (sigma<=0).any():raise ValueError('Invalid known historical score/risk input')
            residual=y-beta*market;residual-=residual.mean();response=residual/sigma
            ic=float(np.corrcoef(score,response)[0,1]) if np.std(score)>=1e-12 and np.std(response)>=1e-12 else np.nan
            cache[history_date]=dict(ic=ic,label_end=label_end);bisect.insort(accepted,history_date)
        history=accepted[-252:];values=np.asarray([cache[d]['ic'] for d in history],dtype=float)
        age=positions.loc[date]-positions.reindex(pd.DatetimeIndex(history)).to_numpy()
        weights=np.exp2(-age/63.);valid=np.isfinite(values)&np.isfinite(weights)&(weights>0)
        w=weights[valid];v=values[valid];mass=float(w.sum())
        mean=float(w@v/mass) if mass else 0.;effective=float(mass*mass/(w@w)) if mass else 0.
        warm=len(history)<63 or int(valid.sum())<63;multiplier=1. if warm else float(mean>0)
        records.append(dict(decision_date=date,multiplier=multiplier,gate=multiplier,warmup=warm,warmup_baseline_fallback=warm,
            mature_dates=len(history),valid_ic_dates=int(valid.sum()),effective_dates_kish=effective,ew_ic=mean,pooled_ic_mean=mean,
            history_first_date=min(history,default=pd.NaT),history_last_date=max(history,default=pd.NaT),
            history_label_end_max=max((cache[d]['label_end'] for d in history),default=pd.NaT),
            half_life_sessions=63,lookback_mature_dates=252,minimum_mature_dates=63,minimum_valid_ic_dates=63,
            negative_forecast_sign_flipped=False,provenance_column=origin,seed_is_final_in_sample_backfill=False,
            gate_kind='positive_prior_OOS_Pearson_IC_mean;not_profit_or_confidence',data_cutoff_exclusive=str(boundary.date())))
    return pd.DataFrame(records).set_index('decision_date')
