"""Fixed rank/EWMA/weekly risk sizing and strictly matured OOS Pearson-IC gate.

No files, prices or outcomes are read at import. Labels are used only inside
the gate, after every member of a historical date query has strictly matured.
The final learner is never fitted or refitted here. Seed forecasts must carry
a real per-row refit boundary at or before their decision, not backfilled final
in-sample predictions. Upstream fit seals establish the training-label cutoff.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
STOCK_SLOTS=110
import numpy as np
import pandas as pd
from scipy.stats import rankdata

_spec=importlib.util.spec_from_file_location('_v10_signal_explicit_risk',Path(__file__).with_name('risk_engine.py'))
_risk=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_risk)


def _boundary(value):
    value=pd.Timestamp(value)
    if pd.isna(value) or value.tz is not None or value!=value.normalize():raise ValueError('Explicit normalized naive end_before required')
    return value


def _clock(value):
    dates=pd.DatetimeIndex(value)
    if dates.tz is not None or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:raise ValueError('Unique increasing naive observed calendar required')
    return dates


def _known(frame):
    return frame.model_eligible.eq(True) if 'model_eligible' in frame else frame.eligible.eq(True)&frame.sector_available.eq(1)


def _queries(predictions,panel,calendar,end_before,through=None,*,provenance=False):
    boundary=_boundary(end_before);p=panel.copy(deep=False);p.attrs={};f=predictions.copy(deep=False);f.attrs={}
    pd_dates=pd.to_datetime(p.decision_date);fd=pd.to_datetime(f.decision_date)
    last=pd.Timestamp(through) if through is not None else fd.loc[fd<boundary].max()
    p=p.loc[pd_dates.lt(boundary)&pd_dates.le(last)].copy();p['decision_date']=pd_dates.loc[p.index]
    f=f.loc[fd.lt(boundary)&fd.le(last)].copy();f['decision_date']=fd.loc[f.index]
    if f.empty:raise ValueError('No bounded forecasts')
    if f.duplicated(['decision_date','ticker']).any() or p.duplicated(['decision_date','ticker']).any():raise ValueError('Duplicate forecast/panel member date')
    clock=_clock(calendar);clock=clock[(clock<boundary)&(clock<=last)]
    if not f.decision_date.isin(clock).all():raise ValueError('Forecast outside observed clock')
    p=p.loc[_known(p)&p.decision_date.ge(f.decision_date.min())].copy()
    keys=['decision_date','ticker'];expected=p[keys].sort_values(keys).reset_index(drop=True);actual=f[keys].sort_values(keys).reset_index(drop=True)
    if not expected.equals(actual):raise ValueError('Forecasts must preserve EVERY complete current-known date query, including unknown final labels')
    origin=next((c for c in ('refit_boundary','model_available_at','fit_cutoff','training_cutoff') if c in f),None)
    if provenance:
        if origin is None:raise ValueError('Per-row OOS refit_boundary/model_available_at required; final-model in-sample backfill prohibited')
        available=pd.to_datetime(f[origin],errors='raise')
        if available.isna().any() or not available.le(f.decision_date).all():raise ValueError('Forecast model unavailable at decision; in-sample backfill prohibited')
        if 'max_training_label_end' in f:
            end=pd.to_datetime(f.max_training_label_end,errors='raise')
            if end.isna().any() or not end.lt(available).all():raise ValueError('Seed/final training label maturity invalid')
    return f,p,clock,origin


def compute_gate(predictions,panel,calendar,query_dates,*,end_before='2024-01-01'):
    """Literal v9 efficacy_ic_gate: positive past EW Pearson-IC mean or cash.

    History: last252 whole complete mature label dates, then finite ICs receive
    session-age half-life63 weights. Warmup remains1 until BOTH63 history dates
    and63 finite Pearson-IC dates exist, exactly as sealed v9 mode_warm. Rank
    IC, basket profit and in-sample fitted scores are not substituted here.
    """
    boundary=_boundary(end_before);queries=_clock(query_dates)
    if not len(queries) or (queries>=boundary).any():raise ValueError('Nonempty bounded gate queries required')
    f,p,clock,origin=_queries(predictions,panel,calendar,boundary,queries[-1],provenance=True)
    if not queries.isin(f.decision_date).all():raise ValueError('Gate query without complete current OOS forecasts')
    if not queries.isin(clock).all():raise ValueError('Gate query outside observed clock')
    spy=next((c for c in ('future_spy_h5','v8_future_spy_h5') if c in p),None)
    if spy is None:raise ValueError('Independent H5 SPY label required')
    columns=['decision_date','ticker','label_end_h5','future_return_h5','beta_spy63','idio_vol21',spy]
    combined=p[columns].merge(f[['decision_date','ticker','score']],on=['decision_date','ticker'],how='left',validate='one_to_one')
    combined['label_end_h5']=pd.to_datetime(combined.label_end_h5,errors='raise')
    groups={pd.Timestamp(d):g for d,g in combined.groupby('decision_date',sort=True)}
    positions=pd.Series(np.arange(len(clock)),index=clock);cache={};records=[]
    for date in queries:
        history=[]
        for history_date,g in groups.items():
            if history_date>=date:break
            end=g.label_end_h5
            if len(g)<10 or end.isna().any() or not end.lt(date).all() or not end.gt(history_date).all():continue
            if history_date not in cache:
                # Future outcome numeric parsing happens ONLY after the strict
                # whole-date maturity test above; current unknown labels stay.
                y=pd.to_numeric(g.future_return_h5,errors='raise').to_numpy(dtype=float)
                market=pd.to_numeric(g[spy],errors='raise').to_numpy(dtype=float)
                if not np.isfinite(y).all() or not np.isfinite(market).all():cache[history_date]=None
                else:
                    score=pd.to_numeric(g.score,errors='raise').to_numpy(dtype=float)
                    beta=pd.to_numeric(g.beta_spy63,errors='raise').to_numpy(dtype=float)
                    sigma=pd.to_numeric(g.idio_vol21,errors='raise').to_numpy(dtype=float)*np.sqrt(5.)
                    if not np.isfinite(score).all() or not np.isfinite(beta).all() or not np.isfinite(sigma).all() or (sigma<=0).any():raise ValueError('Invalid known historical score/risk input')
                    residual=y-beta*market;residual-=residual.mean();response=residual/sigma
                    ic=float(np.corrcoef(score,response)[0,1]) if np.std(score)>=1e-12 and np.std(response)>=1e-12 else np.nan
                    cache[history_date]=dict(ic=ic,label_end=end.max())
            if cache[history_date] is not None:history.append(history_date)
        history=history[-252:];values=np.asarray([cache[d]['ic'] for d in history],dtype=float)
        age=positions.loc[date]-positions.reindex(pd.DatetimeIndex(history)).to_numpy()
        weights=np.exp2(-age/63.);valid=np.isfinite(values)&np.isfinite(weights)&(weights>0)
        w=weights[valid];v=values[valid];mass=float(w.sum())
        mean=float(w@v/mass) if mass else 0.;effective=float(mass*mass/(w@w)) if mass else 0.
        warm=len(history)<63 or int(valid.sum())<63
        multiplier=1. if warm else float(mean>0)
        records.append(dict(decision_date=date,multiplier=multiplier,gate=multiplier,warmup=warm,warmup_baseline_fallback=warm,
            mature_dates=len(history),valid_ic_dates=int(valid.sum()),effective_dates_kish=effective,ew_ic=mean,pooled_ic_mean=mean,
            history_first_date=min(history,default=pd.NaT),history_last_date=max(history,default=pd.NaT),
            history_label_end_max=max((cache[d]['label_end'] for d in history),default=pd.NaT),
            half_life_sessions=63,lookback_mature_dates=252,minimum_mature_dates=63,minimum_valid_ic_dates=63,
            negative_forecast_sign_flipped=False,provenance_column=origin,seed_is_final_in_sample_backfill=False,
            gate_kind='positive_prior_OOS_Pearson_IC_mean;not_profit_or_confidence',data_cutoff_exclusive=str(boundary.date())))
    return pd.DataFrame(records).set_index('decision_date')


def _rank_signal(scores,mask):
    result=np.zeros(len(scores));idx=np.flatnonzero(mask)
    if len(idx)<3:return result
    centered=rankdata(np.asarray(scores)[idx],method='average')-(len(idx)+1)/2
    positive=np.maximum(centered,0);negative=np.minimum(centered,0)
    if positive.sum()>1e-12 and -negative.sum()>1e-12:
        result[idx]=.5*positive/positive.sum()+.5*negative/-negative.sum()
    return result


def _base_projection(raw,beta,mask,gross):
    """Exact v6/v8 projection normalization for the initial unlevered book."""
    result=np.zeros(len(raw))
    if gross<=1e-12 or mask.sum()<3:return result
    rows=np.vstack([np.ones(mask.sum()),beta[mask]])
    _,singular,right=np.linalg.svd(rows,full_matrices=False)
    rank=int(np.count_nonzero(singular>max(rows.shape)*np.finfo(float).eps*singular[0]))
    projected=raw[mask]-right[:rank].T@(right[:rank]@raw[mask]);size=float(np.abs(projected).sum())
    if size<1e-12:return result
    projected*=gross/size;projected*=min(1.,.10/float(np.abs(projected).max()))
    if np.abs(rows@projected).max()>1e-10:return result
    result[mask]=projected;return result


def build_weekly_targets(predictions,panel,calendar,names,covariances,*,anchor,end_before='2024-01-01'):
    """Cold daily score EWMA.25 → ranks → base projection → weekly12% sizing.

    The fixed cadence is five OBSERVED sessions from the explicit anchor.
    EWMA resets only once, on the first query of this single supplied path;
    it does not reset at year boundaries. It is not seeded with fitted scores.
    The gate is separate: multiply scheduled targets by compute_gate.multiplier.
    """
    boundary=_boundary(end_before);anchor=pd.Timestamp(anchor);names=list(names)
    if not names or len(names)>STOCK_SLOTS or len(set(names))!=len(names):raise ValueError('One-to110 unique declared stock names required')
    f,p,clock,_=_queries(predictions,panel,calendar,boundary)
    if not set(f.ticker).issubset(names):raise ValueError('Forecast member outside declared cohort')
    dates=pd.DatetimeIndex(sorted(f.decision_date.unique()))
    if anchor not in clock or dates[0]<anchor:raise ValueError('Explicit observed anchor must not follow the cold-start first query')
    scores=f.pivot(index='decision_date',columns='ticker',values='score').reindex(index=dates,columns=names)
    risk=p.set_index(['decision_date','ticker']);anchor_position=int(clock.get_indexer([anchor])[0])
    stages={key:[] for key in ('raw_score','ewma_score','rank','signal','projected')}
    diagnostics=[];desired=[];old=np.full(len(names),np.nan)
    for date in dates:
        index=pd.MultiIndex.from_product([[date],names],names=['decision_date','ticker']);meta=risk.reindex(index)
        raw=scores.loc[date].to_numpy(dtype=float);beta=meta.beta_spy63.to_numpy(dtype=float)
        available=_known(meta).to_numpy();mask=available&np.isfinite(raw)&np.isfinite(beta)
        smoothed=raw.copy() if date==dates[0] else .75*np.where(np.isfinite(old),old,raw)+.25*raw
        smoothed[~mask]=np.nan;old=smoothed.copy()
        rank=np.zeros(len(names));rank[mask]=rankdata(smoothed[mask],method='average')-(mask.sum()+1)/2
        signal=_rank_signal(smoothed,mask);gross=min(float(mask.sum()/STOCK_SLOTS),float(np.abs(signal).sum()))
        projected=_base_projection(signal,beta,mask,gross)
        number=int(clock.get_indexer([date])[0]);scheduled=(number-anchor_position)%5==0
        detail=dict(decision_date=date,target_scheduled=scheduled,eligible_count=int(mask.sum()),
            base_gross=float(abs(projected).sum()),base_net=float(projected.sum()),base_estimated_beta=float(projected[mask]@beta[mask]),
            smoothing_alpha=.25,ewma_initialization='cold_first_query_once;no_annual_reset;no_IS_seed',cadence_sessions=5,
            cadence_anchor=str(anchor.date()),stock_budget_denominator=STOCK_SLOTS,data_cutoff_exclusive=str(boundary.date()))
        if scheduled:
            if date not in covariances:raise ValueError('Missing exact scheduled risk record')
            target,record=_risk.allocate_target(pd.Series(projected,index=names),pd.Series(beta,index=names),covariances[date],pd.Series(mask,index=names),
                date=date,vol_target=.12,gross_cap=3.,name_cap=.15,data_cutoff=boundary)
            desired.append(target.rename(date));detail.update(record)
        for key,value in dict(raw_score=raw,ewma_score=smoothed,rank=rank,signal=signal,projected=projected).items():stages[key].append(pd.Series(value,index=names,name=date))
        diagnostics.append(detail)
    frames={key:pd.DataFrame(rows,columns=names).rename_axis('decision_date') for key,rows in stages.items()}
    targets=pd.DataFrame(desired,columns=names).rename_axis('decision_date')
    audit=pd.DataFrame(diagnostics);audit['decision_date']=pd.to_datetime(audit.decision_date)
    return targets,audit,frames
