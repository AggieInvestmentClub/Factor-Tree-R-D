"""Disk-backed exact diag50 covariance, storing centered returns, not N^2.

This is a storage change only. Invalid histories remain invalid placeholders;
the covariance estimator, session window and portfolio constraints are fixed.
"""
from pathlib import Path
from collections.abc import Mapping
import json,hashlib
import numpy as np,pandas as pd
from risk_engine import _clock,_slots,_cutoff

class LazyCovariances(Mapping):
    def __init__(self,directory):
        self.directory=Path(directory);self.manifest=json.loads((self.directory/'manifest.json').read_text())
        self.names=tuple(self.manifest['names']);self.dates=pd.DatetimeIndex(self.manifest['dates'])
        self._last=None;self._record=None
    def __len__(self):return len(self.dates)
    def __iter__(self):return iter(self.dates)
    def __contains__(self,key):return pd.Timestamp(key) in self.dates
    def __getitem__(self,key):
        date=pd.Timestamp(key)
        if date not in self:raise KeyError(key)
        if date==self._last:return self._record
        with np.load(self.directory/f'{date:%Y-%m-%d}.npz',allow_pickle=False) as stored:
            idx=stored['indices'];centered=stored['centered'];metadata=json.loads(str(stored['metadata']))
        sample=centered.T@centered/62
        sub=.5*sample+.5*np.diag(np.diag(sample));sub=(sub+sub.T)/2
        matrix=np.zeros((len(self.names),len(self.names)));matrix[np.ix_(idx,idx)]=sub
        valid=pd.Series(False,index=self.names);valid.iloc[idx]=True
        self._record=dict(names=self.names,pooled=matrix,valid=valid,
            stock_slot_mask=pd.Series(self.manifest['stock_slot_mask'],index=self.names),metadata=metadata)
        self._last=date;return self._record

def build(prices,names,dates,directory,source_hash,*,stock_slot_mask=None,data_cutoff='2024-01-01'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    boundary=_cutoff(data_cutoff);names=list(names);slots=_slots(names,stock_slot_mask);queries=_clock(dates,'query')
    if not len(queries) or (queries>=boundary).any():raise ValueError('Nonempty pre-boundary query clock required')
    last=queries[-1];calendar=pd.DatetimeIndex(prices['SPY'].index);calendar=_clock(calendar[(calendar<boundary)&(calendar<=last)],'SPY')
    assert queries.isin(calendar).all()
    schema=dict(names=names,dates=[str(d.date()) for d in queries],stock_slot_mask=slots.tolist(),source_hash=source_hash,
        estimator='diag50',window=63,boundary=str(boundary.date()),storage='float64 centered qualified daily returns; expand exactly on demand')
    manifest=directory/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==schema,'Risk cache source/schema changed'
        if all((directory/f'{date:%Y-%m-%d}.npz').exists() for date in queries):return LazyCovariances(directory)
    returns=np.full((len(calendar),len(names)),np.nan)
    for j,name in enumerate(names):
        source=prices[name];ix=pd.DatetimeIndex(source.index);keep=(ix<boundary)&(ix<=last)
        frame=source.loc[keep].copy(deep=False);frame.index=_clock(ix[keep],'source');frame=frame.reindex(calendar)
        if 'q' in frame:q=frame.q.eq(1)
        elif {'source_available','tr_chain_available'}.issubset(frame):q=frame.source_available.eq(1)&frame.tr_chain_available.eq(1)
        elif 'tr_qualified' in frame:q=frame.tr_qualified.eq(1)
        elif 'price_qualified' in frame:q=frame.price_qualified.eq(1)
        else:raise ValueError('Explicit qualification required')
        close=pd.to_numeric(frame.adj_close,errors='raise');segment=frame.tr_segment_id if 'tr_segment_id' in frame else pd.Series(1.,index=calendar)
        good=q&np.isfinite(close)&close.gt(0)&segment.notna()
        returns[:,j]=(close/close.shift(1)-1).where(good&good.shift(1,fill_value=False)&segment.eq(segment.shift(1))).to_numpy()
    for date in queries:
        file=directory/f'{date:%Y-%m-%d}.npz'
        if file.exists():continue
        end=calendar.get_loc(date)+1;start=max(0,end-63);x=returns[start:end]
        valid=np.isfinite(x).all(axis=0) if len(x)==63 else np.zeros(len(names),bool);indices=np.flatnonzero(valid)
        work=x[:,indices];centered=work-work.mean(axis=0)
        metadata=dict(decision_date=str(date.date()),window_start=str(calendar[start].date()),window_end=str(date.date()),
            max_source_date=str(date.date()),window_sessions=len(x),required_sessions=63,estimator='diag50',covariance_horizon=1,
            covariance_units='daily_return_squared',shrinkage=.5 if valid.any() else None,normalization='sample_ddof1',
            valid_assets=int(valid.sum()),valid_stock_assets=int((valid&slots.to_numpy()).sum()),hedge_names=list(slots.index[~slots]),
            stock_budget_denominator=500,nonlinear_2017_replication=False,data_cutoff_exclusive=str(boundary.date()),numeric_prefix_end=str(last.date()))
        tmp=file.with_suffix('.partial')
        with tmp.open('wb') as stream:np.savez_compressed(stream,indices=indices,centered=centered,metadata=np.array(json.dumps(metadata)))
        tmp.replace(file)
    manifest.write_text(json.dumps(schema,indent=2));return LazyCovariances(directory)
