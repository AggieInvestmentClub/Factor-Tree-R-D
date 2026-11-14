"""Train-only regularized cross-sectional model; no files/data read at import."""
from __future__ import annotations
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

TRAIN_START=pd.Timestamp('2002-01-01')
FINAL_CUTOFF=pd.Timestamp('2024-01-01')

def known(frame):
 return frame.model_eligible.eq(True) if 'model_eligible' in frame else frame.eligible.eq(True)&frame.sector_available.eq(1)

def mature_training(panel,cutoff,calendar,*,minimum_dates=252):
 """Expanding2002-start window; whole known-member queries, strict label purge."""
 cutoff=pd.Timestamp(cutoff)
 if cutoff>FINAL_CUTOFF:raise ValueError('Model weights cannot train on holdout')
 dates=pd.to_datetime(panel.decision_date)
 p=panel.loc[dates.ge(TRAIN_START)&dates.lt(cutoff)].copy();p.attrs={}
 p['decision_date']=dates.loc[p.index];p=p.loc[known(p)].copy()
 if p.duplicated(['decision_date','ticker']).any():raise ValueError('Duplicate current member/date')
 p['label_end_h5']=pd.to_datetime(p.label_end_h5)
 mature=p.label_end_h5.notna()&p.label_end_h5.lt(cutoff)
 y=pd.Series(np.nan,index=p.index,dtype=float)
 y.loc[mature]=pd.to_numeric(p.loc[mature,'target_risk_resid_h5'],errors='raise')
 valid=np.isfinite(y)&mature
 # Preserve the old compact control's complete-query comparison census.
 for name in ('v8_target_absolute_h5','v8_target_spy_h5'):
  if name in p:valid &= p[name].notna()
 okay=valid.groupby(p.decision_date).all();counts=p.groupby('decision_date').size()
 accepted=okay.index[okay&counts.ge(10)]
 p=p.loc[p.decision_date.isin(accepted)].sort_values(['decision_date','ticker']).reset_index(drop=True)
 if len(accepted)<minimum_dates:raise ValueError('Too few complete mature date queries')
 intervals=p.groupby('decision_date').agg(start=('execution_date','first'),end=('label_end_h5','max'))
 clock=pd.DatetimeIndex(calendar);a=clock.get_indexer(pd.to_datetime(intervals.start));b=clock.get_indexer(intervals.end)
 if (a<0).any() or (b<a).any():raise ValueError('Unknown interval on observed calendar')
 active=np.zeros(len(clock)+1);np.add.at(active,a,1);np.add.at(active,b+1,-1);active=np.cumsum(active[:-1])
 uniqueness=pd.Series([float(np.mean(1/active[left:right+1])) for left,right in zip(a,b)],index=intervals.index)
 p['sample_weight']=p.decision_date.map(uniqueness)/p.groupby('decision_date').ticker.transform('size')
 p['sample_weight']*=len(p)/p.sample_weight.sum()
 return p

class FrozenRidge:
 """Compact Ridge100 semantics with explicitly declared optional availability pairs."""
 def __init__(self,columns,availability_pairs=None):
  self.columns=list(columns);self.availability_pairs=availability_pairs or {}
  if not self.columns or len(set(self.columns))!=len(self.columns):raise ValueError('Unique feature schema required')
  if any(c.startswith(('target_','future_','label_end','execution_')) or c=='decision_date' for c in self.columns):raise ValueError('Outcome metadata cannot be X')

 def raw(self,frame,end_before):
  dates=pd.to_datetime(frame.decision_date)
  if not dates.lt(pd.Timestamp(end_before)).all():raise ValueError('Prediction outside explicitly permitted prefix')
  return frame[self.columns].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan).fillna(0.).astype(float)

 def design(self,frame,end_before):
  raw=self.raw(frame,end_before);raw[self.continuous_]=raw[self.continuous_].clip(self.low_,self.high_,axis=1)
  for value,flags in self.availability_pairs.items():
   if value not in self.columns:continue
   flags=[flags] if isinstance(flags,str) else list(flags)
   mask=np.ones(len(frame),dtype=bool)
   for flag in flags:
    s=pd.to_numeric(frame[flag],errors='raise')
    if not s.isin([0,1,False,True]).all():raise ValueError('Availability must be literal0/1')
    mask &= s.eq(1).to_numpy()
   raw.loc[~mask,value]=0.
  if not np.isfinite(raw.to_numpy()).all():raise ValueError('Nonfinite model inputs')
  return raw

 def fit(self,train,cutoff):
  cutoff=pd.Timestamp(cutoff)
  if cutoff>FINAL_CUTOFF:raise ValueError('No test outcomes may refit Ridge coefficients')
  dates=pd.to_datetime(train.decision_date);ends=pd.to_datetime(train.label_end_h5)
  if not dates.lt(cutoff).all() or not dates.ge(TRAIN_START).all() or not ends.lt(cutoff).all():raise ValueError('Immature/out-of-window training data')
  y=pd.to_numeric(train.target_risk_resid_h5,errors='raise').to_numpy(dtype=float)
  weights=pd.to_numeric(train.sample_weight,errors='raise').to_numpy(dtype=float)
  if not np.isfinite(y).all() or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:raise ValueError('Invalid target/weights')
  self.cutoff_=cutoff;raw=self.raw(train,cutoff)
  literal=[c for c in self.columns if c.startswith(('sector_','is_')) or c.endswith('_available')]
  self.continuous_=[c for c in self.columns if c not in literal]
  self.low_=raw[self.continuous_].quantile(.005);self.high_=raw[self.continuous_].quantile(.995)
  matrix=self.design(train,cutoff).to_numpy(dtype=float)
  self.scaler_=StandardScaler().fit(matrix,sample_weight=weights)
  clipped=np.clip(y,-8,8)
  self.estimator_=Ridge(alpha=100).fit(self.scaler_.transform(matrix),clipped,sample_weight=weights)
  self.fit_record_=dict(cutoff=str(cutoff.date()),training_start=str(dates.min().date()),training_last_decision=str(dates.max().date()),
   max_training_label_end=str(ends.max().date()),rows=len(train),dates=dates.nunique(),features=len(self.columns),alpha=100.,
   coefficients_fixed_after_fit=True,feature_names=self.columns,availability_pairs=self.availability_pairs,
   scaler_mean=self.scaler_.mean_.tolist(),scaler_scale=self.scaler_.scale_.tolist(),
   clipping_low=self.low_.to_dict(),clipping_high=self.high_.to_dict(),
   coefficients=self.estimator_.coef_.tolist(),intercept=float(self.estimator_.intercept_),
   training_sha256=hashlib.sha256(matrix.tobytes()+clipped.tobytes()+weights.tobytes()).hexdigest())
  return self

 def predict(self,frame,*,end_before='2024-01-01'):
  if pd.to_datetime(frame.decision_date).lt(self.cutoff_).any():raise ValueError('Do not backdate final-model forecasts')
  x=self.design(frame,end_before).to_numpy(dtype=float)
  result=np.clip(self.estimator_.predict(self.scaler_.transform(x)),-8,8)
  if not np.isfinite(result).all():raise ValueError('Nonfinite predictions')
  return result
