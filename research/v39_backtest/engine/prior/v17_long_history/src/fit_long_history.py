"""Frozen, annual, purged forecasting comparisons on the historical candidate panel.

This creates forecasts, never silently approves account performance. Post2023
rows are forbidden, and every model sees the same current-known date queries.
"""
import os,sys
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,datetime,json,pickle,hashlib,time
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.preprocessing import StandardScaler,SplineTransformer
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import RandomForestRegressor
from model_core_long import known,mature_training,FrozenRidge
ROOT=Path(__file__).resolve().parents[1]

PARAMETERS=dict(
 rf=dict(n_estimators=96,max_depth=5,min_samples_leaf=500,max_features=.5,max_samples=.25,n_jobs=3,random_state=2026),
 lgbm=dict(n_estimators=120,num_leaves=7,learning_rate=.03,reg_lambda=100,min_child_samples=500,n_jobs=3,verbosity=-1,random_state=2026),
 xgb=dict(n_estimators=120,max_depth=3,learning_rate=.03,reg_lambda=100,min_child_weight=500,subsample=.7,colsample_bytree=.7,tree_method='hist',n_jobs=3,random_state=2026),
 mlp=dict(width=32,activation='tanh',learning_rate=.001,weight_decay=.01,max_epochs=40,patience=6,batch_size=8192,seed=2026),
 elastic=dict(alpha=.001,l1_ratio=.1,max_iter=5000,tol=1e-5,selection='cyclic'),
 huber_ridge=dict(iterations=10,huber_delta=1.345,minimum_residual_scale=.1),
 spline_ridge=dict(n_knots=4,degree=2,knots='quantile',include_bias=False),
 rff_ridge=dict(n_components=128,gamma=1/36,random_state=2026))

def prepare(tr,frames,cols,cut):
    obj=FrozenRidge(cols);raw=obj.raw(tr,cut)
    obj.continuous_=[c for c in cols if not(c.startswith(('sector_','is_')) or c.endswith('_available'))]
    obj.low_=raw[obj.continuous_].quantile(.005);obj.high_=raw[obj.continuous_].quantile(.995)
    x=obj.design(tr,cut).to_numpy(dtype=float)
    scaler=StandardScaler().fit(x,sample_weight=tr.sample_weight.to_numpy())
    return [scaler.transform(x)]+[scaler.transform(obj.design(f,'2024-01-01').to_numpy(dtype=float)) for f in frames],dict(preprocessor=obj,scaler=scaler)

def neural(x,y,w,query,epochs,validation=None):
    import torch
    torch.set_num_threads(2);torch.manual_seed(2026)
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.Tanh(),torch.nn.Linear(32,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    X=torch.from_numpy(np.asarray(x,np.float32));Y=torch.from_numpy(np.asarray(y,np.float32));W=torch.from_numpy(np.asarray(w,np.float32))
    best=np.inf;chosen=epochs;wait=0
    for ep in range(epochs):
        model.train()
        # Fixed chronological row order. Validation uses a later date block;
        # no shuffled split, full-panel normalization or validation refitting.
        for left in range(0,len(x),8192):
            sl=slice(left,left+8192);loss=((model(X[sl]).flatten()-Y[sl])**2*W[sl]).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if validation is not None:
            model.eval();vx,vy,vw=validation
            with torch.no_grad():
                pred=model(torch.from_numpy(np.asarray(vx,np.float32))).flatten().numpy()
            value=float(np.average((pred-vy)**2,weights=vw))
            if value<best-1e-6:best=value;chosen=ep+1;wait=0
            else:wait+=1
            if wait>=6:break
    model.eval()
    with torch.no_grad():pred=model(torch.from_numpy(np.asarray(query,np.float32))).flatten().numpy()
    return pred,chosen,{k:v.detach().numpy() for k,v in model.state_dict().items()}

def shuffle_within_sector(train,y,seed):
    """Conditional null: preserve each date/sector outcome distribution.

    This removes within-sector stock assignment, not sector-level predictability.
    Group-specific seeds make the same historical query invariant to adding
    earlier or later dates to an annual training window.
    """
    result=np.asarray(y).copy()
    for (date,sector),indices in train.groupby(['decision_date','sector'],dropna=False,sort=True).indices.items():
        key=f'{seed}|{pd.Timestamp(date).date()}|{sector}'.encode()
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'little'))
        result[indices]=rng.permutation(result[indices])
    return result

def fit_year(panel,clock,columns,recipe,year,panel_hash=None):
    name=recipe['name'];out=ROOT/'forecast_runs';out.mkdir(exist_ok=True)
    models=ROOT/'forecast_models';models.mkdir(exist_ok=True)
    target=out/f'{name}_{year}.csv.gz';meta=models/f'{name}_{year}.json'
    panel_hash=panel_hash or hashlib.sha256((ROOT/'data/candidate_features/panel.pkl').read_bytes()).hexdigest()
    if target.exists() and meta.exists():
        old=json.loads(meta.read_text())
        assert old['candidate_panel_sha256']==panel_hash and old['recipe']==recipe,'Forecast cache source/recipe changed; do not reuse'
        assert hashlib.sha256(target.read_bytes()).hexdigest()==old['forecast_sha256']
        return
    cut=pd.Timestamp(year,1,1);window=panel
    if recipe['years'] is not None:window=panel[panel.decision_date.ge(cut-pd.DateOffset(years=recipe['years']))]
    train=mature_training(window,cut,clock)
    q=panel[known(panel)&panel.decision_date.dt.year.eq(year)].sort_values(['decision_date','ticker']).reset_index(drop=True)
    if not len(q):raise ValueError('No complete current-known forecast queries')
    if recipe['decay_sessions']:
        location=pd.Series(np.arange(len(clock)),index=clock);age=clock.searchsorted(cut)-train.decision_date.map(location)
        train['sample_weight']*=np.exp2(-age/recipe['decay_sessions']);train['sample_weight']*=len(train)/train.sample_weight.sum()
    y=np.clip(train.target_risk_resid_h5.to_numpy(float),-8,8);w=train.sample_weight.to_numpy(float)
    if 'null_seed' in recipe:y=shuffle_within_sector(train,y,recipe['null_seed'])
    arrays,prep=prepare(train,[q],columns,cut);x,query=arrays;model_kind=recipe['model'];penalty=recipe['ridge_penalty']
    if isinstance(penalty,str):
        short=mature_training(panel[panel.decision_date.ge(cut-pd.DateOffset(years=5))],cut,clock)
        penalty=100*len(train)/len(short)
    record=dict(recipe=recipe,year=year,features=columns,rows=len(train),dates=train.decision_date.nunique(),training_start=str(train.decision_date.min()),
       max_training_label_end=str(train.label_end_h5.max()),refit_boundary=str(cut),training_weight_sum=float(w.sum()),effective_row_weights=float(w.sum()**2/(w@w)),
       target_clipped_rows=int((abs(train.target_risk_resid_h5)>8).sum()),ridge_penalty=penalty,query_rows=len(q),query_dates=q.decision_date.nunique(),
       candidate_panel_sha256=panel_hash,
       prediction_eligibility_uses_labels=False,account_qualification='Not granted by forecast fit',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    started=time.time()
    if model_kind=='ridge':model=Ridge(alpha=penalty).fit(x,y,sample_weight=w)
    elif model_kind=='elastic':model=ElasticNet(**PARAMETERS['elastic']).fit(x,y,sample_weight=w)
    elif model_kind=='huber_ridge':
        rw=w.copy();model=Ridge(alpha=penalty)
        for _ in range(10):
            model.fit(x,y,sample_weight=rw);resid=y-model.predict(x);scale=max(.1,1.4826*np.median(abs(resid-np.median(resid))))
            rw=w*np.minimum(1,1.345*scale/np.maximum(abs(resid),1e-12));rw*=len(rw)/rw.sum()
        record['final_residual_scale']=float(scale)
    elif model_kind=='rf':model=RandomForestRegressor(**PARAMETERS['rf']).fit(x,y,sample_weight=w)
    elif model_kind=='lgbm':
        from lightgbm import LGBMRegressor
        model=LGBMRegressor(**PARAMETERS['lgbm']).fit(x,y,sample_weight=w)
    elif model_kind=='xgb':
        from xgboost import XGBRegressor
        model=XGBRegressor(**PARAMETERS['xgb']).fit(x,y,sample_weight=w)
    elif model_kind in ['spline_ridge','rff_ridge']:
        transform=SplineTransformer(**PARAMETERS['spline_ridge']) if model_kind=='spline_ridge' else RBFSampler(**PARAMETERS['rff_ridge'])
        if model_kind=='spline_ridge':
            selected=[i for i,c in enumerate(columns) if c in prep['preprocessor'].continuous_ and np.std(x[:,i])>1e-10]
            other=[i for i in range(len(columns)) if i not in selected]
            xx=transform.fit_transform(x[:,selected]).astype(np.float32);qq=transform.transform(query[:,selected]).astype(np.float32)
            x=np.column_stack([xx,x[:,other]]).astype(np.float32);query=np.column_stack([qq,query[:,other]]).astype(np.float32)
            prep['spline_columns']=selected;prep['linear_columns']=other
        else:
            x=transform.fit_transform(x).astype(np.float32);query=transform.transform(query).astype(np.float32)
        prep['nonlinear_transform']=transform
        model=Ridge(alpha=penalty).fit(x,y,sample_weight=w)
    elif model_kind=='mlp':
        vd=sorted(train.decision_date.unique())[-252:];validation_start=pd.Timestamp(vd[0])
        inner=mature_training(window,validation_start,clock);val=train[train.decision_date.isin(vd)].reset_index(drop=True)
        inner_arrays,_=prepare(inner,[val],columns,validation_start)
        _,epochs,_=neural(inner_arrays[0],np.clip(inner.target_risk_resid_h5.to_numpy(),-8,8),inner.sample_weight.to_numpy(),inner_arrays[1],40,
             (inner_arrays[1],np.clip(val.target_risk_resid_h5.to_numpy(),-8,8),val.sample_weight.to_numpy()))
        pred,_,model=neural(x,y,w,query,epochs)
        record.update(epochs=epochs,inner_cutoff=str(validation_start),inner_max_label_end=str(inner.label_end_h5.max()))
    else:raise ValueError(model_kind)
    if model_kind!='mlp':
        pred=model.predict(query);insample=model.predict(x)
        record.update(training_weighted_mse=float(np.average((y-insample)**2,weights=w)),training_null_weighted_mse=float(np.average((y-np.average(y,weights=w))**2,weights=w)))
        if hasattr(model,'coef_') and len(model.coef_)==len(columns):record['standardized_coefficients']=dict(zip(columns,np.asarray(model.coef_,float)))
        if hasattr(model,'feature_importances_'):record['training_split_importance']=dict(zip(columns,np.asarray(model.feature_importances_,float)))
    pred=np.clip(pred,-8,8);assert np.isfinite(pred).all()
    forecast=q[['decision_date','ticker']].copy();forecast['score']=pred;forecast['refit_boundary']=cut;forecast['max_training_label_end']=train.label_end_h5.max()
    temporary=target.with_suffix('.partial');forecast.to_csv(temporary,index=False,compression='gzip');temporary.replace(target)
    with (models/f'{name}_{year}.pkl').open('wb') as stream:pickle.dump(dict(model=model,preprocessing=prep,record=record),stream,protocol=5)
    record['fit_seconds']=time.time()-started;record['forecast_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    meta.write_text(json.dumps(record,indent=2,default=str));print('LONG_FIT',name,year,len(train),round(record['fit_seconds'],1),flush=True)

def main(years,arms=None,controls=False):
    approval_file=ROOT/'data/assembled/approval.json'
    if not approval_file.exists():raise ValueError('Finish historical source qualification before fitting the comparison')
    approval=json.loads(approval_file.read_text())
    assert approval['source_qualification_passed'] and approval['prices_sha256']==hashlib.sha256((ROOT/'data/assembled/prices_candidate.pkl').read_bytes()).hexdigest()
    protocol=ROOT/'long_history_protocol.json';assert hashlib.sha256(protocol.read_bytes()).hexdigest()==(ROOT/'long_history_protocol.sha256').read_text().strip()
    config=json.loads(protocol.read_text());recipes=config['arms'];parameter_file=ROOT/'frozen_model_parameters.json'
    if parameter_file.exists():assert json.loads(parameter_file.read_text())==PARAMETERS
    else:parameter_file.write_text(json.dumps(PARAMETERS,indent=2))
    if arms:recipes=[r for r in recipes if r['name'] in arms];assert len(recipes)==len(arms)
    if controls:
        null_file=ROOT/'null_control_protocol.json'
        assert hashlib.sha256(null_file.read_bytes()).hexdigest()==null_file.with_suffix('.sha256').read_text().strip()
        recipes=json.loads(null_file.read_text())['arms']
    panel=pd.read_pickle(ROOT/'data/candidate_features/panel.pkl');assert pd.to_datetime(panel.decision_date).max()<pd.Timestamp('2024-01-01')
    feature_status=json.loads((ROOT/'data/candidate_features/status.json').read_text())
    assert feature_status['source_prices_sha256']==approval['prices_sha256'],'Feature panel source mismatch'
    assert feature_status['source_events_sha256']==approval['events_sha256']==hashlib.sha256((ROOT/'data/assembled/events.csv').read_bytes()).hexdigest(),'Feature terminal-event source mismatch'
    clock=pd.DatetimeIndex(pd.read_csv(ROOT/'data/assembled/calendar.csv',parse_dates=['Date']).Date)
    columns=json.loads((ROOT/'data/candidate_features/columns.json').read_text())
    panel_hash=hashlib.sha256((ROOT/'data/candidate_features/panel.pkl').read_bytes()).hexdigest()
    assert panel_hash==approval['panel_sha256'],'Approved feature panel changed'
    for year in years:
        assert 2007<=year<=2023
        for recipe in recipes:fit_year(panel,clock,columns,recipe,year,panel_hash)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('years',type=int,nargs='+');p.add_argument('--arms',nargs='+');p.add_argument('--controls',action='store_true');a=p.parse_args();main(a.years,a.arms,a.controls)
