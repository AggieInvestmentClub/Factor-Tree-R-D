"""Unchanged annual H5 Huber gate, explicitly extended to the authorized boundary."""
from common import *
import numpy as np,pandas as pd,importlib.util,pickle,gc,time
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(SOURCE/'src'))

def main():
    core=SOURCE/'src/model_core_long.py';code=core.read_text().replace("FINAL_CUTOFF=pd.Timestamp('2024-01-01')","FINAL_CUTOFF=pd.Timestamp('"+BOUNDARY+"')")
    local=HERE/'src/model_core_v39.py';local.write_text(code);sys.path.insert(0,str(HERE/'src'))
    from model_core_v39 import known,mature_training,FrozenRidge
    panel=pd.read_pickle(HERE/'data/panel.pkl');clock=pd.DatetimeIndex(pd.read_csv(HERE/'data/assembled/calendar.csv',parse_dates=['Date']).Date);cols=read(SOURCE/'data/candidate_features/columns.json');folder=HERE/'forecasts';folder.mkdir(exist_ok=True)
    recipe=read(SOURCE/'forecast_models/huber_ridge_10y_2023.json')['recipe'];assert recipe==dict(name='huber_ridge_10y',model='huber_ridge',years=10,ridge_penalty=100,decay_sessions=None)
    for year in (2024,2025,2026):
        file=folder/f'huber_{year}.csv.gz';meta=file.with_suffix('.json')
        if meta.exists():assert read(meta)['forecast_sha256']==sha(file);continue
        cut=pd.Timestamp(year,1,1);window=panel[panel.decision_date.ge(cut-pd.DateOffset(years=10))];tr=mature_training(window,cut,clock);q=panel[known(panel)&panel.decision_date.dt.year.eq(year)].sort_values(['decision_date','ticker']).reset_index(drop=True)
        obj=FrozenRidge(cols);raw=obj.raw(tr,cut);obj.continuous_=[c for c in cols if not(c.startswith(('sector_','is_')) or c.endswith('_available'))];obj.low_=raw[obj.continuous_].quantile(.005);obj.high_=raw[obj.continuous_].quantile(.995)
        w=tr.sample_weight.to_numpy();y=np.clip(tr.target_risk_resid_h5.to_numpy(),-8,8);x=obj.design(tr,cut).to_numpy();scaler=StandardScaler().fit(x,sample_weight=w);x=scaler.transform(x);query=scaler.transform(obj.design(q,BOUNDARY).to_numpy());rw=w.copy();model=Ridge(alpha=100)
        for _ in range(10):
            model.fit(x,y,sample_weight=rw);resid=y-model.predict(x);scale=max(.1,1.4826*np.median(abs(resid-np.median(resid))));rw=w*np.minimum(1,1.345*scale/np.maximum(abs(resid),1e-12));rw*=len(rw)/rw.sum()
        forecast=q[['decision_date','ticker']].copy();forecast['score']=np.clip(model.predict(query),-8,8);forecast['refit_boundary']=cut;forecast['max_training_label_end']=tr.label_end_h5.max();forecast.to_csv(file,index=False)
        weightfile=HERE/f'models/huber_{year}.pkl';weightfile.write_bytes(pickle.dumps(dict(model=model,preprocessor=obj,scaler=scaler)))
        write(meta,dict(year=year,recipe=recipe,rows=len(tr),dates=tr.decision_date.nunique(),max_training_label_end=str(tr.label_end_h5.max()),refit_boundary=str(cut),forecast_sha256=sha(file),weights_sha256=sha(weightfile),panel_sha256=sha(HERE/'data/panel.pkl'),boundary_adapter_sha256=sha(local),design_sha256=sha(HERE/'design.json')))
        print('HUBER_FIT',year,len(tr),flush=True);del tr,q,x,query,raw,window;gc.collect()
    # A bounded2017+ prefix supplies the252 complete mature-date lookback; verify below.
    forecasts=[pd.read_csv(SOURCE/f'forecast_runs/huber_ridge_10y_{y}.csv.gz',parse_dates=['decision_date','refit_boundary','max_training_label_end']) for y in range(2017,2024)]
    forecasts += [pd.read_csv(folder/f'huber_{y}.csv.gz',parse_dates=['decision_date','refit_boundary','max_training_label_end']) for y in (2024,2025,2026)]
    f=pd.concat(forecasts,ignore_index=True);meta=read(V29/'data/metadata.json');anchor=pd.Timestamp(meta['anchor']);known_dates=pd.DatetimeIndex(sorted(panel.loc[known(panel)&panel.decision_date.ge('2023-01-01'),'decision_date'].unique()));dates=known_dates[(clock.get_indexer(known_dates)-clock.get_loc(anchor))%21==0]
    from fast_gate import compute_gate
    g=compute_gate(f,panel[panel.decision_date.ge('2017-01-01')],clock,dates,end_before=BOUNDARY)
    old=pd.read_csv(V23/'account_inputs/original_huber_gate_cadence21.csv',index_col=0,parse_dates=True);overlap=g.index.intersection(old.index)
    assert np.array_equal(g.loc[overlap,'multiplier'],old.loc[overlap,'multiplier'])
    # Insufficient historical complete-date coverage would invalidate truncation parity.
    assert (g.loc[g.index>=START,'mature_dates']==252).all()
    combined=pd.concat([old,g[g.index>=START]]);combined.to_csv(HERE/'data/gate.csv')
    write(HERE/'data/gate_qa.json',dict(passed=True,prior_2023_gate_multiplier_exact=True,overlap_dates=len(overlap),forward_queries=int((g.index>=START).sum()),strict_label_end_before_query=bool((pd.to_datetime(g.history_label_end_max)<g.index).all()),gate_sha256=sha(HERE/'data/gate.csv')));print('GATE_EXTENDED',len(combined),flush=True)
if __name__=='__main__':main()
