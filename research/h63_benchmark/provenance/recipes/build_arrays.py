"""Same causal58-feature state, covariance, masks and monthly training census."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('v39_config',Path(__file__).with_name('common.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
c.sys.path.insert(0,str(c.V34/'src'))
from risk_learning import correlation_loadings,orthogonal_rows
import numpy as np,pandas as pd,gc

def main():
    panel=pd.read_pickle(c.HERE/'data/panel.pkl');labels=pd.read_pickle(c.HERE/'data/horizon_labels.pkl');assert panel[['decision_date','ticker']].equals(labels[['decision_date','ticker']])
    clock=pd.DatetimeIndex(pd.read_csv(c.HERE/'data/assembled/calendar.csv',parse_dates=['Date']).Date);meta=c.read(c.V29/'data/metadata.json');cols=meta['columns'];oldnames=meta['names'];names=sorted(set(oldnames)|set(panel.loc[panel.model_eligible,'ticker']));mapping={n:i for i,n in enumerate(names)};oldix=np.array([mapping[n] for n in oldnames]);anchor=pd.Timestamp(meta['anchor']);oldates=pd.DatetimeIndex(meta['dates']);known=panel.model_eligible.eq(True)
    alldates=pd.DatetimeIndex(sorted(panel.loc[known,'decision_date'].unique()));dates=alldates[(clock.get_indexer(alldates)-clock.get_loc(anchor))%21==0];assert dates[dates<pd.Timestamp('2024-01-01')].equals(oldates)
    nt,nn=len(dates),len(names);out=c.HERE/'data/model';out.mkdir(exist_ok=True)
    fields=[f'{key}_h{h}' for h in (5,10,21) for key in ('future_return','future_parent_return','future_spy','target_risk_resid')];valid=np.isfinite(labels[fields].to_numpy()).all(1)&panel.v8_target_absolute_h5.notna().to_numpy()&panel.v8_target_spy_h5.notna().to_numpy();quality=pd.DataFrame(dict(date=panel.decision_date,known=known,valid=valid));quality=quality[quality.known]
    complete=(quality.groupby('date').valid.all()&quality.groupby('date').size().ge(10)).reindex(dates).to_numpy(bool);ends=labels.groupby('decision_date').label_end_h21.max().reindex(dates)
    future=labels.loc[known&labels.decision_date.isin(dates)];returns=future.pivot(index='decision_date',columns='ticker',values='future_return_h21').reindex(index=dates,columns=names).to_numpy();del labels,future,quality;gc.collect()
    beta=panel.pivot(index='decision_date',columns='ticker',values='beta_spy63').reindex(index=clock,columns=names);eligible=panel.pivot(index='decision_date',columns='ticker',values='model_eligible').reindex(index=clock,columns=names).fillna(False).astype(bool);beta.to_pickle(out/'daily_beta.pkl');eligible.to_pickle(out/'daily_eligible.pkl')
    mask=eligible.reindex(dates).to_numpy(copy=True);b=np.where(mask,beta.reindex(dates).to_numpy(),np.nan);del beta,eligible
    x=np.zeros((nt,nn,58));state=np.full((2,nn,36),np.nan);dmap={d:i for i,d in enumerate(dates)};source=panel.loc[known,['decision_date','ticker',*cols]];del panel;gc.collect()
    for date,g in source.groupby('decision_date',sort=True):
        ix=g.ticker.map(mapping).to_numpy(int);active=np.zeros(nn,bool);active[ix]=True;raw=np.nan_to_num(g[cols].to_numpy(float))
        for j,a in enumerate((.25,1-2**(-1/21))):
            prev=state[j,ix];state[j,ix]=(1-a)*np.where(np.isfinite(prev),prev,raw)+a*raw;state[j,~active]=np.nan
        if date in dmap:x[dmap[date]]=np.nan_to_num(np.concatenate([state[0],state[1,:,:22]],axis=1))
    oldx=np.load(c.V29/'data/x.npy',mmap_mode='r');assert np.array_equal(x[:len(oldates),oldix],oldx);del source,oldx;gc.collect()
    prices=pd.read_pickle(c.HERE/'data/assembled/prices_candidate.pkl');daily=np.full((len(clock),nn),np.nan)
    def rets(f):
        good=f.q.eq(1)&f.adj_close.gt(0)&f.tr_segment_id.notna();return (f.adj_close/f.adj_close.shift()-1).where(good&good.shift(fill_value=False)&f.tr_segment_id.eq(f.tr_segment_id.shift()))
    for name,j in mapping.items():daily[:,j]=rets(prices[name].reindex(clock)).to_numpy()
    market=rets(prices['SPY'].reindex(clock));del prices;gc.collect();np.save(out/'qualified_returns.npy',daily);market.to_csv(out/'market_returns.csv',header=['return'])
    centered=np.zeros((nt,63,nn));orth=np.zeros((nt,2,nn));corr=np.zeros((nt,5,nn));loads=np.zeros((nt,3,nn));diag=np.zeros((nt,nn));records=[]
    for j,date in enumerate(dates):
        end=clock.get_loc(date)+1;window=daily[end-63:end];valid=np.isfinite(window).all(0);ix=np.flatnonzero(valid);centered[j][:,ix]=window[:,ix]-window[:,ix].mean(0);diag[j]=np.sum(centered[j]**2,axis=0)/62;mask[j]&=valid&np.isfinite(b[j]);ones=mask[j].astype(float);bb=np.nan_to_num(b[j])*ones
        orth[j],_=orthogonal_rows(np.vstack([ones,bb]),mask[j],2);active=np.flatnonzero(mask[j]);mr=market.loc[:date].tail(63).to_numpy();assert np.isfinite(mr).all();loading,fractions=correlation_loadings(centered[j][:,active],mr);loads[j][:,active]=loading;corr[j],_=orthogonal_rows(np.vstack([ones,bb,loads[j]]),mask[j],5)
        records.append(dict(date=str(date.date()),valid_risk=int(valid.sum()),active=int(mask[j].sum()),complete_labels=bool(complete[j]),label_end=str(ends.iloc[j]),risk_source_end=str(date.date())))
    checks={}
    for key,array in [('centered',centered),('diag',diag),('mask',mask),('beta',np.nan_to_num(b))]:
        old=np.load(c.V29/f'data/{key}.npy',mmap_mode='r');a=array[:len(oldates),:,oldix] if key=='centered' else array[:len(oldates),oldix]
        # Advanced indexing for3D can reorder axes; explicit last-axis take avoids ambiguity.
        if key=='centered':a=np.take(array[:len(oldates)],oldix,axis=2)
        error=float(np.max(abs(a.astype(float)-old.astype(float)),initial=0));assert error<1e-10,(key,error);checks[key]=error
    # Keep old SVD/eigenspaces exactly, avoiding arbitrary sign/rotation differences.
    for key,array in [('orth',orth),('corr_orth',corr),('factor_loadings',loads)]:
        old=np.load(c.V29/'data/orth.npy') if key=='orth' else np.load(c.V34/f'data/{key}.npy')
        for j in range(len(oldates)):array[j][:,oldix]=old[j]
    gate=pd.read_csv(c.HERE/'data/gate.csv',index_col=0,parse_dates=True).multiplier.reindex(dates).fillna(1).to_numpy()
    arrays=dict(x=x,centered=centered,diag=diag,orth=orth,corr_orth=corr,factor_loadings=loads,mask=mask,beta=np.nan_to_num(b),gate=gate,returns=returns,complete=complete)
    for key,value in arrays.items():np.save(out/f'{key}.npy',value)
    pd.DataFrame(records).to_csv(out/'query_audit.csv',index=False);c.write(out/'metadata.json',dict(dates=[str(d.date()) for d in dates],label_ends=[None if pd.isna(d) else str(d.date()) for d in ends],names=names,columns=cols,anchor=str(anchor.date()),calendar=[str(d.date()) for d in clock],feature_dimension=58))
    c.write(c.HERE/'data/array_qa.json',dict(passed=True,old_feature_arrays_exact=True,old_risk_parity=checks,scheduled_queries=nt,forward_queries=int((dates>=c.START).sum()),stored_names=nn,risk_budget_denominator=500,prediction_eligibility_uses_future_labels=False,files={f.name:c.sha(f) for f in out.iterdir() if f.is_file()}));print('ARRAYS_EXTENDED',nt,nn,flush=True)
if __name__=='__main__':main()
