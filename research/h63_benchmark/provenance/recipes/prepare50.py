"""Qualified horizon labels and past-only daily context,without changing sources."""
from study50 import *
import importlib.util
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def conditional(row):
    return ((pd.notna(row.get('cvr_upper')) and row.get('cvr_upper',0)>0) or (pd.notna(row.get('unpriced_rights')) and bool(row.get('unpriced_rights',False))) or (pd.notna(row.get('label_wealth_qualified')) and not bool(row.get('label_wealth_qualified'))))
def labels(d):
    specpath=ROOT/'work/v44_mc_workspace/prior/v17_long_history/src';sys.path.insert(0,str(specpath));from historical_sources import label_quotes
    f=load_module('v50_stock_primitives',SOURCE/'src/stock_features_v39.py');clock=pd.DatetimeIndex(d['calendar']);dates=d['dates'];prices=pd.read_pickle(SOURCE/'data/assembled/prices_candidate.pkl');events=pd.read_csv(SOURCE/'data/assembled/events.csv',parse_dates=['effective_date']);zero=pd.Series(0.,index=clock);one=zero+1
    output={h:np.full(d['mask'].shape,np.nan) for h in HORIZONS};records=[]
    for k,name in enumerate(d['names']):
        q=label_quotes(prices[name],clock);event=events[events.ticker.eq(name)]
        if len(event):
            assert len(event)==1
            if not conditional(event.iloc[0]):q,record=f._label_wealth(q,event.iloc[0]);records.append(record)
        values=f._labels(q,q,zero,one,one,HORIZONS,'close')
        for h in HORIZONS:output[h][:,k]=values.reindex(dates)[f'future_return_h{h}'].to_numpy()
        if k%200==0:print('LABELS',k,flush=True)
    good=d['mask']&np.isfinite(d['returns']);error=float(abs(output[21][good]-d['returns'][good]).max());assert error<1e-10
    # Eligibility may be broader than label qualification;never change prediction masks.
    original_missing=int((d['mask']&~np.isfinite(d['returns'])).sum());new_missing=int((d['mask']&~np.isfinite(output[21])).sum())
    assert np.array_equal(np.isnan(output[21][d['mask']]),np.isnan(d['returns'][d['mask']]))
    ends={h:pd.Series(clock,index=clock).shift(-(h+1)).reindex(dates).to_numpy() for h in HORIZONS};common=d['complete'].copy();coverage=[]
    for h in HORIZONS:
        valid=(np.isfinite(output[h])|~d['mask']).all(1)&~pd.isna(ends[h]);common &=valid;np.save(HERE/f'data/returns_h{h}.npy',output[h]);np.save(HERE/f'data/ends_h{h}.npy',ends[h]);coverage.append(dict(horizon=h,complete_queries=int(valid.sum()),eligible_missing_labels=int((d['mask']&~np.isfinite(output[h])).sum())))
    np.save(HERE/'data/common_complete.npy',common);pd.DataFrame(coverage).to_csv(HERE/'reports/label_coverage.csv',index=False)
    market=pd.read_csv(SOURCE/'data/model/market_returns.csv',index_col=0,parse_dates=True).iloc[:,0]
    for h in HORIZONS:
        m=[]
        for date,end in zip(dates,ends[h]):
            if pd.isna(end):m.append(np.nan);continue
            a=clock.get_loc(date);b=clock.get_loc(end);r=market.reindex(clock[a+2:b+1]).to_numpy();assert len(r)==h and np.isfinite(r).all();m.append(np.prod(1+r)-1)
        np.save(HERE/f'data/market_h{h}.npy',np.array(m))
    write(HERE/'reports/labels_qa.json',dict(passed=True,h21_error=error,h21_missingness_identical=True,common_complete_queries=int(common.sum()),original_complete_queries=int(d['complete'].sum()),terminal_wealth=records))
    del prices,output;gc.collect()
def context(d):
    clock=pd.DatetimeIndex(d['calendar']);market=pd.read_csv(SOURCE/'data/model/market_returns.csv',index_col=0,parse_dates=True).iloc[:,0].reindex(clock);daily=np.load(SOURCE/'data/model/qualified_returns.npy');stock=pd.DataFrame(daily,index=clock,columns=d['names']);eligible=pd.read_pickle(SOURCE/'data/model/daily_eligible.pkl').reindex(index=clock,columns=d['names']);beta=pd.read_pickle(SOURCE/'data/model/daily_beta.pkl').reindex(index=clock,columns=d['names'])
    log=np.log1p(market);down=log.clip(upper=0).pow(2);dd={h:np.sqrt(down.ewm(halflife=h,min_periods=126,adjust=False).mean()) for h in (20,60,120)}
    r21=np.expm1(np.log1p(stock).rolling(21,min_periods=21).sum());res=stock.subtract(beta.multiply(market,axis=0));resvol=res.rolling(21,min_periods=21).std();qualified=eligible&np.isfinite(r21)&np.isfinite(resvol);count=qualified.sum(1)
    z=pd.DataFrame(dict(downside20=dd[20],downside20_minus60=dd[20]-dd[60],downside60_minus120=dd[60]-dd[120],mean120=log.ewm(halflife=120,min_periods=126,adjust=False).mean(),dispersion21=r21.where(qualified).std(1),median_residual_vol21=resvol.where(qualified).median(1),breadth21=(r21.gt(0)&qualified).sum(1)/count.where(count>=10)),index=clock)
    correlation=np.full(len(clock),np.nan);em=eligible.to_numpy(bool)
    for t in range(62,len(clock)):
        w=daily[t-62:t+1];mask=em[t]&np.isfinite(w).all(0);w=w[:,mask];c=w-w.mean(0);sd=np.sqrt((c*c).sum(0)/62);c=c[:,sd>1e-10]/sd[sd>1e-10];n=c.shape[1]
        if n>=10:correlation[t]=((c.sum(1)**2).sum()/62-n)/(n*(n-1))
    z['mean_correlation63']=correlation;z=z[list(CONTEXT)];assert np.isfinite(z.reindex(d['dates']).to_numpy()).all()
    z.to_csv(HERE/'data/daily_context.csv.gz');np.save(HERE/'data/query_context.npy',z.reindex(d['dates']).to_numpy(np.float32))
    write(HERE/'reports/context_qa.json',dict(passed=True,features=list(z),first_complete=str(z.dropna().index[0]),all_forecast_context_finite=True,source_end_is_decision_close=True));print('CONTEXT_READY',flush=True)
def main():
    restrict();freeze();d=data();labels(d);context(d);write(HERE/'reports/preparation_qa.json',dict(passed=True,source_unchanged=source_unchanged()));print('PREPARATION_DONE',flush=True)
if __name__=='__main__':main()
