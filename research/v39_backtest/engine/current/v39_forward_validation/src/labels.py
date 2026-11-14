from common import *
import numpy as np,pandas as pd,gc
sys.path.insert(0,str(SOURCE/'src'));sys.path.insert(0,str(V23/'src'))
from build_labels import conditional_event,columns,HORIZONS
from historical_sources import label_quotes
from stock_features_v39 import _prepare,_pair,_labels,_label_wealth,SECTORS,IDIO_FLOOR

def main():
    panel=pd.read_pickle(HERE/'data/panel.pkl');panel=panel[panel.decision_date.ge('2023-11-01')].reset_index(drop=True)
    prices=pd.read_pickle(HERE/'data/assembled/prices_candidate.pkl');calendar=pd.DatetimeIndex(pd.read_csv(HERE/'data/assembled/calendar.csv',parse_dates=['Date']).Date);events=pd.read_csv(HERE/'data/assembled/events.csv',parse_dates=['effective_date'])
    out=panel[['decision_date','ticker']].copy();clock=pd.Series(calendar,index=calendar);out['execution_date']=panel.decision_date.map(clock.shift(-1))
    etfs={k:_prepare(prices[k],calendar) for k in set(SECTORS)|{'IYZ','SPY'}};le={k:label_quotes(prices[k],calendar) for k in etfs}
    for h in HORIZONS:
        out[f'label_end_h{h}']=panel.decision_date.map(clock.shift(-(h+1)))
        for col in columns(h):out[col]=np.nan
        out[f'future_spy_h{h}']=panel.decision_date.map(le['SPY'].adj_close.shift(-(h+1))/le['SPY'].adj_close.shift(-1)-1)
    null=pd.Series(np.nan,index=calendar);records=[]
    for count,(ticker,rows) in enumerate(panel.groupby('ticker',sort=True)):
        stock=_prepare(prices.get(ticker),calendar);sl=label_quotes(prices[ticker],calendar) if ticker in prices else stock;event=events[events.ticker.eq(ticker)]
        if len(event):
            assert len(event)==1
            if not conditional_event(event.iloc[0]):sl,record=_label_wealth(sl,event.iloc[0]);records.append(record)
        for sector in rows.sector.dropna().unique():
            selected=rows[rows.sector.eq(sector)];beta,_,idio=_pair(stock,etfs[sector]);sigma=idio.clip(lower=IDIO_FLOOR)
            a=selected.decision_date.map(sigma).to_numpy();b=selected.idio_vol21.to_numpy();assert np.array_equal(a,b,equal_nan=True)
            labels=_labels(sl,le[sector],beta,sigma,null,HORIZONS,'close');wanted=[col for h in HORIZONS for col in columns(h)];out.loc[selected.index,wanted]=labels.loc[pd.DatetimeIndex(selected.decision_date),wanted].to_numpy()
        if count%100==0:print('FORWARD_LABELS',count,flush=True)
    for col in ('future_return_h5','future_parent_return_h5','target_risk_resid_h5','future_spy_h5'):
        assert np.array_equal(out[col].to_numpy(),panel[col].to_numpy(),equal_nan=True),col
    old=pd.read_pickle(V23/'data/horizon_labels.pkl').set_index(['decision_date','ticker']);new=out.set_index(['decision_date','ticker']);overlap=old.index.intersection(new.index)
    for col in old.columns:
        a=old.loc[overlap,col];b=new.loc[overlap,col];valid=a.notna()
        if pd.api.types.is_numeric_dtype(a):assert np.allclose(a[valid],b[valid],rtol=0,atol=1e-10)
        else:pd.testing.assert_series_equal(a[valid],b[valid],check_names=False)
        old.loc[overlap,col]=b
    combined=pd.concat([old,new[new.index.get_level_values(0)>=pd.Timestamp(START)]]).sort_index().reset_index();combined.to_pickle(HERE/'data/horizon_labels.pkl');write(HERE/'data/label_qa.json',dict(passed=True,rows=len(combined),old_mature_labels_identical=True,terminal_applications=records,labels_sha256=sha(HERE/'data/horizon_labels.pkl')));print('LABELS_EXTENDED',len(combined),flush=True)
if __name__=='__main__':main()
