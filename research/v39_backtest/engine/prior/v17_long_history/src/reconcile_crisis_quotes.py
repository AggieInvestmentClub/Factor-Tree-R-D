"""Observed consolidated OC windows for five 2008 NYSE-venue gaps.

Same dated ISIN, USD, no intervening distribution, matching boundary prices,
and a constant split scale are required. These are valuation/return repairs,
not synthetic bars: venue-specific range/volume features are withheld.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'
WINDOWS=[('FNM.N^G10','FNMA.PK','sep'),('FRE.N^G10','FMCC.PK','sep'),
         ('GNW.N','GNW','nov'),('GGP.N^D09','GGP^H18','nov'),('UIS.N','UIS','nov')]

def read_capture(key):
    p=DATA/'audit_requests'/f'{key}.pkl';m=json.loads(p.with_suffix('.json').read_text())
    assert m['state']=='captured_not_audited'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
    return pd.read_pickle(p),dict(path=str(p.relative_to(ROOT)),sha256=m['sha256'])

def dated_isin(frame,ric,date):
    g=frame[frame.Instrument.eq(ric)][['TR.ISIN.DATE','TR.ISIN']].dropna().drop_duplicates()
    g=g[g['TR.ISIN.DATE'].le(date)].sort_values('TR.ISIN.DATE')
    assert len(g) and g.iloc[-1]['TR.ISIN'].startswith('US')
    return g.iloc[-1]['TR.ISIN']

def observed_series(g,field):
    z=g[[f'TR.{field}.DATE',f'TR.{field}(ADJUSTED=0)']].dropna().drop_duplicates()
    z.columns=['date','value'];assert not z.date.duplicated().any()
    return z.set_index('date').value.astype(float).sort_index()

def reconcile_window(original,quotes,calendar):
    """Keep anchor return unchanged; recompute every subsequent window return.

    Recomputing the resumption day avoids double counting the missing days.
    The right boundary's identical raw price joins back to the old series.
    """
    q=quotes.reindex(calendar);assert q[['open','close']].notna().all().all()
    assert q[['open','close']].gt(0).all().all()
    a,b=calendar[0],calendar[-1]
    scale=original.loc[a,'PriceClose']/q.loc[a,'close']
    assert np.isfinite(scale) and scale>0
    assert np.isclose(original.loc[b,'PriceClose'],q.loc[b,'close']*scale,rtol=1e-8,atol=1e-8),'Boundary basis mismatch'
    out=original.copy();ix=calendar[1:]
    out.loc[ix,'PriceOpen']=q.loc[ix,'open']*scale
    out.loc[ix,'PriceClose']=q.loc[ix,'close']*scale
    out.loc[ix,'RawClose']=q.loc[ix,'close']
    out.loc[ix,'TotalReturn1D']=q.close.pct_change(fill_method=None).loc[ix]*100
    # Do not splice a new venue's range or volume into old technical features.
    out.loc[ix,['PriceHigh','PriceLow','Volume']]=np.nan
    assert out.loc[a,'TotalReturn1D']==original.loc[a,'TotalReturn1D']
    assert np.isclose(np.prod(1+out.loc[ix,'TotalReturn1D']/100),q.loc[b,'close']/q.loc[a,'close'])
    return out,scale

def apply_windows(combined):
    if combined.index.get_level_values('date').max()<pd.Timestamp('2008-01-01'):return combined,[]
    identity,ip=read_capture('00_crisis_security_histories')
    clock=pd.DatetimeIndex(pd.read_csv(DATA/'direct_spy_calendar.csv',parse_dates=['Date']).Date)
    records=[]
    for ric,route,tag in WINDOWS:
        raw,rp=read_capture(f'00_gap_direct_prices_{tag}_primary')
        dividends,dp=read_capture(f'00_gap_direct_dividends_{tag}_primary')
        g=raw[raw.Instrument.eq(route)];close=observed_series(g,'CLOSEPRICE');op=observed_series(g,'OPENPRICE')
        a,b=close.index.min(),close.index.max();dates=clock[(clock>=a)&(clock<=b)]
        assert b<pd.Timestamp('2024-01-01')
        for d in dates:assert dated_isin(identity,ric,d)==dated_isin(identity,route,d),'Historical issue mismatch'
        currency=set(g['TR.PRICECLOSE.CURRENCY'].dropna().astype(str).str.strip())-{''}
        assert currency=={'USD'}
        div=dividends[dividends.Instrument.eq(route)];assert len(div)
        ex=pd.to_datetime(div['TR.DIVEXDATE'],errors='coerce')
        assert not ex.between(a,b).any(),'Cash or noncash action needs separate wealth reconstruction'
        split=(pd.to_numeric(g['TR.CLOSEPRICE(ADJUSTED=1)'],errors='coerce')/
               pd.to_numeric(g['TR.CLOSEPRICE(ADJUSTED=0)'],errors='coerce')).dropna().astype(float)
        assert split.max()/split.min()-1<1e-5,'Nonconstant share basis in window'
        original=combined.xs(ric).copy();old_missing=int(original.reindex(dates).PriceClose.isna().sum())
        repaired,scale=reconcile_window(original,pd.DataFrame({'open':op,'close':close}),dates)
        for col in repaired:
            if col not in combined:combined[col]=np.nan
        ix=pd.MultiIndex.from_product([[ric],dates[1:]],names=['ric','date'])
        combined.loc[ix,repaired.columns]=repaired.loc[dates[1:]].to_numpy()
        records.append(dict(ric=ric,route=route,from_date=str(a.date()),through_date=str(b.date()),
            recovered_missing_closes=old_missing,split_price_scale=scale,sources=[ip,rp,dp],
            policy='Observed consolidated OC and cash-audited raw returns; both boundary closes reconcile; no range/volume feature splice. Missing NYSE venue quote is not a market-wide halt.'))
    return combined,records

def test():
    dates=pd.bdate_range('2008-09-05',periods=4)
    old=pd.DataFrame({'PriceOpen':[10,np.nan,np.nan,8.], 'PriceClose':[10,np.nan,np.nan,8.],
                      'PriceHigh':[10,np.nan,np.nan,8.], 'PriceLow':[10,np.nan,np.nan,8.],
                      'Volume':1.,'TotalReturn1D':[2,np.nan,np.nan,-20.]},index=dates)
    q=pd.DataFrame({'open':[10,7,9,8.],'close':[10,7,9,8.]},index=dates)
    z,s=reconcile_window(old,q,dates)
    assert np.isclose((1+z.TotalReturn1D.iloc[1:]/100).prod(),.8)
    assert np.isclose(z.TotalReturn1D.iloc[-1],100*(8/9-1))
    assert z.loc[dates[1:],'PriceHigh'].isna().all() and z.iloc[0].TotalReturn1D==2
    try:reconcile_window(old,q.assign(close=[10,7,9,7.]),dates)
    except AssertionError:pass
    else:raise AssertionError('Mismatched right boundary accepted')
    report=dict(passed=True,checks=4,lookahead=False,synthetic_test_only=True)
    (ROOT/'reports/crisis_quote_tests.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':test()
