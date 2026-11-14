"""One documented ex-date entitlement without contemporaneous traded marks.

Do not move the ex-date to first trading or backfill the first trading price.
Sears' Orchard entitlement is separately settled by the account engine.
"""
import sys,json,hashlib
sys.dont_write_bytecode=True
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'

def observed(ric,date):
    values=set();used=[];currency=set()
    for p in sorted((DATA/'audit_requests').glob('01_spin_components_*.pkl')):
        m=json.loads(p.with_suffix('.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
        f=pd.read_pickle(p);g=f[f.Instrument.eq(ric)]
        z=pd.to_numeric(g.loc[pd.to_datetime(g['TR.CLOSEPRICE.DATE']).eq(date),'TR.CLOSEPRICE(ADJUSTED=0)'],errors='coerce').dropna()
        if len(z):
            values.update(z.tolist());used.append(dict(request=p.stem,sha256=m['sha256']))
            currency.update(set(g['TR.PRICECLOSE.CURRENCY'].dropna())-{''})
    assert len(values)==1 and min(values)>0 and currency=={'USD'},(ric,date,values,currency)
    return next(iter(values)),used

def definition():
    ticker='SHLD.OQ^J18';ex=pd.Timestamp('2011-12-14');settle=pd.Timestamp('2012-01-03')
    previous,s0=observed(ticker,pd.Timestamp('2011-12-13'));parent,s1=observed(ticker,ex)
    common,s2=observed('OSH.OQ^F13',settle);preferred,s3=observed('OSHSQ.PK^B14',settle)
    row=dict(ticker=ticker,ex_date=str(ex.date()),settlement_date=str(settle.date()),
        raw_parent_previous=previous,raw_parent_ex_close=parent,
        receivers=[dict(ric='OSH.OQ^F13',ratio=1/22.141777,raw_settlement_close=common),
                   dict(ric='OSHSQ.PK^B14',ratio=1/22.141777,raw_settlement_close=preferred)],
        raw_settlement_per_parent=(common+preferred)/22.141777,
        valuation_policy='zero_until_observed_settlement; conditional_only',
        source='https://infomemo.theocc.com/infomemos?date=201112&number=29943',
        terms_source='https://www.sec.gov/Archives/edgar/data/896842/000119312511344104/d198486d424b3.htm',
        quotes=s0+s1+s2+s3,
        qualification='Unobserved receivable marks; interim NAV, Sharpe and risk path remain conditional; no future price backfill.')
    return row

def apply(frame):
    row=definition();key=(row['ticker'],pd.Timestamp(row['ex_date']))
    assert key in frame.index
    old=float(frame.loc[key,'TotalReturn1D']);value=100*(row['raw_parent_ex_close']/row['raw_parent_previous']-1)
    frame.loc[key,'TotalReturn1D']=value
    if 'UnpricedDistributionClaim' not in frame:frame['UnpricedDistributionClaim']=0
    dates=frame.index.get_level_values('date');names=frame.index.get_level_values('ric')
    interval=(names==row['ticker'])&(dates>=pd.Timestamp(row['ex_date']))&(dates<=pd.Timestamp(row['settlement_date']))
    frame.loc[interval,'UnpricedDistributionClaim']=1
    # Incomplete economic wealth is excluded from feature/label construction,
    # while observed parent quotes remain executable for existing holdings.
    path=DATA/'distribution_audit/delayed_claims.json';path.write_text(json.dumps([row],indent=2))
    return frame,[dict(policy='Separate delayed receivable; no future price assigned on ex-date',
        source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ticker=row['ticker'],ex_date=row['ex_date'],reported_return_pct=old,parent_only_return_pct=value)]

if __name__=='__main__':
    row=definition();(DATA/'distribution_audit/delayed_claims.json').write_text(json.dumps([row],indent=2));print(json.dumps(row,indent=2))
