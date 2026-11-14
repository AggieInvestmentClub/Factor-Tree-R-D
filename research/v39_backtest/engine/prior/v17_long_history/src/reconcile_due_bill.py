"""Remove a duplicated distribution only with explicit cum-rights evidence.

GGP traded with due bills through Aug27. Its closing quote included the
dividend entitlement. Pay that entitlement once at the Aug28 terminal event;
do not add it again to the preceding cum-rights close. No quotes are changed.
"""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def apply(frame):
    ric='GGP.N^H18';date=pd.Timestamp('2018-08-27');previous=pd.Timestamp('2018-08-24')
    key=(ric,date)
    if key not in frame.index:return frame,[]
    path=ROOT/'data/audit_requests/05_terminal_quotes_2018_ggp.pkl'
    meta=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
    raw=pd.read_pickle(path);g=raw[raw.Instrument.eq(ric)]
    assert set(g['TR.ISIN'].dropna())-{''}=={'US36174X1019'}
    assert set(g['TR.PRICECLOSE.CURRENCY'].dropna())-{''}=={'USD'}
    observed=[]
    for d in [previous,date]:
        a=pd.to_numeric(g.loc[pd.to_datetime(g['TR.CLOSEPRICE.DATE']).eq(d),'TR.CLOSEPRICE(ADJUSTED=0)']).dropna().unique()
        assert len(a)==1 and a[0]>0
        assert np.isclose(frame.loc[(ric,d),'PriceClose'],a[0],rtol=1e-8)
        observed.append(float(a[0]))
    corrected=100*(observed[1]/observed[0]-1)
    reported=float(frame.loc[key,'TotalReturn1D'])
    assert np.isclose(reported,65.9370725034,atol=1e-5), 'Re-review if provider revises the disputed return'
    frame.loc[key,'TotalReturn1D']=corrected
    return frame,[dict(ric=ric,date=str(date.date()),reported_return_pct=reported,corrected_return_pct=corrected,
        quote_sha256=meta['sha256'],policy='Cum-rights quote return; no dividend reinvestment before entitlement separates. Aggregate default cash14.642 + .376BPY units settled once Aug28.',
        due_bill_source='https://www.sec.gov/Archives/edgar/data/1496048/000119312518219153/d495214d8k.htm',
        final_terms_source='https://www.sec.gov/Archives/edgar/data/1545772/000119312518263673/d618193dsc13e3a.htm')]
