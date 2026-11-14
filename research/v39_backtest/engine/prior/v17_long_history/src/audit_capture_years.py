"""Incremental, hash-checked annual field alignment. No market-data imputation."""
from pathlib import Path
import hashlib,json
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'
FIELDS=['TR.PriceOpen','TR.PriceHigh','TR.PriceLow','TR.PriceClose','TR.Volume','TR.TotalReturn1D']
AUDIT_VERSION=3

def align(frame,year):
    columns=[];issues=[]
    for field in FIELDS:
        x=pd.DataFrame(dict(ric=frame.Instrument,date=pd.to_datetime(frame[(field+'.date').upper()],errors='coerce'),value=pd.to_numeric(frame[field.upper()],errors='coerce')))
        x=x[x.date.dt.year.eq(year)&x.value.notna()].copy()
        counts=x.groupby(['ric','date']).value.nunique();bad=counts[counts>1].index
        issues.extend(dict(ric=r,date=str(d.date()),field=field) for r,d in bad)
        x=x.set_index(['ric','date']);x=x.loc[~x.index.isin(bad)]
        x=x.loc[~x.index.duplicated()]
        columns.append(x.value.rename(field.removeprefix('TR.')))
    return pd.concat(columns,axis=1),issues

def main():
    raw=DATA/'annual_dated_raw';out=DATA/'annual_aligned';out.mkdir(exist_ok=True)
    records=json.loads((raw/'manifest.json').read_text());byfile={}
    for r in records:
        if 'sha256' in r:byfile[f"{r['year']}_{r['batch']:03}.pkl"]=r
    for year in range(2000,2024):
        done=out/f'{year}_status.json'
        if done.exists() and json.loads(done.read_text()).get('audit_version')==AUDIT_VERSION:continue
        namespath=raw/f'{year}_names.json';planpath=raw/f'{year}_batch_plan.json'
        if not namespath.exists() or not planpath.exists():continue
        names=json.loads(namespath.read_text());plan=json.loads(planpath.read_text());n=(len(names)+plan['batch_size']-1)//plan['batch_size']
        paths=[raw/f'{year}_{i:03}.pkl' for i in range(n)]
        if not all(p.name in byfile and p.exists() for p in paths):continue
        frames=[];conflicts=[]
        for p in paths:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==byfile[p.name]['sha256'], 'Raw hash mismatch'
            frame=pd.read_pickle(p);assert set(frame.Instrument).issubset(names)
            z,issues=align(frame,year);frames.append(z);conflicts+=issues
        z=pd.concat(frames).sort_index();assert not z.index.duplicated().any()
        ohlc=z[['PriceOpen','PriceHigh','PriceLow','PriceClose']]
        # Nullable pandas columns propagate pd.NA into isfinite. .all(skipna=True)
        # alone would call an entirely missing bar valid: require explicit notna.
        z['complete_ohlc']=ohlc.notna().all(axis=1)&np.isfinite(ohlc.astype(float)).all(axis=1)&ohlc.gt(0).all(axis=1)
        tolerance=ohlc.abs().max(axis=1).clip(lower=1)*1e-7
        z['consistent_ohlc']=z.complete_ohlc&z.PriceHigh.add(tolerance).ge(ohlc.max(axis=1))&z.PriceLow.sub(tolerance).le(ohlc.min(axis=1))
        z['qualified_ohlc_tr']=z.consistent_ohlc&z.TotalReturn1D.notna()&z.TotalReturn1D.gt(-100)
        z.to_pickle(out/f'{year}.pkl')
        rows=[]
        for name in names:
            g=z.xs(name,level='ric') if name in z.index.get_level_values('ric') else z.iloc[:0]
            rows.append(dict(year=year,ric=name,dates=len(g),complete_ohlc=int(g.complete_ohlc.sum()),ohlc_tr=int(g.qualified_ohlc_tr.sum()),first=str(g.index.min()) if len(g) else None,last=str(g.index.max()) if len(g) else None))
        pd.DataFrame(rows).to_csv(out/f'{year}_coverage.csv',index=False)
        rec=dict(audit_version=AUDIT_VERSION,year=year,requested_names=len(names),raw_batches=n,field_dates=len(z),qualified_ohlc_tr=int(z.qualified_ohlc_tr.sum()),conflicts=conflicts,inconsistent_ohlc=int((z.complete_ohlc&~z.consistent_ohlc).sum()),state='candidate; no lineage or terminal accounting approval')
        done.write_text(json.dumps(rec,indent=2));print('ALIGNED',year,len(z),int(z.qualified_ohlc_tr.sum()),flush=True)
    summaries=[json.loads(p.read_text()) for p in out.glob('*_status.json')]
    pd.DataFrame(summaries).drop(columns=['conflicts'],errors='ignore').sort_values('year').to_csv(out/'summary.csv',index=False)

if __name__=='__main__':main()
