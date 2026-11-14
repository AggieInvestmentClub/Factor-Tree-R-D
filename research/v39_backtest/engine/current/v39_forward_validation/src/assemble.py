"""Date-aligned forward sources. Candidate output is not account approval."""
from common import *
import numpy as np,pandas as pd,argparse,gc
sys.path.insert(0,str(SOURCE/'src'))
from historical_sources import canonical_ric,convert,membership_names

FIELDS={f'TR.{f.upper()}':(f'TR.{f.upper()}.DATE',f) for f in ('PriceOpen','PriceHigh','PriceLow','PriceClose','Volume','TotalReturn1D')}
FIELDS.update({'TR.CLOSEPRICE(ADJUSTED=0)':('TR.CLOSEPRICE.DATE','RawClose'),'TR.CLOSEPRICE(ADJUSTED=1)':('TR.CLOSEPRICE.DATE','DirectAdjustedClose'),'TR.OPENPRICE(ADJUSTED=0)':('TR.OPENPRICE.DATE','RawOpen'),'TR.OPENPRICE(ADJUSTED=1)':('TR.OPENPRICE.DATE','DirectAdjustedOpen')})

def quote_jobs():
    plans=['capture_plan.json','supplement_plan.json','event_quotes_plan.json','xlre_repair_plan.json']
    return [j for p in plans if (HERE/'data'/p).exists() for j in read(HERE/'data'/p)['requests'] if j['kind']=='quotes']

def align():
    jobs=quote_jobs();out=HERE/'data/aligned';out.mkdir(exist_ok=True);records=[]
    for job in jobs:
        f=HERE/'data/raw'/(job['name']+'.pkl');dest=out/f.name
        if not f.exists():continue
        meta=read(f.with_suffix('.json'));assert sha(f)==meta['response_sha256']
        if dest.exists():assert read(dest.with_suffix('.json'))['raw_sha256']==sha(f);continue
        frame=pd.read_pickle(f);pieces=[];conflicts=[]
        for field,(datecol,target) in FIELDS.items():
            z=pd.DataFrame(dict(ric=frame.Instrument.map(canonical_ric),date=pd.to_datetime(frame[datecol],errors='coerce'),value=pd.to_numeric(frame[field],errors='coerce')))
            z=z[z.date.ge(job['parameters']['SDate'])&z.date.le(job['parameters']['EDate'])&z.value.notna()].copy()
            multiple=z.groupby(['ric','date']).value.nunique();bad=multiple[multiple>1]
            conflicts.extend(dict(ric=r,date=str(d.date()),field=field) for r,d in bad.index)
            assert bad.empty,conflicts
            pieces.append(z.drop_duplicates(['ric','date']).set_index(['ric','date']).value.astype(float).rename(target))
        joined=pd.concat(pieces,axis=1).sort_index();joined.to_pickle(dest)
        m=dict(raw_sha256=sha(f),aligned_sha256=sha(dest),rows=len(joined),conflicts=conflicts,separately_dated_fields=list(FIELDS));write(dest.with_suffix('.json'),m);records.append(job['name'])
    print('ALIGNED_NEW',len(records),flush=True)

def assemble():
    align();jobs=quote_jobs();files=[HERE/'data/aligned'/(j['name']+'.pkl') for j in jobs];assert all(f.exists() for f in files),'Wait for complete quote acquisition'
    z=pd.concat([pd.read_pickle(f) for f in files]).sort_index();assert not z.index.duplicated().any()
    spy=z.xs('SPY');observed=spy.index[spy.PriceClose.gt(0)&spy.TotalReturn1D.notna()&spy.Volume.gt(0)];direct=spy.index[spy.RawClose.gt(0)]
    assert observed.equals(direct),'Fundamental/direct SPY calendars disagree'
    assert observed[-1]==pd.Timestamp(END)
    oldclock=pd.DatetimeIndex(pd.read_csv(SOURCE/'data/assembled/calendar.csv',parse_dates=['Date']).Date)
    clock=oldclock.union(observed[observed>oldclock[-1]]);clock.name='Date';assert clock.is_unique
    # Compare with the earlier independently qualified direct-source session calendar.
    prior=read(PRIOR/'v10_final_holdout/holdout_data_qualified.json')
    oldforward=pd.read_pickle(PRIOR/'v10_final_holdout/data/qualified/etfs_full.pkl')
    if isinstance(oldforward,dict):
        pspy=oldforward.get('SPY')
        if pspy is not None:
            d=pspy.index;d=pd.DatetimeIndex(d[(d>=pd.Timestamp('2024-01-01'))&(d<=pd.Timestamp('2026-11-05'))]);assert clock[(clock>=d.min())&(clock<=d.max())].equals(d)
    del oldforward;gc.collect()
    prices=pd.read_pickle(SOURCE/'data/assembled/prices_candidate.pkl');legacy_unpriced=set(membership_names(pd.read_csv(SOURCE/'data/membership_candidate_snapshots.csv.gz')).ric)-set(prices);audit=[];overlaps=[];quote_checks=[]
    for name,g in z.groupby(level='ric',sort=False):
        f=g.droplevel('ric');n=convert(f,observed)
        n['raw_open']=f.RawOpen.reindex(observed);n['direct_adjusted_close']=f.DirectAdjustedClose.reindex(observed)
        for a,b in [('PriceClose','DirectAdjustedClose'),('PriceOpen','DirectAdjustedOpen')]:
            both=f[a].notna()&f[b].notna();relative=(f.loc[both,a]/f.loc[both,b]-1).abs();bad=relative>1e-5
            quote_checks.extend(dict(ric=name,date=str(d.date()),field=a,relative_error=float(relative[d])) for d in relative.index[bad])
        if name in prices:
            old=prices[name];both=old.index.intersection(n.index);valid=old.loc[both,'q'].eq(1)&n.loc[both,'q'].eq(1)
            ret=(old.loc[both,'total_return']-n.loc[both,'total_return']).abs();overlaps.append(dict(ric=name,comparable=int(valid.sum()),max_return_difference=float(ret[valid].max()) if valid.any() else None,dates_difference_over_1bp=int((ret[valid]>1e-4).sum())))
            endpoint=oldclock[-1]
            if endpoint in n.index and old.loc[endpoint,'q']==1 and n.loc[endpoint,'q']==1:
                factor=old.loc[endpoint,'adj_close']/n.loc[endpoint,'adj_close'];n.loc[:,['adj_open','adj_high','adj_low','adj_close']]*=factor
                # Segment identifiers have no economic magnitude. Join only the verified adjacent segment.
                n['tr_segment_id']+=old.loc[endpoint,'tr_segment_id']-n.loc[endpoint,'tr_segment_id']
                join='observed_overlap'
            else:join='no_continuous_boundary';n['tr_segment_id']+=old.tr_segment_id.max()+10000
            prices[name]=pd.concat([old,n[n.index>endpoint]]).reindex(clock)
        else:
            join='new_issue_warmup'
            if name in legacy_unpriced:
                n=n[n.index>oldclock[-1]];join='legacy_missing_issue_preserved_pre2024; forward_quotes_restart_normal_warmup'
            prices[name]=n.reindex(clock)
        audit.append(dict(ric=name,bridge=join,first=str(n[n.q.eq(1)].index.min()),last=str(n[n.q.eq(1)].index.max()),forward_missing_OC=int(n.loc[n.index>=START,'q'].ne(1).sum()),malformed_high_low=int((n.q.eq(1)&n.source_available.ne(1)).sum())))
    # All original names remain in the arrays; inactive old names have no manufactured forward quotes.
    for name in list(prices):prices[name]=prices[name].reindex(clock)
    out=HERE/'data/assembled';out.mkdir(exist_ok=True);pd.to_pickle(prices,out/'prices_candidate.pkl');pd.DataFrame({'Date':clock}).to_csv(out/'calendar.csv',index=False);z.to_pickle(out/'aligned_fields.pkl')
    pd.DataFrame(audit).to_csv(out/'coverage.csv',index=False);pd.DataFrame(overlaps).to_csv(out/'overlap_2023.csv',index=False);pd.DataFrame(quote_checks,columns=['ric','date','field','relative_error']).to_csv(out/'direct_quote_disagreements.csv',index=False)
    write(out/'status.json',dict(state='CANDIDATE; event and distribution qualification required',source_conflicts=[],end_year=2026,calendar_sessions=len(clock),forward_sessions=int((clock>=START).sum()),calendar_counts=clock.to_series().groupby(clock.year).size().to_dict(),prices_sha256=sha(out/'prices_candidate.pkl'),old_source_sha256=sha(SOURCE/'data/assembled/prices_candidate.pkl'),all_old_prefixes_preserved=True,quote_disagreements=len(quote_checks)))
    print('ASSEMBLED',len(prices),int((clock>=START).sum()),'QUOTE_DISAGREEMENTS',len(quote_checks),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['align','assemble']);a=p.parse_args();align() if a.stage=='align' else assemble()
