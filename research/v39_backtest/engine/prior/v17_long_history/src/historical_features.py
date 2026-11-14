"""Historical membership/sector-aware compact features; no portfolio fitting.

Uses the sealed feature primitives without altering their files or globals.
Terminal-label gaps remain missing and are measured, never imputed as returns.
"""
import sys,os
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json
import numpy as np,pandas as pd
from historical_sources import ROOT,DATA,membership_names,label_quotes
sys.path.insert(0,str(ROOT.parent/'v16_payoff_alignment/src'))
from stock_features import _prepare,_technicals,_pair,_labels,_rank,_label_wealth,OWN,SECTOR_OWN,SECTORS,IDIO_FLOOR,VOLUME_OWN,MICRO_OWN

def dated_masks(calendar,names):
    broad=membership_names(pd.read_csv(DATA/'membership_candidate_snapshots.csv.gz'))
    sectors=membership_names(pd.read_csv(DATA/'sector_candidate_snapshots.csv.gz'))
    lookup={k:i for i,k in enumerate(names)};shape=(len(calendar),len(names))
    member=np.zeros(shape,bool);sector=np.zeros(shape,np.int8);count=np.zeros(shape,np.int8)
    def spans(frame,start=None):
        groups=list(frame.groupby('date',sort=True))
        for i,(d,g) in enumerate(groups):
            left=max(pd.Timestamp(d),pd.Timestamp(start)) if start else pd.Timestamp(d)
            right=pd.Timestamp(groups[i+1][0]) if i+1<len(groups) else pd.Timestamp('2024-01-01')
            a=calendar.searchsorted(left);b=calendar.searchsorted(right)
            ids=[lookup[r] for r in set(g.ric) if r in lookup]
            if b>a and ids:yield a,b,ids
    for a,b,ids in spans(broad):member[a:b,ids]=True
    for k,s in enumerate(SECTORS,1):
        start='2016-09-19' if s=='XLRE' else '2001-12-03'
        for a,b,ids in spans(sectors[sectors.sector.eq(s)],start):
            sector[a:b,ids]=k;count[a:b,ids]+=1
    sector[count!=1]=0
    return member,sector,count

def main():
    directory=DATA/'assembled';status=json.loads((directory/'status.json').read_text())
    assert not status['source_conflicts'],'Conflicting fields must be resolved first'
    prices=pd.read_pickle(directory/'prices_candidate.pkl')
    calendar=pd.DatetimeIndex(pd.read_csv(directory/'calendar.csv',parse_dates=['Date']).Date)
    assert calendar.max()<pd.Timestamp('2024-01-01')
    membership=membership_names(pd.read_csv(DATA/'membership_candidate_snapshots.csv.gz'))
    names=sorted(set(membership.ric));member,sectors,sector_counts=dated_masks(calendar,names)
    benchmark_names=set(SECTORS)|{'IYZ','SPY'}
    absent=[k for k in benchmark_names if k not in prices or not prices[k].q.eq(1).any()]
    if absent:
        raise ValueError('Required parent/benchmark source absent; do not silently exclude its stocks: '+str(sorted(absent)))
    etfs={k:_prepare(prices.get(k),calendar) for k in benchmark_names}
    label_etfs={k:label_quotes(prices[k],calendar) for k in benchmark_names if k in prices}
    technical={k:_technicals(v) for k,v in etfs.items()}
    eventfile=directory/'events.csv'
    events=pd.read_csv(eventfile,parse_dates=['effective_date']) if eventfile.exists() else pd.DataFrame(columns=['ticker','effective_date'])
    frames=[];coverage=[];cash_records=[];empty=pd.DataFrame(index=calendar)
    for index,name in enumerate(names):
        included=member[:,index]&(calendar>=pd.Timestamp('2002-01-01'))
        if not included.any():continue
        stock=_prepare(prices.get(name),calendar);raw=_technicals(stock)
        stock_labels=label_quotes(prices[name],calendar) if name in prices else stock
        event=events[events.ticker.eq(name)]
        if len(event):
            assert len(event)==1
            row=event.iloc[0]
            conditional=(pd.notna(row.get('cvr_upper')) and row.get('cvr_upper',0)>0) or (pd.notna(row.get('unpriced_rights')) and bool(row.get('unpriced_rights',False)))
            conditional |= pd.notna(row.get('label_wealth_qualified')) and not bool(row.get('label_wealth_qualified'))
            if conditional:
                cash_records.append(dict(ticker=name,effective_date=str(row.effective_date),applied=False,
                    reason='Unpriced contingent rights: cash-only payout is not a verified complete shareholder-wealth training label'))
            else:
                stock_labels,detail=_label_wealth(stock_labels,row);cash_records.append(detail)
        beta,_,_= _pair(stock,etfs['SPY']);raw['beta_spy63']=beta
        out=pd.DataFrame(index=calendar);out['ticker']=name
        out['sector']=[SECTORS[k-1] if k else None for k in sectors[:,index]]
        out['sector_taxonomy']=out.sector.where(~(out.sector.eq('XLC')&(out.index<pd.Timestamp('2018-09-24'))),'Telecommunication Services')
        out['eligible']=stock.price_available&stock.consecutive_bars.ge(127)&member[:,index]
        out['beta_spy63']=beta;out['sector_available']=0.;out['idio_vol21']=np.nan
        out['execution_date']=pd.Series(calendar,index=calendar).shift(-1)
        out['label_end_h5']=pd.Series(calendar,index=calendar).shift(-6)
        for c in ['future_return_h5','future_parent_return_h5','target_risk_resid_h5','future_spy_h5']:out[c]=np.nan
        for s in SECTORS:
            current=out.sector.eq(s)
            if not current.any():continue
            routes=[('IYZ',current&(out.index<pd.Timestamp('2018-09-24'))),('XLC',current&(out.index>=pd.Timestamp('2018-09-24')))] if s=='XLC' else [(s,current)]
            for route,mask in routes:
                if not mask.any():continue
                parent=etfs[route];bs,corr,idio=_pair(stock,parent)
                sigma=idio.clip(lower=IDIO_FLOOR)
                # The label scale is an ex-ante input. A recovered63-session
                # beta can coexist with21 missing residual-vol observations
                # after a parent quote gap; do not call that context available.
                # This uses only decision-date information, never future labels.
                availability=np.isfinite(bs)&np.isfinite(corr)&np.isfinite(sigma)&sigma.gt(0)&np.isfinite(technical[route].return63)&np.isfinite(raw.return21)&np.isfinite(technical[route].vol21)
                labels=_labels(stock_labels,label_etfs.get(route,parent),bs,sigma,raw.atr14_pct,(5,),'close')
                out.loc[mask,'sector_available']=availability.loc[mask].astype(float)
                out.loc[mask,'idio_vol21']=sigma.loc[mask]
                for col in ['future_return_h5','future_parent_return_h5','target_risk_resid_h5']:out.loc[mask,col]=labels.loc[mask,col]
        sp=label_etfs['SPY'];out['future_spy_h5']=sp.adj_close.shift(-6)/sp.adj_close.shift(-1)-1
        out['response_spy']=(out.future_return_h5-beta*out.future_spy_h5)/(out.idio_vol21*np.sqrt(5))
        out['v8_target_absolute_h5']=out.future_return_h5
        out['v8_target_spy_h5']=out.future_return_h5-out.future_spy_h5
        out['sector_unique']=(sector_counts[:,index]==1)
        out['model_eligible']=out.eligible&out.sector_available.eq(1)&out.sector_unique
        for flag in ['volume_available','micro_available','momentum252_available']:out[flag]=raw[flag].fillna(0.)
        for c in OWN:out['raw_'+c]=raw[c]
        out.index.name='decision_date';frames.append(out.loc[included].reset_index())
        coverage.append(dict(ticker=name,member_dates=int(included.sum()),known_model_rows=int(out.loc[included,'model_eligible'].sum()),missing_h5_labels=int((out.model_eligible&out.target_risk_resid_h5.isna()&included).sum())))
        if index%100==0:print('HISTORICAL_FEATURES',index,len(names),flush=True)
    panel=pd.concat(frames,ignore_index=True).sort_values(['decision_date','ticker']).reset_index(drop=True)
    residual=panel.future_return_h5-panel.beta_spy63*panel.future_spy_h5
    residual_mean=residual.where(panel.model_eligible).groupby(panel.decision_date).transform('mean')
    denominator=panel.idio_vol21*np.sqrt(5)
    panel['response_spy']=(residual-residual_mean)/denominator
    panel['v8_target_absolute_h5']=panel.future_return_h5/denominator
    panel['v8_target_spy_h5']=panel.response_spy
    for feature in OWN:
        valid=panel.eligible.copy()
        if feature=='return252':valid &= panel.momentum252_available.eq(1)
        elif feature in VOLUME_OWN:valid &= panel.volume_available.eq(1)
        elif feature in MICRO_OWN:valid &= panel.micro_available.eq(1)
        panel['cs_'+feature]=_rank(panel,'raw_'+feature,valid,unknown=0.)
    for s in SECTORS:panel['sector_'+s]=panel.sector.eq(s).astype(float)
    cols=['cs_'+f for f in OWN]+['volume_available','momentum252_available','micro_available']+['sector_'+s for s in SECTORS]
    assert len(cols)==36 and not panel.duplicated(['decision_date','ticker']).any()
    panel[cols]=panel[cols].replace([np.inf,-np.inf],np.nan).fillna(0.).astype(np.float32)
    panel=panel.drop(columns=['raw_'+f for f in OWN]);out=DATA/'candidate_features';out.mkdir(exist_ok=True)
    panel.to_pickle(out/'panel.pkl');(out/'columns.json').write_text(json.dumps(cols,indent=2))
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    pd.DataFrame(cash_records).to_csv(out/'terminal_label_coverage.csv',index=False)
    rows=[]
    for year,g in panel.groupby(panel.decision_date.dt.year):
        known=g[g.model_eligible];complete=known.target_risk_resid_h5.notna().groupby(known.decision_date).all()
        rows.append(dict(year=int(year),dated_members=g.ticker.nunique(),rows=len(g),eligible_rows=len(known),known_dates=known.decision_date.nunique(),complete_label_dates=int(complete.sum()),median_eligible=float(known.groupby('decision_date').size().median()),complete_date_fraction=float(complete.mean())))
    pd.DataFrame(rows).to_csv(out/'query_completeness.csv',index=False)
    import hashlib
    (out/'status.json').write_text(json.dumps(dict(end_year=status['end_year'],rows=len(panel),features=36,training_start='2002-01-01',reserve='2024-01-01',state='CANDIDATE_FEATURES; terminal labels/account remain unqualified',no_label_based_prediction_eligibility=True,
        source_prices_sha256=hashlib.sha256((directory/'prices_candidate.pkl').read_bytes()).hexdigest(),
        source_events_sha256=hashlib.sha256(eventfile.read_bytes()).hexdigest()),indent=2))
    print('PANEL',len(panel),rows,flush=True)

if __name__=='__main__':main()
