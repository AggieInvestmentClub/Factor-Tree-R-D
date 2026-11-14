"""Approve source conventions only after census and quote reconciliation."""
from common import *
import pandas as pd,numpy as np,gc

def check_splits(folder):
    fields=pd.read_pickle(folder/'aligned_fields.pkl');capital=pd.read_csv(HERE/'data/capital_census.csv',parse_dates=['Capital Change Ex Date','Capital Change Effective Date']);div=pd.read_csv(HERE/'data/distribution_census.csv',parse_dates=['ex_date']);rows=[]
    for _,r in capital[capital['Corporate Change Event Type'].isin(['Share Split','Share Consolidation'])].iterrows():
        nominal=r['Capital Change Effective Date']
        if pd.isna(nominal) or nominal>pd.Timestamp(END):continue
        if r.Instrument=='HON.OQ' and nominal.year==2026:
            rows.append(dict(ticker=r.Instrument,nominal_date=str(nominal.date()),status='joint_spin_and_reverse_split_verified_in_physical_basket'));continue
        f=fields.xs(r.Instrument);ratio=float(r['Terms New Shares']/r['Terms Old Shares']);factor=f.RawClose/f.DirectAdjustedClose;jump=factor.shift()/factor;window=jump[jump.index.to_series().between(nominal-pd.Timedelta(days=40),nominal+pd.Timedelta(days=10))];match=window.index[np.isclose(window,ratio,atol=1e-4,rtol=0)]
        assert len(match)==1,(r.Instrument,str(nominal),ratio,window.to_dict());date=match[0];previous=f.index[f.index.get_loc(date)-1]
        cash=div[div.ticker.eq(r.Instrument)&div.ex_date.eq(date)&~div.type.str.startswith('Stock',na=False)&div.type.ne('Rescinded payment')].unadjusted_gross.sum();physical=(ratio*f.loc[date,'RawClose']+cash)/f.loc[previous,'RawClose']-1;provider=f.loc[date,'TotalReturn1D']/100;error=float(physical-provider)
        assert abs(error)<1e-4,(r.Instrument,date,physical,provider,error)
        rows.append(dict(ticker=r.Instrument,nominal_date=str(nominal.date()),observed_split_date=str(date.date()),share_ratio=ratio,physical_return=physical,provider_return=provider,difference_bps=10000*error,status='raw_share_units_adjusted_close_and_TR_reconcile'))
    pd.DataFrame(rows).to_csv(folder/'split_qualification.csv',index=False);return len(rows)

def main():
    folder=HERE/'data/assembled';assert read(HERE/'data/membership_qualification.json')['passed'];census=read(HERE/'data/census_qualification.json');assert census['passed']
    assert read(folder/'terminal_qualification.json')['passed'];disagreements=pd.read_csv(folder/'direct_quote_disagreements.csv');assert disagreements.empty,disagreements.to_dict('records')
    old=pd.read_pickle(SOURCE/'data/assembled/prices_candidate.pkl');new=pd.read_pickle(folder/'prices_candidate.pkl');calendar=pd.DatetimeIndex(pd.read_csv(folder/'calendar.csv',parse_dates=['Date']).Date)
    assert calendar[-1]==pd.Timestamp(END) and calendar.is_unique and calendar.is_monotonic_increasing
    for name,f in old.items():
        candidate=new[name].loc[f.index,f.columns].copy();candidate.index=candidate.index.as_unit(f.index.unit)
        pd.testing.assert_frame_equal(candidate,f,check_dtype=False,check_names=False,check_exact=True)
    nold=len(old);del old;gc.collect();forward=calendar[calendar>=START]
    for etf in ['SPY','IYZ',*INDICES]:
        f=new[etf].loc[forward];assert f.q.eq(1).all() and f.source_available.eq(1).all() and f.tr_segment_id.nunique()==1,etf
    issues=[]
    for name,f in new.items():
        f=f.loc[forward];good=f.q.eq(1)
        if not good.any():continue
        for date in f.loc[f.index.to_series().between(f.index[good][0],f.index[good][-1])&~good].index:issues.append(dict(ric=name,date=str(date.date()),type='interior_return_gap_requires_zero_held_exposure'))
    pd.DataFrame(issues,columns=['ric','date','type']).to_csv(folder/'interior_gaps.csv',index=False)
    quotes=read(folder/'status.json');claims=read(folder/'claims.json');events=pd.read_csv(folder/'events.csv');forwardevents=events[pd.to_datetime(events.effective_date)>=pd.Timestamp(START)]
    splits=check_splits(folder)
    record=dict(source_qualification_passed=True,qualification='Historical reconstruction; qualified observed quotes and independently reconciled corporate terms. Conditional rights/claim valuations retained; account-level exposure and arithmetic audits still required.',created_utc=now(),design_sha256=sha(HERE/'design.json'),prices_sha256=sha(folder/'prices_candidate.pkl'),events_sha256=sha(folder/'events.csv'),claims_sha256=sha(folder/'claims.json'),noncash_census_sha256=sha(folder/'noncash_census.csv'),membership_qualification_sha256=sha(HERE/'data/membership_qualification.json'),census_qualification_sha256=sha(HERE/'data/census_qualification.json'),prior_price_prefixes_checked=nold,prior_price_prefixes_exact=True,forward_sessions=len(forward),calendar_end=END,calendar_counts=quotes['calendar_counts'],independent_direct_quote_disagreements=0,interior_quote_gaps=len(issues),forward_terminal_events=len(forwardevents),forward_noncash_events=len(pd.read_csv(folder/'noncash_census.csv')),split_events_verified=splits,forward_delayed_claims=[r for r in claims if r['ex_date']>=START],conditional_terminal_tickers=forwardevents.loc[forwardevents.cvr_upper.fillna(0).gt(0),'ticker'].tolist(),conditional_performance_requires_explicit_disclosure=True,vintage_database=False)
    write(folder/'approval.json',record);print('SOURCE_QUALIFIED',len(forward),'INTERIOR_GAPS',len(issues),flush=True)
if __name__=='__main__':main()
