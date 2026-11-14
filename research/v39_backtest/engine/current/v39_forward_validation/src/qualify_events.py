"""Reconcile forward corporate wealth and costs using verified issuer/exchange terms."""
from common import *
import numpy as np,pandas as pd,gc
sys.path.insert(0,str(SOURCE/'src'))
from historical_sources import convert

def main():
    folder=HERE/'data/assembled';prices=pd.read_pickle(folder/'prices_candidate.pkl');fields=pd.read_pickle(folder/'aligned_fields.pkl');calendar=pd.DatetimeIndex(pd.read_csv(folder/'calendar.csv',parse_dates=['Date']).Date);terms=pd.read_csv(HERE/'data/terminal_terms.csv',parse_dates=['market_effective_date']);events=[];termchecks=[]
    for r in terms.itertuples():
        date=r.market_effective_date;previous=calendar[calendar.get_loc(date)-1];old=prices[r.ticker];last=old.index[old.q.eq(1)].max();assert last==previous,(r.ticker,'last quote',last,previous)
        assert old.loc[previous,'raw_cash_conversion_qualified'] and np.isfinite(old.loc[previous,'raw_close'])
        stock=0.;opening=np.nan
        if r.exchange_ratio:
            assert r.receiver_ric in prices and prices[r.receiver_ric].loc[date,'q']==1
            opening=fields.loc[(r.receiver_ric,date),'RawOpen'];assert np.isfinite(opening) and opening>0;stock=r.exchange_ratio*opening
        cash=r.cash_component;events.append(dict(ticker=r.ticker,effective_date=date,raw_cash_price=cash+stock,event_type='stock_merger_liquidation' if stock else 'cash_merger',event_phase='open',stock_component=stock,cash_component=cash,receiver_ric=r.receiver_ric,exchange_ratio=r.exchange_ratio,receiver_raw_open=opening,source_qualified=True,source_url=r.source_url,official_source_url=r.clock_source_url,cvr_upper=r.cvr_upper,unpriced_rights=bool(r.cvr_upper>0),label_wealth_qualified=not bool(r.cvr_upper>0),conditional_reason='Unpriced non-transferable contingent right; cash-only NAV scenario; signed0-to-cap sensitivity required' if r.cvr_upper>0 else None,last_raw_quote_date=previous,last_raw_close=old.loc[previous,'raw_close']))
        termchecks.append(dict(ticker=r.ticker,effective_date=str(date.date()),previous_qualified_raw_quote=str(previous.date()),last_quote_exact=True,receiver_open_dated_exact=bool(stock),source_url=r.source_url,clock_source_url=r.clock_source_url,conditional_right_upper=r.cvr_upper))
    prior=pd.read_csv(SOURCE/'data/assembled/events.csv',parse_dates=['effective_date']);combined=pd.concat([prior,pd.DataFrame(events)],ignore_index=True);assert not combined.ticker.duplicated().any();combined.to_csv(folder/'events.csv',index=False)
    write(folder/'terminal_qualification.json',dict(passed=True,events=termchecks,scope='Same immediate successor regular-open liquidation and fractional synthetic share convention as prior benchmark; no actual fractional cash or broker settlement lag claim'))
    # Forward spin-off terms may not be inferred from the vendor return series.
    # Ordinary same-date cash, if any, must be separately reconciled in this file.
    terms=pd.read_csv(HERE/'data/noncash_terms.csv',parse_dates=['ex_date']);cashfile=HERE/'data/noncash_same_date_cash.csv';cashrows=pd.read_csv(cashfile,parse_dates=['ex_date']) if cashfile.exists() else pd.DataFrame(columns=['ticker','ex_date','cash_per_old_share']);changes=[]
    for r in terms.itertuples():
        date=r.ex_date;previous=calendar[calendar.get_loc(date)-1];f=fields.xs(r.ticker);oldclose=f.loc[previous,'RawClose'];parent=f.loc[date,'RawClose'];child=fields.loc[(r.child,date),'RawClose'];assert np.isfinite([oldclose,parent,child]).all() and min(oldclose,parent,child)>0
        ordinary=cashrows[cashrows.ticker.eq(r.ticker)&cashrows.ex_date.eq(date)];assert len(ordinary)<=1;cash=float(ordinary.cash_per_old_share.iloc[0]) if len(ordinary) else 0.
        distributed=r.child_ratio*child;wealth=r.parent_ratio*parent+distributed+cash;value=100*(wealth/oldclose-1);oldreturn=fields.loc[(r.ticker,date),'TotalReturn1D'];fields.loc[(r.ticker,date),'TotalReturn1D']=value;fields.loc[(r.ticker,date),'DistributionReinvestmentTurnover']=2*abs(distributed)/wealth
        changes.append(dict(ticker=r.ticker,date=str(date.date()),child=r.child,child_ratio=r.child_ratio,parent_ratio=r.parent_ratio,previous_raw_close=oldclose,parent_raw_close=parent,child_raw_close=child,ordinary_cash=cash,distributed_value=distributed,old_provider_return_pct=oldreturn,physical_return_pct=value,change_bps=100*(value-oldreturn),reinvestment_turnover=2*abs(distributed)/wealth,source=r.source_url))
    claims=read(SOURCE/'data/distribution_audit/delayed_claims.json');delayed=read(HERE/'data/delayed_distribution_terms.json')
    for r in delayed:
        date=pd.Timestamp(r['ex_date']);settle=pd.Timestamp(r['settlement_date']);previous=calendar[calendar.get_loc(date)-1];ticker=r['ticker'];oldclose=fields.loc[(ticker,previous),'RawClose'];parent=fields.loc[(ticker,date),'RawClose'];child=fields.loc[(r['child'],settle),'RawOpen'];assert np.isfinite([oldclose,parent,child]).all() and min(oldclose,parent,child)>0
        ordinary=cashrows[cashrows.ticker.eq(ticker)&cashrows.ex_date.eq(date)];assert len(ordinary)<=1;cash=float(ordinary.cash_per_old_share.iloc[0]) if len(ordinary) else 0.
        oldreturn=fields.loc[(ticker,date),'TotalReturn1D'];value=100*((parent+cash)/oldclose-1);fields.loc[(ticker,date),'TotalReturn1D']=value
        if 'UnpricedDistributionClaim' not in fields:fields['UnpricedDistributionClaim']=0
        ix=(fields.index.get_level_values('ric')==ticker)&(fields.index.get_level_values('date')>=date)&(fields.index.get_level_values('date')<=settle);fields.loc[ix,'UnpricedDistributionClaim']=1
        claim=dict(r,raw_parent_previous=float(oldclose),raw_parent_ex_close=float(parent),raw_settlement_per_parent=float(r['child_ratio']*child),receivers=[dict(ric=r['child'],ratio=r['child_ratio'],raw_settlement_open=float(child))]);claims.append(claim)
        changes.append(dict(ticker=ticker,date=r['ex_date'],child=r['child'],child_ratio=r['child_ratio'],parent_ratio=1,previous_raw_close=oldclose,parent_raw_close=parent,child_raw_close=None,ordinary_cash=cash,distributed_value=None,old_provider_return_pct=oldreturn,physical_return_pct=value,change_bps=100*(value-oldreturn),reinvestment_turnover=0,source=r['source'],state='delayed_claim_separate_from_parent',settlement_date=r['settlement_date']))
    # Regenerate only changed issues. The pre2024 prefix and boundary wealth unit
    # remain immutable, including all prior source corrections and claim handling.
    observed=calendar[calendar>='2023-01-01'];original=pd.read_pickle(SOURCE/'data/assembled/prices_candidate.pkl');cut=pd.Timestamp('2023-12-29')
    for name in sorted(set(terms.ticker)|{r['ticker'] for r in delayed}):
        f=fields.xs(name);new=convert(f,observed);new['raw_open']=f.RawOpen.reindex(observed);new['direct_adjusted_close']=f.DirectAdjustedClose.reindex(observed)
        if name in original:
            old=original[name]
            if old.loc[cut,'q']==1 and new.loc[cut,'q']==1:
                factor=old.loc[cut,'adj_close']/new.loc[cut,'adj_close'];new.loc[:,['adj_open','adj_high','adj_low','adj_close']]*=factor;new.tr_segment_id+=old.loc[cut,'tr_segment_id']-new.loc[cut,'tr_segment_id']
            else:new.tr_segment_id+=old.tr_segment_id.max()+10000
            prices[name]=pd.concat([old,new[new.index>cut]]).reindex(calendar)
        else:prices[name]=new.reindex(calendar)
    pd.to_pickle(prices,folder/'prices_candidate.pkl');fields.to_pickle(folder/'reconciled_fields.pkl');pd.DataFrame(changes).to_csv(folder/'noncash_wealth_reconciliation.csv',index=False);pd.DataFrame([dict(ticker=r['ticker'],date=r['date'],child=r['child'],state=r.get('state','verified_physical_basket_reinvested'),source=r['source']) for r in changes]).to_csv(folder/'noncash_census.csv',index=False)
    write(folder/'noncash_qualification.json',dict(state='Physical terms and separately dated raw closes reconciled; census coverage still required',events=len(changes),physical_cash_and_shares_not_double_counted=True,ordinary_dividend_same_date_audit_required=True,price_sha256=sha(folder/'prices_candidate.pkl'),records=changes))
    write(folder/'claims.json',claims)
    print('EVENTS_RECONCILED',len(events),len(changes),'CENSUS_AND_QUOTES_REQUIRE_FINAL_QA',flush=True)
if __name__=='__main__':main()
