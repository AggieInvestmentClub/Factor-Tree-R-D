"""Accounting-only source comparisons; never reads portfolio predictions or returns."""
from common import *
import pandas as pd,numpy as np
from assemble import align

def main():
    align();files=list((HERE/'data/aligned').glob('*.pkl'));z=pd.concat([pd.read_pickle(f) for f in files]).sort_index();assert not z.index.duplicated().any();terms=pd.read_csv(HERE/'data/noncash_terms.csv',parse_dates=['ex_date']);rows=[]
    for r in terms.itertuples():
        if r.ticker not in z.index.get_level_values(0):continue
        parent=z.xs(r.ticker);date=r.ex_date
        if date not in parent.index or pd.isna(parent.loc[date,'RawClose']):continue
        previous=parent[parent.index<date].RawClose.dropna();child=z.xs(r.child).loc[date,'RawClose'] if r.child in z.index.get_level_values(0) and date in z.xs(r.child).index else np.nan
        if previous.empty:continue
        gross=r.parent_ratio*parent.loc[date,'RawClose']+r.child_ratio*child;physical=gross/previous.iloc[-1]-1;provider=parent.loc[date,'TotalReturn1D']/100
        rows.append(dict(ticker=r.ticker,ex_date=str(date.date()),child=r.child,parent_ratio=r.parent_ratio,child_ratio=r.child_ratio,prior_date=str(previous.index[-1].date()),previous_raw_close=previous.iloc[-1],parent_raw_close=parent.loc[date,'RawClose'],child_raw_close=child,physical_gross_return=physical,provider_gross_return=provider,difference_bps=(physical-provider)*10000,source_url=r.source_url,state='needs dividend census; physical basket observed' if np.isfinite(physical) else 'child quote pending'))
    pd.DataFrame(rows).to_csv(HERE/'data/noncash_reconciliation_candidate.csv',index=False)
    terminal=pd.read_csv(HERE/'data/terminal_terms.csv',parse_dates=['market_effective_date']);tr=[]
    for r in terminal.itertuples():
        if r.ticker not in z.index.get_level_values(0):continue
        f=z.xs(r.ticker);valid=f.RawClose.notna();last=f.index[valid].max()
        if pd.isna(last) or f.index.max()<r.market_effective_date:continue
        tr.append(dict(ticker=r.ticker,market_effective_date=str(r.market_effective_date.date()),last_raw_quote=str(last.date()),no_quotes_from_effective=bool(last<r.market_effective_date),cash=r.cash_component,receiver=r.receiver_ric,ratio=r.exchange_ratio,cvr_upper=r.cvr_upper))
    pd.DataFrame(tr).to_csv(HERE/'data/terminal_quote_checks.csv',index=False)
    print('NONCASH_SOURCE_CHECKS',pd.DataFrame(rows)[['ticker','ex_date','difference_bps','state']].to_string(index=False) if rows else 'pending',flush=True)
    print('TERMINAL_CHECKS',tr,flush=True)
if __name__=='__main__':main()
