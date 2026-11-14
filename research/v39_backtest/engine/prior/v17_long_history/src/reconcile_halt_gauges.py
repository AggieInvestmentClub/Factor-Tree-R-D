"""Link resumed return units across documented halts, without creating quotes."""
import numpy as np,pandas as pd

def reconcile(frame,source,halt_dates):
    f=frame.copy();clock=f.index;raw=source.reindex(clock);records=[]
    locations=clock.get_indexer(pd.DatetimeIndex(halt_dates));locations=locations[locations>=0]
    if not len(locations):return f,records
    for part in np.split(np.sort(locations),np.flatnonzero(np.diff(np.sort(locations))>1)+1):
        if not len(part):continue
        left,right=int(part[0])-1,int(part[-1])+1
        record=dict(halt_first=str(clock[part[0]].date()),halt_last=str(clock[part[-1]].date()),state='no_observed_adjacent_resumption')
        assert not f.iloc[part].q.eq(1).any(),'Official full-session halt conflicts with supplied executable quotes'
        if left<0 or right>=len(clock) or f.iloc[left].q!=1 or f.iloc[right].q!=1:
            records.append(record);continue
        previous,resume=clock[left],clock[right]
        actual=float(raw.loc[resume,'PriceClose']/raw.loc[previous,'PriceClose']-1)
        reported=float(raw.loc[resume,'TotalReturn1D']/100)
        if not np.isfinite([actual,reported]).all() or abs(actual-reported)>1e-7:
            record.update(state='unresolved_return_basis',raw_price_return=actual,reported_return=reported);records.append(record);continue
        # A split/distribution across the halt would fail this price-return
        # reconciliation; no current-gauge ratio is assumed without the check.
        segment=f.loc[resume,'tr_segment_id'];old=f.loc[previous,'tr_segment_id']
        subset=f.tr_segment_id.eq(segment)&f.q.eq(1)
        factor=float(f.loc[previous,'adj_close']*(1+reported)/f.loc[resume,'adj_close'])
        for column in ['adj_open','adj_high','adj_low','adj_close']:f.loc[subset,column]*=factor
        f.loc[subset,'tr_segment_id']=old
        record.update(state='observed_resumption_reconciled;halt_marks_still_provisional',resume=str(resume.date()),
            raw_price_return=actual,reported_return=reported,reconciliation_error=abs(actual-reported),gauge_factor=factor,
            halt_quotes_created=False)
        records.append(record)
    return f,records
