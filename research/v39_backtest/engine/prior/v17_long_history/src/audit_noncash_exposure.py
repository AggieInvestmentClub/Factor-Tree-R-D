"""Check every unreviewed event against actual holdings, never assume no exposure.

Events outside eligible feature context remain harmless only when the account
also holds no affected parent around their potential entitlement dates.
"""
import json
import numpy as np,pandas as pd

class NoncashExposureAudit:
    def __init__(self,census):
        self.rows=[];self.events=[]
        for row in census.to_dict('records'):
            if row['state']!='no_eligible_feature_context; account_exposure_check_required':continue
            a=pd.Timestamp(row['source_ex_date']);b=pd.Timestamp(row['effective_date'])
            # Covers due-bill/provider date differences without deciding which
            # date is right from profitable outcomes. False positives need review.
            self.events.append(dict(ticker=row['parent'],source_date=str(a.date()),
                start=min(a,b)-pd.Timedelta(days=14),end=max(a,b)+pd.Timedelta(days=14)))

    def consume(self,positions):
        if positions.empty:return
        minimum=positions.Date.min();maximum=positions.Date.max()
        for event in self.events:
            if event['end']<minimum or event['start']>maximum:continue
            g=positions[positions.ticker.eq(event['ticker'])&positions.Date.between(event['start'],event['end'])]
            g=g[g.previous_closing_shares.abs().gt(1e-14)]
            for row in g.itertuples():
                self.rows.append(dict(date=str(pd.Timestamp(row.Date).date()),ticker=event['ticker'],
                    kind='unreviewed_noncash_event_exposure',source_event_date=event['source_date'],
                    gauge_shares=float(row.previous_closing_shares)))

    def report(self):
        # This audit never subtracts problem dates from performance.
        return dict(events_checked=len(self.events),passed=not self.rows,issues=self.rows)
