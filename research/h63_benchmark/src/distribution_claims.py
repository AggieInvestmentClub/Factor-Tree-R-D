"""Explicit delayed distribution receivables under a provisional zero-mark case.

The legal entitlement follows the ex-date position even after the parent is
sold. Future settlement prices never enter interim NAV, risk or decisions.
Zero is a scenario, not a certified market value (especially for short claims).
"""
import numpy as np
import pandas as pd

class DistributionClaims:
    def __init__(self,records,names,start,end):
        self.names=list(names);self.open=[];self.log=[];self.records=[]
        for raw in records or []:
            row=dict(raw);row['ex_date']=pd.Timestamp(row['ex_date']);row['settlement_date']=pd.Timestamp(row['settlement_date'])
            assert row['ex_date']<row['settlement_date']<=pd.Timestamp(end)
            assert np.isfinite(row['raw_parent_previous']) and row['raw_parent_previous']>0
            assert np.isfinite(row['raw_settlement_per_parent']) and row['raw_settlement_per_parent']>=0
            assert row['valuation_policy']=='zero_until_observed_settlement; conditional_only'
            if row['ticker'] in names and row['ex_date']>=pd.Timestamp(start):self.records.append(row)

    def on_day(self,date,prior_shares,previous_close,fee):
        n=len(self.names);cash=np.zeros(n);long=np.zeros(n);short=np.zeros(n);lc=np.zeros(n);sc=np.zeros(n);notional=np.zeros(n);issues=[]
        for row in self.records:
            if date!=row['ex_date']:continue
            j=self.names.index(row['ticker'])
            if prior_shares[j]==0:continue
            assert np.isfinite(previous_close[j]) and previous_close[j]>0
            # Gauge shares times yesterday's gauge/raw ratio = actual shares.
            quantity=float(prior_shares[j]*previous_close[j]/row['raw_parent_previous'])
            self.open.append(dict(row,quantity=quantity))
            self.log.append(dict(ticker=row['ticker'],date=str(date.date()),kind='entitlement_created',raw_parent_shares=quantity))
        remaining=[]
        for row in self.open:
            j=self.names.index(row['ticker']);quantity=row['quantity']
            if date<row['settlement_date']:
                issues.append(dict(date=str(date.date()),ticker=row['ticker'],kind='unresolved_distribution_claim_mark',
                    raw_share_equivalent=quantity,reason='Entitled distributed shares not yet tradable; explicit zero-mark scenario, not observed NAV'))
                remaining.append(row);continue
            assert date==row['settlement_date'],'Missed observed settlement session'
            paid=quantity*row['raw_settlement_per_parent'];cash[j]+=paid;notional[j]+=abs(paid)
            if quantity>0:long[j]+=paid;lc[j]+=fee*abs(paid)
            else:short[j]+=paid;sc[j]+=fee*abs(paid)
            self.log.append(dict(ticker=row['ticker'],date=str(date.date()),kind='observed_receiver_liquidation',
                raw_parent_shares=quantity,cashflow=paid,cost=fee*abs(paid),source=row['source']))
        self.open=remaining
        return dict(cash=cash,long_pnl=long,short_pnl=short,long_costs=lc,short_costs=sc,notional=notional,issues=issues)
