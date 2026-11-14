"""Enumerate every returned distribution/capital event before approving wealth."""
from common import *
import pandas as pd,numpy as np
sys.path.insert(0,str(SOURCE/'src'))
from historical_sources import canonical_ric

def main():
    raw=HERE/'data/raw';plan=read(HERE/'data/capture_plan.json');repairs=read(HERE/'data/xlre_repair_plan.json')['requests'];jobs=[j for j in plan['requests']+repairs if j['kind']=='distributions']
    assert len(jobs)==21;parts=[]
    for j in jobs:
        p=raw/(j['name']+'.pkl');assert p.exists() and sha(p)==read(p.with_suffix('.json'))['response_sha256'];z=pd.read_pickle(p);assert len(z.columns)==5
        z.columns=['ticker','ex_date','adjusted_gross','unadjusted_gross','type'];z['source_request']=j['name'];parts.append(z)
    div=pd.concat(parts,ignore_index=True);div.ticker=div.ticker.map(canonical_ric);assert set(div.ticker)=={canonical_ric(n) for n in plan['issues']};div.ex_date=pd.to_datetime(div.ex_date);div=div[div.ex_date.between(START,END)].copy();div.to_csv(HERE/'data/distribution_census.csv',index=False)
    terms=pd.read_csv(HERE/'data/noncash_terms.csv',parse_dates=['ex_date']);delayed=pd.DataFrame(read(HERE/'data/delayed_distribution_terms.json'));delayed.ex_date=pd.to_datetime(delayed.ex_date);terms=pd.concat([terms,delayed.rename(columns={'source':'source_url'})],ignore_index=True)
    cash_types={'Interim','Final','Special','Extra','Capital gains','Capital gains, long term','Capital gains, short term','Approximate interim','Approximate final','Approximate, miscellaneous','Liquidation'}
    allowed=cash_types|{'Stock dividend, different stock','Stock dividend, same stock','Rescinded payment'};assert set(div.type.dropna())<=allowed,sorted(set(div.type.dropna())-allowed)
    dispositions=[];unresolved=[];cash=[]
    for row in div[div.type.eq('Stock dividend, different stock')].itertuples():
        match=terms[terms.ticker.eq(row.ticker)&terms.ex_date.eq(row.ex_date)]
        if len(match)!=1:unresolved.append(dict(ticker=row.ticker,date=str(row.ex_date.date()),kind='dividend_census_noncash_unmatched'))
        dispositions.append(dict(ticker=row.ticker,date=str(row.ex_date.date()),kind=row.type,status='verified_physical_or_claim_terms' if len(match)==1 else 'unresolved',source_request=row.source_request))
    for row in terms.itertuples():
        same=div[div.ticker.eq(row.ticker)&div.ex_date.eq(row.ex_date)&div.type.isin(cash_types)];assert same.unadjusted_gross.notna().all(),same.to_dict('records')
        cash.append(dict(ticker=row.ticker,ex_date=str(row.ex_date.date()),cash_per_old_share=float(same.unadjusted_gross.sum()),cash_records=len(same),source_requests='|'.join(sorted(set(same.source_request)))))
    pd.DataFrame(cash).to_csv(HERE/'data/noncash_same_date_cash.csv',index=False)
    capital=[]
    for j in read(HERE/'data/supplement_plan.json')['requests']+repairs:
        if not (j['name'].startswith('capital_') or j['name']=='XLRE_explicit_capital'):continue
        p=raw/(j['name']+'.pkl');assert sha(p)==read(p.with_suffix('.json'))['response_sha256'];z=pd.read_pickle(p);z['source_request']=j['name'];capital.append(z)
    cap=pd.concat(capital,ignore_index=True);cap.Instrument=cap.Instrument.map(canonical_ric);cap.to_csv(HERE/'data/capital_census.csv',index=False)
    for _,row in cap.iterrows():
        kind=row['Corporate Change Event Type'];ticker=row.Instrument
        if pd.isna(kind) or not kind:continue
        if pd.notna(row['Capital Change Is Rescinded']) and bool(row['Capital Change Is Rescinded']):state='rescinded_no_entitlement'
        elif kind in ('Buyback','Exchange Offer'):state='no_automatic_entitlement; frozen strategy does not tender'
        elif kind in ('Share Split','Share Consolidation'):state='same_issue_share_units_in_adjusted_OHLC_and_TR; no_separate_child_cashflow'
        elif kind=='Demerger':
            old=row['Terms Old Shares'];new=row['Terms New Shares'];date=row['Capital Change Effective Date']
            if pd.isna(old) or pd.isna(new) or pd.isna(date) or date>pd.Timestamp(END):state='proposal_or_outside_sample; no_completed_ex_date_terms'
            else:
                match=terms[terms.ticker.eq(ticker)&(terms.ex_date-pd.Timestamp(date)).abs().le(pd.Timedelta(days=45))]
                if len(match)!=1:unresolved.append(dict(ticker=ticker,date=str(pd.Timestamp(date).date()),kind='capital_demerger_unmatched'));state='unresolved'
                else:state='verified_issuer_exchange_terms_override_vendor_record_or_payment_date'
        elif kind=='Rights Issue' and ticker=='LUV.N':state='attached_poison_pill_untriggered_expired_2024-10-25; https://www.southwestairlinesinvestorrelations.com/sec-filings/all-sec-filings/content/0001193125-24-243473/0001193125-24-243473.pdf'
        else:unresolved.append(dict(ticker=ticker,kind=str(kind)));state='unresolved'
        dispositions.append(dict(ticker=ticker,date=str(row['Capital Change Effective Date']),kind=kind,status=state,source_request=row.source_request))
    pd.DataFrame(dispositions).to_csv(HERE/'data/census_dispositions.csv',index=False)
    result=dict(passed=not unresolved,unresolved=unresolved,distribution_requests=len(jobs),distribution_rows=len(div),distribution_type_counts=div.type.value_counts().to_dict(),capital_requests=len(capital),capital_rows=len(cap),physical_distribution_terms=len(terms)-len(delayed),separate_delayed_claims=len(delayed),all_575_acquisition_issues_censused=True,same_date_cash_sha256=sha(HERE/'data/noncash_same_date_cash.csv'),terms_sha256=sha(HERE/'data/noncash_terms.csv'),delayed_terms_sha256=sha(HERE/'data/delayed_distribution_terms.json'),classification='Capital records include announcement/record/payment dates and proposed events; official exchange ex-dates control. Mandatory distributions reconciled; no automatic participation in voluntary exchange offers.')
    write(HERE/'data/census_qualification.json',result);print('CENSUS',json.dumps(result,default=str),flush=True);assert not unresolved,unresolved
if __name__=='__main__':main()
