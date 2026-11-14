"""Reconcile explicit cash terms with direct final quotes and observed sessions.

Only fixed USD cash/common-share consideration with no quote gap can receive
a source-level settlement candidate. Account approval still requires a
qualified pre-event return basis and a reconciled whole historical panel.
"""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from audit_capture_years import align
from historical_sources import canonical_ric
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'
QUOTE_ALIASES={'AYE^B11':'AYE.N^B11'} # Primary issue in identity audit, US0173611064.

def quote_ric(ric):return QUOTE_ALIASES.get(ric,canonical_ric(ric))

def full_calendar():
    direct=DATA/'direct_spy_calendar.csv'
    if direct.exists():
        clock=pd.DatetimeIndex(pd.read_csv(direct,parse_dates=['Date']).Date)
        assert clock.is_unique and clock.is_monotonic_increasing and clock.max()<pd.Timestamp('2024-01-01')
        return clock
    parts=[]
    for path in (DATA/'audit_requests').glob('*benchmark*.json'):
        meta=json.loads(path.read_text())
        if meta.get('state')!='captured_not_audited':continue
        raw=pd.read_pickle(path.with_suffix('.pkl'));raw=raw[raw.Instrument.eq('SPY')]
        for year in range(2000,2024):
            z,_=align(raw,year)
            if len(z):
                good=z.PriceClose.notna()&z.Volume.gt(0)&z.TotalReturn1D.notna()
                parts.extend(z.index.get_level_values('date')[good])
    clock=pd.DatetimeIndex(sorted(set(parts)));assert clock.max()<pd.Timestamp('2024-01-01')
    return clock

def direct_quotes():
    rows=[];coverage=[];currencies={}
    for path in (DATA/'audit_requests').glob('05_terminal_quotes_*.json'):
        meta=json.loads(path.read_text())
        if meta.get('state')!='captured_not_audited':continue
        source=path.with_suffix('.pkl');assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['sha256']
        raw=pd.read_pickle(source)
        if 'TR.PRICECLOSE.CURRENCY' in raw:
            for ric,g in raw.groupby('Instrument'):
                # A static currency is often populated only on the first row;
                # empty continuation cells are missing, not a second currency.
                codes=g['TR.PRICECLOSE.CURRENCY'].dropna().astype(str).str.strip()
                currencies.setdefault(quote_ric(ric),set()).update(codes[codes.ne('')].unique())
        for tag in ['CLOSE','OPEN']:
            f=pd.DataFrame(dict(ric=raw.Instrument.map(quote_ric),date=pd.to_datetime(raw[f'TR.{tag}PRICE.DATE'],errors='coerce'),
                value=pd.to_numeric(raw[f'TR.{tag}PRICE(ADJUSTED=0)'],errors='coerce'),field=tag.lower()))
            rows.append(f.dropna())
        for ric in meta['request']['universe']:coverage.append((quote_ric(ric),pd.Timestamp(meta['request']['parameters']['EDate'])))
    if not rows:return pd.DataFrame(),{},{}
    f=pd.concat(rows,ignore_index=True);assert f.date.lt('2024-01-01').all()
    f=f[f.value.gt(0)];g=f.groupby(['ric','date','field']).value;bad=g.nunique().gt(1)
    z=g.first().mask(bad).unstack('field');ends={}
    for ric,date in coverage:ends[ric]=max(ends.get(ric,date),date)
    return z,ends,currencies

def main():
    candidates=pd.read_csv(DATA/'terminal_audit/matched_consideration_candidates.csv',dtype={'deal_number':str});clock=full_calendar();quotes,ends,currencies=direct_quotes()
    if quotes.empty:return
    override_file=DATA/'verified_terminal_overrides.json';overrides={}
    if override_file.exists():
        for row in json.loads(override_file.read_text()):
            key=(row['ric'],str(row['deal_number']));assert key not in overrides
            assert pd.Timestamp(row['official_effective_date'])<pd.Timestamp('2024-01-01')
            overrides[key]=row
    # An independently documented common-share event does not require the
    # vendor M&A search to find the historical ticker. Preserve a distinct
    # official: identifier rather than inventing an SDC deal number.
    known=set(zip(candidates.ric,candidates.deal_number));extra=[]
    for (ric,number),row in overrides.items():
        if (ric,number) in known:continue
        assert number.startswith('official:'),'Missing provider row cannot silently become an official-only event'
        extra.append(dict(ric=ric,deal_number=number,terms_status='independently_verified_terms',
            provider_effective_date=row['official_effective_date'],source_request='official:'+row['source_url'],
            consideration_text=row['reason'],cash_per_common_share=row['cash_per_common_share'],
            receiver_shares_per_common_share=row['receiver_shares_per_common_share'],receiver_routes=json.dumps(row['receiver_routes'])))
    if extra:candidates=pd.concat([candidates,pd.DataFrame(extra)],ignore_index=True)
    # A later vendor search may also find an officially reconciled issuer.
    # Retain the explicitly reviewed record, not two competing settlements.
    reviewed_rics={key[0] for key in overrides}
    candidates=candidates[[r.ric not in reviewed_rics or (r.ric,r.deal_number) in overrides
                           for r in candidates.itertuples()]].copy()
    records=[];settlements=[]
    for row in candidates.itertuples():
        override=overrides.get((row.ric,row.deal_number))
        if override:
            row=row._replace(cash_per_common_share=override['cash_per_common_share'],
                receiver_shares_per_common_share=override['receiver_shares_per_common_share'],
                receiver_routes=json.dumps(override['receiver_routes']),provider_effective_date=override['official_effective_date'],terms_status='explicit_terms_candidate')
        ric=canonical_ric(row.ric);rec=dict(ric=ric,deal_number=row.deal_number,terms_status=row.terms_status,
            provider_effective_date=row.provider_effective_date,source_request=row.source_request,consideration_text=row.consideration_text,
            state='manual_review',approved_for_account=False)
        if override:rec.update(official_source_url=override['source_url'],official_override_reason=override['reason'])
        if row.terms_status!='explicit_terms_candidate':rec['reason']='Consideration not uniquely fixed cash/common shares'
        elif ric not in quotes.index.get_level_values('ric'):rec['reason']='No direct predecessor raw quotes'
        else:
            q=quotes.xs(ric).dropna(subset=['close']);last=q.index.max();effective=pd.Timestamp(row.provider_effective_date)
            if pd.isna(last):rec['reason']='No predecessor closing price'
            elif ends[ric]<(last+pd.Timedelta(days=7)):rec['reason']='Capture ends too close to final quote to establish natural termination'
            elif abs((last-effective).days)>7:rec['reason']='Deal date does not reconcile to predecessor quote termination'
            else:
                k=clock.searchsorted(last,side='right');e=clock.searchsorted(effective)
                if max(k,e)>=len(clock):rec['reason']='Settlement outside observed pre2024 calendar'
                elif k!=max(k,e):rec['reason']='Unpriced market session between last quote and effective consideration'
                else:
                    settlement=clock[k];rec.update(last_quote_date=str(last.date()),last_raw_close=float(q.loc[last,'close']),
                        settlement_date=str(settlement.date()),cash=float(row.cash_per_common_share),share_ratio=float(row.receiver_shares_per_common_share))
                    if row.receiver_shares_per_common_share>0:
                        routes=json.loads(row.receiver_routes)
                        if ric=='BMS.N^F19':routes=['AMCR.N'] # New holding company; issuer announcement, June11,2019.
                        if len(routes)!=1:rec['reason']='Receiver quote identity not uniquely mapped'
                        else:
                            receiver=canonical_ric(routes[0]);key=(receiver,settlement)
                            if currencies.get(receiver)!= {'USD'} or currencies.get(ric)!= {'USD'}:
                                rec['reason']='Raw predecessor/receiver USD quote denomination not uniquely verified'
                            elif '.TO' in receiver or '.L' in receiver or '.AX' in receiver:
                                rec['reason']='Foreign share/ADR ratio requires separate reconciliation'
                            elif key not in quotes.index or not np.isfinite(quotes.loc[key,'open']):
                                rec['reason']='No observed receiver opening quote on settlement session'
                            else:
                                opening=float(quotes.loc[key,'open']);stock=float(row.receiver_shares_per_common_share)*opening
                                payout=float(row.cash_per_common_share)+stock
                                if not .5<payout/q.loc[last,'close']<1.5:rec['reason']='Converted stock/cash value fails broad final-quote sanity range'
                                else:
                                    rec.update(state='reconciled_stock_cash_candidate',reason='Explicit fixed common-share ratio; exact provider receiver route; USD direct raw opening quote; no intervening market session gap',receiver=receiver,receiver_raw_open=opening,stock_component=stock,total_consideration=payout)
                                    settlements.append(dict(ticker=ric,effective_date=str(settlement.date()),raw_cash_price=payout,
                                        event_type='stock_merger_liquidation',event_phase='open',stock_component=stock,cash_component=float(row.cash_per_common_share),
                                        receiver_ric=receiver,exchange_ratio=float(row.receiver_shares_per_common_share),receiver_raw_open=opening,
                                        deal_number=row.deal_number,source_request=row.source_request,last_raw_quote_date=str(last.date()),last_raw_close=float(q.loc[last,'close']),source_qualified=True))
                    elif currencies.get(ric)!= {'USD'}:rec['reason']='Raw predecessor USD quote denomination not uniquely verified'
                    elif not row.cash_per_common_share>0:rec['reason']='No positive fixed cash consideration'
                    elif not .5<row.cash_per_common_share/q.loc[last,'close']<1.5:rec['reason']='Cash/final quote ratio outside broad reconciliation range; manual content audit'
                    else:
                        rec.update(state='reconciled_fixed_cash_candidate',reason='Explicit USD/common-share cash; direct final raw close; next observed session; no intervening market quote gap')
                        settlements.append(dict(ticker=ric,effective_date=str(settlement.date()),raw_cash_price=float(row.cash_per_common_share),
                            event_type='cash_merger',event_phase='open',deal_number=row.deal_number,source_request=row.source_request,
                            last_raw_quote_date=str(last.date()),last_raw_close=float(q.loc[last,'close']),source_qualified=True))
        records.append(rec)
    out=DATA/'terminal_audit';pd.DataFrame(records).to_csv(out/'quote_reconciliation.csv',index=False)
    s=pd.DataFrame(settlements)
    if len(s):
        s['official_source_url']=[overrides.get((r.ticker,str(r.deal_number)),{}).get('source_url') for r in s.itertuples()]
        duplicates=s.ticker.duplicated(False);s=s[~duplicates];s.to_csv(out/'reconciled_settlement_candidates.csv',index=False)
        s[s.event_type.eq('cash_merger')].to_csv(out/'fixed_cash_candidates.csv',index=False)
    report=dict(quote_dates=len(quotes),consideration_rows=len(records),fixed_cash_candidates=int(s.event_type.eq('cash_merger').sum()) if len(s) else 0,
        stock_cash_candidates=int(s.event_type.eq('stock_merger_liquidation').sum()) if len(s) else 0,account_approved=False,
        limitation='Source reconciliation candidates only; qualified predecessor TR/raw basis and account-specific held-event audits remain required. Stock receipt liquidated at first regular receiver open, with2bps on stock component.')
    (out/'quote_reconciliation_status.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
