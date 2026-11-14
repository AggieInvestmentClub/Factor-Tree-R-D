"""Date-qualified source assembly. This does not certify terminal-event accounting."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from audit_capture_years import align,FIELDS
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'
ALIASES={'BKR.N':'BKR.OQ','HST.N':'HST.OQ','KMB.N':'KMB.OQ','CPB.N':'CPB.OQ',
 'RF.N':'RF','BMC.OQ^I13':'BMC.O^I13','SAF.N^I08':'SAF^I08','FITB.N':'FITB.O','FITB.OQ':'FITB.O',
 'XLRE.P':'XLRE'} # Exact real-estate ETF ISIN US81369Y8600, dated source captures2015-2023.

def canonical_ric(value):return ALIASES.get(value,value)

def membership_names(frame):
    f=frame.copy()
    # Add the explicit identity boundary even if it was not an index-rebalance
    # date in the vendor's normalized event table.
    boundary='2019-06-11';extra=[]
    blocks=f.groupby('sector') if 'sector' in f else [(None,f)]
    for _,g in blocks:
        earlier=g[g.date.lt(boundary)]
        if len(earlier) and not g.date.eq(boundary).any():
            last=earlier[earlier.date.eq(earlier.date.max())].copy()
            if last.ric.eq('AMCR.N').any():last['date']=boundary;extra.append(last)
    if extra:f=pd.concat([f]+extra,ignore_index=True)
    f['ric']=f.ric.map(canonical_ric)
    # Independent WRDS PERMNO intervals and dated LSEG RIC history identify
    # Bemis before June11,2019. Never attribute those prices to an existing Amcor.
    f.loc[f.ric.eq('AMCR.N')&pd.to_datetime(f.date).lt('2019-06-11'),'ric']='BMS.N^F19'
    return f.drop_duplicates()

def last_complete_year():
    out=DATA/'annual_aligned';years=[]
    for y in range(2000,2024):
        if (out/f'{y}_status.json').exists():years.append(y)
        else:break
    if not years:raise ValueError('No complete primary capture year')
    return max(years)

def load_fields(end_year):
    chunks=[];provenance=[];conflicts=[]
    for year in range(2000,end_year+1):
        path=DATA/f'annual_aligned/{year}.pkl';z=pd.read_pickle(path)[[f.removeprefix('TR.') for f in FIELDS]]
        chunks.append(z);provenance.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    # Only the same separately dated OHLC/return field schema is admitted.
    for meta in sorted((DATA/'audit_requests').glob('*.json')):
        record=json.loads(meta.read_text());request=record['request']
        required_fields={k for f in FIELDS for k in (f+'.date',f)}
        if record.get('state')!='captured_not_audited' or not required_fields.issubset(request.get('fields',[])):continue
        if pd.Timestamp(request['parameters']['SDate']).year>end_year:continue
        path=meta.with_suffix('.pkl');assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
        raw=pd.read_pickle(path)
        for year in range(max(2000,pd.Timestamp(request['parameters']['SDate']).year),min(end_year,pd.Timestamp(request['parameters']['EDate']).year)+1):
            z,issues=align(raw,year);chunks.append(z);conflicts.extend(issues)
        provenance.append(dict(path=str(path.relative_to(ROOT)),sha256=record['sha256']))
    all_data=pd.concat(chunks).reset_index();all_data['source_route']=all_data.ric
    all_data['ric']=all_data.ric.map(canonical_ric)
    # Same issue != same venue OHLC/volume. Use one declared continuous quote
    # route when present; never blend two exchanges' intraday bars/volume.
    all_data['route_priority']=all_data.source_route.ne(all_data.ric).astype(int)
    preferred=all_data.groupby(['ric','date']).route_priority.transform('min')
    discarded=int(all_data.route_priority.gt(preferred).sum())
    provenance.append(dict(policy='Prefer declared canonical quote route for entire source row; no venue field blending',discarded_alternate_venue_rows=discarded))
    all_data=all_data[all_data.route_priority.eq(preferred)]
    columns=[f.removeprefix('TR.') for f in FIELDS]
    all_data[columns]=all_data[columns].astype(float)
    groups=all_data.groupby(['ric','date'],sort=False);low=groups[columns].min();high=groups[columns].max()
    bad=(high-low).abs().gt(1e-7*np.maximum(1,np.maximum(low.abs(),high.abs())))
    for (ric,date),row in bad[bad.any(axis=1)].iterrows():
        conflicts.extend(dict(ric=ric,date=str(date.date()),field=field) for field in row.index[row])
    combined=groups[columns].first().mask(bad).sort_index()
    routefile=ROOT/'direct_source_routes.json';directfile=DATA/'direct_history/candidates.pkl'
    if routefile.exists() and directfile.exists():
        routes=json.loads(routefile.read_text());direct=pd.read_pickle(directfile)
        replacements=[];selected=set(routes['routes'])
        for ric in selected:
            assert ric in direct,'Declared direct history missing'
            frame=direct[ric].loc[direct[ric].index.year<=end_year,columns+['raw_close']].copy()
            frame=frame.rename(columns={'raw_close':'RawClose'});frame['ric']=canonical_ric(ric)
            frame.index.name='date';frame=frame.reset_index().set_index(['ric','date']);replacements.append(frame)
        combined=combined.loc[~combined.index.get_level_values('ric').isin(selected)]
        combined=pd.concat([combined]+replacements).sort_index()
        discarded_conflicts=[c for c in conflicts if c['ric'] in selected]
        conflicts=[c for c in conflicts if c['ric'] not in selected]
        provenance.append(dict(policy='Whole-issue direct quote/cash-dividend replacement under explicit source route contract',
            routes=sorted(selected),source=str(directfile.relative_to(ROOT)),sha256=hashlib.sha256(directfile.read_bytes()).hexdigest(),
            contract_sha256=hashlib.sha256(routefile.read_bytes()).hexdigest(),discarded_fundamental_conflicts=discarded_conflicts))
    from reconcile_crisis_quotes import apply_windows
    combined,crisis_records=apply_windows(combined)
    provenance.extend(crisis_records)
    distribution_file=DATA/'distribution_audit/shareholder_value_reconciliation.csv'
    if distribution_file.exists():
        distributions=pd.read_csv(distribution_file,parse_dates=['ex_date']);changed=[]
        combined['DistributionReinvestmentTurnover']=0.
        for action in distributions.to_dict('records'):
            # Exact issuer terms and observed parent/child closes establish
            # gross shareholder wealth. Distributed shares are liquidated and
            # their proceeds reinvested in the parent at that close. Both legs
            # carry the same declared one-way fee in the account, not labels.
            if not action['state'].startswith('Observed raw'):continue
            key=(action['ric'],action['ex_date']);value=action.get('shareholder_close_return_pct',np.nan)
            if key not in combined.index or not np.isfinite(value) or value<=-100:continue
            old=combined.loc[key,'TotalReturn1D'];combined.loc[key,'TotalReturn1D']=value
            wealth=action['previous_parent_raw_close']*(1+value/100)
            combined.loc[key,'DistributionReinvestmentTurnover']=2*abs(action['omitted_distribution_value'])/wealth
            changed.append(dict(ric=key[0],date=str(key[1].date()),old_return_pct=float(old),verified_gross_return_pct=float(value),source=action['source']))
        provenance.append(dict(policy='Verified parent+distributed-child closing gross wealth replaces omitted noncash distributions; no quote fabricated. Liquidate distributed shares and reinvest at observed closes; both market legs charged at declared one-way rate in accounts.',
            source=str(distribution_file.relative_to(ROOT)),sha256=hashlib.sha256(distribution_file.read_bytes()).hexdigest(),corrections=changed))
    from reconcile_due_bill import apply as reconcile_due_bill
    combined,due_bill_records=reconcile_due_bill(combined)
    provenance.extend(due_bill_records)
    from reconcile_delayed_distributions import apply as delayed_distributions
    combined,delayed_records=delayed_distributions(combined)
    provenance.extend(delayed_records)
    assert combined.index.get_level_values('date').max()<pd.Timestamp('2024-01-01')
    return combined,provenance,conflicts

def convert(frame,calendar):
    """Forward total-return gauge within observed complete segments, no fills.

    New segments have arbitrary positive initial price gauges. An account may
    not cross their breaks without a separately verified reconciliation/event.
    Raw cash conversion is intentionally unavailable until raw prices are read.
    """
    z=frame.reindex(calendar).astype(float);p=z[['PriceOpen','PriceHigh','PriceLow','PriceClose']]
    bar=np.isfinite(p.to_numpy()).all(axis=1)&p.gt(0).all(axis=1).to_numpy()
    tolerance=p.abs().max(axis=1).clip(lower=1)*1e-7
    bar&=z.PriceHigh.add(tolerance).ge(p.max(axis=1)).to_numpy()&z.PriceLow.sub(tolerance).le(p.min(axis=1)).to_numpy()
    # Malformed high/low ranges invalidate technical features, but do not erase
    # separately recorded positive OC quotes or their valid total-return chain.
    # No high/low value is repaired or invented.
    oc=z[['PriceOpen','PriceClose']]
    quotes=np.isfinite(oc.to_numpy()).all(axis=1)&oc.gt(0).all(axis=1).to_numpy()
    ret=z.TotalReturn1D/100.;good=quotes&np.isfinite(ret.to_numpy())&ret.gt(-1).to_numpy()
    starts=~pd.Series(good,index=calendar).shift(1,fill_value=False).to_numpy()|~good
    segment=np.cumsum(starts);close=np.full(len(z),np.nan)
    # The first observation establishes a price unit, not a fabricated return.
    previous=False
    for i in range(len(z)):
        if good[i]:close[i]=close[i-1]*(1+ret.iloc[i]) if previous else z.PriceClose.iloc[i]
        previous=good[i]
    ratio=close/z.PriceClose.to_numpy();out=pd.DataFrame(index=calendar)
    for source,target in [('PriceOpen','adj_open'),('PriceHigh','adj_high'),('PriceLow','adj_low'),('PriceClose','adj_close')]:out[target]=z[source].to_numpy()*ratio
    out['Volume']=z.Volume;out['raw_close']=z.RawClose if 'RawClose' in z else np.nan
    out['distribution_reinvestment_turnover']=z.DistributionReinvestmentTurnover.fillna(0.) if 'DistributionReinvestmentTurnover' in z else 0.
    out['total_return']=ret;out['split_adjusted_close']=z.PriceClose
    unpriced=z.UnpricedDistributionClaim.fillna(0).eq(1) if 'UnpricedDistributionClaim' in z else pd.Series(False,index=calendar)
    out['source_available']=(bar&~unpriced).astype(int);out['tr_chain_available']=good.astype(int)
    out['wealth_labels_qualified']=(~unpriced).astype(int)
    out['q']=good.astype(int)
    out['tr_segment_id']=np.where(good,segment,np.nan)
    out['raw_cash_conversion_qualified']=good&out.raw_close.notna()&out.raw_close.gt(0)
    out.index.name='Date'
    return out

def label_quotes(source,calendar):
    """Observed OC wealth for close-return labels, separate from OHLC features."""
    z=source.reindex(calendar).copy();z['price_available']=z.q.eq(1);z['segment_id']=z.tr_segment_id
    if 'wealth_labels_qualified' in z:
        z['price_available'] &= z.wealth_labels_qualified.eq(1)
        breaks=(~z.price_available)|(~z.price_available.shift(1,fill_value=False))|z.tr_segment_id.ne(z.tr_segment_id.shift(1))
        z['segment_id']=breaks.cumsum()
    same=z.price_available&z.price_available.shift(1,fill_value=False)&z.segment_id.eq(z.segment_id.shift(1))
    z['close_return']=(z.adj_close/z.adj_close.shift(1)-1).where(same)
    z['open_return']=(z.adj_open/z.adj_open.shift(1)-1).where(same)
    return z

def assemble(end_year=None):
    end_year=last_complete_year() if end_year is None else end_year
    assert 2000<=end_year<=2023
    raw,provenance,conflicts=load_fields(end_year)
    spy=raw.xs('SPY');ohlc=spy[['PriceOpen','PriceHigh','PriceLow','PriceClose']]
    clockfile=DATA/'direct_spy_calendar.csv'
    if not clockfile.exists():raise ValueError('Independent direct SPY session clock required')
    calendar=pd.DatetimeIndex(pd.read_csv(clockfile,parse_dates=['Date']).Date)
    calendar=calendar[(calendar.year>=2000)&(calendar.year<=end_year)]
    # TR fundamental Frq=D omits four real US exchange sessions in2006/2007.
    # Recover their observed fields with Frq=C; never shorten the session clock
    # to hide missing quotes or accidentally alter the five-session horizon.
    valid_spy=ohlc.notna().all(axis=1)&spy.TotalReturn1D.notna()&spy.Volume.gt(0)
    absent=calendar[~calendar.isin(spy.index[valid_spy])]
    if len(absent):raise ValueError('Observed SPY sessions awaiting source recovery: '+str(absent.astype(str).tolist()))
    assert calendar.is_monotonic_increasing and calendar.is_unique
    counts=pd.Series(calendar.year).value_counts();assert counts.ge(240).all(), 'Incomplete annual benchmark calendar'
    from reconcile_terminal_quotes import direct_quotes
    terminal_quotes,_,terminal_currencies=direct_quotes()
    from reconcile_halt_gauges import reconcile as reconcile_halts
    haltfile=DATA/'terminal_audit/known_halts.csv'
    halts=pd.read_csv(haltfile,parse_dates=['date']) if haltfile.exists() else pd.DataFrame(columns=['ticker','date'])
    prices={};records=[];halt_records=[]
    for ric,g in raw.groupby(level='ric',sort=False):
        z=convert(g.droplevel('ric'),calendar)
        halted=halts[halts.ticker.eq(ric)]
        if len(halted):
            z,checks=reconcile_halts(z,g.droplevel('ric'),halted.date)
            halt_records.extend(dict(ric=ric,**check) for check in checks)
        if len(terminal_quotes) and ric in terminal_quotes.index.get_level_values('ric') and terminal_currencies.get(ric)=={'USD'}:
            direct=terminal_quotes.xs(ric).close.reindex(calendar)
            both=direct.notna()&z.raw_close.notna()
            assert np.allclose(direct[both],z.loc[both,'raw_close'],rtol=1e-7,atol=1e-8),'Conflicting raw cash conversion basis'
            z.loc[direct.notna(),'raw_close']=direct[direct.notna()]
            z['raw_cash_conversion_qualified']=z.q.eq(1)&z.raw_close.notna()&z.raw_close.gt(0)
        prices[ric]=z
        records.append(dict(ric=ric,complete_bars=int(z.source_available.sum()),return_qualified_bars=int(z.tr_chain_available.sum()),segments=int(z.tr_segment_id.nunique()),
            observed_OC_with_rejected_range=int((z.q.eq(1)&z.source_available.eq(0)).sum()),raw_cash_basis_dates=int(z.raw_cash_conversion_qualified.sum())))
    out=DATA/'assembled';out.mkdir(exist_ok=True)
    pd.to_pickle(prices,out/'prices_candidate.pkl');pd.DataFrame({'Date':calendar}).to_csv(out/'calendar.csv',index=False)
    pd.DataFrame(records).to_csv(out/'coverage.csv',index=False)
    (out/'halt_gauge_reconciliation.json').write_text(json.dumps(halt_records,indent=2))
    candidate_events=DATA/'terminal_audit/reconciled_settlement_candidates.csv'
    if candidate_events.exists():
        events=pd.read_csv(candidate_events,parse_dates=['effective_date'])
        official_file=DATA/'terminal_audit/official_settlement_candidates.csv'
        if official_file.exists():
            official=pd.read_csv(official_file,parse_dates=['effective_date'])
            assert not official.ticker.duplicated().any()
            overlaps=events.merge(official,on='ticker',suffixes=('_provider','_official'))
            provenance.append(dict(policy='Independently raw-quote-reconciled issuer/SEC terms take precedence over parsed provider consideration text',
                source=str(official_file.relative_to(ROOT)),sha256=hashlib.sha256(official_file.read_bytes()).hexdigest(),
                superseded_provider_candidates=int(len(overlaps)),
                note='SIVB missing halt marks and ABMD contingent rights are still conditional, not certified market valuations'))
            events=pd.concat([events[~events.ticker.isin(official.ticker)],official],ignore_index=True)
        otc_file=DATA/'terminal_audit/otc_settlement_candidates.csv'
        if otc_file.exists():
            otc=pd.read_csv(otc_file,parse_dates=['effective_date'])
            assert not otc.ticker.duplicated().any()
            events=pd.concat([events[~events.ticker.isin(otc.ticker)],otc],ignore_index=True)
            provenance.append(dict(source=str(otc_file.relative_to(ROOT)),sha256=hashlib.sha256(otc_file.read_bytes()).hexdigest(),
                policy='Observed first alternate-venue opening exit; historical security and raw share basis verified;2bps liquidity assumption explicit'))
        complex_file=DATA/'terminal_audit/complex_settlement_candidates.csv'
        if complex_file.exists():
            extra=pd.read_csv(complex_file,parse_dates=['effective_date'])
            assert not extra.ticker.duplicated().any()
            events=pd.concat([events[~events.ticker.isin(extra.ticker)],extra],ignore_index=True)
            provenance.append(dict(source=str(complex_file.relative_to(ROOT)),sha256=hashlib.sha256(complex_file.read_bytes()).hexdigest(),
                policy='Conditional CVR/default-election records require explicit account sensitivity; not complete verified training-label wealth'))
        events=events[events.effective_date.le(calendar[-1])&events.ticker.isin(prices)]
        events.to_csv(out/'events.csv',index=False)
        provenance.append(dict(source=str(candidate_events.relative_to(ROOT)),sha256=hashlib.sha256(candidate_events.read_bytes()).hexdigest(),state='Source-reconciled events; account-level qualification pending'))
    status=dict(end_year=end_year,calendar_sessions=len(calendar),calendar_counts={str(k):int(v) for k,v in counts.items()},
                instruments=len(prices),source_conflicts=conflicts,sources=provenance,quote_aliases=ALIASES,
                state='FEATURE_CANDIDATE_ONLY; account forbidden until segment breaks and terminal payouts reconciled',
                identity_policy='Historical provider route, not current issuer identity; Bemis is separately named before2019-06-11.',
                terminal_cash_conversion=False,forward_or_backward_fill=False,
                calendar_source='Independent direct SPY close dates; not inferred from available fundamental rows',
                calendar_sha256=hashlib.sha256(clockfile.read_bytes()).hexdigest(),
                quote_policy='Separate strict OHLC feature validity from observed OC/return-chain valuation validity; no price or range repairs')
    (out/'status.json').write_text(json.dumps(status,indent=2));print('ASSEMBLED',end_year,len(calendar),len(prices),len(conflicts),flush=True)
    return prices,calendar,status

if __name__=='__main__':assemble()
