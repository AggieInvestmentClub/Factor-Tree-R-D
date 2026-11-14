"""Rebuild only H5/H10/H21 labels on the immutable historical panel census.

Uses the original qualified quote, terminal-wealth, pair and label primitives.
No features, eligibility flags, memberships, account settings or sources mutate.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(str(Path(__file__).resolve().parents[3] / 'prior/v17_long_history'))
sys.path.insert(0, str(SOURCE / 'src'))
from historical_features import label_quotes, _prepare, _pair, _labels, _label_wealth, SECTORS, IDIO_FLOOR
from model_core_long import known

HORIZONS = (5, 10, 21)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def conditional_event(row):
    """Exact qualification gate from historical_features.main."""
    conditional = ((pd.notna(row.get('cvr_upper')) and row.get('cvr_upper', 0) > 0)
                   or (pd.notna(row.get('unpriced_rights')) and bool(row.get('unpriced_rights', False))))
    conditional |= pd.notna(row.get('label_wealth_qualified')) and not bool(row.get('label_wealth_qualified'))
    return bool(conditional)


def columns(h):
    return [f'future_return_h{h}', f'future_parent_return_h{h}',
            f'target_sector_residual_h{h}', f'target_risk_resid_h{h}']


def synthetic_source(calendar):
    close = pd.Series(100 + np.arange(len(calendar), dtype=float), index=calendar)
    frame = pd.DataFrame(index=calendar)
    for c in ('adj_open', 'adj_close', 'adj_high', 'adj_low'):
        frame[c] = close
    frame['Volume'] = 1000.
    frame['raw_close'] = close / 2
    frame['q'] = 1
    frame['tr_segment_id'] = 1
    frame['wealth_labels_qualified'] = 1
    frame['source_available'] = 1
    frame['tr_chain_available'] = 1
    return frame


def test_primitives():
    calendar = pd.bdate_range('2010-01-04', periods=180)
    source = synthetic_source(calendar)
    quotes = label_quotes(source, calendar)
    zero = pd.Series(0., index=calendar)
    one = pd.Series(1., index=calendar)
    labels = _labels(quotes, quotes, zero, one, one, HORIZONS, 'close')
    checks = []
    for h in HORIZONS:
        i = 10
        assert labels.loc[calendar[i], 'execution_date'] == calendar[i+1]
        assert labels.loc[calendar[i], f'label_end_h{h}'] == calendar[i+h+1]
        expected = source.adj_close.iloc[i+h+1] / source.adj_close.iloc[i+1] - 1
        assert labels.loc[calendar[i], f'future_return_h{h}'] == expected
        assert labels.loc[calendar[i], f'target_risk_resid_h{h}'] == expected / np.sqrt(h)
        assert labels[f'label_end_h{h}'].tail(h+1).isna().all()
        assert labels[f'future_return_h{h}'].tail(h+1).isna().all()
    checks.append('H5/H10/H21 next-close entry t+1, endpoint t+h+1, sqrt(h) scale and calendar tail')

    missing = source.copy()
    missing.loc[calendar[14], 'q'] = 0
    missing.loc[calendar[14], 'adj_close'] = np.nan
    no_quotes = label_quotes(missing, calendar)
    gap_labels = _labels(no_quotes, quotes, zero, one, one, HORIZONS, 'close')
    assert np.isnan(gap_labels.loc[calendar[10], 'future_return_h5'])
    assert gap_labels.loc[calendar[5], 'future_return_h5'] == labels.loc[calendar[5], 'future_return_h5']
    assert np.isfinite(gap_labels.loc[calendar[20], 'future_return_h5'])
    checks.append('Missing internal future quote breaks the return chain; no forward fill across gap')

    terminated = source.copy()
    terminated.loc[calendar[40]:, ['q', 'source_available', 'tr_chain_available']] = 0
    terminated.loc[calendar[40]:, ['adj_open', 'adj_close', 'adj_low', 'adj_high', 'raw_close']] = np.nan
    before = terminated.copy(deep=True)
    unextended = label_quotes(terminated, calendar)
    event = pd.Series(dict(ticker='SYNTH', effective_date=calendar[40], raw_cash_price=100.))
    extended, record = _label_wealth(unextended, event)
    assert record['applied'] and record['adjusted_cash_wealth'] == 200.
    terminal_labels = _labels(extended, quotes, zero, one, one, HORIZONS, 'close')
    assert terminal_labels.loc[calendar[35], 'future_return_h5'] == 200/source.adj_close.iloc[36]-1
    pd.testing.assert_frame_equal(terminated, before)
    assert not _prepare(terminated, calendar).price_available.iloc[40]
    assert conditional_event(pd.Series(dict(cvr_upper=1.)))
    assert conditional_event(pd.Series(dict(label_wealth_qualified=False)))
    assert not conditional_event(event)
    checks.append('Verified terminal settlement extends label wealth only; contingent/unqualified terms excluded')

    parent = source.copy()
    r = .001 + .008*np.sin(np.arange(len(calendar))*.41)
    p = 100*np.cumprod(1+r)
    s = 100*np.cumprod(1+1.3*r+.002*np.cos(np.arange(len(calendar))*.61))
    for c in ('adj_open', 'adj_close', 'adj_high', 'adj_low'):
        parent[c] = p
        source[c] = s
    original_pair = _pair(_prepare(source, calendar), _prepare(parent, calendar))
    altered = source.copy()
    for c in ('adj_open', 'adj_close', 'adj_high', 'adj_low'):
        altered.loc[calendar[120]:, c] *= 100
    altered_pair = _pair(_prepare(altered, calendar), _prepare(parent, calendar))
    for a, b in zip(original_pair, altered_pair):
        pd.testing.assert_series_equal(a.iloc[:120], b.iloc[:120])
    checks.append('Ex-ante beta/idiosyncratic scale unchanged by later quotes')
    return checks


def identity(new, original, name, mask):
    a, b = new.loc[mask, name], original.loc[mask, name]
    pd.testing.assert_index_equal(a.index, b.index)
    assert np.array_equal(a.isna().to_numpy(), b.isna().to_numpy()), name + ' missingness differs'
    if pd.api.types.is_datetime64_any_dtype(a):
        pd.testing.assert_series_equal(a, b, check_names=False)
        return dict(column=name, rows=int(mask.sum()), comparable=int(a.notna().sum()), exact=True)
    valid = a.notna()
    aa, bb = a.loc[valid].to_numpy(float), b.loc[valid].to_numpy(float)
    delta = float(np.max(np.abs(aa-bb), initial=0.))
    assert np.array_equal(aa, bb), f'{name} differs; maximum absolute error {delta}'
    return dict(column=name, rows=int(mask.sum()), comparable=int(valid.sum()), exact=True, max_abs_error=delta)


def main():
    start = time.time()
    tests = test_primitives()
    data = ROOT/'data'
    reports = ROOT/'reports/labels'
    data.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    assembled = SOURCE/'data/assembled'
    files = dict(prices=assembled/'prices_candidate.pkl', events=assembled/'events.csv',
                 panel=SOURCE/'data/candidate_features/panel.pkl', calendar=assembled/'calendar.csv',
                 approval=assembled/'approval.json', historical_features=SOURCE/'src/historical_features.py',
                 historical_sources=SOURCE/'src/historical_sources.py',
                 stock_features=SOURCE.parent/'v16_payoff_alignment/src/stock_features.py',
                 builder=Path(__file__))
    hashes = {name:sha(path) for name, path in files.items()}
    approval = json.loads(files['approval'].read_text())
    assert approval['source_qualification_passed'] and not approval['blockers']
    for name in ('prices', 'events', 'panel'):
        assert hashes[name] == approval[name+'_sha256'], 'Approved source hash mismatch: '+name
    prices = pd.read_pickle(files['prices'])
    panel = pd.read_pickle(files['panel'])
    events = pd.read_csv(files['events'], parse_dates=['effective_date'])
    calendar = pd.DatetimeIndex(pd.read_csv(files['calendar'], parse_dates=['Date']).Date)
    assert calendar.is_unique and calendar.is_monotonic_increasing
    assert calendar.max() < pd.Timestamp('2024-01-01')
    assert panel.decision_date.max() < pd.Timestamp('2024-01-01')
    assert panel.index.is_unique and not panel.duplicated(['decision_date', 'ticker']).any()
    assert not events.ticker.duplicated().any()
    out = panel[['decision_date', 'ticker']].copy()
    clock = pd.Series(calendar, index=calendar)
    out['execution_date'] = panel.decision_date.map(clock.shift(-1))
    for h in HORIZONS:
        out[f'label_end_h{h}'] = panel.decision_date.map(clock.shift(-(h+1)))
        for col in columns(h):
            out[col] = np.nan
    etfs = {k:_prepare(prices[k], calendar) for k in set(SECTORS)|{'IYZ', 'SPY'}}
    label_etfs = {k:label_quotes(prices[k], calendar) for k in etfs}
    for h in HORIZONS:
        spy = label_etfs['SPY'].adj_close
        # Exact original historical_features SPY formula; approved calendar is complete.
        spy_return = spy.shift(-(h+1))/spy.shift(-1)-1
        out[f'future_spy_h{h}'] = panel.decision_date.map(spy_return)
    null_atr = pd.Series(np.nan, index=calendar)
    terminal_records = []
    sigma_errors = []
    grouped = panel.groupby('ticker', sort=True)
    for counter, (ticker, rows) in enumerate(grouped, 1):
        stock = _prepare(prices.get(ticker), calendar)
        stock_labels = label_quotes(prices[ticker], calendar) if ticker in prices else stock
        event = events.loc[events.ticker.eq(ticker)]
        if len(event):
            row = event.iloc[0]
            if conditional_event(row):
                terminal_records.append(dict(ticker=ticker, effective_date=str(row.effective_date), applied=False,
                    reason='Unpriced contingent rights or incomplete qualified shareholder wealth'))
            else:
                stock_labels, detail = _label_wealth(stock_labels, row)
                terminal_records.append(detail)
        for sector in rows.sector.dropna().unique():
            current = rows.sector.eq(sector)
            routes = [('IYZ', current & rows.decision_date.lt('2018-09-24')),
                      ('XLC', current & rows.decision_date.ge('2018-09-24'))] if sector == 'XLC' else [(sector, current)]
            for route, mask in routes:
                if not mask.any():
                    continue
                beta, _, idio = _pair(stock, etfs[route])
                sigma = idio.clip(lower=IDIO_FLOOR)
                selected = rows.loc[mask]
                expected_sigma = selected.decision_date.map(sigma).set_axis(selected.index)
                a, b = expected_sigma.to_numpy(float), selected.idio_vol21.to_numpy(float)
                assert np.array_equal(np.isnan(a), np.isnan(b)), 'Ex-ante sigma missingness changed'
                valid = np.isfinite(a)
                assert np.array_equal(a[valid], b[valid]), 'Ex-ante sigma changed'
                sigma_errors.append(float(np.max(np.abs(a[valid]-b[valid]), initial=0.)))
                labels = _labels(stock_labels, label_etfs[route], beta, sigma, null_atr, HORIZONS, 'close')
                wanted = [col for h in HORIZONS for col in columns(h)]
                out.loc[selected.index, wanted] = labels.loc[pd.DatetimeIndex(selected.decision_date), wanted].to_numpy()
        if counter % 100 == 0:
            print('LABELS', counter, grouped.ngroups, round(time.time()-start, 1), flush=True)
    pd.testing.assert_index_equal(out.index, panel.index)
    pd.testing.assert_frame_equal(out[['decision_date','ticker']], panel[['decision_date','ticker']])
    names = ['execution_date', 'label_end_h5', 'future_return_h5', 'future_parent_return_h5',
             'future_spy_h5', 'target_risk_resid_h5']
    identities = []
    for scope, mask in [('all_original_rows', pd.Series(True, index=panel.index)), ('known_current_census', known(panel))]:
        for name in names:
            identities.append(dict(scope=scope, **identity(out, panel, name, mask)))
    eligible = known(panel)
    audit = []
    for h in HORIZONS:
        q = out.loc[eligible, ['decision_date', f'label_end_h{h}']].copy()
        q['known_rows'] = 1
        q['calendar_tail_rows'] = q[f'label_end_h{h}'].isna().astype(int)
        for base in ('future_return', 'future_parent_return', 'future_spy', 'target_risk_resid'):
            q['missing_'+base] = (~np.isfinite(out.loc[eligible, f'{base}_h{h}'].to_numpy(float))).astype(int)
        row_ok = q[[c for c in q if c.startswith('missing_')]].eq(0).all(axis=1) & q.calendar_tail_rows.eq(0)
        q['complete_rows'] = row_ok.astype(int)
        q = q.drop(columns=[f'label_end_h{h}']).groupby('decision_date', sort=True).sum()
        q['complete_date'] = q.known_rows.eq(q.complete_rows)
        q['horizon'] = h
        audit.append(q.reset_index())
    census = pd.concat(audit, ignore_index=True)
    census.to_csv(reports/'whole_date_missing_audit.csv', index=False)
    yearly = census.assign(year=census.decision_date.dt.year).groupby(['horizon','year']).agg(
        known_dates=('complete_date','size'), complete_dates=('complete_date','sum'), known_rows=('known_rows','sum'),
        complete_rows=('complete_rows','sum'), calendar_tail_rows=('calendar_tail_rows','sum'),
        missing_stock_rows=('missing_future_return','sum'), missing_parent_rows=('missing_future_parent_return','sum'),
        missing_spy_rows=('missing_future_spy','sum'), missing_sector_risk_rows=('missing_target_risk_resid','sum')).reset_index()
    yearly.to_csv(reports/'annual_label_coverage.csv', index=False)
    pd.DataFrame(terminal_records).to_csv(reports/'terminal_label_applications.csv', index=False)
    out.to_pickle(data/'horizon_labels.pkl')
    result = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='complete',
        rows=len(out), known_rows=int(eligible.sum()), horizons=list(HORIZONS), columns=list(out.columns),
        original_index_preserved=True, features_and_eligibility_unchanged=True, no_rows_dropped=True,
        calendar_start=str(calendar.min().date()), calendar_end=str(calendar.max().date()),
        endpoint='next-close entry at t+1, terminal close t+h+1; h intervening close-to-close returns',
        source_paths={k:str(v) for k,v in files.items()}, source_hashes=hashes,
        label_pickle_sha256=sha(data/'horizon_labels.pkl'), tests=tests, h5_identity=identities,
        exante_sigma_max_absolute_error=max(sigma_errors),
        complete_dates_by_horizon={str(h):int(g.complete_date.sum()) for h,g in census.groupby('horizon')},
        terminal_applied=int(sum(bool(r['applied']) for r in terminal_records)),
        terminal_excluded=int(sum(not bool(r['applied']) for r in terminal_records)),
        attrition='Whole-date audit uses original known-member census; never filters prediction rows by future label availability',
        source_qualification=approval['state'], seconds=time.time()-start)
    (reports/'label_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','rows','known_rows','complete_dates_by_horizon','exante_sigma_max_absolute_error','seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    if args.test_only:
        print(json.dumps(test_primitives(), indent=2))
    else:
        main()
