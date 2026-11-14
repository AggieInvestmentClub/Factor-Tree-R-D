"""Audit sealed inputs, timing and portable rebuild arrays without changing them."""
import argparse
import hashlib
from pathlib import Path
from v39 import ROOT, VERSION, read, write, sha, external_path, workspace, check_data, limit_resources


def audit(input_root, work_root):
    limit_resources()
    import numpy as np
    import pandas as pd
    source = external_path(input_root)
    work = workspace(work_root)
    input_check = check_data(source, 'rebuild')
    original = source / VERSION
    rebuilt = work / VERSION
    metadata = read(original / 'data/model/metadata.json')
    regenerated = read(rebuilt / 'data/model/metadata.json')
    assert metadata == regenerated, 'Metadata ordering changed'
    dates = pd.DatetimeIndex(metadata['dates'])
    clock = pd.DatetimeIndex(metadata['calendar'])
    ends = pd.DatetimeIndex(metadata['label_ends'])
    names = metadata['names']
    assert dates.is_unique and dates.is_monotonic_increasing
    assert clock.is_unique and clock.is_monotonic_increasing
    assert len(names) == len(set(names))
    assert len(metadata['columns']) == len(set(metadata['columns'])) == 36
    assert len(ends) == len(dates)
    positions = clock.get_indexer(dates)
    assert (positions >= 0).all()
    assert ((positions - clock.get_loc(metadata['anchor'])) % 21 == 0).all()
    for pos, end in zip(positions, ends):
        if pd.notna(end):
            assert end == clock[pos + 22], 'H21 execution/label clock changed'
    errors = {}
    shapes = {}
    for name in ('x', 'centered', 'diag', 'orth', 'corr_orth', 'factor_loadings',
                 'mask', 'beta', 'gate', 'returns', 'complete', 'qualified_returns'):
        a = np.load(original / f'data/model/{name}.npy', mmap_mode='r', allow_pickle=False)
        b = np.load(rebuilt / f'data/model/{name}.npy', mmap_mode='r', allow_pickle=False)
        assert a.shape == b.shape and a.dtype == b.dtype, name
        maximum = 0.0
        for left, right in zip(a, b):
            np.testing.assert_allclose(left, right, rtol=0, atol=1e-10, equal_nan=True)
            delta = np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
            maximum = max(maximum, float(np.max(delta[np.isfinite(delta)], initial=0)))
        errors[name] = maximum
        shapes[name] = list(a.shape)
    assert shapes['x'] == [len(dates), len(names), 58]
    assert shapes['centered'] == [len(dates), 63, len(names)]
    mask = np.load(original / 'data/model/mask.npy', allow_pickle=False)
    complete = np.load(original / 'data/model/complete.npy', allow_pickle=False)
    returns = np.load(original / 'data/model/returns.npy', mmap_mode='r', allow_pickle=False)
    assert np.isfinite(returns[complete][mask[complete]]).all()
    for key in ('x', 'beta', 'gate', 'centered', 'diag', 'orth', 'corr_orth', 'factor_loadings'):
        values = np.load(original / f'data/model/{key}.npy', mmap_mode='r', allow_pickle=False)
        assert all(np.isfinite(row).all() for row in values), key
    training = {}
    for year in (2024, 2025, 2026):
        model = read(original / f'models/log_growth_{year}.json')
        cut = pd.Timestamp(year, 1, 1)
        split = cut - pd.DateOffset(years=2)
        selected = (dates >= cut - pd.DateOffset(years=10)) & (dates < cut) & (ends < cut) & complete
        assert int(selected.sum()) == model['training_periods']
        assert hashlib.sha256(dates[selected].asi8.tobytes()).hexdigest() == model['training_query_sha256']
        assert pd.Timestamp(model['max_training_label_end']) < cut
        assert pd.Timestamp(model['inner_max_training_label_end']) < split
        assert model['weights_sha256'] == sha(original / f'models/log_growth_{year}.pt')
        training[str(year)] = {'training_periods': int(selected.sum()),
                               'max_label_end': model['max_training_label_end'],
                               'refit_boundary': str(cut.date()), 'strict_maturity_passed': True}
    series = {}
    for period, count in [('development', 4277), ('forward', 719), ('full', 4996)]:
        frame = pd.read_csv(ROOT / f'reference/daily_returns_{period}.csv.gz', index_col=0, parse_dates=True)
        assert frame.index.is_unique and frame.index.is_monotonic_increasing
        assert len(frame) == count and frame.columns.tolist() == ['benchmark', 'vol2sigma', 'SPY']
        assert np.isfinite(frame.to_numpy()).all() and (frame.to_numpy() > -1).all()
        series[period] = {'sessions': len(frame), 'start': str(frame.index[0].date()),
                          'end': str(frame.index[-1].date())}
    forward = pd.read_csv(ROOT / 'reference/daily_returns_forward.csv.gz', index_col=0, parse_dates=True)
    account_checks = {}
    for arm in ('benchmark', 'vol2sigma'):
        ledger = pd.read_csv(rebuilt / f'accounts/{arm}_ledger.csv.gz', index_col=0, parse_dates=True)
        assert ledger.index.equals(forward.index)
        np.testing.assert_allclose(ledger.daily_return, forward[arm], rtol=0, atol=1e-10)
        np.testing.assert_allclose(np.cumprod(1 + ledger.daily_return), ledger.nav, rtol=0, atol=1e-10)
        assert not ledger[['unpriced_held_open', 'unpriced_held_close', 'order_batch_aborted', 'margin_calls']].to_numpy().any()
        audit = read(rebuilt / f'accounts/{arm}_complete.json')['audit']
        assert audit['arithmetic_passed'] and not audit['validated_performance']
        assert audit['provisional_sessions'] == 15
        account_checks[arm] = {'arithmetic_passed': True, 'provisional_sessions': 15,
                              'validated_performance': False, 'held_quote_gaps': 0,
                              'aborted_batches': 0, 'margin_calls': 0}
    for name in ('feature', 'label', 'gate', 'array'):
        assert read(rebuilt / f'data/{name}_qa.json')['passed'], name
    gate = pd.read_csv(rebuilt / 'data/gate.csv', index_col=0, parse_dates=True)
    gate = gate.loc[gate.index >= '2024-01-02']
    assert (pd.to_datetime(gate.history_label_end_max) < gate.index).all()
    assert (gate.mature_dates == 252).all()
    report = {'passed': True, 'scope': 'Frozen inputs and forward rebuild integrity; not an independent vendor recapture',
              'inputs': input_check, 'model_queries': len(dates), 'stored_names': len(names),
              'unique_sorted_dates_and_names': True, 'h21_label_clock_passed': True,
              'annual_training': training, 'rebuilt_array_max_absolute_errors': errors,
              'array_shapes': shapes, 'gate_strict_maturity': True, 'return_series': series,
              'accounts': account_checks, 'historical_vintage_database': False,
              'data_fully_unconditional': False}
    write(work / 'data_check.json', report)
    print('DATA_CHECK_PASSED: hashes, arrays, timing, account arithmetic; conditional marks retained', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--work-root', required=True)
    args = parser.parse_args()
    audit(args.input_root, args.work_root)
