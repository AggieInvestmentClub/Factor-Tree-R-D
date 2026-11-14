"""v23 horizon/target accounts with only the declared cadence change.

Prior project and v21 are read-only dependencies. All candidates share the
original Huber H5 efficacy gate evaluated on their own fixed schedule. The
gate is not trained on a candidate or changed to a horizon-specific response.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V21 = ROOT.parent / 'v21_forecast_research'
spec = importlib.util.spec_from_file_location('_v23_readonly_v21_account', V21/'src/account_runner.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
SOURCE = legacy.SOURCE
sha, read, write = legacy.sha, legacy.read, legacy.write
signal_engine, account_engine = legacy.signal_engine, legacy.account_engine
NoncashExposureAudit = legacy.NoncashExposureAudit
from fast_gate import compute_gate
from lazy_covariances import build as build_covariances

ORIGINAL = 'huber_ridge_10y'
REFERENCE = 'h5_sector_risk_weekly'
HORIZONS = (5, 10, 21)
TARGETS = ('sector_risk', 'projected_scaled', 'projected_rank')
MODELS = tuple(f'h{h}_{target}' for h in HORIZONS for target in TARGETS)
WEEKLY = tuple(f'{model}_weekly' for model in MODELS)
MATCHED = tuple(f'{model}_matched' for model in MODELS if not model.startswith('h5_'))
ACCOUNTS = (ORIGINAL, *WEEKLY, *MATCHED)
ALTERNATIVES = tuple(name for name in (*WEEKLY, *MATCHED) if name != REFERENCE)
BOUNDARY = pd.Timestamp('2024-01-01')


def account_spec(name):
    if name == ORIGINAL:
        return dict(account=name, model=ORIGINAL, horizon=5, target='original_sector_risk', cadence=5, schedule='weekly')
    if name not in (*WEEKLY, *MATCHED):
        raise ValueError(f'Unknown account {name}')
    model, schedule = name.rsplit('_', 1)
    horizon = int(model.split('_')[0][1:])
    return dict(account=name, model=model, horizon=horizon, target=model.split('_', 1)[1],
                cadence=5 if schedule == 'weekly' else horizon, schedule=schedule)


def selected_accounts(stage):
    return list((ORIGINAL, *WEEKLY) if stage == 'weekly' else MATCHED if stage == 'matched' else ACCOUNTS)


def current_contract():
    path = ROOT/'protocol.json'
    if not path.exists() or not path.with_suffix('.sha256').exists():
        raise ValueError('Root must freeze protocol.json before account preparation')
    assert sha(path) == path.with_suffix('.sha256').read_text().strip()
    return sha(path)


def account_contract():
    protocol = current_contract()
    code = {p.name: sha(p) for p in [Path(__file__), ROOT/'src/evaluate_horizons.py']}
    payload = dict(protocol_sha256=protocol, adapter_code_hashes=code,
        readonly_loader_sha256=sha(V21/'src/account_runner.py'),
        readonly_analysis_sha256=sha(SOURCE/'src/analyze_state_v20.py'),
        accounts=[account_spec(name) for name in ACCOUNTS], reference=REFERENCE,
        alternatives=list(ALTERNATIVES), position_serialization=False,
        gate='Original Huber H5 mature OOS Pearson IC gate recomputed only at each account cadence; common across targets/horizons on same cadence',
        unchanged='EWMA .25; next-close execution; 12% target; 2bps one-way; buffer .2; gross3; name cap.15; original daily guards; no borrow/financing')
    path = ROOT/'account_contract.json'
    if path.exists():
        assert read(path) == payload, 'Frozen account implementation or protocol changed'
    else:
        write(path, payload)
    return sha(path)


def scheduled_dates(predictions, calendar, anchor, cadence):
    if cadence not in (5, 10, 21):
        raise ValueError('Only prespecified 5/10/21 observed-session schedules')
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(predictions.decision_date).unique()))
    clock = pd.DatetimeIndex(calendar)
    assert dates.isin(clock).all() and anchor in clock
    return dates[(clock.get_indexer(dates)-clock.get_loc(anchor)) % cadence == 0]


def build_targets(predictions, panel, calendar, names, covariances, *, anchor,
                  cadence=5, end_before='2024-01-01'):
    """Literal original score/rank/projection/risk sequence; cadence is parameterized.

    Only EWMA scores are retained as a daily stage to save memory. Arithmetic
    is identical to signal_engine.build_weekly_targets when cadence equals 5.
    """
    if cadence not in (5, 10, 21):
        raise ValueError('Unspecified cadence')
    boundary = signal_engine._boundary(end_before)
    anchor = pd.Timestamp(anchor)
    names = list(names)
    if not names or len(names)>signal_engine.MAX_HISTORY_NAMES or len(set(names))!=len(names):
        raise ValueError('Unique declared names required')
    f, p, clock, _ = signal_engine._queries(predictions, panel, calendar, boundary)
    if not set(f.ticker).issubset(names):
        raise ValueError('Forecast outside declared names')
    dates = pd.DatetimeIndex(sorted(f.decision_date.unique()))
    if anchor not in clock or dates[0]<anchor:
        raise ValueError('Invalid cold-start anchor')
    scores = f.pivot(index='decision_date', columns='ticker', values='score').reindex(index=dates, columns=names)
    risk = p.set_index(['decision_date','ticker'])
    anchor_position = int(clock.get_indexer([anchor])[0])
    diagnostics, desired, smoothed_rows = [], [], []
    old = np.full(len(names), np.nan)
    for date in dates:
        index = pd.MultiIndex.from_product([[date], names], names=['decision_date','ticker'])
        meta = risk.reindex(index)
        raw = scores.loc[date].to_numpy(dtype=float)
        beta = meta.beta_spy63.to_numpy(dtype=float)
        available = signal_engine._known(meta).to_numpy()
        mask = available & np.isfinite(raw) & np.isfinite(beta)
        smoothed = raw.copy() if date==dates[0] else .75*np.where(np.isfinite(old),old,raw)+.25*raw
        smoothed[~mask] = np.nan
        old = smoothed.copy()
        signal = signal_engine._rank_signal(smoothed, mask)
        gross = min(float(mask.sum()/signal_engine.STOCK_SLOTS),float(np.abs(signal).sum()))
        projected = signal_engine._base_projection(signal,beta,mask,gross)
        number = int(clock.get_indexer([date])[0])
        scheduled = (number-anchor_position)%cadence==0
        detail = dict(decision_date=date,target_scheduled=scheduled,eligible_count=int(mask.sum()),
            base_gross=float(abs(projected).sum()),base_net=float(projected.sum()),base_estimated_beta=float(projected[mask]@beta[mask]),
            smoothing_alpha=.25,ewma_initialization='cold_first_query_once;no_annual_reset;no_IS_seed',cadence_sessions=cadence,
            cadence_anchor=str(anchor.date()),stock_budget_denominator=signal_engine.STOCK_SLOTS,data_cutoff_exclusive=str(boundary.date()))
        if scheduled:
            if date not in covariances:
                raise ValueError('Missing exact scheduled covariance')
            target, record = signal_engine._risk.allocate_target(pd.Series(projected,index=names),pd.Series(beta,index=names),
                covariances[date],pd.Series(mask,index=names),date=date,vol_target=.12,gross_cap=3.,name_cap=.15,data_cutoff=boundary)
            desired.append(target.rename(date))
            detail.update(record)
        smoothed_rows.append(pd.Series(smoothed,index=names,name=date))
        diagnostics.append(detail)
    targets = pd.DataFrame(desired,columns=names).rename_axis('decision_date')
    audit = pd.DataFrame(diagnostics)
    audit['decision_date'] = pd.to_datetime(audit.decision_date)
    smooth = pd.DataFrame(smoothed_rows,columns=names).rename_axis('decision_date')
    return targets, audit, smooth


def forecasts(model):
    if model not in (ORIGINAL,*MODELS):
        raise ValueError(model)
    directory = SOURCE if model==ORIGINAL else ROOT
    original_manifest = read(SOURCE/f'accounts/{ORIGINAL}_complete.json')
    frames, hashes = [], {}
    for year in range(2007,2024):
        path = directory/f'forecast_runs/{model}_{year}.csv.gz'
        digest = sha(path)
        if model==ORIGINAL:
            assert digest == original_manifest['forecast_hashes'][path.name]
        else:
            meta = read(ROOT/f'forecast_models/{model}_{year}.json')
            assert meta['protocol_sha256'] == current_contract()
            assert digest == meta['forecast_sha256']
        frame = pd.read_csv(path,parse_dates=['decision_date','refit_boundary','max_training_label_end'])
        assert frame.decision_date.dt.year.eq(year).all()
        assert frame.refit_boundary.le(frame.decision_date).all()
        assert frame.max_training_label_end.lt(frame.refit_boundary).all()
        assert frame.decision_date.max()<BOUNDARY and np.isfinite(frame.score).all()
        frames.append(frame)
        hashes[path.name] = digest
    result = pd.concat(frames,ignore_index=True)
    assert not result.duplicated(['decision_date','ticker']).any()
    return result, hashes


def cadence_inputs(loaded, baseline, cadence):
    """Original H5 gate, evaluated at the declared schedule; no candidate labels."""
    scheduled = scheduled_dates(baseline,loaded['clock'],loaded['anchor'],cadence)
    if cadence==5:
        cov = loaded['cov']
    else:
        if 'prices' not in loaded:
            loaded['prices'] = pd.read_pickle(SOURCE/'data/assembled/prices_candidate.pkl')
        cov = build_covariances(loaded['prices'],loaded['names'],scheduled,
            ROOT/f'account_inputs/covariances_cadence{cadence}',loaded['status']['prices_sha256'])
    assert pd.DatetimeIndex(list(cov)).equals(scheduled)
    gates = compute_gate(baseline,loaded['panel'],loaded['clock'],scheduled,end_before='2024-01-01')
    assert gates.index.equals(scheduled) and gates.multiplier.isin([0.,1.]).all()
    ends = pd.to_datetime(gates.history_label_end_max)
    assert ends.dropna().lt(gates.index[ends.notna()]).all()
    if cadence==5:
        old = loaded['gate']
        assert gates.index.equals(old.index)
        np.testing.assert_array_equal(gates.multiplier.to_numpy(),old.multiplier.to_numpy())
        np.testing.assert_allclose(gates.ew_ic.to_numpy(),old.ew_ic.to_numpy(),rtol=0,atol=1e-12)
    path = ROOT/f'account_inputs/original_huber_gate_cadence{cadence}.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    gates.to_csv(path)
    return cov, gates, sha(path)


def prepare(accounts):
    protocol, contract = current_contract(), account_contract()
    loaded = legacy.inputs()
    baseline, baseline_hashes = forecasts(ORIGINAL)
    accounts = list(dict.fromkeys([ORIGINAL,*accounts]))
    caches = {cadence:cadence_inputs(loaded,baseline,cadence) for cadence in sorted({account_spec(a)['cadence'] for a in accounts})}
    directory = ROOT/'account_inputs'
    reports = ROOT/'reports/accounts'
    directory.mkdir(exist_ok=True)
    reports.mkdir(parents=True,exist_ok=True)
    for name in accounts:
        info = account_spec(name)
        model, cadence = info['model'], info['cadence']
        cov, gates, gate_hash = caches[cadence]
        predictions, hashes = (baseline,baseline_hashes) if model==ORIGINAL else forecasts(model)
        complete = directory/f'{name}_prepared.json'
        target_path = directory/f'{name}_desired.pkl'
        smooth_path = directory/f'{model}_smoothed.pkl'
        if complete.exists():
            old = read(complete)
            assert old['protocol_sha256']==protocol and old['account_contract_sha256']==contract
            assert old['forecast_hashes']==hashes and old['source_status']==loaded['status']
            assert old['target_sha256']==sha(target_path) and old['smoothed_sha256']==sha(smooth_path)
            assert old['gate_sha256']==gate_hash
            print('HORIZON_PREPARED_CACHED',name,flush=True)
            continue
        print('HORIZON_PREPARE_BEGIN',name,flush=True)
        started = time.monotonic()
        targets, audit, smooth = build_targets(predictions,loaded['panel'],loaded['clock'],loaded['names'],cov,
            anchor=loaded['anchor'],cadence=cadence)
        assert targets.index.equals(gates.index)
        gated = targets.mul(gates.multiplier,axis=0)
        error = None
        if name==ORIGINAL:
            old = pd.read_pickle(SOURCE/f'account_inputs/{ORIGINAL}_desired.pkl')
            assert old.index.equals(gated.index) and old.columns.equals(gated.columns)
            error = float(np.max(np.abs(old.to_numpy()-gated.to_numpy())))
            assert error<1e-11
        gated.to_pickle(target_path)
        if smooth_path.exists():
            old = pd.read_pickle(smooth_path)
            pd.testing.assert_frame_equal(old,smooth,check_exact=True)
        else:
            smooth.to_pickle(smooth_path)
        audit.to_csv(reports/f'{name}_risk.csv.gz',index=False)
        write(complete,dict(**info,protocol_sha256=protocol,account_contract_sha256=contract,
            forecast_hashes=hashes,target_sha256=sha(target_path),smoothed_sha256=sha(smooth_path),
            source_status=loaded['status'],baseline_target_max_absolute_error=error,
            gate_sha256=gate_hash,gate_file=f'original_huber_gate_cadence{cadence}.csv',
            scheduled_dates=len(gated),seconds=time.monotonic()-started,
            gate_policy='Original Huber H5 efficacy gate at declared cadence; not a candidate/horizon-specific gate'))
        del targets,audit,smooth,gated,predictions
        gc.collect()
        print('HORIZON_PREPARED',name,'seconds',round(time.monotonic()-started,1),'baseline_error',error,flush=True)


def run(accounts):
    protocol, contract = current_contract(), account_contract()
    loaded = legacy.inputs(need_prices=True)
    del loaded['panel']
    gc.collect()
    out = ROOT/'accounts'
    out.mkdir(exist_ok=True)
    baseprep = read(ROOT/f'account_inputs/{ORIGINAL}_prepared.json')
    assert baseprep['baseline_target_max_absolute_error']<1e-11
    for name in accounts:
        info = account_spec(name)
        prepared = read(ROOT/f'account_inputs/{name}_prepared.json')
        assert prepared['protocol_sha256']==protocol and prepared['account_contract_sha256']==contract
        assert prepared['source_status']==loaded['status']
        target_file = ROOT/f'account_inputs/{name}_desired.pkl'
        assert sha(target_file)==prepared['target_sha256']
        assert sha(ROOT/'account_inputs'/prepared['gate_file'])==prepared['gate_sha256']
        complete = out/f'{name}_complete.json'
        if complete.exists():
            old = read(complete)
            assert old['protocol_sha256']==protocol and old['account_contract_sha256']==contract
            assert old['target_sha256']==prepared['target_sha256']
            assert old['ledger_sha256']==sha(out/f'{name}_ledger.csv.gz')
            print('HORIZON_ACCOUNT_CACHED',name,flush=True)
            continue
        if name!=ORIGINAL:
            old = read(out/f'{ORIGINAL}_complete.json')
            assert max(old['baseline_identity_max_errors'].values())<1e-10
        print('HORIZON_ACCOUNT_BEGIN',name,flush=True)
        started = time.monotonic()
        targets = pd.read_pickle(target_file)
        assert targets.index.max()<BOUNDARY and list(targets.columns)==loaded['names']
        noncash = NoncashExposureAudit(pd.read_csv(SOURCE/'data/distribution_audit/spin_census_dispositions.csv'))
        rows = 0
        def consume(frame):
            nonlocal rows
            noncash.consume(frame)
            rows += len(frame)
        ledger,positions,actual,decisions,audit = account_engine.simulate_stock_portfolio(
            loaded['prices'],targets,loaded['events'],
            start=loaded['clock'][loaded['clock'].get_loc(loaded['anchor'])+1],end=loaded['clock'][-1],
            cost_bps=2,buffer_l1=.2,risk_betas=loaded['beta'],neutral=True,dollar_neutral=True,beta_neutral=True,
            eligibility=loaded['eligible'],position_cap=.15,gross_cap=3,gross_guard=3.3,
            data_cutoff='2024-01-01',record_positions=True,position_sink=consume,distribution_claims=loaded['claims'])
        assert positions.empty and rows==audit['streamed_position_rows'] and audit['arithmetic_passed']
        audit['noncash_exposure_audit'] = noncash.report()
        if noncash.rows:
            audit['issues'].extend(noncash.rows)
            audit['validated_performance'] = False
        audit['position_storage'] = 'Every batch passed original noncash audit; no row-level position CSV retained'
        identity = None
        if name==ORIGINAL:
            old = pd.read_csv(SOURCE/f'accounts/{ORIGINAL}_ledger.csv.gz',index_col=0,parse_dates=True)
            assert old.index.equals(ledger.index)
            identity = {column:float(np.max(abs(old[column].to_numpy()-ledger[column].to_numpy())))
                        for column in ('nav','daily_return','turnover','costs','long_net_pnl','short_net_pnl')}
            assert max(identity.values())<1e-10,identity
        for suffix,frame,index in [('ledger',ledger,True),('targets',actual,True),('decisions',decisions,False)]:
            frame.to_csv(out/f'{name}_{suffix}.csv.gz',index=index)
        record = dict(**info,protocol_sha256=protocol,account_contract_sha256=contract,
            target_sha256=prepared['target_sha256'],forecast_hashes=prepared['forecast_hashes'],
            gate_sha256=prepared['gate_sha256'],gate_policy=prepared['gate_policy'],
            audit=audit,baseline_identity_max_errors=identity,ledger_sha256=sha(out/f'{name}_ledger.csv.gz'),
            state='validated' if audit['validated_performance'] else 'provisional; conditional research performance only',
            seconds=time.monotonic()-started,source_status=loaded['status'])
        write(complete,record)
        print('HORIZON_ACCOUNT_DONE',name,round(time.monotonic()-started,1),record['state'],flush=True)
        del ledger,positions,actual,decisions,targets
        gc.collect()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['seal','prepare','run'])
    parser.add_argument('--stage',choices=['weekly','matched','all'],default='weekly')
    parser.add_argument('--accounts',nargs='+',choices=ACCOUNTS)
    args = parser.parse_args()
    if args.action=='seal':
        print(account_contract())
    else:
        (prepare if args.action=='prepare' else run)(args.accounts or selected_accounts(args.stage))
