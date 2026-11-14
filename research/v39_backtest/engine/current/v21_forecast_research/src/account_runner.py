"""Isolated v21 forecast-to-account bridge; prior research is read-only.

All challengers use the saved Huber efficacy gate. The original allocator and
full accounting engine are reused without modification. Position batches are
audited in memory; ledgers, trades, targets, and audit findings are retained.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import datetime as dt
import gc
import hashlib
import json
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(str(Path(__file__).resolve().parents[3] / 'prior/v17_long_history'))
sys.path.insert(0, str(SOURCE / "src"))
import signal_engine
import account_engine
from lazy_covariances import LazyCovariances
from audit_noncash_exposure import NoncashExposureAudit

REFERENCE = "huber_ridge_10y"
ARMS = ("sector_huber", "median_logistic", "equal_ensemble")
BOUNDARY = pd.Timestamp("2024-01-01")
CODE = ("account_engine.py", "dated_quote_cache.py", "distribution_claims.py",
        "signal_engine.py", "risk_engine.py", "fast_gate.py", "lazy_covariances.py",
        "audit_noncash_exposure.py", "forecast_stage_attribution.py")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def current_contract():
    path = ROOT / "protocol.json"
    if not path.exists():
        raise ValueError("Freeze protocol.json before account preparation")
    seal = path.with_suffix(".sha256")
    if seal.exists():
        assert sha(path) == seal.read_text().strip()
    return sha(path)


def sources():
    approval_path = SOURCE / "data/assembled/approval.json"
    status = read(approval_path)
    assert status["source_qualification_passed"] and not status.get("blockers")
    baseline = read(SOURCE / f"accounts/{REFERENCE}_complete.json")
    files = {
        "panel_sha256": SOURCE / "data/candidate_features/panel.pkl",
        "prices_sha256": SOURCE / "data/assembled/prices_candidate.pkl",
        "events_sha256": SOURCE / "data/assembled/events.csv",
        "claims_sha256": SOURCE / "data/distribution_audit/delayed_claims.json",
        "noncash_census_sha256": SOURCE / "data/distribution_audit/spin_census_dispositions.csv",
    }
    for key, file in files.items():
        assert sha(file) == status[key], ("Source changed", key)
        if key in baseline:
            assert baseline[key] == status[key]
    status["source_approval_sha256"] = sha(approval_path)
    status["account_code_hashes"] = {name: sha(SOURCE / "src" / name) for name in CODE}
    for key in ("source_approval_sha256", "account_code_hashes"):
        assert status[key] == baseline[key], key
    status["baseline_gate_sha256"] = sha(SOURCE / f"reports/accounts/{REFERENCE}_gate.csv")
    status["baseline_desired_sha256"] = sha(SOURCE / f"account_inputs/{REFERENCE}_desired.pkl")
    status["baseline_ledger_sha256"] = sha(SOURCE / f"accounts/{REFERENCE}_ledger.csv.gz")
    assert status["baseline_ledger_sha256"] == baseline["ledger_sha256"]
    status["covariance_manifest_sha256"] = sha(SOURCE / "account_inputs/covariances/manifest.json")
    return status


def inputs(*, need_prices=False):
    """Verify existing artifacts; never invoke the source cache-writing loader."""
    status = sources()
    panel = pd.read_pickle(SOURCE / "data/candidate_features/panel.pkl")
    assert panel.decision_date.max() < BOUNDARY
    clock = pd.DatetimeIndex(pd.read_csv(SOURCE / "data/assembled/calendar.csv", parse_dates=["Date"]).Date)
    assert clock.is_unique and clock.is_monotonic_increasing and clock.max() < BOUNDARY
    names = sorted(panel.loc[panel.model_eligible, "ticker"].unique())
    beta = pd.read_pickle(SOURCE / "account_inputs/betas.pkl")
    eligible = pd.read_pickle(SOURCE / "account_inputs/eligible.pkl")
    assert list(beta.columns) == names == list(eligible.columns)
    assert beta.index.equals(clock) and eligible.index.equals(clock)
    expected_beta = panel.pivot(index="decision_date", columns="ticker", values="beta_spy63").reindex(index=clock, columns=names)
    expected_eligible = panel.pivot(index="decision_date", columns="ticker", values="model_eligible").reindex(index=clock, columns=names).fillna(False).astype(bool)
    assert np.array_equal(beta.to_numpy(), expected_beta.to_numpy(), equal_nan=True)
    assert np.array_equal(eligible.to_numpy(), expected_eligible.to_numpy())
    del expected_beta, expected_eligible
    cov = LazyCovariances(SOURCE / "account_inputs/covariances")
    assert list(cov.names) == names and cov.manifest["source_hash"] == status["prices_sha256"]
    assert cov.manifest["boundary"] == "2024-01-01"
    dates = pd.DatetimeIndex(sorted(panel.loc[panel.model_eligible & panel.decision_date.ge("2007-01-01"), "decision_date"].unique()))
    anchor = dates[0]
    scheduled = dates[(clock.get_indexer(dates) - clock.get_loc(anchor)) % 5 == 0]
    assert scheduled.equals(cov.dates)
    gate = pd.read_csv(SOURCE / f"reports/accounts/{REFERENCE}_gate.csv", index_col=0, parse_dates=True)
    assert gate.index.equals(scheduled)
    assert gate.data_cutoff_exclusive.eq("2024-01-01").all()
    label_ends = pd.to_datetime(gate.history_label_end_max)
    assert label_ends.dropna().lt(gate.index[label_ends.notna()]).all()
    result = dict(panel=panel, clock=clock, names=names, beta=beta, eligible=eligible,
                  cov=cov, anchor=anchor, gate=gate, status=status)
    if need_prices:
        result["prices"] = pd.read_pickle(SOURCE / "data/assembled/prices_candidate.pkl")
        result["events"] = pd.read_csv(SOURCE / "data/assembled/events.csv", parse_dates=["effective_date"])
        result["claims"] = read(SOURCE / "data/distribution_audit/delayed_claims.json")
    return result


def forecasts(arm):
    if arm not in (*ARMS, REFERENCE):
        raise ValueError(arm)
    directory = SOURCE if arm == REFERENCE else ROOT
    files = [directory / f"forecast_runs/{arm}_{year}.csv.gz" for year in range(2007, 2024)]
    hashes = {file.name: sha(file) for file in files}
    if arm == REFERENCE:
        assert hashes == read(SOURCE / f"accounts/{REFERENCE}_complete.json")["forecast_hashes"]
    frames = []
    for year, file in zip(range(2007, 2024), files):
        frame = pd.read_csv(file, parse_dates=["decision_date", "refit_boundary", "max_training_label_end"])
        assert frame.decision_date.dt.year.eq(year).all()
        assert frame.refit_boundary.le(frame.decision_date).all()
        assert frame.max_training_label_end.lt(frame.refit_boundary).all()
        assert frame.decision_date.max() < BOUNDARY and np.isfinite(frame.score.to_numpy()).all()
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    assert not combined.duplicated(["decision_date", "ticker"]).any()
    return combined, hashes


def prepare(arms):
    contract = current_contract()
    loaded = inputs()
    directory = ROOT / "account_inputs"
    directory.mkdir(exist_ok=True)
    reports = ROOT / "reports/accounts"
    reports.mkdir(parents=True, exist_ok=True)
    for arm in dict.fromkeys([REFERENCE, *arms]):
        print("V21_PREPARE_BEGIN", arm, flush=True)
        start = time.monotonic()
        predictions, hashes = forecasts(arm)
        complete = directory / f"{arm}_prepared.json"
        if complete.exists():
            previous = read(complete)
            assert previous["protocol_sha256"] == contract
            assert previous["forecast_hashes"] == hashes
            assert previous["target_sha256"] == sha(directory / f"{arm}_desired.pkl")
            assert previous["source_status"] == loaded["status"]
            print("V21_PREPARE_CACHED", arm, flush=True)
            continue
        targets, diagnostics, stages = signal_engine.build_weekly_targets(
            predictions, loaded["panel"], loaded["clock"], loaded["names"], loaded["cov"],
            anchor=loaded["anchor"], end_before="2024-01-01")
        assert targets.index.equals(loaded["gate"].index)
        gated = targets.mul(loaded["gate"].multiplier, axis=0)
        baseline_error = None
        if arm == REFERENCE:
            saved = pd.read_pickle(SOURCE / f"account_inputs/{REFERENCE}_desired.pkl")
            assert saved.index.equals(gated.index) and saved.columns.equals(gated.columns)
            baseline_error = float(np.max(np.abs(saved.to_numpy() - gated.to_numpy())))
            assert baseline_error < 1e-11, baseline_error
        gated.to_pickle(directory / f"{arm}_desired.pkl")
        stages["ewma_score"].to_pickle(directory / f"{arm}_smoothed.pkl")
        diagnostics.to_csv(reports / f"{arm}_risk.csv.gz", index=False)
        loaded["gate"].to_csv(reports / f"{arm}_gate.csv")
        write(complete, dict(arm=arm, protocol_sha256=contract, forecast_hashes=hashes,
             target_sha256=sha(directory / f"{arm}_desired.pkl"),
             smoothed_sha256=sha(directory / f"{arm}_smoothed.pkl"), source_status=loaded["status"],
             baseline_target_max_absolute_error=baseline_error, scheduled_dates=len(gated),
             gate_policy="Frozen saved Huber10 gate shared by all challengers; not candidate-reestimated",
             runner_sha256=sha(__file__), seconds=time.monotonic()-start,
             created_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        del predictions, targets, diagnostics, stages, gated
        gc.collect()
        print("V21_PREPARED", arm, round(time.monotonic()-start, 1), "baseline_error", baseline_error, flush=True)


def run(arms):
    contract = current_contract()
    loaded = inputs(need_prices=True)
    # Forecast/feature arrays are unnecessary during a full account replay.
    del loaded["panel"]
    gc.collect()
    out = ROOT / "accounts"
    out.mkdir(exist_ok=True)
    baseprep = read(ROOT / f"account_inputs/{REFERENCE}_prepared.json")
    assert baseprep["baseline_target_max_absolute_error"] < 1e-11
    for arm in arms:
        if arm not in (*ARMS, REFERENCE):
            raise ValueError(arm)
        prepared = read(ROOT / f"account_inputs/{arm}_prepared.json")
        assert prepared["protocol_sha256"] == contract and prepared["source_status"] == loaded["status"]
        target_file = ROOT / f"account_inputs/{arm}_desired.pkl"
        assert sha(target_file) == prepared["target_sha256"]
        complete = out / f"{arm}_complete.json"
        if complete.exists():
            saved = read(complete)
            assert saved["target_sha256"] == prepared["target_sha256"]
            assert saved["protocol_sha256"] == contract
            assert saved["ledger_sha256"] == sha(out / f"{arm}_ledger.csv.gz")
            print("V21_ACCOUNT_CACHED", arm, flush=True)
            continue
        print("V21_ACCOUNT_BEGIN", arm, flush=True)
        start = time.monotonic()
        targets = pd.read_pickle(target_file)
        assert targets.index.max() < BOUNDARY and list(targets.columns) == loaded["names"]
        noncash = NoncashExposureAudit(pd.read_csv(SOURCE / "data/distribution_audit/spin_census_dispositions.csv"))
        written = 0
        def audit_positions(frame):
            nonlocal written
            noncash.consume(frame)
            written += len(frame)
        ledger, positions, actual, decisions, audit = account_engine.simulate_stock_portfolio(
            loaded["prices"], targets, loaded["events"],
            start=loaded["clock"][loaded["clock"].get_loc(loaded["anchor"])+1], end=loaded["clock"][-1],
            cost_bps=2, buffer_l1=.2, risk_betas=loaded["beta"], neutral=True, dollar_neutral=True,
            beta_neutral=True, eligibility=loaded["eligible"], position_cap=.15, gross_cap=3,
            gross_guard=3.3, data_cutoff="2024-01-01", record_positions=True,
            position_sink=audit_positions, distribution_claims=loaded["claims"])
        assert positions.empty and written == audit["streamed_position_rows"]
        assert audit["arithmetic_passed"]
        audit["noncash_exposure_audit"] = noncash.report()
        if noncash.rows:
            audit["issues"].extend(noncash.rows)
            audit["validated_performance"] = False
        audit["position_storage"] = "All position batches consumed by original noncash audit; row-level positions not serialized"
        identity = None
        if arm == REFERENCE:
            saved = pd.read_csv(SOURCE / f"accounts/{REFERENCE}_ledger.csv.gz", index_col=0, parse_dates=True)
            assert saved.index.equals(ledger.index)
            identity = {}
            for column in ("nav", "daily_return", "turnover", "costs", "long_net_pnl", "short_net_pnl"):
                identity[column] = float(np.max(np.abs(saved[column].to_numpy()-ledger[column].to_numpy())))
                assert identity[column] < 1e-10, (column, identity[column])
        for suffix, frame, index in (("ledger", ledger, True), ("targets", actual, True), ("decisions", decisions, False)):
            frame.to_csv(out / f"{arm}_{suffix}.csv.gz", index=index)
        source = loaded["status"]
        record = dict(arm=arm, protocol_sha256=contract, target_sha256=prepared["target_sha256"],
             forecast_hashes=prepared["forecast_hashes"], audit=audit, baseline_identity_max_errors=identity,
             ledger_sha256=sha(out / f"{arm}_ledger.csv.gz"),
             state="validated" if audit["validated_performance"] else "provisional; conditional research performance only",
             gate_policy=prepared["gate_policy"], baseline_gate_sha256=source["baseline_gate_sha256"],
             allocation="Unchanged original next-close weekly EWMA .25; 12% target; 2bps one-way; buffer .2; gross3; cap.15; no borrow/financing interest",
             runner_sha256=sha(__file__), seconds=time.monotonic()-start,
             created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        for key in ("source_approval_sha256", "prices_sha256", "events_sha256", "claims_sha256", "noncash_census_sha256", "account_code_hashes"):
            record[key] = source[key]
        write(complete, record)
        print("V21_ACCOUNT_DONE", arm, round(time.monotonic()-start, 1), record["state"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--arms", nargs="+", choices=(*ARMS, REFERENCE), default=list(ARMS))
    args = parser.parse_args()
    (prepare if args.action == "prepare" else run)(args.arms)
