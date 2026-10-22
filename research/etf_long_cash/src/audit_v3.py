"""Independent chronology/accounting audit for controlled v3 experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "reports/v3_baseline_hashes.json"
INFERENCE_PATH = ROOT / "reports/v3_inference_plan.json"
RUN_DIRECTORY = "v3_1"


def declare_inference_plan():
    if INFERENCE_PATH.exists():
        return json.loads(INFERENCE_PATH.read_text())
    plan = {"declared_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Numerical-resolution amendment before any v3 bootstrap inference; no model, strategy, asset, split or promotion rule changes.",
            "original_experiment_config_sha256": digest(ROOT / "config/experiments_v3.json"),
            "original_replications": 500, "bootstrap_replications": 9999,
            "reason": "With 500 replications and +1 tail correction, minimum p is 1/501, so the smallest possible Holm-adjusted p across 225 tests is about 0.449. 9999 replications give minimum p 0.0001 and permit the declared family-level threshold to be numerically resolved. This change is independent of arm performance.",
            "cost_bps": 5, "block_sessions": 20, "batch_size": 128, "seed": 42,
            "pairing": "Each ETF jointly resamples the same block starts for its frozen baseline and all 15 new variants.",
            "comparisons": 225, "p_value": "Approximate one-sided centered bootstrap with +1 correction; alternative delta Sharpe > 0",
            "multiplicity": "Holm step-down across the complete 225-comparison family",
            "raw_intervals": "Use the same 9999 paired draws for raw two-sided 95% delta-Sharpe and annualized mean-return-difference intervals.",
            "limitations": "Fitted strategies stay fixed. Model fitting, selection and prior exploratory choices are not resampled; adjusted historical p-values are not prospective proof."}
    INFERENCE_PATH.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("V3 inference amendment frozen before bootstrap: 9999 replications, 225 comparisons", flush=True)
    return plan


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_baseline():
    """Record preserved v2 artifacts once, before any new experimental outputs."""
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text())
    paths = sorted(path for path in (ROOT / "runs/v2").rglob("*") if path.is_file())
    paths += [ROOT / "src/research.py", ROOT / "reports/metrics.csv",
              ROOT / "reports/sharpe_comparison.csv", ROOT / "reports/result_audit.json",
              ROOT / "reports/benchmark_diagnostics.csv", ROOT / "research_notebook_v2.ipynb",
              ROOT / "reports/research_report_v2.html"]
    manifest = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "Frozen v2 source, predictions, models, ledgers, metrics and preserved deliverables; new experiments must not modify these.",
                "files": {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}}
    BASELINE_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen v2 baseline: {len(manifest['files'])} hashes", flush=True)
    return manifest


def baseline_changes():
    manifest = json.loads(BASELINE_PATH.read_text())
    return [relative for relative, expected in manifest["files"].items()
            if not (ROOT / relative).is_file() or digest(ROOT / relative) != expected]


def holm_adjust(p_values):
    """Holm step-down adjustment across the complete declared comparison family."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values))
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(p_values) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted


def sampled_block_moments(values, starts, block=20):
    """Exact resampled means/sample variances without materializing session draws.

    ``values`` is variants by sessions and ``starts`` is draws by block count.
    The final block is shortened to preserve the original session count.
    """
    values = np.asarray(values, dtype=float)
    starts = np.asarray(starts, dtype=int)
    n = values.shape[1]
    full, remainder = divmod(n, block)
    sums = np.concatenate([np.zeros((len(values), 1)), values.cumsum(axis=1)], axis=1)
    squares = np.concatenate([np.zeros((len(values), 1)), np.square(values).cumsum(axis=1)], axis=1)
    total = np.zeros((len(values), len(starts)))
    total_square = np.zeros_like(total)
    if full:
        block_sums = sums[:, block:] - sums[:, :-block]
        block_squares = squares[:, block:] - squares[:, :-block]
        total += block_sums[:, starts[:, :full]].sum(axis=2)
        total_square += block_squares[:, starts[:, :full]].sum(axis=2)
    if remainder:
        last = starts[:, -1]
        total += sums[:, last + remainder] - sums[:, last]
        total_square += squares[:, last + remainder] - squares[:, last]
    means = total / n
    variances = np.maximum(0., (total_square - np.square(total) / n) / (n - 1))
    return means, variances


def joint_paired_bootstrap(returns, baseline="baseline", seed=42, replications=500, block=20):
    """Keep paired sessions and all variants together in every block resample.

    The centered-bootstrap tail is an approximate one-sided descriptive test.
    It holds fitted strategies fixed, so it does not reproduce model selection
    or the complete exploratory research process.
    """
    if baseline not in returns or returns.isna().any().any():
        raise ValueError("Complete aligned baseline and variant returns required")
    values = returns.to_numpy(dtype=float).T
    if not np.isfinite(values).all() or values.shape[1] < 2:
        raise ValueError("Finite aligned return histories required")

    def sharpes(matrix):
        sd = matrix.std(axis=1, ddof=1)
        return np.divide(matrix.mean(axis=1) * np.sqrt(252), sd,
                         out=np.zeros(len(matrix)), where=sd > 0)

    observed = sharpes(values)
    base_index = list(returns.columns).index(baseline)
    delta = observed - observed[base_index]
    rng = np.random.default_rng(seed)
    n = values.shape[1]
    draws = np.empty((replications, len(values)))
    return_difference_draws = np.empty_like(draws)
    for first in range(0, replications, 128):
        count = min(128, replications - first)
        starts = rng.integers(0, max(1, n - block + 1), size=(count, int(np.ceil(n / block))))
        means, variances = sampled_block_moments(values, starts, block)
        sd = np.sqrt(variances)
        ss = np.divide(means * np.sqrt(252), sd, out=np.zeros_like(means), where=sd > 0)
        draws[first:first + count] = (ss - ss[base_index]).T
        return_difference_draws[first:first + count] = ((means - means[base_index]) * 252).T
    rows = []
    for index, variant in enumerate(returns.columns):
        if variant == baseline:
            continue
        low, high = np.quantile(draws[:, index], [.025, .975])
        rlow, rhigh = np.quantile(return_difference_draws[:, index], [.025, .975])
        centered = draws[:, index] - delta[index]
        p_value = (1 + np.count_nonzero(centered >= delta[index] - 1e-12)) / (replications + 1)
        rows.append({"variant": variant, "baseline_variant": baseline,
                     "variant_sharpe": float(observed[index]), "baseline_sharpe": float(observed[base_index]),
                     "delta_sharpe": float(delta[index]), "delta_sharpe_95pct_low": float(low),
                     "delta_sharpe_95pct_high": float(high),
                     "p_centered_bootstrap_one_sided_approx": float(p_value),
                     "annualized_mean_daily_return_difference": float((values[index] - values[base_index]).mean() * 252),
                     "annualized_mean_daily_return_difference_95pct_low": float(rlow),
                     "annualized_mean_daily_return_difference_95pct_high": float(rhigh),
                     "bootstrap_replications": replications, "block_sessions": block, "seed": seed,
                     "uncertainty_caveat": "Exploratory historical comparison of fixed fitted strategies. Joint paired blocks retain cross-arm pairing; fits/selection are not replayed. Raw intervals are unadjusted. Approximate centered-bootstrap p-values and Holm adjustment do not establish prospective significance."})
    return rows


def experiment_statistics():
    plan = json.loads((ROOT / "config/experiments_v3.json").read_text())
    inference = json.loads(INFERENCE_PATH.read_text())
    rows = []
    for ticker in plan["universe"]:
        series = []
        reference_index = None
        for variant in plan["variants"]:
            path = ROOT / "runs" / RUN_DIRECTORY / ticker / f"ledger_{variant}_5bps.csv"
            ledger = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
            if reference_index is None:
                reference_index = ledger.index
            elif not ledger.index.equals(reference_index):
                raise ValueError(f"Unaligned account dates: {ticker}/{variant}")
            series.append(ledger.daily_return.rename(variant))
        rr = pd.concat(series, axis=1)
        for result in joint_paired_bootstrap(rr, replications=inference["bootstrap_replications"]):
            rows.append({"ticker": ticker, "start": str(rr.index[0].date()),
                         "end": str(rr.index[-1].date()), "days": len(rr), **result})
        print(f"Joint paired experiment bootstrap: {ticker}", flush=True)
    frame = pd.DataFrame(rows)
    expected = len(plan["universe"]) * (len(plan["variants"]) - 1)
    if len(frame) != expected:
        raise RuntimeError("Incomplete declared multiplicity family")
    frame["p_holm"] = holm_adjust(frame.p_centered_bootstrap_one_sided_approx)
    frame["multiplicity_family_size"] = expected
    frame["multiplicity_family"] = "Every declared new variant versus frozen v2 control, across all 15 ETFs; no subset selected by results"
    frame.to_csv(ROOT / "reports/v3_uncertainty.csv", index=False)
    frame[["ticker", "variant", "delta_sharpe", "p_centered_bootstrap_one_sided_approx", "p_holm", "multiplicity_family_size", "multiplicity_family", "uncertainty_caveat"]].to_csv(ROOT / "reports/v3_multiplicity.csv", index=False)
    return frame


def audit_results(allow_partial=False):
    import research
    plan_path = ROOT / "config/experiments_v3.json"
    plan = json.loads(plan_path.read_text())
    reserve = pd.Timestamp(plan["reserved_start"])
    metrics = pd.read_csv(ROOT / "reports/v3_metrics.csv")
    old_metrics = pd.read_csv(ROOT / "reports/metrics.csv")
    prices = research.load_prices()
    checks = []

    def check(name, passed, detail=None):
        item = {"check": name, "passed": bool(passed)}
        if detail is not None:
            item["detail"] = detail
        checks.append(item)

    changes = baseline_changes()
    check("all_frozen_v2_hashes_unchanged", not changes, changes)
    check("frozen_v3_config_snapshot_matches", digest(ROOT / "runs" / RUN_DIRECTORY / "experiment_config.json") == digest(plan_path))
    check("frozen_v3_source_snapshot_matches", digest(ROOT / "runs" / RUN_DIRECTORY / "source_snapshot.py") == digest(ROOT / "src/experiments_v3.py"))
    expected_rows = len(plan["universe"]) * len(plan["variants"]) * len(plan["cost_bps"])
    if not allow_partial:
        check("all_predeclared_variant_metrics_present", len(metrics) == expected_rows and set(metrics.variant) == set(plan["variants"]) and set(metrics.ticker) == set(plan["universe"]), {"expected": expected_rows, "actual": len(metrics)})
    check("no_duplicate_metrics", not metrics.duplicated(["ticker", "variant", "cost_bps"]).any())
    counts = []
    missing = []
    for ticker in plan["universe"]:
        folder = ROOT / "runs" / RUN_DIRECTORY / ticker
        if not (folder / "completed.json").exists():
            missing.append(ticker)
            if allow_partial:
                continue
            raise RuntimeError(f"Incomplete v3 instrument {ticker}")
        old_folder = ROOT / "runs/v2" / ticker
        original = pd.read_csv(old_folder / "forecasts.csv", parse_dates=["decision_date", "entry_date"])
        original = original.loc[original.family.eq("CART")].sort_values("decision_date")
        original_dates = original.decision_date.tolist()
        old_engineering = pd.read_csv(ROOT / "runs/v3" / ticker / "metrics.csv", float_precision="round_trip")
        corrected_metrics = pd.read_csv(folder / "metrics.csv", float_precision="round_trip")
        prior_non_ewma = old_engineering.loc[~old_engineering.variant.eq("ewma_vol")].sort_values(["variant", "cost_bps"]).reset_index(drop=True)
        current_non_ewma = corrected_metrics.loc[~corrected_metrics.variant.eq("ewma_vol")].sort_values(["variant", "cost_bps"]).reset_index(drop=True)
        check(f"{ticker}:all_non_EWMA_metrics_exactly_match_preserved_engineering_run", current_non_ewma.equals(prior_non_ewma))
        # Exact float round trips matter when a repeated leaf prediction equals
        # an OOF regression-quantile edge stored separately in JSON.
        forecasts = pd.read_csv(folder / "forecasts.csv", float_precision="round_trip")
        for column in ["decision_date", "entry_date", "fit_cutoff", "train_latest_label", "calibration_cutoff", "calibration_latest_label", "label_end", "label_end20"]:
            if column in forecasts:
                forecasts[column] = pd.to_datetime(forecasts[column])
        check(f"{ticker}:all_variant_forecasts_present", set(forecasts.variant) == set(plan["variants"]))
        check(f"{ticker}:no_duplicate_variant_decision", not forecasts.duplicated(["variant", "decision_date"]).any())
        check(f"{ticker}:no_reserved_forecast_or_execution", (forecasts.decision_date < reserve).all() and (forecasts.entry_date < reserve).all())
        check(f"{ticker}:next_open_after_completed_decision", (forecasts.entry_date > forecasts.decision_date).all())
        check(f"{ticker}:forecast_weights_long_cash", forecasts.target_weight.between(0, 1).all())
        ewma = forecasts.loc[forecasts.variant.eq("ewma_vol")]
        expected_ewma = ewma.base_signal.astype(float) * np.minimum(1., .1 / ewma.ewma_vol.clip(lower=.05))
        check(f"{ticker}:EWMA_variant_uses_its_recorded_EWMA_risk_estimate", np.allclose(ewma.target_weight, expected_ewma, rtol=1e-12, atol=1e-12))
        fitted = forecasts.loc[forecasts.uses_fitted_model]
        check(f"{ticker}:model_training_strictly_matured", (fitted.train_latest_label < fitted.fit_cutoff).all() and (fitted.train_rows >= 104).all() and (fitted.fit_cutoff <= fitted.decision_date).all())
        payoff = forecasts.loc[forecasts.variant.isin(["payoff_only", "payoff_context", "payoff_buffer"]) & forecasts.expected_payoff.notna()]
        check(f"{ticker}:payoff_calibration_strictly_matured", (payoff.calibration_latest_label < payoff.calibration_cutoff).all() and (payoff.oof_rows >= 52).all())
        model_hashes = {digest(path) for path in folder.glob("*.pkl")}
        check(f"{ticker}:all_fitted_forecast_model_hashes_resolve", fitted.model_sha256.isin(model_hashes).all())
        for variant in plan["variants"]:
            ff = forecasts.loc[forecasts.variant.eq(variant)].sort_values("decision_date")
            check(f"{ticker}:{variant}:forecast_dates_match_v2", ff.decision_date.tolist() == original_dates and ff.entry_date.tolist() == original.entry_date.tolist())
            for bps in plan["cost_bps"]:
                path = folder / f"ledger_{variant}_{bps}bps.csv"
                ledger = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
                old = pd.read_csv(old_folder / f"ledger_CART_selected_{bps}bps.csv", index_col="Date", parse_dates=["Date"])
                stamp = f"{ticker}:{variant}:{bps}bps"
                check(stamp + ":dates_match_frozen_v2", ledger.index.equals(old.index))
                check(stamp + ":no_reserved_account_dates", (ledger.index < reserve).all())
                check(stamp + ":positive_NAV_and_long_cash_bounds", (np.isfinite(ledger.nav) & (ledger.nav > 0)).all() and (ledger.cash >= -1e-10).all() and ledger.weight.between(-1e-10, 1 + 1e-10).all() and (ledger.shares >= -1e-10).all())
                underlying = prices[ticker].adj_close.reindex(ledger.index)
                check(stamp + ":cash_units_conserve_NAV", np.allclose(ledger.cash + ledger.shares * underlying, ledger.nav, rtol=1e-9, atol=1e-10))
                implied = ledger.nav.pct_change()
                implied.iloc[0] = ledger.nav.iloc[0] - 1
                check(stamp + ":daily_returns_reconcile", np.allclose(implied, ledger.daily_return, rtol=1e-9, atol=1e-11))
                check(stamp + ":actual_notional_costs", np.allclose(ledger.costs, ledger.trade_value.abs() * bps / 10000 + ledger.terminal_cost, rtol=1e-9, atol=1e-11))
                if variant == "baseline":
                    check(stamp + ":exact_original_control_replay", np.allclose(ledger.nav, old.nav, rtol=1e-9, atol=1e-10) and np.allclose(ledger.daily_return, old.daily_return, rtol=1e-9, atol=1e-10))
                mm = metrics.loc[metrics.ticker.eq(ticker) & metrics.variant.eq(variant) & metrics.cost_bps.eq(bps)]
                observed = research.metrics(ledger)
                check(stamp + ":summary_recomputes", len(mm) == 1 and all(np.isclose(mm.iloc[0][key], observed[key], rtol=1e-8, atol=1e-10) for key in ["total_return", "cagr", "sharpe", "volatility", "max_drawdown", "average_exposure"]))
        oof = pd.read_csv(folder / "calibration_oof.csv", float_precision="round_trip")
        for column in ["decision_date", "entry_date", "label_end", "label_end20", "fit_cutoff", "calibration_cutoff", "oof_train_latest_label"]:
            oof[column] = pd.to_datetime(oof[column])
        check(f"{ticker}:calibration_predictions_from_earlier_fits", ((oof.fit_cutoff <= oof.decision_date) & (oof.fit_cutoff < oof.calibration_cutoff)).all())
        check(f"{ticker}:calibration_outcomes_and_20gate_strictly_matured", ((oof.label_end < oof.calibration_cutoff) & (oof.label_end20 < oof.calibration_cutoff)).all())
        check(f"{ticker}:calibration_never_uses_reserved_dates", all((oof[column] < reserve).all() for column in ["decision_date", "entry_date", "label_end", "label_end20"]))
        check(f"{ticker}:calibration_predictions_and_outcomes_finite", np.isfinite(oof.prediction).all() and np.isfinite(oof.outcome).all())
        check(f"{ticker}:OOF_training_labels_matured_and_minimum_rows", (oof.oof_train_latest_label < oof.fit_cutoff).all() and (oof.oof_train_rows >= 104).all())
        calibrators = json.loads((folder / "calibrators.json").read_text())
        calibration_replay_ok = True
        forecast_payoff_ok = True
        for record in calibrators:
            data = oof.loc[oof.outer_year.eq(record["outer_year"]) & oof.context.eq(record["context"])]
            cal = record["calibrator"]
            calibration_replay_ok &= cal["n"] == len(data)
            if not cal["eligible"]:
                calibration_replay_ok &= len(data) < plan["calibration"]["min_matured_oof_rows"]
                continue
            calibration_replay_ok &= len(data) >= plan["calibration"]["min_matured_oof_rows"]
            calibration_replay_ok &= np.isclose(cal["global_mean"], data.outcome.mean(), atol=1e-12)
            buckets = np.searchsorted(np.asarray(cal["edges"]), data.prediction.to_numpy(), side="right")
            pseudo = plan["calibration"]["pseudo_observations"]
            for bin_index, bin_record in enumerate(cal["bins"]):
                values = data.outcome.to_numpy()[buckets == bin_index]
                expected_mean = (values.sum() + pseudo * data.outcome.mean()) / (len(values) + pseudo)
                calibration_replay_ok &= bin_record["n"] == len(values) and np.isclose(bin_record["expected_payoff"], expected_mean, atol=1e-12)
            variants = ["payoff_context"] if record["context"] else ["payoff_only", "payoff_buffer"]
            ff = forecasts.loc[forecasts.outer_year.eq(record["outer_year"]) & forecasts.variant.isin(variants)]
            indices = np.searchsorted(np.asarray(cal["edges"]), ff.prediction.to_numpy(), side="right")
            expected = np.asarray([cal["bins"][index]["expected_payoff"] for index in indices])
            forecast_payoff_ok &= np.allclose(ff.expected_payoff, expected, atol=1e-12)
        check(f"{ticker}:calibrator_bin_counts_and_shrinkage_replay", calibration_replay_ok)
        check(f"{ticker}:recorded_payoff_forecasts_match_prior_calibrator", forecast_payoff_ok)
        counts.append({"ticker": ticker, "forecast_rows": len(forecasts), "calibration_oof_rows": len(oof), "variants": len(plan["variants"]), "first_decision": str(forecasts.decision_date.min().date()), "last_decision": str(forecasts.decision_date.max().date())})
    if not allow_partial:
        stats = pd.read_csv(ROOT / "reports/v3_uncertainty.csv")
        expected_comparisons = len(plan["universe"]) * (len(plan["variants"]) - 1)
        check("full_joint_bootstrap_family_present", len(stats) == expected_comparisons and stats.multiplicity_family_size.eq(expected_comparisons).all())
        check("holm_adjustment_reproduces_full_family", np.allclose(stats.p_holm, holm_adjust(stats.p_centered_bootstrap_one_sided_approx), atol=1e-12))
        inference = json.loads(INFERENCE_PATH.read_text())
        check("bootstrap_replications_match_frozen_resolution_amendment", stats.bootstrap_replications.eq(inference["bootstrap_replications"]).all())
        check("inference_amendment_matches_original_frozen_config", inference["original_experiment_config_sha256"] == digest(plan_path))
    violations = [item for item in checks if not item["passed"]]
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "FAIL" if violations else "PARTIAL" if missing else "PASS",
              "checks_passed": len(checks) - len(violations), "checks_total": len(checks), "violations": violations,
              "reserved_start": str(reserve.date()), "reserved_scored": False,
              "v2_baseline_files_checked": len(json.loads(BASELINE_PATH.read_text())["files"]), "v2_changed_files": changes,
              "experiment_config_sha256": digest(plan_path), "instruments": counts, "checks": checks,
              "run_directory": f"runs/{RUN_DIRECTORY}",
              "interpretation": "Engineering and chronology consistency only. Inspected historical experiments and fixed-fit bootstrap comparisons do not establish prospective significance or resolve vendor corporate-action limitations."}
    (ROOT / "reports/v3_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"V3 audit {report['status']}: {report['checks_passed']}/{len(checks)} checks", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-baseline", action="store_true")
    parser.add_argument("--statistics", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--declare-inference-plan", action="store_true")
    args = parser.parse_args()
    if args.declare_inference_plan:
        declare_inference_plan()
    elif args.capture_baseline:
        capture_baseline()
    else:
        if args.statistics:
            experiment_statistics()
        result = audit_results(allow_partial=args.allow_partial)
        if result["status"] == "FAIL":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
