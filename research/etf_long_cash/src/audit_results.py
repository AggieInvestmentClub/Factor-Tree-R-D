"""Audit saved historical research and run a bounded prior-fold family comparison.

This reads existing local snapshots; it never downloads data, changes forecasts,
promotes a learner, opens a trading session, or examines another project.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

import research

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs" / "v2"
FAMILIES = ["CART", "RF", "XGB"]
XLF_SOURCE = "https://www.sec.gov/Archives/edgar/data/1064641/000119312516699787/d246445d497.htm"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_events(path):
    frame = pd.read_csv(path, index_col="decision_date", parse_dates=["decision_date"])
    for column in ["entry_date", "label_end_5", "label_end_10", "label_end_20"]:
        frame[column] = pd.to_datetime(frame[column])
    return frame


def paired_sharpe_comparison(ticker, primary, buy_hold, seed=42, replications=500, block=20):
    """Paired moving-block intervals for descriptive historical differences."""
    aligned = pd.concat([primary.daily_return.rename("primary"),
                         buy_hold.daily_return.rename("buy_hold")], axis=1).dropna()
    a = aligned.primary.to_numpy()
    b = aligned.buy_hold.to_numpy()

    def sharpe(values):
        sd = values.std(ddof=1)
        return float(values.mean() / sd * np.sqrt(252)) if sd > 0 else 0.

    rng = np.random.default_rng(seed)
    n = len(a)
    delta_sharpes = []
    mean_return_differences = []
    for _ in range(replications):
        starts = rng.integers(0, max(1, n - block + 1), size=int(np.ceil(n / block)))
        indices = np.concatenate([np.arange(start, min(start + block, n)) for start in starts])[:n]
        aa, bb = a[indices], b[indices]
        delta_sharpes.append(sharpe(aa) - sharpe(bb))
        mean_return_differences.append(float((aa - bb).mean() * 252))
    sharpe_ci = np.quantile(delta_sharpes, [.025, .975])
    return_ci = np.quantile(mean_return_differences, [.025, .975])
    return {"ticker": ticker, "primary_strategy": "CART_selected_5bps", "benchmark": "ETF_buy_hold_5bps",
            "start": str(aligned.index[0].date()), "end": str(aligned.index[-1].date()), "days": n,
            "primary_sharpe": sharpe(a), "buy_hold_sharpe": sharpe(b),
            "delta_sharpe": sharpe(a) - sharpe(b),
            "delta_sharpe_95pct_low": float(sharpe_ci[0]), "delta_sharpe_95pct_high": float(sharpe_ci[1]),
            "annualized_mean_daily_return_difference": float((a - b).mean() * 252),
            "annualized_mean_daily_return_difference_95pct_low": float(return_ci[0]),
            "annualized_mean_daily_return_difference_95pct_high": float(return_ci[1]),
            "primary_measure": "net_daily_Sharpe_difference_vs_ETF_buy_hold",
            "sharpe_convention": "sqrt(252)*mean(daily net simple return)/sample_std(daily net simple return); zero cash and risk-free rate",
            "interval_method": "paired overlapping moving-block bootstrap; same sampled session indices for both return series",
            "bootstrap_replications": replications, "block_sessions": block, "seed": seed,
            "uncertainty_caveat": "Unadjusted post-selection historical uncertainty; no correction for multiple comparisons or model selection, no significance or prospective-performance proof. Annualized mean return difference is not a CAGR difference.",
            "risk_caveat": "Exposure levels, serial dependence, tail and gap risks differ. Sharpe uses the declared zero cash/risk-free convention; assess drawdown and absolute return alongside the ratio."}


def xlf_action_audit():
    """Diagnose the provider representation without asserting economic accuracy."""
    frame = pd.read_csv(ROOT / "data/normalized/prices/XLF.csv", parse_dates=["Date"]).set_index("Date")
    actions = pd.read_csv(ROOT / "data/normalized/actions/XLF.csv")
    window = frame.loc["2016-09-14":"2016-09-22"].copy()
    window["provider_close_return"] = frame.Close.pct_change().reindex(window.index)
    window["adjusted_close_return"] = frame.adj_close.pct_change().reindex(window.index)
    window["adjustment_factor"] = window["Adj Close"] / window.Close
    report = window.reset_index()
    report.to_csv(REPORTS / "xlf_2016_action_window.csv", index=False)
    action_rows = actions.loc[actions.Date.between("2016-09-14", "2016-09-22")]
    after = window.loc[pd.Timestamp("2016-09-19")]
    before = window.loc[pd.Timestamp("2016-09-16")]
    return {
        "status": "LIMITATION_REQUIRING_INDEPENDENT_ACTION_RECONCILIATION",
        "issuer_filing": XLF_SOURCE,
        "issuer_filing_context": "Issuer filing describes removing real-estate holdings and an in-kind distribution of XLRE shares to XLF holders around the September 2016 rebalance.",
        "provider_action_rows": json.loads(action_rows.to_json(orient="records")),
        "provider_pre_event_close": float(before.Close),
        "provider_event_close": float(after.Close),
        "provider_event_close_return": float(after.provider_close_return),
        "provider_event_adjusted_close_return": float(after.adjusted_close_return),
        "adjustment_factor_before": float(before.adjustment_factor),
        "adjustment_factor_after": float(after.adjustment_factor),
        "finding": "Yahoo encodes the 2016-09-19 event as a 1231:1000 split. Its historical OHLC already reflect that representation. A separate received-XLRE distribution is not recorded in this action extract. The adjusted series is smooth across the event, but smoothness does not establish the exact economic total return.",
        "research_treatment": "Fractional adjusted-unit total-return approximation; no separate dividend or XLRE credit. Do not treat this as a verified raw-share corporate-action ledger.",
        "official_validation_requirement": "Reconcile contemporaneous unadjusted XLF prices, exact XLRE units, distribution/payable timing and disposal/reinvestment policy against issuer or independently verified action data.",
    }


def audit(allow_partial=False):
    cfg = research.CFG
    reserve = pd.Timestamp(cfg["reserved_start"])
    checks = []
    violations = []
    tickers = []
    sharpe_comparisons = []

    def check(name, passed, ticker=None, detail=None):
        record = {"check": name, "passed": bool(passed)}
        if ticker:
            record["ticker"] = ticker
        if detail is not None:
            record["detail"] = detail
        checks.append(record)
        if not passed:
            violations.append(record)

    manifest_path = REPORTS / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        check("run_manifest_config_hash", manifest.get("config_sha256") == sha256(ROOT / "config/run_config.json"))
        check("run_manifest_research_hash", manifest.get("research_sha256") == sha256(ROOT / "src/research.py"))
    else:
        manifest = {}
    check("selection_objective_is_declared_net_Sharpe", "sharpe" in cfg["inner_objective"].lower())

    prices = research.load_prices()
    historical = {ticker: frame.loc[frame.index < reserve] for ticker, frame in prices.items()}
    raw_manifest = json.loads((ROOT / "data/raw_download_manifest.json").read_text())
    raw_checks = []
    for source in raw_manifest["sources"]:
        path = ROOT / source["raw_path"]
        raw_checks.append(path.exists() and sha256(path) == source["sha256"])
    check("all_archived_raw_source_hashes_match", all(raw_checks), detail={"sources": len(raw_checks)})

    global_metrics = pd.read_csv(REPORTS / "metrics.csv") if (REPORTS / "metrics.csv").exists() else pd.DataFrame()
    missing = []
    for ticker in cfg["universe"]:
        folder = RUNS / ticker
        needed = [folder / "forecasts.csv", folder / "selections.csv", folder / "events.csv",
                  folder / "candidate_ledger.csv", folder / "ledger_CART_selected_5bps.csv",
                  folder / "ledger_buy_hold_5bps.csv", folder / "ledger_SPY_context_5bps.csv",
                  folder / "uncertainty.json"]
        if not all(path.exists() for path in needed):
            missing.append(ticker)
            continue
        selections = pd.read_csv(folder / "selections.csv", parse_dates=["train_last_decision", "train_latest_label", "cutoff"])
        forecasts = pd.read_csv(folder / "forecasts.csv", parse_dates=["decision_date", "entry_date", "label_end", "fit_cutoff"])
        events = read_events(folder / "events.csv")
        candidates = pd.read_csv(folder / "candidate_ledger.csv")
        check("training_decisions_strictly_before_cutoff", (selections.train_last_decision < selections.cutoff).all(), ticker)
        check("training_labels_strictly_matured_before_cutoff", (selections.train_latest_label < selections.cutoff).all(), ticker)
        check("minimum_matured_weekly_training_rows", (selections.train_rows >= cfg["minimum_matured_weekly_rows"]).all(), ticker)
        check("selection_cutoffs_equal_outer_year_start", all(row.cutoff == pd.Timestamp(int(row.outer_year), 1, 1) for row in selections.itertuples()), ticker)
        check("macro_recipes_quarantined", selections.recipe.isin(cfg["primary_eligible_recipes"]).all(), ticker)
        check("event_feature_dates_before_reserve", (events.index < reserve).all(), ticker)
        check("event_entry_dates_before_reserve", (events.entry_date.dropna() < reserve).all(), ticker)
        check("event_label_ends_before_reserve", all((events[f"label_end_{h}"].dropna() < reserve).all() for h in cfg["horizons"]), ticker)
        check("no_duplicate_forecast_per_family_decision", not forecasts.duplicated(["family", "decision_date"]).any(), ticker)
        check("forecast_dates_before_reserve", (forecasts.decision_date < reserve).all() and (forecasts.entry_date < reserve).all() and (forecasts.label_end.dropna() < reserve).all(), ticker)
        check("forecast_after_fit_cutoff_and_before_execution", ((forecasts.decision_date >= forecasts.fit_cutoff) & (forecasts.entry_date > forecasts.decision_date)).all(), ticker)
        check("finite_predictions", np.isfinite(forecasts.prediction).all(), ticker)
        classified = forecasts.loc[forecasts.target.eq("classification"), "prediction"]
        check("classifier_probabilities_in_bounds", classified.between(0, 1).all(), ticker)

        chosen_ok = True
        chosen_errors = []
        forecast_specs_ok = True
        model_hashes_ok = True
        train_replay_ok = True
        forecast_coverage_ok = True
        for row in selections.itertuples():
            pool = candidates.loc[(candidates.outer_year == row.outer_year) & candidates.status.eq("eligible")].copy()
            pool["fixed_preference"] = np.where(pool.sizing.eq("fixed"), 0, 1)
            pool = pool.sort_values(["score", "recipe_id", "fixed_preference"], ascending=[False, True, True])
            if pool.empty or pool.iloc[0].recipe_id != row.recipe_id or pool.iloc[0].sizing != row.sizing or not np.isclose(pool.iloc[0].score, row.inner_score, atol=1e-10):
                chosen_ok = False
                chosen_errors.append(int(row.outer_year))
            ff = forecasts.loc[forecasts.outer_year == row.outer_year]
            forecast_specs_ok &= (ff.horizon.eq(row.horizon) & ff.target.eq(row.target) & ff.recipe.eq(row.recipe) & ff.sizing.eq(row.sizing) & ff.fit_cutoff.eq(row.cutoff)).all()
            columns = research.feature_columns(events, row.recipe)
            training = research.mature_rows(events, row.cutoff, columns)
            train_replay_ok &= len(training) == row.train_rows and training.index[-1] == row.train_last_decision and training.label_end_20.max() == row.train_latest_label
            next_year = pd.Timestamp(int(row.outer_year) + 1, 1, 1)
            expected_test = events.loc[(events.index >= row.cutoff) & (events.index < next_year)].dropna(subset=columns + ["entry_date"])
            for family in FAMILIES:
                file = folder / f"model_{row.outer_year}_{family}.pkl"
                recorded = ff.loc[ff.family.eq(family), "model_sha256"]
                model_hashes_ok &= file.exists() and len(recorded) > 0 and recorded.eq(sha256(file)).all()
                forecast_coverage_ok &= ff.loc[ff.family.eq(family), "decision_date"].sort_values().tolist() == expected_test.index.tolist()
        check("selected_recipe_matches_prior_inner_winner_and_tie_break", chosen_ok, ticker, chosen_errors)
        check("forecasts_use_registered_pre_outer_specification", forecast_specs_ok, ticker)
        check("all_forecast_model_hashes_match_saved_models", model_hashes_ok, ticker)
        check("registered_training_dates_replay_exact_maturity_and_features", train_replay_ok, ticker)
        check("all_eligible_outer_decisions_have_each_family_forecast", forecast_coverage_ok, ticker)

        reference_dates = forecasts.loc[forecasts.family.eq("CART"), "decision_date"].sort_values().tolist()
        check("learner_forecast_dates_aligned", all(forecasts.loc[forecasts.family.eq(family), "decision_date"].sort_values().tolist() == reference_dates for family in FAMILIES), ticker)
        next_open_ok = True
        labels_ok = True
        outcomes_ok = True
        source = historical[ticker]
        for row in forecasts.itertuples():
            pos = source.index.get_indexer([row.decision_date])[0]
            next_open_ok &= pos >= 0 and pos + 1 < len(source) and source.index[pos + 1] == row.entry_date
            if pd.notna(row.label_end):
                end_pos = pos + 1 + int(row.horizon)
                labels_ok &= end_pos < len(source) and source.index[end_pos] == row.label_end
                expected = source.adj_open.iloc[end_pos] / source.adj_open.iloc[pos + 1] - 1 - cfg["label_roundtrip_hurdle"]
                outcomes_ok &= np.isclose(expected, row.matured_outcome, atol=1e-10)
        check("all_forecasts_execute_next_actual_open", next_open_ok, ticker)
        check("forecast_label_ends_match_session_horizons", labels_ok, ticker)
        check("recorded_matured_outcomes_match_adjusted_open_returns", outcomes_ok, ticker)

        # Rebuild from reserved-truncated prices and compare actual saved features.
        rebuilt = research.build_features(historical, ticker, None)
        feature_cols = [c for c in events if c.startswith(("tech_", "rel_"))]
        saved = events[feature_cols].to_numpy(dtype=float)
        causal = rebuilt.reindex(events.index)[feature_cols].to_numpy(dtype=float)
        check("saved_technical_relationship_features_match_causal_rebuild", np.allclose(saved, causal, rtol=1e-10, atol=1e-10, equal_nan=True), ticker)

        primary = pd.read_csv(folder / "ledger_CART_selected_5bps.csv", index_col="Date", parse_dates=["Date"])
        buy_hold = pd.read_csv(folder / "ledger_buy_hold_5bps.csv", index_col="Date", parse_dates=["Date"])
        sharpe_comparisons.append(paired_sharpe_comparison(ticker, primary, buy_hold))
        ledgers = sorted(folder.glob("ledger_*.csv"))
        for path in ledgers:
            ledger = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
            name = path.stem
            stamp = f"{ticker}:{name}"
            check("ledger_dates_unique_sorted_and_before_reserve", ledger.index.is_unique and ledger.index.is_monotonic_increasing and (ledger.index < reserve).all(), stamp)
            check("ledger_comparison_dates_match_primary", ledger.index.equals(primary.index), stamp)
            check("ledger_long_cash_bounds", np.isfinite(ledger.nav).all() and (ledger.nav > 0).all() and (ledger.cash >= -1e-10).all() and ledger.weight.between(-1e-10, 1 + 1e-10).all() and (ledger.shares >= -1e-10).all(), stamp)
            close_prices = historical["SPY" if "SPY_context" in name else ticker].adj_close.reindex(ledger.index)
            check("ledger_cash_plus_units_equals_nav", np.allclose(ledger.cash + ledger.shares * close_prices, ledger.nav, atol=1e-10, rtol=1e-10), stamp)
            expected_returns = ledger.nav.pct_change()
            expected_returns.iloc[0] = ledger.nav.iloc[0] - 1
            check("ledger_daily_returns_reconcile_nav", np.allclose(expected_returns, ledger.daily_return, atol=1e-12, rtol=1e-10), stamp)
            fee = int(re.search(r"_(\d+)bps$", name).group(1)) / 10000
            terminal_cost = ledger.terminal_cost if "terminal_cost" in ledger else 0
            check("ledger_costs_match_actual_traded_notional", np.allclose(ledger.costs, ledger.trade_value.abs() * fee + terminal_cost, atol=1e-12, rtol=1e-10) and (ledger.costs >= 0).all(), stamp)

        tt = global_metrics.loc[global_metrics.ticker.eq(ticker)] if len(global_metrics) else pd.DataFrame()
        check("global_metrics_include_instrument", len(tt) > 0, ticker)
        for row in tt.itertuples():
            if row.strategy in FAMILIES:
                tag = f"{row.strategy}_{row.sizing}_{row.cost_bps}bps"
            elif row.strategy == "SPY_context":
                tag = f"SPY_context_{row.cost_bps}bps"
            else:
                tag = f"{row.strategy}_{row.cost_bps}bps"
            path = folder / f"ledger_{tag}.csv"
            if not path.exists():
                check("metrics_ledger_exists", False, ticker, tag)
                continue
            ledger = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
            recomputed = research.metrics(ledger)
            keys = ["total_return", "cagr", "volatility", "sharpe", "max_drawdown", "average_exposure"]
            check("summary_metrics_recompute_from_saved_ledger", all(np.isclose(getattr(row, key), recomputed[key], atol=1e-10, rtol=1e-9) for key in keys), f"{ticker}:{tag}")
        tickers.append({"ticker": ticker, "outer_years": selections.outer_year.tolist(), "selection_rows": len(selections),
                        "forecast_rows": len(forecasts), "ledger_files": len(ledgers), "evaluation_start": str(primary.index[0].date()),
                        "evaluation_end": str(primary.index[-1].date()), "minimum_train_rows": int(selections.train_rows.min()),
                        "maximum_training_label": str(selections.train_latest_label.max().date())})

    if not allow_partial:
        check("all_15_instruments_complete", not missing, detail={"missing": missing})
    challenger_path = REPORTS / "challenger_inner_metrics.csv"
    challenger_rows = len(pd.read_csv(challenger_path)) if challenger_path.exists() else 0
    if sharpe_comparisons:
        pd.DataFrame(sharpe_comparisons).to_csv(REPORTS / "sharpe_comparison.csv", index=False)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "FAIL" if violations else ("PARTIAL" if missing else "PASS"),
              "scope": "Saved historical outputs only. This audits recorded chronology, accounting and consistency; it does not prove profitability, point-in-time vendor adjustment correctness, or independence of historical research.",
              "reserved_start": cfg["reserved_start"], "reserved_scored": False,
              "expected_instruments": len(cfg["universe"]), "audited_instruments": len(tickers), "missing_instruments": missing,
              "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks),
              "violations": violations, "instruments": tickers, "checks": checks,
              "challenger_inner_comparison_rows": challenger_rows,
              "primary_outperformance_measure": "Net daily Sharpe difference versus the same ETF buy/hold, at matched dates and 5 bps one-way costs, with zero cash/risk-free rate.",
              "sharpe_comparison_file": "reports/sharpe_comparison.csv",
              "sharpe_uncertainty": "500 paired moving-block bootstrap replications, 20 sessions per block. Unadjusted post-selection historical intervals; no multiple-comparison adjustment or significance claim.",
              "supplemental_benchmark_file": "reports/benchmark_diagnostics.csv",
              "supplemental_benchmark_scope": "Causal fixed 10% volatility-target always-long reference and a separately labeled ex-post weekly rebalanced average-exposure reference; primary metrics and forecasts unchanged.",
              "xlf_2016_action": xlf_action_audit()}
    (REPORTS / "result_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Result audit {report['status']}: {report['checks_passed']}/{len(checks)} checks; {len(tickers)}/15 instruments", flush=True)
    return report


def compare_inner_challengers():
    """One fixed trial per family on the last CART-shortlisted pre-2023 recipe."""
    cfg = research.CFG
    year = 2023
    plan_path = REPORTS / "challenger_inner_plan.json"
    plan = {"purpose": "Descriptive matched earlier-fold family comparison; no retuning or promotion from these comparisons or outer results.",
            "outer_cutoff": "2023-01-01", "inner_validation_years": [2021, 2022],
            "shortlist": "One recipe and sizing per ETF from the saved pre-2023 CART screen; no additional recipe search.",
            "families": FAMILIES, "trials_per_family_per_instrument": 1, "fits_per_trial": 2,
            "expected_trials": len(cfg["universe"]) * len(FAMILIES), "deployed_primary_family": "CART",
            "promotion_allowed": False, "cpu_jobs_per_model": 1,
            "config_sha256": sha256(ROOT / "config/run_config.json"),
            "research_sha256": sha256(ROOT / "src/research.py")}
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        if any(existing.get(key) != value for key, value in plan.items()):
            raise RuntimeError("Existing challenger plan differs; preserve it and create a new declared version")
    else:
        plan["declared_at_utc"] = datetime.now(timezone.utc).isoformat()
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    prices = {ticker: frame.loc[frame.index < cfg["reserved_start"]]
              for ticker, frame in research.load_prices().items()}
    rows = []
    result_path = REPORTS / "challenger_inner_metrics.csv"
    for ticker in cfg["universe"]:
        selection = pd.read_csv(RUNS / ticker / "selections.csv")
        chosen = selection.loc[selection.outer_year.eq(year)]
        if len(chosen) != 1:
            raise RuntimeError(f"{ticker}: one saved pre-2023 selection required")
        chosen = chosen.iloc[0]
        if chosen.recipe not in cfg["primary_eligible_recipes"]:
            raise RuntimeError("Unqualified recipe cannot enter challenger comparison")
        events = read_events(RUNS / ticker / "events.csv")
        for family in FAMILIES:
            score, folds = research.eval_inner(events, prices[ticker], year, chosen.recipe,
                                               int(chosen.horizon), chosen.target, family=family,
                                               sizing=chosen.sizing)
            if not np.isfinite(score):
                raise RuntimeError(f"{ticker}/{family}: comparison unexpectedly ineligible")
            row = {"ticker": ticker, "family": family, "outer_cutoff": "2023-01-01",
                   "validation_years": "2021,2022", "recipe_id": chosen.recipe_id,
                   "horizon": int(chosen.horizon), "target": chosen.target, "recipe": chosen.recipe,
                   "sizing": chosen.sizing, "inner_score": score,
                   "saved_CART_selection_score": float(chosen.inner_score),
                   "matched_CART_score_reproduced": bool(np.isclose(score, chosen.inner_score, rtol=1e-9, atol=1e-10)) if family == "CART" else None,
                   "fold_metrics": json.dumps(folds), "trial_parameters": json.dumps(cfg["cart" if family == "CART" else "random_forest" if family == "RF" else "xgboost"], sort_keys=True),
                   "primary_family": "CART", "promotion_allowed": False,
                   "comparison_status": "descriptive_prior_fold_challenger_on_CART_shortlist"}
            if family == "CART" and not row["matched_CART_score_reproduced"]:
                raise RuntimeError(f"{ticker}: saved CART inner score failed reproduction")
            rows.append(row)
            pd.DataFrame(rows).to_csv(result_path, index=False)
            print(f"Prior-fold challenger {ticker} {family}: Sharpe {score:.4f}; no promotion", flush=True)
    return rows


def supplemental_benchmarks():
    """Separate timing signals from a predeclared volatility-scaling control."""
    cfg = research.CFG
    prices = {ticker: frame.loc[frame.index < cfg["reserved_start"]]
              for ticker, frame in research.load_prices().items()}
    rows = []
    for ticker in cfg["universe"]:
        folder = RUNS / ticker
        primary = pd.read_csv(folder / "ledger_CART_selected_5bps.csv", index_col="Date", parse_dates=["Date"])
        forecasts = pd.read_csv(folder / "forecasts.csv", parse_dates=["entry_date"])
        forecasts = forecasts.loc[forecasts.family.eq("CART")].sort_values("entry_date")
        vol_weights = np.minimum(1., cfg["volatility_target"] /
                                 forecasts.vol63.clip(lower=cfg["volatility_floor"]))
        exposure = research.metrics(primary)["average_exposure"]
        primary_sharpe = research.metrics(primary)["sharpe"]
        references = [
            ("vol_always", vol_weights.to_numpy(),
             "Causal predeclared always-long volatility target, using only saved decision-time 63-session volatility, 10% annual target, 5% volatility floor and 100% cap; no model signal or future exposure estimate.",
             None),
            ("exposure_rebalanced", np.repeat(exposure, len(forecasts)),
             "Ex-post descriptive reference: constant target set from the primary strategy's full-sample average held exposure, rebalanced at the same weekly entries. It uses future sample information and is not a deployable causal baseline.",
             exposure),
        ]
        for name, scale, note, target in references:
            weights = pd.Series(scale, index=pd.DatetimeIndex(forecasts.entry_date))
            ledger = research.simulate(prices[ticker], weights, start=primary.index[0],
                                       end=primary.index[-1], cost_bps=5)
            path = folder / f"ledger_benchmark_{name}_5bps.csv"
            ledger.to_csv(path)
            summary = research.metrics(ledger)
            rows.append({"ticker": ticker, "benchmark": name,
                         "start": str(ledger.index[0].date()), "end": str(ledger.index[-1].date()),
                         "cost_bps": 5, "primary_sharpe": primary_sharpe,
                         "delta_sharpe_primary_minus_benchmark": primary_sharpe - summary["sharpe"],
                         "constant_target_exposure": target, "causality_and_interpretation": note,
                         "ledger_path": path.relative_to(ROOT).as_posix(), **summary})
    pd.DataFrame(rows).to_csv(REPORTS / "benchmark_diagnostics.csv", index=False)
    print(f"Supplemental references saved: {len(rows)} ledgers/metric rows", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--challengers", action="store_true", help="Run the frozen 2021/2022 family comparison after all saved outputs exist")
    parser.add_argument("--benchmarks", action="store_true", help="Save causal always-long volatility and ex-post exposure references separately from primary metrics")
    args = parser.parse_args()
    if args.challengers:
        compare_inner_challengers()
    if args.benchmarks:
        supplemental_benchmarks()
    result = audit(allow_partial=args.allow_partial)
    if result["violations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
