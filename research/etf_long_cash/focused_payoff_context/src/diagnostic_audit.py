"""Independent, read-only checks for the focused saved-result diagnostic bundle.

No fitting, price retrieval, model execution or new trading simulation occurs.
The callable returns an ordinary dictionary; writing an audit file is left to
the caller. CLI output is JSON on stdout and a nonzero exit on failed checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = {"payoff_context", "etf_hold", "spy_hold"}


def _csv(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def _near(a, b, rtol=1e-8, atol=1e-10):
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def _window_values(ledger):
    """Use saved returns, preserving carry-in and first-session fees."""
    r = ledger.daily_return.to_numpy(dtype=float)
    n = len(r)
    start_nav = float(ledger.nav.iloc[0] / (1 + r[0]))
    relative = ledger.nav.to_numpy(dtype=float) / start_nav
    total = float(np.prod(1 + r) - 1)
    deviation = float(np.std(r, ddof=1)) if n > 1 else 0.
    downside = float(np.sqrt(np.mean(np.square(np.minimum(r, 0)))) * np.sqrt(252))
    drawdown = relative / np.maximum.accumulate(np.r_[1., relative])[1:] - 1
    cagr = float((1 + total) ** (252 / n) - 1)
    worst_n = int(np.ceil(.05 * n))
    return {"days": n, "total_return": total, "cagr": cagr,
            "sharpe": float(np.mean(r) / deviation * np.sqrt(252)) if deviation > 0 else 0.,
            "volatility": deviation * np.sqrt(252), "max_drawdown": float(drawdown.min()),
            "downside_deviation": downside,
            "sortino": float(np.mean(r) * 252 / downside) if downside > 0 else np.nan,
            "calmar": cagr / abs(float(drawdown.min())) if drawdown.min() < 0 else np.nan,
            "worst_daily_return": float(r.min()), "best_daily_return": float(r.max()),
            "cvar5_daily_return": float(np.sort(r)[:worst_n].mean()),
            "average_exposure": float(ledger.held_exposure.mean()),
            "turnover_per_year": float(ledger.turnover.sum() / (n / 252)),
            "fees_per_window_start_NAV": float(ledger.costs.sum() / start_nav)}


def _booleans(series):
    if series.dtype == bool:
        return series
    return series.map({True: True, False: False, "True": True, "False": False,
                       "true": True, "false": False, 1: True, 0: False})


def _extended_inputs(root, manifest, forecasts, ledgers, check, pending):
    """Audit saved richer inputs without calling their metric implementation."""
    reserve = pd.Timestamp(manifest["reserved_start"])
    names = ["forecast_outcomes", "weekly_policy_intervals", "context_availability",
             "coverage_pre2024", "prior_calibration_oof", "paired_benchmark_uncertainty",
             "v3_common_period", "v3_uncertainty", "v3_multiplicity", "passive_vol_control"]
    tables = {}
    for name in names:
        path = root / f"data/{name}.csv"
        if not path.is_file():
            pending.append(f"Missing extended input: {path.name}")
        else:
            tables[name] = _csv(path)
    if len(tables) != len(names):
        return
    outcomes = tables["forecast_outcomes"]
    for column in ["decision_date", "entry_date", "label_end"]:
        outcomes[column] = pd.to_datetime(outcomes[column])
    check("outer_outcomes:one_to_one_with_saved_forecasts", len(outcomes) == len(forecasts)
          and not outcomes.duplicated(["ticker", "decision_date"]).any())
    merged = forecasts.merge(outcomes, on=["ticker", "decision_date"], suffixes=("_f", "_o"), validate="one_to_one")
    comparable = ["entry_date", "outer_year", "target", "horizon", "prediction", "expected_payoff",
                  "signal", "target_weight", "label_end", "payoff_bin_n", "oof_rows", "feature_count"]
    same = True
    for column in comparable:
        a, b = merged[column + "_f"], merged[column + "_o"]
        if pd.api.types.is_numeric_dtype(a):
            same &= _near(a, b)
        else:
            same &= bool((a.eq(b) | (a.isna() & b.isna())).all())
    check("outer_outcomes:saved_forecast_fields_unchanged", same)
    available = _booleans(outcomes.label_available)
    check("outer_outcomes:available_label_contract", available.notna().all()
          and available.eq(outcomes.label_end.notna() & outcomes.actual_net_horizon_return.notna()
                           & outcomes.actual_gross_horizon_return.notna()).all())
    check("outer_outcomes:no_reserved_label_or_execution", (outcomes.entry_date < reserve).all()
          and (outcomes.label_end.dropna() < reserve).all())
    check("outer_outcomes:original_fixed_hurdle", _near(outcomes.loc[available, "actual_gross_horizon_return"]
          - outcomes.loc[available, "actual_net_horizon_return"], .001))
    check("outer_outcomes:available_returns_finite", np.isfinite(outcomes.loc[available,
          ["actual_net_horizon_return", "actual_gross_horizon_return"]]).all().all())

    weekly = tables["weekly_policy_intervals"]
    for column in ["decision_date", "entry_date", "interval_end"]:
        weekly[column] = pd.to_datetime(weekly[column])
    check("policy_intervals:one_per_saved_decision", len(weekly) == len(forecasts)
          and not weekly.duplicated(["ticker", "decision_date"]).any())
    for ticker in manifest["tickers"]:
        d = ledgers[(ticker, "payoff_context")]
        f = forecasts.loc[forecasts.ticker.eq(ticker)].sort_values("entry_date")
        w = weekly.loc[weekly.ticker.eq(ticker)].sort_values("entry_date")
        s = f.entry_date.to_numpy()
        ends = np.r_[s[1:], d.index[-1].to_datetime64()]
        boundary_values = np.r_[d.loc[f.entry_date.iloc[1:], "opening_nav"].to_numpy(), d.nav.iloc[-1]]
        start_values = d.loc[f.entry_date, "opening_nav"].to_numpy()
        session_n = [int(((d.index >= a) & (d.index < b)).sum()) for a, b in zip(s[:-1], s[1:])]
        session_n += [int((d.index >= s[-1]).sum())]
        partial = _booleans(w.partial_terminal_interval)
        check(f"{ticker}:policy_boundaries_keep_pretrade_NAV_and_single_fees",
              len(w) == len(f) and np.array_equal(w.entry_date.to_numpy(), s)
              and np.array_equal(w.interval_end.to_numpy(), ends)
              and _near(w.start_opening_nav, start_values) and _near(w.end_nav, boundary_values)
              and _near(w.policy_net_return, boundary_values / start_values - 1)
              and np.array_equal(w.sessions.to_numpy(), session_n)
              and np.array_equal(partial.to_numpy(), np.r_[np.zeros(len(w)-1, dtype=bool), True]))
        check(f"{ticker}:policy_intervals_compound_to_whole_saved_account",
              _near(np.prod(1 + w.policy_net_return), d.nav.iloc[-1]))
        check(f"{ticker}:policy_signals_and_weights_unchanged", _near(w.signal, f.signal)
              and _near(w.target_weight, f.target_weight)
              and np.array_equal(w.decision_date.to_numpy(), f.decision_date.to_numpy()))

    coverage = tables["coverage_pre2024"]
    check("coverage:one_record_per_ETF", len(coverage) == 15 and not coverage.ticker.duplicated().any()
          and set(coverage.ticker) == set(manifest["tickers"]))
    for ticker in manifest["tickers"]:
        row = coverage.loc[coverage.ticker.eq(ticker)].iloc[0]
        d = ledgers[(ticker, "payoff_context")]
        o = outcomes.loc[outcomes.ticker.eq(ticker)]
        a = _booleans(o.label_available)
        check(f"{ticker}:coverage_matches_saved_accounts_and_outcomes",
              pd.Timestamp(row.strategy_start) == d.index.min() and pd.Timestamp(row.strategy_end) == d.index.max()
              and row.strategy_sessions == len(d) and row.forecast_decisions == len(o)
              and row.available_horizon_outcomes == a.sum()
              and row.excluded_trailing_horizon_outcomes == (~a).sum()
              and row.duplicate_price_dates == 0 and row.invalid_adjusted_open_close == 0
              and pd.Timestamp(row.first_price_date) <= d.index.min()
              and pd.Timestamp(row.last_included_price_date) < reserve)
    availability = tables["context_availability"]
    check("context_availability:unique_flag_instances_and_valid_rates", len(availability) == 76
          and not availability.duplicated(["ticker", "availability_flag"]).any()
          and availability.mean_available_fraction.between(0, 1).all()
          and availability.unmatched_rows.eq(0).all()
          and availability.apply(lambda x: x.decision_rows == forecasts.ticker.eq(x.ticker).sum(), axis=1).all())

    oof = tables["prior_calibration_oof"]
    for column in ["decision_date", "entry_date", "label_end", "label_end20", "fit_cutoff", "calibration_cutoff", "oof_train_latest_label"]:
        oof[column] = pd.to_datetime(oof[column])
    check("prior_OOF:unique_within_each_annual_calibrator_only",
          not oof.duplicated(["ticker", "outer_year", "decision_date"]).any()
          and _booleans(oof.context).eq(True).all())
    check("prior_OOF:training_and_calibration_maturity",
          (oof.oof_train_latest_label < oof.fit_cutoff).all() and oof.oof_train_rows.ge(104).all()
          and (oof.fit_cutoff <= oof.decision_date).all() and (oof.entry_date > oof.decision_date).all()
          and (oof.label_end < oof.calibration_cutoff).all() and (oof.label_end20 < oof.calibration_cutoff).all()
          and (oof.calibration_cutoff < reserve).all()
          and pd.DatetimeIndex(oof.calibration_cutoff).year.to_numpy().tolist() == oof.outer_year.tolist())
    calibrators = json.loads((root / "data/calibrators.json").read_text())
    check("prior_OOF:one_context_calibrator_per_saved_model_year",
          len(calibrators) == forecasts.groupby(["ticker", "outer_year"]).ngroups
          and all(c["context"] is True for c in calibrators))
    for c in calibrators:
        ticker, year, cal = c["ticker"], c["outer_year"], c["calibrator"]
        g = oof.loc[oof.ticker.eq(ticker) & oof.outer_year.eq(year)]
        f = forecasts.loc[forecasts.ticker.eq(ticker) & forecasts.outer_year.eq(year)]
        expected_n = len(g)
        valid = bool(cal["n"] == expected_n and bool(cal["eligible"]) == (expected_n >= 52))
        if cal["eligible"]:
            valid &= _near(cal["global_mean"], g.outcome.mean()) and cal["pseudo_observations"] == 26
            bins = np.searchsorted(cal["edges"], g.prediction.to_numpy(), side="right")
            for i, b in enumerate(cal["bins"]):
                values = g.loc[bins == i, "outcome"]
                shrunk = (values.sum() + 26 * g.outcome.mean()) / (len(values) + 26)
                valid &= b["n"] == len(values) and _near(b["expected_payoff"], shrunk)
            fb = np.searchsorted(cal["edges"], f.prediction.to_numpy(), side="right")
            applied = np.asarray([cal["bins"][i]["expected_payoff"] for i in fb])
            counts = np.asarray([cal["bins"][i]["n"] for i in fb])
            valid &= _near(f.expected_payoff, applied) and _near(f.payoff_bin_n, counts)
        else:
            valid &= f.expected_payoff.isna().all()
        check(f"{ticker}:{year}:calibrator_matches_prior_OOF_and_outer_mapping", valid)

    paired = tables["paired_benchmark_uncertainty"]
    plan_path = root / "data/paired_uncertainty_plan.json"
    plan = json.loads(plan_path.read_text())
    check("benchmark_uncertainty:separate_30_comparison_descriptive_family", len(paired) == 30
          and set(paired.benchmark) == {"etf_hold", "spy_hold"}
          and not paired.duplicated(["ticker", "benchmark"]).any()
          and paired.strategy.eq("payoff_context").all() and paired.bootstrap_replications.eq(9999).all()
          and paired.block_sessions.eq(20).all() and paired.uncertainty_scope.eq("posthoc_descriptive_fixed_fit").all()
          and not any("pvalue" in col.lower() or "p_value" in col.lower() or "p_holm" in col.lower() for col in paired))
    check("benchmark_uncertainty:declared_input_and_plan_hashes_match",
          paired.input_sha256.eq(hashlib.sha256((root / "data/ledgers.csv").read_bytes()).hexdigest()).all()
          and paired.plan_sha256.eq(hashlib.sha256(plan_path.read_bytes()).hexdigest()).all()
          and plan["input_sha256"] == paired.input_sha256.iloc[0]
          and plan["comparison_count"] == 30 and plan["bootstrap_replications"] == 9999
          and plan["source_sha256"] == hashlib.sha256((root / "src/paired_uncertainty.py").read_bytes()).hexdigest())
    for row in paired.itertuples():
        s = _window_values(ledgers[(row.ticker, "payoff_context")])
        b = _window_values(ledgers[(row.ticker, row.benchmark)])
        d = ledgers[(row.ticker, "payoff_context")]
        check(f"{row.ticker}:{row.benchmark}:uncertainty_observed_values_match_saved_accounts",
              _near(row.strategy_sharpe, s["sharpe"]) and _near(row.benchmark_sharpe, b["sharpe"])
              and _near(row.delta_sharpe, s["sharpe"] - b["sharpe"])
              and row.days == len(d) and pd.Timestamp(row.start) == d.index[0] and pd.Timestamp(row.end) == d.index[-1]
              and np.isfinite([row.delta_sharpe_95pct_low, row.delta_sharpe_95pct_high]).all()
              and row.delta_sharpe_95pct_low <= row.delta_sharpe_95pct_high
              and row.moment_equivalence_validated and row.matched_axes_validated and row.joint_three_account_draws)
    old = tables["v3_uncertainty"]
    multiplicity = tables["v3_multiplicity"]
    check("parent_inference:retains_original_225_family_and_baseline_comparator", len(old) == 15
          and len(multiplicity) == 15 and old.variant.eq("payoff_context").all()
          and old.baseline_variant.eq("baseline").all() and old.multiplicity_family_size.eq(225).all()
          and multiplicity.multiplicity_family_size.eq(225).all())
    for row in old.itertuples():
        s = _window_values(ledgers[(row.ticker, "payoff_context")])
        m = multiplicity.loc[multiplicity.ticker.eq(row.ticker)].iloc[0]
        check(f"{row.ticker}:parent_inference_is_separate_and_unchanged",
              _near(row.variant_sharpe, s["sharpe"]) and _near(row.delta_sharpe, row.variant_sharpe-row.baseline_sharpe)
              and _near(m.delta_sharpe, row.delta_sharpe) and _near(m.p_holm, row.p_holm))

    passive = tables["passive_vol_control"]
    passive["Date"] = pd.to_datetime(passive.Date)
    check("passive_vol_reference:only_pre_reserved_dates", (passive.Date < reserve).all()
          and not passive.duplicated(["ticker", "Date"]).any())
    for ticker in manifest["tickers"]:
        d = ledgers[(ticker, "payoff_context")]
        p = passive.loc[passive.ticker.eq(ticker)].set_index("Date").sort_index()
        check(f"{ticker}:passive_vol_reference_aligned", p.index.equals(d.index)
              and p.held_exposure.between(-1e-10, 1+1e-10).all())
    common = tables["v3_common_period"]
    check("common_period:only_combined_saved_continuing_accounts", len(common) == 45
          and common.variant.eq("payoff_context").all() and common.analysis_run_directory.eq("runs/v3_1").all()
          and set(common.window) == {"common_full", "common_early", "common_late"})
    keys = ["days", "sharpe", "total_return", "cagr", "volatility", "max_drawdown", "average_exposure",
            "turnover_per_year", "fees_per_window_start_NAV", "worst_daily_return", "cvar5_daily_return"]
    for row in common.itertuples():
        d = ledgers[(row.ticker, "payoff_context")].loc[row.start:row.end]
        actual = _window_values(d)
        p = passive.loc[passive.ticker.eq(row.ticker)].set_index("Date").loc[row.start:row.end]
        reference_sharpe = _window_values(p)["sharpe"]
        check(f"{row.ticker}:{row.window}:common_window_retains_carry_in_and_saved_costs",
              all(_near(getattr(row, k), actual[k]) for k in keys)
              and _near(row.carry_in_nav_original, d.nav.iloc[0]/(1+d.daily_return.iloc[0]))
              and _near(row.first_included_daily_return, d.daily_return.iloc[0])
              and _near(row.causal_vol_reference_sharpe, reference_sharpe)
              and _near(row.delta_sharpe_vs_causal_vol_reference, row.sharpe-reference_sharpe))


def _diagnostic_exports(root, manifest, ledgers, check, pending):
    names = ["monthly_returns", "annual_performance", "risk_statistics", "drawdown_episodes",
             "forecast_summary", "probability_reliability", "payoff_reliability", "weekly_interval_summary"]
    tables = {}
    for name in names:
        path = root / f"data/{name}.csv"
        if not path.is_file():
            pending.append(f"Missing diagnostic export: {path.name}")
        else:
            tables[name] = _csv(path)
    if len(tables) != len(names):
        return
    calendar = pd.DatetimeIndex(pd.to_datetime(_csv(root / "data/session_calendar.csv").Date))
    check("reporting_calendar:unique_increasing_pre_reserved_sessions", calendar.is_unique
          and calendar.is_monotonic_increasing and (calendar < pd.Timestamp(manifest["reserved_start"])).all())
    scalar_keys = {"days": "days", "sessions": "days", "net_return": "total_return",
                   "annualized_return": "cagr", "sharpe": "sharpe", "volatility": "volatility",
                   "max_drawdown": "max_drawdown", "average_exposure": "average_exposure",
                   "annualized_turnover": "turnover_per_year", "costs_per_start_NAV": "fees_per_window_start_NAV",
                   "downside_semideviation_annual": "downside_deviation", "sortino": "sortino",
                   "calmar": "calmar", "worst_daily_return": "worst_daily_return",
                   "best_daily_return": "best_daily_return", "cvar5_daily_return": "cvar5_daily_return"}
    for ticker in manifest["tickers"]:
        for strategy in sorted(STRATEGIES):
            d = ledgers[(ticker, strategy)]
            stamp = f"{ticker}:{strategy}"
            for table_name, frequency in [("monthly_returns", "M"), ("annual_performance", "Y")]:
                data = tables[table_name]
                records = data.loc[data.ticker.eq(ticker) & data.strategy.eq(strategy)]
                grouping = d.groupby(d.index.to_period(frequency))
                check(stamp + f":{table_name}:one_record_per_observed_calendar_period", len(records) == grouping.ngroups)
                aggregate = 1.
                for period, window in grouping:
                    selected = records.loc[records.year.eq(period.year)]
                    if frequency == "M":
                        selected = selected.loc[selected.month.eq(period.month)]
                    valid = len(selected) == 1
                    if valid:
                        row = selected.iloc[0]
                        actual = _window_values(window)
                        # New descriptive tables flag zero-risk ratios undefined;
                        # the frozen parent audit uses Sharpe zero in that case.
                        if window.daily_return.std(ddof=1) <= 1e-15:
                            actual["sharpe"] = np.nan
                        if actual["downside_deviation"] <= 1e-15 * np.sqrt(252):
                            actual["sortino"] = np.nan
                        if abs(actual["max_drawdown"]) <= 1e-15:
                            actual["calmar"] = np.nan
                        actual_calendar = calendar[(calendar >= period.start_time.normalize()) & (calendar <= period.end_time.normalize())]
                        partial = not window.index.equals(actual_calendar)
                        valid &= all(_near(row[k], actual[v]) for k, v in scalar_keys.items() if k in row)
                        valid &= pd.Timestamp(row.start) == window.index[0] and pd.Timestamp(row.end) == window.index[-1]
                        valid &= int(row.calendar_days) == (window.index[-1]-window.index[0]).days+1
                        valid &= _near(row.carry_in_nav_original, window.nav.iloc[0]/(1+window.daily_return.iloc[0]))
                        valid &= bool(_booleans(pd.Series([row.partial_period])).iloc[0]) == partial
                        valid &= row.partial_period_basis == "provided_session_calendar"
                        valid &= _near(row.turnover, window.turnover.sum()) and _near(row.costs, window.costs.sum())
                        aggregate *= 1 + row.net_return
                    check(stamp + f":{table_name}:{period}:saved_returns_costs_carry_and_calendar", valid)
                check(stamp + f":{table_name}:period_returns_compound_to_full_account", _near(aggregate, d.nav.iloc[-1]))
            risk = tables["risk_statistics"]
            selected = risk.loc[risk.ticker.eq(ticker) & risk.strategy.eq(strategy)]
            valid = len(selected) == 1
            if valid:
                row = selected.iloc[0]
                actual = _window_values(d)
                r = d.daily_return
                relative = (1+r).cumprod()
                dd = relative / relative.cummax().clip(lower=1) - 1
                valid &= all(_near(row[k], actual[v]) for k, v in scalar_keys.items() if k in row)
                valid &= _near(row.var5_daily_return, r.quantile(.05))
                valid &= row.cvar5_tail_sessions == int(np.ceil(.05*len(d)))
                valid &= _near(row.daily_win_fraction, r.gt(0).mean()) and _near(row.daily_loss_fraction, r.lt(0).mean())
                valid &= _near(row.daily_flat_fraction, r.eq(0).mean()) and _near(row.ulcer_index, np.sqrt(np.mean(dd**2)))
                valid &= _near(row["skew"], r.skew()) and _near(row.excess_kurtosis, r.kurt())
                valid &= _near(row.turnover, d.turnover.sum()) and _near(row.costs, d.costs.sum())
            check(stamp + ":risk_statistics:independent_daily_tail_and_downside_formulas", valid)
            nav = (1 + d.daily_return).cumprod().to_numpy()
            peaks = np.maximum.accumulate(np.r_[1., nav])[1:]
            underwater = nav < peaks * (1-1e-12)
            starts = np.flatnonzero(underwater & ~np.r_[False, underwater[:-1]])
            records = tables["drawdown_episodes"]
            records = records.loc[records.ticker.eq(ticker) & records.strategy.eq(strategy)].sort_values("episode")
            valid = len(records) == len(starts)
            max_underwater = max_duration = max_calendar = 0
            ongoing_any = False
            if valid:
                for (_, row), start in zip(records.iterrows(), starts):
                    recovered = np.flatnonzero(~underwater[start:])
                    ongoing = len(recovered) == 0
                    end = len(d)-1 if ongoing else start + int(recovered[0])
                    peak_i = start-1
                    trough_i = start + int(np.argmin(nav[start:end+1]))
                    peak_nav = 1. if peak_i < 0 else nav[peak_i]
                    peak_date = d.index[max(0, peak_i)]
                    underwater_count = end-start + (1 if ongoing else 0)
                    duration = end-peak_i
                    elapsed = (d.index[end]-peak_date).days
                    valid &= pd.Timestamp(row.peak_date) == peak_date and pd.Timestamp(row.drawdown_start_date) == d.index[start]
                    valid &= pd.Timestamp(row.trough_date) == d.index[trough_i] and pd.Timestamp(row.end_date) == d.index[end]
                    valid &= _near(row.depth, nav[trough_i]/peak_nav-1)
                    valid &= _near(row.peak_NAV, peak_nav) and _near(row.trough_NAV, nav[trough_i])
                    valid &= row.underwater_sessions == underwater_count and row.duration_sessions == duration
                    valid &= row.peak_to_end_calendar_days == elapsed and bool(row.ongoing) == ongoing
                    valid &= bool(row.initial_peak_before_first_session) == (peak_i < 0)
                    if ongoing:
                        valid &= pd.isna(row.recovery_date) and pd.isna(row.peak_to_recovery_calendar_days)
                    else:
                        valid &= pd.Timestamp(row.recovery_date) == d.index[end]
                        valid &= row.peak_to_recovery_calendar_days == elapsed
                        valid &= row.trough_to_recovery_sessions == end-trough_i
                        valid &= row.trough_to_recovery_calendar_days == (d.index[end]-d.index[trough_i]).days
                    max_underwater = max(max_underwater, underwater_count)
                    max_duration = max(max_duration, duration)
                    max_calendar = max(max_calendar, elapsed)
                    ongoing_any |= ongoing
            check(stamp + ":drawdown_episodes:all_underwater_runs_dates_and_recovery_censoring", valid)
            if len(selected) == 1:
                row = selected.iloc[0]
                check(stamp + ":drawdown_summary:duration_and_open_status_match_episodes",
                      row.drawdown_episodes == len(starts) and row.longest_underwater_sessions == max_underwater
                      and row.longest_peak_to_end_sessions == max_duration
                      and row.longest_peak_to_end_calendar_days == max_calendar
                      and bool(row.ongoing_drawdown) == ongoing_any)
    _forecast_exports(root, manifest, tables, check)


def _forecast_exports(root, manifest, tables, check):
    outcomes = _csv(root / "data/forecast_outcomes.csv")
    intervals = _csv(root / "data/weekly_policy_intervals.csv")
    for ticker in manifest["tickers"]:
        facts = outcomes.loc[outcomes.ticker.eq(ticker)]
        summary = tables["forecast_summary"].loc[tables["forecast_summary"].ticker.eq(ticker)]
        check(f"{ticker}:forecast_summary:all_target_horizon_groups_retained",
              len(summary) == facts.groupby(["target", "horizon"]).ngroups)
        for (target, horizon), g in facts.groupby(["target", "horizon"]):
            a = g.loc[_booleans(g.label_available)]
            row = summary.loc[summary.target.eq(target) & summary.horizon_sessions.eq(horizon)]
            valid = len(row) == 1
            if valid:
                row = row.iloc[0]
                valid &= row.forecasts == len(g) and row.available_horizon_labels == len(a)
                valid &= row.unavailable_horizon_labels == len(g)-len(a)
                if len(a):
                    truth, signal = a.actual_net_horizon_return.gt(0), a.signal.eq(1)
                    tp, fp = int((signal & truth).sum()), int((signal & ~truth).sum())
                    tn, fn = int((~signal & ~truth).sum()), int((~signal & truth).sum())
                    valid &= _near(row.mean_actual_net_horizon_return, a.actual_net_horizon_return.mean())
                    valid &= _near(row.positive_outcome_rate, truth.mean()) and _near(row.long_decision_rate, signal.mean())
                    valid &= _near(row.payoff_signal_direction_accuracy, signal.eq(truth).mean())
                    valid &= (row.true_positive, row.false_positive, row.true_negative, row.false_negative) == (tp, fp, tn, fn)
                    if target == "classification":
                        probability = a.prediction
                        clipped = probability.clip(1e-12, 1-1e-12)
                        valid &= probability.between(0, 1).all()
                        valid &= _near(row.raw_probability_brier, ((probability-truth.astype(float))**2).mean())
                        valid &= _near(row.raw_probability_log_loss, -(truth*np.log(clipped)+(~truth)*np.log(1-clipped)).mean())
                    else:
                        error = a.prediction-a.actual_net_horizon_return
                        valid &= _near(row.raw_regression_mae, error.abs().mean())
                        valid &= _near(row.raw_regression_rmse, np.sqrt((error**2).mean()))
                    payoff = a.loc[a.expected_payoff.notna()]
                    if len(payoff):
                        error = payoff.expected_payoff-payoff.actual_net_horizon_return
                        valid &= row.payoff_eligible_labels == len(payoff)
                        valid &= _near(row.payoff_mae, error.abs().mean()) and _near(row.payoff_rmse, np.sqrt((error**2).mean()))
            check(f"{ticker}:{target}:{horizon}:forecast_accuracy_uses_only_available_horizon_labels", valid)
        probability_rows = tables["probability_reliability"]
        probability_rows = probability_rows.loc[probability_rows.ticker.eq(ticker)]
        p = facts.loc[facts.target.eq("classification") & _booleans(facts.label_available)].copy()
        p["bucket"] = np.searchsorted([.2, .4, .6, .8], p.prediction.to_numpy(), side="right")
        valid = len(probability_rows) == p.groupby(["horizon", "bucket"]).ngroups
        for (horizon, bucket), g in p.groupby(["horizon", "bucket"]):
            row = probability_rows.loc[probability_rows.horizon_sessions.eq(horizon) & probability_rows.probability_bin.eq(bucket)]
            valid &= len(row) == 1
            if len(row) == 1:
                row = row.iloc[0]
                truth = g.actual_net_horizon_return.gt(0)
                valid &= row.n == len(g) and _near(row.mean_probability, g.prediction.mean())
                valid &= _near(row.observed_positive_fraction, truth.mean())
                valid &= _near(row.brier, ((g.prediction-truth.astype(float))**2).mean())
        check(f"{ticker}:raw_probability_reliability:fixed_bins_and_classification_only", valid)
        payoff_rows = tables["payoff_reliability"]
        payoff_rows = payoff_rows.loc[payoff_rows.ticker.eq(ticker)]
        p = facts.loc[_booleans(facts.label_available) & facts.expected_payoff.notna()].copy()
        p["bucket"] = np.searchsorted([-.01, -.002, 0, .002, .01], p.expected_payoff.to_numpy(), side="right")
        labels = ["below −1%", "−1% to −0.2%", "−0.2% to 0%", "0% to 0.2%", "0.2% to 1%", "at least 1%"]
        valid = len(payoff_rows) == p.groupby(["target", "horizon", "bucket"]).ngroups
        for (target, horizon, bucket), g in p.groupby(["target", "horizon", "bucket"]):
            row = payoff_rows.loc[payoff_rows.target.eq(target) & payoff_rows.horizon_sessions.eq(horizon)
                                  & payoff_rows.payoff_display_bin.eq(labels[bucket])]
            valid &= len(row) == 1
            if len(row) == 1:
                row = row.iloc[0]
                valid &= row.n == len(g) and _near(row.mean_predicted_payoff, g.expected_payoff.mean())
                valid &= _near(row.mean_observed_payoff, g.actual_net_horizon_return.mean())
                valid &= _near(row.observed_positive_fraction, g.actual_net_horizon_return.gt(0).mean())
        check(f"{ticker}:payoff_reliability:separate_outer_descriptive_fixed_bins", valid)
        w = intervals.loc[intervals.ticker.eq(ticker)]
        records = tables["weekly_interval_summary"]
        records = records.loc[records.ticker.eq(ticker)]
        valid = len(records) == w.groupby("partial_terminal_interval").ngroups
        for partial, g in w.groupby("partial_terminal_interval"):
            label = "final partial interval" if partial else "completed weekly decision interval"
            row = records.loc[records.interval_type.eq(label)]
            valid &= len(row) == 1
            if len(row) == 1:
                row = row.iloc[0]
                returns = g.policy_net_return
                pos, neg, flat = returns.gt(1e-12).sum(), returns.lt(-1e-12).sum(), returns.abs().le(1e-12).sum()
                valid &= row.intervals == len(g) and (row.positive_intervals, row.negative_intervals, row.flat_intervals) == (pos, neg, flat)
                valid &= _near(row.positive_fraction, pos/len(g)) and _near(row.mean_net_return, returns.mean())
                valid &= _near(row.mean_invested_target, g.target_weight.mean())
        check(f"{ticker}:weekly_policy_summary:disjoint_win_loss_flat_and_partial_terminal", valid)


def audit_bundle(bundle_root=ROOT, require_extended=True, check_manifest=True):
    root = Path(bundle_root)
    manifest = json.loads((root / "manifest.json").read_text())
    reserve = pd.Timestamp(manifest["reserved_start"])
    ledger_data = _csv(root / "data/ledgers.csv", parse_dates=["Date"])
    forecasts = _csv(root / "data/forecasts.csv")
    episodes = _csv(root / "data/trade_episodes.csv", parse_dates=["entry_date", "exit_date"])
    summary = _csv(root / "data/strategy_summary.csv")
    models = _csv(root / "data/annual_model_statistics.csv")
    checks = []
    pending = []

    def check(name, passed, detail=None):
        record = {"check": name, "passed": bool(passed)}
        if detail is not None:
            record["detail"] = detail
        checks.append(record)

    check("only_combined_and_two_declared_benchmarks", set(ledger_data.strategy) == STRATEGIES)
    check("fifteen_declared_independent_ETFs", len(manifest["tickers"]) == 15 and set(ledger_data.ticker) == set(manifest["tickers"]))
    check("only_combined_forecast_variant", set(forecasts.variant) == {"payoff_context"})
    check("no_reserved_account_dates", (ledger_data.Date < reserve).all())
    for column in ["decision_date", "entry_date", "fit_cutoff", "train_latest_label", "calibration_cutoff", "calibration_latest_label", "label_end"]:
        if column in forecasts:
            forecasts[column] = pd.to_datetime(forecasts[column])
    check("no_reserved_forecast_or_label_dates", all((forecasts[column].dropna() < reserve).all() for column in ["decision_date", "entry_date", "label_end"]))
    check("training_outcomes_strictly_matured", (forecasts.train_latest_label < forecasts.fit_cutoff).all())
    eligible = forecasts.loc[forecasts.expected_payoff.notna()]
    check("calibration_outcomes_strictly_matured", (eligible.calibration_latest_label < eligible.calibration_cutoff).all())
    check("next_open_follows_weekly_decision", (forecasts.entry_date > forecasts.decision_date).all())
    check("one_forecast_per_ETF_decision", not forecasts.duplicated(["ticker", "decision_date"]).any())

    ledgers = {}
    for ticker in manifest["tickers"]:
        reference_dates = None
        for strategy in sorted(STRATEGIES):
            data = ledger_data.loc[ledger_data.ticker.eq(ticker) & ledger_data.strategy.eq(strategy)].set_index("Date").sort_index()
            ledgers[(ticker, strategy)] = data
            stamp = f"{ticker}:{strategy}"
            check(stamp + ":nonempty_unique_dates", len(data) > 0 and data.index.is_unique)
            if reference_dates is None:
                reference_dates = data.index
            check(stamp + ":comparison_dates_match", data.index.equals(reference_dates))
            numeric = ["nav", "cash", "shares", "weight", "held_exposure", "costs", "turnover", "trade_value", "terminal_sale", "terminal_cost", "opening_nav", "daily_return"]
            check(stamp + ":finite_long_cash_account", np.isfinite(data[numeric]).all().all() and (data.nav > 0).all() and (data.cash >= -1e-10).all() and data.held_exposure.between(-1e-10, 1 + 1e-10).all())
            previous_cash = data.cash.shift(1)
            previous_cash.iloc[0] = 1.
            check(stamp + ":cashflows_conserve", _near(data.cash, previous_cash - data.trade_value - data.costs + data.terminal_sale))
            check(stamp + ":actual_turnover_fees", _near(data.costs, data.trade_value.abs() * manifest["cost_bps_one_way"] / 10000 + data.terminal_cost))
            returns = data.nav.pct_change()
            returns.iloc[0] = data.nav.iloc[0] - 1
            check(stamp + ":daily_returns_reconcile", _near(returns, data.daily_return))
        data = ledgers[(ticker, "payoff_context")]
        ep = episodes.loc[episodes.ticker.eq(ticker)]
        stat = summary.loc[summary.ticker.eq(ticker)].iloc[0]
        episode_checks = []
        for row in ep.itertuples():
            window = data.loc[row.entry_date:row.exit_date]
            buys = window.trade_value.clip(lower=0).sum()
            sales = -window.trade_value.clip(upper=0).sum() + window.terminal_sale.sum()
            fees = window.costs.sum()
            net = sales - buys - fees
            first = window.iloc[0]
            last = window.iloc[-1]
            episode_checks.append(_near(net, row.net_pnl) and _near(row.net_pnl, last.nav - first.opening_nav)
                                  and _near(row.net_account_return, row.net_pnl / first.opening_nav)
                                  and _near(row.fees, fees)
                                  and int(row.holding_sessions) == int((window.held_exposure > 1e-10).sum()))
        check(f"{ticker}:every_complete_episode_reconciles_cashflows_and_duration", all(episode_checks))
        check(f"{ticker}:episodes_explain_whole_account", _near(ep.net_pnl.sum(), data.nav.iloc[-1] - 1))
        check(f"{ticker}:episode_count_and_winrate", stat.completed_episodes == len(ep) and stat.net_wins == ep.outcome.eq("win").sum() and stat.net_losses == ep.outcome.eq("loss").sum() and _near(stat.win_rate, ep.outcome.eq("win").mean()))
        facts = forecasts.loc[forecasts.ticker.eq(ticker)]
        check(f"{ticker}:decision_clock_statistics", _near(stat.decision_horizon_mean_sessions, facts.horizon.mean()) and _near(stat.mean_target_weight_all_decisions, facts.target_weight.mean()) and _near(stat.mean_target_weight_when_long, facts.loc[facts.signal.eq(1), "target_weight"].mean()) and _near(stat.mean_realized_held_exposure, data.held_exposure.mean()))
        for year, group in facts.groupby("outer_year"):
            record = models.loc[models.ticker.eq(ticker) & models.model_year.eq(year)]
            check(f"{ticker}:{year}:model_facts_match_saved_forecasts", len(record) == 1 and _near(record.iloc[0].decision_horizon_mean_sessions, group.horizon.mean()) and int(record.iloc[0].decision_weeks) == len(group))

    if check_manifest:
        changed = []
        for item in manifest["export_files"]:
            path = root / item["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                changed.append(item["path"])
        check("curated_manifest_hashes_match", not changed, changed)

    # Added table checks are intentionally explicit: missing richer diagnostics
    # must not silently yield an apparently complete audit.
    if require_extended:
        _extended_inputs(root, manifest, forecasts, ledgers, check, pending)
        _diagnostic_exports(root, manifest, ledgers, check, pending)
    failures = [record for record in checks if not record["passed"]]
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAIL" if failures else "INCOMPLETE" if pending else "PASS",
            "checks_passed": len(checks) - len(failures), "checks_total": len(checks),
            "violations": failures, "pending": pending, "checks": checks,
            "scope": "Read-only independent checks of saved pre-2024 focused diagnostics, not model validation or prospective significance.",
            "bundle_root": str(root.resolve())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()
    result = audit_bundle(args.root, require_extended=not args.base_only, check_manifest=not args.skip_manifest)
    print(json.dumps(result, indent=2))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
