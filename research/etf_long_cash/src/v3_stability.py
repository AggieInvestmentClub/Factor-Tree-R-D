"""Read-only annual and common-window diagnostics for completed v3 ledgers.

No fitting, hypothesis selection, reserve data or account simulation occurs.
Window NAV is a measurement normalization of existing net daily returns, with
the account state carried into the first included session explicitly reported.
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
COMMON_START = "2012-01-09"
COMMON_END = "2023-12-29"
WINDOWS = {
    "common_full": (COMMON_START, COMMON_END),
    "common_early": (COMMON_START, "2017-12-29"),
    "common_late": ("2018-01-01", COMMON_END),
}
REFERENCE = "causal_always_long_vol_reference"


def read_ledger(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Invalid date index: {path}")
    if frame.index.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError(f"Reserved dates in saved ledger: {path}")
    required = ["daily_return", "nav", "cash", "shares", "weight", "held_exposure", "turnover", "costs", "trade_value", "terminal_sale"]
    if not np.isfinite(frame[required].to_numpy()).all() or frame.nav.le(0).any() or frame.daily_return.le(-1).any():
        raise ValueError(f"Nonfinite or invalid account data: {path}")
    return frame


def window_stats(full: pd.DataFrame, start, end) -> dict:
    frame = full.loc[start:end]
    if frame.empty:
        raise ValueError("Empty measurement window")
    r = frame.daily_return
    # NAV=1 immediately BEFORE the first included session. The first saved
    # return remains intact, including any carry-in overnight movement/trade.
    normalized_nav = (1 + r).cumprod()
    years = len(frame) / 252
    stdev = r.std(ddof=1)
    earlier = full.loc[full.index < frame.index[0]]
    has_previous = len(earlier) > 0
    prior = earlier.iloc[-1] if has_previous else None
    prior_nav = float(prior.nav) if has_previous else 1.0
    inferred_prior_nav = float(frame.nav.iloc[0] / (1 + r.iloc[0]))
    if not np.isclose(prior_nav, inferred_prior_nav, rtol=1e-10, atol=1e-10):
        raise AssertionError("Carried prior NAV inconsistent with saved first-day return")
    tail_count = max(1, int(np.ceil(.05 * len(r))))
    worst_tail = np.sort(r.to_numpy())[:tail_count]
    return {
        "start": str(frame.index[0].date()), "end": str(frame.index[-1].date()), "days": len(frame),
        "sharpe": float(r.mean() / stdev * np.sqrt(252)) if stdev > 0 else 0.,
        "total_return": float(normalized_nav.iloc[-1] - 1),
        "cagr": float(normalized_nav.iloc[-1] ** (1 / years) - 1),
        "volatility": float(stdev * np.sqrt(252)),
        "max_drawdown": float((normalized_nav / normalized_nav.cummax().clip(lower=1) - 1).min()),
        "average_exposure": float(frame.held_exposure.mean()),
        "turnover_per_year": float(frame.turnover.sum() / years),
        "trade_events": int(frame.trade_value.abs().gt(1e-10).sum() + frame.terminal_sale.gt(0).sum()),
        "fee_cash_original_account_currency": float(frame.costs.sum()),
        "fees_per_window_start_NAV": float(frame.costs.sum() / prior_nav),
        "worst_daily_return": float(r.min()),
        "var5_daily_return": float(r.quantile(.05)),
        "cvar5_daily_return": float(worst_tail.mean()),
        "cvar5_tail_observations": tail_count,
        "had_prior_ledger_session": bool(has_previous),
        "prior_ledger_date": str(earlier.index[-1].date()) if has_previous else None,
        "carry_in_nav_original": prior_nav,
        "carry_in_shares_original": float(prior.shares) if has_previous else 0.,
        "carry_in_exposure": float(prior.weight) if has_previous else 0.,
        "carry_in_cash_fraction": float(prior.cash / prior_nav) if has_previous else 1.,
        "carry_in_position": bool(has_previous and prior.shares > 1e-10),
        "first_included_daily_return": float(r.iloc[0]),
        "first_included_turnover": float(frame.turnover.iloc[0]),
        "actual_terminal_liquidation_in_window": bool(frame.terminal_sale.gt(0).any()),
        "normalization": "NAV 1 immediately before first included session; compound unchanged saved daily returns; no resimulation or cost reset",
    }


def add_comparisons(row: dict, baseline: dict, reference: dict) -> dict:
    if (row["start"], row["end"], row["days"]) != (baseline["start"], baseline["end"], baseline["days"]):
        raise AssertionError("Unmatched baseline comparison dates")
    if (row["start"], row["end"], row["days"]) != (reference["start"], reference["end"], reference["days"]):
        raise AssertionError("Unmatched causal reference comparison dates")
    return dict(row, baseline_sharpe=baseline["sharpe"], delta_sharpe_vs_baseline=row["sharpe"]-baseline["sharpe"],
                causal_vol_reference_sharpe=reference["sharpe"], delta_sharpe_vs_causal_vol_reference=row["sharpe"]-reference["sharpe"],
                delta_exposure_vs_baseline=row["average_exposure"]-baseline["average_exposure"],
                delta_exposure_vs_causal_vol_reference=row["average_exposure"]-reference["average_exposure"],
                delta_turnover_vs_baseline=row["turnover_per_year"]-baseline["turnover_per_year"],
                delta_turnover_vs_causal_vol_reference=row["turnover_per_year"]-reference["turnover_per_year"],
                delta_fees_per_start_NAV_vs_baseline=row["fees_per_window_start_NAV"]-baseline["fees_per_window_start_NAV"],
                delta_fees_per_start_NAV_vs_causal_vol_reference=row["fees_per_window_start_NAV"]-reference["fees_per_window_start_NAV"])


def primary_summary(common: pd.DataFrame, primary: list[str]) -> list[dict]:
    summary = []
    tol = 1e-10
    for variant in primary:
        entry = {"variant": variant, "windows": []}
        for window in WINDOWS:
            d = common.loc[common.variant.eq(variant) & common.window.eq(window)]
            entry["windows"].append({
                "window": window, "independent_ETFs": len(d),
                "above_baseline_sharpe_count": int(d.delta_sharpe_vs_baseline.gt(tol).sum()),
                "below_baseline_sharpe_count": int(d.delta_sharpe_vs_baseline.lt(-tol).sum()),
                "above_causal_vol_reference_sharpe_count": int(d.delta_sharpe_vs_causal_vol_reference.gt(tol).sum()),
                "median_individual_delta_sharpe_vs_baseline": float(d.delta_sharpe_vs_baseline.median()),
                "median_individual_delta_sharpe_vs_causal_vol_reference": float(d.delta_sharpe_vs_causal_vol_reference.median()),
                "median_individual_exposure": float(d.average_exposure.median()),
                "median_individual_annual_turnover": float(d.turnover_per_year.median()),
                "median_individual_delta_exposure_vs_causal_vol_reference": float(d.delta_exposure_vs_causal_vol_reference.median()),
                "median_individual_delta_turnover_vs_causal_vol_reference": float(d.delta_turnover_vs_causal_vol_reference.median()),
                "median_individual_delta_fees_per_start_NAV_vs_causal_vol_reference": float(d.delta_fees_per_start_NAV_vs_causal_vol_reference.median()),
                "median_individual_worst_daily_return": float(d.worst_daily_return.median()),
                "median_individual_cvar5_daily_return": float(d.cvar5_daily_return.median()),
                "interpretation": "Descriptive summary of 15 independent accounts; never a portfolio return or portfolio Sharpe",
            })
        early = common.loc[common.variant.eq(variant) & common.window.eq("common_early")].set_index("ticker")
        late = common.loc[common.variant.eq(variant) & common.window.eq("common_late")].set_index("ticker")
        e, l = early.delta_sharpe_vs_baseline, late.delta_sharpe_vs_baseline
        entry["split_delta_sharpe_signs"] = {
            "positive_in_both_ETFs": list(e.index[e.gt(tol) & l.gt(tol)]),
            "negative_in_both_ETFs": list(e.index[e.lt(-tol) & l.lt(-tol)]),
            "positive_early_negative_late_ETFs": list(e.index[e.gt(tol) & l.lt(-tol)]),
            "negative_early_positive_late_ETFs": list(e.index[e.lt(-tol) & l.gt(tol)]),
            "near_zero_in_either_ETFs": list(e.index[e.abs().le(tol) | l.abs().le(tol)]),
        }
        summary.append(entry)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-directory", default="runs/v3_1", help="Explicit completed run directory relative to project root")
    args=parser.parse_args()
    run_directory=(ROOT / args.run_directory).resolve()
    config_path = run_directory / "experiment_config.json"
    config = json.loads(config_path.read_text())
    tickers = config["universe"]
    variants = list(config["variants"])
    primary = config["primary_ablation_variants"]
    upstream = pd.read_csv(ROOT / "reports/v3_metrics.csv")
    upstream = upstream.loc[upstream.cost_bps.eq(5)].set_index(["ticker", "variant"])
    annual_rows, common_rows, sources, metric_checks = [], [], [], []
    for ticker in tickers:
        source_frames = {}
        for variant in variants:
            path = run_directory / ticker / f"ledger_{variant}_5bps.csv"
            source_frames[variant] = read_ledger(path)
            sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        reference_path = ROOT / f"runs/v2/{ticker}/ledger_benchmark_vol_always_5bps.csv"
        reference = read_ledger(reference_path)
        sources.append({"path": reference_path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest()})
        baseline = source_frames["baseline"]
        if baseline.index[0] > pd.Timestamp(COMMON_START) or baseline.index[-1] < pd.Timestamp(COMMON_END):
            raise AssertionError("Declared common window is outside baseline coverage")
        for variant, frame in source_frames.items():
            if not frame.index.equals(baseline.index):
                raise AssertionError(f"Unmatched v3 ledger dates: {ticker} {variant}")
            if not reference.index.equals(baseline.index):
                raise AssertionError(f"Unmatched reference dates: {ticker}")
            whole = window_stats(frame, frame.index[0], frame.index[-1])
            expected = upstream.loc[(ticker, variant)]
            for metric in ["sharpe", "cagr", "volatility", "max_drawdown", "average_exposure", "turnover_per_year", "total_return"]:
                if not np.isclose(whole[metric], expected[metric], rtol=1e-10, atol=1e-10):
                    raise AssertionError(f"Saved metric mismatch: {ticker} {variant} {metric}")
                metric_checks.append({"ticker": ticker, "variant": variant, "metric": metric, "passed": True})
            for year in sorted(frame.index.year.unique()):
                start, end = f"{year}-01-01", f"{year}-12-31"
                row = dict(ticker=ticker, variant=variant, cost_bps=5, year=int(year), **window_stats(frame, start, end))
                row = add_comparisons(row, window_stats(baseline, start, end), window_stats(reference, start, end))
                annual_rows.append(row)
            for window, (start, end) in WINDOWS.items():
                if window != "common_full" and variant not in primary:
                    continue
                row = dict(ticker=ticker, variant=variant, source_kind="v3_variant", cost_bps=5, window=window,
                           requested_start=start, requested_end=end, **window_stats(frame, start, end))
                common_rows.append(add_comparisons(row, window_stats(baseline, start, end), window_stats(reference, start, end)))
        for window, (start,end) in WINDOWS.items():
            stats = window_stats(reference,start,end)
            row = dict(ticker=ticker,variant=REFERENCE,source_kind="v2_causal_reference",cost_bps=5,window=window,
                       requested_start=start,requested_end=end,**stats)
            common_rows.append(add_comparisons(row,window_stats(baseline,start,end),stats))
        print(f"{ticker}: 16 variants, annual windows and matched common-period diagnostics complete",flush=True)
    annual = pd.DataFrame(annual_rows)
    common = pd.DataFrame(common_rows)
    annual["analysis_run_directory"] = run_directory.relative_to(ROOT).as_posix()
    common["analysis_run_directory"] = run_directory.relative_to(ROOT).as_posix()
    reports = ROOT / "reports"
    annual_path = reports / "v3_annual_stability.csv"
    common_path = reports / "v3_common_period.csv"
    annual.to_csv(annual_path,index=False)
    common.to_csv(common_path,index=False)
    summaries = primary_summary(common,primary)
    findings=[]
    for entry in summaries:
        full=entry["windows"][0];signs=entry["split_delta_sharpe_signs"]
        findings.append(f"{entry['variant']}: common-period net Sharpe exceeds v3 baseline in {full['above_baseline_sharpe_count']}/15 separate ETF accounts and causal always-long vol reference in {full['above_causal_vol_reference_sharpe_count']}/15. {len(signs['positive_in_both_ETFs'])}/15 show positive delta versus baseline in both early and late subperiods. These are descriptive counts, not promotions or significance claims.")
    summary = {
        "schema_version":1,"generated_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS",
        "scope":"Saved historical ledgers only; no fitting, source/config alteration, reserve observations, resimulation, winner selection or promotion",
        "actual_run_directory":run_directory.relative_to(ROOT).as_posix(),
        "source_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "annual_rows":len(annual),"common_period_rows":len(common),"independent_ETFs":len(tickers),"variants":len(variants),
        "cost_bps":5,"common_period_requested":WINDOWS,"reserved_rows_used":0,
        "saved_whole_period_metric_checks_passed":len(metric_checks),
        "continuing_strategy_measurement":{
            "NAV":"Set measurement NAV to 1 immediately before the first included return; compound the existing saved net daily returns without resetting the strategy.",
            "carry_in":"Prior ledger NAV, shares, cash fraction and exposure are reported. The first included daily return retains overnight movement, original trades and actual costs.",
            "boundaries":"No fresh initial allocation or artificial liquidation at year/subperiod boundaries. Existing final liquidation at the original 2023 endpoint remains included.",
            "fees":"Raw fees retain original-account NAV-1 currency. fees_per_window_start_NAV divides those recorded fees by actual prior-session account NAV; it is a measurement normalization, not a recomputed execution cost or return drag.",
            "drawdown":"Window drawdown resets only the measurement high-water mark to initial normalized NAV 1; it does not alter source positions or trades.",
            "Sharpe":"sqrt(252)*mean(existing net simple daily return)/sample std(ddof=1), zero risk-free/cash convention.",
            "tail":"Worst daily return, empirical 5th percentile and mean of the worst ceil(5%*n) daily returns. The latter is reported as a signed daily-return CVaR/expected-shortfall estimate; no independent-sample or tail-model claim."
        },
        "primary_four_arm_summaries":summaries,"findings":findings,
        "limitations":["All pre-2024 history is exploratory and repeatedly inspected; subperiods are stability diagnostics, not new untouched holdouts.",
                       "No aggregate portfolio returns, portfolio Sharpe or shared-capital account is constructed. Cross-ETF counts and medians summarize separate accounts only.",
                       "Annual and split samples are short and serially dependent; these diagnostics add no multiplicity correction or statistical significance.",
                       "Historical Sharpe changes can reflect exposure and cost differences; compare the recorded exposure, turnover, fees, drawdown and tail outcomes.",
                       "The causal always-long vol reference is the saved v2 benchmark with original weekly volatility sizing; it is not a new fit or optimized risk match."],
        "outputs":[{"path":p.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in [annual_path,common_path]],
        "source_ledgers":sources
    }
    (reports / "v3_stability_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(f"PASS: {len(annual)} annual rows, {len(common)} common-window rows, {len(metric_checks)} saved metric checks; no reserve rows",flush=True)
    for finding in findings:print(finding,flush=True)


if __name__ == "__main__":
    main()
