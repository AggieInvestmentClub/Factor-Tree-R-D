"""Descriptive paired Sharpe intervals from saved focused-account returns.

This module neither imports model/report engines nor fits, selects, resimulates
or promotes a strategy. It jointly block-resamples the three existing net return
histories within each ETF. The plan is written before real-return computation.

Public API:
    sampled_block_moments(values, starts, block=20)
    validate_return_axes(series_by_strategy)
    paired_sharpe_intervals(returns, replications=9999, block=20, seed=42)
    validate_implementation()

Run with the separate research environment:
    python focused_payoff_context/src/paired_uncertainty.py
The notebook should normally read data/paired_benchmark_uncertainty.csv only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUT = DATA / "ledgers.csv"
PLAN = DATA / "paired_uncertainty_plan.json"
OUTPUT = DATA / "paired_benchmark_uncertainty.csv"
STRATEGIES = ["payoff_context", "etf_hold", "spy_hold"]
TICKERS = "XLK XLF XLE XLV XLI XLY XLP XLU XLB IEF TLT TIP GLD DBC VNQ".split()
RESERVED_START = "2024-01-01"
CAVEAT = (
    "Post-hoc descriptive fixed-fit paired moving-block percentile interval. "
    "Saved fitted-strategy returns are held fixed; model fitting, calibration, "
    "recipe selection and earlier exploratory decisions are not resampled. "
    "Raw 95% intervals are not multiplicity-adjusted and establish neither "
    "prospective significance nor a promotion decision. These 30 market-benchmark "
    "comparisons are not the old 225 variant-versus-v2-baseline Holm family."
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def freeze_plan() -> dict:
    """Create once; refuse silent changes to a declared real-data computation."""
    specification = {
        "schema_version": 1,
        "purpose": "Post-hoc descriptive fixed-fit uncertainty for focused payoff_context versus its ETF and SPY holdings; no model or strategy change.",
        "input_path": "data/ledgers.csv", "input_sha256": digest(INPUT),
        "source_path": "src/paired_uncertainty.py", "source_sha256": digest(Path(__file__)),
        "strategy": STRATEGIES[0], "benchmarks": STRATEGIES[1:], "tickers": TICKERS,
        "comparison_count": 30, "reserved_start": RESERVED_START,
        "bootstrap_replications": 9999, "block_sessions": 20, "batch_size": 128,
        "seed": 42, "rng": "numpy.random.default_rng; reset seed 42 independently for each ETF",
        "pairing": "Within each ETF, payoff_context, etf_hold and spy_hold share exactly the same sampled block starts. No joint cross-ETF inference is asserted.",
        "sampling": "Non-circular overlapping moving blocks. Start indices uniform on 0..n-block inclusive. Draw ceil(n/block) full-block starts; truncate only the final block to retain exactly n original sessions.",
        "statistic": "Sharpe=sqrt(252)*mean(saved net simple daily returns)/sample standard deviation(ddof=1); zero cash/risk-free convention unchanged. Delta=strategy Sharpe minus benchmark Sharpe.",
        "interval": "Raw two-sided 95% percentile interval using NumPy linear 2.5% and 97.5% quantiles of paired delta-Sharpe draws.",
        "zero_variance_convention": "Zero bootstrap sample standard deviation maps Sharpe to zero, matching the existing audit convention; degenerate-draw counts are reported. Observed real histories must have positive sample variance.",
        "accounting": "Use already recorded net daily returns, including their existing costs and terminal liquidation. No account restart, new fees, changed cash yield, resimulation or fresh trading occurs.",
        "eligibility_checks": "All three histories must have identical unique increasing session axes, finite returns and dates strictly before 2024-01-01; no inner join or price/return filling.",
        "validation_before_computation": [
            "Exact block-moment means, ddof=1 variances and paired Sharpe differences versus explicit resampled return vectors, including a partial final block.",
            "Divisible block length, one-session remainder, shorter-than-block software case, zero-return case, and identical-arm pairing checks.",
            "Reject mismatched, duplicate or unordered benchmark session axes; validate actual input axes for all 15 ETFs.",
        ],
        "p_values": "None computed or emitted.",
        "multiplicity": "These 30 market-benchmark comparisons have raw intervals only. Previous Holm-adjusted p values for the different 225-comparison family must not be attached to these intervals.",
        "inference_scope": CAVEAT,
        "outputs": ["data/paired_benchmark_uncertainty.csv"],
        "packages": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    if PLAN.exists():
        existing = json.loads(PLAN.read_text(encoding="utf-8"))
        if {key: value for key, value in existing.items() if key != "declared_at_utc"} != specification:
            raise ValueError("Existing plan differs from source, input or settings; do not overwrite the frozen plan")
        return existing
    plan = {"declared_at_utc": datetime.now(timezone.utc).isoformat(), **specification}
    DATA.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing a concurrently frozen plan.
    with PLAN.open("x", encoding="utf-8") as output:
        json.dump(plan, output, indent=2, ensure_ascii=False, allow_nan=False)
    print("Frozen focused paired-uncertainty plan before real-return computation", flush=True)
    return plan


def sampled_block_moments(values: np.ndarray, starts: np.ndarray, block: int = 20):
    """Exact bootstrap means/variances, preserving a shortened final block.

    Inputs are arms × sessions and draws × ceil(sessions/block). Computation
    adapts the original audit_v3 prefix-moment approach without importing it.
    """
    values = np.asarray(values, dtype=float)
    starts = np.asarray(starts)
    if values.ndim != 2 or starts.ndim != 2 or block < 1:
        raise ValueError("Two-dimensional arrays and a positive block length required")
    if not np.issubdtype(starts.dtype, np.integer):
        raise ValueError("Block starts must be integer indices")
    n = values.shape[1]
    if n < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite return observations required")
    if starts.shape[1] != int(np.ceil(n / block)):
        raise ValueError("Wrong number of blocks for original sample length")
    if (starts < 0).any() or (starts > max(0, n - block)).any():
        raise ValueError("Invalid full-block start indices")
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


def _sharpe_from_moments(means, variances):
    standard_deviations = np.sqrt(variances)
    return np.divide(means * np.sqrt(252), standard_deviations,
                     out=np.zeros_like(means), where=standard_deviations > 0)


def validate_return_axes(series_by_strategy: dict[str, pd.Series]) -> pd.DataFrame:
    """Require exact aligned dates; never silently intersect or fill histories."""
    if set(series_by_strategy) != set(STRATEGIES):
        raise ValueError("Exactly payoff_context, etf_hold and spy_hold required")
    reference = series_by_strategy[STRATEGIES[0]].index
    if not isinstance(reference, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex required")
    if len(reference) < 20 or not reference.is_unique or not reference.is_monotonic_increasing:
        raise ValueError("At least twenty unique increasing matched sessions required")
    if reference.hasnans or (reference >= pd.Timestamp(RESERVED_START)).any():
        raise ValueError("Missing or reserved session date detected")
    for name in STRATEGIES:
        if not series_by_strategy[name].index.equals(reference):
            raise ValueError(f"Unmatched session axis: {name}")
    frame = pd.DataFrame({name: series_by_strategy[name] for name in STRATEGIES})
    matrix = frame.to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or (matrix <= -1).any():
        raise ValueError("Finite valid simple daily returns required; no filling")
    return frame


def paired_sharpe_intervals(returns: pd.DataFrame, replications: int = 9999,
                            block: int = 20, seed: int = 42) -> list[dict]:
    """Two raw paired intervals from one joint three-history resample per ETF."""
    if list(returns.columns) != STRATEGIES:
        raise ValueError("Columns must be ordered payoff_context, etf_hold, spy_hold")
    returns = validate_return_axes({name: returns[name] for name in STRATEGIES})
    if replications < 2 or block < 1 or block > len(returns):
        raise ValueError("Valid replication and block counts required")
    values = returns.to_numpy(dtype=float).T
    observed_variances = values.var(axis=1, ddof=1)
    if (observed_variances <= 0).any():
        raise ValueError("Observed real-history Sharpe undefined for a constant account")
    observed = _sharpe_from_moments(values.mean(axis=1), observed_variances)
    n = values.shape[1]
    rng = np.random.default_rng(seed)
    draws = np.empty((replications, 2))
    degenerate = np.zeros(2, dtype=int)
    for first in range(0, replications, 128):
        count = min(128, replications - first)
        starts = rng.integers(0, n - block + 1, size=(count, int(np.ceil(n / block))))
        means, variances = sampled_block_moments(values, starts, block)
        sharpes = _sharpe_from_moments(means, variances)
        draws[first:first+count] = (sharpes[0:1] - sharpes[1:]).T
        for index in range(2):
            degenerate[index] += int(((variances[0] <= 0) | (variances[index+1] <= 0)).sum())
    result = []
    for index, benchmark in enumerate(STRATEGIES[1:]):
        low, high = np.quantile(draws[:, index], [.025, .975], method="linear")
        result.append({
            "strategy": STRATEGIES[0], "benchmark": benchmark,
            "start": returns.index[0].strftime("%Y-%m-%d"),
            "end": returns.index[-1].strftime("%Y-%m-%d"), "days": n,
            "strategy_sharpe": float(observed[0]), "benchmark_sharpe": float(observed[index+1]),
            "delta_sharpe": float(observed[0] - observed[index+1]),
            "delta_sharpe_95pct_low": float(low), "delta_sharpe_95pct_high": float(high),
            "bootstrap_replications": replications, "block_sessions": block, "seed": seed,
            "degenerate_bootstrap_draws": int(degenerate[index]),
            "interval_type": "raw_two_sided_95pct_percentile",
            "uncertainty_scope": "posthoc_descriptive_fixed_fit",
            "joint_three_account_draws": True, "uncertainty_caveat": CAVEAT,
        })
    return result


def validate_implementation() -> dict:
    """Compare moment acceleration with materialized draws, not itself."""
    rng = np.random.default_rng(314159)
    cases = []
    for n, block in [(40, 20), (43, 20), (21, 20), (7, 20)]:
        values = rng.normal(.0005, .015, size=(3, n))
        values[1] += .3 * values[0]
        starts = rng.integers(0, max(1, n-block+1), size=(31, int(np.ceil(n/block))))
        means, variances = sampled_block_moments(values, starts, block)
        direct_means, direct_variances, direct_deltas = [], [], []
        for draw_starts in starts:
            indices = np.concatenate([np.arange(start, min(start+block, n)) for start in draw_starts])[:n]
            materialized = values[:, indices]
            assert materialized.shape == values.shape
            direct_means.append(materialized.mean(axis=1))
            direct_variances.append(materialized.var(axis=1, ddof=1))
            sd = materialized.std(axis=1, ddof=1)
            direct_sharpe = materialized.mean(axis=1) * np.sqrt(252) / sd
            direct_deltas.append(direct_sharpe[0] - direct_sharpe[1:])
        np.testing.assert_allclose(means, np.asarray(direct_means).T, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(variances, np.asarray(direct_variances).T, rtol=1e-12, atol=1e-18)
        fast = _sharpe_from_moments(means, variances)
        np.testing.assert_allclose((fast[0:1]-fast[1:]).T, direct_deltas, rtol=1e-11, atol=1e-12)
        cases.append({"sessions": n, "block": block, "last_block_sessions": n % block or block, "draws": len(starts), "matches_materialized": True})
    zero_means, zero_variances = sampled_block_moments(np.zeros((3, 43)), np.array([[0, 5, 10]]), 20)
    assert np.array_equal(_sharpe_from_moments(zero_means, zero_variances), np.zeros((3, 1)))
    dates = pd.bdate_range("2010-01-01", periods=43)
    signal = pd.Series(rng.normal(.0005, .02, 43), index=dates)
    third = pd.Series(rng.normal(.0001, .01, 43), index=dates)
    frame = validate_return_axes({"payoff_context": signal, "etf_hold": signal.copy(), "spy_hold": third})
    identical = paired_sharpe_intervals(frame, replications=73)[0]
    assert identical["delta_sharpe"] == identical["delta_sharpe_95pct_low"] == identical["delta_sharpe_95pct_high"] == 0.
    rejection_checks = 0
    for altered in [signal.iloc[:-1], signal.iloc[::-1], pd.concat([signal, signal.iloc[-1:]])]:
        try:
            validate_return_axes({"payoff_context": signal, "etf_hold": altered, "spy_hold": third})
        except ValueError:
            rejection_checks += 1
        else:
            raise AssertionError("Invalid benchmark date axis was accepted")
    assert rejection_checks == 3
    return {"status": "passed", "materialized_equivalence_cases": cases,
            "identical_arm_paired_delta_exactly_zero": True, "invalid_axes_rejected": rejection_checks,
            "zero_return_bootstrap_convention_verified": True}


def main() -> None:
    plan = freeze_plan()
    plan_hash = digest(PLAN)
    checks = validate_implementation()
    print(json.dumps(checks), flush=True)
    frame = pd.read_csv(INPUT, usecols=["ticker", "strategy", "Date", "daily_return"],
                        parse_dates=["Date"], float_precision="round_trip")
    if set(frame.ticker) != set(TICKERS) or set(frame.strategy) != set(STRATEGIES):
        raise ValueError("Input does not contain exactly the fifteen ETFs and three declared strategies")
    if frame.Date.isna().any() or (frame.Date >= pd.Timestamp(RESERVED_START)).any():
        raise ValueError("Reserved or missing dates present; no observed-return statistics computed")
    if frame.duplicated(["ticker", "strategy", "Date"]).any():
        raise ValueError("Duplicate account/date records")
    aligned = {}
    for ticker in TICKERS:
        subset = frame.loc[frame.ticker.eq(ticker)]
        aligned[ticker] = validate_return_axes({name: subset.loc[subset.strategy.eq(name)].set_index("Date").daily_return for name in STRATEGIES})
    if digest(INPUT) != plan["input_sha256"]:
        raise ValueError("Input changed after the plan was frozen")
    results = []
    for ticker in TICKERS:
        rows = paired_sharpe_intervals(aligned[ticker], replications=plan["bootstrap_replications"],
                                      block=plan["block_sessions"], seed=plan["seed"])
        for row in rows:
            results.append({"ticker": ticker, **row,
                            "input_sha256": plan["input_sha256"], "plan_sha256": plan_hash,
                            "moment_equivalence_validated": True, "matched_axes_validated": True})
        print(f"Paired fixed-fit intervals: {ticker}", flush=True)
    output = pd.DataFrame(results)
    if len(output) != 30 or output.duplicated(["ticker", "benchmark"]).any():
        raise AssertionError("Wrong or duplicate comparison rows")
    if not np.isfinite(output[["delta_sharpe", "delta_sharpe_95pct_low", "delta_sharpe_95pct_high"]].to_numpy()).all():
        raise AssertionError("Nonfinite comparison statistics")
    if digest(INPUT) != plan["input_sha256"] or digest(PLAN) != plan_hash:
        raise ValueError("Input or frozen plan changed during computation")
    if any(name.startswith("p_") or "holm" in name for name in output.columns):
        raise AssertionError("P-values or unrelated multiplicity fields must not be emitted")
    output.to_csv(OUTPUT, index=False)
    print(json.dumps({"status": "complete", "comparisons": len(output),
                      "output": str(OUTPUT), "input_unchanged": True, "plan_unchanged": True,
                      "reserved_rows_used": 0, "no_models_fit": True, "no_p_values_emitted": True}), flush=True)


if __name__ == "__main__":
    main()
