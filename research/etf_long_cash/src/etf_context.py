"""Fixed, causal ETF-specific market context for a separate incremental study.

No macro releases, current holdings, target labels, fitted transformations or
reserved-period observations enter these definitions. Each value has an explicit
availability flag. An unavailable value is represented by zero plus flag=0;
prices themselves are never forward/backfilled or synthetically extended.

Public interface:
    build_context(prices, ticker) -> Date-indexed DataFrame of ctx_* columns
    load_context(ticker) -> the saved equivalent DataFrame

All input prices are truncated strictly before 2024-01-01 before any return,
rolling window, cross-sectional statistic or correlation is calculated.
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
RESERVED_START = "2024-01-01"
FEATURE_VERSION = "etf_context_v1"
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
CYCLICALS = ["XLK", "XLF", "XLE", "XLI", "XLY", "XLB"]
DEFENSIVES = ["XLV", "XLP", "XLU"]
BONDS = ["IEF", "TLT", "TIP"]
UNIVERSE = SECTORS + BONDS + ["GLD", "DBC", "VNQ"]
SOURCES = UNIVERSE + ["SPY", "UUP"]


def _price_frame(frame: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Only date parsing/slicing happens before the mandatory reserve gate."""
    if "Date" in frame.columns:
        frame = frame.set_index("Date")
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    frame = frame.set_axis(index)
    frame = frame.loc[index < pd.Timestamp(cutoff)].copy()
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Context prices require unique, increasing session dates")
    return frame


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b.abs() > 1e-16)


def _definitions(ticker: str) -> list[dict]:
    """The finite feature menu is written before any incremental model fit."""
    def spec(name, formula, inputs, unit, lookback, caveat):
        return {"ticker": ticker, "feature_name": name, "kind": "value", "formula": formula,
                "inputs": "|".join(inputs), "unit": unit, "lookback_sessions": lookback,
                "minimum_history_sessions": 50 if name == "ctx_sector_breadth50" else lookback + 1,
                "interpretation_caveat": caveat}
    if ticker in SECTORS:
        return [
            spec("ctx_own_vs_other_sectors_mom20", "own r20 - equal-weight mean r20 of other eight sector ETFs", SECTORS, "decimal relative return", 20, "Fixed selected sector cohort; not historical holdings or a survivorship-neutral sector universe."),
            spec("ctx_sector_momentum_dispersion20", "population std(ddof=0) of nine sector ETF r20 values", SECTORS, "decimal cross-sectional dispersion", 20, "All nine momentum values must exist. Dispersion is descriptive, not a fitted regime classifier."),
            spec("ctx_sector_breadth50", "fraction of nine sector adjusted closes above their own trailing 50-session mean", SECTORS, "fraction in [0,1]", 50, "Every close and 50-session mean must exist. This is ETF-sector breadth, not constituent-stock breadth."),
            spec("ctx_cyclical_defensive_mom20", "mean r20(XLK,XLF,XLE,XLI,XLY,XLB) - mean r20(XLV,XLP,XLU)", SECTORS, "decimal relative return", 20, "Static economic grouping declared before fitting; equal weights; groups include the subject sector when applicable."),
            spec("ctx_own_cyclical_spread_corr63", "corr63(own daily return, equal-weight cyclical daily return minus defensive daily return)", SECTORS, "correlation in [-1,1]", 63, "Own sector participates in its group, creating a mechanical component. This is contemporaneous past-data context, not causal attribution."),
        ]
    if ticker in BONDS:
        return [
            spec("ctx_nominal_duration_mom20", "r20(TLT) - r20(IEF)", ["TLT", "IEF"], "decimal relative return", 20, "Longer-minus-intermediate Treasury ETF momentum; not a measured duration, yield spread or yield-curve estimate."),
            spec("ctx_inflation_linked_mom20", "r20(TIP) - r20(IEF)", ["TIP", "IEF"], "decimal relative return", 20, "Inflation-linked minus nominal ETF returns have duration/real-rate confounds; this is not breakeven inflation or an inflation surprise."),
            spec("ctx_equity_bond_corr63", f"corr63(r1({ticker}), r1(SPY))", [ticker, "SPY"], "correlation in [-1,1]", 63, "Historical return correlation only; no assumed stable stock/bond hedge."),
            spec("ctx_own_vs_other_bonds_mom20", f"r20({ticker}) - mean r20 of other two ETFs in IEF,TLT,TIP", BONDS, "decimal relative return", 20, "Requires both other listed instruments; differing duration and inflation exposure prevent a pure factor interpretation."),
            spec("ctx_bond_equity_relative20", f"r20({ticker}) - r20(SPY)", [ticker, "SPY"], "decimal relative return", 20, "Relative asset return; not excess return over cash or a credit spread."),
        ]
    if ticker == "GLD":
        return [
            spec("ctx_usd_mom20", "r20(UUP)", ["UUP"], "decimal return", 20, "Real listed UUP ETF history only, starting 2007-03-01; no synthetic prelisting dollar history."),
            spec("ctx_gold_usd_corr63", "corr63(r1(GLD), r1(UUP))", ["GLD", "UUP"], "correlation in [-1,1]", 63, "ETF price-return relationship; UUP is a currency-futures fund, not the spot dollar index."),
            spec("ctx_gold_bond_relative20", "r20(GLD) - r20(IEF)", ["GLD", "IEF"], "decimal relative return", 20, "Gold ETF versus nominal Treasury ETF momentum; not a real-yield estimate."),
            spec("ctx_gold_bond_corr63", "corr63(r1(GLD), r1(IEF))", ["GLD", "IEF"], "correlation in [-1,1]", 63, "Historical return association only; no stable safe-haven assumption."),
            spec("ctx_gold_vs_commodity_mom20", "r20(GLD) - r20(DBC)", ["GLD", "DBC"], "decimal relative return", 20, "Requires actual DBC listing history; not gold futures basis or commodity carry."),
        ]
    if ticker == "DBC":
        return [
            spec("ctx_energy_materials_mom20", "mean(r20(XLE), r20(XLB))", ["XLE", "XLB"], "decimal return", 20, "Equity-sector proxy containing equity/company risks; not physical commodity returns or futures carry."),
            spec("ctx_commodity_vs_energy_materials20", "r20(DBC) - mean(r20(XLE), r20(XLB))", ["DBC", "XLE", "XLB"], "decimal relative return", 20, "Relative ETF price behavior; neither futures roll yield, term structure nor a carry measure is available."),
            spec("ctx_usd_mom20", "r20(UUP)", ["UUP"], "decimal return", 20, "Real listed UUP history only; unavailable prelisting history is flagged, never synthesized."),
            spec("ctx_commodity_usd_corr63", "corr63(r1(DBC), r1(UUP))", ["DBC", "UUP"], "correlation in [-1,1]", 63, "Historical ETF return relationship; not proof of dollar-driven commodity changes."),
            spec("ctx_energy_vs_materials_mom20", "r20(XLE) - r20(XLB)", ["XLE", "XLB"], "decimal relative return", 20, "Energy-equity versus materials-equity rotation; cannot isolate commodity supply or inflation shocks."),
            spec("ctx_cyclical_defensive_mom20", "mean r20(XLK,XLF,XLE,XLI,XLY,XLB) - mean r20(XLV,XLP,XLU)", SECTORS, "decimal relative return", 20, "Broad equity-sector cyclical backdrop; fixed economic grouping, not fitted business-cycle state."),
        ]
    if ticker == "VNQ":
        return [
            spec("ctx_property_vs_bonds_mom20", "r20(VNQ) - r20(IEF)", ["VNQ", "IEF"], "decimal relative return", 20, "Listed REIT versus Treasury ETF momentum; not a property cap-rate spread."),
            spec("ctx_property_ief_beta63", "cov63(r1(VNQ),r1(IEF)) / var63(r1(IEF))", ["VNQ", "IEF"], "return beta", 63, "Historical sensitivity to an IEF price proxy, not an estimated real-estate duration or yield sensitivity."),
            spec("ctx_property_equity_corr63", "corr63(r1(VNQ),r1(SPY))", ["VNQ", "SPY"], "correlation in [-1,1]", 63, "Listed-property/equity return association; not private property fundamentals."),
            spec("ctx_property_vs_defensives_mom20", "r20(VNQ) - mean(r20(XLP),r20(XLU))", ["VNQ", "XLP", "XLU"], "decimal relative return", 20, "Relative to defensive and interest-sensitive equity proxies; a declared hypothesis rather than a holdings map."),
            spec("ctx_nominal_duration_mom20", "r20(TLT) - r20(IEF)", ["TLT", "IEF"], "decimal relative return", 20, "Nominal-bond price backdrop; not a directly observed mortgage or cap-rate factor."),
        ]
    raise ValueError(f"Unsupported ETF {ticker}")


def build_context(prices: dict[str, pd.DataFrame], ticker: str, reserved_start: str = RESERVED_START) -> pd.DataFrame:
    """Build a fixed context pack using completed closes and earlier observations.

    This function performs no I/O, looks at no labels and fits no parameters.
    Missing sources remain missing in price space. Each output value is finite
    and paired with an availability flag so prelisting data is distinguishable.
    """
    if ticker not in prices:
        raise ValueError(f"No price frame for {ticker}")
    truncated = {symbol: _price_frame(frame, reserved_start) for symbol, frame in prices.items()}
    own = truncated[ticker]
    if own.empty:
        raise ValueError(f"No pre-reserve data for {ticker}")
    end = own.index[-1]
    if "SPY" in truncated:
        calendar = truncated["SPY"].index
    else:
        calendar = own.index
        for frame in truncated.values():
            calendar = calendar.union(frame.index)
    calendar = calendar[calendar <= end].sort_values()
    closes = {}
    for symbol in SOURCES:
        frame = truncated.get(symbol)
        if frame is None:
            closes[symbol] = pd.Series(np.nan, index=calendar, dtype=float)
        else:
            key = "adj_close" if "adj_close" in frame else "Adj Close"
            if key not in frame:
                raise ValueError(f"{symbol} requires adjusted close")
            closes[symbol] = pd.to_numeric(frame[key], errors="raise").reindex(calendar)
    close = pd.DataFrame(closes, index=calendar)
    ret = close.pct_change(fill_method=None)
    mom = close.pct_change(20, fill_method=None)
    def mean(frame, names):
        return frame[names].mean(axis=1, skipna=False)
    def corr(a, b):
        return ret[a].rolling(63, min_periods=63).corr(ret[b])
    cyc_mom = mean(mom, CYCLICALS) - mean(mom, DEFENSIVES)
    duration = mom.TLT - mom.IEF
    values = {}
    if ticker in SECTORS:
        sma50 = close[SECTORS].rolling(50, min_periods=50).mean()
        panel_ready = close[SECTORS].notna().all(axis=1) & sma50.notna().all(axis=1)
        breadth = (close[SECTORS] > sma50).mean(axis=1).where(panel_ready)
        cyc_daily = mean(ret, CYCLICALS) - mean(ret, DEFENSIVES)
        values = {
            "ctx_own_vs_other_sectors_mom20": mom[ticker] - mean(mom, [t for t in SECTORS if t != ticker]),
            "ctx_sector_momentum_dispersion20": mom[SECTORS].std(axis=1, ddof=0, skipna=False),
            "ctx_sector_breadth50": breadth,
            "ctx_cyclical_defensive_mom20": cyc_mom,
            "ctx_own_cyclical_spread_corr63": ret[ticker].rolling(63, min_periods=63).corr(cyc_daily),
        }
    elif ticker in BONDS:
        values = {
            "ctx_nominal_duration_mom20": duration,
            "ctx_inflation_linked_mom20": mom.TIP - mom.IEF,
            "ctx_equity_bond_corr63": corr(ticker, "SPY"),
            "ctx_own_vs_other_bonds_mom20": mom[ticker] - mean(mom, [t for t in BONDS if t != ticker]),
            "ctx_bond_equity_relative20": mom[ticker] - mom.SPY,
        }
    elif ticker == "GLD":
        values = {"ctx_usd_mom20": mom.UUP, "ctx_gold_usd_corr63": corr("GLD", "UUP"),
                  "ctx_gold_bond_relative20": mom.GLD - mom.IEF, "ctx_gold_bond_corr63": corr("GLD", "IEF"),
                  "ctx_gold_vs_commodity_mom20": mom.GLD - mom.DBC}
    elif ticker == "DBC":
        basket = mean(mom, ["XLE", "XLB"])
        values = {"ctx_energy_materials_mom20": basket, "ctx_commodity_vs_energy_materials20": mom.DBC - basket,
                  "ctx_usd_mom20": mom.UUP, "ctx_commodity_usd_corr63": corr("DBC", "UUP"),
                  "ctx_energy_vs_materials_mom20": mom.XLE - mom.XLB, "ctx_cyclical_defensive_mom20": cyc_mom}
    elif ticker == "VNQ":
        values = {"ctx_property_vs_bonds_mom20": mom.VNQ - mom.IEF,
                  "ctx_property_ief_beta63": _safe_divide(ret.VNQ.rolling(63, min_periods=63).cov(ret.IEF), ret.IEF.rolling(63, min_periods=63).var()),
                  "ctx_property_equity_corr63": corr("VNQ", "SPY"),
                  "ctx_property_vs_defensives_mom20": mom.VNQ - mean(mom, ["XLP", "XLU"]),
                  "ctx_nominal_duration_mom20": duration}
    else:
        raise ValueError(f"Unsupported ETF {ticker}")
    expected = [s["feature_name"] for s in _definitions(ticker)]
    if list(values) != expected:
        raise AssertionError("Feature implementation and registry disagree")
    result = pd.DataFrame(index=own.index)
    result.index.name = "Date"
    for name, series in values.items():
        aligned = series.reindex(result.index).replace([np.inf, -np.inf], np.nan)
        result[name] = aligned.fillna(0.0)
        result[name + "_available"] = aligned.notna().astype(float)
    return result


def load_context(ticker: str, root: Path = ROOT) -> pd.DataFrame:
    path = Path(root) / "data/normalized/etf_context" / f"{ticker}.csv"
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    if len(frame) and frame.index.max() >= pd.Timestamp(RESERVED_START):
        raise ValueError("Saved context contains reserved dates")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Saved context failed date/finite checks")
    return frame


def _load_pre_reserve_prices(root: Path) -> dict[str, pd.DataFrame]:
    result = {}
    for symbol in SOURCES:
        path = root / "data/normalized/prices" / f"{symbol}.csv"
        if path.exists():
            # Reserve filtering precedes all mathematical transformations.
            raw = pd.read_csv(path, usecols=["Date", "adj_close"])
            result[symbol] = _price_frame(raw, RESERVED_START)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    prices = _load_pre_reserve_prices(root)
    provenance = json.loads((root / "data/data_manifest.json").read_text(encoding="utf-8"))
    provider_sources = {entry["identifier"]: entry for entry in provenance["sources"]}
    source_hashes = {symbol: hashlib.sha256(frame[["adj_close"]].to_csv(float_format="%.17g").encode()).hexdigest()
                     for symbol, frame in prices.items()}
    output = root / "data/normalized/etf_context"
    output.mkdir(parents=True, exist_ok=True)
    definitions, audits, checks = [], [], []
    source_code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for ticker in UNIVERSE:
        frame = build_context(prices, ticker)
        path = output / f"{ticker}.csv"
        frame.to_csv(path)
        if not np.isfinite(frame.to_numpy()).all() or frame.index.max() >= pd.Timestamp(RESERVED_START):
            raise AssertionError(f"Invalid context for {ticker}")
        for spec in _definitions(ticker):
            base = dict(spec, feature_version=FEATURE_VERSION, available_at_rule="Completed decision session close; next actual session open execution",
                        missingness_policy="No price filling. Undefined/insufficient-history factor -> value 0 plus corresponding _available=0. Flag distinguishes missing from measured zero.",
                        reserved_start_exclusive=RESERVED_START, source_code_sha256=source_code_sha,
                        preregistered_definition="Fixed before any ETF-context incremental fit; no target labels or learned parameter")
            definitions.append(base)
            definitions.append(dict(base, feature_name=spec["feature_name"] + "_available", kind="availability_flag",
                                    formula=f"1 iff {spec['feature_name']} is finite using sufficient available source history, else 0", unit="binary flag"))
        # Prefix equality is checked at four prior cutoffs. Complete future
        # source frames are withheld from every prefix build.
        dates = [frame.index[int((len(frame)-1)*fraction)] for fraction in [.2, .5, .8, .95]]
        prefix_checks = []
        for end in dates:
            prefix_prices = {s: p.loc[p.index <= end] for s, p in prices.items()}
            prefix = build_context(prefix_prices, ticker)
            expected = frame.loc[prefix.index]
            equal = bool(np.allclose(prefix.to_numpy(), expected.to_numpy(), rtol=1e-11, atol=1e-12, equal_nan=False))
            if not equal:
                raise AssertionError(f"Causal prefix mismatch: {ticker} {end}")
            prefix_checks.append({"cutoff": str(end.date()), "rows": len(prefix), "passed": equal})
        flags = [c for c in frame if c.endswith("_available")]
        availability = []
        for flag in flags:
            available = frame[flag].eq(1)
            if not frame[flag].isin([0., 1.]).all():
                raise AssertionError("Availability flag outside zero/one")
            if not frame.loc[~available, flag.removesuffix("_available")].eq(0).all():
                raise AssertionError("Unavailable factor did not retain explicit zero placeholder")
            availability.append({"feature": flag.removesuffix("_available"), "available_rows": int(available.sum()),
                                 "placeholder_rows": int((~available).sum()),
                                 "first_available_date": str(frame.index[available][0].date()) if available.any() else None})
        audits.append({"ticker": ticker, "rows": len(frame), "first_date": str(frame.index.min().date()),
                       "last_date": str(frame.index.max().date()), "value_features": len(flags), "availability_flags": len(flags),
                       "columns": list(frame), "path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "availability": availability, "prefix_causality_checks": prefix_checks})
        checks.extend(prefix_checks)
        print(f"{ticker}: {len(frame)} pre-reserve rows, {len(flags)} values + {len(flags)} flags; four prefix checks PASS", flush=True)
    registry = root / "data/etf_context_registry.csv"
    pd.DataFrame(definitions).to_csv(registry, index=False)
    audit = {"schema_version": 1, "feature_version": FEATURE_VERSION, "generated_utc": datetime.now(timezone.utc).isoformat(),
             "status": "PASS", "reserved_start_exclusive": RESERVED_START, "reserved_feature_rows": 0,
             "new_downloads": False, "source_code_sha256": source_code_sha, "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
             "source_inputs": [{"ticker": s, "pre_reserve_adjusted_close_sha256": source_hashes[s],
                                "provider_source_url": provider_sources.get(s, {}).get("source_url"),
                                "provider_raw_sha256": provider_sources.get(s, {}).get("sha256"),
                                "provider_raw_path": provider_sources.get(s, {}).get("raw_path"),
                                "provider_retrieved_utc": provider_sources.get(s, {}).get("retrieved_utc"),
                                "provider_normalized_file_sha256": provider_sources.get(s, {}).get("normalized_sha256"),
                                "first_date": str(p.index.min().date()),
                                "last_date": str(p.index.max().date()), "rows": len(p)} for s,p in prices.items()],
             "causal_prefix_checks_passed": len(checks), "instruments": audits,
             "limitations": ["Fixed current ETF cohort and static cyclical/defensive groups are declared hypotheses, not a point-in-time constituent universe.",
                             "All source adjusted prices retain the v2 vendor total-return/corporate-action limitations; no independent adjustment audit is implied.",
                             "UUP has real listed coverage only from 2007-03-01. Missing prelisting dollar context uses explicit flags and zero factor placeholders; prices are never filled.",
                             "TIP-minus-IEF return is not breakeven inflation; TLT-minus-IEF return is not a measured yield curve or pure duration factor.",
                             "DBC equity-sector proxies are not commodity futures term structure, roll yield or carry. Those inputs remain unavailable.",
                             "Correlation and beta are historical associations, not causal exposures. Global sector panels may include the subject sector.",
                             "No latest-vintage macro, sentiment, holdings or target labels are used. All factors are computed strictly before 2024 before rolling/return transforms."]}
    (root / "reports/context_data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"PASS: {len(UNIVERSE)} instruments, {len(checks)} causal prefix checks; no reserved context rows", flush=True)


if __name__ == "__main__":
    main()
