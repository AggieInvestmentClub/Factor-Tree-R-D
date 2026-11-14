"""Compare new state-model account ledgers with the frozen v17 Ridge accounts.

This module analyzes completed accounts only. It does not fit models, change
positions, select dates, or read any 2024+ outcomes. The input arm family should
be declared in a protocol before candidate account results are inspected.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ("huber_ridge_10y", "ridge_5y")
PERIODS = (
    ("primary", "2012-01-01", "2023-12-31"),
    ("early", "2007-01-01", "2011-12-31"),
    ("full", "2007-01-01", "2023-12-31"),
)
SOURCE_KEYS = (
    "source_approval_sha256", "prices_sha256", "events_sha256",
    "claims_sha256", "noncash_census_sha256", "account_code_hashes",
)
REQUIRED_LEDGER = (
    "daily_return", "nav", "turnover", "costs", "long_net_pnl", "short_net_pnl",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use --arm name=ledger.csv.gz")
    name, raw_path = value.split("=", 1)
    if not name or name in REFERENCES or not raw_path:
        raise argparse.ArgumentTypeError("Candidate name must be nonempty and distinct from the references")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Ledger does not exist: {path}")
    return name, path


def account_metadata(name: str, path: Path, baseline: dict | None) -> dict:
    """Verify source lineage when a completion manifest accompanies the ledger."""
    candidate = path.with_name(f"{name}_complete.json")
    if not candidate.is_file():
        return {"metadata_present": False, "source_match_verified": False}
    state = json.loads(candidate.read_text(encoding="utf-8"))
    if state.get("ledger_sha256") != sha(path):
        raise ValueError(f"Ledger hash does not match completion manifest: {name}")
    if baseline is not None:
        for key in SOURCE_KEYS:
            if key not in state or state[key] != baseline.get(key):
                raise ValueError(f"Source/economics mismatch for {name}: {key}")
    return {
        "metadata_present": True,
        "source_match_verified": baseline is not None,
        "metadata_path": str(candidate),
        "metadata_sha256": sha(candidate),
        "account_state": state.get("state"),
    }


def load_ledger(path: Path, name: str) -> pd.DataFrame:
    ledger = pd.read_csv(path, index_col=0, parse_dates=True)
    missing = set(REQUIRED_LEDGER) - set(ledger)
    if missing:
        raise ValueError(f"Ledger {name} lacks columns {sorted(missing)}")
    if ledger.empty or ledger.index.hasnans or not ledger.index.is_unique or not ledger.index.is_monotonic_increasing:
        raise ValueError(f"Ledger {name} has an invalid daily clock")
    if ledger.index.min() < pd.Timestamp("2007-01-01") or ledger.index.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError(f"Ledger {name} includes out-of-scope dates")
    if ledger.index.tz is not None or not (ledger.index == ledger.index.normalize()).all():
        raise ValueError(f"Ledger {name} dates must be normalized and timezone naive")
    columns = list(REQUIRED_LEDGER) + [c for c in ("long_price_pnl", "short_price_pnl", "gross_exposure", "net_exposure") if c in ledger]
    if not np.isfinite(ledger[columns].to_numpy(float)).all():
        raise ValueError(f"Ledger {name} has nonfinite account values")
    returns = ledger.daily_return.to_numpy(float)
    if (returns <= -1).any() or (ledger.nav <= 0).any():
        raise ValueError(f"Ledger {name} has nonpositive NAV")
    if np.max(np.abs(np.cumprod(1 + returns) - ledger.nav.to_numpy(float))) > 1e-8:
        raise ValueError(f"Ledger {name} NAV does not reconcile to returns")
    prior = ledger.nav.shift(1, fill_value=1.)
    leg_return = (ledger.long_net_pnl + ledger.short_net_pnl) / prior
    if np.max(np.abs(leg_return.to_numpy(float) - returns)) > 1e-9:
        raise ValueError(f"Ledger {name} long/short net P&L does not reconcile")
    if (ledger.turnover < -1e-12).any() or (ledger.costs < -1e-12).any():
        raise ValueError(f"Ledger {name} has negative turnover or costs")
    return ledger


def stats(values: np.ndarray) -> dict:
    x = np.asarray(values, float)
    if len(x) < 2 or not np.isfinite(x).all() or (x <= -1).any():
        raise ValueError("A complete return clock with positive NAV is required")
    nav = np.cumprod(1 + x)
    peaks = np.maximum.accumulate(np.r_[1., nav])[1:]
    volatility = float(x.std(ddof=1) * np.sqrt(252))
    return {
        "sessions": len(x),
        "sharpe": float(x.mean() * 252 / volatility) if volatility > 1e-15 else np.nan,
        "annual_mean_return": float(x.mean() * 252),
        "cagr": float(nav[-1] ** (252 / len(x)) - 1),
        "net_return": float(nav[-1] - 1),
        "volatility": volatility,
        "max_drawdown": float(np.min(nav / peaks - 1)),
    }


def performance(name: str, ledger: pd.DataFrame, prior: pd.Series, period: str,
                benchmark: pd.Series | None) -> dict:
    row = {
        "arm": name,
        "period": period,
        "start": ledger.index.min(),
        "end": ledger.index.max(),
        **stats(ledger.daily_return.to_numpy(float)),
        "turnover_per_year": float(ledger.turnover.mean() * 252),
        "cost_per_year": float((ledger.costs / prior).mean() * 252),
        "annual_long_net_contribution": float((ledger.long_net_pnl / prior).mean() * 252),
        "annual_short_net_contribution": float((ledger.short_net_pnl / prior).mean() * 252),
    }
    if not np.isclose(row["annual_long_net_contribution"] + row["annual_short_net_contribution"],
                      row["annual_mean_return"], atol=1e-9):
        raise ValueError(f"Net leg contribution mismatch: {name}, {period}")
    for side in ("long", "short"):
        column = f"{side}_price_pnl"
        if column in ledger:
            row[f"annual_{side}_gross_contribution"] = float((ledger[column] / prior).mean() * 252)
    for column in ("gross_exposure", "net_exposure"):
        if column in ledger:
            row[f"average_{column}"] = float(ledger[column].mean())
    if benchmark is not None:
        spy = benchmark.reindex(ledger.index)
        if spy.isna().any():
            raise ValueError("SPY benchmark has a missing account date")
        fit = np.linalg.lstsq(np.column_stack([np.ones(len(spy)), spy.to_numpy(float)]),
                              ledger.daily_return.to_numpy(float), rcond=None)[0]
        row["spy_beta"] = float(fit[1])
        row["annual_spy_alpha_descriptive"] = float(fit[0] * 252)
    return row


def holm(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Finite probabilities required for the full family")
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty(len(p))
    result[order] = adjusted
    return result


def paired_family(frame: pd.DataFrame, arms: list[str], block: int,
                  replications: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Resample identical time blocks for all accounts and both references."""
    names = list(REFERENCES) + arms
    x = frame[names].to_numpy(float)
    n = len(x)
    if n <= block or not np.isfinite(x).all():
        raise ValueError("Complete, common-clock primary returns required")
    rng = np.random.default_rng(seed)
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    observed = np.divide(np.sqrt(252) * mu, sd,
                         out=np.full(len(names), np.nan), where=sd > 1e-15)
    draws = np.empty((replications, len(names)), float)
    for start in range(0, replications, 50):
        count = min(50, replications - start)
        starts = rng.integers(0, n, size=(count, int(np.ceil(n / block))))
        indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(count, -1)[:, :n]
        sampled = x[indices]
        sample_mu = sampled.mean(axis=1)
        sample_sd = sampled.std(axis=1, ddof=1)
        draws[start:start + count] = np.divide(np.sqrt(252) * sample_mu, sample_sd,
                                               out=np.full_like(sample_mu, np.nan), where=sample_sd > 1e-15)
    rows = []
    for reference in REFERENCES:
        k = names.index(reference)
        for arm in arms:
            j = names.index(arm)
            delta = observed[j] - observed[k]
            distribution = draws[:, j] - draws[:, k]
            valid = bool(np.isfinite(delta) and np.isfinite(distribution).all())
            rows.append({
                "arm": arm,
                "baseline": reference,
                "delta_sharpe": float(delta),
                "ci_low": float(np.quantile(distribution, .025)) if valid else np.nan,
                "ci_high": float(np.quantile(distribution, .975)) if valid else np.nan,
                "p_one_sided": float((1 + np.sum(distribution - delta >= delta)) / (replications + 1)) if valid else 1.,
                "block_length": block,
                "block_role": "primary" if block == 60 else "sensitivity",
                "replications": replications,
                "seed": seed,
                "family_size": len(arms) if reference == REFERENCES[0] else 0,
                "valid_sharpe_replicates": int(np.isfinite(distribution).sum()),
                "interval_scope": "unadjusted_95_percentile_per_contrast;not_simultaneous",
                "endpoint_role": "net_daily_Sharpe_primary_vs_Huber10;Ridge5_historical_control",
                "start": frame.index.min(),
                "end": frame.index.max(),
                "sessions": n,
            })
    result = pd.DataFrame(rows)
    primary = result.baseline.eq(REFERENCES[0])
    result["p_holm"] = np.nan
    result.loc[primary, "p_holm"] = holm(result.loc[primary, "p_one_sided"].to_numpy(float))
    return result


def analyze(arms: list[tuple[str, Path]], output: Path, *, null: tuple[str, Path] | None = None,
            protocol: Path | None = None,
            replications: int = 5000, seed: int = 42, force: bool = False) -> None:
    if not arms:
        raise ValueError("At least one candidate arm is required")
    names = [name for name, _ in arms]
    if len(set(names)) != len(names):
        raise ValueError("Candidate arm names must be unique")
    if null is not None and (null[0] in names or null[0] in REFERENCES):
        raise ValueError("Null arm name must be distinct from challengers and references")
    if protocol is not None:
        config = json.loads(protocol.read_text(encoding="utf-8"))
        declared = [a["name"] if isinstance(a, dict) else a for a in config["arms"]]
        if names != declared:
            raise ValueError(f"Candidate arms differ from the declared protocol: {names} != {declared}")
        declared_null = config.get("null_arm")
        if isinstance(declared_null, dict):
            declared_null = declared_null.get("name")
        if declared_null is not None and (null is None or null[0] != declared_null):
            raise ValueError("Null arm differs from the declared protocol")
        if config.get("reserve_exclusive") != "2024-01-01":
            raise ValueError("Protocol must explicitly reserve 2024 onward")
    output = output.resolve()
    if (output / "analysis_status.json").exists() and not force:
        raise ValueError("Existing completed analysis; use a new output directory or explicit --force")

    paths = {ref: ROOT / "accounts" / f"{ref}_ledger.csv.gz" for ref in REFERENCES}
    paths.update(arms)
    if null is not None:
        paths[null[0]] = null[1]
    reference_meta_path = ROOT / "accounts" / "huber_ridge_10y_complete.json"
    reference_meta = json.loads(reference_meta_path.read_text(encoding="utf-8"))
    ledgers = {name: load_ledger(path, name) for name, path in paths.items()}
    clock = ledgers[REFERENCES[0]].index
    for name, ledger in ledgers.items():
        if not ledger.index.equals(clock):
            raise ValueError(f"Unequal account clock for {name}; date intersection is forbidden")
    for period, start, end in PERIODS:
        if ledgers[REFERENCES[0]].loc[start:end].empty:
            raise ValueError(f"Missing required period {period}")
    lineage = {name: account_metadata(name, path, reference_meta)
               for name, path in paths.items()}

    benchmark_path = ROOT / "reports/economic_v18/daily_returns.csv.gz"
    if not benchmark_path.is_file():
        benchmark_path = ROOT / "reports/long_history/daily_returns.csv.gz"
    benchmark = None
    if benchmark_path.is_file():
        benchmark_frame = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
        if "SPY" in benchmark_frame:
            benchmark = benchmark_frame.SPY
            if not benchmark.index.equals(clock) or not np.isfinite(benchmark.to_numpy(float)).all():
                raise ValueError("SPY benchmark clock differs from account clock")

    performance_rows = []
    annual_rows = []
    for name, ledger in ledgers.items():
        full_prior = ledger.nav.shift(1, fill_value=1.)
        for period, start, end in PERIODS:
            piece = ledger.loc[start:end]
            performance_rows.append(performance(name, piece, full_prior.reindex(piece.index), period, benchmark))
        for year, piece in ledger.groupby(ledger.index.year):
            annual_rows.append({**performance(name, piece, full_prior.reindex(piece.index), "calendar_year", benchmark), "year": year})
    returns = pd.DataFrame({name: ledger.daily_return for name, ledger in ledgers.items()})
    if benchmark is not None:
        returns["SPY"] = benchmark
    paired = pd.concat([paired_family(returns.loc["2012":"2023"], names, block,
                                     replications=replications, seed=seed)
                        for block in (60, 20)], ignore_index=True)
    null_diagnostic = None
    if null is not None:
        primary = pd.DataFrame(performance_rows).set_index(["arm", "period"])
        null_diagnostic = {
            "arm": null[0], "period": "primary",
            "delta_sharpe_vs_huber10": float(primary.loc[(null[0], "primary"), "sharpe"]
                                             - primary.loc[(REFERENCES[0], "primary"), "sharpe"]),
            "delta_annual_mean_vs_huber10": float(primary.loc[(null[0], "primary"), "annual_mean_return"]
                                                  - primary.loc[(REFERENCES[0], "primary"), "annual_mean_return"]),
            "role": "shuffled_state_negative_control;descriptive_not_in_Holm_family",
        }

    output.mkdir(parents=True, exist_ok=True)
    reports = {
        "performance.csv": pd.DataFrame(performance_rows),
        "annual.csv": pd.DataFrame(annual_rows),
        "paired_comparisons.csv": paired,
    }
    for filename, frame in reports.items():
        frame.to_csv(output / filename, index=False)
    returns.to_csv(output / "daily_returns.csv.gz")
    if null_diagnostic is not None:
        pd.DataFrame([null_diagnostic]).to_csv(output / "negative_control.csv", index=False)
    input_paths = [Path(__file__), reference_meta_path, *paths.values()]
    if protocol is not None:
        input_paths.append(protocol)
    if benchmark is not None:
        input_paths.append(benchmark_path)
    for state in lineage.values():
        if state.get("metadata_present"):
            input_paths.append(Path(state["metadata_path"]))
    status = {
        "completed": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": "conditional_pre2024_reused_development;no_independent_confirmation",
        "references": list(REFERENCES),
        "candidate_arms": names,
        "null_arm": null[0] if null is not None else None,
        "negative_control_primary": null_diagnostic,
        "periods": {name: [start, end] for name, start, end in PERIODS},
        "reserve_exclusive": "2024-01-01",
        "account_clock_sessions": len(clock),
        "same_complete_account_clock": True,
        "primary_endpoint": "net_daily_Sharpe_difference_vs_huber_ridge_10y",
        "bootstrap": {"kind": "paired_circular_time_blocks", "replications": replications,
                      "seed": seed, "primary_block_sessions": 60, "sensitivity_block_sessions": 20,
                      "holm_family": f"{len(names)}_non_null_arms_vs_huber_ridge_10y",
                      "ridge5_role": "descriptive_historical_control;not_in_Holm_family"},
        "lineage": lineage,
        "input_hashes": {str(path.resolve()): sha(path) for path in input_paths},
        "output_hashes": {filename: sha(output / filename) for filename in
                          [*reports, "daily_returns.csv.gz"] + (["negative_control.csv"] if null is not None else [])},
        "limitations": [
            "All pre-2024 data have been repeatedly used for development; within-family Holm p-values do not correct the project-wide search.",
            "Only non-null contrasts versus Huber10 enter the within-study Holm family; Ridge5 contrasts are descriptive.",
            "Percentile confidence intervals are per contrast, not simultaneous family-wise intervals.",
            "Long/short figures are contributions to total account NAV, not standalone leg returns; hedge and risk sizing couple the legs.",
            "Historical valuation/source caveats of the reference study carry through to any matched accounts.",
            "A candidate without completion metadata has arithmetic verification only; fixed economics are not independently verified.",
        ],
    }
    (output / "analysis_status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(performance_rows).loc[lambda x: x.period.eq("primary"),
                                         ["arm", "sharpe", "cagr", "volatility", "turnover_per_year"]].to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=parse_arm, action="append", required=True,
                        help="Declared candidate name and completed ledger path, name=path; repeat for all arms")
    parser.add_argument("--null", type=parse_arm,
                        help="Optional shuffled-state negative-control ledger, name=path; excluded from the Holm family")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/state_v20")
    parser.add_argument("--protocol", type=Path, help="Optional frozen protocol with ordered arms and reserve_exclusive")
    parser.add_argument("--replications", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Explicitly replace a completed report in --out")
    args = parser.parse_args()
    if args.replications < 100:
        raise ValueError("At least 100 bootstrap replications are required")
    analyze(args.arm, args.out, null=args.null, protocol=args.protocol, replications=args.replications,
            seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
