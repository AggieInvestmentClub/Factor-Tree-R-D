"""Generate documentation from source and pre-reserve feature columns; no fit."""
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import research

CLOSE_RULE = "After instrument and required peer completed close on date t; used only after the last completed W-FRI session close, execution next actual session open. Daily files do not prove original vendor receipt timestamps."
MISSING = "No price forward/backfill. Rolling windows require pandas default full-window nonmissing inputs. NaN and nonfinite features remain missing; each fitted recipe drops rows missing any required feature."
PEER_MISSING = "Peer prices reindexed to instrument sessions without price fill. Uncomputable n-session peer momentum (prelisting, initial warmup or missing source) becomes explicit zero with paired rel_PEER_available_n=0. Zero is a placeholder, not a price/return observation."
MACRO_RULE = "Source observation date + pandas BDay(2), weekdays only (not an exchange/US holiday calendar); backward merge_asof onto SPY sessions. This assumed date is not verified original publication/receipt or vintage availability; quarantined."
MACRO_MISSING = "Drop missing raw macro observations; backward asof carry last assumed-available observation with no staleness cap. Preserve missing before first available observation; change20 also missing for first 20 aligned sessions. No prices are filled."
PRICE_CA = "Yahoo adjusted OHLC uses provider OHLC*(Adj Close/Close), a retrospective total-return approximation; vendor corrections/current adjustment factors may revise history. Do not double-credit cash dividends."
rows = {}


def add(name, family, formula, inputs, unit, lookback, minimum, missing=MISSING, caveat=PRICE_CA):
    macro = family == "macro"
    rows[name] = dict(feature_name=name, family=family, formula=formula, inputs=inputs, unit=unit,
                      lookback_sessions=lookback, minimum_history=minimum,
                      available_at_rule=MACRO_RULE if macro else CLOSE_RULE, missingness_policy=missing,
                      feature_version="v1", eligibility="quarantined_not_used_in_primary" if macro else "primary_eligible_v1_historical_research",
                      primary_recipes="none" if macro else ("technical|relationships" if family == "technical" else "relationships"),
                      source_file="src/data_ingest.py + src/research.py" if macro else "src/research.py",
                      source_function="macro_features + build_features" if macro else "build_features", corporate_action_caveat=caveat)


for n in [5, 10, 20, 63, 126, 252]:
    add(f"tech_mom_{n}", "technical", f"C_t / C_(t-{n}) - 1", "adj_close", "decimal return (0.05 = 5%)", n, f"{n+1} session rows; current and lagged adjusted-close endpoints valid")
for n in [20, 63]:
    add(f"tech_vol_{n}", "technical", f"sample_std(r, {n}, ddof=1) * sqrt(252)", "r_t=adj_close_t/adj_close_(t-1)-1", "annualized decimal volatility", n, f"{n} valid daily returns / {n+1} adjusted closes")
add("tech_downside_20", "technical", "sqrt(mean(min(r,0)^2,20))*sqrt(252); positive days count as zero in denominator", "adj_close daily returns", "annualized decimal downside deviation", 20, "20 valid daily returns / 21 closes")
for n in [50, 200]:
    add(f"tech_trend_{n}", "technical", f"C_t / rolling_mean(C,{n}) - 1", "adj_close", "dimensionless decimal ratio minus one", n, f"{n} closes")
add("tech_drawdown_252", "technical", "C_t / rolling_max(C,252) - 1", "adj_close", "decimal drawdown, ordinarily <= 0", 252, "252 closes")
add("tech_range_20", "technical", "rolling_mean((H-L)/C,20)", "adj_high|adj_low|adj_close", "dimensionless decimal intraday range", 20, "20 complete adjusted HLC rows")
add("tech_gap_5", "technical", "rolling_mean(O_t/C_(t-1)-1,5)", "adj_open|lagged adj_close", "decimal overnight gap", 5, "5 valid gaps / 6 consecutive OHLC sessions")
add("tech_reversal_5", "technical", "-rolling_sum(r,5); simple-return sum, not compounded", "adj_close daily returns", "negative sum of decimal returns", 5, "5 daily returns / 6 closes")
add("tech_volume_ratio", "technical", "rolling_mean(Volume,5)/rolling_mean(Volume,63)-1", "provider Volume retained unchanged", "dimensionless relative-volume ratio minus one", 63, "63 volume observations", caveat="Volume is retained as returned by the provider; split or other corporate-action changes can distort relative volume. No additional share-volume split adjustment is applied or verified; prices separately use adjusted OHLC.")
add("tech_shock", "technical", "r_t / sample_std(r,63,ddof=1)_(t-1)", "adj_close daily returns; lagged 63-return standard deviation", "dimensionless lagged daily-volatility units", "63 returns ending t-1 plus current return", "64 returns / 65 closes; denominator strictly lagged")
add("tech_vol_jump", "technical", "sample_std(r,10,ddof=1)/sample_std(r,63,ddof=1)-1", "adj_close daily returns", "dimensionless volatility ratio minus one", 63, "63 returns / 64 closes")
add("tech_acceleration", "technical", "tech_mom_20_t - tech_mom_20_(t-20)", "adj_close; current and 20-session-lagged 20-session momentum", "difference in decimal returns", 40, "41 closes")
add("tech_fast_slow", "technical", "EMA(C,span=10,adjust=False)/EMA(C,span=63,adjust=False)-1; recursive EMA seeded at first available close", "adj_close", "dimensionless EMA ratio minus one", "unbounded decaying history; spans 10 and 63", "1 close; no enforced EMA warmup (first output is zero)")
for peer in ["SPY", "IEF", "GLD", "DBC"]:
    for n in [20, 63]:
        policy = PEER_MISSING.replace("PEER", peer)
        add(f"rel_{peer}_available_{n}", "relationships", f"1 if pct_change(peer adjusted close,{n},fill_method=None) is nonmissing else 0", f"{peer} adj_close reindexed to instrument sessions", "binary availability flag (0 or 1)", n, "Flag emitted from first instrument session; 1 requires both valid n-session endpoints", missing=policy)
        add(f"rel_{peer}_mom_{n}", "relationships", f"peer C_t/C_(t-{n})-1; fill missing momentum with zero only alongside paired availability flag", f"{peer} adj_close reindexed to instrument sessions", "decimal return when paired availability=1; zero placeholder otherwise", n, f"Observed momentum needs {n+1} aligned peer closes/endpoints; placeholder emitted earlier", missing=policy)
    if peer == "SPY":
        add("rel_market_beta_63", "relationships", "rolling_cov(r,SPY_r,63)/rolling_var(SPY_r,63); pandas sample statistics ddof=1", "instrument and SPY adjusted-close daily returns", "dimensionless market beta", 63, "63 paired valid returns / 64 aligned sessions")
        add("rel_market_correlation_63", "relationships", "rolling_corr(r,SPY_r,63)", "instrument and SPY adjusted-close daily returns", "correlation coefficient [-1,1]", 63, "63 paired valid returns / 64 aligned sessions")
        add("rel_market_residual_20", "relationships", "rolling_sum(r_t - beta63_(t-1)*SPY_r_t,20)", "instrument and SPY daily returns; strictly lagged rel_market_beta_63", "sum of decimal beta-adjusted daily returns", "lagged 63-return beta plus 20 residual returns", "84 aligned price rows when uninterrupted (first beta at row64, lag at row65, 20 residuals)")
        add("rel_strength_63", "relationships", "tech_mom_63 - rel_SPY_mom_63", "instrument adjusted close; SPY 63-session momentum plus availability flag", "difference in decimal returns", 63, "64 instrument closes; SPY unavailability represented by paired zero/flag policy", missing=MISSING + " SPY momentum uses zero/availability convention.")
macro_units = {
    "DGS10": "percent per annum, investment-basis Treasury yield (5 = 5%, not 0.05)",
    "DGS2": "percent per annum, investment-basis Treasury yield (5 = 5%, not 0.05)",
    "DTB3": "percent per annum, Treasury bill discount basis (5 = 5%, not 0.05)",
    "T10YIE": "percent annualized breakeven inflation rate (2 = 2%, not 0.02)",
    "VIXCLS": "VIX index points (20 corresponds to 20% annualized implied volatility; stored 20, not 0.20)",
    "BAMLH0A0HYM2": "percent option-adjusted spread (3 = 3 percentage points = 300 basis points)",
}
for series, unit in macro_units.items():
    name = "macro_" + series.lower()
    caveat = "Latest-vintage daily FRED observations, not verified original vintages; no price corporate action involved."
    if series == "BAMLH0A0HYM2":
        caveat += " Downloaded archive begins October 2023 because source history is capped; preserve structural precoverage missingness."
    add(name, "macro", "Latest nonmissing FRED level with assumed available_date <= SPY session date", series, unit, "asof level; no finite maximum age", "1 eligible source observation at assumed date", missing=MACRO_MISSING, caveat=caveat)
    change_unit = "VIX index-point difference" if series == "VIXCLS" else "percentage-point difference (0.50 = 50 basis points), not a decimal return or relative percent change"
    add(name + "_change20", "macro", f"{name}_t - {name}_(t-20 aligned SPY trading sessions)", series + " asof-aligned level", change_unit, 20, "21 aligned SPY sessions with nonmissing endpoint levels; may reuse stale observations", missing=MACRO_MISSING, caveat=caveat)
add("macro_term_spread", "macro", "macro_dgs10 - macro_dgs2", "DGS10|DGS2 lagged asof levels", "percentage points (1 = 100 basis points)", "asof levels", "Both assumed-available source levels nonmissing", missing=MACRO_MISSING, caveat="Latest-vintage Treasury yields; release timestamps/vintages unverified.")
add("macro_vix_log", "macro", "natural_log(max(macro_vixcls,0.001))", "VIXCLS lagged asof level", "natural log of numeric VIX index points; log(20), not log(0.20)", "asof level", "1 nonmissing assumed-available VIX level", missing=MACRO_MISSING, caveat="Latest-vintage VIX; floor 0.001 before log; release timestamps/vintages unverified.")

# Enumerate only development-period feature outputs. No fitting, selection,
# forecasts, backtests, or reserved-period feature computation is performed.
cutoff = pd.Timestamp(research.CFG["reserved_start"])
prices = {name: frame.loc[frame.index < cutoff] for name, frame in research.load_prices().items()}
macro = pd.read_csv(ROOT / "data/normalized/macro_features.csv", parse_dates=["Date"]).set_index("Date")
macro = macro.loc[macro.index < cutoff]
features = {ticker: research.build_features(prices, ticker, macro) for ticker in research.CFG["universe"]}
columns = list(features["XLK"].columns)
assert set(rows) == set(columns), {"undocumented": sorted(set(columns)-set(rows)), "nonexistent": sorted(set(rows)-set(columns))}
assert len(columns) == 54
assert all(list(frame.columns) == columns for frame in features.values())
assert all(len(frame.select_dtypes(include="number").columns) == 54 for frame in features.values())
registry_path = ROOT / "data/feature_registry.csv"
with registry_path.open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=list(next(iter(rows.values()))))
    writer.writeheader()
    writer.writerows(rows[col] for col in columns)
source_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ["src/research.py", "src/data_ingest.py", "config/run_config.json"]}
representative = features["XLK"]
contract = {
    "schema_version": 1, "feature_version": "v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "derivation": "Read actual build_features/macro_features source and enumerate actual numerical build_features output for all 15 instruments after truncating every price series and macro frame strictly before 2024-01-01. No trained models were rerun.",
    "source_sha256": source_hashes, "registry_path": "data/feature_registry.csv", "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    "counts": {"all_numeric_features": len(columns), "primary_eligible": 40, "technical": 20, "relationships": 20, "quarantined_macro": 14},
    "actual_column_order": columns,
    "formula_notation": {"C/O/H/L": "Instrument adj_close/adj_open/adj_high/adj_low", "r_t": "C_t/C_(t-1)-1 using pct_change(fill_method=None)", "session": "One row of that instrument's observed trading-session index; macro change20 uses aligned SPY session rows", "std/cov/var": "pandas rolling sample statistics, default ddof=1", "sqrt(252)": "Fixed 252-session annualization"},
    "unit_rules": ["All price returns and volatility features use decimal units: 0.05 means 5%.", "FRED interest/breakeven/spread levels remain percent units: 5 means 5%, not 0.05. Their differences are percentage points, not relative changes.", "VIX remains index points; log-VIX logs that numeric level. No FRED feature is divided by 100 in the current source."],
    "timing": {"decision": research.CFG["weekly_clock"], "execution": "Next actual instrument session open; no same-close fill", "technical_relationship_availability": CLOSE_RULE, "macro_assumption": MACRO_RULE, "reserved_start": str(cutoff.date()), "reserved_features_computed_for_this_enumeration": False},
    "missingness": {"price_fill": "none", "rolling": MISSING, "peer_momentum": PEER_MISSING, "macro": MACRO_MISSING, "nonfinite": "build_features replaces positive/negative infinity by NaN", "model_rows": "Recipe-specific complete-case drop; no learned imputer or scaler fitted"},
    "eligibility": {"primary": "20 technical features for technical recipe; all 40 technical+relationship features for relationships recipe", "macro": "14 archived diagnostic candidates quarantined; macro/all recipes ineligible under current macro PIT gate", "monthly": "CPIAUCSL, UNRATE, INDPRO archived only; no numerical feature columns emitted", "news": "FinBERT outputs are separate archived demonstrations; no FinBERT/news features enter these 54 model input columns", "official_validation": "Primary eligibility is for retrospective v1 research, not certification of original-vintage market data. Corporate actions and independent price verification remain necessary."},
    "implementation_notes": ["tech_shock divides current return by strictly lagged 63-return standard deviation.", "rel_market_residual_20 uses strictly lagged 63-return market beta.", "GLD/DBC beforelisting peer momenta use zero placeholders paired with availability flags. The same actual implementation covers initial warmup and any other uncomputable peer-momentum endpoint; underlying prices are never filled.", "tech_fast_slow emits recursive EMA ratio from the first close; there is no separate 63-session EMA warmup gate.", "Macro BDay(2) offset is weekday-based and has no intraday publication clock or maximum staleness limit.", "Volume is retained as the provider returns it, with no additional verified split adjustment; splits and other actions can distort relative-volume ratios.", "UUP price history was archived as a candidate proxy but build_features currently uses only SPY, IEF, GLD and DBC peers."],
    "representative_development_enumeration": {"ticker": "XLK", "first_input_date": str(representative.index.min().date()), "last_input_date": str(representative.index.max().date()), "rows": len(representative), "features": {name: {"nonmissing": int(representative[name].notna().sum()), "missing": int(representative[name].isna().sum()), "first_nonmissing": str(representative[name].first_valid_index().date()) if representative[name].notna().any() else None} for name in columns}},
    "instrument_enumeration": {ticker: {"rows": len(frame), "numeric_features": len(frame.columns), "last_input_date": str(frame.index.max().date()), "primary_complete_rows": int(frame[[name for name in columns if not name.startswith("macro_")]].notna().all(axis=1).sum())} for ticker, frame in features.items()},
    "macro_unit_sources": {series: f"https://fred.stlouisfed.org/series/{series}" for series in macro_units},
    "notebook_usage": {"registry": "Load data/feature_registry.csv for formulas, units, minimum history, availability rules and eligibility filters", "contract": "Use counts, actual_column_order and representative_development_enumeration for input library documentation; macro/news exclusion remains visible", "existing_outputs": "No models, forecasts, ledgers, model math or source code were modified by this documentation task"},
}
(ROOT / "reports/feature_contract.json").write_text(json.dumps(contract, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"registry": str(registry_path), "rows": len(columns), "families": contract["counts"], "contract": str(ROOT / "reports/feature_contract.json"), "source_sha256": source_hashes}, indent=2))
