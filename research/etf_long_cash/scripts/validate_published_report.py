"""Validate retained report data; no downloads, fitting or reserved scoring.

This checks hashes, curves, cached-statistic consistency and notebook structure.
The full original cash/share and OOF audits require the complete local study.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import nbformat
import numpy as np
import pandas as pd

project = Path(__file__).resolve().parents[1]
root = project / "focused_payoff_context"
sys.path.insert(0, str(root / "src"))
from notebook_views import FocusedReport, LABELS, read_table

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--record", action="store_true", help="Update the lean validation record")
args = parser.parse_args()
checks = 0


def check(condition, message):
    global checks
    if not bool(condition):
        raise AssertionError(message)
    checks += 1


def near(actual, expected, message):
    check(np.allclose(actual, expected, rtol=1e-9, atol=1e-10, equal_nan=True), message)


manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for name, entry in manifest["inputs"].items():
    raw = (root / "data" / name).read_bytes()
    check(len(raw) == entry["bytes"], f"Input size: {name}")
    check(hashlib.sha256(raw).hexdigest() == entry["sha256"], f"Input hash: {name}")
report = FocusedReport(root)
check(len(manifest["tickers"]) == 15, "ETF universe")
check(set(report.ledgers.ticker) == set(manifest["tickers"]), "Curve universe")
check(set(report.ledgers.strategy) == set(LABELS), "Three displayed strategies")
check(report.ledgers.Date.max() < pd.Timestamp("2024-01-01"), "Curve cutoff")
check(report.forecasts.entry_date.max() < pd.Timestamp("2024-01-01"), "Forecast cutoff")
check(manifest["reserved_scored"] is False, "Reserved period unscored")
tables = {name: read_table(root / "data", name) for name in (
    "monthly_returns", "annual_performance", "risk_statistics", "drawdown_episodes",
    "weekly_policy_intervals", "forecast_outcomes", "paired_benchmark_uncertainty")}
check(len(tables["paired_benchmark_uncertainty"]) == 30, "Paired benchmark rows")
outcomes = tables["forecast_outcomes"]
check(pd.to_datetime(outcomes.label_end).dropna().max() < pd.Timestamp("2024-01-01"),
      "Outcome labels precede reserved period")

for ticker in manifest["tickers"]:
    forecasts = report.forecasts.loc[report.forecasts.ticker.eq(ticker)]
    check(forecasts.horizon.isin([5, 10, 20]).all(), f"{ticker}: horizons")
    check(forecasts.target_weight.between(0, 1).all(), f"{ticker}: long/cash weights")
    model = report.ledger(ticker)
    summary = report.summary(ticker)
    episodes = report.completed_episodes(ticker)
    check(summary["completed_episodes"] == len(episodes), f"{ticker}: episode count")
    check(sum(summary[k] for k in ("net_wins", "net_losses", "net_flat")) == len(episodes),
          f"{ticker}: outcome counts")
    near(episodes.net_pnl.sum(), model.nav.iloc[-1] - 1, f"{ticker}: cached episode PnL")
    weekly = tables["weekly_policy_intervals"].query("ticker == @ticker")
    near((1 + weekly.policy_net_return).prod(), model.nav.iloc[-1], f"{ticker}: weekly NAV")
    near(weekly.policy_net_return, weekly.end_nav / weekly.start_opening_nav - 1,
         f"{ticker}: weekly intervals")
    for year, forecasts_year in forecasts.groupby("outer_year"):
        usage = report.usage.loc[report.usage.ticker.eq(ticker) & report.usage.year.eq(year)]
        check(forecasts_year.feature_count.eq(len(usage)).all(), f"{ticker}/{year}: input count")
    for strategy in LABELS:
        series = report.ledger(ticker, strategy)
        check(series.index.equals(model.index), f"{ticker}/{strategy}: aligned dates")
        check(series.nav.gt(0).all(), f"{ticker}/{strategy}: positive NAV")
        check(series.held_exposure.between(-1e-10, 1 + 1e-10).all(), f"{ticker}/{strategy}: exposure")
        near(series.daily_return, series.nav / series.nav.shift(1).fillna(1) - 1,
             f"{ticker}/{strategy}: daily NAV returns")
        for name in ("monthly_returns", "annual_performance"):
            rows = tables[name].query("ticker == @ticker and strategy == @strategy")
            near((1 + rows.net_return).prod(), series.nav.iloc[-1], f"{ticker}/{strategy}: {name}")
        risk = tables["risk_statistics"].query("ticker == @ticker and strategy == @strategy")
        check(len(risk) == 1, f"{ticker}/{strategy}: risk row")
        near(risk.sharpe.iloc[0], np.sqrt(252) * series.daily_return.mean() / series.daily_return.std(ddof=1),
             f"{ticker}/{strategy}: Sharpe")
        max_dd = (series.nav / series.nav.cummax().clip(lower=1) - 1).min()
        near(risk.max_drawdown.iloc[0], max_dd, f"{ticker}/{strategy}: maximum drawdown")
        drawdowns = tables["drawdown_episodes"].query("ticker == @ticker and strategy == @strategy")
        near(drawdowns.depth.min(), max_dd, f"{ticker}/{strategy}: deepest drawdown episode")

nb = nbformat.read(root / "payoff_context_review.ipynb", as_version=4)
nbformat.validate(nb)
cells = [c for c in nb.cells if c.cell_type == "code"]
check(len(cells) == 139 and all(c.execution_count is not None for c in cells), "Executed cells")
outputs = [o for c in cells for o in c.outputs]
check(not any(o.output_type == "error" for o in outputs), "No notebook errors")
images = sum("image/png" in o.get("data", {}) for o in outputs)
check(images == 61, "Embedded figures")
check(nb.metadata.swing_tree.variant == "payoff_context", "Notebook variant")
check(nb.metadata.swing_tree.source_run == "v3.1", "Notebook source run")
check(nb.metadata.swing_tree.reserved_start == "2024-01-01", "Notebook reserved cutoff")
result = {"status": "PASS", "checks_passed": checks, "executed_cells": len(cells),
          "embedded_figures": images, "reserved_scored": False,
          "scope": "Retained-data hashes, curves, cached-statistic consistency and notebook structure"}
if args.record:
    path = root / "verification.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["lean_validation"] = result
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
