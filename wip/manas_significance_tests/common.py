"""Shared pieces for the three significance tests of the v3.1 ETF long/cash trees.

- Weekly evaluator: hold each weekly decision from its entry open to the next entry open,
  5 bps one-way cost per position change, equal-capital portfolio across ETFs.
- `replicate()`: a re-implementation of the core v2/v3.1 training loop from
  research/etf_long_cash/src/research.py. Per ETF and outer year 2008-2023 it runs the same
  inner selection (horizon 5/10/20 x classification/regression x technical/relationships
  x fixed/vol sizing, scored on the two prior years), then refits the chosen CART on all
  matured prior rows (expanding window) and predicts that year's weekly decisions.
  Differences from the published v3.1: no ETF-context features, no payoff calibrator, and
  inner scores use the weekly evaluator instead of the daily account simulator.

Everything uses only pre-2024 data; the downloaded prices end 2023-12-29.
Run with the Python 3.12 environment in research/etf_long_cash/.venv.
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ETF = REPO / "research" / "etf_long_cash"
sys.path.insert(0, str(ETF / "src"))
import research as base  # noqa: E402  (the original pipeline module)

CFG = base.CFG
UNIVERSE = CFG["universe"]
PEERS = ["SPY", "IEF", "GLD", "DBC"]
COST = 5 / 10000
FIRST_YEAR, LAST_YEAR = CFG["historical_outer_first_year"], CFG["historical_outer_last_year"]
RESERVED = pd.Timestamp(CFG["reserved_start"])
RESULTS = REPO / "wip" / "manas_significance_tests" / "results"


def load_prices() -> dict[str, pd.DataFrame]:
    prices = base.load_prices()
    if not prices:
        raise FileNotFoundError("No prices. Run research/etf_long_cash/src/data_ingest.py first.")
    prices = {k: v.loc[v.index < RESERVED] for k, v in prices.items()}
    return prices


# ---------------------------------------------------------------- weekly evaluation
def weekly_returns(price: pd.DataFrame, entry_dates: pd.Series) -> pd.Series:
    """Return earned by a position opened at entry_date's open and held to the next entry's open."""
    op = price["adj_open"]
    nxt = entry_dates.shift(-1)
    r = nxt.map(op) / entry_dates.map(op) - 1
    return r


def net_weekly(weights: pd.Series, rets: pd.Series) -> pd.Series:
    """Weights and returns indexed by decision_date (same index). Costs on |change in weight|."""
    w = weights.astype(float)
    turnover = w.diff().abs().fillna(w.abs())
    return w * rets - turnover * COST


def sharpe(x: pd.Series | np.ndarray) -> float:
    x = pd.Series(x).dropna()
    s = x.std()
    return float(x.mean() / s * np.sqrt(52)) if s > 0 else 0.0


def portfolio(net_by_ticker: dict[str, pd.Series]) -> pd.Series:
    """Equal-capital across the ETFs that have a decision in that week."""
    return pd.DataFrame(net_by_ticker).mean(axis=1, skipna=True)


def evaluate(signals: dict[str, pd.Series], rets: dict[str, pd.Series]) -> dict:
    """signals[t]: 0/1 (or weight) by decision_date. Returns per-ETF and portfolio Sharpe."""
    net = {}
    for t, s in signals.items():
        r = rets[t].reindex(s.index)
        keep = r.notna()
        net[t] = net_weekly(s[keep], r[keep])
    port = portfolio(net)
    return {"portfolio": sharpe(port), "per_etf": {t: sharpe(v) for t, v in net.items()},
            "net": net, "port_series": port}


# ---------------------------------------------------------------- replica of the training loop
def build_events(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in UNIVERSE:
        f = base.build_features(prices, t)
        e = base.weekly_events(prices[t], f)
        e["wk_ret"] = weekly_returns(prices[t], e["entry_date"])
        out[t] = e
    return out


def _fit_predict(train, test, cols, h, target):
    pred, _ = base.fit_predict(train, test, cols, h, target, "CART")
    return np.asarray(pred, float)


def _signal(pred, target):
    return (pred > (0.5 if target == "classification" else 0.0)).astype(float)


def _scale(frame, sizing):
    if sizing == "fixed":
        return np.ones(len(frame))
    return np.minimum(1, CFG["volatility_target"] / frame["tech_vol_63"].clip(lower=CFG["volatility_floor"]).to_numpy())


SPECS = list(itertools.product(CFG["horizons"], CFG["targets"], CFG["primary_eligible_recipes"]))


def replicate_ticker(events: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """Returns the fixed-size 0/1 signal by decision_date for 2008-2023, plus the selections."""
    signals, chosen_log = [], []
    cols_by_recipe = {r: base.feature_columns(events, r) for r in CFG["primary_eligible_recipes"]}
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        ystart = pd.Timestamp(year, 1, 1)
        best = None
        for h, target, recipe in SPECS:
            cols = cols_by_recipe[recipe]
            folds, ok = [], True
            for valyear in range(year - CFG["inner_years"], year):
                cut = pd.Timestamp(valyear, 1, 1)
                tr = base.mature_rows(events, cut, cols)
                te = events.loc[(events.index >= cut) & (events.index < pd.Timestamp(valyear + 1, 1, 1))
                                & (events.entry_date < ystart)].dropna(subset=cols + ["entry_date"])
                if len(tr) < CFG["minimum_matured_weekly_rows"] or len(te) < 40:
                    ok = False
                    break
                folds.append((te, _fit_predict(tr, te, cols, h, target)))
            if not ok:
                continue
            for sizing in ["fixed", "vol"]:
                parts = [net_weekly(pd.Series(_signal(p, target) * _scale(te, sizing), index=te.index),
                                    te["wk_ret"]) for te, p in folds]
                score = sharpe(pd.concat(parts))
                key = (-score, f"{h:02}_{target}_{recipe}", 0 if sizing == "fixed" else 1)
                if best is None or key < best[0]:
                    best = (key, h, target, recipe, sizing)
        if best is None:
            continue
        _, h, target, recipe, sizing = best
        cols = cols_by_recipe[recipe]
        train = base.mature_rows(events, ystart, cols)
        test = events.loc[(events.index >= ystart) & (events.index < pd.Timestamp(year + 1, 1, 1))].dropna(
            subset=cols + ["entry_date"])
        if len(test) < 40:
            continue
        pred = _fit_predict(train, test, cols, h, target)
        signals.append(pd.Series(_signal(pred, target), index=test.index))
        chosen_log.append(dict(year=year, horizon=h, target=target, recipe=recipe, sizing=sizing))
    return (pd.concat(signals) if signals else pd.Series(dtype=float)), chosen_log


def replicate(events: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.Series], dict]:
    sig, logs = {}, {}
    for t, e in events.items():
        sig[t], logs[t] = replicate_ticker(e)
    return sig, logs


def always_long(signals: dict[str, pd.Series]) -> dict[str, pd.Series]:
    return {t: pd.Series(1.0, index=s.index) for t, s in signals.items()}
