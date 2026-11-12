"""Short-side diagnosis for the v3.1 ETF trees (inherited by the v4.0 long/flat/short study).

Question: when the model leaves an ETF (signal = 0, "cash"), does that ETF go on to fall?
If not, turning cash weeks into shorts cannot make money.

Uses only saved outputs already in the repo:
    research/etf_long_cash/focused_payoff_context/data/forecast_outcomes.csv.gz

Run from the repo root:
    python scripts/short_side_diagnosis.py
"""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

DATA = Path("research/etf_long_cash/focused_payoff_context/data/forecast_outcomes.csv.gz")


def load() -> pd.DataFrame:
    d = pd.read_csv(DATA)
    d = d[d["label_available"]].copy()
    # Horizons differ (5/10/20 sessions); convert each gross forward return to a
    # per-5-session (weekly) equivalent so rows are comparable.
    d["fwd_wk"] = (1 + d["actual_gross_horizon_return"]) ** (5 / d["horizon"]) - 1
    d["cash"] = d["signal"] == 0
    d["z"] = d.groupby("ticker")["fwd_wk"].transform(lambda x: (x - x.mean()) / x.std())
    return d


def main() -> None:
    d = load()
    print(f"Decision weeks with matured outcomes: {len(d)}   cash share: {d['cash'].mean():.1%}\n")

    print("1) Forward return after LONG vs CASH decisions (pooled, per-week equivalent)")
    pooled = d.groupby("cash")["fwd_wk"].agg(
        n="size", mean_bp=lambda x: x.mean() * 1e4, frac_down=lambda x: (x < 0).mean()
    )
    print(pooled.round(3).to_string(), "\n")

    print("2) Per ETF: what happened after the model went to cash")
    rows = []
    for t, x in d.groupby("ticker"):
        c, l = x[x["cash"]]["fwd_wk"], x[~x["cash"]]["fwd_wk"]
        ic = spearmanr(x["expected_payoff"], x["fwd_wk"])[0]
        rows.append([t, len(c), c.mean() * 1e4, (c < 0).mean(), l.mean() * 1e4, ic])
    per = pd.DataFrame(rows, columns=["etf", "n_cash", "cash_fwd_bp_wk", "cash_frac_down",
                                      "long_fwd_bp_wk", "rank_ic_expected_payoff"])
    print(per.round(3).to_string(index=False), "\n")

    print("3) Dispersion: are cash weeks 'down' weeks or just 'volatile' weeks? (z-scored within ETF)")
    disp = d.groupby("cash")["z"].agg(
        std="std", mean="mean", p05=lambda x: x.quantile(0.05), p95=lambda x: x.quantile(0.95)
    )
    print(disp.round(3).to_string(), "\n")

    print("4) Forward return by expected-payoff quintile (pooled)")
    d["quintile"] = pd.qcut(d["expected_payoff"].rank(method="first"), 5,
                            labels=["Q1 bearish", "Q2", "Q3", "Q4", "Q5 bullish"])
    q = d.groupby("quintile", observed=True).agg(
        n=("fwd_wk", "size"), exp_payoff=("expected_payoff", "mean"),
        fwd_bp_wk=("fwd_wk", lambda x: x.mean() * 1e4), frac_down=("fwd_wk", lambda x: (x < 0).mean())
    )
    print(q.round(4).to_string())
    print("\nNote: weekly decisions with 10/20-session labels overlap, so row counts overstate "
          "independent observations. Descriptive only; no significance claimed.")


if __name__ == "__main__":
    main()
