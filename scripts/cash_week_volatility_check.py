"""Volatility check for the v3.1 long/cash trees.

Question: is going to cash worth it? Compare what happened after LONG decisions with what
happened after CASH decisions, in return AND risk. If cash weeks have clearly worse return per
unit of risk, the model is skipping bad weeks. If they look the same, the lower drawdown comes
only from being out of the market part of the time, not from picking which weeks to skip.

Uses the same saved file as scripts/short_side_diagnosis.py (pre-2024 only; 2024+ untouched):
    research/etf_long_cash/focused_payoff_context/data/forecast_outcomes.csv.gz

Run from the repo root:
    python scripts/cash_week_volatility_check.py
"""
from math import comb
from pathlib import Path

import pandas as pd

DATA = Path("research/etf_long_cash/focused_payoff_context/data/forecast_outcomes.csv.gz")


def sign_test_p(k: int, n: int) -> float:
    """Two-sided p-value for k of n under a fair coin (no scipy needed)."""
    probs = [comb(n, i) / 2**n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] + 1e-12))


def load() -> pd.DataFrame:
    d = pd.read_csv(DATA)
    d = d[d["label_available"]].copy()
    # Same conversion as the diagnosis script: per-5-session (weekly) equivalent return.
    d["fwd_wk"] = (1 + d["actual_gross_horizon_return"]) ** (5 / d["horizon"]) - 1
    d["cash"] = d["signal"] == 0
    return d


def stats(x: pd.Series) -> pd.Series:
    worst = x[x <= x.quantile(0.05)]
    return pd.Series({
        "n": len(x),
        "mean_bp": x.mean() * 1e4,
        "std_bp": x.std() * 1e4,
        "ret_per_risk": x.mean() / x.std(),
        "p05_bp": x.quantile(0.05) * 1e4,
        "worst5pct_avg_bp": worst.mean() * 1e4,
    })


def main() -> None:
    d = load()
    print(f"Decisions: {len(d)}  ({d['decision_date'].min()} to {d['decision_date'].max()})\n")

    print("1) Pooled: after LONG vs after CASH decisions (per-week equivalent)")
    pooled = d.groupby(d["cash"].map({False: "long", True: "cash"}))["fwd_wk"].apply(stats).unstack()
    print(pooled.round(3).to_string(), "\n")

    print("2) Per ETF (ratio = mean / std of forward weekly return)")
    rows = []
    for t, x in d.groupby("ticker"):
        c, l, a = x[x["cash"]]["fwd_wk"], x[~x["cash"]]["fwd_wk"], x["fwd_wk"]
        model = x["fwd_wk"] * (~x["cash"])  # earn the return only when long, 0 in cash
        rows.append({
            "etf": t, "n_cash": len(c),
            "std_cash/std_long": c.std() / l.std(),
            "ratio_long": l.mean() / l.std(),
            "ratio_cash": c.mean() / c.std(),
            "ratio_always_long": a.mean() / a.std(),
            "ratio_model": model.mean() / model.std(),
        })
    per = pd.DataFrame(rows).set_index("etf")
    print(per.round(3).to_string(), "\n")

    n = len(per)
    more_vol = int((per["std_cash/std_long"] > 1).sum())
    worse_cash = int((per["ratio_cash"] < per["ratio_long"]).sum())
    model_beats = int((per["ratio_model"] > per["ratio_always_long"]).sum())
    print("3) Scorecard across ETFs")
    print(f"   Cash weeks more volatile than long weeks:        {more_vol}/{n}"
          f"  (sign-test p={sign_test_p(more_vol, n):.3f})")
    print(f"   Cash weeks had worse return per risk:            {worse_cash}/{n}"
          f"  (sign-test p={sign_test_p(worse_cash, n):.3f})")
    print(f"   Model's ratio beat always-long's ratio:          {model_beats}/{n}"
          f"  (sign-test p={sign_test_p(model_beats, n):.3f})")
    print(f"   Median ratio: model {per['ratio_model'].median():.3f} vs "
          f"always-long {per['ratio_always_long'].median():.3f}\n")

    print("Notes: ratios are per-decision, not annualized Sharpe. Labels of 10/20 sessions overlap,")
    print("so rows are not independent; sign tests across ETFs treat ETFs as independent, which")
    print("they are not fully (shared market moves). Gross returns, no costs. Descriptive.")


if __name__ == "__main__":
    main()
