"""Test #1: is v3.1's timing better than random timing with the same exposure?

Takes the published v3.1 weekly long/cash decisions (forecast_outcomes.csv.gz). For each ETF it
splits the decision sequence into runs (consecutive long weeks, consecutive cash weeks), then
shuffles the order of the long runs and of the cash runs. Each shuffled model has exactly the
same number of cash weeks, the same run lengths and the same number of trades as the real one;
only WHICH weeks are cash changes. Repeat 10,000 times and see where the real model lands.

Run from the repo root:
    research/etf_long_cash/.venv/Scripts/python wip/manas_significance_tests/random_timing_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

N_SIM = 10_000
SEED = 20261117
FORECASTS = c.ETF / "focused_payoff_context" / "data" / "forecast_outcomes.csv.gz"


def runs(x: np.ndarray) -> tuple[list[int], list[int], float]:
    edges = np.flatnonzero(np.diff(x)) + 1
    starts = np.r_[0, edges]
    lengths = np.diff(np.r_[starts, len(x)])
    values = x[starts]
    return list(lengths[values == 1]), list(lengths[values == 0]), float(values[0])


def shuffled(longs, cashes, first, rng) -> np.ndarray:
    a, b = rng.permutation(longs), rng.permutation(cashes)
    first_list, second_list = (a, b) if first == 1 else (b, a)
    out, fv = [], first
    for i in range(max(len(first_list), len(second_list))):
        if i < len(first_list):
            out.append(np.full(first_list[i], fv))
        if i < len(second_list):
            out.append(np.full(second_list[i], 1 - fv))
    return np.concatenate(out)


def main() -> None:
    prices = c.load_prices()
    fo = pd.read_csv(FORECASTS, parse_dates=["decision_date", "entry_date"])
    signals, rets = {}, {}
    for t, g in fo.groupby("ticker"):
        g = g.sort_values("decision_date").set_index("decision_date")
        r = c.weekly_returns(prices[t], g["entry_date"])
        keep = r.notna()
        signals[t] = g.loc[keep, "signal"].astype(float)
        rets[t] = r[keep]

    real = c.evaluate(signals, rets)
    allin = c.evaluate(c.always_long(signals), rets)
    expo = {t: s.mean() for t, s in signals.items()}
    matched = c.evaluate({t: pd.Series(expo[t], index=s.index) for t, s in signals.items()}, rets)

    rng = np.random.default_rng(SEED)
    structure = {t: runs(s.to_numpy().astype(int)) for t, s in signals.items()}
    R = pd.DataFrame(rets)  # weeks x ETFs
    port_null = np.empty(N_SIM)
    etf_null = {t: np.empty(N_SIM) for t in signals}
    for k in range(N_SIM):
        net = {}
        for t, s in signals.items():
            fake = pd.Series(shuffled(*structure[t], rng), index=s.index, dtype=float)
            net[t] = c.net_weekly(fake, rets[t])
            etf_null[t][k] = c.sharpe(net[t])
        port_null[k] = c.sharpe(c.portfolio(net))

    def pval(real_value, null):
        return float((1 + np.sum(null >= real_value)) / (len(null) + 1))

    rows = []
    for t in sorted(signals):
        rows.append(dict(etf=t, cash_share=round(1 - expo[t], 3),
                         sharpe_model=real["per_etf"][t], sharpe_always_long=allin["per_etf"][t],
                         sharpe_matched_exposure=matched["per_etf"][t],
                         null_median=float(np.median(etf_null[t])),
                         null_95th=float(np.quantile(etf_null[t], 0.95)),
                         percentile=float((etf_null[t] < real["per_etf"][t]).mean()),
                         p_value=pval(real["per_etf"][t], etf_null[t])))
    per = pd.DataFrame(rows)
    summary = dict(
        n_simulations=N_SIM, seed=SEED, weeks=int(sum(len(s) for s in signals.values())),
        portfolio_sharpe_model=real["portfolio"], portfolio_sharpe_always_long=allin["portfolio"],
        portfolio_sharpe_matched_exposure=matched["portfolio"],
        null_median=float(np.median(port_null)), null_5th=float(np.quantile(port_null, 0.05)),
        null_95th=float(np.quantile(port_null, 0.95)),
        model_percentile=float((port_null < real["portfolio"]).mean()),
        p_value_one_sided=pval(real["portfolio"], port_null),
        etfs_beating_95pct_of_random=int((per.percentile >= 0.95).sum()),
        etfs_below_null_median=int((per.sharpe_model < per.null_median).sum()),
    )
    c.RESULTS.mkdir(parents=True, exist_ok=True)
    per.to_csv(c.RESULTS / "random_timing_per_etf.csv", index=False)
    np.save(c.RESULTS / "random_timing_null_portfolio.npy", port_null)
    (c.RESULTS / "random_timing_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 160)
    print("Test #1: random timing with identical exposure, run lengths and trade count")
    print(f"{N_SIM:,} shuffles, weekly decisions 2008-2023, 5 bps costs, fixed (0/1) sizing\n")
    print(per.round(3).to_string(index=False), "\n")
    for k, v in summary.items():
        print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")


if __name__ == "__main__":
    main()
