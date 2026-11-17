"""Summarize tests #2 (label permutation) and #3 (synthetic markets) from results/*.jsonl.

Run from the repo root:
    research/etf_long_cash/.venv/Scripts/python wip/manas_significance_tests/summarize_retrain_tests.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402


def load(name: str) -> list[dict]:
    path = c.RESULTS / f"{name}_runs.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def p_ge(real: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= real)) / (len(null) + 1))


def main() -> None:
    real = load("real")[0]
    perm, syn = load("permutation"), load("synthetic")
    summary = {"real_replica": {k: real[k] for k in ["portfolio_sharpe", "portfolio_always_long",
                                                       "delta_vs_always_long", "cash_share"]}}

    if perm:
        ps = np.array([r["portfolio_sharpe"] for r in perm])
        per = []
        for t, v in real["per_etf_sharpe"].items():
            null = np.array([r["per_etf_sharpe"][t] for r in perm])
            per.append(dict(etf=t, real=v, null_median=np.median(null), null_95th=np.quantile(null, .95),
                            percentile=(null < v).mean(), p_value=p_ge(v, null)))
        per = pd.DataFrame(per)
        per.to_csv(c.RESULTS / "permutation_per_etf.csv", index=False)
        summary["label_permutation"] = dict(
            runs=len(perm), real_portfolio_sharpe=real["portfolio_sharpe"],
            null_median=float(np.median(ps)), null_5th=float(np.quantile(ps, .05)),
            null_95th=float(np.quantile(ps, .95)), real_percentile=float((ps < real["portfolio_sharpe"]).mean()),
            p_value_one_sided=p_ge(real["portfolio_sharpe"], ps),
            etfs_beating_95pct=int((per.percentile >= .95).sum()))
        print("Test #2 label permutation, per ETF:")
        print(per.round(3).to_string(index=False), "\n")

    if syn:
        d = np.array([r["delta_vs_always_long"] for r in syn])
        s = np.array([r["portfolio_sharpe"] for r in syn])
        a = np.array([r["portfolio_always_long"] for r in syn])
        summary["synthetic_markets"] = dict(
            runs=len(syn), real_delta_vs_always_long=real["delta_vs_always_long"],
            synthetic_delta_median=float(np.median(d)), synthetic_delta_5th=float(np.quantile(d, .05)),
            synthetic_delta_95th=float(np.quantile(d, .95)),
            real_delta_percentile=float((d < real["delta_vs_always_long"]).mean()),
            p_value_one_sided=p_ge(real["delta_vs_always_long"], d),
            synthetic_model_sharpe_median=float(np.median(s)),
            synthetic_always_long_sharpe_median=float(np.median(a)),
            share_synthetic_runs_model_sharpe_above_real=float((s >= real["portfolio_sharpe"]).mean()),
            synthetic_cash_share_median=float(np.median([r["cash_share"] for r in syn])))

    (c.RESULTS / "retrain_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
