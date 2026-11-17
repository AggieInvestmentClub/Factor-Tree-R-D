"""Tests #2 and #3: retrain the v3.1 core trees on data with no real signal.

#2 label permutation (--test permutation)
    Real prices and real features, but each ETF's labels (forward returns used for training) are
    circularly shifted by the same random number of weeks. The labels keep their own
    distribution and autocorrelation, but no longer line up with the features, so anything the
    trees "learn" is noise. The retrained models are scored on the REAL weekly returns.

#3 synthetic markets (--test synthetic)
    Fake price histories made by a stationary block bootstrap of real daily bars (all ETFs on the
    same sampled days, mean block 20 sessions, pool 2006-02 to 2023-12). They keep realistic
    drift, volatility, volatility clustering and cross-ETF correlation, but have no predictable
    pattern beyond ~1 month. The whole pipeline (features, labels, selection, fit) runs on them.

Both use the same replica of the training loop (see common.py), and compare with the replica
run on the real data (--test real). Each run's result is appended to results/*.jsonl, so a
partial run keeps its results.

Run from the repo root, e.g.:
    research/etf_long_cash/.venv/Scripts/python wip/manas_significance_tests/retrain_tests.py --test real
    research/etf_long_cash/.venv/Scripts/python wip/manas_significance_tests/retrain_tests.py --test permutation --n 200
    research/etf_long_cash/.venv/Scripts/python wip/manas_significance_tests/retrain_tests.py --test synthetic --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

LABELS = ["y_5", "y_10", "y_20"]
MIN_SHIFT = 104  # weeks; keeps shifted labels at least two years away from their true dates
_PRICES = None
_EVENTS = None


def _init():
    global _PRICES, _EVENTS
    _PRICES = c.load_prices()
    _EVENTS = c.build_events(_PRICES)


# ------------------------------------------------------------------ data generators
def permuted_events(seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_min = min(int(e[LABELS].notna().all(axis=1).sum()) for e in _EVENTS.values())
    k = int(rng.integers(MIN_SHIFT, n_min - MIN_SHIFT))
    out = {}
    for t, e in _EVENTS.items():
        e = e.copy()
        valid = e[LABELS].notna().all(axis=1).to_numpy()
        e.loc[valid, LABELS] = np.roll(e.loc[valid, LABELS].to_numpy(), k, axis=0)
        out[t] = e
    return out


def synthetic_prices(seed: int, mean_block: int = 20) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = c.UNIVERSE + ["SPY"]
    real = {t: _PRICES[t] for t in names}
    pool_dates = real["DBC"].index  # latest-starting ETF: all 16 series exist from here
    pool_dates = pool_dates[1:]
    rel = {}
    for t in names:
        p = real[t].reindex(real["DBC"].index)
        prev = p["adj_close"].shift(1)
        rel[t] = np.column_stack([(p[f"adj_{x}"] / prev).to_numpy()[1:] for x in ["open", "high", "low", "close"]]
                                 + [p["Volume"].to_numpy()[1:]])
    calendar = real["XLK"].index  # 1998-12 onward
    n, m = len(calendar), len(pool_dates)
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = int(rng.integers(0, m))
        length = int(rng.geometric(1 / mean_block))
        take = (start + np.arange(length)) % m
        idx[i:i + length] = take[: n - i]
        i += length
    out = {}
    for t in names:
        first = real[t].index[0]
        cal = calendar[calendar >= first] if t != "SPY" else calendar
        bars = rel[t][idx[-len(cal):]]
        close = 100 * np.cumprod(bars[:, 3])
        prev = np.r_[100.0, close[:-1]]
        frame = pd.DataFrame({"adj_open": prev * bars[:, 0], "adj_high": prev * bars[:, 1],
                              "adj_low": prev * bars[:, 2], "adj_close": close, "Volume": bars[:, 4]},
                             index=cal)
        frame.index.name = "Date"
        out[t] = frame
    return out


# ------------------------------------------------------------------ one run
def one_run(test: str, seed: int) -> dict:
    t0 = time.time()
    if test == "real":
        events = _EVENTS
    elif test == "permutation":
        events = permuted_events(seed)
    elif test == "synthetic":
        events = c.build_events(synthetic_prices(seed))
    else:
        raise ValueError(test)
    signals, logs = c.replicate(events)
    rets = {t: events[t]["wk_ret"] for t in signals}
    model = c.evaluate(signals, rets)
    allin = c.evaluate(c.always_long(signals), rets)
    return dict(test=test, seed=seed, seconds=round(time.time() - t0, 1),
                portfolio_sharpe=model["portfolio"], portfolio_always_long=allin["portfolio"],
                delta_vs_always_long=model["portfolio"] - allin["portfolio"],
                cash_share=float(np.mean([1 - s.mean() for s in signals.values()])),
                per_etf_sharpe=model["per_etf"], per_etf_always_long=allin["per_etf"],
                signals={t: {str(d.date()): int(v) for d, v in s.items()} for t, s in signals.items()} if test == "real" else None,
                selections=logs if test == "real" else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["real", "permutation", "synthetic"], required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()
    c.RESULTS.mkdir(parents=True, exist_ok=True)
    out = c.RESULTS / f"{args.test}_runs.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["seed"] for line in out.read_text().splitlines() if line.strip()}
    seeds = [s for s in range(args.seed0, args.seed0 + args.n) if s not in done]
    if args.test == "real":
        seeds = [0] if 0 not in done else []
    print(f"{args.test}: {len(seeds)} runs to do ({len(done)} already saved), {args.workers} workers", flush=True)
    start = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, max(1, len(seeds))), initializer=_init) as pool:
        futures = [pool.submit(one_run, args.test, s) for s in seeds]
        for k, f in enumerate(as_completed(futures), 1):
            r = f.result()
            with out.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"[{k}/{len(seeds)}] seed {r['seed']}: model {r['portfolio_sharpe']:.3f} "
                  f"always-long {r['portfolio_always_long']:.3f} ({(time.time() - start) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
