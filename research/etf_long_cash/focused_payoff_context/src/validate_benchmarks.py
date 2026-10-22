"""Independent, read-only validation of matched ETF and SPY buy/hold accounts.

The trading result is the existing runs/v3_1 payoff_context account. Benchmarks
buy at its first execution-session adjusted open, use 5 bps entry/exit costs,
mark at adjusted closes and liquidate at the same final close. No model, price
download, reserved-period analysis or portfolio is created by this module.

The closed-form benchmark below independently reproduces research.simulate for
a single initial weight=1 order and comparable terminal liquidation. It can be
used to validate exported bundle curves as well as the original saved ledgers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = "runs/v3_1"
RESERVED_START = "2024-01-01"
LEDGER_COLUMNS = ["nav", "cash", "shares", "weight", "held_exposure", "costs", "turnover",
                  "trade_value", "terminal_sale", "terminal_cost", "opening_nav", "daily_return"]


def load_pre_reserve_prices(path: Path) -> pd.DataFrame:
    """Stop at the first reserved date before parsing any reserved price fields."""
    rows=[]
    with Path(path).open(newline="",encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            day=row["Date"]
            if day >= RESERVED_START:
                break
            rows.append({"Date":pd.Timestamp(day),"adj_open":float(row["adj_open"]),"adj_close":float(row["adj_close"])})
    frame=pd.DataFrame(rows).set_index("Date")
    if frame.empty or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Invalid pre-reserve price dates: {path}")
    if not np.isfinite(frame.to_numpy()).all() or frame.le(0).any().any():
        raise ValueError(f"Invalid adjusted prices: {path}")
    return frame


def read_ledger(path: Path) -> pd.DataFrame:
    frame=pd.read_csv(path,parse_dates=["Date"]).set_index("Date")
    if frame.empty or frame.index.max() >= pd.Timestamp(RESERVED_START):
        raise ValueError(f"Missing or reserved ledger rows: {path}")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Invalid ledger dates: {path}")
    if not np.isfinite(frame[LEDGER_COLUMNS].to_numpy()).all():
        raise ValueError(f"Nonfinite ledger values: {path}")
    return frame


def construct_buyhold(prices: pd.DataFrame, start, end, cost_bps: float=5.) -> pd.DataFrame:
    """Exact algebraic equivalent of a single 100% buy/hold base.simulate order.

    Entry investable notional is 1/(1+fee), not 1-fee. Constant adjusted units
    are held until the final close. First/last-day returns retain both price
    changes and actual entry/liquidation costs; the first close is not rebased.
    """
    price=prices.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if price.empty or price.index[0] != pd.Timestamp(start) or price.index[-1] != pd.Timestamp(end):
        raise ValueError("Benchmark must cover the exact account first/last sessions")
    if price.index.max() >= pd.Timestamp(RESERVED_START):
        raise ValueError("Reserved benchmark prices are not allowed")
    fee=float(cost_bps)/10000.
    if fee<0 or fee>=1:
        raise ValueError("Invalid one-way cost")
    value=1./(1.+fee)
    units=value/float(price.adj_open.iloc[0])
    gross_nav=units*price.adj_close.to_numpy()
    nav=gross_nav.copy()
    terminal=float(gross_nav[-1]);exit_cost=terminal*fee
    nav[-1]-=exit_cost
    n=len(price)
    cash=np.zeros(n);cash[-1]=nav[-1]
    shares=np.full(n,units);shares[-1]=0.
    weight=np.ones(n);weight[-1]=0.
    costs=np.zeros(n);costs[0]=value*fee;costs[-1]+=exit_cost
    turnover=np.zeros(n);turnover[0]=value;turnover[-1]+=1.
    trade_value=np.zeros(n);trade_value[0]=value
    terminal_sale=np.zeros(n);terminal_sale[-1]=terminal
    terminal_cost=np.zeros(n);terminal_cost[-1]=exit_cost
    opening_nav=units*price.adj_open.to_numpy();opening_nav[0]=1.
    daily_return=np.r_[nav[0]-1.,nav[1:]/nav[:-1]-1.]
    result=pd.DataFrame({"nav":nav,"cash":cash,"shares":shares,"weight":weight,"held_exposure":np.ones(n),
                         "costs":costs,"turnover":turnover,"trade_value":trade_value,"terminal_sale":terminal_sale,
                         "terminal_cost":terminal_cost,"opening_nav":opening_nav,"daily_return":daily_return},index=price.index)
    result.index.name="Date"
    return result


def validate_benchmark(strategy: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.DataFrame,
                       name: str, cost_bps: float=5.) -> dict:
    """Validate any exported/saved benchmark against the independent formula."""
    start,end=strategy.index[0],strategy.index[-1]
    expected=construct_buyhold(prices,start,end,cost_bps)
    if not expected.index.equals(strategy.index) or not benchmark.index.equals(strategy.index):
        raise AssertionError(f"{name}: strategy and benchmark sessions differ")
    errors={}
    for column in LEDGER_COLUMNS:
        if column not in benchmark:
            raise AssertionError(f"{name}: missing benchmark column {column}")
        errors[column]=float(np.max(np.abs(expected[column].to_numpy()-benchmark[column].to_numpy())))
        if not np.allclose(expected[column],benchmark[column],rtol=1e-10,atol=1e-10):
            raise AssertionError(f"{name}: {column} differs from exact adjusted-open/cost construction")
    price=prices.loc[start:end]
    identity=benchmark.cash+benchmark.shares*price.adj_close
    if not np.allclose(identity,benchmark.nav,rtol=1e-10,atol=1e-10):
        raise AssertionError(f"{name}: cash/units/NAV identity failed")
    return {"benchmark":name,"status":"PASS","start":str(start.date()),"end":str(end.date()),"days":len(expected),
            "cost_bps_one_way":cost_bps,"initial_NAV_before_first_session":1.,
            "entry_adjusted_open":float(price.adj_open.iloc[0]),"entry_notional":float(expected.trade_value.iloc[0]),
            "entry_fee":float(expected.costs.iloc[0]-expected.terminal_cost.iloc[0]),
            "final_adjusted_close":float(price.adj_close.iloc[-1]),"terminal_fee":float(expected.terminal_cost.iloc[-1]),
            "first_recorded_NAV":float(expected.nav.iloc[0]),"last_recorded_NAV":float(expected.nav.iloc[-1]),
            "first_daily_return":float(expected.daily_return.iloc[0]),"last_daily_return":float(expected.daily_return.iloc[-1]),
            "maximum_absolute_difference_by_column":errors,
            "checks":"Exact session pairing; 12 ledger-column formula checks; cash+units*adjusted-close NAV identity"}


def validate_project(project_root: Path=PROJECT_ROOT) -> dict:
    """Read original sources only and return an audit suitable for bundle export."""
    root=Path(project_root).resolve()
    config=json.loads((root/SOURCE_RUN/"experiment_config.json").read_text())
    tickers=config["universe"]
    spy=load_pre_reserve_prices(root/"data/normalized/prices/SPY.csv")
    results=[]
    for ticker in tickers:
        source=root/SOURCE_RUN/ticker/"ledger_payoff_context_5bps.csv"
        strategy=read_ledger(source)
        # The account begins at the first scheduled model execution, including
        # cases where its first signal is cash. It is not reset at the first buy.
        forecasts=pd.read_csv(root/SOURCE_RUN/ticker/"forecasts.csv",usecols=["variant","entry_date"])
        entries=pd.to_datetime(forecasts.loc[forecasts.variant.eq("payoff_context"),"entry_date"])
        if entries.empty or entries.min()!=strategy.index[0]:
            raise AssertionError(f"{ticker}: account start is not first payoff-context execution session")
        if ((strategy.cash < -1e-10) | (strategy.weight < -1e-10) | (strategy.weight > 1+1e-10)).any():
            raise AssertionError(f"{ticker}: strategy account violates long/cash bounds")
        own=load_pre_reserve_prices(root/f"data/normalized/prices/{ticker}.csv")
        own_period=own.loc[strategy.index]
        if not np.allclose(strategy.cash+strategy.shares*own_period.adj_close,strategy.nav,rtol=1e-10,atol=1e-10):
            raise AssertionError(f"{ticker}: saved strategy account identity failed")
        expected_return=strategy.nav/pd.Series(np.r_[1.,strategy.nav.to_numpy()[:-1]],index=strategy.index)-1
        if not np.allclose(expected_return,strategy.daily_return,rtol=1e-10,atol=1e-10):
            raise AssertionError(f"{ticker}: saved strategy first/subsequent return convention failed")
        pair={"ticker":ticker,"actual_source_run_directory":SOURCE_RUN,"strategy_source":source.relative_to(root).as_posix(),
              "strategy_source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"first_strategy_execution":str(entries.min().date()),
              "pre_reserve_own_price_sha256":hashlib.sha256(own.to_csv(float_format="%.17g").encode()).hexdigest(),
              "benchmarks":[]}
        for name,prices,filename in [("ETF_buy_hold",own,"ledger_buy_hold_5bps.csv"),("SPY_buy_hold",spy,"ledger_SPY_context_5bps.csv")]:
            # Original v2 reference ledgers were generated with base.simulate.
            # They have precisely the same evaluation dates as v3_1 accounts.
            path=root/"runs/v2"/ticker/filename
            benchmark=read_ledger(path)
            validated=validate_benchmark(strategy,prices,benchmark,name)
            validated["original_saved_source"]=path.relative_to(root).as_posix()
            validated["original_saved_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            pair["benchmarks"].append(validated)
        results.append(pair)
    return {"status":"PASS","actual_source_run_directory":SOURCE_RUN,"strategy_variant":"payoff_context",
            "independent_ETF_accounts":len(results),"validated_benchmarks":2*len(results),"reserved_rows_used":0,
            "benchmark_contract":{
                "clock":"Enter at each ETF strategy account's first actual execution-session adjusted open; mark at adjusted closes; liquidate at its identical final close.",
                "account":"Each curve has separate initial NAV 1 before the first session, zero cash yield, no leverage, fractional adjusted units.",
                "entry_cost":"fee=5/10000; purchased notional=1/(1+fee); shares=notional/first adjusted open; entry fee=notional*fee.",
                "exit_cost":"Final gross holding=shares*final adjusted close; deduct gross holding*fee and liquidate.",
                "returns":"First return=first recorded NAV/1-1. Subsequent returns=current NAV/prior recorded NAV-1. Do not rebase first close to1.",
                "distributions":"Use existing consistently adjusted OHLC; no separate dividend credit, avoiding double counting.",
                "timing":"If the first strategy decision is cash, benchmarks still enter at that same first scheduled execution open; do not postpone benchmarks to first positive strategy trade.",
                "scope":"No aggregate portfolio, new model version, simulation of strategy signals, new data acquisition, fitting or reserved-period analysis."
            },"instruments":results}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,default=PROJECT_ROOT)
    parser.add_argument("--json",action="store_true",help="Print full audit JSON for the bundle builder to save")
    args=parser.parse_args()
    audit=validate_project(args.project_root)
    if args.json:
        print(json.dumps(audit,indent=2,allow_nan=False))
    else:
        print(f"PASS: {audit['independent_ETF_accounts']} independent payoff-context accounts, {audit['validated_benchmarks']} exactly paired ETF/SPY benchmarks; reserved rows used=0.")
        for item in audit["instruments"]:
            checks=item["benchmarks"]
            max_error=max(max(b["maximum_absolute_difference_by_column"].values()) for b in checks)
            print(f"{item['ticker']}: {checks[0]['start']} .. {checks[0]['end']}, {checks[0]['days']} sessions; maximum ledger difference {max_error:.3g}")


if __name__=="__main__":
    main()
