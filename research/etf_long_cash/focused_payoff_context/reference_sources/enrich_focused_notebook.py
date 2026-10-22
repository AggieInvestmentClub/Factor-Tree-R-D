"""Build the statistics-heavy reporting edition; preserve saved trading results."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,shutil
import numpy as np
import pandas as pd
import nbformat as nbf
import build_focused_notebook as base

PROJECT=base.PROJECT
ROOT=base.DEST


def prepare_diagnostics():
    archived=base.WORKSPACE/"work/focused-initial-edition"
    archived.mkdir(exist_ok=True)
    for name in ["payoff_context_review.ipynb","payoff_context_review.html","manifest.json","verification.json"]:
        src=ROOT/name;dst=archived/name
        if src.exists() and not dst.exists():shutil.copyfile(src,dst)
    base.prepare()
    plan={"declared_utc":datetime.now(timezone.utc).isoformat(),"purpose":"Descriptive reporting extension of saved v3.1 payoff_context; no new strategy or selection",
          "reserved_start":"2024-01-01","reserved_scored":False,
          "forecast_labels":"Cached adjusted open at saved label_end / saved entry open -1 - original fixed0.001hurdle. Crop prices before2024 before deriving outcomes; missing reserved/trailing outcomes remain unavailable.",
          "policy_intervals":"Consecutive saved pre-trade execution-opening NAV ratios; final partial interval uses final net close NAV. All starting rebalance fees enter once; compound to the full saved account return.",
          "reliability":"Outer saved predictions only, split by classification/regression and horizon. Labels overlap; no IID accuracy intervals or promotion. Prior OOF fit records remain a separate artifact.",
          "payoff_display_bins":[-.01,-.002,0,.002,.01],"payoff_display_bins_used_for_trading":False,
          "metrics":"Net daily Sharpe primary; zero cash/risk-free convention; annual and monthly compounding retains carry-in and saved costs.",
          "inference":"Existing225-family baseline comparison preserved; new30ETF/SPY paired intervals are descriptive fixed-fit and have no pvalues.",
          "equity_curves_per_etf":["payoff_context","etf_hold","spy_hold"]}
    (ROOT/"data/diagnostic_reporting_plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    manifest=json.loads((ROOT/"manifest.json").read_text())
    forecasts=pd.read_csv(ROOT/"data/forecasts.csv",parse_dates=["decision_date","entry_date","label_end"])
    ledgers=pd.read_csv(ROOT/"data/ledgers.csv",parse_dates=["Date"])
    spy_path=PROJECT/"data/normalized/prices/SPY.csv"
    session_dates=pd.read_csv(spy_path,usecols=["Date"],parse_dates=["Date"])
    session_dates.loc[session_dates.Date<pd.Timestamp("2024-01-01")].to_csv(ROOT/"data/session_calendar.csv",index=False)
    hurdle=json.loads((PROJECT/"config/run_config.json").read_text())["label_roundtrip_hurdle"]
    outcomes=[];intervals=[];oof=[];coverage=[];availability=[]
    for ticker in base.TICKERS:
        path=PROJECT/f"data/normalized/prices/{ticker}.csv"
        prices=pd.read_csv(path,parse_dates=["Date"]).set_index("Date").sort_index()
        prices=prices.loc[prices.index<pd.Timestamp("2024-01-01")].copy()
        manifest["source_files"].append({"path":str(path.relative_to(PROJECT)),"sha256":base.sha(path),"use":"Prices cropped before2024; only pre-reserve target outcomes/data coverage derived"})
        f=forecasts.loc[forecasts.ticker.eq(ticker)].sort_values("decision_date").copy()
        actual=[]
        for r in f.itertuples():
            if pd.isna(r.label_end):
                actual.append(np.nan);continue
            assert r.entry_date in prices.index and r.label_end in prices.index
            j=prices.index.get_loc(r.entry_date)
            assert prices.index[j+int(r.horizon)]==r.label_end
            actual.append(float(prices.loc[r.label_end,"adj_open"]/prices.loc[r.entry_date,"adj_open"]-1-hurdle))
        selected=["ticker","decision_date","entry_date","outer_year","target","horizon","prediction","expected_payoff","signal","target_weight","label_end","payoff_bin_n","oof_rows","feature_count"]
        out=f[selected].copy();out["actual_net_horizon_return"]=actual
        out["actual_gross_horizon_return"]=out.actual_net_horizon_return+hurdle
        out["label_available"]=out.actual_net_horizon_return.notna()
        outcomes.append(out)
        d=ledgers.loc[ledgers.ticker.eq(ticker)&ledgers.strategy.eq("payoff_context")].set_index("Date").sort_index()
        for i,r in enumerate(f.itertuples()):
            start=r.entry_date
            startnav=float(d.loc[start,"opening_nav"])
            partial=i==len(f)-1
            end=d.index[-1] if partial else f.entry_date.iloc[i+1]
            endnav=float(d.nav.iloc[-1]) if partial else float(d.loc[end,"opening_nav"])
            intervals.append({"ticker":ticker,"decision_date":r.decision_date,"entry_date":start,"interval_end":end,
                              "partial_terminal_interval":partial,"start_opening_nav":startnav,"end_nav":endnav,
                              "policy_net_return":endnav/startnav-1,"signal":r.signal,"target_weight":r.target_weight,
                              "sessions":len(d.loc[start:end])-(0 if partial else 1)})
        path=PROJECT/f"runs/v3_1/{ticker}/calibration_oof.csv"
        prior=pd.read_csv(path,float_precision="round_trip").loc[lambda x:x.context.eq(True)].copy()
        prior.insert(0,"ticker",ticker);oof.append(prior)
        manifest["source_files"].append({"path":str(path.relative_to(PROJECT)),"sha256":base.sha(path)})
        coverage.append({"ticker":ticker,"first_price_date":prices.index.min(),"last_included_price_date":prices.index.max(),
                         "pre2024_price_rows":len(prices),"duplicate_price_dates":int(prices.index.duplicated().sum()),
                         "invalid_adjusted_open_close":int(((prices.adj_open<=0)|(prices.adj_close<=0)|prices[["adj_open","adj_close"]].isna().any(axis=1)).sum()),
                         "strategy_start":d.index.min(),"strategy_end":d.index.max(),"strategy_sessions":len(d),
                         "forecast_decisions":len(f),"available_horizon_outcomes":sum(np.isfinite(actual)),
                         "excluded_trailing_horizon_outcomes":sum(~np.isfinite(actual))})
        path=PROJECT/f"data/normalized/etf_context/{ticker}.csv"
        context=pd.read_csv(path,index_col=0,parse_dates=True)
        context=context.reindex(pd.DatetimeIndex(f.decision_date))
        for name in context:
            if name.endswith("_available"):
                availability.append({"ticker":ticker,"availability_flag":name,"decision_rows":len(context),
                                     "mean_available_fraction":float(context[name].mean()),"unmatched_rows":int(context[name].isna().sum())})
    pd.concat(outcomes,ignore_index=True).to_csv(ROOT/"data/forecast_outcomes.csv",index=False)
    pd.DataFrame(intervals).to_csv(ROOT/"data/weekly_policy_intervals.csv",index=False)
    pd.concat(oof,ignore_index=True).to_csv(ROOT/"data/prior_calibration_oof.csv",index=False)
    pd.DataFrame(coverage).to_csv(ROOT/"data/coverage_pre2024.csv",index=False)
    pd.DataFrame(availability).to_csv(ROOT/"data/context_availability.csv",index=False)
    for filename in ["v3_uncertainty.csv","v3_multiplicity.csv","v3_common_period.csv"]:
        p=pd.read_csv(PROJECT/"reports"/filename)
        p=p.loc[p.variant.eq("payoff_context")]
        p.to_csv(ROOT/"data"/filename,index=False)
    supplemental=[]
    for ticker in base.TICKERS:
        p=PROJECT/f"runs/v2/{ticker}/ledger_benchmark_vol_always_5bps.csv"
        d=pd.read_csv(p);d.insert(0,"ticker",ticker);supplemental.append(d)
    pd.concat(supplemental,ignore_index=True).to_csv(ROOT/"data/passive_vol_control.csv",index=False)
    for filename in ["literature_review.json","context_source_roadmap.json"]:
        shutil.copyfile(PROJECT/"reports"/filename,ROOT/"reference_sources"/filename)
    manifest["reporting_edition"]="statistics-heavy"
    manifest["source_files"].append({"path":str(spy_path.relative_to(PROJECT)),"sha256":base.sha(spy_path),
                                      "use":"Pre2024 SPY-observed session dates only, for exact partial-month/year flags"})
    manifest["reporting_plan_sha256"]=base.sha(ROOT/"data/diagnostic_reporting_plan.json")
    (ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


def expanded_notebook():
    original=base.notebook()
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    intro=original.cells[:3]
    intro[0].source=intro[0].source.replace("focused research notebook","statistics-heavy research notebook")
    intro[1].source += "\nfrom statistical_views import StatisticalReport\nS = StatisticalReport(ROOT)\n"
    cells=intro+[
        md("""## Research dashboard · all 15 separate accounts

The tables below summarize the current combined procedure. Cross-ETF summaries do not combine capital or create a portfolio. Main equity charts retain exactly the three requested curves.

All derived diagnostics use saved results through 2023. Forecast-target outcomes and weekly account intervals are separate views; neither is a new trading strategy. New benchmark confidence intervals are descriptive after historical inspection."""),
        code("display(HTML(S.overview_tables()))\nS.overview_figure()"),
        md("""### Metric definitions and data eligibility

Sharpe uses net simple daily returns and zero risk-free rate. Sortino uses a zero target and the square root of the mean squared negative daily returns across all days. Calmar divides CAGR by the magnitude of maximum drawdown. Tail returns are signed: a more negative 5% tail mean is worse. Annual/monthly returns compound the existing account's saved returns, retaining first-session costs and carry-in positions. Partial periods are identified.

Forecast-label reliability uses only saved outer forecasts with outcomes observable before 2024. The 5/10/20-session labels overlap. Raw classification probabilities, regression returns and calibrated expected payoffs have distinct diagnostics. Fitted prior OOF calibration bins are shown separately from outer forecast evaluation.

Factor importance describes splits in fitted trees and does not establish causality. Model recipe choices, horizons and sizing were chosen on earlier folds; their retrospective stability is descriptive."""),
        code("display(HTML(S.coverage_tables()))"),
    ]
    for ticker in base.TICKERS:
        cells += [
            md(f"## {ticker} · equity, trades and model settings"),
            code(f"R.plot('{ticker}')"),
            code(f"display(HTML(R.details('{ticker}')))"),
            md(f"### {ticker} · risk, return distribution and time stability"),
            code(f"display(HTML(S.risk_tables('{ticker}')))"),
            code(f"S.temporal_figure('{ticker}')"),
            code(f"display(HTML(S.temporal_tables('{ticker}')))"),
            md(f"### {ticker} · forecasts, payoff reliability and actual weekly intervals"),
            code(f"display(HTML(S.forecast_tables('{ticker}')))"),
            code(f"S.forecast_figure('{ticker}')"),
            md(f"### {ticker} · factors, exposure and benchmark evidence"),
            code(f"S.factor_figure('{ticker}')"),
            code(f"display(HTML(S.benchmark_tables('{ticker}')))"),
        ]
    cells += [
        md("""## Literature, limitations and next research questions

This is a statistical reporting extension of the same v3.1 model. It does not add strategy variants or use the reserved period. The parent study's 225-comparison multiplicity history remains relevant even when only the combined procedure is displayed.

Priorities are calibration support and stability, the effect of turnover and exposure, simple trend/passive-volatility controls, and qualified direct economic factors. Macro and FinBERT remain outside the fitted model until historical data availability is qualified. Vendor-adjusted synthetic total returns, XLF's 2016 distribution, zero cash yield and constant fees remain practical limitations."""),
        code("display(HTML(S.literature_tables()))\ndisplay(HTML(R.verification()))"),
    ]
    original.cells=cells
    original.metadata.swing_tree["reporting_edition"]="statistics-heavy"
    original.metadata.swing_tree["strategy_unchanged"]=True
    return original


if __name__=="__main__":
    prepare_diagnostics()
    if "--prepare-only" not in sys.argv:
        sys.path.insert(0,str(ROOT/"src"))
        from statistical_views import StatisticalReport
        S=StatisticalReport(ROOT)
        S.export_tables()
        base.execute(expanded_notebook())
