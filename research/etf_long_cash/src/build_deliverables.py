"""Build the executed, read-only research notebook and self-contained HTML report.

This module only reads saved research outputs. It never fits a trading model,
reselects a recipe, downloads data, or consumes reserved prices for analysis.
Run with the project's isolated Python: python src/build_deliverables.py.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT.parent.parent / "work" / "report-runtime"
for name in ["mpl", "jupyter", "jupyter-runtime", "ipython"]:
    (SCRATCH / name).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(SCRATCH / "mpl"))
os.environ.setdefault("JUPYTER_CONFIG_DIR", str(SCRATCH / "jupyter"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(SCRATCH / "jupyter-runtime"))
os.environ.setdefault("IPYTHONDIR", str(SCRATCH / "ipython"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import PercentFormatter, LogLocator, FuncFormatter
import numpy as np
import pandas as pd

COLORS = {"primary": "#087e82", "buy": "#26364d", "spy": "#9a879d", "bad": "#b75e49", "good": "#087e82", "muted": "#697586"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                    "axes.spines.right": False, "axes.labelcolor": "#354154", "text.color": "#1c2b40",
                    "axes.titleweight": "bold", "axes.titlesize": 13, "figure.facecolor": "#ffffff",
                    "axes.facecolor": "#ffffff", "grid.color": "#e6eaf0", "grid.alpha": 0.8,
                    "savefig.facecolor": "white"})


class ResearchReport:
    """Convenience readers and descriptive figures for already completed runs."""

    def __init__(self, root=ROOT, require_complete=True):
        self.root = Path(root)
        self.runs = self.root / "runs/v2"
        self.config = json.loads((self.runs / "run_config.json").read_text())
        self.tickers = self.config["universe"]
        self.metrics = pd.read_csv(self.root / "reports/metrics.csv")
        self.coverage = pd.read_csv(self.root / "data/coverage.csv")
        self.data_manifest = json.loads((self.root / "data/data_manifest.json").read_text())
        self.finbert = json.loads((self.root / "reports/finbert_status.json").read_text())
        self.figure_dir = self.root / "reports/figures"
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        expected = len(self.tickers) * 34
        if require_complete and (len(self.metrics) != expected or set(self.metrics.ticker) != set(self.tickers)):
            raise RuntimeError(f"Wait for all 15 completed ETF runs: expected {expected} metrics rows, got {len(self.metrics)}")
        self.tickers = [t for t in self.tickers if t in set(self.metrics.ticker)]
        self.primary = self.rows("CART", "selected", 5).set_index("ticker").reindex(self.tickers)
        self.buy = self.rows("buy_hold", "static", 5).set_index("ticker").reindex(self.tickers)
        self.spy = self.rows("SPY_context", "static", 5).set_index("ticker").reindex(self.tickers)
        self.selection = pd.concat([self.read(t, "selections.csv").assign(ticker=t) for t in self.tickers], ignore_index=True)
        self.usage = pd.concat([self.read(t, "feature_usage.csv").assign(ticker=t) for t in self.tickers], ignore_index=True)
        self.intervals = pd.DataFrame([dict(ticker=t, lower=self.uncertainty(t)[0], upper=self.uncertainty(t)[1]) for t in self.tickers]).set_index("ticker")
        comparison_path=self.root/"reports/sharpe_comparison.csv"
        if comparison_path.exists():
            comparison=pd.read_csv(comparison_path).set_index("ticker")
            if not set(self.tickers).issubset(comparison.index):raise RuntimeError("Canonical Sharpe audit is incomplete")
            self.sharpe_intervals=comparison[["delta_sharpe_95pct_low","delta_sharpe_95pct_high"]].rename(columns={"delta_sharpe_95pct_low":"lower","delta_sharpe_95pct_high":"upper"}).reindex(self.tickers)
            self.sharpe_interval_source="reports/sharpe_comparison.csv (canonical audit)"
        else:
            if require_complete:raise RuntimeError("Wait for canonical reports/sharpe_comparison.csv before final report execution")
            self.sharpe_intervals = self.sharpe_bootstrap()
            self.sharpe_interval_source="temporary preview calculation"

    def rows(self, strategy, sizing, bps):
        d = self.metrics
        return d.loc[d.strategy.eq(strategy) & d.sizing.eq(sizing) & d.cost_bps.eq(bps)].copy()

    def read(self, ticker, filename):
        try:
            return pd.read_csv(self.runs / ticker / filename)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def ledger(self, ticker, name="CART_selected_5bps"):
        return self.read(ticker, f"ledger_{name}.csv").set_index("Date").rename_axis("Date").pipe(lambda d: d.set_axis(pd.to_datetime(d.index)))

    def uncertainty(self, ticker):
        return json.loads((self.runs / ticker / "uncertainty.json").read_text())["annual_mean_return_difference_95pct"]

    def sharpe_bootstrap(self):
        rows=[]
        for i,ticker in enumerate(self.tickers):
            a=self.ledger(ticker).daily_return.to_numpy()
            b=self.ledger(ticker,"buy_hold_5bps").daily_return.to_numpy()
            if len(a)!=len(b):raise ValueError("Sharpe bootstrap requires matched dates")
            rng=np.random.default_rng(42+i);n=len(a);block=20
            starts=rng.integers(0,max(1,n-block+1),size=(500,int(np.ceil(n/block))))
            ix=(starts[:,:,None]+np.arange(block)[None,None,:]).reshape(500,-1)[:,:n]
            aa=a[ix];bb=b[ix]
            sa=np.divide(aa.mean(axis=1),aa.std(axis=1,ddof=1),out=np.zeros(500),where=aa.std(axis=1,ddof=1)>0)*np.sqrt(252)
            sb=np.divide(bb.mean(axis=1),bb.std(axis=1,ddof=1),out=np.zeros(500),where=bb.std(axis=1,ddof=1)>0)*np.sqrt(252)
            low,high=np.quantile(sa-sb,[.025,.975])
            rows.append({"ticker":ticker,"lower":low,"upper":high})
        return pd.DataFrame(rows).set_index("ticker")

    def save(self, fig, name):
        path = self.figure_dir / f"{name}.png"
        fig.savefig(path, dpi=145, bbox_inches="tight")
        fig.savefig(self.figure_dir / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def table(frame, percentages=(), decimals=(), integers=()):
        d = frame.copy()
        for col in percentages:
            if col in d:
                d[col] = d[col].map(lambda v: f"{v:.2%}" if pd.notna(v) else "—")
        for col in decimals:
            if col in d:
                d[col] = d[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        for col in integers:
            if col in d:
                d[col] = d[col].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        return d.to_html(index=False, border=0, classes="research-table", escape=True, na_rep="—")

    def overview_html(self):
        n = len(self.tickers)
        wins = int((self.primary.sharpe > self.buy.sharpe).sum())
        safer = int((self.primary.max_drawdown > self.buy.max_drawdown).sum())
        ci_positive = int((self.sharpe_intervals.lower > 0).sum())
        ci_negative = int((self.sharpe_intervals.upper < 0).sum())
        body = f"""<div class='cards'><div><strong>{n}</strong><span>independent ETF accounts</span></div>
        <div><strong>{wins} / {n}</strong><span>CART net Sharpe above ETF buy/hold</span></div>
        <div><strong>{safer} / {n}</strong><span>smaller worst drawdown</span></div>
        <div><strong>{ci_positive} / {n}</strong><span>Sharpe-difference intervals above zero</span></div></div>
        <p>These counts compare the selected CART control at 5 bps one-way with each ETF's buy/hold account on identical dates.
        They describe separate accounts and different eligible starting dates. They are not portfolio results.</p>
        <p>{ci_negative} Sharpe-difference intervals lie entirely below zero; all other intervals cross zero. Sharpe uses net daily returns with a zero cash/risk-free rate.
        Paired moving-block bootstrap intervals are descriptive after selection; they do not establish robust outperformance.</p>"""
        return body

    def summary_table(self):
        d = pd.DataFrame({"ETF": self.tickers, "Start": self.primary.start.values, "End": self.primary.end.values,
                          "CART Sharpe": self.primary.sharpe.values, "ETF Sharpe": self.buy.sharpe.values,
                          "SPY context Sharpe": self.spy.sharpe.values, "CART − ETF Sharpe": (self.primary.sharpe-self.buy.sharpe).values,
                          "CART drawdown": self.primary.max_drawdown.values, "ETF drawdown": self.buy.max_drawdown.values})
        return self.table(d, percentages=d.columns[-2:],decimals=d.columns[3:-2])

    def return_table(self):
        d=pd.DataFrame({"ETF":self.tickers,"CART CAGR":self.primary.cagr.values,"ETF CAGR":self.buy.cagr.values,
                        "SPY context CAGR":self.spy.cagr.values,"CART total return":self.primary.total_return.values,
                        "ETF total return":self.buy.total_return.values,"CART annual vol":self.primary.volatility.values,
                        "ETF annual vol":self.buy.volatility.values})
        return self.table(d,percentages=d.columns[1:])

    def execution_table(self):
        d = self.primary.reset_index()[["ticker", "average_exposure", "turnover_per_year", "trade_events", "average_holding_sessions", "cost_normalized_currency"]]
        d.columns = ["ETF", "Mean invested weight", "Annual turnover (× NAV)", "Trade events", "Mean holding sessions", "Fees in NAV-1 currency"]
        return self.table(d, percentages=["Mean invested weight"], decimals=["Annual turnover (× NAV)", "Mean holding sessions", "Fees in NAV-1 currency"], integers=["Trade events"])

    def comparison_figure(self):
        fig, ax = plt.subplots(figsize=(13, 6.5))
        y = np.arange(len(self.tickers)); height = .26
        ax.barh(y-height/2, self.primary.sharpe, height, color=COLORS["primary"], label="Selected CART · 5 bps")
        ax.barh(y+height/2, self.buy.sharpe, height, color=COLORS["buy"], label="ETF buy/hold · 5 bps")
        ax.set(yticks=y, yticklabels=self.tickers, xlabel="Annualized Sharpe ratio · net daily returns · zero risk-free rate", title="Primary criterion: net Sharpe versus each ETF")
        ax.invert_yaxis(); ax.axvline(0, color="#8b95a5", lw=.8); ax.grid(axis="x"); ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right")
        fig.text(.125, .015, "Separate accounts; dates vary by eligibility. Comparison does not establish predictive alpha.", color=COLORS["muted"], fontsize=9)
        return self.save(fig, "01_individual_sharpe")

    def workflow_figure(self):
        fig, ax = plt.subplots(figsize=(14, 8)); ax.axis("off"); ax.set(xlim=(0, 14), ylim=(0, 8))
        boxes = [(0.2,6.2,"01 · Data + provenance","15 ETF histories, SPY/UUP\nRaw responses, actions, SHA-256"),
                 (4.9,6.2,"02 · Quality + availability gate","Coverage/calendar/price checks\nMacro and news remain gated"),
                 (9.6,6.2,"03 · Causal weekly features","20 technical + 20 relationship\nCompleted Friday week; next open"),
                 (9.6,3.7,"04 · Earlier inner research","5/10/20 sessions × 2 targets\nCART screen + fixed/vol sizing"),
                 (4.9,3.7,"05 · Freeze annual specification","≥104 matured weekly rows\nStrict 20-session maturity gate"),
                 (0.2,3.7,"06 · Refit + stored forecasts","Annual expanding, uniform weights\nCART + RF/XGB challengers"),
                 (0.2,1.2,"07 · Independent long/cash ledger","Weight 0–100%; NAV starts at 1\n0/5/10 bps on actual turnover"),
                 (4.9,1.2,"08 · Outer scoring + diagnostics","Yearly outer intervals to 2023\nCosts, exposure, stability, uncertainty"),
                 (9.6,1.2,"09 · Freeze next version","2024+ reserve left unscored\nThen prospective validation")]
        for x,y,title,body in boxes:
            patch = FancyBboxPatch((x,y),4.1,1.35,boxstyle="round,pad=0.1,rounding_size=0.12",facecolor="#eef6f6",edgecolor="#bad6d7",linewidth=1)
            ax.add_patch(patch); ax.text(x+.18,y+1.05,title,fontsize=11,fontweight="bold",va="top")
            ax.text(x+.18,y+.65,body,fontsize=10,va="top",linespacing=1.65)
        pairs=[((4.4,6.9),(4.8,6.9)),((9.1,6.9),(9.5,6.9)),((11.65,6.1),(11.65,5.2)),
               ((9.5,4.4),(9.1,4.4)),((4.8,4.4),(4.4,4.4)),((2.25,3.6),(2.25,2.7)),
               ((4.4,1.9),(4.8,1.9)),((9.1,1.9),(9.5,1.9))]
        for start,end in pairs:
            ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=14,color=COLORS["primary"],linewidth=1.7))
        fig.suptitle("The requested workflow, staged by data eligibility",x=.055,ha="left",fontsize=17,fontweight="bold")
        fig.text(.055,.02,"Implemented now: baseline factors, nested chronology, model ladder, fixed/vol sizing. Later gates: PIT macro/news, calibrated payoff sizing, recency/drift policies.",fontsize=9,color=COLORS["muted"])
        return self.save(fig, "02_workflow")

    def coverage_table(self):
        d = self.coverage.loc[self.coverage.ticker.isin(self.tickers)].set_index("ticker").reindex(self.tickers).reset_index()
        d = d[["ticker","first_date","last_date","rows","dividend_events","split_events","missing_spy_sessions","status"]]
        d.columns=["ETF","First session","Last session","Daily rows","Dividends","Splits","Missing sessions","QA"]
        return self.table(d,integers=["Daily rows","Dividends","Splits","Missing sessions"])

    def macro_table(self):
        rows=[]
        for source in self.data_manifest["sources"]:
            if "series" in source:
                rows.append({"Series":source["series"],"First returned":source["first_date"],"Last returned":source["last_date"],
                             "Rows":source["rows"],"Source missing rows":source["missing_rows"],
                             "Use in fitted models":"Excluded · vintage gate"})
        return self.table(pd.DataFrame(rows),integers=["Rows","Source missing rows"])

    def macro_figure(self):
        d=pd.read_csv(self.root/"data/normalized/macro_features.csv",parse_dates=["Date"]).set_index("Date")
        d=d.loc[(d.index>="2000-01-01")&(d.index<self.config["reserved_start"])]
        fig,axes=plt.subplots(2,1,figsize=(13,5.8),sharex=True)
        axes[0].plot(d.index,d.macro_term_spread,color=COLORS["primary"],lw=1);axes[0].axhline(0,color="#a5adb8",lw=.7)
        axes[0].set(ylabel="10y − 2y (pp)",title="Archived macro examples · diagnostic only, excluded from fitted strategies")
        axes[1].plot(d.index,d.macro_vixcls,color=COLORS["buy"],lw=.9);axes[1].set(ylabel="VIX level")
        for ax in axes: ax.grid(axis="y");ax.set_axisbelow(True)
        fig.text(.125,.01,"Latest-vintage FRED observations shifted by two business days. This lag does not establish original release availability or revision history.",fontsize=9,color=COLORS["muted"])
        fig.tight_layout(rect=(0,.04,1,1));return self.save(fig,"03_macro_archive")

    def instrument_figure(self,ticker):
        fig,axes=plt.subplots(2,1,figsize=(13,6.3),sharex=True,gridspec_kw={"height_ratios":[2.1,1]})
        for tag,label,color,style in [("CART_selected_5bps","Selected CART",COLORS["primary"],"-"),("buy_hold_5bps",f"{ticker} buy/hold",COLORS["buy"],"-"),("SPY_context_5bps","SPY context",COLORS["spy"],"--")]:
            d=self.ledger(ticker,tag);peak=d.nav.cummax().clip(lower=1);dd=d.nav/peak-1
            axes[0].plot(d.index,d.nav,label=label,color=color,ls=style,lw=1.9 if tag.startswith("CART") else 1.2)
            axes[1].plot(d.index,dd,color=color,ls=style,lw=1.2)
        p=self.primary.loc[ticker];b=self.buy.loc[ticker]
        axes[0].set(yscale="log",ylabel="NAV · logarithmic scale",title=f"{ticker}  |  {p.start} to {p.end}")
        axes[0].yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:g}"))
        axes[0].legend(frameon=False,ncol=3,loc="upper left");axes[1].set(ylabel="Drawdown")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].axhline(0,color="#b0b8c4",lw=.5)
        for ax in axes:ax.grid(axis="y");ax.set_axisbelow(True)
        fig.text(.125,.015,f"5 bps one-way. Sharpe: CART {p.sharpe:.2f}, ETF {b.sharpe:.2f}. CAGR: {p.cagr:.2%} / {b.cagr:.2%}. Worst drawdown: {p.max_drawdown:.2%} / {b.max_drawdown:.2%}. Separate NAV-1 account.",fontsize=9,color=COLORS["muted"])
        fig.tight_layout(rect=(0,.045,1,1));return self.save(fig,f"equity_drawdown_{ticker}")

    def model_table(self):
        rows=[]
        for ticker in self.tickers:
            row={"ETF":ticker}
            for family in ["CART","RF","XGB"]:
                d=self.rows(family,"selected",5).set_index("ticker").loc[ticker]
                row[family+" Sharpe"]=d.sharpe;row[family+" CAGR"]=d.cagr
            rows.append(row)
        return self.table(pd.DataFrame(rows),percentages=[x+" CAGR" for x in ["CART","RF","XGB"]],decimals=[x+" Sharpe" for x in ["CART","RF","XGB"]])

    def sizing_table(self):
        rows=[]
        for ticker in self.tickers:
            row={"ETF":ticker}
            for size,label in [("fixed","Fixed"),("vol","10% vol target"),("selected","Inner-selected")]:
                d=self.rows("CART",size,5).set_index("ticker").loc[ticker]
                row[label+" Sharpe"]=d.sharpe;row[label+" CAGR"]=d.cagr;row[label+" drawdown"]=d.max_drawdown
            rows.append(row)
        d=pd.DataFrame(rows);return self.table(d,percentages=[c for c in d.columns if "CAGR" in c or "drawdown" in c],decimals=[c for c in d.columns if "Sharpe" in c])

    def cost_table(self):
        rows=[]
        for ticker in self.tickers:
            vals={bps:self.rows("CART","selected",bps).set_index("ticker").loc[ticker] for bps in [0,5,10]}
            rows.append({"ETF":ticker,"0 bps Sharpe":vals[0].sharpe,"5 bps Sharpe":vals[5].sharpe,"10 bps Sharpe":vals[10].sharpe,
                         "0→5 bps CAGR reduction":vals[0].cagr-vals[5].cagr,"Annual turnover":vals[5].turnover_per_year})
        return self.table(pd.DataFrame(rows),percentages=["0→5 bps CAGR reduction"],decimals=["0 bps Sharpe","5 bps Sharpe","10 bps Sharpe","Annual turnover"])

    def selection_figure(self):
        fig,axes=plt.subplots(2,2,figsize=(14,8))
        specs=[("recipe",["technical","relationships"],"Factor recipe"),("horizon",[5,10,20],"Target horizon (sessions)"),
               ("target",["classification","regression"],"Target type"),("sizing",["fixed","vol"],"Sizing selected on earlier folds")]
        palette=[COLORS["primary"],"#425a76","#a4c6c5"]
        for ax,(field,options,title) in zip(axes.flat,specs):
            table=pd.crosstab(self.selection.ticker,self.selection[field],normalize="index").reindex(self.tickers).reindex(columns=options,fill_value=0)
            bottom=np.zeros(len(self.tickers))
            for i,col in enumerate(table):
                ax.bar(self.tickers,table[col],bottom=bottom,color=palette[i],label=str(col));bottom+=table[col].to_numpy()
            ax.set(ylim=(0,1));ax.set_title(title,pad=36,fontsize=12);ax.tick_params(axis="x",rotation=60);ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.legend(frameon=False,fontsize=9,ncol=len(options),loc="lower center",bbox_to_anchor=(.5,1.0));ax.grid(axis="y");ax.set_axisbelow(True)
        fig.suptitle("What earlier inner windows selected for each ETF",x=.055,ha="left",fontsize=17,fontweight="bold")
        fig.text(.055,.01,"Shares of qualified annual outer fits, not shares of capital or evidence of causality. Macro recipes were registered but rejected by the availability gate.",fontsize=9,color=COLORS["muted"])
        fig.tight_layout(rect=(0,.04,1,.95));return self.save(fig,"04_recipe_selections")

    def factor_figure(self):
        u=self.usage.loc[self.usage.family.eq("CART")].copy()
        u["group"]=u.feature.str.split("_").str[0].map({"tech":"Technical","rel":"Relationships","macro":"Macro"})
        perfit=u.groupby(["ticker","year","group"]).importance.sum().unstack(fill_value=0)
        means=perfit.groupby("ticker").mean().reindex(self.tickers).fillna(0)
        fig,axes=plt.subplots(1,2,figsize=(14,6),gridspec_kw={"width_ratios":[1.1,1]})
        left=np.zeros(len(self.tickers))
        for group,color in [("Technical",COLORS["primary"]),("Relationships",COLORS["buy"])]:
            vals=means[group] if group in means else np.zeros(len(means))
            axes[0].barh(self.tickers,vals,left=left,color=color,label=group);left+=np.asarray(vals)
        axes[0].invert_yaxis();axes[0].set(xlabel="Mean share of impurity reduction")
        axes[0].set_title("CART importance by eligible family",pad=34)
        axes[0].xaxis.set_major_formatter(PercentFormatter(1));axes[0].legend(frameon=False,loc="lower center",bbox_to_anchor=(.5,1.0),ncol=2)
        top=u.groupby("feature").importance.sum().div(self.selection.shape[0]).sort_values().tail(12)
        axes[1].barh(top.index,top.values,color=COLORS["primary"]);axes[1].set(title="Most used features across annual CART fits",xlabel="Mean impurity importance across all fits")
        axes[1].tick_params(axis="y",labelsize=9)
        for ax in axes:ax.grid(axis="x");ax.set_axisbelow(True)
        fig.text(.07,.015,"Descriptive fitted usage; correlated factors and tree impurity biases prevent causal interpretation. Unselected relationship recipes contribute zero usage.",fontsize=9,color=COLORS["muted"])
        fig.tight_layout(rect=(0,.05,1,1));return self.save(fig,"05_factor_usage")

    def factor_example(self,ticker="XLK"):
        if ticker not in self.tickers:ticker=self.tickers[0]
        e=self.read(ticker,"events.csv");e["decision_date"]=pd.to_datetime(e.decision_date);e=e.set_index("decision_date").loc["2017-01-01":"2023-12-31"]
        p=pd.read_csv(self.root/f"data/normalized/prices/{ticker}.csv",parse_dates=["Date"]).set_index("Date").loc["2017-01-01":"2023-12-31"]
        fig,axes=plt.subplots(4,1,figsize=(13,8),sharex=True)
        axes[0].plot(p.index,p.adj_close/p.adj_close.iloc[0],color=COLORS["buy"]);axes[0].set(ylabel="Price index",title=f"{ticker} · examples of the actual saved causal factors")
        for ax,col,label in [(axes[1],"tech_mom_20","20-session momentum"),(axes[2],"tech_vol_63","63-session annual vol"),(axes[3],"rel_strength_63","63-session relative strength")]:
            if col in e:ax.plot(e.index,e[col],color=COLORS["primary"],lw=1);ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylabel(label,fontsize=9);ax.axhline(0,color="#a5adb8",lw=.5)
        for ax in axes:ax.grid(axis="y");ax.set_axisbelow(True)
        fig.text(.125,.01,"Weekly factors use the completed decision close and past observations; trades execute at the next session open. Relative strength subtracts SPY momentum.",fontsize=9,color=COLORS["muted"])
        fig.tight_layout(rect=(0,.04,1,1));return self.save(fig,"06_factor_example")

    def feature_inventory_html(self):
        d=pd.read_csv(self.root/"data/feature_registry.csv");parts=[]
        parts.append("<p><strong>54 computed numeric features: 20 technical, 20 relationships and 14 quarantined macro.</strong> Only the 40 price-based factors are eligible in this run. The feature formula version remains v1; run v2 corrects calendar handling without changing the factor formulas.</p>")
        for family,label in [("technical","Technical · 20 eligible"),("relationships","Relationships · 20 eligible"),("macro","Macro · 14 computed, excluded")]:
            x=d.loc[d.family.eq(family),["feature_name","formula","unit","lookback_sessions"]].copy()
            x.columns=["Feature","Exact formula / derivation","Unit","Lookback sessions"]
            parts.append(f"<details><summary>{label}</summary><div class='registry-table'>"+self.table(x)+"</div></details>")
        parts.append("<p>Full units, source inputs, minimum history, missingness policy and availability rules: <a href='data/feature_registry.csv'>feature_registry.csv</a> and <a href='reports/feature_contract.json'>feature_contract.json</a>.</p>")
        return "".join(parts)

    def annual_figure(self):
        frames=[self.read(t,"annual_primary.csv").assign(ticker=t) for t in self.tickers]
        d=pd.concat(frames).pivot(index="ticker",columns="year",values="total_return").reindex(self.tickers)
        limit=max(.10,float(np.nanpercentile(np.abs(d.to_numpy()),95)))
        fig,ax=plt.subplots(figsize=(14,7));cmap=plt.get_cmap("RdYlGn").copy();cmap.set_bad("#e8edf1")
        im=ax.imshow(d.to_numpy(),aspect="auto",cmap=cmap,vmin=-limit,vmax=limit)
        ax.set(xticks=np.arange(len(d.columns)),xticklabels=d.columns,yticks=np.arange(len(d)),yticklabels=d.index,
               title="Annual net returns · selected CART, 5 bps · separate accounts")
        for i in range(len(d)):
            for j in range(len(d.columns)):
                val=d.iloc[i,j]
                ax.text(j,i,"—" if pd.isna(val) else f"{val:.0%}",ha="center",va="center",fontsize=8,color="#26364d")
        fig.colorbar(im,ax=ax,pad=.02,format=PercentFormatter(1),label="Calendar-year net return")
        fig.text(.125,.015,"Grey cells: insufficient earlier history for a qualified annual selection. Color scale clips at the 95th percentile; printed returns are untruncated.",fontsize=9,color=COLORS["muted"])
        fig.tight_layout(rect=(0,.04,1,1));return self.save(fig,"07_annual_stability")

    def uncertainty_figure(self):
        fig,ax=plt.subplots(figsize=(13,6.5));y=np.arange(len(self.tickers));ci=self.intervals.reindex(self.tickers)
        for i,ticker in enumerate(self.tickers):
            lower,upper=ci.loc[ticker,["lower","upper"]]
            color=COLORS["good"] if lower>0 else COLORS["bad"] if upper<0 else "#697b92"
            diff=(self.ledger(ticker).daily_return-self.ledger(ticker,"buy_hold_5bps").daily_return).mean()*252
            ax.plot([lower,upper],[i,i],color=color,lw=3);ax.plot(diff,i,"o",color=color,ms=5)
        ax.axvline(0,color="#26364d",lw=1,ls="--");ax.set(yticks=y,yticklabels=self.tickers,xlabel="Annualized mean daily return difference versus ETF buy/hold",title="Historical uncertainty · 95% moving-block bootstrap intervals")
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.invert_yaxis();ax.grid(axis="x");ax.set_axisbelow(True)
        fig.text(.125,.01,"20-session blocks · 500 resamples · includes overlapping selected outcomes and multiple comparisons. These are descriptive intervals, not a validated alpha test.",fontsize=9,color=COLORS["muted"])
        fig.tight_layout(rect=(0,.04,1,1));return self.save(fig,"08_bootstrap_uncertainty")

    def sharpe_uncertainty_figure(self):
        fig,ax=plt.subplots(figsize=(13,6.5));ci=self.sharpe_intervals.reindex(self.tickers)
        for i,ticker in enumerate(self.tickers):
            lower,upper=ci.loc[ticker,["lower","upper"]]
            color=COLORS["good"] if lower>0 else COLORS["bad"] if upper<0 else "#697b92"
            ax.plot([lower,upper],[i,i],color=color,lw=3)
            ax.plot(self.primary.loc[ticker,"sharpe"]-self.buy.loc[ticker,"sharpe"],i,"o",color=color,ms=5)
        ax.axvline(0,color="#26364d",lw=1,ls="--");ax.set(yticks=np.arange(len(self.tickers)),yticklabels=self.tickers,
            xlabel="CART net Sharpe − ETF buy/hold net Sharpe",title="Primary-metric uncertainty · 95% paired moving-block intervals")
        ax.invert_yaxis();ax.grid(axis="x");ax.set_axisbelow(True)
        fig.text(.125,.01,"Canonical saved audit · 20-session blocks · 500 paired resamples. Descriptive after selection; no multiple-comparison or full selection-process correction.",fontsize=9,color=COLORS["muted"])
        fig.tight_layout(rect=(0,.04,1,1));return self.save(fig,"08a_sharpe_uncertainty")

    def calibration_figure(self):
        cols=3;rows=int(np.ceil(len(self.tickers)/cols));fig,axes=plt.subplots(rows,cols,figsize=(13,2.6*rows),squeeze=False)
        for ax,ticker in zip(axes.flat,self.tickers):
            d=self.read(ticker,"calibration.csv")
            d=d.loc[d.family.eq("CART")] if "family" in d else pd.DataFrame()
            ax.plot([0,1],[0,1],color="#a5adb8",lw=.8,ls="--")
            if len(d):
                ax.plot(d.predicted,d.observed,color=COLORS["primary"],lw=1.2)
                ax.scatter(d.predicted,d.observed,s=12+np.sqrt(d.n)*5,color=COLORS["primary"])
                brier=np.average(d.brier,weights=d.n)
                ax.set_title(f"{ticker} · n={int(d.n.sum())} · Brier {brier:.3f}",fontsize=10)
            else:ax.set_title(f"{ticker} · no classification folds",fontsize=10)
            ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1]);ax.grid(alpha=.4)
        for ax in list(axes.flat)[len(self.tickers):]:ax.axis("off")
        fig.supxlabel("Raw predicted probability",fontsize=10);fig.supylabel("Observed positive net-horizon outcome fraction",fontsize=10)
        fig.suptitle("Raw CART reliability · selected classification years only",fontsize=16,fontweight="bold",y=.995)
        fig.tight_layout(rect=(.02,.03,1,.97));return self.save(fig,"09_calibration")

    def finbert_html(self):
        m=self.finbert.get("model",{});corpus=self.finbert.get("news_corpus",{});scored=self.finbert.get("archive_inference",{})
        labels=scored.get("label_counts",{})
        return f"""<p><strong>Local inference is working.</strong> ProsusAI/finbert revision <code>{html.escape(m.get('revision','unknown'))}</code>
        is pinned and its downloaded files are hashed. The CPU smoke test completed. The real archive contains
        {corpus.get('articles',0)} official FOMC statements ({html.escape(str(corpus.get('first_publication',''))[:10])} to {html.escape(str(corpus.get('last_publication',''))[:10])}).</p>
        <p>Retrospective scoring demonstration: {scored.get('rows_scored',0)} statements; positive {labels.get('positive',0)},
        negative {labels.get('negative',0)}, neutral {labels.get('neutral',0)}. Statements are truncated to the first 384 tokenizer tokens.
        These labels are sentiment outputs, not return predictions or an accuracy benchmark.</p>
        <p><strong>No FinBERT feature enters any reported trading result.</strong> The retrieved checkpoint was modified in 2026;
        original article versions, historical receipt timestamps, complete ETF relevance and source outage coverage are unverified.
        News export templates and an ingestion contract are included for a later qualified ablation. Current license: {html.escape(str(m.get('license','unknown')))}.</p>"""

    def sector_news_html(self):
        status_path=self.root/"reports/sector_news_status.json";csv_path=self.root/"data/news/sector_etf_news_scored.csv"
        if not status_path.exists() or not csv_path.exists():
            return "<p>ETF/sector news ingestion has not completed; do not treat the FOMC-only archive as broad ETF news coverage.</p>"
        status=json.loads(status_path.read_text());d=pd.read_csv(csv_path)
        parts=[f"<p><strong>Broader ETF and sector news has been retrieved and scored locally.</strong> {status.get('unique_articles',d.article_id.nunique())} unique public articles; {status.get('scored_articles',d.article_id.nunique())} scored articles; {len(d)} article–ETF associations. One canonical article can map to more than one ETF. Feed inclusion alone does not establish relevance.</p>"]
        rows=[]
        for ticker in self.tickers:
            x=d.loc[d.ticker.eq(ticker)];q=x.loc[x.relevance_label.isin(["direct_etf_mention","sector_keyword_context"])]
            rows.append({"ETF":ticker,"Direct ETF mentions":int(x.relevance_label.eq("direct_etf_mention").sum()),
                         "Sector keyword context":int(x.relevance_label.eq("sector_keyword_context").sum()),
                         "Feed association only":int(x.relevance_label.eq("feed_association_only").sum()),
                         "Qualified associations":len(q),"Mean sentiment balance":q.sentiment_balance.mean() if len(q) else np.nan})
        parts.append(self.table(pd.DataFrame(rows),integers=["Direct ETF mentions","Sector keyword context","Feed association only","Qualified associations"],decimals=["Mean sentiment balance"]))
        parts.append(f"<p>Returned publication range: {html.escape(str(status.get('oldest_publication','unknown')))} to {html.escape(str(status.get('newest_publication','unknown')))}. Direct-mention and sector-context associations contribute to the descriptive sentiment summary. Feed-only associations are retained for audit and excluded from that summary.</p>")
        parts.append("<p><strong>This is a current public-feed feasibility sample, not a historical backtest corpus.</strong> Availability is no earlier than actual current receipt and scoring. Coverage is shallow and source-dependent; original article versions, outage histories and complete licensed historical ETF coverage are unavailable. Sector keyword relevance is a transparent rule, not a validated entity map. All rows remain excluded from the reported trading models.</p>")
        examples=d.loc[d.relevance_label.isin(["direct_etf_mention","sector_keyword_context"])].drop_duplicates("article_id").head(6)
        if len(examples):
            parts.append("<details><summary>Inspect six real news examples and relevance evidence</summary><table><tr><th>ETF</th><th>Article</th><th>Relevance</th><th>Published</th></tr>")
            for _,r in examples.iterrows():
                parts.append(f"<tr><td>{html.escape(str(r.ticker))}</td><td><a href='{html.escape(str(r.source_url),quote=True)}'>{html.escape(str(r.headline))}</a></td><td>{html.escape(str(r.relevance_label))}<br>{html.escape(str(r.matched_tokens))}</td><td>{html.escape(str(r.published_at))}</td></tr>")
            parts.append("</table></details>")
        parts.append("<p>Auditable files: <a href='data/news/sector_etf_news_scored.csv'>article–ETF scored associations</a> and <a href='reports/sector_news_status.json'>coverage and source status</a>.</p>")
        return "".join(parts)

    def sector_news_figure(self):
        d=pd.read_csv(self.root/"data/news/sector_etf_news_scored.csv")
        q=d.loc[d.relevance_label.isin(["direct_etf_mention","sector_keyword_context"])].copy()
        means=q.groupby("ticker").sentiment_balance.mean().reindex(self.tickers)
        counts=q.groupby("ticker").size().reindex(self.tickers,fill_value=0)
        fig,ax=plt.subplots(figsize=(13,6.3));y=np.arange(len(self.tickers))
        ax.barh(y,means.fillna(0),color=[COLORS["good"] if pd.notna(v) and v>=0 else COLORS["bad"] for v in means])
        ax.set(yticks=y,yticklabels=[f"{t} · n={counts[t]}" for t in self.tickers],xlim=(-1,1),
               xlabel="Mean FinBERT positive probability − negative probability",title="Current ETF/sector news · descriptive sentiment, excluded from backtests")
        ax.invert_yaxis();ax.axvline(0,color="#748195",lw=.8);ax.grid(axis="x");ax.set_axisbelow(True)
        fig.text(.125,.01,"Direct mentions + sector-keyword context only. Overlapping article mappings, shallow public-feed coverage and current receipt prevent historical trading inference.",fontsize=9,color=COLORS["muted"])
        fig.tight_layout(rect=(0,.04,1,1));return self.save(fig,"10_sector_news_sentiment")

    def benchmark_html(self):
        path=self.root/"reports/benchmark_diagnostics.csv"
        if not path.exists():return "<p>Supplemental benchmark diagnostics are not available in this report snapshot.</p>"
        d=pd.read_csv(path)
        wanted=[c for c in ["ticker","strategy","benchmark","sharpe","primary_sharpe","delta_sharpe_primary_minus_benchmark","average_exposure","volatility","max_drawdown"] if c in d]
        d=d[wanted].copy()
        if "benchmark" in d:d["benchmark"]=d["benchmark"].replace({"vol_always":"Always-long vol target","exposure_rebalanced":"Ex-post weekly exposure"})
        labels={"ticker":"ETF","strategy":"Reference","benchmark":"Reference","sharpe":"Reference Sharpe","primary_sharpe":"CART Sharpe", "delta_sharpe_primary_minus_benchmark":"CART − reference Sharpe","average_exposure":"Mean exposure","volatility":"Annual volatility","max_drawdown":"Worst drawdown"}
        d=d.rename(columns=labels)
        return self.table(d,percentages=["Mean exposure","Annual volatility","Worst drawdown"],decimals=["Reference Sharpe","CART Sharpe","CART − reference Sharpe"])

    def audit_html(self):
        path=self.root/"reports/result_audit.json"
        if not path.exists():return "<p>Independent audit has not completed.</p>"
        d=json.loads(path.read_text())
        return f"<p><strong>Independent saved-output audit: {html.escape(str(d.get('status','unknown')))}.</strong> {d.get('audited_instruments',0)}/{d.get('expected_instruments',15)} instruments; {d.get('checks_passed',0):,}/{d.get('checks_total',0):,} checks passed. Reserved dates scored: {html.escape(str(d.get('reserved_scored','unknown')))}. Checks cover chronology, label maturity, weekly timing, account consistency, hashes and metrics; they cannot prove point-in-time provider corrections or future profitability.</p>"

    def improvements_html(self):
        delta=self.primary.sharpe-self.buy.sharpe;dd=self.primary.max_drawdown-self.buy.max_drawdown
        drag=self.rows("CART","selected",0).set_index("ticker").cagr-self.primary.cagr
        turnover=self.primary.turnover_per_year
        weak=delta.nsmallest(min(3,len(delta)))
        cross=int(((self.sharpe_intervals.lower<=0)&(self.sharpe_intervals.upper>=0)).sum())
        return f"""<ol class='priorities'><li><strong>Test whether timing adds value after reducing exposure.</strong>
        CART trails ETF buy/hold net Sharpe in {int((delta<0).sum())}/{len(delta)} studies. The largest Sharpe gaps are
        {', '.join(f'{t} ({v:+.2f})' for t,v in weak.items())}.
        Compare frozen timing against a prespecified static weight and a volatility-matched control; the supplied ex-post initial exposure reference is descriptive and its weight drifts.</li>
        <li><strong>Reduce unnecessary turnover before adding model complexity.</strong> The median 0-to-5-bps CAGR reduction across independent studies is {drag.median()*100:.2f} percentage points/year;
        {turnover.idxmax()} has the most turnover ({turnover.max():.1f}× NAV/year). A new version can compare a prior-selected no-trade buffer or slower rebalance rule on the same inner chronology.</li>
        <li><strong>Check year and regime dependence.</strong> {cross}/{len(delta)} Sharpe-difference bootstrap intervals cross zero, while {int((dd<0).sum())} studies have worse maximum drawdown than buy/hold.
        Inspect the annual heatmap and crisis periods, then register any recency or refit-frequency challenger before scoring its next interval.</li>
        <li><strong>Calibrate probabilities and economic payoff separately.</strong> Current classifiers use a fixed 0.5 signal threshold and raw probabilities.
        Add strictly prior out-of-fold calibration, sample-size shrinkage and gain/loss estimates before testing payoff-based sizing. Regression years require their own error and payoff diagnostics.</li>
        <li><strong>Qualify macro and text before enabling them.</strong> Obtain release/vintage-aware macro data and versioned timestamped news.
        The lagged FRED archive and FinBERT feasibility work are available, but the reported baseline cannot establish their incremental trading value.</li>
        <li><strong>Freeze the next recipe and preserve validation.</strong> Lower CAGR alone is not a failure under the declared Sharpe objective; judge net risk-adjusted return with drawdown, exposure and uncertainty.
        Treat RF/XGBoost and all sizing/cost views as challenger diagnostics.
        Do not promote an outer-period winner from these results. Preserve the 2024+ historical reserve, then declare a new prospective evaluation after choosing the next version.</li></ol>"""

    def provenance_html(self):
        config_bytes=(self.runs/"run_config.json").read_bytes();source_bytes=(self.runs/"research_source.py").read_bytes()
        raw=len(self.data_manifest["sources"])
        return f"""<p>Run version: <code>{html.escape(self.config['run_version'])}</code>. Saved metric rows: {len(self.metrics)}.
        Raw source archives: {raw}; price QA passed for all 17 downloaded instruments.
        <br>Frozen config SHA-256: <code>{hashlib.sha256(config_bytes).hexdigest()}</code>
        <br>Frozen research source SHA-256: <code>{hashlib.sha256(source_bytes).hexdigest()}</code></p>
        <p>Reproducible evidence lives in <code>runs/v2/&lt;ETF&gt;/</code>: candidate/rejection ledger, annual selections,
        dated forecasts with model hashes, fitted models, tree rules, positions/cash/NAV and costs, annual results, reliability bins and bootstrap intervals.
        Source responses and data SHA-256 hashes live under <code>data/</code>. This notebook only reads those files and creates figures.</p>
        <p>The declared next-step file is <a href='reports/validation_plan.json'>reports/validation_plan.json</a>; primary evaluation rules are in
        <a href='reports/evaluation_criteria.json'>reports/evaluation_criteria.json</a>. See <a href='reports/result_audit.json'>reports/result_audit.json</a> for independent audit results, when complete.</p>"""


def emit(path):
    """Display a saved PNG in a notebook without requiring an interactive backend."""
    from IPython.display import Image, display
    display(Image(filename=str(path)))


def notebook_cells():
    import nbformat as nbf
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s):cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))
    md("""# Swing Tree · local ETF research
    **15 independent long/cash studies · executed research notebook**

    Read the outcomes, then inspect how the decisions were formed. All performance is historical research after model and recipe selection on earlier inner windows. Each account starts with NAV 1; these accounts are never pooled into a portfolio.

    This notebook loads saved results and produces charts. **Running every cell never retrains a model, selects a recipe, downloads data or scores the 2024+ reserve.** Use a Python environment containing the pinned project requirements; start Jupyter in this project folder.
    """)
    code("""
    from pathlib import Path
    import sys
    ROOT = Path.cwd()
    if not (ROOT / 'src/build_deliverables.py').exists():
        raise RuntimeError('Open this notebook with the swing-tree-local folder as the working directory.')
    sys.path.insert(0, str(ROOT / 'src'))
    from build_deliverables import ResearchReport, emit
    from IPython.display import HTML, display
    R = ResearchReport(ROOT)
    display(HTML(R.overview_html()))
    """)
    md("""## 1 · Measured outcomes
    **Primary outperformance criterion: annualized net daily Sharpe above the ETF's own buy/hold Sharpe on identical dates.** The primary result is CART with the recipe, horizon, target and fixed/volatility sizing selected using only earlier inner years, charged **5 bps one-way** on actual turnover. Buy/hold and SPY context use the same ETF-specific start and end dates and transaction-cost rate. Cash and the risk-free convention are zero.

    CAGR compounds the account's actual daily path. Drawdown includes the initial NAV 1. Sharpe uses daily returns and a zero cash rate. Dates vary because each ETF must satisfy the same minimum history requirements.
    """)
    code("display(HTML(R.summary_table()))\nemit(R.comparison_figure())\ndisplay(HTML(R.return_table()))")
    md("""## 2 · Workflow and research boundaries
    The handoff's mapped workflow is followed in stages. Price-based technical and relationship factors pass the first data gate. Macro and FinBERT remain behind their availability gates. Later payoff sizing, frontier sizing, recency, calibration updates and drift-triggered policies are registered future stages, not completed experiments.

    Each annual outer year is preceded by two inner validation years. Only labels matured strictly before each fit cutoff can enter training; a common 20-session gate purges overlaps across candidate horizons. At least 104 matured weekly observations are required. CART screens 5/10/20-session horizons, classification/regression targets, and two eligible factor recipes with fixed/vol sizing. The other two macro recipes remain in the rejection ledger.

    A candidate label is the next-open to horizon-end open return minus a 10-bps round-trip hurdle. The actual account separately charges costs on realized turnover; the label hurdle is not deducted from the account again.

    RF and XGBoost use the CART-selected specification as separately reported challengers. CART remains the primary control throughout. No learner is promoted based on these outer results.
    """)
    code("emit(R.workflow_figure())")
    md("""## 3 · Data depth, quality and missingness
    Public Yahoo daily provider OHLC, adjusted close, volume and action histories are retained with immutable raw responses and SHA-256 hashes. Adjusted OHLC equals provider OHLC multiplied by the same-date adjusted-close/close ratio. These fractional adjusted units approximate dividend-reinvested total return; dividends are not credited a second time.

    Coverage goes to the completed **2026-10-19** session. The strategy pipeline removes all **2024-01-01 onward** prices before building features, labels, fitting or scoring. Downloading and quality-auditing reserved prices does not make that historical reserve a guaranteed unseen sample; prospective validation is still required.
    """)
    code("display(HTML(R.coverage_table()))")
    md("""### Macro archive · excluded from fitted strategies
    Six daily FRED series are aligned using a conservative two-business-day observation lag; they are latest-vintage data and remain diagnostic only. A lag cannot recreate original release timestamps or revisions. CPI, unemployment and industrial production are revision-sensitive monthly archives and are excluded even from the derived daily feature file. HY OAS is structurally limited to three years by FRED's source terms since April 2026. Missing early history is not backfilled.
    """)
    code("display(HTML(R.macro_table()))\nemit(R.macro_figure())")
    md("""## 4 · Every ETF's equity and drawdown
    Each chart is a separate normalized account. Selected CART, the ETF's own buy/hold and SPY context share identical dates and 5 bps one-way costs. Equity axes are logarithmic; lower panels show percentage drawdown from the running peak. Weekly predictions do not impose an automatic horizon exit: exposure persists until the next weekly sizing decision.
    """)
    for ticker in ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","IEF","TLT","TIP","GLD","DBC","VNQ"]:
        md(f"### {ticker}")
        code(f"emit(R.instrument_figure('{ticker}'))")
    md("""## 5 · Model ladder, sizing and costs
    These views reuse saved forecasts. They do not launch a new search. RF/XGBoost are fixed-family challengers fitted to the CART-selected annual recipe. Their outer results cannot be used to promote a family without a new registered evaluation.
    """)
    code("display(HTML(R.model_table()))")
    md("""### Fixed versus volatility targeting
    Fixed sizing invests 100% on a LONG signal. Volatility targeting uses `min(1, 10% / max(5%, trailing 63-session annualized volatility))`; CASH has weight zero. The inner-selected rule freezes fixed or vol sizing before each outer year. All accounts are long/cash with nonnegative cash and no leverage.
    """)
    code("display(HTML(R.sizing_table()))")
    md("""### Cost stress and actual trading behavior
    Zero cost is a diagnostic, 5 bps is the selection/base assumption, and 10 bps is stress. Costs are charged on actual turnover at next-open trades and comparable terminal liquidation. Annual turnover is traded notional divided by current NAV per year. Fees below are summed cash fees in the account's NAV-1 currency, not basis points. Trade events include final liquidation. Holding sessions describe continuous periods with positive exposure.

    The saved `matched_exposure` account is an **ex-post initial exposure reference**: it invests once at the CART sample-average exposure and lets shares drift. It is not a continuously exposure-matched or risk-matched benchmark and is not used for selection.
    """)
    code("display(HTML(R.cost_table()))\ndisplay(HTML(R.execution_table()))")
    md("""### Supplemental exposure and volatility references
    These additional saved references help distinguish timing behavior from exposure reduction. The **weekly exposure reference** rebalances to a constant target set from CART's full-sample mean exposure; it uses future sample information and is ex-post, unlike a deployable strategy. It is separate from the one-time initial-allocation reference discussed above. The **always-long capped volatility target** is a causal risk-control reference using earlier volatility; it does not predict direction. These diagnostics do not replace the declared ETF buy/hold Sharpe objective.
    """)
    code("display(HTML(R.benchmark_html()))")
    md("""## 6 · What was selected and which factors were used
    These charts show annual inner selections and fitted CART impurity usage. Factor importance describes usage in fitted trees; correlated inputs and tree biases prevent interpreting it as causal contribution or validated alpha. Technical factors cover momentum, volatility, downside risk, moving-average trend, drawdown, range, gaps, reversal, volume, shocks and fast/slow divergence. Relationship candidates use SPY, IEF, GLD and DBC returns plus market beta, correlation and residual behavior.

    Where a peer had not listed, relationship momentum is explicitly zero with a separate availability flag. Qualified history and missingness remain visible in the saved events and selection files. UUP is downloaded as an available relationship source but is not used by this baseline.
    """)
    code("display(HTML(R.feature_inventory_html()))\nemit(R.selection_figure())\nemit(R.factor_figure())\nemit(R.factor_example())")
    md("""## 7 · Stability and uncertainty
    Annual returns reveal whether results depend on a small number of years. Grey cells indicate years that lacked sufficient earlier qualified history. For the primary objective, **paired moving-block bootstrap Sharpe-difference intervals** resample the same 20-session blocks for the strategy and ETF benchmark (500 replicates). The canonical audit file records seeds and the full procedure. A separate chart preserves the saved annualized mean-return difference intervals. Neither is CAGR uncertainty. Both are descriptive after selection and do not correct the full research process, multiple comparisons, current-cohort hindsight or all dependence from overlapping outcomes.
    """)
    code("emit(R.annual_figure())\nemit(R.sharpe_uncertainty_figure())\nemit(R.uncertainty_figure())")
    md("""### Raw probability reliability
    Only selected classification years contribute. Points show observed positive *net horizon* outcomes in probability bins; marker size reflects sample count. Selected horizons change across fits, so each panel mixes distinct prediction horizons. Probabilities are uncalibrated, outcomes overlap, and these descriptive bins are not a fitted calibration layer. Regression years do not appear. Lower Brier loss is better within the same comparable prediction task.
    """)
    code("emit(R.calibration_figure())")
    md("""## 8 · FinBERT and the real news feasibility demonstration
    Sentiment infrastructure and a real historical text archive are available locally. Their trading eligibility is evaluated separately from software feasibility.
    """)
    code("display(HTML(R.finbert_html()))")
    md("""### ETF and sector news · broader public coverage
    Current feeds add ETF-specific and sector-specific text beyond Federal Reserve statements. Relevance evidence and receipt times remain visible; current sentiment is reported separately from historical strategy performance.
    """)
    code("display(HTML(R.sector_news_html()))\nemit(R.sector_news_figure())")
    md("""## 9 · Measurable improvement priorities
    These are hypotheses suggested by the observed diagnostics. Test them under a new version using earlier windows and a bounded trial ledger; no improvement is guaranteed.
    """)
    code("display(HTML(R.improvements_html()))")
    md("""## 10 · Reproducibility and practical limits
    This is an initial retrospective engineering study. The 15 currently chosen ETFs are not a survivorship-neutral historical universe. Public provider corrections, adjustment conventions and historical tradability need independent validation before official results. No live order execution, options, shared portfolio allocation or account access is present.

    Every scored account uses fractional adjusted units, zero cash yield, simplified constant trading costs and no financing. Fixed spreads/slippage do not capture every execution condition. Outer outcomes are historical evaluation of the declared selection procedure, not an untouched prospective guarantee. The 2024+ historical reserve remains unscored in this version.
    """)
    code("display(HTML(R.audit_html()))\ndisplay(HTML(R.provenance_html()))")
    return cells


def simple_markdown(source):
    """Small, safe renderer for the deliberately simple notebook prose."""
    source=html.escape(source)
    source=re.sub(r"\*\*(.*?)\*\*",r"<strong>\1</strong>",source)
    source=re.sub(r"`([^`]+)`",r"<code>\1</code>",source)
    blocks=[]
    for part in re.split(r"\n\s*\n",source):
        lines=part.strip().splitlines()
        if not lines:continue
        if lines[0].startswith("#"):
            n=min(4,len(lines[0])-len(lines[0].lstrip("#")))
            blocks.append(f"<h{n}>{lines[0].lstrip('# ').strip()}</h{n}>")
            if len(lines)>1:blocks.append("<p>"+" ".join(lines[1:])+"</p>")
        else:blocks.append("<p>"+" ".join(lines)+"</p>")
    return "\n".join(blocks)


CSS="""
:root{--navy:#1c2b40;--teal:#087e82;--muted:#637186;--line:#e1e7ed}
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:var(--navy);font:16px/1.65 'Segoe UI',Arial,sans-serif}
header{background:#142c40;color:white;padding:42px max(30px,calc((100vw - 1240px)/2));border-bottom:6px solid #3bb1af}
header p{color:#bed0de;max-width:900px;margin-bottom:0}.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#93d4d1}
header h1{font-size:38px;font-weight:650;margin:8px 0}.wrap{max-width:1320px;margin:auto;padding:26px 34px 70px}
nav{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 28px}nav a{color:var(--teal);background:white;border:1px solid var(--line);padding:7px 14px;border-radius:20px;text-decoration:none;font-size:13px}
section{background:white;padding:26px 34px;margin:20px 0;border:1px solid var(--line);border-radius:12px;overflow:hidden}
h1,h2,h3{line-height:1.25}h2{font-size:27px;margin:0 0 22px}h3{font-size:20px;margin:30px 0 14px}p{max-width:1120px;margin:14px 0}strong{font-weight:650}
.output{overflow-x:auto;margin:18px 0}img.figure{display:block;max-width:100%;height:auto;margin:12px auto}pre{white-space:pre-wrap;font:12px/1.5 Consolas,monospace}
code{font:12px Consolas,monospace;overflow-wrap:anywhere;background:#f1f4f7;padding:2px 4px;border-radius:3px}
table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th{text-align:right;background:#edf4f5;border-bottom:2px solid #bad5d8;color:#25424e;padding:11px 12px}td{text-align:right;border-bottom:1px solid #e9edf1;padding:10px 12px}th:first-child,td:first-child{text-align:left}tr:nth-child(even)td{background:#f8fafb}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:10px 0 22px}.cards>div{border-left:4px solid var(--teal);background:#f0f7f7;padding:18px 14px}.cards strong{font-size:32px;display:block;line-height:1.2}.cards span{font-size:12px;color:var(--muted);display:block;margin-top:8px}
.priorities{padding-left:22px}.priorities li{padding:7px 0 12px;max-width:1110px}footer{font-size:12px;color:var(--muted);padding-top:20px}
details{border:1px solid var(--line);border-radius:8px;margin:14px 0;padding:12px 16px}summary{cursor:pointer;font-weight:650;color:var(--teal)}.registry-table table{white-space:normal}.registry-table td{vertical-align:top;text-align:left;min-width:130px}.registry-table td:nth-child(2){min-width:340px}
@media(max-width:750px){.wrap{padding:14px}section{padding:20px 16px}header{padding:26px 20px}header h1{font-size:28px}.cards{grid-template-columns:repeat(2,1fr)}h2{font-size:24px}}
@media print{body{background:white}header{background:white;color:#142c40;padding:15px}header p{color:#637186}nav{display:none}.wrap{padding:0}section{border:0;break-inside:avoid}h2{break-before:page}.output{overflow:visible}.cards{grid-template-columns:repeat(4,1fr)}}
"""


def html_from_notebook(nb,target):
    parts=[];section_open=False;nav=[]
    for cell in nb.cells:
        if cell.cell_type=="markdown":
            source=cell.source
            if source.startswith("# "):
                # The branded report header carries the notebook title.
                source="\n".join(source.splitlines()[1:]).strip()
                if not section_open:parts.append("<section id='overview'>");section_open=True
            elif source.startswith("## "):
                if section_open:parts.append("</section>")
                title=source.splitlines()[0][3:];slug="section-"+str(len(nav)+1)
                parts.append(f"<section id='{slug}'>");section_open=True;nav.append((slug,title))
            parts.append(simple_markdown(source))
        else:
            for output in cell.get("outputs",[]):
                if output.output_type=="error":raise RuntimeError("Executed notebook contains an error")
                data=output.get("data",{})
                if "image/png" in data:
                    parts.append(f"<div class='output'><img class='figure' alt='Research chart generated from saved results' src='data:image/png;base64,{data['image/png']}'></div>")
                elif "text/html" in data:parts.append("<div class='output'>"+data["text/html"]+"</div>")
                elif output.output_type=="stream" and output.get("name")=="stderr":
                    # No silent hiding of execution warnings.
                    parts.append("<pre class='execution-note'>"+html.escape(output.get("text",""))+"</pre>")
                elif "text/plain" in data and not data["text/plain"].startswith("<IPython.core.display"):
                    parts.append("<pre>"+html.escape(data["text/plain"])+"</pre>")
    if section_open:parts.append("</section>")
    links="".join(f"<a href='#{slug}'>{html.escape(title)}</a>" for slug,title in nav)
    stamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page=f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>Swing Tree — Local ETF Research</title><style>{CSS}</style></head><body>
    <header><div class='eyebrow'>University research · local run v2 · historical evaluation to 2023</div><h1>Swing Tree · measured ETF outcomes</h1>
    <p>Fifteen separate long/cash studies, chronological selection and an auditable path from source data to forecasts and accounts. All charts and tables are embedded for offline review.</p></header>
    <main class='wrap'><nav>{links}</nav>{''.join(parts)}<footer>Built from an executed read-only notebook · {stamp}. This local HTML makes no external network requests.</footer></main></body></html>"""
    page=page.replace("href='reports/","href='").replace('href="reports/','href="')
    for folder in ["data/","runs/","src/"]:
        page=page.replace("href='"+folder,"href='../"+folder).replace('href="'+folder,'href="../'+folder)
    target.write_text(page,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--draft-only",action="store_true",help="Write the notebook without executing; does not create a final report")
    args=parser.parse_args()
    import nbformat as nbf
    nb=nbf.v4.new_notebook(cells=notebook_cells())
    nb.metadata.update({"kernelspec":{"display_name":"Python 3 (Swing Tree research)","language":"python","name":"python3"},
                        "language_info":{"name":"python","version":sys.version.split()[0]},
                        "swing_tree":{"purpose":"read saved artifacts only; no model retraining","run_version":"v2"}})
    target=ROOT/"research_notebook.ipynb"
    nbf.write(nb,target)
    if args.draft_only:
        print(f"Notebook draft saved: {target}");return
    print("Validating all 15 completed v2 instruments and canonical audit...",flush=True)
    ResearchReport(ROOT,require_complete=True)
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    # Resolve this interpreter explicitly using a task-local kernelspec only.
    # This does not install or modify the user's global Jupyter configuration.
    kernel_base=SCRATCH/"kernels";kernel_dir=kernel_base/"python3";kernel_dir.mkdir(parents=True,exist_ok=True)
    (kernel_dir/"kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
        "display_name":"Python 3 (Swing Tree isolated environment)","language":"python"}),encoding="utf-8")
    km=KernelManager(kernel_name="python3",kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_base)]))
    def progress(**kwargs):
        print(f"Executed notebook cell {kwargs.get('cell_index',0)+1}",flush=True)
    print("Starting isolated notebook kernel and executing saved-result views...",flush=True)
    client=NotebookClient(nb,km=km,timeout=180,startup_timeout=30,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}},allow_errors=False,on_cell_executed=progress)
    try:
        client.execute(cwd=str(ROOT))
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    code_cells=[c for c in nb.cells if c.cell_type=="code"]
    if not all(c.execution_count is not None for c in code_cells):raise RuntimeError("Notebook has unexecuted code cells")
    nb.metadata["swing_tree"]["executed_utc"]=datetime.now(timezone.utc).isoformat()
    nbf.write(nb,target)
    html_from_notebook(nb,ROOT/"reports/research_report.html")
    print(f"Executed {len(code_cells)} code cells with no errors; notebook and self-contained HTML saved.")


if __name__=="__main__":main()
