"""Readable plotting/reporting helpers for the focused saved-result notebook."""
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
from trade_statistics import summary_statistics, completed_episodes, annual_model_statistics

LABELS = {"payoff_context": "Payoff + context", "etf_hold": "ETF buy & hold", "spy_hold": "S&P 500 (SPY) buy & hold"}
NAMES = {"XLK":"Technology","XLF":"Financials","XLE":"Energy","XLV":"Health care",
         "XLI":"Industrials","XLY":"Consumer discretionary","XLP":"Consumer staples",
         "XLU":"Utilities","XLB":"Materials","IEF":"7–10 year Treasuries",
         "TLT":"20+ year Treasuries","TIP":"Inflation-protected Treasuries",
         "GLD":"Gold","DBC":"Broad commodities","VNQ":"U.S. real estate"}


def performance(d):
    r = d.daily_return
    return {"Net Sharpe": r.mean()/r.std(ddof=1)*np.sqrt(252) if r.std(ddof=1)>0 else 0,
            "CAGR": d.nav.iloc[-1]**(252/len(d))-1,
            "Max drawdown": (d.nav/d.nav.cummax().clip(lower=1)-1).min(),
            "Volatility": r.std(ddof=1)*np.sqrt(252),
            "Total return": d.nav.iloc[-1]-1,
            "Daily exposure": d.held_exposure.mean(),
            "Turnover/year": d.turnover.sum()/(len(d)/252)}


def tab(d, percent=()):
    fmt = {c: (lambda x: f"{x:.1%}") for c in percent if c in d}
    return d.to_html(index=False, border=0, escape=True, formatters=fmt,
                     float_format=lambda x: f"{x:,.3f}", na_rep="—")


class FocusedReport:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        self.ledgers = pd.read_csv(self.root/"data/ledgers.csv", parse_dates=["Date"])
        self.forecasts = pd.read_csv(self.root/"data/forecasts.csv", parse_dates=["decision_date","entry_date","fit_cutoff","train_latest_label","calibration_cutoff","calibration_latest_label","label_end"])
        self.usage = pd.read_csv(self.root/"data/feature_usage.csv")
        self.factors = pd.read_csv(self.root/"data/factor_dictionary.csv").fillna("")
        self.costs = pd.read_csv(self.root/"data/cost_sensitivity.csv")
        calendar_path=self.root/"data/session_calendar.csv"
        self.session_calendar=pd.DatetimeIndex(pd.read_csv(calendar_path,parse_dates=["Date"]).Date) if calendar_path.exists() else None
        assert set(self.ledgers.strategy) == set(LABELS)
        assert set(self.forecasts.variant) == {"payoff_context"}
        assert self.ledgers.Date.max() < pd.Timestamp("2024-01-01")
        assert self.forecasts.decision_date.max() < pd.Timestamp("2024-01-01")
        (self.root/"figures").mkdir(exist_ok=True)

    def ledger(self, ticker, strategy="payoff_context"):
        d=self.ledgers.loc[self.ledgers.ticker.eq(ticker)&self.ledgers.strategy.eq(strategy)].set_index("Date").sort_index()
        if self.session_calendar is not None:d.attrs["session_calendar"]=self.session_calendar
        return d

    def workflow(self):
        return """<p><strong>Research workflow:</strong> completed weekly close → next-session open →
annual expanding CART fit using earlier matured labels → ETF-context features →
prior out-of-fold predictions mapped to shrunk expected net payoff → LONG when
expected payoff is positive, otherwise cash → inherited fixed or capped10%volatility sizing →
actual-turnover fees and final liquidation.</p>
<p>CART depth3, minimum leaf fraction0.08, seed42. Earlier chronological folds chose
classification/regression,5/10/20-session horizons, technical/relationship recipes and sizing.
The common training maturity gate is20sessions. Payoff mapping uses up to5prior OOFyears,
5bins,26pseudo-observations and a minimum52mature OOFrows; insufficient history falls back
to the original forecast signal. Weight limits are 0–100%, with a 5% volatility floor.</p>""".replace("10%volatility","10% volatility").replace("depth3","depth 3").replace("fraction0.08","fraction 0.08").replace("seed42","seed 42").replace("classification/regression,5/10/20-session","classification/regression, 5/10/20-session").replace("is20sessions","is 20 sessions").replace("up to5prior OOFyears","up to 5 prior OOF years").replace("5bins,26pseudo-observations","5 bins, 26 pseudo-observations").replace("minimum52mature OOFrows","minimum 52 mature OOF rows")

    def plot(self, ticker):
        colors = {"payoff_context":"#087f82","etf_hold":"#93649a","spy_hold":"#758399"}
        fig, axes = plt.subplots(2,1,figsize=(12,6.4),sharex=True,gridspec_kw={"height_ratios":[2.3,1]})
        for strategy in LABELS:
            d = self.ledger(ticker,strategy)
            # Add a visual NAV1 starting anchor only. Saved daily returns stay intact.
            anchor = d.index[0]-pd.Timedelta(seconds=1)
            dates = pd.DatetimeIndex([anchor]).append(d.index)
            nav = pd.Series(np.r_[1.0,d.nav.to_numpy()], index=dates)
            style = "--" if strategy=="spy_hold" else "-"
            axes[0].plot(nav.index,nav,style,color=colors[strategy],lw=1.65,label=LABELS[strategy])
            axes[1].plot(nav.index,nav/nav.cummax()-1,style,color=colors[strategy],lw=1.2)
        axes[0].set_title(f"{ticker} · {NAMES[ticker]}",loc="left",fontsize=15,fontweight="bold")
        axes[0].set_ylabel("Equity · log scale (NAV 1 start)")
        axes[0].set_yscale("log")
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axes[0].legend(frameon=False,ncol=3,loc="upper left",fontsize=9)
        axes[1].set_ylabel("Drawdown")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        d = self.ledger(ticker)
        axes[1].set_xlim(d.index[0]-pd.Timedelta(seconds=1),d.index[-1])
        for ax in axes:
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(alpha=.15)
        fig.tight_layout()
        fig.savefig(self.root/"figures"/f"{ticker}.png",dpi=140,bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def details(self,ticker):
        d = self.ledger(ticker)
        f = self.forecasts.loc[self.forecasts.ticker.eq(ticker)]
        usage = self.usage.loc[self.usage.ticker.eq(ticker)]
        stats = summary_statistics(d,f)
        rows = [{"Account": LABELS[s], **performance(self.ledger(ticker,s))} for s in LABELS]
        perf = tab(pd.DataFrame(rows), ["CAGR","Max drawdown","Volatility","Total return","Daily exposure"])
        # Preserve every statistic and its exact denominator in an expandable table.
        statistic_rows = []
        for key,value in stats.items():
            if isinstance(value,(dict,list)): value=json.dumps(value,default=str)
            if value is None: value="—"
            statistic_rows.append({"Measure":key.replace("_"," "), "Value":value})
        annual = annual_model_statistics(f,d,usage)
        brief_annual = annual[["model_year","target_types","horizons_sessions","base_factor_recipes",
                               "sizing_rules","decision_weeks","training_rows_min",
                               "forecast_feature_count_min","active_tree_feature_count"]].rename(
            columns={"model_year":"Year","target_types":"Target","horizons_sessions":"Horizon (sessions)",
                     "base_factor_recipes":"Base factors","sizing_rules":"Sizing","decision_weeks":"Decisions",
                     "training_rows_min":"Training rows","forecast_feature_count_min":"Inputs",
                     "active_tree_feature_count":"Inputs used in splits"})
        factors = self.factors.loc[self.factors.ticker.eq(ticker)]
        ranking = usage.groupby("feature",as_index=False).agg(mean_split_importance=("importance","mean"),years_in_model=("year","nunique"))
        ranking = ranking.sort_values(["mean_split_importance","feature"],ascending=[False,True])
        definitions = factors[["feature_name","family","formula","inputs","unit","available_at_rule","missingness_policy","interpretation"]].copy()
        definitions = definitions.rename(columns={"feature_name":"Feature","family":"Family","formula":"Definition","inputs":"Inputs","unit":"Unit","available_at_rule":"Availability","missingness_policy":"Missing values","interpretation":"Limits"})
        cost = self.costs.loc[self.costs.ticker.eq(ticker),["cost_bps","sharpe","cagr","max_drawdown","average_exposure","turnover_per_year"]].copy()
        episodes = completed_episodes(d)
        rules=[]
        for path in sorted((self.root/"data/model_rules"/ticker).glob("*.txt")):
            rules.append(f"<details><summary>{html.escape(path.stem.replace('_context',''))} CART rules</summary><pre>{html.escape(path.read_text(encoding='utf-8'))}</pre></details>")
        recent=f.sort_values("decision_date").tail(12)[["decision_date","entry_date","target","horizon","prediction","expected_payoff","payoff_bin_n","oof_rows","feature_count","signal","target_weight"]]
        def pct(key):
            value=stats.get(key)
            return "—" if value is None else f"{value:.1%}"
        def number(key):
            value=stats.get(key)
            return "—" if value is None else f"{value:,.2f}"
        brief = [
            ("Completed long episodes", str(stats["completed_episodes"])),
            ("Net wins / losses / flat", f"{stats['net_wins']} / {stats['net_losses']} / {stats['net_flat']}"),
            ("Win rate (completed episodes)", pct("win_rate")),
            ("Average winning / losing episode return", pct("mean_win_account_return")+" / "+pct("mean_loss_account_return")),
            ("Profit factor (net monetary gains / losses)", number("net_profit_factor")+" · "+stats["net_profit_factor_status"]),
            ("Average / median holding sessions", number("mean_holding_sessions")+" / "+number("median_holding_sessions")),
            ("Average forecast horizon (sessions)", number("decision_horizon_mean_sessions")),
            ("Average target position while LONG", pct("mean_target_weight_when_long")),
            ("Average target position across all decision weeks", pct("mean_target_weight_all_decisions")),
            ("Average realized daily exposure, including cash days", pct("mean_realized_held_exposure")),
            ("LONG decision rate", pct("long_signal_rate")),
            ("Calibration fallback rate", pct("calibration_fallback_rate")),
        ]
        out = f"<p><strong>Dates:</strong> {d.index[0].date()}–{d.index[-1].date()} · source v3.1 · 5bps one-way fees.</p>"+perf
        out += "<p><strong>Combined strategy: wins, holding periods and sizing</strong></p>"+tab(pd.DataFrame(brief,columns=["Measure","Value"]))
        out += "<details><summary><strong>All strategy statistics and reconciliation</strong></summary>"+tab(pd.DataFrame(statistic_rows))+"</details>"
        out += "<details><summary><strong>Annual forecasting settings</strong></summary>"+tab(brief_annual)+"</details>"
        out += "<details><summary><strong>Full annual training/calibration provenance</strong></summary>"+tab(annual)+"</details>"
        out += "<details><summary><strong>Factors offered to the model and split importance</strong></summary><p>Importance is averaged over annual fitted trees; zero means no tree split used that input. Availability flags count as inputs, not distinct economic factors. Importance is in-sample and cannot establish a predictive or causal contribution.</p>"+tab(ranking)+tab(definitions)+"</details>"
        out += "<details><summary><strong>Saved CART decision rules by year</strong></summary>"+"".join(rules)+"</details>"
        out += "<details><summary><strong>Cost sensitivity: combined strategy only</strong></summary>"+tab(cost,["cagr","max_drawdown","average_exposure"])+"</details>"
        out += "<details><summary><strong>Last 12 saved weekly forecasts, all before 2024</strong></summary>"+tab(recent)+"</details>"
        out += "<details><summary><strong>Completed net trade episodes</strong></summary>"+tab(episodes)+"</details>"
        return out

    def verification(self):
        audit=json.loads((self.root/"reference_sources/v3_audit.json").read_text())
        return f"<p>Parent saved-result audit: <strong>{audit['status']}</strong>, {audit['checks_passed']:,}/{audit['checks_total']:,} checks. Reserved period starts2024-01-01 and remains unscored. Source/config hashes and the curated-input provenance are in manifest.json. Local helper source and factor formulas accompany the notebook.</p>"
