"""Statistics-heavy, saved-result notebook views for the combined procedure."""
from pathlib import Path
import html,json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from notebook_views import FocusedReport, LABELS, performance, tab
from temporal_statistics import monthly_returns, annual_statistics, drawdown_episodes, risk_statistics
from trade_statistics import completed_episodes, summary_statistics
from forecast_statistics import forecast_summary, probability_reliability, payoff_reliability, weekly_summary

COLORS={"payoff_context":"#087f82","etf_hold":"#93649a","spy_hold":"#758399"}


def table(frame):
    return tab(frame)


def fold(title,body,opened=False):
    return f"<details {'open' if opened else ''}><summary><strong>{html.escape(title)}</strong></summary>{body}</details>"


def metrics_matrix(records):
    """Keep names/units explicit; percentages remain decimal fractions here."""
    records=[{k:json.dumps(v,default=str) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in records]
    data=pd.DataFrame(records).set_index("Account").T.reset_index(names="Measure")
    data["Measure"]=data.Measure.str.replace("_"," ",regex=False)
    return table(data)


class StatisticalReport:
    def __init__(self,root):
        self.root=Path(root);self.R=FocusedReport(root)
        self.outcomes=pd.read_csv(self.root/"data/forecast_outcomes.csv",parse_dates=["decision_date","entry_date","label_end"])
        self.intervals=pd.read_csv(self.root/"data/weekly_policy_intervals.csv",parse_dates=["decision_date","entry_date","interval_end"])
        self.paired=pd.read_csv(self.root/"data/paired_benchmark_uncertainty.csv")
        self.original_uncertainty=pd.read_csv(self.root/"data/v3_uncertainty.csv")
        self.common=pd.read_csv(self.root/"data/v3_common_period.csv")
        self.passive=pd.read_csv(self.root/"data/passive_vol_control.csv",parse_dates=["Date"])
        self.availability=pd.read_csv(self.root/"data/context_availability.csv")
        self.calibrators=json.loads((self.root/"data/calibrators.json").read_text())
        self.tickers=self.R.manifest["tickers"]
        self.cache={}
        for name in ["monthly_returns","annual_performance","risk_statistics","drawdown_episodes"]:
            path=self.root/"data"/f"{name}.csv"
            if path.exists():self.cache[name]=pd.read_csv(path)
        assert set(self.outcomes.ticker)==set(self.tickers)

    def cached(self,name,t,strategy="payoff_context"):
        data=self.cache[name]
        return data.loc[data.ticker.eq(t)&data.strategy.eq(strategy)].drop(columns=["ticker","strategy"]).copy()

    def save(self,fig,name):
        fig.tight_layout()
        fig.savefig(self.root/"figures"/f"{name}.png",dpi=110,bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def overview_tables(self):
        perf=[];trades=[];bench=[];recipes=[]
        for t in self.tickers:
            d=self.R.ledger(t);f=self.R.forecasts.loc[self.R.forecasts.ticker.eq(t)]
            s=summary_statistics(d,f)
            perf.append({"ETF":t,**performance(d)})
            trades.append({"ETF":t,"Episodes":s["completed_episodes"],"Wins":s["net_wins"],"Losses":s["net_losses"],
                           "Win rate":s["win_rate"],"Profit factor":s["net_profit_factor"],
                           "Payoff ratio":s["payoff_ratio_account_return"],"Average holding sessions":s["mean_holding_sessions"],
                           "Forecast horizon mean":s["decision_horizon_mean_sessions"],
                           "LONG target size mean":s["mean_target_weight_when_long"],"LONG decision rate":s["long_signal_rate"]})
            p=performance(d);q=performance(self.R.ledger(t,"etf_hold"));z=performance(self.R.ledger(t,"spy_hold"))
            v=self.passive.loc[self.passive.ticker.eq(t)].set_index("Date")
            bench.append({"ETF":t,"Model Sharpe":p["Net Sharpe"],"ETF hold Sharpe":q["Net Sharpe"],
                          "SPY hold Sharpe":z["Net Sharpe"],"Passive 10% vol Sharpe":performance(v)["Net Sharpe"],
                          "Delta vs ETF":p["Net Sharpe"]-q["Net Sharpe"],"Delta vs SPY":p["Net Sharpe"]-z["Net Sharpe"]})
            recipes.append({"ETF":t,"Classification decision fraction":f.target.eq("classification").mean(),
                            "Regression decision fraction":f.target.eq("regression").mean(),
                            "Fixed sizing decision fraction":f.base_sizing.eq("fixed").mean(),
                            "Vol sizing decision fraction":f.base_sizing.eq("vol").mean(),
                            "Mean offered inputs":f.feature_count.mean(),"Model years":f.outer_year.nunique()})
        return ("<h3>Net performance and risk</h3>"+tab(pd.DataFrame(perf),["CAGR","Max drawdown","Volatility","Total return","Daily exposure"])+
                "<h3>Trade and forecasting summary</h3>"+tab(pd.DataFrame(trades),["Win rate","LONG target size mean","LONG decision rate"])+
                "<h3>Benchmark Sharpe comparisons</h3><p>The passive volatility reference is supplemental tabular context, with the same 10% target and no timing signal; it is not an extra equity curve.</p>"+table(pd.DataFrame(bench))+
                fold("Recipe, target and sizing mix",tab(pd.DataFrame(recipes),["Classification decision fraction","Regression decision fraction","Fixed sizing decision fraction","Vol sizing decision fraction"])))

    def overview_figure(self):
        fig,axes=plt.subplots(1,3,figsize=(15,6))
        y=np.arange(len(self.tickers))
        for offset,strategy in zip([-.23,0,.23],LABELS):
            sharpes=[performance(self.R.ledger(t,strategy))["Net Sharpe"] for t in self.tickers]
            axes[0].barh(y+offset,sharpes,height=.22,color=COLORS[strategy],label=LABELS[strategy])
        axes[0].set_yticks(y,self.tickers);axes[0].invert_yaxis();axes[0].axvline(0,color="#bbb",lw=.8)
        axes[0].set_title("Net daily Sharpe");axes[0].legend(fontsize=7,loc="lower right")
        wins=[]
        for t in self.tickers:
            f=self.R.forecasts.loc[self.R.forecasts.ticker.eq(t)]
            wins.append(summary_statistics(self.R.ledger(t),f)["win_rate"])
        axes[1].barh(y,wins,color=COLORS["payoff_context"])
        axes[1].set_yticks(y,self.tickers);axes[1].invert_yaxis();axes[1].set_xlim(0,1)
        axes[1].xaxis.set_major_formatter(PercentFormatter(1));axes[1].set_title("Completed-episode win rate")
        left=np.zeros(len(self.tickers))
        for horizon,color in [(5,"#bed8dd"),(10,"#50969e"),(20,"#274f66")]:
            values=[self.R.forecasts.loc[self.R.forecasts.ticker.eq(t),"horizon"].eq(horizon).mean() for t in self.tickers]
            axes[2].barh(y,values,left=left,color=color,label=f"{horizon} sessions");left+=values
        axes[2].set_yticks(y,self.tickers);axes[2].invert_yaxis();axes[2].set_xlim(0,1)
        axes[2].xaxis.set_major_formatter(PercentFormatter(1));axes[2].legend(fontsize=8)
        axes[2].set_title("Forecast horizon mix by decision")
        for ax in axes:ax.spines[["top","right"]].set_visible(False)
        self.save(fig,"dashboard_statistics")

    def coverage_tables(self):
        coverage=pd.read_csv(self.root/"data/coverage_pre2024.csv")
        return table(coverage)+fold("Context-factor availability on actual decision dates",tab(self.availability,["mean_available_fraction"]))+"""
<p>Only technical, market-relationship and ETF-price-context inputs entered the model. Missing context uses a zero value and an explicit availability flag. Macro archives and FinBERT news are excluded from fitted historical performance. Data coverage is cropped before2024; trailing labels that would need reserved observations remain missing.</p>"""

    def risk_tables(self,t):
        records=[{"Account":LABELS[s],**self.cached("risk_statistics",t,s).iloc[0].to_dict()} for s in LABELS]
        return ("<p>Return and exposure quantities in this detailed matrix are decimal fractions: 0.01 means 1%. Ratios, counts and durations are separately named. Undefined ratios are blank. CVaR5 is the signed mean of the worst ceil(5%×N) daily returns; VaR5 is the empirical fifth percentile. Daily positive-return frequency is distinct from completed-trade win rate.</p>"+
                metrics_matrix(records))

    def temporal_figure(self,t):
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        own=self.R.ledger(t);m=self.cached("monthly_returns",t)
        matrix=m.pivot(index="year",columns="month",values="net_return").reindex(columns=range(1,13))
        limit=max(.025,float(np.nanmax(np.abs(matrix.to_numpy()))))
        im=axes[0,0].imshow(matrix.to_numpy(),cmap="RdBu",vmin=-limit,vmax=limit,aspect="auto")
        axes[0,0].set_yticks(range(len(matrix)),matrix.index)
        axes[0,0].set_xticks(range(12),list("JFMAMJJASOND"))
        axes[0,0].set_title("Monthly net returns · model")
        for i in range(len(matrix)):
            for j in range(12):
                value=matrix.iloc[i,j]
                if np.isfinite(value):axes[0,0].text(j,i,f"{value:.0%}",ha="center",va="center",fontsize=6,color="white" if abs(value)>.65*limit else "#253342")
        fig.colorbar(im,ax=axes[0,0],fraction=.03,format=PercentFormatter(1))
        for offset,s in zip([-.25,0,.25],LABELS):
            annual=self.cached("annual_performance",t,s)
            axes[0,1].bar(annual.year.to_numpy()+offset,annual.net_return,width=.23,color=COLORS[s],label=LABELS[s])
        axes[0,1].set_title("Calendar-year net returns");axes[0,1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0,1].tick_params(axis="x",rotation=45);axes[0,1].legend(fontsize=7)
        for s in LABELS:
            d=self.R.ledger(t,s);r=d.daily_return
            rolling=r.rolling(252).mean()/r.rolling(252).std(ddof=1)*np.sqrt(252)
            axes[1,0].plot(d.index,rolling,color=COLORS[s],label=LABELS[s],lw=1)
        axes[1,0].set_title("Rolling252-session net Sharpe");axes[1,0].axhline(0,color="#aaa",lw=.7)
        axes[1,1].hist(own.daily_return.to_numpy(),bins=65,color=COLORS["payoff_context"],alpha=.8)
        axes[1,1].axvline(0,color="#aaa",lw=.8);axes[1,1].xaxis.set_major_formatter(PercentFormatter(1))
        axes[1,1].set_title("Net daily-return distribution · model");axes[1,1].set_ylabel("Sessions")
        for ax in axes.flat:ax.spines[["top","right"]].set_visible(False)
        fig.suptitle(f"{t} · temporal performance diagnostics",fontsize=15,fontweight="bold")
        self.save(fig,f"{t}_temporal")

    def temporal_tables(self,t):
        annual=[]
        for s in LABELS:
            d=self.cached("annual_performance",t,s);d.insert(0,"Account",LABELS[s]);annual.append(d)
        own=self.R.ledger(t);episodes=self.cached("drawdown_episodes",t)
        trades=completed_episodes(own)
        pnl=trades.net_pnl
        gross_profit=pnl.clip(lower=0).sum()
        concentration=pd.DataFrame([{"completed_episodes":len(trades),"mean_episode_net_pnl":pnl.mean(),
                                     "median_episode_net_pnl":pnl.median(),"best_episode_net_pnl":pnl.max(),
                                     "worst_episode_net_pnl":pnl.min(),
                                     "top_three_wins_fraction_of_positive_net_episode_pnl":pnl.nlargest(3).clip(lower=0).sum()/gross_profit if gross_profit>0 else np.nan,
                                     "largest_holding_sessions":trades.holding_sessions.max(),
                                     "mean_fees_per_episode":trades.fees.mean(),
                                     "total_resize_events":trades.resize_events.sum()}])
        worst=own[["daily_return","held_exposure","turnover","costs"]].sort_values("daily_return").head(10).reset_index()
        common=self.common.loc[self.common.ticker.eq(t),["window","start","end","days","sharpe","total_return","cagr","volatility","max_drawdown","average_exposure","turnover_per_year","delta_sharpe_vs_baseline","delta_sharpe_vs_causal_vol_reference"]]
        out="<h4>Common-period and early/late stability</h4>"+table(common)
        annual_frame=pd.concat(annual,ignore_index=True)
        annual_columns=["Account","year","net_return","sharpe","volatility","max_drawdown","average_exposure","annualized_turnover","costs_per_start_NAV","days","partial_period","carry_in_position"]
        out+="<h4>Annual account performance · continuing positions and costs</h4><p>Calendar returns retain actual carry-in positions and costs. Within-year drawdown resets the measurement peak to the year's starting NAV; it does not reset the trading account. Partial-period flags use the bundled pre2024 SPY-observed session calendar. All annual fields are also exported in data/annual_performance.csv.</p>"+table(annual_frame[annual_columns])
        monthly=self.cached("monthly_returns",t)
        out+=fold("Monthly return table · combined strategy",table(monthly[["year","month","net_return","sessions","partial_period","average_exposure","turnover","costs_per_start_NAV","carry_in_position"]]))
        out+="<h4>Ten deepest drawdowns · depth, duration and recovery</h4><p>All episodes for all three accounts are retained in data/drawdown_episodes.csv. An ongoing drawdown has no recovery date; duration is censored at the final observed session.</p>"+table(episodes.sort_values("depth").head(10))
        out+="<h4>Episode expectancy and concentration</h4><p>P&L and fees here are original NAV1 account currency, not percentage returns. Concentration measures the share of positive net episode P&L attributable to the three largest winners.</p>"+table(concentration)
        out+=fold("Worst10net daily returns and their exposure/cost context",table(worst))
        return out

    def forecast_tables(self,t):
        d=self.outcomes.loc[self.outcomes.ticker.eq(t)]
        intervals=self.intervals.loc[self.intervals.ticker.eq(t)]
        summary=forecast_summary(d)
        general_cols=["target","horizon_sessions","forecasts","available_horizon_labels","unavailable_horizon_labels",
                      "mean_actual_net_horizon_return","positive_outcome_rate","long_decision_rate",
                      "payoff_signal_direction_accuracy","true_positive","false_positive","true_negative","false_negative",
                      "precision_when_long","recall_positive_outcomes"]
        general=summary[[c for c in general_cols if c in summary]]
        raw_cols=["target","horizon_sessions","raw_probability_brier","raw_probability_log_loss",
                  "raw_probability_direction_accuracy","raw_regression_mae","raw_regression_rmse",
                  "raw_regression_r2_vs_zero_return","raw_prediction_actual_pearson","raw_prediction_actual_rank_correlation"]
        payoff_cols=["target","horizon_sessions","payoff_eligible_labels","mean_predicted_net_payoff",
                     "mean_observed_net_payoff","payoff_mean_error","payoff_mae","payoff_rmse","payoff_actual_rank_correlation"]
        out="""<p>These are saved outer-year forecasts with outcomes available by2023. Actual horizon targets use adjusted open-to-open returns minus the fixed0.1%round-trip label hurdle. They are not actual strategy account returns. Horizons overlap, and counts do not imply independent samples. Probability scores apply only to classification years; regression scores remain in return units. R² is explicitly versus a zero-return forecast.</p>"""
        out+="<h4>Horizon support, direction and decision confusion counts</h4>"+table(general)
        out+="<h4>Raw model forecasting statistics</h4>"+table(summary[[c for c in raw_cols if c in summary]])
        out+="<h4>Calibrated expected-payoff errors</h4>"+table(summary[[c for c in payoff_cols if c in summary]])
        out+=fold("Raw classifier probability reliability by horizon",table(probability_reliability(d)))
        out+=fold("Outer expected-payoff display bins", "<p>Fixed reporting bins are distinct from the earlier fitted OOF mapping bins. They do not change the trading policy. Dispersion is descriptive; no independent-sample error bars are claimed.</p>"+table(payoff_reliability(d)))
        out+="<h4>Actual weekly account intervals</h4><p>Return=next execution's pre-trade opening NAV/current execution's pre-trade opening NAV−1. The last partial interval ends at final net liquidation NAV. Each starting rebalance fee is counted once; these intervals compound exactly to the saved full account result.</p>"+table(weekly_summary(intervals))
        bins=[]
        for item in self.calibrators:
            if item["ticker"]!=t:continue
            c=item["calibrator"]
            for i,b in enumerate(c.get("bins",[])):
                bins.append({"year":item["outer_year"],"bin":i,"eligible":c["eligible"],"total_prior_oof":c["n"],
                             "bin_support":b["n"],"shrunk_expected_payoff":b["expected_payoff"],
                             "outcome_std_descriptive":b["outcome_std"],"pseudo_observations":c.get("pseudo_observations"),
                             "latest_matured_label":item["latest_matured_label"],"cutoff":item["calibration_cutoff"]})
        out+=fold("Prior OOF calibration fit records by year and bin", "<p>The same historical observation can appear in multiple annual calibration windows. These records are not pooled as independent outer evaluation outcomes.</p>"+table(pd.DataFrame(bins)))
        return out

    def forecast_figure(self,t):
        d=self.outcomes.loc[self.outcomes.ticker.eq(t)]
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        reliability=probability_reliability(d)
        axes[0,0].plot([0,1],[0,1],"--",color="#aaa")
        if len(reliability):
            for h,g in reliability.groupby("horizon_sessions"):
                axes[0,0].plot(g.mean_probability,g.observed_positive_fraction,"o-",label=f"{h} sessions")
            axes[0,0].legend(fontsize=8)
        else:axes[0,0].text(.5,.5,"No classification years",ha="center")
        axes[0,0].set(xlim=(0,1),ylim=(0,1),xlabel="Raw probability",ylabel="Observed positive fraction",title="Outer probability reliability")
        reg=d.loc[d.target.eq("regression")&d.label_available]
        if len(reg):
            for h,g in reg.groupby("horizon"):
                axes[0,1].scatter(g.prediction,g.actual_net_horizon_return,s=8,alpha=.3,label=f"{h} sessions")
            axes[0,1].legend(fontsize=8)
        else:axes[0,1].text(.5,.5,"No regression years",ha="center",transform=axes[0,1].transAxes)
        axes[0,1].set(xlabel="Predicted net target",ylabel="Observed net target",title="Outer regression forecasts")
        axes[0,1].xaxis.set_major_formatter(PercentFormatter(1));axes[0,1].yaxis.set_major_formatter(PercentFormatter(1))
        pay=payoff_reliability(d)
        if len(pay):
            for (target,h),g in pay.groupby(["target","horizon_sessions"]):
                axes[1,0].plot(g.mean_predicted_payoff,g.mean_observed_payoff,"o-",label=f"{target[:3]} · {h}")
            lo=min(pay.mean_predicted_payoff.min(),pay.mean_observed_payoff.min())
            hi=max(pay.mean_predicted_payoff.max(),pay.mean_observed_payoff.max())
            axes[1,0].plot([lo,hi],[lo,hi],"--",color="#aaa");axes[1,0].legend(fontsize=7)
        axes[1,0].set(xlabel="Mean predicted payoff",ylabel="Mean observed payoff",title="Outer payoff reliability · display bins")
        axes[1,0].xaxis.set_major_formatter(PercentFormatter(1));axes[1,0].yaxis.set_major_formatter(PercentFormatter(1))
        trade=completed_episodes(self.R.ledger(t))
        axes[1,1].hist(trade.holding_sessions,bins=min(15,max(3,len(trade)//3)),color=COLORS["payoff_context"])
        axes[1,1].set(xlabel="Invested sessions per completed episode",ylabel="Episodes",title="Actual holding-period distribution")
        for ax in axes.flat:ax.spines[["top","right"]].set_visible(False);ax.grid(alpha=.12)
        fig.suptitle(f"{t} · forecast, payoff and holding diagnostics",fontsize=15,fontweight="bold")
        self.save(fig,f"{t}_forecast")

    def factor_figure(self,t):
        fig,axes=plt.subplots(2,1,figsize=(12,7))
        usage=self.R.usage.loc[self.R.usage.ticker.eq(t)].copy()
        usage["family"]=np.where(usage.feature.str.startswith("ctx_"),"ETF context",np.where(usage.feature.str.startswith("rel_"),"Relationships","Technical"))
        family=usage.groupby(["year","family"]).importance.sum().unstack(fill_value=0)
        family.plot.bar(stacked=True,ax=axes[0],color=["#69a3a8","#b0b6c8","#8e699a"],width=.75)
        axes[0].set_title("Fitted split-importance share by input family");axes[0].set_ylabel("Importance share")
        axes[0].legend(fontsize=8,loc="upper right");axes[0].tick_params(axis="x",rotation=45)
        d=self.R.ledger(t);f=self.R.forecasts.loc[self.R.forecasts.ticker.eq(t)]
        axes[1].plot(d.index,d.held_exposure,color="#087f82",label="Realized daily exposure",lw=.8,alpha=.8)
        axes[1].step(f.entry_date,f.target_weight,where="post",color="#a66b3e",label="Weekly target weight",lw=.8,alpha=.8)
        axes[1].set_ylim(-.02,1.04);axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].set_title("Exposure and target position sizing");axes[1].legend(fontsize=8)
        for ax in axes:ax.spines[["top","right"]].set_visible(False)
        fig.suptitle(f"{t} · model inputs and exposure",fontsize=15,fontweight="bold")
        self.save(fig,f"{t}_factors")

    def benchmark_tables(self,t):
        d=self.paired.loc[self.paired.ticker.eq(t),["benchmark","strategy_sharpe","benchmark_sharpe","delta_sharpe","delta_sharpe_95pct_low","delta_sharpe_95pct_high","days","bootstrap_replications","block_sessions"]].copy()
        d["benchmark"]=d.benchmark.map(LABELS)
        original=self.original_uncertainty.loc[self.original_uncertainty.ticker.eq(t),["baseline_variant","delta_sharpe","delta_sharpe_95pct_low","delta_sharpe_95pct_high","p_holm","multiplicity_family_size"]]
        return "<h4>Paired Sharpe uncertainty versus ETF and SPY holdings</h4><p>Raw95% intervals from9,999 joint20-session block draws. These post-hoc descriptive intervals hold fitted strategies fixed, do not repeat model selection and carry no new p-values or promotion rule.</p>"+table(d)+"<h4>Preserved parent-study comparison with the original baseline</h4><p>This table retains the original225-comparison Holm adjustment. It applies to that parent comparison family, not to the30new ETF/SPY reporting intervals.</p>"+table(original)

    def literature_tables(self):
        p=json.loads((self.root/"reference_sources/literature_review.json").read_text())
        rows=[]
        for item in p["papers"]:
            rows.append({"Paper":f"<a href='{html.escape(item['primary_url'],quote=True)}'>{html.escape(item['title'])}</a> ({item['year']})",
                         "Evidence":html.escape(item["finding"]),"Limits for this model":html.escape(item["limits_for_this_project"])})
        return pd.DataFrame(rows).to_html(index=False,border=0,escape=False)

    def export_tables(self):
        collections={key:[] for key in ["monthly_returns","annual_performance","risk_statistics","drawdown_episodes","forecast_summary","probability_reliability","payoff_reliability","weekly_interval_summary"]}
        signature=self.cache_signature()
        cache_plan=self.root/"data/statistical_cache_signature.json"
        if cache_plan.exists() and json.loads(cache_plan.read_text())==signature and all((self.root/"data"/f"{name}.csv").exists() for name in collections):
            return
        for t in self.tickers:
            for s in LABELS:
                d=self.R.ledger(t,s)
                for key,fn in [("monthly_returns",monthly_returns),("annual_performance",annual_statistics),("drawdown_episodes",drawdown_episodes)]:
                    result=fn(d);result.insert(0,"strategy",s);result.insert(0,"ticker",t);collections[key].append(result)
                collections["risk_statistics"].append(pd.DataFrame([{"ticker":t,"strategy":s,**risk_statistics(d)}]))
            o=self.outcomes.loc[self.outcomes.ticker.eq(t)]
            for key,fn in [("forecast_summary",forecast_summary),("probability_reliability",probability_reliability),("payoff_reliability",payoff_reliability)]:
                result=fn(o);result.insert(0,"ticker",t);collections[key].append(result)
            result=weekly_summary(self.intervals.loc[self.intervals.ticker.eq(t)])
            result.insert(0,"ticker",t);collections["weekly_interval_summary"].append(result)
        for name,parts in collections.items():
            pd.concat(parts,ignore_index=True).to_csv(self.root/"data"/f"{name}.csv",index=False)
        cache_plan.write_text(json.dumps(signature,indent=2),encoding="utf-8")
        for name in ["monthly_returns","annual_performance","risk_statistics","drawdown_episodes"]:
            self.cache[name]=pd.read_csv(self.root/"data"/f"{name}.csv")

    def cache_signature(self):
        names=["data/ledgers.csv","data/forecast_outcomes.csv","data/weekly_policy_intervals.csv",
               "data/session_calendar.csv","src/temporal_statistics.py","src/forecast_statistics.py"]
        return {name:hashlib.sha256((self.root/name).read_bytes()).hexdigest() for name in names}
