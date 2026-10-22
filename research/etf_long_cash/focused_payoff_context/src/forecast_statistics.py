"""Descriptive diagnostics of saved chronological outer forecasts.

Historical target labels overlap and are not independent observations. These
diagnostics never fit a model, choose a threshold or use reserved outcomes.
Raw probability reliability and expected-payoff reliability are different.
"""
import numpy as np
import pandas as pd


def safe_mean(x):
    s=pd.Series(x,dtype=float).dropna()
    return float(s.mean()) if len(s) else None


def correlation(a,b,rank=False):
    d=pd.DataFrame({"a":a,"b":b}).dropna()
    if len(d)<3 or d.a.nunique()<2 or d.b.nunique()<2:return None
    if rank:d=d.rank()
    return float(d.a.corr(d.b))


def forecast_summary(outcomes):
    records=[]
    for (target,horizon),g in outcomes.groupby(["target","horizon"],sort=True):
        d=g.loc[g.label_available.astype(bool)].copy()
        rec={"target":target,"horizon_sessions":int(horizon),"forecasts":len(g),
             "available_horizon_labels":len(d),"unavailable_horizon_labels":len(g)-len(d)}
        if not len(d):
            records.append(rec);continue
        actual=d.actual_net_horizon_return
        truth=actual.gt(0)
        signal=d.signal.eq(1)
        tp=int((signal&truth).sum());fp=int((signal&~truth).sum())
        tn=int((~signal&~truth).sum());fn=int((~signal&truth).sum())
        rec.update(mean_actual_net_horizon_return=float(actual.mean()),
                   positive_outcome_rate=float(truth.mean()),long_decision_rate=float(signal.mean()),
                   payoff_signal_direction_accuracy=float(signal.eq(truth).mean()),
                   true_positive=tp,false_positive=fp,true_negative=tn,false_negative=fn,
                   precision_when_long=tp/(tp+fp) if tp+fp else None,
                   recall_positive_outcomes=tp/(tp+fn) if tp+fn else None,
                   specificity_negative_outcomes=tn/(tn+fp) if tn+fp else None,
                   mean_actual_return_when_long=safe_mean(actual.loc[signal]),
                   mean_actual_return_when_cash=safe_mean(actual.loc[~signal]))
        if target=="classification":
            probability=d.prediction
            clipped=probability.clip(1e-12,1-1e-12)
            rec.update(raw_probability_brier=float(((probability-truth.astype(float))**2).mean()),
                       raw_probability_log_loss=float(-(truth*np.log(clipped)+(~truth)*np.log(1-clipped)).mean()),
                       raw_probability_direction_accuracy=float(probability.gt(.5).eq(truth).mean()),
                       mean_raw_predicted_probability=float(probability.mean()))
        else:
            error=d.prediction-actual
            denominator=float(actual.pow(2).sum())
            rec.update(raw_regression_mae=float(error.abs().mean()),
                       raw_regression_rmse=float(np.sqrt(error.pow(2).mean())),
                       raw_regression_r2_vs_zero_return=1-float(error.pow(2).sum())/denominator if denominator>0 else None,
                       raw_prediction_actual_pearson=correlation(d.prediction,actual),
                       raw_prediction_actual_rank_correlation=correlation(d.prediction,actual,True))
        eligible=d.loc[d.expected_payoff.notna()]
        if len(eligible):
            error=eligible.expected_payoff-eligible.actual_net_horizon_return
            rec.update(payoff_eligible_labels=len(eligible),
                       mean_predicted_net_payoff=float(eligible.expected_payoff.mean()),
                       mean_observed_net_payoff=float(eligible.actual_net_horizon_return.mean()),
                       payoff_mean_error=float(error.mean()),payoff_mae=float(error.abs().mean()),
                       payoff_rmse=float(np.sqrt(error.pow(2).mean())),
                       payoff_actual_rank_correlation=correlation(eligible.expected_payoff,eligible.actual_net_horizon_return,True))
        records.append(rec)
    return pd.DataFrame(records)


def probability_reliability(outcomes):
    rows=[]
    d=outcomes.loc[outcomes.target.eq("classification")&outcomes.label_available.astype(bool)].copy()
    for horizon,g in d.groupby("horizon"):
        buckets=np.searchsorted([.2,.4,.6,.8],g.prediction.to_numpy(),side="right")
        for bucket,group in g.groupby(buckets):
            truth=group.actual_net_horizon_return.gt(0).astype(float)
            rows.append({"horizon_sessions":int(horizon),"probability_bin":int(bucket),
                         "bin_lower":int(bucket)*.2,"bin_upper":(int(bucket)+1)*.2,
                         "n":len(group),"mean_probability":float(group.prediction.mean()),
                         "observed_positive_fraction":float(truth.mean()),
                         "brier":float(((group.prediction-truth)**2).mean())})
    return pd.DataFrame(rows)


def payoff_reliability(outcomes):
    """Post-hoc fixed display bins; these are not the fitted OOF calibration bins."""
    edges=[-.01,-.002,0,.002,.01]
    labels=["below −1%","−1% to −0.2%","−0.2% to 0%","0% to 0.2%","0.2% to 1%","at least 1%"]
    d=outcomes.loc[outcomes.label_available.astype(bool)&outcomes.expected_payoff.notna()].copy()
    rows=[]
    for (target,horizon),g in d.groupby(["target","horizon"]):
        bins=np.searchsorted(edges,g.expected_payoff.to_numpy(),side="right")
        for bucket,group in g.groupby(bins):
            rows.append({"target":target,"horizon_sessions":int(horizon),"payoff_display_bin":labels[int(bucket)],
                         "n":len(group),"mean_predicted_payoff":float(group.expected_payoff.mean()),
                         "mean_observed_payoff":float(group.actual_net_horizon_return.mean()),
                         "observed_positive_fraction":float(group.actual_net_horizon_return.gt(0).mean()),
                         "outcome_std_descriptive":float(group.actual_net_horizon_return.std(ddof=1)) if len(group)>1 else None})
    return pd.DataFrame(rows)


def weekly_summary(intervals):
    rows=[]
    for label,g in intervals.groupby("partial_terminal_interval"):
        returns=g.policy_net_return
        rows.append({"interval_type":"final partial interval" if label else "completed weekly decision interval",
                     "intervals":len(g),"positive_intervals":int(returns.gt(1e-12).sum()),
                     "negative_intervals":int(returns.lt(-1e-12).sum()),
                     "flat_intervals":int(returns.abs().le(1e-12).sum()),
                     "positive_fraction":float(returns.gt(1e-12).mean()),
                     "mean_net_return":float(returns.mean()),"median_net_return":float(returns.median()),
                     "worst_net_return":float(returns.min()),"best_net_return":float(returns.max()),
                     "mean_invested_target":float(g.target_weight.mean())})
    return pd.DataFrame(rows)
