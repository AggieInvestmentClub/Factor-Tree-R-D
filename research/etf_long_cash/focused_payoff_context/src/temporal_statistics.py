"""Pure pandas/numpy descriptive statistics for saved daily account ledgers.

All returns, drawdowns, volatility and tail figures are decimal fractions.
The analysis NAV starts at 1 immediately BEFORE the first supplied daily return;
that return is retained. Subperiods measure the continuing saved strategy and
never simulate a fresh allocation, trading cost or artificial liquidation.

Ratios annualize with 252 sessions and a zero target/risk-free convention.
Undefined ratios return Python None, never infinity. Calendar days and trading
sessions are separately named. No function fits models, accesses files, fetches
prices or changes the caller's ledger.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

MONTH_COLUMNS=["year","month","period","net_return","sessions","days","calendar_days","partial_period","partial_period_basis",
               "start","end","average_exposure","turnover","annualized_turnover","costs","costs_per_start_NAV",
               "carry_in_nav_original","carry_in_position","had_prior_ledger_session"]
EPISODE_COLUMNS=["episode","peak_date","initial_peak_before_first_session","drawdown_start_date","trough_date","recovery_date",
                 "end_date","depth","peak_NAV","trough_NAV","underwater_sessions","duration_sessions",
                 "peak_to_recovery_calendar_days","peak_to_end_calendar_days","trough_to_recovery_sessions",
                 "trough_to_recovery_calendar_days","ongoing"]


def _number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _result_frame(rows,columns=None):
    """Keep undefined scalar statistics as Python None in tabular records too."""
    frame=pd.DataFrame(rows,columns=columns)
    nullable=["sharpe","sortino","calmar","skew","excess_kurtosis","volatility","annualized_return",
              "peak_to_recovery_calendar_days","trough_to_recovery_sessions","trough_to_recovery_calendar_days"]
    for name in frame:
        if name in nullable or frame[name].isna().any():
            frame[name]=pd.Series([None if pd.isna(x) else x for x in frame[name]],index=frame.index,dtype=object)
    return frame


def _prepare(ledger):
    attrs=dict(getattr(ledger,"attrs",{}))
    d=ledger.copy()
    if "Date" in d.columns:
        d=d.set_index("Date")
    d.index=pd.DatetimeIndex(pd.to_datetime(d.index))
    if d.index.tz is not None:
        d.index=d.index.tz_localize(None)
    d.index=d.index.normalize()
    d.index.name="Date"
    if not d.index.is_unique or not d.index.is_monotonic_increasing:
        raise ValueError("Ledger requires unique, increasing daily session dates")
    if "daily_return" not in d:
        if "nav" not in d:
            if d.empty:d["daily_return"]=pd.Series(index=d.index,dtype=float)
            else:raise ValueError("Ledger requires daily_return, or nav with initial NAV 1")
        else:
            nav=pd.to_numeric(d.nav,errors="raise")
            d["daily_return"]=nav.div(nav.shift(1).fillna(1))-1
    d["daily_return"]=pd.to_numeric(d.daily_return,errors="raise").astype(float)
    if not np.isfinite(d.daily_return).all() or d.daily_return.le(-1).any():
        raise ValueError("Daily returns must be finite and greater than -100%")
    if "nav" in d:
        d["nav"]=pd.to_numeric(d.nav,errors="raise").astype(float)
        if not np.isfinite(d.nav).all() or d.nav.le(0).any():raise ValueError("Source NAV must be finite and positive")
        if len(d)>1 and not np.allclose(d.nav.iloc[1:].to_numpy()/d.nav.iloc[:-1].to_numpy()-1,d.daily_return.iloc[1:],rtol=1e-9,atol=1e-11):
            raise ValueError("Saved daily returns disagree with successive source NAV values")
    for c in ["cash","shares","weight","held_exposure","turnover","costs","trade_value","terminal_sale","terminal_cost","opening_nav"]:
        if c in d:
            d[c]=pd.to_numeric(d[c],errors="raise").astype(float)
            if not np.isfinite(d[c]).all():raise ValueError(f"Nonfinite ledger field {c}")
    d.attrs=attrs
    return d


def _normalized_nav(d):
    return (1+d.daily_return).cumprod()


def _source_prior_nav(d):
    if d.empty:return None
    return float(d.nav.iloc[0]/(1+d.daily_return.iloc[0])) if "nav" in d else 1.


def _core(d):
    n=len(d);r=d.daily_return;nav=_normalized_nav(d)
    total=float(nav.iloc[-1]-1) if n else None
    std=float(r.std(ddof=1)) if n>1 else None
    downside=float(np.sqrt(np.mean(np.minimum(r.to_numpy(),0)**2))) if n else None
    dd=nav/nav.cummax().clip(lower=1)-1
    maxdd=float(dd.min()) if n else None
    with np.errstate(over="ignore",invalid="ignore"):
        annual_return=_number(np.expm1(np.log(nav.iloc[-1])*252/n)) if n else None
    tail_n=max(1,math.ceil(.05*n)) if n else 0
    exposure="held_exposure" if "held_exposure" in d else "weight" if "weight" in d else None
    prior=_source_prior_nav(d)
    turnover=float(d.turnover.sum()) if "turnover" in d and n else None
    costs=float(d.costs.sum()) if "costs" in d and n else None
    trade_events=int(d.trade_value.abs().gt(1e-10).sum()+(d.terminal_sale.gt(0).sum() if "terminal_sale" in d else 0)) if "trade_value" in d else None
    return {
        "start":str(d.index[0].date()) if n else None,"end":str(d.index[-1].date()) if n else None,
        "sessions":n,"days":n,"calendar_days":int((d.index[-1]-d.index[0]).days+1) if n else 0,
        "elapsed_calendar_days":int((d.index[-1]-d.index[0]).days) if n else 0,
        "annualization_years_252":n/252,"net_return":total,"annualized_return":annual_return,
        "annualized_mean_return":float(r.mean()*252) if n else None,
        "volatility":_number(std*np.sqrt(252)) if std is not None else None,
        "sharpe":_number(np.sqrt(252)*r.mean()/std) if std is not None and std>1e-15 else None,
        "downside_semideviation_daily":downside,
        "downside_semideviation_annual":_number(downside*np.sqrt(252)) if downside is not None else None,
        "sortino":_number(np.sqrt(252)*r.mean()/downside) if downside is not None and downside>1e-15 else None,
        "max_drawdown":maxdd,
        "calmar":_number(annual_return/abs(maxdd)) if annual_return is not None and maxdd is not None and maxdd < -1e-15 else None,
        "ulcer_index":float(np.sqrt(np.mean(dd.to_numpy()**2))) if n else None,
        "daily_win_fraction":float(r.gt(0).mean()) if n else None,
        "daily_loss_fraction":float(r.lt(0).mean()) if n else None,
        "daily_flat_fraction":float(r.eq(0).mean()) if n else None,
        "best_daily_return":float(r.max()) if n else None,"worst_daily_return":float(r.min()) if n else None,
        "var5_daily_return":float(r.quantile(.05)) if n else None,
        "cvar5_daily_return":float(np.sort(r.to_numpy())[:tail_n].mean()) if n else None,
        "cvar5_tail_sessions":tail_n,
        "skew":_number(r.skew()) if n>=3 and std is not None and std>1e-15 else None,
        "excess_kurtosis":_number(r.kurt()) if n>=4 and std is not None and std>1e-15 else None,
        "average_exposure":float(d[exposure].mean()) if exposure and n else None,"exposure_field":exposure,
        "mean_postclose_cash_fraction":float((d.cash/d.nav).mean()) if "cash" in d and "nav" in d and n else None,
        "turnover":turnover,"annualized_turnover":turnover/(n/252) if turnover is not None and n else None,
        "costs":costs,"costs_per_start_NAV":costs/prior if costs is not None and prior else None,
        "trade_events":trade_events,"carry_in_nav_original":prior,
    }


def _partial(d,period,group):
    calendar=d.attrs.get("session_calendar")
    if calendar is not None:
        cal=pd.DatetimeIndex(pd.to_datetime(calendar))
        if cal.tz is not None:cal=cal.tz_localize(None)
        expected=cal[(cal>=period.start_time.normalize())&(cal<=period.end_time.normalize())].normalize()
        return not group.index.equals(expected),"provided_session_calendar"
    weekdays=pd.bdate_range(period.start_time.normalize(),period.end_time.normalize())
    boundary=(group.index[0]==d.index[0] and group.index[0]>weekdays[0]) or (group.index[-1]==d.index[-1] and group.index[-1]<weekdays[-1])
    return bool(boundary),"conservative_boundary_flag_without_exchange_calendar"


def _carry(d,group):
    earlier=d.loc[d.index<group.index[0]]
    if len(earlier):
        previous=earlier.iloc[-1]
        active=bool(previous.shares>1e-10) if "shares" in earlier else bool(previous.weight>1e-10) if "weight" in earlier else None
        return {"carry_in_position":active,"had_prior_ledger_session":True,"prior_ledger_date":str(earlier.index[-1].date())}
    return {"carry_in_position":_initial_position(d),"had_prior_ledger_session":False,"prior_ledger_date":None}


def monthly_returns(ledger):
    """Monthly compounded saved net returns, retaining each month's first return.

    Columns include year/month/net_return/sessions/partial_period plus observed
    calendar span, exposure, turnover, actual costs and carry-in state. Turnover
    is summed traded-notional/current-NAV; annualized_turnover multiplies by
    252/sessions. Costs retain source NAV currency. An optional
    ledger.attrs['session_calendar'] makes partial_period an exact calendar
    coverage comparison. Without it, the flag conservatively identifies dataset
    boundary truncation using weekdays; exchange holidays may also trigger it.
    """
    d=_prepare(ledger);rows=[]
    for period,g in d.groupby(d.index.to_period("M")):
        c=_core(g);partial,basis=_partial(d,period,g)
        rows.append(dict(year=period.year,month=period.month,period=str(period),partial_period=partial,partial_period_basis=basis,**c,**_carry(d,g)))
    return _result_frame(rows) if rows else pd.DataFrame(columns=MONTH_COLUMNS)


def annual_statistics(ledger):
    """Calendar-year continuing-account metrics; no annual trading/cost reset.

    net_return is the observed year's compounded return. annualized_return is
    its 252-session CAGR convention, which differs for partial years. 'days' is
    trading sessions, while calendar_days is the inclusive observed date span.
    First-session returns, overnight carry and any actual original liquidation
    costs remain included. Partial-period calendar caveats match monthly_returns.
    """
    d=_prepare(ledger);rows=[]
    for period,g in d.groupby(d.index.to_period("Y")):
        partial,basis=_partial(d,period,g)
        rows.append(dict(year=period.year,partial_period=partial,partial_period_basis=basis,**_core(g),**_carry(d,g)))
    return _result_frame(rows) if rows else pd.DataFrame(columns=["year","net_return","sharpe","max_drawdown","volatility","average_exposure","turnover","days"])


def drawdown_episodes(ledger):
    """Peak/trough/recovery episodes from return-compounded NAV, including NAV 1.

    Depth is a negative decimal return. underwater_sessions counts closes below
    the old peak and excludes a recovered close. duration_sessions counts session
    intervals from peak to recovery/as-of, using a virtual pre-first-session
    position -1 for an initial loss. For that virtual peak, peak_date is the first
    session date with initial_peak_before_first_session=True; no fictitious prior
    calendar date is invented. Calendar durations are elapsed date differences.
    Ongoing episodes end at the last supplied date; recovery_date and actual
    peak_to_recovery_calendar_days remain None, but peak_to_end_calendar_days
    reports their censored duration through that last date.
    """
    d=_prepare(ledger)
    if d.empty:return pd.DataFrame(columns=EPISODE_COLUMNS)
    nav=_normalized_nav(d).to_numpy();dates=d.index;peak=1.;peak_i=-1;active=None;rows=[]
    def finish(end_i,recovered):
        pdate=dates[active["peak_i"]] if active["peak_i"]>=0 else dates[0]
        tdate=dates[active["trough_i"]];edate=dates[end_i]
        rows.append({"episode":len(rows)+1,"peak_date":str(pdate.date()),"initial_peak_before_first_session":active["peak_i"]<0,
                     "drawdown_start_date":str(dates[active["start_i"]].date()),"trough_date":str(tdate.date()),
                     "recovery_date":str(edate.date()) if recovered else None,"end_date":str(edate.date()),
                     "depth":float(active["trough_nav"]/active["peak_nav"]-1),"peak_NAV":float(active["peak_nav"]),"trough_NAV":float(active["trough_nav"]),
                     "underwater_sessions":end_i-active["start_i"]+(0 if recovered else 1),
                     "duration_sessions":end_i-active["peak_i"],
                     "peak_to_recovery_calendar_days":int((edate-pdate).days) if recovered else None,
                     "peak_to_end_calendar_days":int((edate-pdate).days),
                     "trough_to_recovery_sessions":end_i-active["trough_i"] if recovered else None,
                     "trough_to_recovery_calendar_days":int((edate-tdate).days) if recovered else None,"ongoing":not recovered})
    for i,value in enumerate(nav):
        if active is None:
            if value < peak*(1-1e-12):
                active={"peak_i":peak_i,"peak_nav":peak,"start_i":i,"trough_i":i,"trough_nav":value}
            else:
                peak=max(peak,value);peak_i=i
        elif value >= active["peak_nav"]*(1-1e-12):
            finish(i,True);active=None;peak=max(peak,value);peak_i=i
        elif value < active["trough_nav"]:
            active["trough_nav"]=value;active["trough_i"]=i
    if active is not None:finish(len(nav)-1,False)
    return _result_frame(rows,columns=EPISODE_COLUMNS)


def _initial_position(d):
    """Infer carry-in from self-financing ledger fields, otherwise return None."""
    required={"opening_nav","cash","trade_value","costs","terminal_sale","terminal_cost"}
    if d.empty or not required.issubset(d.columns):return None
    x=d.iloc[0]
    cash_after_open=x.cash-x.terminal_sale+x.terminal_cost
    opening_cost=x.costs-x.terminal_cost
    prior_asset_value=x.opening_nav-(cash_after_open+x.trade_value+opening_cost)
    return bool(prior_asset_value>1e-9)


def _episode_statistics(d):
    """Net whole-account returns over continuous positive-position episodes."""
    if not {"shares","held_exposure"}.issubset(d.columns):
        return {"closed_long_episodes":None,"open_long_episodes":None,"trade_episode_win_fraction":None,
                "trade_episode_loss_fraction":None,"trade_episode_flat_fraction":None,
                "trade_episode_note":"Not computed: requires saved shares and held_exposure, distinct from daily win/loss fractions."}
    initial=_initial_position(d);episodes=[];start=None;left=False
    for i,(_,x) in enumerate(d.iterrows()):
        during=x.held_exposure>1e-10;after=x.shares>1e-10
        if start is None and (during or after or (i==0 and initial is True)):
            start=i;left=i==0 and initial is not False
        if start is not None and not after:
            rr=float((1+d.daily_return.iloc[start:i+1]).prod()-1)
            episodes.append({"return":rr,"closed":True,"left_censored":left});start=None
    if start is not None:
        episodes.append({"return":float((1+d.daily_return.iloc[start:]).prod()-1),"closed":False,"left_censored":left})
    eligible=[x["return"] for x in episodes if x["closed"] and not x["left_censored"]]
    return {"closed_long_episodes":sum(x["closed"] for x in episodes),"open_long_episodes":sum(not x["closed"] for x in episodes),
            "left_censored_long_episodes":sum(x["left_censored"] for x in episodes),"win_rate_eligible_closed_episodes":len(eligible),
            "trade_episode_win_fraction":sum(x>1e-12 for x in eligible)/len(eligible) if eligible else None,
            "trade_episode_loss_fraction":sum(x<-1e-12 for x in eligible)/len(eligible) if eligible else None,
            "trade_episode_flat_fraction":sum(abs(x)<=1e-12 for x in eligible)/len(eligible) if eligible else None,
            "trade_episode_note":"A continuous long-position episode ends when shares become zero, including the exit day's overnight return/fees and actual terminal liquidation. Rebalances within positive exposure are not new episodes. Episode return is net whole-account return over that interval, not a per-lot or per-dollar-invested trade return. Open/left-censored episodes are excluded from win fractions; zero cash yield is the declared source-account convention."}


def risk_statistics(ledger):
    """Whole supplied-window descriptive statistics with portable undefined None.

    VaR5 is the empirical 5th return percentile (pandas linear interpolation).
    CVaR5 is the signed mean of the worst ceil(5%*N) daily returns, including ties
    only as needed to keep that count. Downside semideviation uses target zero
    and denominator N: sqrt(mean(min(r,0)^2)); annualized Sortino is
    sqrt(252)*mean(r)/that_daily_semideviation. Skew/kurtosis use pandas unbiased
    sample estimators; kurtosis is excess (normal=0). Best/worst month/year can
    include boundary partial periods. No p values or significance claims occur.
    """
    d=_prepare(ledger);result=_core(d);monthly=monthly_returns(d);annual=annual_statistics(d);dd=drawdown_episodes(d)
    for label,frame,keys in [("month",monthly,["year","month"]),("year",annual,["year"])]:
        for direction,fn in [("best","idxmax"),("worst","idxmin")]:
            if frame.empty:result[f"{direction}_{label}"]=None
            else:
                row=frame.loc[getattr(frame.net_return,fn)()]
                result[f"{direction}_{label}"]={**{k:int(row[k]) for k in keys},"net_return":float(row.net_return),"sessions":int(row.sessions),"partial_period":bool(row.partial_period)}
    result.update({"drawdown_episodes":len(dd),"ongoing_drawdown":bool(dd.ongoing.any()) if len(dd) else False,
                   "longest_underwater_sessions":int(dd.underwater_sessions.max()) if len(dd) else 0,
                   "longest_peak_to_end_sessions":int(dd.duration_sessions.max()) if len(dd) else 0,
                   "longest_peak_to_end_calendar_days":int(dd.peak_to_end_calendar_days.max()) if len(dd) else 0,
                   **_episode_statistics(d),
                   "units":{"returns_volatility_drawdown_tail":"decimal fractions; multiply by100 for percent display","annualization":"252 trading sessions; sample volatility ddof=1; zero risk-free and downside target","days":"days/sessions are observations; calendar_days is inclusive observed span; drawdown calendar durations are elapsed date differences","turnover":"sum of actual traded notional/current NAV; annualized_turnover=sum*252/sessions","costs":"summed original account NAV currency; costs_per_start_NAV divides by inferred prior original NAV, not an estimate of compounded cost drag","NAV":"measurement NAV1 before first supplied return; saved return and source trades/costs retained","ulcer_index":"sqrt(mean(daily drawdown^2)), decimal drawdown units","daily_vs_trade_wins":"daily positive/negative/zero fractions use all daily observations; episode fractions use only completed, uncensored positive-position intervals"},
                   "limitations":"Descriptive realized sample only. Serial dependence, overlapping forecasts, prior model selection and multiple comparisons remain; no prospective or statistical-significance conclusion follows."})
    return result
