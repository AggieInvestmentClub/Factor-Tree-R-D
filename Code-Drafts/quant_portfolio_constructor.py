"""Drop-in LONG/PUT/EXCLUDE portfolio constructor. Research prototype, no order execution.
Install: pip install yfinance pandas scipy
"""
import math
from datetime import datetime, date
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Keep variables set by code above this block; otherwise use standalone examples.
if "model_signals" not in globals():
    model_signals = [
        {"ticker": "AAPL", "signal": "LONG"},
        {"ticker": "ADBE", "signal": "LONG"},
        {"ticker": "NVDA", "signal": "PUT"},
        {"ticker": "META", "signal": "PUT"},
        {"ticker": "TSLA", "signal": "EXCLUDE"},
        {"ticker": "AMZN", "signal": "EXCLUDE"},
    ]
if "available_capital" not in globals():
    available_capital = 100_000.0
if "put_expiration_weeks" not in globals():
    put_expiration_weeks = 2
if "min_abs_delta" not in globals():
    min_abs_delta = 0.95
if "max_extrinsic_fraction" not in globals():
    max_extrinsic_fraction = 0.03  # ask-side extrinsic / total ask premium
if "max_spread_fraction" not in globals():
    max_spread_fraction = 0.08  # (ask-bid)/midpoint
if "min_open_interest" not in globals():
    min_open_interest = 10
if "risk_free_rate" not in globals():
    risk_free_rate = 0.04  # annual continuously compounded approximation
if "dividend_yield" not in globals():
    dividend_yield = 0.0  # prototype; replace per ticker for production
if "max_price_age_days" not in globals():
    max_price_age_days = 4  # includes weekends; block stale daily closes
if "max_gross_delta_leverage" not in globals():
    max_gross_delta_leverage = None  # e.g. 2.0; None = report only
if "export_csv" not in globals():
    export_csv = True


def standardized_signals(raw):
    if isinstance(raw, pd.DataFrame):
        data = raw[["ticker", "signal"]].to_dict("records")
    elif isinstance(raw, dict):
        data = [{"ticker": k, "signal": v} for k, v in raw.items()]
    else:
        data = list(raw)
    result = {}
    for row in data:
        ticker = str(row["ticker"]).strip().upper()
        signal = str(row["signal"]).strip().upper()
        if not ticker or signal not in {"LONG", "PUT", "EXCLUDE"}:
            raise ValueError(f"Invalid signal row: {row}")
        if ticker in result:
            raise ValueError(f"Duplicate ticker: {ticker}")
        result[ticker] = signal
    return result


def estimate_put_delta(spot, strike, time_years, iv, rate, yield_rate):
    """European BSM put delta: estimate only, especially for American equity puts."""
    if min(spot, strike, time_years, iv) <= 0:
        return float("nan")
    d1 = (math.log(spot / strike) + (rate - yield_rate + 0.5*iv*iv)*time_years) / (iv*math.sqrt(time_years))
    return math.exp(-yield_rate*time_years)*(norm.cdf(d1)-1)


def choose_expiry(expiries, weeks, today):
    """Nearest listed Friday >= target; avoid same-day expiry."""
    target_days = int(7*weeks)
    candidates = [date.fromisoformat(e) for e in expiries
                  if date.fromisoformat(e) > today and date.fromisoformat(e).weekday() == 4]
    if not candidates:
        raise ValueError("No future Friday expirations supplied by provider")
    return min(candidates, key=lambda d: (abs((d-today).days-target_days),
                                          (d-today).days < target_days)).isoformat()


def valid_spot(ticker_obj, ticker, now):
    history = ticker_obj.history(period="7d", interval="1d", auto_adjust=False)
    if history.empty:
        raise ValueError(f"No stock price data for {ticker}")
    last_stamp = pd.Timestamp(history.index[-1])
    last_date = last_stamp.date()
    if (now.date() - last_date).days > max_price_age_days:
        raise ValueError(f"Stale price for {ticker}: {last_date}")
    spot = float(history["Close"].iloc[-1])
    if not math.isfinite(spot) or spot <= 0:
        raise ValueError(f"Invalid stock price for {ticker}")
    return spot, last_date


def choose_put(ticker_obj, ticker, spot, now):
    exp = choose_expiry(ticker_obj.options, put_expiration_weeks, now.date())
    chain = ticker_obj.option_chain(exp).puts
    if chain is None or chain.empty:
        raise ValueError(f"No puts returned for {ticker} {exp}")
    # Conservative: use calendar days until 4 pm New York expiration.
    expiry_time = datetime.combine(date.fromisoformat(exp), datetime.min.time(),
                                   tzinfo=ZoneInfo("America/New_York")).replace(hour=16)
    years = (expiry_time - now).total_seconds()/(365.25*86400)
    if years <= 0:
        raise ValueError("Expiration already passed")
    passing = []
    for _, opt in chain.iterrows():
        try:
            strike = float(opt["strike"])
            bid, ask = float(opt["bid"]), float(opt["ask"])
            iv = float(opt["impliedVolatility"])
            oi = float(opt["openInterest"])
            if not all(math.isfinite(x) for x in (strike, bid, ask, iv, oi)):
                continue
            if not (strike > spot and 0 < bid <= ask and 0.01 <= iv <= 5 and oi >= min_open_interest):
                continue
            mid = (bid+ask)/2
            if (ask-bid)/mid > max_spread_fraction:
                continue
            # Marked extrinsic can be negative due to asynchronous/invalid quotes.
            intrinsic = strike-spot
            extrinsic = ask-intrinsic
            if extrinsic < -0.02:  # reject economically inconsistent quotes
                continue
            if max(0, extrinsic)/ask > max_extrinsic_fraction:
                continue
            delta = estimate_put_delta(spot, strike, years, iv, risk_free_rate, dividend_yield)
            if not math.isfinite(delta) or delta > -min_abs_delta:
                continue
            passing.append({"strike":strike, "bid":bid, "ask":ask,
                            "iv":iv,"delta":delta,"extrinsic":extrinsic,
                            "contractSymbol":str(opt["contractSymbol"]),
                            "expiration":exp,"oi":int(oi)})
        except (ValueError, TypeError, KeyError, ZeroDivisionError):
            continue
    if not passing:
        raise ValueError(f"No qualifying put for {ticker} ({exp}); adjust criteria or source")
    return min(passing, key=lambda x:(x["strike"], x["ask"]-x["bid"]))


def build_portfolio(signals, capital):
    signals = standardized_signals(signals)
    if not math.isfinite(float(capital)) or capital <= 0:
        raise ValueError("available_capital must be positive")
    now = datetime.now(ZoneInfo("America/New_York"))
    active = []
    for ticker, signal in signals.items():
        if signal == "EXCLUDE":
            continue
        obj = yf.Ticker(ticker)
        spot, price_date = valid_spot(obj, ticker, now)
        if signal == "LONG":
            active.append(dict(ticker=ticker, signal=signal, instrument=ticker,
                               price=spot, unit_cost=spot, unit_exposure=spot,
                               delta=1.0, strike=None, expiration=None,
                               bid=None, ask=None, price_date=price_date))
        else:
            put = choose_put(obj, ticker, spot, now)
            active.append(dict(ticker=ticker, signal=signal,
                               instrument=put["contractSymbol"], price=spot,
                               unit_cost=put["ask"]*100,
                               unit_exposure=abs(put["delta"])*spot*100,
                               delta=put["delta"], strike=put["strike"],
                               expiration=put["expiration"],bid=put["bid"],
                               ask=put["ask"],price_date=price_date))
    if not active:
        return pd.DataFrame(), {"cash_remaining": float(capital), "gross_leverage":0.0}

    # Continuous target: equal absolute delta-exposure by ticker, cost scaled to NAV.
    exposure_per_name = capital/sum(p["unit_cost"]/p["unit_exposure"] for p in active)
    for p in active:
        p["quantity"] = math.floor(exposure_per_name/p["unit_exposure"])

    spent = sum(p["quantity"]*p["unit_cost"] for p in active)
    # Whole-unit refinement: add to most underweight affordable ticker.
    # Do not exceed continuous exposure target to avoid concentrating in cheap options.
    while True:
        candidates = [p for p in active if p["unit_cost"] <= capital-spent+1e-8 and
                      (p["quantity"]+1)*p["unit_exposure"] <= exposure_per_name+1e-8]
        if not candidates:
            break
        p = max(candidates, key=lambda x: exposure_per_name-x["quantity"]*x["unit_exposure"])
        p["quantity"] += 1
        spent += p["unit_cost"]

    for p in active:
        p["cash_cost"] = p["quantity"]*p["unit_cost"]
        p["signed_delta_exposure"] = (1 if p["signal"]=="LONG" else -1)*p["quantity"]*p["unit_exposure"]

    gross = sum(abs(p["signed_delta_exposure"]) for p in active)
    net = sum(p["signed_delta_exposure"] for p in active)
    leverage = gross/capital
    if max_gross_delta_leverage is not None and leverage > max_gross_delta_leverage:
        raise ValueError(f"Gross delta leverage {leverage:.2f} exceeds cap {max_gross_delta_leverage:.2f}")
    summary = {"as_of":now.isoformat(timespec="seconds"), "capital":capital,
               "spent":spent, "cash_remaining":capital-spent,
               "deployment_pct":spent/capital*100,
               "gross_delta_exposure":gross, "net_delta_exposure":net,
               "gross_leverage":leverage,
               "target_abs_exposure_per_ticker":exposure_per_name}
    columns = ["ticker","signal","instrument","quantity","price","strike", "expiration",
               "delta","bid","ask","cash_cost","signed_delta_exposure", "price_date"]
    return pd.DataFrame(active)[columns], summary


portfolio_orders, portfolio_summary = build_portfolio(model_signals, float(available_capital))
print("\nPROPOSED FRIDAY PORTFOLIO (research quotes; not verified executable prices)")
if portfolio_orders.empty:
    print("No active positions")
else:
    print(portfolio_orders.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
print("\nSUMMARY")
for k, v in portfolio_summary.items():
    print(f"{k}: {v:,.2f}" if isinstance(v, (int,float)) else f"{k}: {v}")
if export_csv:
    portfolio_orders.to_csv("proposed_portfolio.csv", index=False)
    print("\nSaved: proposed_portfolio.csv")
