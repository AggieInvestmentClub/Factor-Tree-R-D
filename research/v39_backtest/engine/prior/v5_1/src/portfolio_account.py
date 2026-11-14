"""Joint signed ETF account; targets are fractions of NAV after trading fees.

The account has no external flows, cash yield or short borrowing charge. Target
gross exposure may not exceed one. It is a research account, without historical
locate/recall or exchange short-sale execution data. Prices are supplied already
adjusted; this module does not separately credit/debit distributions.

``simulate_portfolio`` returns ``(daily_ledger, daily_positions)``. Positions are
long-form Date/ticker records. Open margin breaches skip scheduled targets. Close
breaches act at that close, without reversing earlier open trades. Mandatory
short covering can also require the minimum proportional long sale to avoid a
cash loan. Insolvency raises a dated exception instead of inventing later returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class AccountInsolvencyError(ValueError):
    """A complete arm must be recorded as failed when its NAV becomes nonpositive."""

    def __init__(self, date, stage, nav):
        self.date = pd.Timestamp(date)
        self.stage = stage
        self.nav = float(nav)
        super().__init__(f"Portfolio insolvent on {self.date.date()} at {stage}: NAV={nav:.17g}")


def _positive_nav(value, date, stage):
    if not np.isfinite(value) or value <= 0:
        raise AccountInsolvencyError(date, stage, value)


def _post_fee_values(wealth, old, weights, fee):
    """Solve E + fee*sum(abs(weights*E-old)) = wealth monotonically."""
    if fee == 0:
        return weights * wealth
    if wealth <= fee * np.abs(old).sum():
        return None
    lower, upper = 0.0, wealth
    # With fee < 1 and sum(abs(weights)) <= 1 the residual is strictly increasing.
    for _ in range(60):
        middle = (lower + upper) * .5
        residual = middle + fee * np.abs(weights * middle - old).sum() - wealth
        if residual > 0:
            upper = middle
        else:
            lower = middle
    return weights * ((lower + upper) * .5)


def _margin_cover(cash, shares, marks, fee, date, stage):
    """Cover shorts, selling the minimum proportional long value if necessary."""
    before = shares * marks
    delta = np.where(before < 0, -before, 0.0)
    cover = float(delta.sum())
    cash -= cover * (1 + fee)
    updated = shares.copy()
    updated[before < 0] = 0.0
    long_sale = 0.0
    if cash < 0:
        available = np.maximum(before, 0.0)
        total_long = float(available.sum())
        long_sale = -cash / (1 - fee)
        if long_sale >= total_long:
            raise AccountInsolvencyError(date, stage + " forced sale", cash + total_long * (1 - fee))
        sales = available * (long_sale / total_long)
        delta -= sales
        updated -= sales / marks
        cash += long_sale * (1 - fee)
    cash = max(0.0, float(cash))
    values = updated * marks
    _positive_nav(cash + values.sum(), date, stage + " after cover")
    return cash, updated, delta, cover, long_sale


def _inputs(prices, targets, start, end, cost_bps, maintenance_margin):
    if not isinstance(prices, dict) or not prices:
        raise ValueError("prices must contain at least one ETF")
    if not isinstance(targets, pd.DataFrame):
        raise ValueError("targets must be a complete execution-date by ETF DataFrame")
    tickers = list(prices)
    if not targets.columns.is_unique or set(targets.columns) != set(tickers):
        raise ValueError("Each target row must contain exactly all price tickers")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("cost_bps must be finite and in [0,10000)")
    if not np.isfinite(maintenance_margin) or not 0 <= maintenance_margin <= 1:
        raise ValueError("maintenance_margin must be in [0,1]")
    target_index = pd.DatetimeIndex(targets.index)
    if target_index.hasnans or not target_index.is_unique:
        raise ValueError("Target execution dates must be unique and nonmissing")
    weights = targets.loc[:, tickers].to_numpy(dtype=float, copy=True)
    if not np.isfinite(weights).all() or (np.abs(weights).sum(axis=1) > 1 + 1e-12).any():
        raise ValueError("Every target must be finite with total absolute weight <=1")
    # Tolerate arithmetic roundoff only; never scale a meaningfully leveraged row.
    gross = np.abs(weights).sum(axis=1)
    tiny_excess = gross > 1
    weights[tiny_excess] /= gross[tiny_excess, None]
    index = None
    openings, closes = [], []
    for ticker in tickers:
        frame = prices[ticker]
        if not {"adj_open", "adj_close"}.issubset(frame.columns):
            raise ValueError(f"Missing adjusted open/close for {ticker}")
        dates = pd.DatetimeIndex(frame.index)
        if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError(f"Invalid or unsorted price dates for {ticker}")
        mask = np.ones(len(dates), dtype=bool)
        if start is not None:
            mask &= dates >= pd.Timestamp(start)
        if end is not None:
            mask &= dates <= pd.Timestamp(end)
        selected = dates[mask]
        if not len(selected):
            raise ValueError(f"Empty selected price history for {ticker}")
        if index is None:
            index = selected
        elif not index.equals(selected):
            raise ValueError(f"Unmatched session calendar for {ticker}; no price fills allowed")
        values = frame.loc[mask, ["adj_open", "adj_close"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Prices must be positive and finite for {ticker}")
        openings.append(values[:, 0])
        closes.append(values[:, 1])
    relevant = (target_index >= index[0]) & (target_index <= index[-1])
    rows = index.get_indexer(target_index[relevant])
    if (rows < 0).any():
        raise ValueError("An in-window execution target is not an observed trading session")
    schedule = np.full((len(index), len(tickers)), np.nan)
    schedule[rows] = weights[relevant]
    return tickers, index, np.column_stack(openings), np.column_stack(closes), schedule


def simulate_portfolio(prices, targets, start=None, end=None, cost_bps=5,
                       maintenance_margin=.30):
    """Simulate one shared account from NAV 1 and return (ledger, positions).

    Every target row states signed ETF fractions of *post-fee* account NAV;
    unspecified dates drift existing shares. Flat capital stays cash. A margin
    event occurs when NAV/total short market value is below maintenance_margin.
    Positions close at the final observed close, with one-way fees on all sides.
    """
    tickers, dates, opens, closes, schedule = _inputs(
        prices, targets, start, end, cost_bps, maintenance_margin)
    fee = float(cost_bps) / 10000
    count = len(tickers)
    shares = np.zeros(count)
    cash, previous_nav = 1.0, 1.0
    previous_close = opens[0]
    records, position_records = [], []
    for day, date in enumerate(dates):
        opening, closing = opens[day], closes[day]
        previous_shares = shares.copy()
        old = shares * opening
        opening_nav = float(cash + old.sum())
        _positive_nav(opening_nav, date, "open")
        overnight = previous_shares * (opening - previous_close)
        long_pnl = np.where(previous_shares > 0, overnight, 0.0)
        short_pnl = np.where(previous_shares < 0, overnight, 0.0)
        long_costs, short_costs = np.zeros(count), np.zeros(count)
        absolute_trades = np.zeros(count)
        open_trade, close_trade, rebalance_trade = np.zeros(count), np.zeros(count), np.zeros(count)
        forced_cover_open = forced_cover_close = forced_long_sale = 0.0
        margin_open = margin_close = False
        turnover, trade_events, terminal_cost = 0.0, 0, 0.0
        target_applied = False

        short_open = float(-np.minimum(old, 0).sum())
        if short_open > 0 and opening_nav / short_open < maintenance_margin - 1e-12:
            margin_open = True
            cash, shares, delta, forced_cover_open, sale = _margin_cover(
                cash, shares, opening, fee, date, "open margin")
            forced_long_sale += sale
            open_trade += delta
            absolute_trades += np.abs(delta)
            long_costs += fee * np.where(old > 0, np.abs(delta), 0.0)
            short_costs += fee * np.where(old < 0, np.abs(delta), 0.0)
            turnover += float(np.abs(delta).sum()) / opening_nav
            trade_events += int(np.count_nonzero(np.abs(delta) > 1e-12))
        elif np.isfinite(schedule[day]).all():
            weights = schedule[day]
            values = _post_fee_values(opening_nav, old, weights, fee)
            if values is None:
                raise AccountInsolvencyError(date, "rebalance fees", opening_nav - fee * np.abs(old).sum())
            delta = values - old
            long_turnover = np.abs(np.maximum(values, 0) - np.maximum(old, 0))
            short_turnover = np.abs(np.minimum(values, 0) - np.minimum(old, 0))
            long_costs += fee * long_turnover
            short_costs += fee * short_turnover
            cash = float(opening_nav - values.sum() - fee * np.abs(delta).sum())
            shares = values / opening
            rebalance_trade = delta.copy()
            open_trade += delta
            absolute_trades += np.abs(delta)
            turnover += float(np.abs(delta).sum()) / opening_nav
            trade_events += int(np.count_nonzero(np.abs(delta) > 1e-12))
            target_applied = True
        if cash < -1e-10:
            raise ValueError(f"Cash borrowing on {date.date()}: {cash}")
        cash = max(0.0, cash)
        after_open_shares = shares.copy()
        after_open_values = shares * opening
        post_trade_nav = float(cash + after_open_values.sum())
        _positive_nav(post_trade_nav, date, "after open trades")
        intraday = shares * (closing - opening)
        long_pnl += np.where(shares > 0, intraday, 0.0)
        short_pnl += np.where(shares < 0, intraday, 0.0)
        values = shares * closing
        close_nav = float(cash + values.sum())
        _positive_nav(close_nav, date, "close before liquidation")
        held_net = float(values.sum() / close_nav)
        held_gross = float(np.abs(values).sum() / close_nav)
        held_long = float(np.maximum(values, 0).sum() / close_nav)
        held_short = float(-np.minimum(values, 0).sum() / close_nav)
        short_close = float(-np.minimum(values, 0).sum())
        before_close_actions = values.copy()
        if short_close > 0 and close_nav / short_close < maintenance_margin - 1e-12:
            margin_close = True
            cash, shares, delta, forced_cover_close, sale = _margin_cover(
                cash, shares, closing, fee, date, "close margin")
            forced_long_sale += sale
            close_trade += delta
            absolute_trades += np.abs(delta)
            long_costs += fee * np.where(values > 0, np.abs(delta), 0.0)
            short_costs += fee * np.where(values < 0, np.abs(delta), 0.0)
            turnover += float(np.abs(delta).sum()) / close_nav
            trade_events += int(np.count_nonzero(np.abs(delta) > 1e-12))
        if day == len(dates) - 1:
            values = shares * closing
            before_terminal_nav = float(cash + values.sum())
            delta = -values
            liquidation_costs = fee * np.abs(delta)
            terminal_cost = float(liquidation_costs.sum())
            cash -= float(delta.sum()) + terminal_cost
            long_costs += fee * np.maximum(values, 0)
            short_costs += fee * -np.minimum(values, 0)
            close_trade += delta
            absolute_trades += np.abs(delta)
            turnover += float(np.abs(delta).sum()) / before_terminal_nav
            trade_events += int(np.count_nonzero(np.abs(delta) > 1e-12))
            shares = np.zeros(count)
        closing_values = shares * closing
        nav = float(cash + closing_values.sum())
        _positive_nav(nav, date, "after close actions")
        if cash < -1e-10:
            raise ValueError(f"Cash borrowing after liquidation on {date.date()}: {cash}")
        cash = max(0.0, cash)
        costs_by_asset = long_costs + short_costs
        costs = float(costs_by_asset.sum())
        net_long, net_short = long_pnl - long_costs, short_pnl - short_costs
        net_pnl = float(net_long.sum() + net_short.sum())
        if not np.isclose(nav - previous_nav, net_pnl, rtol=1e-10, atol=1e-11):
            raise ValueError(f"Portfolio P&L conservation failed on {date.date()}")
        records.append(dict(
            Date=date, nav=nav, cash=cash, opening_nav=opening_nav,
            post_trade_nav=post_trade_nav, close_nav_before_actions=close_nav,
            daily_return=nav / previous_nav - 1, costs=costs, turnover=turnover,
            trade_events=trade_events, target_scheduled=bool(np.isfinite(schedule[day]).all()),
            target_applied=target_applied, margin_call_open=margin_open,
            margin_call_close=margin_close, margin_calls=int(margin_open) + int(margin_close),
            forced_short_cover=forced_cover_open + forced_cover_close,
            forced_long_liquidation=forced_long_sale, terminal_cost=terminal_cost,
            held_exposure=held_net, held_gross_exposure=held_gross,
            held_long_exposure=held_long, held_short_exposure=held_short,
            net_exposure=float(closing_values.sum() / nav),
            gross_exposure=float(np.abs(closing_values).sum() / nav),
            long_notional=float(np.maximum(closing_values, 0).sum()),
            short_notional=float(-np.minimum(closing_values, 0).sum()),
            long_price_pnl=float(long_pnl.sum()), short_price_pnl=float(short_pnl.sum()),
            long_costs=float(long_costs.sum()), short_costs=float(short_costs.sum()),
            long_net_pnl=float(net_long.sum()), short_net_pnl=float(net_short.sum()),
            net_pnl=net_pnl, borrow_cost=0.0, cash_interest=0.0))
        for j, ticker in enumerate(tickers):
            position_records.append((
                date, ticker, previous_shares[j], old[j], after_open_shares[j],
                after_open_values[j], schedule[day, j] if np.isfinite(schedule[day, j]) else np.nan,
                after_open_values[j] / post_trade_nav, open_trade[j], rebalance_trade[j],
                before_close_actions[j], close_trade[j], shares[j], closing_values[j],
                costs_by_asset[j], absolute_trades[j], long_pnl[j], short_pnl[j],
                net_long[j], net_short[j]))
        previous_nav, previous_close = nav, closing
    ledger = pd.DataFrame(records).set_index("Date")
    positions = pd.DataFrame(position_records, columns=[
        "Date", "ticker", "opening_shares", "opening_value", "after_open_shares",
        "after_open_value", "scheduled_weight", "after_open_weight", "open_trade_value",
        "rebalance_trade_value", "close_value_before_actions", "close_trade_value",
        "closing_shares", "closing_value", "costs", "absolute_trade_value",
        "long_price_pnl", "short_price_pnl", "long_net_pnl", "short_net_pnl"])
    ledger.attrs.update(borrow_interest_assumption=0.0, cash_yield_assumption=0.0,
                        margin_rule="pooled equity/short notional; minimum forced long sale if needed",
                        target_basis="post-fee NAV; gross at targets <=1; intervening exposure drifts")
    return ledger, positions
