"""Read-only statistics for the saved payoff-plus-context long/cash strategy.

Only pandas, NumPy and the Python standard library are required. This module
does not load models, fit anything, fetch data, resimulate positions, or write
files. Pass a complete saved account ledger for full-study episode statistics.

A completed episode starts with an observed transition from cash into positive
stock exposure and ends with a full exit. Resizing is part of that same episode.
Its net P&L is all sale proceeds minus all purchase notional minus every fee,
including the entry, resizes, exit and any explicit final-close liquidation.
The episode return divides this P&L by account NAV immediately before entry.
That is an account contribution, not a forecast-label return or a return on the
initial stock purchase alone. Zero cash yield and no external flows are assumed.

Holding sessions count positive post-open exposure: entry session included,
exit-at-open session excluded, terminal-close session included. Carry-in and
still-open episodes are excluded from completed-episode win/loss statistics.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

VARIANT = "payoff_context"
_TOL = 1e-10
_REQUIRED = {
    "nav", "cash", "shares", "weight", "held_exposure", "costs", "turnover",
    "trade_value", "terminal_sale", "terminal_cost", "opening_nav", "daily_return",
}
_EPISODE_COLUMNS = [
    "episode_id", "entry_date", "exit_date", "entry_clock", "exit_clock",
    "entry_account_nav", "exit_account_nav", "entry_post_fee_open_weight",
    "buy_notional", "sale_notional", "fees", "gross_pnl_before_fees", "net_pnl",
    "net_account_return", "outcome", "holding_sessions", "calendar_days",
    "mean_held_exposure_during_episode", "one_way_trade_events", "resize_events",
    "terminal_liquidation", "pnl_nav_reconciliation_error",
]

STATISTIC_DEFINITIONS = {
    "win_rate": "Net-winning completed episodes / all completed episodes, including flat episodes in the denominator.",
    "win_rate_excluding_flat": "Net-winning completed episodes / (net-winning + net-losing completed episodes).",
    "net_profit_factor": "Sum of positive completed-episode NET P&L / absolute sum of negative completed-episode NET P&L; fees and resizing included.",
    "payoff_ratio_account_return": "Mean positive completed-episode account return / absolute mean negative completed-episode account return.",
    "episode_net_account_return": "Episode net cashflow P&L / account NAV immediately before its observed entry trade.",
    "holding_sessions": "Sessions with positive post-open exposure; exit-open day excluded, terminal-close day included.",
    "decision_horizon_mean_sessions": "Arithmetic mean of the saved prediction horizon over decision weeks; it is not the holding duration.",
    "mean_target_weight_all_decisions": "Mean saved target stock weight over every weekly decision, including CASH decisions.",
    "mean_target_weight_when_long": "Mean saved target stock weight conditional on a LONG signal; it is not daily realized exposure.",
    "mean_realized_held_exposure": "Mean daily saved held_exposure, including cash days and exposure before terminal liquidation.",
    "calibration_fallback_rate": "Fraction of decision weeks without an eligible expected-payoff estimate; those weeks retain the original signal.",
    "cash_accounting": "Zero cash yield, no external flows, fractional adjusted units, no extra dividend credits; public-provider adjustment limitations remain.",
}


def _ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    data = ledger.copy()
    if "Date" in data:
        data = data.set_index("Date")
    data.index = pd.DatetimeIndex(pd.to_datetime(data.index), name="Date")
    missing = _REQUIRED.difference(data.columns)
    if missing:
        raise ValueError(f"Missing ledger columns: {sorted(missing)}")
    if data.empty or not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("A nonempty ledger with unique increasing dates is required")
    values = data[list(_REQUIRED)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (data.nav <= 0).any() or (data.opening_nav <= 0).any():
        raise ValueError("Ledger values must be finite with strictly positive NAV")
    if (data.cash < -_TOL).any() or (data.shares < -_TOL).any() or (data.costs < -_TOL).any():
        raise ValueError("Long/cash ledger contains negative cash, shares or fees")
    if not data.held_exposure.between(-_TOL, 1 + _TOL).all():
        raise ValueError("Realized exposure is outside long/cash bounds")
    if (data.terminal_cost > data.costs + _TOL).any() or (data.terminal_sale < -_TOL).any():
        raise ValueError("Invalid terminal liquidation fields")
    return data


def _forecast(forecasts: pd.DataFrame | None) -> pd.DataFrame:
    if forecasts is None:
        return pd.DataFrame()
    data = forecasts.copy()
    if "variant" in data:
        data = data.loc[data.variant.eq(VARIANT)].copy()
    if "ticker" in data and data.ticker.nunique() > 1:
        raise ValueError("Pass one ETF at a time; these accounts are not a portfolio")
    if data.empty:
        return data
    required = {"decision_date", "entry_date", "horizon", "signal", "target_weight"}
    if required.difference(data):
        raise ValueError(f"Missing forecast columns: {sorted(required.difference(data))}")
    for name in ["decision_date", "entry_date", "fit_cutoff", "train_latest_label",
                 "calibration_cutoff", "calibration_latest_label", "label_end"]:
        if name in data:
            data[name] = pd.to_datetime(data[name])
    data = data.sort_values("decision_date")
    if data.decision_date.duplicated().any() or data.entry_date.duplicated().any():
        raise ValueError("One forecast/execution per decision is required")
    if not (data.entry_date > data.decision_date).all():
        raise ValueError("Forecast execution must follow the decision close")
    if not data.target_weight.between(0, 1).all():
        raise ValueError("Forecast target weight is outside long/cash bounds")
    if not data.signal.isin([0, 1, False, True]).all():
        raise ValueError("LONG signal must be binary")
    if (data.loc[data.signal.eq(0), "target_weight"].abs() > _TOL).any():
        raise ValueError("CASH signal has a positive target stock weight")
    return data


def _episode_tables(ledger: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    data = _ledger(ledger)
    completed: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    first = data.iloc[0]
    # cash_after_close = previous_cash - open_trade - total_fees + terminal_sale
    previous_cash = float(first.cash - first.terminal_sale + first.trade_value + first.costs)
    initial_old_stock = float(first.opening_nav - previous_cash)
    if initial_old_stock < -_TOL:
        raise ValueError("Initial cash/position state is inconsistent")

    def begin(date, row, carried_in=False):
        return {"entry_date": pd.NaT if carried_in else date,
                "observed_start": date, "entry_account_nav": None if carried_in else float(row.opening_nav),
                "carried_in": carried_in, "buy_notional": 0., "sale_notional": 0., "fees": 0.,
                "holding_sessions": 0, "exposure_sum": 0., "one_way_trade_events": 0,
                "resize_events": 0, "entry_post_fee_open_weight": None}

    def finish(date, row, terminal):
        nonlocal current
        assert current is not None
        if current["carried_in"]:
            excluded.append({"status": "carry_in_closed", "observed_start": current["observed_start"],
                             "observed_end": date, "fees_in_observed_window": current["fees"]})
            current = None
            return
        net_pnl = current["sale_notional"] - current["buy_notional"] - current["fees"]
        nav_change = float(row.nav) - current["entry_account_nav"]
        error = net_pnl - nav_change
        if not np.isclose(net_pnl, nav_change, rtol=1e-8, atol=_TOL):
            raise ValueError("Episode cashflows do not reconcile NAV; zero cash yield/no external flows are required")
        threshold = _TOL * max(1., current["entry_account_nav"])
        outcome = "win" if net_pnl > threshold else "loss" if net_pnl < -threshold else "flat"
        completed.append({"episode_id": len(completed) + 1,
                          "entry_date": current["entry_date"], "exit_date": date,
                          "entry_clock": "open", "exit_clock": "terminal_close" if terminal else "open",
                          "entry_account_nav": current["entry_account_nav"], "exit_account_nav": float(row.nav),
                          "entry_post_fee_open_weight": current["entry_post_fee_open_weight"],
                          "buy_notional": current["buy_notional"], "sale_notional": current["sale_notional"],
                          "fees": current["fees"], "gross_pnl_before_fees": net_pnl + current["fees"],
                          "net_pnl": net_pnl, "net_account_return": net_pnl / current["entry_account_nav"],
                          "outcome": outcome, "holding_sessions": current["holding_sessions"],
                          "calendar_days": (date - current["entry_date"]).days,
                          "mean_held_exposure_during_episode": current["exposure_sum"] / current["holding_sessions"] if current["holding_sessions"] else 0.,
                          "one_way_trade_events": current["one_way_trade_events"], "resize_events": current["resize_events"],
                          "terminal_liquidation": bool(terminal), "pnl_nav_reconciliation_error": error})
        current = None

    for date, row in data.iterrows():
        old_stock = float(row.opening_nav - previous_cash)
        stock_after_open = old_stock + float(row.trade_value)
        scale = max(1., float(row.opening_nav))
        active_before = old_stock > _TOL * scale
        active_after_open = stock_after_open > _TOL * scale
        if stock_after_open < -_TOL * scale:
            raise ValueError("Open trade oversells the stock position")
        if active_after_open != (row.held_exposure > _TOL):
            raise ValueError("Open position and held_exposure disagree")
        expected_cash = previous_cash - row.trade_value - row.costs + row.terminal_sale
        if not np.isclose(expected_cash, row.cash, rtol=1e-9, atol=_TOL):
            raise ValueError("Daily cashflow conservation failed")
        if current is None and active_before:
            if date != data.index[0]:
                raise ValueError("Unexplained carry-in stock position")
            current = begin(date, row, carried_in=True)
        entered = current is None and active_after_open
        if entered:
            current = begin(date, row)
            open_cost = float(row.costs - row.terminal_cost)
            current["entry_post_fee_open_weight"] = stock_after_open / (row.opening_nav - open_cost)
        if current is not None:
            trade = float(row.trade_value)
            current["buy_notional"] += max(trade, 0.)
            current["sale_notional"] += max(-trade, 0.)
            current["fees"] += float(row.costs - row.terminal_cost)
            current["one_way_trade_events"] += int(abs(trade) > _TOL)
            if active_before and active_after_open and not entered and abs(trade) > _TOL:
                current["resize_events"] += 1
            if active_after_open:
                current["holding_sessions"] += 1
                current["exposure_sum"] += float(row.held_exposure)
            else:
                if row.terminal_sale > _TOL:
                    raise ValueError("Terminal sale with no stock exposure")
                finish(date, row, terminal=False)
        elif abs(row.trade_value) > _TOL or row.costs > _TOL:
            raise ValueError("Trade or cost outside an identifiable episode")
        if row.terminal_sale > _TOL:
            if current is None or not active_after_open or row.shares > _TOL:
                raise ValueError("Invalid saved terminal liquidation")
            current["sale_notional"] += float(row.terminal_sale)
            current["fees"] += float(row.terminal_cost)
            current["one_way_trade_events"] += 1
            finish(date, row, terminal=True)
        previous_cash = float(row.cash)
    if current is not None:
        excluded.append({"status": "carry_in_open" if current["carried_in"] else "open",
                         "observed_start": current["observed_start"], "observed_end": data.index[-1],
                         "fees_in_observed_window": current["fees"],
                         "marked_account_pnl_since_observed_entry": None if current["carried_in"] else float(data.nav.iloc[-1] - current["entry_account_nav"])})
    return pd.DataFrame(completed, columns=_EPISODE_COLUMNS), excluded


def completed_episodes(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return observed-entry, fully exited episodes, including every resize fee."""
    return _episode_tables(ledger)[0]


def _ratio(numerator: float, denominator: float) -> tuple[float | None, str]:
    if denominator > _TOL:
        return float(numerator / denominator), "finite"
    return (None, "infinite_no_losses") if numerator > _TOL else (None, "undefined_no_profit_or_loss")


def summary_statistics(ledger: pd.DataFrame, forecasts: pd.DataFrame | None = None) -> dict[str, Any]:
    """Return portable episode, account and decision statistics for one ETF."""
    data = _ledger(ledger)
    forecasts = _forecast(forecasts)
    episodes, excluded = _episode_tables(data)
    wins = episodes.loc[episodes.outcome.eq("win")]
    losses = episodes.loc[episodes.outcome.eq("loss")]
    flats = episodes.loc[episodes.outcome.eq("flat")]
    count = len(episodes)
    positive_pnl = float(wins.net_pnl.sum())
    negative_pnl = float(-losses.net_pnl.sum())
    factor, factor_status = _ratio(positive_pnl, negative_pnl)
    average_win = float(wins.net_account_return.mean()) if len(wins) else None
    average_loss = float(losses.net_account_return.mean()) if len(losses) else None
    payoff, payoff_status = _ratio(average_win or 0., -(average_loss or 0.))
    initial_nav = float(data.nav.iloc[0] / (1 + data.daily_return.iloc[0]))
    years = len(data) / 252
    volatility = float(data.daily_return.std(ddof=1) * np.sqrt(252)) if len(data) > 1 else 0.
    sharp = float(data.daily_return.mean() * 252 / volatility) if volatility > 0 else 0.
    result = {
        "variant": VARIANT, "start": str(data.index[0].date()), "end": str(data.index[-1].date()), "sessions": len(data),
        "completed_episodes": count, "net_wins": len(wins), "net_losses": len(losses), "net_flat": len(flats),
        "win_rate": len(wins) / count if count else None,
        "win_rate_excluding_flat": len(wins) / (len(wins) + len(losses)) if len(wins) + len(losses) else None,
        "sum_positive_episode_net_pnl": positive_pnl, "sum_negative_episode_net_pnl_abs": negative_pnl,
        "net_profit_factor": factor, "net_profit_factor_status": factor_status,
        "mean_win_account_return": average_win, "mean_loss_account_return": average_loss,
        "payoff_ratio_account_return": payoff, "payoff_ratio_status": payoff_status,
        "mean_episode_net_account_return": float(episodes.net_account_return.mean()) if count else None,
        "mean_holding_sessions": float(episodes.holding_sessions.mean()) if count else None,
        "median_holding_sessions": float(episodes.holding_sessions.median()) if count else None,
        "max_holding_sessions": int(episodes.holding_sessions.max()) if count else None,
        "forced_terminal_episode_count": int(episodes.terminal_liquidation.sum()) if count else 0,
        "excluded_carry_in_episodes": sum(item["status"].startswith("carry_in") for item in excluded),
        "excluded_open_episodes": sum(item["status"].endswith("open") for item in excluded),
        "complete_episode_net_pnl_sum": float(episodes.net_pnl.sum()),
        "complete_episode_fees": float(episodes.fees.sum()), "total_fees_normalized_currency": float(data.costs.sum()),
        "all_account_pnl_explained_by_completed_episodes": not excluded and bool(np.isclose(episodes.net_pnl.sum(), data.nav.iloc[-1] - initial_nav, rtol=1e-8, atol=_TOL)),
        "total_return": float(data.nav.iloc[-1] / initial_nav - 1),
        "cagr": float((data.nav.iloc[-1] / initial_nav) ** (1 / years) - 1),
        "annualized_volatility": volatility, "net_sharpe_zero_rf": sharp,
        "max_drawdown": float((data.nav / data.nav.cummax().clip(lower=initial_nav) - 1).min()),
        "annual_turnover": float(data.turnover.sum() / years),
        "one_way_trade_events": int((data.trade_value.abs() > _TOL).sum() + (data.terminal_sale > _TOL).sum()),
        "mean_realized_held_exposure": float(data.held_exposure.mean()),
        "fraction_sessions_with_stock_exposure": float((data.held_exposure > _TOL).mean()),
        "cash_yield_assumed": 0.,
    }
    result.update(_decision_facts(forecasts))
    if len(forecasts) and "ticker" in forecasts:
        result["ticker"] = str(forecasts.ticker.iloc[0])
    return result


def _decision_facts(forecasts: pd.DataFrame) -> dict[str, Any]:
    if forecasts.empty:
        return {"decision_weeks": 0, "decision_horizon_mean_sessions": None,
                "long_signal_rate": None, "calibration_fallback_rate": None,
                "mean_target_weight_all_decisions": None, "mean_target_weight_when_long": None}
    active = forecasts.signal.eq(1)
    eligible = forecasts.expected_payoff.notna() if "expected_payoff" in forecasts else pd.Series(False, index=forecasts.index)
    return {"decision_weeks": len(forecasts),
            "decision_horizon_mean_sessions": float(forecasts.horizon.mean()),
            "decision_horizon_median_sessions": float(forecasts.horizon.median()),
            "decision_horizon_counts": {str(int(key)): int(value) for key, value in forecasts.horizon.value_counts().sort_index().items()},
            "long_signal_rate": float(active.mean()),
            "calibration_eligible_rate": float(eligible.mean()), "calibration_fallback_rate": float((~eligible).mean()),
            "mean_target_weight_all_decisions": float(forecasts.target_weight.mean()),
            "mean_target_weight_when_long": float(forecasts.loc[active, "target_weight"].mean()) if active.any() else None,
            "minimum_target_weight_when_long": float(forecasts.loc[active, "target_weight"].min()) if active.any() else None,
            "maximum_target_weight_when_long": float(forecasts.loc[active, "target_weight"].max()) if active.any() else None}


def annual_model_statistics(forecasts: pd.DataFrame, ledger: pd.DataFrame,
                            feature_usage: pd.DataFrame | None = None) -> pd.DataFrame:
    """Describe annual saved model decisions; calendar exposure is separately named.

    Forecast groups use the model's outer/decision year. A December decision can
    execute in January. Daily realized exposure is grouped by calendar session
    year, so its sampling clock differs and is made explicit in column names.
    Fitted factor importance describes tree usage, not causal contribution.
    """
    forecasts = _forecast(forecasts)
    data = _ledger(ledger)
    if forecasts.empty:
        return pd.DataFrame()
    usage = feature_usage.copy() if feature_usage is not None else pd.DataFrame()
    if len(usage) and "model" in usage:
        usage = usage.loc[usage.model.eq("context")]
    years = forecasts.outer_year if "outer_year" in forecasts else forecasts.decision_date.dt.year
    records = []
    for year, group in forecasts.groupby(years):
        session = data.loc[data.index.year == int(year)]
        record = {"model_year": int(year), "first_decision": group.decision_date.min(),
                  "last_decision": group.decision_date.max(), "first_execution": group.entry_date.min(),
                  "last_execution": group.entry_date.max(), **_decision_facts(group),
                  "horizons_sessions": ",".join(str(int(x)) for x in sorted(group.horizon.unique())),
                  "target_types": ",".join(sorted(group.target.unique())) if "target" in group else "",
                  "base_factor_recipes": ",".join(sorted(group.base_recipe.unique())) if "base_recipe" in group else "",
                  "sizing_rules": ",".join(sorted(group.base_sizing.unique())) if "base_sizing" in group else "",
                  "realized_exposure_calendar_year_mean": float(session.held_exposure.mean()) if len(session) else None,
                  "calendar_sessions": len(session),
                  "forecast_feature_count_min": int(group.feature_count.min()) if "feature_count" in group else None,
                  "forecast_feature_count_max": int(group.feature_count.max()) if "feature_count" in group else None}
        if "train_rows" in group:
            record.update(training_rows_min=int(group.train_rows.min()), training_rows_max=int(group.train_rows.max()))
        if "fit_cutoff" in group and "train_latest_label" in group:
            record.update(first_fit_cutoff=group.fit_cutoff.min(), last_fit_cutoff=group.fit_cutoff.max(),
                          latest_training_label=group.train_latest_label.max(),
                          training_labels_strictly_matured=bool((group.train_latest_label < group.fit_cutoff).all()))
        if "calibration_latest_label" in group and "calibration_cutoff" in group:
            eligible = group.loc[group.expected_payoff.notna()]
            record.update(latest_calibration_label=eligible.calibration_latest_label.max() if len(eligible) else pd.NaT,
                          calibration_labels_strictly_matured=bool((eligible.calibration_latest_label < eligible.calibration_cutoff).all()),
                          calibration_oof_rows_min=int(group.oof_rows.min()) if "oof_rows" in group else None,
                          calibration_oof_rows_max=int(group.oof_rows.max()) if "oof_rows" in group else None)
        if len(usage):
            selected = usage.loc[usage.year.eq(int(year)) & usage.importance.gt(0)].sort_values("importance", ascending=False)
            record["active_tree_feature_count"] = len(selected)
            record["top_five_tree_features"] = "; ".join(f"{row.feature} ({row.importance:.3f})" for row in selected.head(5).itertuples())
            record["tree_usage_families"] = ",".join(sorted({"context" if value.startswith("ctx_") else "relationships" if value.startswith("rel_") else "technical" for value in selected.feature}))
        records.append(record)
    return pd.DataFrame(records)
