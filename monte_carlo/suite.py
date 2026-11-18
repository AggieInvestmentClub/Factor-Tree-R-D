"""Standalone, batched score-permutation backtests and a single HTML report."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np


@dataclass(frozen=True)
class Data:
    dates: np.ndarray             # (days,) observed sessions, increasing
    names: tuple[str, ...]        # (assets,) stable security identifiers
    close: np.ndarray             # (days, assets) total-return adjusted closes
    eligible: np.ndarray          # (days, assets) decision-known booleans
    decision_rows: np.ndarray     # (forecasts,) indices into dates
    scores: np.ndarray            # (forecasts, assets) saved, causal model scores
    groups: np.ndarray            # (forecasts, assets) decision-known risk-group IDs


# ============================================================================
# DATA INGEST — CONNECT YOUR DATA SOURCE HERE. INTENTIONALLY LEFT BLANK.
# ============================================================================
DATA_SOURCE = None


def ingest_data() -> Data:
    """Load ONLY the seven Data fields above from your chosen source.

    Each group ID should identify comparable sector/beta/volatility/size/
    liquidity strata, constructed using information known at that decision.
    Supply actual historical eligibility and issue identifiers. Do not drop
    incomplete names, fill prices, or select a universe using future outcomes.
    """
    raise NotImplementedError("Connect your source in the clearly marked DATA INGEST section of suite.py")


# ============================================================================
# STRATEGY — EDIT THESE RULES TO MATCH THE STRATEGY YOU WANT TO TEST.
# ============================================================================
@dataclass(frozen=True)
class Strategy:
    tail_fraction: float = 0.20
    gross_exposure: float = 1.0
    cost_bps: float = 2.0
    volatility_window: int = 63
    stop_multiple: float = 2.0
    stop_horizon: int = 21
    stop_floor: float = 0.03
    stop_cap: float = 0.20


def target_weights(scores: np.ndarray, eligible: np.ndarray, rule: Strategy) -> np.ndarray:
    """Equal-weight top/bottom tails; half the gross budget on each side.

    This explicit reference strategy is independent of any repository model.
    Replace this function to test another allocation rule. All variations and
    the actual strategy must use the same function and execution assumptions.
    """
    count = int(eligible.sum())
    selected = max(1, int(count * rule.tail_fraction))
    order = np.argsort(np.where(eligible, scores, np.inf), axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(len(eligible)), order.shape), axis=1)
    long = eligible & (rank >= count - selected) & (rank < count)
    short = eligible & (rank < selected)
    return (long.astype(float) - short) * (rule.gross_exposure / (2 * selected))


def validate(data: Data, rule: Strategy, block_forecasts: int) -> None:
    days, assets = data.close.shape
    queries = len(data.decision_rows)
    if (data.eligible.shape != (days, assets) or data.scores.shape != (queries, assets)
            or data.groups.shape != data.scores.shape or len(data.names) != assets or len(data.dates) != days):
        raise ValueError("Data shapes disagree; see Data's field descriptions")
    dates = np.asarray(data.dates, dtype="datetime64[ns]")
    if np.isnat(dates).any() or not np.all(dates[1:] > dates[:-1]) or len(set(data.names)) != assets:
        raise ValueError("Unique security IDs and increasing, nonmissing dates required")
    if data.eligible.dtype != np.bool_ or not np.issubdtype(data.groups.dtype, np.integer):
        raise ValueError("Eligibility must be boolean; matching groups must be integer IDs")
    rows = data.decision_rows
    if (not queries or not np.issubdtype(rows.dtype, np.integer) or not np.all(np.diff(rows) > 0)
            or rows[0] < rule.volatility_window or rows[0] >= days - 2 or rows[-1] >= days - 1):
        raise ValueError("Forecast rows must be increasing, follow volatility warm-up, "
                         "have a next close, and leave at least two evaluation sessions")
    active = data.eligible[rows]
    if (active.sum(axis=1) < 2).any() or not np.isfinite(data.scores[active]).all() or (data.groups[active] < 0).any():
        raise ValueError("Every forecast needs at least two eligible names, finite scores and known matching groups")
    if (not 0 < rule.tail_fraction <= .5 or not 0 < rule.gross_exposure <= 1
            or not 0 <= rule.cost_bps < 100 or rule.volatility_window < 2
            or rule.stop_horizon < 1 or rule.stop_multiple <= 0
            or not 0 < rule.stop_floor <= rule.stop_cap < 1 or block_forecasts < 1):
        raise ValueError("Invalid strategy or permutation settings")
    # Missing inactive prices are allowed, but no required quote is imputed.
    required = data.eligible.copy()
    required[rows + 1] |= active
    if not np.all(np.isfinite(data.close[required]) & (data.close[required] > 0)):
        raise ValueError("Missing/nonpositive eligible or execution prices; no filling or name exclusion is allowed")
    for query, mask in enumerate(active):
        _, counts = np.unique(data.groups[query, mask], return_counts=True)
        if counts[counts > 1].sum() < .2 * mask.sum():
            raise ValueError(f"Forecast {query}: fewer than 20% of eligible names can be permuted within their groups")


def volatility(close: np.ndarray, window: int) -> np.ndarray:
    """Past-only sample volatility. Incomplete windows stay missing."""
    returns = close[1:] / close[:-1] - 1
    windows = np.lib.stride_tricks.sliding_window_view(returns, window, axis=0)
    result = np.full(close.shape, np.nan)
    result[window:] = windows.std(axis=-1, ddof=1)
    return result


def permute_scores(data: Data, query: int, previous: np.ndarray, first_draw: int,
                   seed: int, block_forecasts: int) -> tuple[np.ndarray, np.ndarray]:
    """Persistent destination/source assignments, using only this query's groups.

    Random streams are keyed by forecast/group and advanced by draw index.
    Results therefore do not depend on batch size. Identity permutations are
    allowed. Rematching never looks ahead at later membership or prices.
    """
    mask = data.eligible[data.decision_rows[query]]
    codes = data.groups[query]
    mapping = np.full_like(previous, -1)
    batch = len(previous)
    for group_number, code in enumerate(np.unique(codes[mask])):
        columns = np.flatnonzero(mask & (codes == code))
        if len(columns) == 1:
            mapping[:, columns] = columns
            continue
        old = previous[:, columns]
        keep = (old >= 0) & mask[np.maximum(old, 0)] & (codes[np.maximum(old, 0)] == code)
        if query % block_forecasts == 0:
            keep[:] = False
        used = np.zeros((batch, len(columns)), dtype=bool)
        row, col = np.nonzero(keep)
        used[row, np.searchsorted(columns, old[row, col])] = True
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, query, group_number])))
        generator.bit_generator.advance(first_draw * len(columns))
        order = np.argsort(generator.random(used.shape) + used * 2, axis=1)
        remaining_rank = np.maximum(np.cumsum(~keep, axis=1) - 1, 0)
        donors = columns[np.take_along_axis(order, remaining_rank, axis=1)]
        mapping[:, columns] = np.where(keep, old, donors)
    scores = np.zeros(mapping.shape)
    scores[:, mask] = data.scores[query, mapping[:, mask]]
    return scores, mapping


def backtest(data: Data, rule: Strategy, sigma: np.ndarray, batch: int, *,
             first_draw: int = 0, seed: int = 42, block_forecasts: int = 3,
             randomized: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch actual share/cash accounts: close signals, next-close execution.

    Stops close only the affected holdings, without rescaling survivors.
    New forecasts permit reentry. The final close liquidates all positions.
    """
    assets = len(data.names)
    shares = np.zeros((batch, assets))
    entry = np.zeros_like(shares)
    threshold = np.zeros_like(shares)
    blocked = np.zeros_like(shares, dtype=bool)
    exits = np.zeros_like(blocked)
    mapping = np.full(shares.shape, -1, dtype=np.int64)
    cash = np.ones(batch)
    turnover = np.zeros(batch)
    costs = np.zeros(batch)
    stops = np.zeros(batch, dtype=np.int64)
    start = int(data.decision_rows[0]) + 1
    equity = np.empty((len(data.dates) - start, batch))
    fee = rule.cost_bps / 10000

    def weights(query):
        nonlocal mapping
        if randomized:
            scores, mapping = permute_scores(data, query, mapping, first_draw, seed, block_forecasts)
        else:
            scores = data.scores[query][None, :]
        return target_weights(scores, data.eligible[data.decision_rows[query]], rule)

    pending = weights(0)
    next_query = 1
    for day in range(start, len(data.dates)):
        price = data.close[day]
        held = shares != 0
        if np.any(held & (~np.isfinite(price) | (price <= 0))):
            raise ValueError(f"Missing held quote at {data.dates[day]}; the entire experiment stops")
        values = np.multiply(shares, price, out=np.zeros_like(shares), where=held)
        nav = cash + values.sum(axis=1)
        if not np.isfinite(nav).all() or (nav <= 0).any():
            raise ValueError("Insolvent/invalid variation; never discard it and draw a replacement")
        if day == len(data.dates) - 1:
            # No final-day new entry; all remaining shares are liquidated once.
            traded = np.abs(values).sum(axis=1)
            costs += fee * traded / nav
            turnover += traded / nav
            equity[day - start] = nav - fee * traded
            break
        if pending is not None:
            needed = (pending != 0).any(axis=0)
            if not np.all(np.isfinite(price[needed]) & (price[needed] > 0)):
                raise ValueError(f"Missing execution quote at {data.dates[day]}")
            after_fee = nav.copy()
            for _ in range(6):
                traded = np.abs(pending * after_fee[:, None] - values).sum(axis=1)
                after_fee = nav - fee * traded
            target = pending * after_fee[:, None]
            delta = target - values
            paid = fee * np.abs(delta).sum(axis=1)
            if not np.allclose(after_fee, nav - paid, rtol=0, atol=1e-10):
                raise ValueError("Post-fee allocation failed to converge")
            new_shares = np.divide(target, price, out=np.zeros_like(shares), where=pending != 0)
            fresh = (new_shares != 0) & (np.sign(new_shares) != np.sign(shares))
            if not np.isfinite(sigma[day, fresh.any(axis=0)]).all():
                raise ValueError(f"Incomplete entry-volatility history at {data.dates[day]}")
            added = (np.sign(new_shares) == np.sign(shares)) & (np.abs(new_shares) > np.abs(shares)) & ~fresh
            addition = np.maximum(np.abs(new_shares) - np.abs(shares), 0)
            added_value = np.abs(shares) * entry + np.multiply(addition, price,
                             out=np.zeros_like(shares), where=addition > 0)
            averaged = np.divide(added_value, np.abs(new_shares), out=np.zeros_like(entry), where=new_shares != 0)
            entry = np.where(fresh, price, np.where(added, averaged, entry))
            new_threshold = np.clip(rule.stop_multiple * sigma[day] * np.sqrt(rule.stop_horizon), rule.stop_floor, rule.stop_cap)
            threshold = np.where(fresh, new_threshold, threshold)
            entry[new_shares == 0] = 0
            cash -= delta.sum(axis=1) + paid
            shares = new_shares
            turnover += np.abs(delta).sum(axis=1) / nav
            costs += paid / nav
            pending = None
        elif exits.any():
            sold = np.where(exits, values, 0)
            traded = np.abs(sold).sum(axis=1)
            cash += sold.sum(axis=1) - fee * traded
            shares[exits] = 0
            entry[exits] = 0
            turnover += traded / nav
            costs += fee * traded / nav
        current = np.multiply(shares, price, out=np.zeros_like(shares), where=shares != 0)
        equity[day - start] = cash + current.sum(axis=1)
        if next_query < len(data.decision_rows) and day == data.decision_rows[next_query]:
            pending = weights(next_query)
            next_query += 1
            blocked[:] = False
            exits[:] = False
        else:
            move = np.divide(price, entry, out=np.ones_like(entry), where=shares != 0) - 1
            hit = (shares != 0) & ~blocked & (np.sign(shares) * move <= -threshold)
            exits = hit | ((shares != 0) & ~data.eligible[day])
            blocked |= exits
            stops += hit.sum(axis=1)
    if not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("Nonpositive/invalid account equity; no variation is omitted")
    return equity, turnover, costs, stops


def metrics(equity: np.ndarray, turnover: np.ndarray, costs: np.ndarray, stops: np.ndarray) -> dict[str, np.ndarray]:
    path = np.vstack([np.ones((1, equity.shape[1])), equity])
    returns = path[1:] / path[:-1] - 1
    years = len(equity) / 252
    vol = returns.std(axis=0, ddof=1) * np.sqrt(252)
    sharpe = np.divide(returns.mean(axis=0) * 252, vol, out=np.full_like(vol, np.nan), where=vol > 0)
    return {"total_return": equity[-1] - 1, "cagr": np.expm1(np.log(equity[-1]) / years),
            "annual_log_growth": np.log(equity[-1]) / years, "sharpe": sharpe,
            "annual_volatility": vol, "max_drawdown": (path / np.maximum.accumulate(path, axis=0) - 1).min(axis=0),
            "annual_turnover": turnover / years, "annual_cost_fraction": costs / years, "stop_count": stops}


@dataclass
class Result:
    actual: dict[str, np.ndarray]
    variations: dict[str, np.ndarray]
    actual_equity: np.ndarray
    variation_equity: np.ndarray
    seconds: float


def run(data: Data, rule: Strategy = Strategy(), *, variations: int = 10000,
        seed: int = 42, batch_size: int = 128, block_forecasts: int = 3) -> Result:
    if variations < 1 or batch_size < 1 or seed < 0:
        raise ValueError("Positive variations/batch size and a nonnegative seed required")
    validate(data, rule, block_forecasts)
    begin = time.perf_counter()
    sigma = volatility(data.close, rule.volatility_window)
    actual_path, actual_turn, actual_costs, actual_stops = backtest(data, rule, sigma, 1, randomized=False)
    actual = metrics(actual_path, actual_turn, actual_costs, actual_stops)
    paths = np.empty((len(actual_path), variations), dtype=np.float32)
    all_metrics = {key: np.empty(variations) for key in actual}
    for first in range(0, variations, batch_size):
        count = min(batch_size, variations - first)
        path, turn, cost, stop = backtest(data, rule, sigma, count, first_draw=first,
                                        seed=seed, block_forecasts=block_forecasts)
        paths[:, first:first + count] = path
        for key, value in metrics(path, turn, cost, stop).items():
            all_metrics[key][first:first + count] = value
    return Result(actual, all_metrics, actual_path[:, 0], paths, time.perf_counter() - begin)


def report(result: Result, data: Data, rule: Strategy, output: Path, *, seed: int, block_forecasts: int) -> None:
    """Write only report.html and variations.csv; plots are embedded in HTML."""
    import csv
    from dataclasses import asdict
    from html import escape
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    keys = list(result.variations)
    with (output / "variations.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variation", *keys])
        writer.writerows((index + 1, *(result.variations[key][index] for key in keys))
                         for index in range(len(result.variation_equity[0])))
    observed = float(result.actual["annual_log_growth"][0])
    values = result.variations["annual_log_growth"]
    exceed = int(np.count_nonzero(values >= observed - 1e-12 * max(1., abs(observed))))
    fraction = (exceed + 1) / (len(values) + 1)
    labels = {"total_return": "Total return", "cagr": "CAGR", "annual_log_growth": "Annual log growth",
              "sharpe": "Sharpe (zero cash rate)", "annual_volatility": "Annual volatility",
              "max_drawdown": "Maximum drawdown", "annual_turnover": "Annual turnover",
              "annual_cost_fraction": "Annual trading-cost fraction", "stop_count": "Stop triggers"}

    def formatted(key, value):
        if not np.isfinite(value):
            return "Undefined"
        return f"{value:.2f}" if key in {"sharpe", "annual_turnover", "stop_count"} else f"{value:.2%}"

    rows = []
    for key in keys:
        finite = result.variations[key][np.isfinite(result.variations[key])]
        percentiles = np.quantile(finite, [.05, .5, .95]) if len(finite) else np.full(3, np.nan)
        rows.append("<tr><th>" + labels[key] + "</th>" + "".join(
            f"<td>{formatted(key, value)}</td>" for value in [result.actual[key][0], *percentiles]) + "</tr>")
    dates = np.asarray(data.dates, dtype="datetime64[D]")[int(data.decision_rows[0]) + 1:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    bands = np.quantile(result.variation_equity, [.05, .5, .95], axis=1)
    axes[0].fill_between(dates, bands[0], bands[2], color="#dbeafe", label="Randomized 5th–95th percentile")
    axes[0].plot(dates, bands[1], color="#64748b", label="Randomized median")
    axes[0].plot(dates, result.actual_equity, color="#1d4ed8", linewidth=2, label="Actual strategy")
    axes[0].set(title="Historical equity paths", ylabel="Account value (starts at 1)")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[1].hist(result.variations["cagr"] * 100, bins=50, color="#93c5fd", edgecolor="white")
    axes[1].axvline(result.actual["cagr"][0] * 100, color="#1d4ed8", linewidth=2, label="Actual strategy")
    axes[1].set(title="Randomized strategy performance", xlabel="CAGR (%)", ylabel="Variations")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
    svg = io.StringIO()
    fig.savefig(svg, format="svg")
    plt.close(fig)
    settings = ", ".join(f"{key}={value}" for key, value in asdict(rule).items())
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Monte Carlo strategy report</title>
<style>body{{font:16px/1.6 system-ui;color:#172033;max-width:1120px;margin:48px auto;padding:0 24px}}
h1{{line-height:1.15}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:10px;border-bottom:1px solid #dce3ed;text-align:right}}th:first-child{{text-align:left}}
svg{{max-width:100%;height:auto}}.stat{{background:#eff6ff;padding:18px;border-radius:8px}}small{{color:#475569}}
@media(max-width:700px){{body{{margin:24px auto}}table{{font-size:12px}}th,td{{padding:6px}}}}</style>
<h1>Monte Carlo strategy report</h1>
<p>{len(values):,} randomized score assignments · {escape(str(dates[0]))}–{escape(str(dates[-1]))}
· {len(data.names)} assets · {len(data.decision_rows)} forecasts · {result.seconds:.2f}s</p>
<p class="stat"><b>{fraction:.2%}</b> upper-tail placebo fraction:
{exceed:,} of {len(values):,} randomized strategies matched or exceeded actual annual net log growth.
Calculated as (1 + exceedances) / (1 + variations), including ties.</p>
{svg.getvalue()[svg.getvalue().find('<svg'):]}
<table><thead><tr><th>Metric</th><th>Actual</th><th>Randomized 5th</th><th>Randomized median</th>
<th>Randomized 95th</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>What was tested</h2><p>Raw score assignments were randomized within supplied decision-known risk groups.
Compatible assignments persisted for {block_forecasts} forecasts; changes in eligibility or groups required rematching.
Market history and forecast dates stayed fixed. Every variation reran selection, share/cash accounting,
next-close orders, entry-frozen volatility stops and actual-turnover fees. Final holdings were liquidated.</p>
<p>These are conditional historical placebo comparisons, not future-return forecasts or automatically calibrated
no-edge p-values. Matching does not establish exchangeability. Prior model selection is not corrected.</p>
<h2>Rules and assumptions</h2><p>The supplied reference allocator buys the highest-score tail and shorts the
lowest-score tail equally. Verify or replace target_weights() before drawing conclusions about a different strategy.
Prices must represent consistent total-return wealth; this is not a broker or corporate-action simulator.
Borrow fees, financing and cash interest are assumed zero. Missing required data and failed variations stop the whole run.</p>
<small>Seed: {seed}. {escape(settings)}. All metrics use 252 sessions/year. Percentiles describe randomized historical
strategies, not confidence intervals for future returns. See variations.csv for every result.</small></html>"""
    (output / "report.html").write_text(html, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-forecasts", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("monte_carlo_report"))
    args = parser.parse_args(argv)
    try:
        data = ingest_data()
        rule = Strategy()
        result = run(data, rule, variations=args.variations, seed=args.seed,
                     batch_size=args.batch_size, block_forecasts=args.block_forecasts)
        report(result, data, rule, args.output, seed=args.seed, block_forecasts=args.block_forecasts)
    except (NotImplementedError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Completed {args.variations:,} variations in {result.seconds:.2f}s: {args.output / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
