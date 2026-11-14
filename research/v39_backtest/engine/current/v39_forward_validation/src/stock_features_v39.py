"""Causal, segment-aware stock X and explicitly separate future labels.

No files, network, model or portfolio fitting. Inputs must already use a qualified,
consistent adjusted OHLC/total-return basis. Missing history is never filled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RESERVED = pd.Timestamp("2026-11-12")
CALENDAR_START = pd.Timestamp("2013-01-01")
FEATURE_START = pd.Timestamp("2014-01-01")
SECTORS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", "XLRE")
CROSS_ASSETS = ("IEF", "TLT", "TIP", "GLD", "DBC", "VNQ")
IDIO_FLOOR = .003
OWN = tuple([f"return{h}" for h in (1, 5, 21, 63, 126, 252)] + [
    "vol21", "beta_spy63", "close_ema21", "drawdown63", "rsi14", "bb20_z",
    "typical_vwap20_distance", "atr14_pct", "volume20_60", "return_volume_corr21",
    "overnight_cum5", "overnight_cum21", "intraday_cum5", "intraday_cum21",
    "overnight_intraday_spread5", "overnight_intraday_spread21"])
SECTOR_OWN = ("beta_sector63", "corr_sector63", "parent_mom5", "parent_mom21", "parent_mom63",
              "parent_vol21", "own_minus_parent_mom5", "own_minus_parent_mom21")
MARKET = ("market_return21", "market_vol21", "market_drawdown126")
VOLUME_OWN = {"typical_vwap20_distance", "volume20_60", "return_volume_corr21"}
MICRO_OWN = {name for name in OWN if name.startswith(("overnight", "intraday"))}


def feature_profiles():
    """Pure schema for freezing the contract without processing real prices."""
    base = ["cs_"+name for name in OWN] + ["volume_available", "momentum252_available", "micro_available"] + list(MARKET) + ["sector_"+s for s in SECTORS]
    sector = ["cs_"+name for name in SECTOR_OWN] + ["sector_available"]
    cross = [col for name in CROSS_ASSETS for col in (f"cs_cross_{name}_beta63", f"cs_cross_{name}_beta_return5",
        f"cross_{name}_available", f"cross_{name}_mom21", f"cross_{name}_vol21")]
    profiles = dict(base=base, sector=base+sector, crossasset=base+cross, all=base+sector+cross)
    assert {key:len(value) for key,value in profiles.items()} == dict(base=39, sector=48, crossasset=69, all=78)
    return profiles


def _bounded(frame):
    frame = frame.copy()
    if "Date" in frame:
        frame = frame.set_index("Date")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    # Reserved numeric objects are discarded before any numeric conversion.
    frame = frame.loc[frame.index < RESERVED].copy()
    if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Input dates must be unique, increasing daily sessions")
    return frame


def _prepare(frame, calendar):
    bounded = _bounded(frame) if frame is not None else pd.DataFrame(index=pd.DatetimeIndex([]))
    raw = bounded.reindex(calendar)
    out = pd.DataFrame(index=calendar)
    for col in ("adj_open", "adj_high", "adj_low", "adj_close", "Volume"):
        out[col] = pd.to_numeric(raw[col], errors="raise") if col in raw else np.nan
    if "volume_comparable" in raw:
        out["Volume"] = out.Volume.where(raw.volume_comparable.fillna(1).eq(1))
    raw_close = "raw_close" if "raw_close" in raw else "Close" if "Close" in raw else None
    out["raw_close"] = pd.to_numeric(raw[raw_close], errors="raise") if raw_close else np.nan
    for col in ("source_available", "tr_chain_available"):
        out[col] = pd.to_numeric(raw[col], errors="raise").eq(1) if col in raw else raw.index.isin(bounded.index)
    segment_col = "tr_segment_id" if "tr_segment_id" in raw else "segment_id" if "segment_id" in raw else None
    segment = raw[segment_col].astype(object) if segment_col else pd.Series("default", index=calendar)
    bars = out[["adj_open", "adj_high", "adj_low", "adj_close"]]
    valid = pd.Series(np.isfinite(bars).all(axis=1), index=calendar) & bars.gt(0).all(axis=1)
    tolerance = bars.abs().max(axis=1) * 1e-7
    valid &= bars.adj_high.add(tolerance).ge(bars[["adj_open", "adj_close", "adj_low"]].max(axis=1))
    valid &= bars.adj_low.sub(tolerance).le(bars[["adj_open", "adj_close", "adj_high"]].min(axis=1))
    valid &= out.source_available & out.tr_chain_available
    if segment_col:
        valid &= segment.notna()
    breaks = ~valid | ~valid.shift(1, fill_value=False) | segment.ne(segment.shift())
    out["segment_id"] = breaks.cumsum().astype(int)
    out["price_available"] = valid
    out["consecutive_bars"] = valid.astype(int).groupby(out.segment_id, sort=False).cumsum()
    out.loc[~valid, ["adj_open", "adj_high", "adj_low", "adj_close"]] = np.nan
    out["close_return"] = out.adj_close / out.adj_close.groupby(out.segment_id, sort=False).shift(1) - 1
    out["open_return"] = out.adj_open / out.adj_open.groupby(out.segment_id, sort=False).shift(1) - 1
    return out


def _groups(p):
    good = p.loc[p.price_available]
    return good.groupby("segment_id", sort=False).groups.values()


def _technicals(p):
    out = pd.DataFrame(np.nan, index=p.index, columns=list(OWN)+["volume_available", "micro_available", "momentum252_available", "drawdown126"])
    for indices in _groups(p):
        block = p.loc[indices]
        c, o, high, low = block.adj_close, block.adj_open, block.adj_high, block.adj_low
        ret = c.pct_change(fill_method=None)
        for h in (1, 5, 21, 63, 126, 252):
            out.loc[indices, f"return{h}"] = c.pct_change(h, fill_method=None)
        out.loc[indices, "vol21"] = ret.rolling(21).std(ddof=1)
        out.loc[indices, "close_ema21"] = c/c.ewm(span=21, adjust=False, min_periods=21).mean()-1
        for h in (63, 126):
            out.loc[indices, f"drawdown{h}"] = c/c.rolling(h).max()-1
        # Wilder smoothing applies to close-price changes, not percent returns.
        difference = c.diff()
        gain = difference.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = -difference.clip(upper=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rsi = 100-100/(1+gain/loss)
        rsi = rsi.where(loss.ne(0), 100.).where(~(gain.eq(0)&loss.eq(0)), 50.)
        rsi = rsi.where(gain.notna()&loss.notna())
        out.loc[indices, "rsi14"] = rsi
        average, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
        out.loc[indices, "bb20_z"] = ((c-average)/sd).where(sd.ne(0), 0).where(sd.notna())
        true_range = pd.concat([high-low, (high-c.shift()).abs(), (low-c.shift()).abs()], axis=1).max(axis=1)
        out.loc[indices, "atr14_pct"] = true_range.ewm(alpha=1/14, adjust=False, min_periods=14).mean()/c
        volume = block.Volume.where(np.isfinite(block.Volume)&block.Volume.gt(0))
        volume_ok = volume.rolling(60).count().eq(60)
        volume20, volume60 = volume.rolling(20).mean(), volume.rolling(60).mean()
        out.loc[indices, "volume20_60"] = (volume20/volume60-1).where(volume_ok)
        typical = (high+low+c)/3
        # Daily typical-price/volume proxy; not exchange intraday VWAP.
        vwap_proxy = (typical*volume).rolling(20).sum()/volume.rolling(20).sum()
        out.loc[indices, "typical_vwap20_distance"] = (c/vwap_proxy-1).where(volume_ok)
        log_volume_change = np.log(volume).diff()
        corr = ret.rolling(21).corr(log_volume_change)
        corr_known = ret.rolling(21).count().eq(21)&log_volume_change.rolling(21).count().eq(21)
        corr = corr.where(corr.notna(), 0.).where(corr_known)
        out.loc[indices, "return_volume_corr21"] = corr.where(volume_ok)
        out.loc[indices, "volume_available"] = (volume_ok&corr_known).astype(float)
        overnight = np.log(o/c.shift())
        intraday = np.log(c/o)
        micro_ok = c.rolling(22).count().eq(22)
        for h in (5, 21):
            on = np.expm1(overnight.rolling(h).sum())
            inside = np.expm1(intraday.rolling(h).sum())
            out.loc[indices, f"overnight_cum{h}"] = on.where(micro_ok)
            out.loc[indices, f"intraday_cum{h}"] = inside.where(micro_ok)
            out.loc[indices, f"overnight_intraday_spread{h}"] = (on-inside).where(micro_ok)
        out.loc[indices, "micro_available"] = micro_ok.astype(float)
        out.loc[indices, "momentum252_available"] = c.rolling(253).count().eq(253).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def _pair(stock, other):
    """Rolling relationships reset whenever either member's segment changes."""
    good = stock.price_available & other.price_available
    breaks = (~good | ~good.shift(1, fill_value=False) |
              stock.segment_id.ne(stock.segment_id.shift()) | other.segment_id.ne(other.segment_id.shift()))
    key = breaks.cumsum()
    beta = pd.Series(np.nan, index=stock.index)
    corr = beta.copy()
    idio = beta.copy()
    for indices in key.loc[good].groupby(key.loc[good], sort=False).groups.values():
        if len(indices) < 64:
            continue
        sr, er = stock.loc[indices, "close_return"], other.loc[indices, "close_return"]
        variance = er.rolling(63).var(ddof=1)
        b = (sr.rolling(63).cov(er)/variance).where(variance.gt(1e-16))
        beta.loc[indices] = b
        corr.loc[indices] = sr.rolling(63).corr(er)
        residual = sr-b.shift(1)*er  # today's close response uses yesterday's known beta.
        idio.loc[indices] = residual.rolling(21).std(ddof=1)
    return beta, corr, idio


def _labels(stock, parent, beta, idio, atr, horizons, execution_price="open"):
    result = pd.DataFrame(index=stock.index)
    price_col, return_col = "adj_"+execution_price, execution_price+"_return"
    execution = pd.Series(stock.index, index=stock.index).shift(-1)
    result["execution_date"] = execution
    for h in horizons:
        end = pd.Series(stock.index, index=stock.index).shift(-(h+1))
        result[f"label_end_h{h}"] = end
        stock_path = (stock.price_available & stock.price_available.shift(-(h+1), fill_value=False) &
                      stock.segment_id.eq(stock.segment_id.shift(-(h+1))))
        parent_path = (parent.price_available & parent.price_available.shift(-(h+1), fill_value=False) &
                       parent.segment_id.eq(parent.segment_id.shift(-(h+1))))
        entry = stock[price_col].shift(-1)
        sr = (stock[price_col].shift(-(h+1))/entry-1).where(stock_path)
        pr = (parent[price_col].shift(-(h+1))/parent[price_col].shift(-1)-1).where(parent_path)
        residual = sr-beta*pr
        result[f"future_return_h{h}"] = sr
        result[f"future_parent_return_h{h}"] = pr
        result[f"target_sector_residual_h{h}"] = residual
        result[f"target_risk_resid_h{h}"] = residual/(idio*np.sqrt(h))
        # Open entry: e..e+h-1; close entry: e+1..e+h, excluding pre-entry extremes.
        extreme_shift = h + int(execution_price=="close")
        future_low = stock.adj_low.rolling(h).min().shift(-extreme_shift)
        future_high = stock.adj_high.rolling(h).max().shift(-extreme_shift)
        long_mae = (1-future_low/entry).clip(lower=0).where(stock_path)
        short_mae = (future_high/entry-1).clip(lower=0).where(stock_path)
        result[f"future_long_mae_h{h}"] = long_mae
        result[f"future_short_mae_h{h}"] = short_mae
        result[f"target_long_tail_h{h}"] = (long_mae>1.5*atr).astype(float).where(long_mae.notna()&atr.notna())
        result[f"target_short_tail_h{h}"] = (short_mae>1.5*atr).astype(float).where(short_mae.notna()&atr.notna())
        squares = pd.Series(0., index=stock.index)
        for step in range(2, h+2):
            eps = stock[return_col].shift(-step)-beta*parent[return_col].shift(-step)
            squares = squares + eps.pow(2)
        result[f"target_idio_vol_h{h}"] = np.sqrt(squares/h).where(stock_path&parent_path&beta.notna())
    return result


def _cash_events(events):
    if events is None:
        return pd.DataFrame(columns=["ticker", "effective_date", "raw_cash_price"])
    events = pd.DataFrame(events).copy()
    required = {"ticker", "effective_date", "raw_cash_price"}
    if not required.issubset(events):
        raise ValueError("Cash events require ticker, effective_date and raw_cash_price")
    events["effective_date"] = pd.to_datetime(events.effective_date, errors="raise")
    events = events.loc[events.effective_date < RESERVED].copy()
    events["raw_cash_price"] = pd.to_numeric(events.raw_cash_price, errors="raise")
    if not np.isfinite(events.raw_cash_price).all() or not events.raw_cash_price.gt(0).all():
        raise ValueError("Verified raw cash payments must be positive and finite")
    if "event_phase" in events and not events.event_phase.eq("open").all():
        raise ValueError("Only verified open-effective cash events are supported")
    if "event_type" in events and not events.event_type.isin(["cash_merger","stock_merger_liquidation","otc_liquidation"]).all():
        raise ValueError("Only explicit cash-merger events are supported")
    if events.ticker.duplicated().any():
        raise ValueError("This bounded adapter supports one final cash merger per issue")
    return events


def _label_wealth(stock, event):
    """Extend settled wealth only, never the source used to compute any X/flag."""
    record = dict(ticker=event.ticker, effective_date=str(event.effective_date), applied=False,
                  classification="label-only verified cash settlement, not executable quotes")
    positions = np.flatnonzero(stock.index>=event.effective_date)
    if not len(positions) or positions[0] == 0:
        record["reason"] = "No pre-event basis or event outside observed calendar"
        return stock, record
    first = int(positions[0])
    prior = stock.iloc[first-1]
    # Do not bridge an unobserved price gap immediately preceding settlement.
    if not prior.price_available or not np.isfinite(prior.raw_close) or prior.raw_close <= 0:
        record["reason"] = "Immediately preceding qualified adjusted/raw close basis missing"
        return stock, record
    factor = float(prior.adj_close/prior.raw_close)
    wealth = float(event.raw_cash_price*factor)
    result = stock.copy()
    after = result.index>=result.index[first]
    result.loc[after, ["adj_open", "adj_high", "adj_low", "adj_close"]] = wealth
    result.loc[after, "price_available"] = True
    result.loc[after, "segment_id"] = int(prior.segment_id)
    result["open_return"] = result.adj_open/result.adj_open.groupby(result.segment_id, sort=False).shift(1)-1
    result["close_return"] = result.adj_close/result.adj_close.groupby(result.segment_id, sort=False).shift(1)-1
    record.update(applied=True, prior_basis_date=str(stock.index[first-1]),
                  raw_cash_price=float(event.raw_cash_price), exante_adjustment_factor=factor,
                  adjusted_cash_wealth=wealth, first_cash_session=str(result.index[first]),
                  cash_sessions=int(after.sum()), interest_rate=0.)
    return result, record


def _rank(panel, raw_col, valid, by="decision_date", *, unknown=np.nan):
    source = panel[raw_col].where(valid & np.isfinite(panel[raw_col]))
    group = source.groupby([panel[col] for col in ([by] if isinstance(by, str) else by)], sort=False)
    count, rank = group.transform("count"), group.rank(method="average")
    result = (2*(rank-1)/(count-1)-1).where(count.gt(1), 0.).where(source.notna(), unknown)
    return result


def build_stock_panel(pricesByISIN, etfPrices, cohort, horizons=(3, 5, 10), *, cash_events=None, execution_price="open", keep_raw=False):
    """Return panel, {base39/sector48/crossasset69/all78}, SPY calendar, diagnostics.

    Every emitted decision has one row per declared stock, irrespective of future
    label availability. Caller may query-filter only known eligible/sector flags.
    keep_raw retains causal raw_* diagnostics outside every model feature profile.
    """
    horizons = tuple(int(h) for h in horizons)
    if execution_price not in ("open", "close"):
        raise ValueError("execution_price must be open or close")
    if not horizons or len(set(horizons)) != len(horizons) or min(horizons) < 1:
        raise ValueError("Horizons must be distinct positive session counts")
    cohort = pd.DataFrame(cohort).copy()
    if "ticker" not in cohort and "ISIN" in cohort:
        cohort = cohort.rename(columns={"ISIN": "ticker"})
    if "sector" not in cohort and "parentETF" in cohort:
        cohort = cohort.rename(columns={"parentETF": "sector"})
    if not {"ticker", "sector"}.issubset(cohort) or cohort.empty:
        raise ValueError("Cohort requires ticker (ISIN) and sector (parent ETF)")
    if cohort.ticker.isna().any() or cohort.ticker.duplicated().any() or not cohort.sector.isin(SECTORS).all():
        raise ValueError("Stock identities must be unique and have declared sector ETFs")
    if "SPY" not in etfPrices:
        raise ValueError("SPY defines the observed execution-session calendar")
    spy_bound = _bounded(etfPrices["SPY"])
    calendar = spy_bound.index[spy_bound.index>=CALENDAR_START]
    if len(calendar) < 128:
        raise ValueError("Insufficient pre-reserve SPY sessions")
    needed = set(cohort.sector) | set(CROSS_ASSETS) | {"SPY"}
    etfs = {name: _prepare(etfPrices.get(name), calendar) for name in needed}
    if not etfs["SPY"].price_available.all():
        raise ValueError("SPY session clock requires qualified complete adjusted OHLC")
    etf_tech = {name: _technicals(frame) for name, frame in etfs.items()}
    events = _cash_events(cash_events)
    market = pd.DataFrame(dict(market_return21=etf_tech["SPY"].return21,
        market_vol21=etf_tech["SPY"].vol21, market_drawdown126=etf_tech["SPY"].drawdown126), index=calendar)
    frames, source_records, cash_records = [], [], []
    for row in cohort.itertuples(index=False):
        name, sector = row.ticker, row.sector
        p = _prepare(pricesByISIN.get(name), calendar)
        raw = _technicals(p)
        raw["beta_spy63"], _, _ = _pair(p, etfs["SPY"])
        parent = etfs[sector]
        b_sector, c_sector, idio_raw = _pair(p, parent)
        raw["beta_sector63"], raw["corr_sector63"] = b_sector, c_sector
        for h in (5, 21, 63):
            raw[f"parent_mom{h}"] = etf_tech[sector][f"return{h}"]
        raw["parent_vol21"] = etf_tech[sector].vol21
        for h in (5, 21):
            raw[f"own_minus_parent_mom{h}"] = raw[f"return{h}"]-raw[f"parent_mom{h}"]
        x = raw.add_prefix("raw_")
        x["eligible"] = p.price_available & p.consecutive_bars.ge(127)
        x["price_available"] = p.price_available
        x["consecutive_bars"] = p.consecutive_bars
        x["segment_id"] = p.segment_id
        x["source_available"], x["tr_chain_available"] = p.source_available, p.tr_chain_available
        x["beta_spy63"], x["beta_sector63"] = raw.beta_spy63, b_sector
        x["idio_vol21_raw"], x["idio_vol21"] = idio_raw, idio_raw.clip(lower=IDIO_FLOOR)
        x["atr14_pct"] = raw.atr14_pct
        x["sector_available"] = np.isfinite(raw[list(SECTOR_OWN)]).all(axis=1).astype(float)
        for flag in ("volume_available", "momentum252_available", "micro_available"):
            x[flag] = raw[flag].fillna(0.)
        for name_etf in CROSS_ASSETS:
            beta, _, _ = _pair(p, etfs[name_etf])
            x[f"raw_cross_{name_etf}_beta63"] = beta
            x[f"raw_cross_{name_etf}_beta_return5"] = beta*etf_tech[name_etf].return5
            x[f"cross_{name_etf}_mom21"] = etf_tech[name_etf].return21
            x[f"cross_{name_etf}_vol21"] = etf_tech[name_etf].vol21
            cols = [f"raw_cross_{name_etf}_beta63", f"raw_cross_{name_etf}_beta_return5",
                    f"cross_{name_etf}_mom21", f"cross_{name_etf}_vol21"]
            x[f"cross_{name_etf}_available"] = np.isfinite(x[cols]).all(axis=1).astype(float)
        label_source = p
        matching = events.loc[events.ticker.eq(name)]
        if len(matching):
            label_source, cash_record = _label_wealth(p, matching.iloc[0])
            cash_records.append(cash_record)
        x = x.join(market).join(_labels(label_source, parent, b_sector, x.idio_vol21, raw.atr14_pct, horizons, execution_price))
        x["execution_clock"] = "NEXT_SESSION_"+execution_price.upper()
        x["ticker"], x["sector"] = name, sector
        x.index.name = "decision_date"
        x = x.loc[(x.index>=FEATURE_START)&x.execution_date.notna()]
        frames.append(x.reset_index())
        source_records.append(dict(ticker=name, sector=sector, source_present=name in pricesByISIN,
            valid_bars=int(p.price_available.sum()), segments=int(p.loc[p.price_available, "segment_id"].nunique()),
            eligible_rows=int(x.eligible.sum()), sector_available_rows=int(x.sector_available.sum())))
    panel = pd.concat(frames, ignore_index=True).sort_values(["decision_date", "ticker"]).reset_index(drop=True).copy()
    if panel.empty:
        raise ValueError("No executable decision dates after feature start")
    for feature in OWN:
        valid = panel.eligible.copy()
        if feature == "return252":
            valid &= panel.momentum252_available.eq(1)
        elif feature in VOLUME_OWN:
            valid &= panel.volume_available.eq(1)
        elif feature in MICRO_OWN:
            valid &= panel.micro_available.eq(1)
        panel["cs_"+feature] = _rank(panel, "raw_"+feature, valid, unknown=0.)
    for feature in SECTOR_OWN:
        panel["cs_"+feature] = _rank(panel, "raw_"+feature, panel.eligible&panel.sector_available.eq(1), unknown=0.)
    cross_columns = []
    for name in CROSS_ASSETS:
        flag = f"cross_{name}_available"
        for stem in ("beta63", "beta_return5"):
            col = f"cs_cross_{name}_{stem}"
            panel[col] = _rank(panel, f"raw_cross_{name}_{stem}", panel.eligible&panel[flag].eq(1), unknown=0.)
            cross_columns.append(col)
        cross_columns += [flag, f"cross_{name}_mom21", f"cross_{name}_vol21"]
        for col in (f"cross_{name}_mom21", f"cross_{name}_vol21"):
            panel[col] = panel[col].where(panel[flag].eq(1), 0.)
    for sector in SECTORS:
        panel["sector_"+sector] = panel.sector.eq(sector).astype(float)
    profiles = feature_profiles()
    for col in profiles["all"]:
        panel[col] = panel[col].replace([np.inf, -np.inf], np.nan).fillna(0.).astype(float)
    panel = panel.copy()  # Consolidate feature blocks before appending label-rank metadata.
    for h in horizons:
        eligible_labels = panel.eligible & panel.sector_available.eq(1)
        for raw, prefix in ((f"target_sector_residual_h{h}", "target_rank"),
                            (f"target_risk_resid_h{h}", "target_risk_rank")):
            panel[f"{prefix}_date_h{h}"] = _rank(panel, raw, eligible_labels)
            panel[f"{prefix}_sector_h{h}"] = _rank(panel, raw, eligible_labels, ["decision_date", "sector"])
        panel[f"target_rank_h{h}"] = panel[f"target_rank_date_h{h}"]
    raw_columns = [col for col in panel if col.startswith("raw_")]
    if not keep_raw:
        panel = panel.drop(columns=raw_columns)
    counts = panel.groupby("decision_date").size()
    if not counts.eq(len(cohort)).all() or panel.duplicated(["decision_date", "ticker"]).any():
        raise AssertionError("The complete declared cohort must be retained on every decision")
    diagnostics = dict(cohort_size=len(cohort), panel_rows=len(panel), decision_dates=len(counts),
        first_decision=str(panel.decision_date.min()), last_decision=str(panel.decision_date.max()),
        last_execution=str(panel.execution_date.max()), calendar_start=str(calendar.min()), calendar_end=str(calendar.max()),
        profile_counts={key:len(value) for key,value in profiles.items()}, horizons=list(horizons),
        eligibility="127 consecutive valid adjusted bars in one segment, no future-label test",
        momentum252="253 consecutive valid bars; zero feature plus availability flag otherwise",
        idio_floor_daily=IDIO_FLOOR, source_records=source_records,
        cash_event_records=cash_records, cash_events_are_features=False,
        eligible_rows=int(panel.eligible.sum()), known_parent_rows=int((panel.eligible&panel.sector_available.eq(1)).sum()),
        unknown_labels={str(h):int(panel[f"target_sector_residual_h{h}"].isna().sum()) for h in horizons},
        vwap="20-session typical adjusted-price/volume proxy, not measured intraday VWAP",
        membership="fixed declared cohort, not historical membership for earlier price-training dates",
        no_reserved_numeric_processing=True, feature_start=str(FEATURE_START))
    diagnostics["execution_clock"] = "NEXT_SESSION_"+execution_price.upper()
    diagnostics["mae_sessions"] = "entry e through e+h-1" if execution_price=="open" else "e+1 through exit e+h; entry-day intraday extremes excluded"
    diagnostics["retained_raw_columns"] = raw_columns if keep_raw else []
    return panel, profiles, calendar, diagnostics
