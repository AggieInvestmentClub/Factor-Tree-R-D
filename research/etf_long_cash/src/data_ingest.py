"""Download auditable public ETF and macro data for local Swing Tree research.

Run from the project folder with ``python src/data_ingest.py``.  Raw provider
responses are content-addressed and never overwritten.  Daily macro observations
are conservatively released two business days later, but are NOT true vintages.
Revised monthly series are archived and excluded from the modeling feature file.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

TICKERS = "XLK XLF XLE XLV XLI XLY XLP XLU XLB IEF TLT TIP GLD DBC VNQ SPY UUP".split()
DAILY_MACRO = ["DGS10", "DGS2", "DTB3", "T10YIE", "VIXCLS", "BAMLH0A0HYM2"]
MONTHLY_MACRO = ["CPIAUCSL", "UNRATE", "INDPRO"]
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36", "Accept": "application/json,text/csv,*/*"}
USE_RAW_CACHE = False


def fetch(url: str) -> bytes:
    """Try ordinary HTTPS first, then an installed browser-compatible client."""
    if USE_RAW_CACHE:
        index = json.loads((DATA / "raw_download_manifest.json").read_text(encoding="utf-8"))
        match = next((x for x in index["sources"] if x["source_url"] == url), None)
        if match:
            payload = (ROOT / match["raw_path"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != match["sha256"]:
                raise RuntimeError("Raw cache integrity failure")
            return payload
    errors = []
    for attempt in range(3):
        try:
            if attempt == 0:
                with urlopen(Request(url, headers=HEADERS), timeout=50) as response:
                    return response.read()
            else:
                from curl_cffi import requests
                response = requests.get(url, impersonate="chrome", timeout=50)
                response.raise_for_status()
                return response.content
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt < 2:
                time.sleep(1 + 2 * attempt)
    raise RuntimeError(f"Download failed for {url}: {'; '.join(errors)}")


def archive(payload: bytes, family: str, identifier: str, extension: str, url: str) -> dict:
    digest = hashlib.sha256(payload).hexdigest()
    folder = DATA / "raw" / family
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{identifier}_{digest[:16]}.{extension}"
    if not target.exists():
        target.write_bytes(payload)
    elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Immutable archive collision: {target}")
    retrieved = datetime.now(timezone.utc).isoformat()
    if USE_RAW_CACHE and (DATA / "raw_download_manifest.json").exists():
        original = json.loads((DATA / "raw_download_manifest.json").read_text(encoding="utf-8"))
        matching = next((x for x in original["sources"] if x["source_url"] == url and x["sha256"] == digest), None)
        if matching:
            retrieved = matching["retrieved_utc"]
    return {"identifier": identifier, "source_url": url, "retrieved_utc": retrieved,
            "raw_path": target.relative_to(ROOT).as_posix(), "sha256": digest, "bytes": len(payload)}


def get_prices(ticker: str, cutoff: str) -> tuple[pd.DataFrame, dict]:
    # The upper bound excludes the next UTC day.  Every returned local session is
    # separately checked against cutoff, so intraday/future bars cannot creep in.
    end = int((datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp())
    query = urlencode({"period1": 0, "period2": end, "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{query}"
    raw = fetch(url)
    entry = archive(raw, "yahoo", ticker, "json", url)
    body = json.loads(raw)
    if body.get("chart", {}).get("error"):
        raise RuntimeError(f"Yahoo {ticker}: {body['chart']['error']}")
    result = body["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj is None:
        raise RuntimeError(f"No adjusted close returned for {ticker}; refusing fabricated adjustment")
    exchange_tz = ZoneInfo(result["meta"].get("exchangeTimezoneName", "America/New_York"))
    dates = [datetime.fromtimestamp(x, timezone.utc).astimezone(exchange_tz).strftime("%Y-%m-%d") for x in timestamps]
    frame = pd.DataFrame({"Date": dates, "Open": quote["open"], "High": quote["high"], "Low": quote["low"],
                          "Close": quote["close"], "Adj Close": adj, "Volume": quote["volume"]})
    frame["Dividends"] = 0.0
    frame["Stock Splits"] = 0.0
    action_details = []
    for kind, events in result.get("events", {}).items():
        for event in events.values():
            day = datetime.fromtimestamp(event["date"], timezone.utc).astimezone(exchange_tz).strftime("%Y-%m-%d")
            item = {"Date": day, "type": kind, **event}
            action_details.append(item)
            if kind == "dividends":
                frame.loc[frame.Date == day, "Dividends"] = float(event["amount"])
            elif kind == "splits":
                ratio = float(event["numerator"]) / float(event["denominator"])
                frame.loc[frame.Date == day, "Stock Splits"] = ratio
    missing_price = frame[["Open", "High", "Low", "Close", "Adj Close"]].isna().any(axis=1)
    entry["provider_null_price_rows"] = int(missing_price.sum())
    frame = frame.loc[(frame.Date <= cutoff) & ~missing_price].copy().sort_values("Date").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"Empty usable price history for {ticker}")
    ratio = frame["Adj Close"] / frame["Close"]
    for field in ("Open", "High", "Low", "Close"):
        frame[f"adj_{field.lower()}"] = frame[field] * ratio
    entry.update({"ticker": ticker, "first_date": frame.Date.min(), "last_date": frame.Date.max(), "rows": len(frame),
                  "provider_name": result["meta"].get("longName", result["meta"].get("shortName")),
                  "currency": result["meta"].get("currency"), "exchange_timezone": str(exchange_tz),
                  "adjustment": "Provider OHLC retained. Adjusted OHLC = provider OHLC * (Adj Close / Close). Volume retained unadjusted.",
                  "action_count": len(action_details)})
    action_path = DATA / "normalized" / "actions" / f"{ticker}.csv"
    action_path.parent.mkdir(parents=True, exist_ok=True)
    if action_details:
        pd.DataFrame(action_details).sort_values("Date").to_csv(action_path, index=False)
    else:
        pd.DataFrame(columns=["Date", "type"]).to_csv(action_path, index=False)
    output = DATA / "normalized" / "prices" / f"{ticker}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    entry["normalized_path"] = output.relative_to(ROOT).as_posix()
    entry["normalized_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    entry["actions_path"] = action_path.relative_to(ROOT).as_posix()
    entry["actions_sha256"] = hashlib.sha256(action_path.read_bytes()).hexdigest()
    return frame, entry


def get_macro(series: str, cutoff: str) -> tuple[pd.DataFrame, dict]:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode({"id": series, "cosd": "1900-01-01", "coed": cutoff})
    raw = fetch(url)
    entry = archive(raw, "fred", series, "csv", url)
    frame = pd.read_csv(io.BytesIO(raw))
    if series not in frame.columns or len(frame.columns) != 2:
        raise RuntimeError(f"Unexpected FRED response for {series}: {list(frame.columns)}")
    frame.columns = ["Date", series]
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    frame[series] = pd.to_numeric(frame[series], errors="coerce")
    frame = frame.loc[frame.Date <= cutoff].sort_values("Date").reset_index(drop=True)
    output = DATA / "normalized" / "macro_archive" / f"{series}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    entry.update({"series": series, "first_date": frame.Date.min(), "last_date": frame.Date.max(),
                  "last_nonmissing_date": frame.loc[frame[series].notna(), "Date"].max(), "rows": len(frame),
                  "missing_rows": int(frame[series].isna().sum()), "normalized_path": output.relative_to(ROOT).as_posix(),
                  "frequency": "daily" if series in DAILY_MACRO else "monthly",
                  "point_in_time": "approximate; latest-vintage archive, no historical vintage guarantee" if series in DAILY_MACRO else "quarantined; revised monthly series; not used in model features"})
    if series == "BAMLH0A0HYM2":
        entry["coverage_limitation"] = "FRED source notes limit this ICE series to three years starting April 2026. Do not fill precoverage values or require this factor in old folds."
        entry["source_notes_url"] = "https://fred.stlouisfed.org/series/BAMLH0A0HYM2"
    entry["normalized_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    return frame, entry


def validate_prices(prices: dict[str, pd.DataFrame], entries: dict[str, dict]) -> list[dict]:
    benchmark_dates = set(prices["SPY"].Date) if "SPY" in prices else set()
    checks = []
    for ticker, frame in prices.items():
        duplicate_count = int(frame.Date.duplicated().sum())
        p = frame[["Open", "High", "Low", "Close", "Adj Close", "adj_open", "adj_high", "adj_low", "adj_close"]]
        nonpositive_count = int((p <= 0).any(axis=1).sum())
        invalid_range_count = int(((frame.High + 1e-8 < frame[["Open", "Close", "Low"]].max(axis=1)) |
                                   (frame.Low - 1e-8 > frame[["Open", "Close", "High"]].min(axis=1))).sum())
        negative_volume_count = int((frame.Volume < 0).sum())
        weekend_count = int((pd.to_datetime(frame.Date).dt.dayofweek >= 5).sum())
        first_date, last_date = frame.Date.min(), frame.Date.max()
        expected = {d for d in benchmark_dates if first_date <= d <= last_date}
        gaps = sorted(expected - set(frame.Date))
        q = {"ticker": ticker, "first_date": frame.Date.min(), "last_date": frame.Date.max(), "rows": len(frame),
             "duplicate_dates": duplicate_count, "nonpositive_price_rows": nonpositive_count,
             "invalid_ohlc_range_rows": invalid_range_count, "negative_volume_rows": negative_volume_count,
             "weekend_rows": weekend_count, "null_volume_rows": int(frame.Volume.isna().sum()),
             "zero_volume_rows": int((frame.Volume == 0).sum()), "dividend_events": int((frame.Dividends != 0).sum()),
             "split_events": int((frame["Stock Splits"] != 0).sum()), "missing_spy_sessions": len(gaps),
             "missing_spy_session_dates": ";".join(gaps), "provider_null_price_rows": entries[ticker]["provider_null_price_rows"]}
        q["status"] = "FAIL" if any([duplicate_count, nonpositive_count, invalid_range_count, negative_volume_count, weekend_count]) else "PASS"
        checks.append(q)
    return checks


def macro_features(macros: dict[str, pd.DataFrame], dates: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    result = pd.DataFrame({"Date": pd.to_datetime(sorted(set(dates)))})
    audit = []
    for series in DAILY_MACRO:
        if series not in macros:
            continue
        source = macros[series].dropna().copy()
        # Business-day shift is stricter than using observation date, but does not
        # prove release timestamps or guarantee historical vintage availability.
        source["available_date"] = pd.to_datetime(source.Date) + pd.offsets.BDay(2)
        source = source[["available_date", series]].sort_values("available_date")
        aligned = pd.merge_asof(result[["Date"]], source, left_on="Date", right_on="available_date", direction="backward")
        name = "macro_" + series.lower()
        result[name] = aligned[series]
        result[name + "_change20"] = result[name].diff(20)
        audit.append({"feature": name, "source_series": series, "availability_lag": "2 business days",
                      "missing_rows": int(result[name].isna().sum()), "nonmissing_rows": int(result[name].notna().sum()),
                      "first_usable_date": result.loc[result[name].notna(), "Date"].min().strftime("%Y-%m-%d") if result[name].notna().any() else None,
                      "vintage_status": "approximate_point_in_time_no_true_vintages"})
    if all(x in result for x in ["macro_dgs10", "macro_dgs2"]):
        result["macro_term_spread"] = result.macro_dgs10 - result.macro_dgs2
    if "macro_vixcls" in result:
        result["macro_vix_log"] = np.log(result.macro_vixcls.clip(lower=0.001))
    result["Date"] = result.Date.dt.strftime("%Y-%m-%d")
    return result, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default="2026-10-19")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--offline", action="store_true", help="Rebuild QA/features from downloaded normalized CSVs")
    parser.add_argument("--raw-only", action="store_true", help="Archive and report source coverage without third-party libraries")
    parser.add_argument("--from-raw", action="store_true", help="Normalize archived raw responses before fetching absent sources")
    parser.add_argument("--prices-only", action="store_true", help="Prepare price histories while macro downloads continue")
    parser.add_argument("--daily-only", action="store_true", help="Prepare daily market/macro features while monthly archives download")
    args = parser.parse_args()
    global USE_RAW_CACHE
    USE_RAW_CACHE = args.from_raw
    DATA.mkdir(parents=True, exist_ok=True)
    if args.raw_only:
        sources, errors = [], []
        end = int((datetime.fromisoformat(args.cutoff).replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp())
        jobs = []
        # First results answer the user's immediate inception-history question.
        for ticker in ["XLK", "DBC"] + [x for x in TICKERS if x not in ["XLK", "DBC"]]:
            query = urlencode({"period1": 0, "period2": end, "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"})
            jobs.append((ticker, "yahoo", "json", f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{query}"))
        for series in DAILY_MACRO + MONTHLY_MACRO:
            jobs.append((series, "fred", "csv", "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode({"id": series, "cosd": "1900-01-01", "coed": args.cutoff})))
        def one(job):
            name, family, ext, url = job
            payload = fetch(url)
            entry = archive(payload, family, name, ext, url)
            if family == "yahoo":
                result = json.loads(payload)["chart"]["result"][0]
                values = result["timestamp"]
                entry.update({"first_date": datetime.fromtimestamp(values[0], timezone.utc).strftime("%Y-%m-%d"),
                              "last_date": datetime.fromtimestamp(values[-1], timezone.utc).strftime("%Y-%m-%d"), "rows": len(values)})
            print(f"RAW OK {name}: {entry.get('rows', 'CSV')} {entry.get('first_date', '')} .. {entry.get('last_date', '')}", flush=True)
            return entry
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one, job): job for job in jobs}
            for future in as_completed(futures):
                try:
                    sources.append(future.result())
                except Exception as exc:
                    name = futures[future][0]
                    errors.append({"identifier": name, "error": str(exc)})
                    print(f"RAW FAILED {name}: {exc}", flush=True)
                (DATA / "raw_download_manifest.json").write_text(json.dumps({"cutoff": args.cutoff, "sources": sources, "errors": errors}, indent=2), encoding="utf-8")
        return
    global pd, np
    import pandas as pd
    import numpy as np
    prices, macros, source_entries, errors = {}, {}, {}, []
    if args.offline:
        manifest_path = DATA / "data_manifest.json"
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_entries = {entry["identifier"]: entry for entry in existing["sources"]}
        for ticker in TICKERS:
            path = DATA / "normalized" / "prices" / f"{ticker}.csv"
            if path.exists():
                prices[ticker] = pd.read_csv(path, dtype={"Date": str})
        for series in DAILY_MACRO + MONTHLY_MACRO:
            path = DATA / "normalized" / "macro_archive" / f"{series}.csv"
            if path.exists():
                macros[series] = pd.read_csv(path, dtype={"Date": str})
        errors = existing.get("errors", [])
    else:
        macro_series = DAILY_MACRO + ([] if args.daily_only else MONTHLY_MACRO)
        jobs = [(ticker, "prices") for ticker in TICKERS] + ([] if args.prices_only else [(series, "macro") for series in macro_series])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(get_prices if kind == "prices" else get_macro, name, args.cutoff): (name, kind) for name, kind in jobs}
            for future in as_completed(futures):
                name, kind = futures[future]
                try:
                    frame, entry = future.result()
                    (prices if kind == "prices" else macros)[name] = frame
                    source_entries[name] = entry
                    print(f"OK {name}: {len(frame)} rows {entry['first_date']} .. {entry['last_date']}", flush=True)
                except Exception as exc:
                    errors.append({"identifier": name, "type": kind, "error": str(exc)})
                    print(f"FAILED {name}: {exc}", flush=True)
    qa = validate_prices(prices, source_entries)
    pd.DataFrame(qa).sort_values("ticker").to_csv(DATA / "coverage.csv", index=False)
    if prices:
        trading_dates = prices["SPY"].Date if "SPY" in prices else pd.Series(sorted(set(d for f in prices.values() for d in f.Date)))
        feature_frame, macro_audit = macro_features(macros, trading_dates)
        feature_frame.to_csv(DATA / "normalized" / "macro_features.csv", index=False)
        pd.DataFrame(macro_audit).to_csv(DATA / "macro_feature_audit.csv", index=False)
    else:
        macro_audit = []
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "requested_cutoff": args.cutoff,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "normalization_libraries": {"pandas": pd.__version__, "numpy": np.__version__},
                "candidate_universe": TICKERS[:15], "relationship_proxies": TICKERS[15:],
                "sources": list(source_entries.values()), "errors": errors, "price_qa": qa, "macro_feature_audit": macro_audit,
                "limitations": ["Public secondary ETF history; provider corrections can change future downloads. Raw source responses are immutable and SHA-256 recorded.",
                                "Yahoo OHLC is provider historical OHLC, generally split adjusted. Total-return adjusted OHLC uses same-date adjusted-close ratio; cash dividends must not be credited again when using adjusted OHLC.",
                                "Daily FRED levels use a two-business-day availability lag but latest available vintage; approximate PIT only. No guarantee of original publication timing or revision history.",
                                "CPIAUCSL, UNRATE, INDPRO are revision-sensitive monthly archives and excluded from macro_features.csv.",
                                "BAMLH0A0HYM2 is capped by FRED at three years of observations since April 2026; its early missingness is structural, and earlier folds must exclude that factor.",
                                "A missing row before inception is not imputed. Calendar gaps checked against SPY sessions within each instrument's actual coverage.",
                                "Cash yield DTB3 is annualized percent at conservative availability; transaction accounting decides whether and how to use it."]}
    # JSON null instead of nonstandard NaN values.
    payload = json.loads(json.dumps(manifest, default=str))
    (DATA / "data_manifest.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Finished: {len(prices)}/{len(TICKERS)} price series, {len(macros)}/9 macro series; {len(errors)} errors", flush=True)
    fatal = [q["ticker"] for q in qa if q["status"] == "FAIL"]
    if fatal:
        raise RuntimeError(f"Fatal price QA failures: {fatal}")
    if any(ticker not in prices for ticker in TICKERS[:15]):
        raise RuntimeError("Missing candidate ETF history; inspect data_manifest.json errors")


if __name__ == "__main__":
    main()
