"""Public current ETF/sector RSS snapshot and local FinBERT demonstration.

No accounts, full-article scraping, historical backfill or trading feature injection.
Run with the dedicated environment: python src/sector_news.py
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from finbert_news import MODEL, REPO, REVISION, classify

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "data/news"
REPORT = ROOT / "reports/sector_news_status.json"
RSS = "https://finance.yahoo.com/rss/headline?s="
HEADERS = {"User-Agent": "Mozilla/5.0"}
SECTORS = {
    "XLK": {"exposure": "US technology-company equity", "fund": r"(?:state street )?technology select sector spdr", "keywords": ["technology", "tech", "semiconductor", "semiconductors", "software", "cloud computing", "artificial intelligence", "chipmakers", "chips"]},
    "XLF": {"exposure": "US financial-company equity", "fund": r"(?:state street )?financial select sector spdr", "keywords": ["financial sector", "financial stocks", "bank", "banks", "banking", "insurance", "insurers", "brokerage", "capital markets"]},
    "XLE": {"exposure": "US energy-company equity; not direct spot-oil exposure", "fund": r"(?:state street )?energy select sector spdr", "keywords": ["energy", "oil", "crude", "natural gas", "petroleum", "oilfield", "refinery", "refineries", "lng"]},
    "XLV": {"exposure": "US healthcare-company equity", "fund": r"(?:state street )?health care select sector spdr", "keywords": ["healthcare", "health care", "biotech", "biotechnology", "pharmaceutical", "pharmaceuticals", "drugmakers", "hospitals", "medical devices"]},
    "XLI": {"exposure": "US industrial-company equity", "fund": r"(?:state street )?industrial select sector spdr", "keywords": ["industrial", "industrials", "aerospace", "defense", "manufacturing", "machinery", "transportation", "railroads", "freight", "airlines"]},
    "XLY": {"exposure": "US consumer-discretionary equity", "fund": r"(?:state street )?consumer discretionary select sector spdr", "keywords": ["consumer discretionary", "retail", "retailers", "automotive", "automakers", "travel", "leisure", "luxury", "homebuilders", "home improvement"]},
    "XLP": {"exposure": "US consumer-staples equity", "fund": r"(?:state street )?consumer staples select sector spdr", "keywords": ["consumer staples", "groceries", "grocery", "grocers", "food", "beverage", "beverages", "tobacco", "supermarkets", "household products"]},
    "XLU": {"exposure": "US utility-company equity", "fund": r"(?:state street )?utilities select sector spdr", "keywords": ["utilities", "utility", "electricity", "electric power", "power grid", "gas utility", "water utility", "regulated utility"]},
    "XLB": {"exposure": "US materials-company equity", "fund": r"(?:state street )?materials select sector spdr", "keywords": ["materials sector", "materials stocks", "mining", "miners", "metals", "steel", "chemicals", "copper", "aluminum", "lithium", "gold mining"]},
    "IEF": {"exposure": "7–10-year US Treasury bonds; rates keywords are broad macro context", "fund": r"ishares\s+7\s*[-–]\s*10\s+year\s+treasury\s+bond", "keywords": ["treasury", "treasuries", "bond yields", "interest rates", "federal reserve", "fed", "monetary policy", "rate cuts", "rate hikes", "10-year yield", "10 year yield"]},
    "TLT": {"exposure": "20-plus-year US Treasury bonds; rates keywords do not prove duration-specific relevance", "fund": r"ishares\s+20\s*(?:\+|plus)\s*year\s+treasury\s+bond", "keywords": ["treasury", "treasuries", "bond yields", "interest rates", "federal reserve", "fed", "monetary policy", "rate cuts", "rate hikes", "long-term bonds", "long duration", "30-year yield"]},
    "TIP": {"exposure": "US Treasury inflation-protected bonds; inflation context", "fund": r"ishares\s+(?:tips|treasury inflation.protected)\s+bond", "keywords": ["inflation", "inflation-protected", "inflation protected", "real yields", "breakeven", "consumer price index", "cpi", "pce", "tips bonds", "tips etf"]},
    "GLD": {"exposure": "Gold bullion trust; no gold-mining equity equivalence assumed", "fund": r"spdr\s+gold\s+(?:shares|trust)", "keywords": ["gold", "bullion", "precious metals"]},
    "DBC": {"exposure": "Broad commodity-futures basket plus collateral; individual commodity headlines are partial context", "fund": r"invesco\s+db\s+commodity\s+index\s+tracking\s+fund", "keywords": ["commodity", "commodities", "commodity futures", "oil", "crude", "natural gas", "metals", "agriculture", "grain", "wheat", "corn", "soybeans", "copper"]},
    "VNQ": {"exposure": "US listed real-estate and REIT equity", "fund": r"vanguard\s+real\s+estate\s+(?:etf|index fund)", "keywords": ["real estate", "reit", "reits", "commercial property", "commercial real estate", "apartments", "office buildings", "property market", "data centers"]},
}


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def plain_text(value: str | None) -> str:
    from bs4 import BeautifulSoup
    return " ".join(BeautifulSoup(html.unescape(value or ""), "html.parser").get_text(" ", strip=True).split())


def canonical_url(value: str) -> str:
    parts = urllib.parse.urlsplit(value.strip())
    kept = [(key, val) for key, val in urllib.parse.parse_qsl(parts.query) if not key.startswith("utm_") and key not in [".tsrc", "guccounter", "guce_referrer", "guce_referrer_sig"]]
    return urllib.parse.urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urllib.parse.urlencode(kept), ""))


def fetch_feed(ticker: str) -> tuple[list[dict], dict]:
    url = RSS + ticker
    started = time.monotonic()
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as response:
        raw = response.read()
    received = stamp()
    digest = hashlib.sha256(raw).hexdigest()
    raw_path = NEWS / "raw_sector_rss" / f"{ticker}_{digest}.xml"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and raw_path.read_bytes() != raw:
        raise ValueError("Immutable raw RSS hash collision")
    if not raw_path.exists():
        raw_path.write_bytes(raw)
    document = ET.fromstring(raw)
    items = document.findall("./channel/item")
    rows, date_errors = [], []
    for item in items:
        title = plain_text(item.findtext("title"))
        summary = plain_text(item.findtext("description"))
        link = canonical_url(item.findtext("link") or "")
        if not title or not link.startswith(("https://", "http://")):
            continue
        published = None
        source_date = item.findtext("pubDate")
        try:
            parsed = parsedate_to_datetime(source_date or "")
            if parsed.tzinfo is None:
                raise ValueError("RSS pubDate has no timezone")
            published = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception as error:
            date_errors.append({"source_url": link, "pubDate": source_date, "error": str(error)})
        body_hash = hashlib.sha256((title + "\n" + summary).encode("utf-8")).hexdigest()
        rows.append({"headline": title, "summary": summary, "source_url": link,
                     "source_name": "Yahoo Finance public ticker RSS", "publisher": item.findtext("source") or None,
                     "source_guid": item.findtext("guid"), "published_at": published,
                     "pubDate_original": source_date, "retrieved_at": received,
                     "source_version": "headline_summary_sha256:" + body_hash, "content_sha256": body_hash,
                     "feed_ticker": ticker, "feed_url": url, "raw_rss_sha256": digest,
                     "version_history_verified": False})
    manifest = {"ticker": ticker, "source_url": url, "retrieved_at": received, "status": "success", "rss_items": len(items), "usable_items": len(rows), "bytes": len(raw), "sha256": digest, "raw_path": raw_path.relative_to(ROOT).as_posix(), "elapsed_seconds": round(time.monotonic()-started, 3), "date_parse_errors": date_errors}
    print(f"RSS {ticker}: {len(rows)} usable articles", flush=True)
    return rows, manifest


def evidence(text: str, pattern: str, field: str, case_sensitive: bool = False) -> list[dict]:
    return [{"field": field, "match": match.group(0)} for match in re.finditer(pattern, text, 0 if case_sensitive else re.I)]


def relevance(article: dict, ticker: str) -> dict:
    exposure = SECTORS[ticker]
    direct = []
    sector = []
    for field in ["headline", "summary"]:
        text = article[field]
        # TIP is an ordinary word too. Require an explicit ticker convention or
        # fund context instead of matching an ordinary upper-case headline word.
        pattern = (r"(?:\$TIP\b|\(TIP\)|\bTIP\s+(?:ETF|fund)\b)" if ticker == "TIP" else rf"(?<![A-Za-z0-9])\$?{ticker}(?![A-Za-z0-9])")
        direct += evidence(text, pattern, field, case_sensitive=True)
        direct += evidence(text, exposure["fund"], field)
        for token in exposure["keywords"]:
            sector += evidence(text, rf"(?<!\w){re.escape(token)}(?!\w)", field)
    selected = direct or sector
    # Feed membership alone never establishes ETF-specific relevance.
    label = "direct_etf_mention" if direct else "sector_keyword_context" if sector else "feed_association_only"
    return {"relevance_label": label, "matched_tokens": sorted(set(item["match"] for item in selected)), "evidence": selected,
            "exposure": exposure["exposure"], "qualification": "direct fund/ticker mention" if direct else "rule-based broad sector/macro context; not a verified ETF-specific return catalyst" if sector else "ticker RSS membership only; no fund/sector text match"}


def main() -> None:
    NEWS.mkdir(parents=True, exist_ok=True)
    all_rows, feeds = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(fetch_feed, ticker): ticker for ticker in SECTORS}
        for job in as_completed(jobs):
            try:
                items, manifest = job.result()
                all_rows.extend(items)
                feeds.append(manifest)
            except Exception as error:
                feeds.append({"ticker": jobs[job], "source_url": RSS + jobs[job], "status": "failure", "error": str(error), "retrieved_at": stamp()})
    # Combine duplicate canonical URL+text versions across feeds. Different
    # text versions are retained and explicitly marked, never silently replaced.
    grouped = {}
    for row in all_rows:
        key = (row["source_url"], row["content_sha256"])
        if key not in grouped:
            article = {name: value for name, value in row.items() if name not in ["feed_ticker", "feed_url", "raw_rss_sha256"]}
            article["article_id"] = "rss-" + hashlib.sha256(row["source_url"].encode()).hexdigest()[:24]
            article["version"] = row["source_version"]
            article["feed_tickers"] = []
            article["source_observations"] = []
            grouped[key] = article
        article = grouped[key]
        article["feed_tickers"].append(row["feed_ticker"])
        article["source_observations"].append({"ticker": row["feed_ticker"], "source_guid": row["source_guid"], "feed_url": row["feed_url"], "retrieved_at": row["retrieved_at"], "raw_rss_sha256": row["raw_rss_sha256"]})
        article["retrieved_at"] = min(article["retrieved_at"], row["retrieved_at"])
    articles = sorted(grouped.values(), key=lambda row: (row["published_at"] or "", row["article_id"]))
    if not articles:
        write_json(REPORT, {"status": "no_usable_public_feed_items", "source_feeds": feeds, "backtest_enabled": False})
        raise RuntimeError("No real RSS articles; no synthetic fallback substituted")
    for article in articles:
        article["feed_tickers"] = sorted(set(article["feed_tickers"]))
        article["relevance_by_etf"] = {}
        for ticker in SECTORS:
            match = relevance(article, ticker)
            if match["relevance_label"] != "feed_association_only" or ticker in article["feed_tickers"]:
                article["relevance_by_etf"][ticker] = match
        article["intended_etfs"] = [ticker for ticker, match in article["relevance_by_etf"].items() if match["relevance_label"] != "feed_association_only"]
    started = time.monotonic()
    scores = classify([article["headline"] + ". " + article["summary"] for article in articles])
    elapsed = round(time.monotonic() - started, 3)
    scored_at = stamp()
    exploded = []
    by_etf = {ticker: {label: 0 for label in ["direct_etf_mention", "sector_keyword_context", "feed_association_only", "total"]} for ticker in SECTORS}
    label_counts = {key: 0 for key in ["direct_etf_mention", "sector_keyword_context", "feed_association_only"]}
    for article, score in zip(articles, scores):
        assert abs(sum(score[key] for key in ["prob_positive", "prob_negative", "prob_neutral"]) - 1) < 1e-5
        article.update(score)
        article.update({"scored_at": scored_at, "available_at": max([article["published_at"] or article["retrieved_at"], article["retrieved_at"], scored_at], key=lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))), "model_repository": REPO, "model_revision": REVISION,
                        "text_policy": "RSS headline plus publisher-supplied RSS description only, truncated at 384 tokenizer tokens", "historical_backtest_eligible": False,
                        "sentiment_scope": "whole headline+summary sentiment, not separately targeted ETF-return sentiment or a trading forecast"})
        for ticker, match in article["relevance_by_etf"].items():
            row = {key: article[key] for key in ["article_id", "version", "headline", "summary", "source_url", "source_name", "publisher", "published_at", "retrieved_at", "available_at", "scored_at", "prob_positive", "prob_negative", "prob_neutral", "sentiment_balance", "predicted_label", "historical_backtest_eligible", "model_revision"]}
            row.update({"ticker": ticker, "relevance_label": match["relevance_label"], "matched_tokens": "|".join(match["matched_tokens"]), "evidence": json.dumps(match["evidence"], ensure_ascii=False), "exposure": match["exposure"], "relevance_qualification": match["qualification"], "feed_membership": ticker in article["feed_tickers"], "feed_tickers": "|".join(article["feed_tickers"]), "intended_etfs": "|".join(article["intended_etfs"])})
            exploded.append(row)
            by_etf[ticker][match["relevance_label"]] += 1
            by_etf[ticker]["total"] += 1
            label_counts[match["relevance_label"]] += 1
    with (NEWS / "sector_etf_news_scored.jsonl").open("w", encoding="utf-8") as output:
        for article in articles:
            output.write(json.dumps(article, ensure_ascii=False, allow_nan=False) + "\n")
    csv_path = NEWS / "sector_etf_news_scored.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(exploded[0]))
        writer.writeheader()
        writer.writerows(exploded)
    published = [row["published_at"] for row in articles if row["published_at"]]
    unique_urls = len({row["source_url"] for row in articles})
    report = {
        "status": "real_current_etf_sector_rss_scored_locally", "generated_at_utc": scored_at,
        "source_feeds": sorted(feeds, key=lambda row: row["ticker"]), "successful_feeds": sum(row["status"] == "success" for row in feeds), "requested_feeds": len(SECTORS),
        "rss_item_observations": len(all_rows), "unique_articles": unique_urls, "article_text_versions": len(articles), "scored_articles": len(articles), "article_etf_associations": len(exploded),
        "qualified_article_etf_associations": label_counts["direct_etf_mention"] + label_counts["sector_keyword_context"],
        "relevance_counts": label_counts, "by_etf": by_etf, "oldest_publication": min(published) if published else None, "newest_publication": max(published) if published else None,
        "source_names": sorted(set(row["source_name"] for row in articles)), "publisher_missing_article_versions": sum(row["publisher"] is None for row in articles),
        "sentiment_label_counts": {label: sum(row["predicted_label"] == label for row in articles) for label in ["positive", "negative", "neutral"]},
        "local_inference_seconds": elapsed, "model_repository": REPO, "model_revision": REVISION,
        "model_revision_last_modified": json.loads((MODEL / "model_manifest.json").read_text(encoding="utf-8"))["revision_last_modified"],
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "historical_backtest_eligible": False, "backtest_enabled": False,
        "deduplication": "Canonical URL strips RSS/UTM tracking and fragments; identical canonical URL+headline-summary SHA256 across feeds scored once. Different text versions retained. Cross-publisher syndicated equivalents with different URLs are not asserted deduplicated.",
        "availability_rule": "available_at=max(exact timezone-aware RSS pubDate when present, actual current retrieval, local scoring completion). Never backdate actual receipt to article publication.",
        "relevance_policy": {"direct_etf_mention": "Exact uppercase ticker with boundaries (TIP requires explicit ticker/fund context), or explicit fund-name match in headline/summary", "sector_keyword_context": "Documented rules match sector/macro keywords in headline/summary; heuristic context, not independently verified ETF-specific causal news", "feed_association_only": "Feed lists article but headline/summary supplies no fund or sector evidence. Not counted as qualified ETF/sector content."},
        "sector_mapping": SECTORS,
        "coverage_limitations": ["Current finite public RSS windows, not a historical archive or complete news census; oldest/newest returned dates describe this snapshot only.", "Yahoo feed membership alone does not establish ETF relevance; generic market content is separately labeled when no explicit fund/sector evidence exists.", "Only headline and supplied RSS summary scored; summaries may be truncated and underlying publisher may be unspecified. No full article scraped.", "Rules map sectors/exposures, not historical holdings. Broad interest-rate news may map both IEF and TLT without distinguishing duration; DBC individual-commodity keywords are partial broad-basket context.", "FinBERT outputs whole-text sentiment probabilities, not an ETF-targeted sentiment label, an ETF return forecast, or evidence of trading profitability.", "Original/update history, historical first receipts, source outages and historical search completeness are unavailable. Current model revision last modified October 2026; no historical deployment claim.", "Distinct syndicated URLs may remain; this is not a licensed/comprehensive historical deduplication service.", "Public access does not grant republication rights; RSS and publisher terms remain applicable to use of the saved local research snapshot."],
        "later_archive_option": {"source": "Alpaca historical news", "documentation": "https://docs.alpaca.markets/us/docs/historical-news-data", "status": "not accessed; future user-supplied entitled export or explicitly authorized account could support a separate historical coverage audit", "caveat": "Documentation describes history from 2015 and symbol queries; actual ETF coverage, original/update timestamps, receipts, licensing and completeness still require verification."},
        "artifacts": {"article_versions_jsonl": "data/news/sector_etf_news_scored.jsonl", "article_etf_associations_csv": "data/news/sector_etf_news_scored.csv", "raw_rss": "data/news/raw_sector_rss"},
        "csv_semantics": "One row per article-version × ETF association. Scoring occurs once per unique URL+text version; do not count duplicated cross-ETF rows as independent articles.",
    }
    assert len({(article["article_id"], article["version"]) for article in articles}) == len(articles)
    assert all(not article["historical_backtest_eligible"] for article in articles)
    raw_checks = sum(hashlib.sha256((ROOT / feed["raw_path"]).read_bytes()).hexdigest() == feed["sha256"] for feed in feeds if feed["status"] == "success")
    assert raw_checks == report["successful_feeds"]
    report["quality_audit"] = {"status": "passed", "raw_rss_hash_matches": raw_checks, "unique_article_version_ids": len(articles), "valid_probability_vectors": len(scores), "availability_not_backdated": len(articles), "historical_backtest_eligible_articles": 0, "date_parse_errors": sum(len(feed.get("date_parse_errors", [])) for feed in feeds)}
    write_json(REPORT, report)
    print(json.dumps({key: report[key] for key in ["successful_feeds", "requested_feeds", "rss_item_observations", "unique_articles", "article_text_versions", "scored_articles", "article_etf_associations", "qualified_article_etf_associations", "relevance_counts", "sentiment_label_counts", "oldest_publication", "newest_publication", "local_inference_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
