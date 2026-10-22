"""Local FinBERT and honest news provenance for Swing Tree research.

No sentiment output from this module is automatically admitted to a backtest.
Historical web pages retrieved now are not point-in-time news records. Run:
  python src/finbert_news.py download-model
  python src/finbert_news.py fetch-fed --start-year 2006 --end-year 2025
  python src/finbert_news.py smoke-test
  python src/finbert_news.py score-fed
  python src/finbert_news.py validate-export path/to/news.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "data" / "news"
MODEL = ROOT / "models" / "finbert"
REPORT = ROOT / "reports" / "finbert_status.json"
REPO = "ProsusAI/finbert"
REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
UNIVERSE = "XLK XLF XLE XLV XLI XLY XLP XLU XLB IEF TLT TIP GLD DBC VNQ".split()
HEADERS = {"User-Agent": "Mozilla/5.0"}
EXPORT_FIELDS = ["id", "version", "headline", "text", "ticker", "entity", "published_at", "updated_at", "first_received_at", "available_at", "source_url", "source", "timestamp_precision", "version_history_verified"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def update_status(section: str, value: dict) -> None:
    status = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    status[section] = value
    status["generated_at_utc"] = now()
    status["backtest_news_enabled"] = False
    status["required_before_news_ablation"] = [
        "Entitled historical ETF-relevant news with original and updated article versions, UTC publication and first-receipt timestamps, and source coverage/outage history.",
        "Documented entity-to-ETF mapping, deduplication, processing latency and source/data usage rights.",
        "A checkpoint with documented availability before the evaluated deployment interval, or explicit retrospective representation-study labeling.",
        "Matched factor-only and factor-plus-news recipes on identical eligible dates, chronological folds and execution costs.",
    ]
    write_json(REPORT, status)


def request_bytes(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=20) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code != 429:
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(url)


def download_model() -> None:
    MODEL.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(request_bytes(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}"))
    if metadata["sha"] != REVISION:
        raise ValueError("Model revision did not match the immutable requested commit")
    write_json(MODEL / "repository_metadata.json", metadata)
    files = ["config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "model.safetensors", "README.md"]
    manifest = []
    for name in files:
        target = MODEL / name
        if not target.exists():
            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true"
            part = target.with_suffix(target.suffix + ".partial")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=180) as response, part.open("wb") as output:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                    part.replace(target)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        digest = hashlib.sha256()
        with target.open("rb") as input_file:
            while block := input_file.read(1024 * 1024):
                digest.update(block)
        manifest.append({"file": name, "bytes": target.stat().st_size, "sha256": digest.hexdigest()})
        print(f"Saved {name}: {target.stat().st_size:,} bytes", flush=True)
    card = (MODEL / "README.md").read_text(encoding="utf-8")
    entry = {
        "status": "downloaded_locally", "repository": REPO, "revision": REVISION,
        "revision_last_modified": metadata.get("lastModified"), "retrieved_at_utc": now(),
        "weights_format": "safetensors", "trust_remote_code": False,
        "license": metadata.get("cardData", {}).get("license"),
        "license_source": f"https://huggingface.co/{REPO}/blob/{REVISION}/README.md",
        "training_provenance": "BERT finance-domain adaptation using Reuters TRC2; sentiment fine-tuning using Financial PhraseBank (Malo et al., 2014), as stated by model card.",
        "training_cutoff_exact": None,
        "vintage_limitation": "This exact repository revision was last modified in 2026. Its availability cannot be backdated for historical deployment claims. No historical trading claim uses it here.",
        "label_mapping": json.loads((MODEL / "config.json").read_text())["id2label"],
        "files": manifest,
    }
    write_json(MODEL / "model_manifest.json", entry)
    update_status("model", entry)


def parse_fed(url: str) -> dict:
    from bs4 import BeautifulSoup
    content = request_bytes(url)
    received = now()
    soup = BeautifulSoup(content, "html.parser")
    date_match = re.search(r"monetary(\d{8})([a-z])\.htm", url)
    if not date_match:
        raise ValueError("Expected dated official FOMC URL")
    release_date = datetime.strptime(date_match[1], "%Y%m%d").replace(tzinfo=timezone.utc)
    release_tag = soup.select_one(".releaseTime")
    release_text = release_tag.get_text(" ", strip=True) if release_tag else ""
    stamp = re.search(r"(\d{1,2}):(\d\d)\s*([ap])\.?m\.?\s*(EDT|EST)", release_text, re.I)
    precision = "date_only_upper_bound"
    published = release_date.replace(hour=23, minute=59, second=59)
    if stamp:
        hour = int(stamp[1]) % 12 + (12 if stamp[3].lower() == "p" else 0)
        utc_offset = 4 if stamp[4].upper() == "EDT" else 5
        published = release_date.replace(hour=hour, minute=int(stamp[2])) + timedelta(hours=utc_offset)
        precision = "minute_from_page_EDT_EST"
    candidates = soup.select("div.col-xs-12.col-sm-8.col-md-8")
    if not candidates:
        raise ValueError("FOMC body selector absent; preserve failure rather than guess content")
    # The same grid classes also wrap the release heading. Select the actual
    # paragraph body by text length, avoiding an empty headline container.
    body = max(candidates, key=lambda div: sum(len(p.get_text()) for p in div.find_all("p", recursive=False)))
    paragraphs = []
    for paragraph in body.find_all("p", recursive=False):
        text = paragraph.get_text(" ", strip=True)
        if text.startswith(("For media inquiries", "Implementation Note", "Voting for", "Voting against")):
            continue
        if text:
            paragraphs.append(text)
    text = "\n\n".join(paragraphs)
    if len(text) < 200:
        raise ValueError("Insufficient policy text")
    title_tag = soup.select_one("h3.title")
    title = title_tag.get_text(" ", strip=True) if title_tag else "Federal Reserve issues FOMC statement"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    ident = "fed-fomc-" + date_match[1] + date_match[2]
    raw_path = NEWS / "raw_fed" / (ident + ".html")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)
    updated = soup.select_one("#lastUpdate")
    return {
        "id": ident, "version": "retrieved_content_sha256:" + digest,
        "headline": title, "text": text, "ticker": "|".join(UNIVERSE),
        "entity": "Federal Open Market Committee; macro-policy exposure, not direct ETF-specific reporting",
        "published_at": published.isoformat().replace("+00:00", "Z"),
        "published_at_precision": precision, "release_time_displayed": release_text,
        "updated_at": None, "updated_at_displayed": updated.get_text(" ", strip=True) if updated else None,
        "first_received_at": received, "available_at": received,
        "hypothetical_release_plus_7d_at": (release_date + timedelta(days=7, hours=23, minutes=59, seconds=59)).isoformat().replace("+00:00", "Z"),
        "hypothetical_availability_warning": "Research illustration only; does not repair missing original-version history. Real available_at is retrieval time.",
        "source_url": url, "source": "Federal Reserve Board public historical statement archive",
        "content_sha256": digest, "raw_html_sha256": hashlib.sha256(content).hexdigest(),
        "version_history_verified": False, "historical_backtest_eligible": False,
        "mapping_method": "Manual broad macro policy relevance to all 15 instruments; no directional attribution assumed",
        "mapping_confidence": "broad_macro_context_only",
    }


def write_export_contract() -> None:
    NEWS.mkdir(parents=True, exist_ok=True)
    with (NEWS / "news_export_template.csv").open("w", newline="", encoding="utf-8") as output:
        csv.writer(output).writerow(EXPORT_FIELDS)
    write_json(NEWS / "news_ingestion_contract.json", {
        "schema_version": 1, "fields": EXPORT_FIELDS,
        "time_format": "Timezone-aware ISO 8601 UTC, e.g. 2024-09-18T18:00:00Z. Never infer a missing timezone.",
        "timestamp_precision_values": ["minute", "second"],
        "id_and_version": "Stable source article ID plus original/update version; preserve each version separately.",
        "required": ["id", "version", "headline", "ticker", "entity", "published_at", "first_received_at", "available_at", "source_url", "source", "version_history_verified"],
        "available_at_rule": "max(published_at, version-specific updated_at when applicable, actual historical first_received_at) plus measured processing latency. Export time today does not establish historical receipt.",
        "entity_mapping": "Use pipe-separated verified ETF tickers, or a dated documented entity-to-ETF mapping; no current holdings backfilled historically.",
        "dedup": "Preserve source ID/version and text SHA256; collapse syndicated duplicates before aggregation; version-specific dedup must not erase earlier availability.",
        "coverage": "Supply coverage start/end, symbols/entities searched, pagination completeness, outages, and entitlements in a sidecar manifest. Zero articles and source unavailable are distinct states.",
        "features": "prob_positive, prob_negative, prob_neutral, balance=positive-negative; aggregate by available_at strictly before weekly decision cutoff; include news count, disagreement, recency, and source outage flag.",
        "backtest_default": "disabled until source/version/availability/coverage gates are reviewed",
    })


def fetch_fed(start_year: int, end_year: int) -> None:
    from bs4 import BeautifulSoup
    write_export_contract()
    urls = set()
    failures = []
    navigation_url = "https://www.federalreserve.gov/newsevents/pressreleases.htm"
    navigation = BeautifulSoup(request_bytes(navigation_url), "html.parser")
    year_indexes = {}
    for anchor in navigation.find_all("a", href=True):
        label = anchor.get_text(" ", strip=True)
        match = re.fullmatch(r"(\d{4})( FOMC)?", label)
        if match:
            key = int(match[1])
            if key not in year_indexes or match[2]:
                year_indexes[key] = urllib.parse.urljoin(navigation_url, anchor["href"])
    for year in range(start_year, end_year + 1):
        index = year_indexes.get(year)
        if not index:
            failures.append({"year": year, "reason": "No year link in official press-release navigation"})
            continue
        try:
            content = request_bytes(index)
            raw = NEWS / "raw_fed" / f"index_{year}.html"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(content)
            soup = BeautifulSoup(content, "html.parser")
            found = set()
            for anchor in soup.find_all("a", href=True):
                title = anchor.get_text(" ", strip=True)
                href = urllib.parse.urljoin(index, anchor["href"])
                if "FOMC statement" in title and re.search(r"monetary\d{8}[a-z]\.htm$", href):
                    found.add(href)
            urls.update(found)
            print(f"Fed archive {year}: {len(found)} statement links", flush=True)
            if not found:
                failures.append({"url": index, "reason": "No recognized FOMC statement links; older format may differ"})
        except Exception as error:
            failures.append({"url": index, "reason": str(error)})
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {pool.submit(parse_fed, url): url for url in sorted(urls)}
        for task in as_completed(tasks):
            try:
                rows.append(task.result())
            except Exception as error:
                failures.append({"url": tasks[task], "reason": str(error)})
    rows.sort(key=lambda row: (row["published_at"], row["id"]))
    # Dedup uses source identity and content hash, never future article revisions.
    dedup = {(row["id"], row["content_sha256"]): row for row in rows}
    rows = list(dedup.values())
    with (NEWS / "fed_fomc_archive.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    entry = {
        "status": "real_historical_archive_retrieved_but_not_point_in_time_eligible",
        "requested_years": [start_year, end_year], "articles": len(rows),
        "first_publication": rows[0]["published_at"] if rows else None,
        "last_publication": rows[-1]["published_at"] if rows else None,
        "source_navigation": navigation_url,
        "source_indexes": {str(year): year_indexes.get(year) for year in range(start_year, end_year + 1)},
        "sources": "Official Federal Reserve historical policy statements only; not comprehensive financial or ETF-specific news",
        "historical_backtest_eligible_articles": 0,
        "availability_policy": "available_at is actual current retrieval timestamp; a separate release-plus-seven-days illustration never enables trading/backtests",
        "missing_original_article_versions": True, "missing_historical_receipt_times": True,
        "source_coverage_or_outage_history_known": False,
        "date_only_publish_rows": sum(row["published_at_precision"] == "date_only_upper_bound" for row in rows),
        "mapping": "All 15 ETFs via broad macro-policy context; manual, no direct ticker relevance assumption",
        "failed_requests": failures,
    }
    write_json(NEWS / "fed_corpus_manifest.json", entry)
    update_status("news_corpus", entry)
    print(f"Saved {len(rows)} real historical statements; {len(failures)} failures; backtest eligible=0", flush=True)


def classify(texts: list[str], batch_size: int = 8) -> list[dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False, use_safetensors=True)
    model.eval()
    labels = model.config.id2label
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=384, return_tensors="pt")
            probabilities = torch.softmax(model(**batch).logits, dim=-1).cpu().tolist()
            for values in probabilities:
                row = {"prob_" + labels[index].lower(): value for index, value in enumerate(values)}
                row["sentiment_balance"] = row["prob_positive"] - row["prob_negative"]
                row["predicted_label"] = labels[max(range(len(values)), key=values.__getitem__)]
                outputs.append(row)
    return outputs


def smoke_test() -> None:
    texts = [
        "The company reported record profits and raised its earnings forecast.",
        "The company reported a large loss and cut its earnings forecast.",
        "The fund publishes its annual report on Tuesday.",
    ]
    started = time.monotonic()
    results = classify(texts)
    entries = [{"text": text, "synthetic": True, "purpose": "software smoke test only; not historical news", **result} for text, result in zip(texts, results)]
    if not all(abs(sum(row[key] for key in ("prob_positive", "prob_negative", "prob_neutral")) - 1) < 1e-5 for row in results):
        raise AssertionError("Probabilities do not sum to one")
    if results[0]["predicted_label"] != "positive" or results[1]["predicted_label"] != "negative":
        raise AssertionError("Simple positive/negative smoke-test behavior failed")
    entry = {
        "status": "passed_real_local_cpu_inference", "model_revision": REVISION,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "safetensors"]},
        "synthetic_test_sentences": entries,
        "accuracy_claim": "None. Three authored examples test the software path only.",
        "historical_performance_claim": "None. FinBERT classifies sentiment, not future ETF returns.",
    }
    write_json(NEWS / "synthetic_smoke_test.json", entry)
    update_status("local_inference", entry)
    print(json.dumps(entry, indent=2), flush=True)


def score_fed() -> None:
    rows = [json.loads(line) for line in (NEWS / "fed_fomc_archive.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("No real archive records to score")
    started = time.monotonic()
    scores = classify([row["text"] for row in rows])
    with (NEWS / "fed_fomc_sentiment.jsonl").open("w", encoding="utf-8") as output:
        for row, score in zip(rows, scores):
            output.write(json.dumps({"id": row["id"], "version": row["version"], "published_at": row["published_at"], "available_at": row["available_at"], "source_url": row["source_url"], "affected_etfs": row["ticker"], "model_repository": REPO, "model_revision": REVISION, "text_policy": "first 384 tokenizer tokens of policy statement body; not ETF return direction", "historical_backtest_eligible": False, **score}, ensure_ascii=False) + "\n")
    labels = {label: sum(score["predicted_label"] == label for score in scores) for label in ["positive", "negative", "neutral"]}
    entry = {"status": "scored_real_archive_locally_excluded_from_backtests", "rows_scored": len(rows), "elapsed_seconds": round(time.monotonic() - started, 3), "label_counts": labels, "model_revision": REVISION, "text_policy": "Truncate each statement to first 384 tokenizer tokens; a reproducible feasibility demonstration, not an approved trading aggregation.", "backtest_eligible": False}
    update_status("archive_inference", entry)
    print(json.dumps(entry, indent=2), flush=True)


def validate_export(path: Path) -> None:
    write_export_contract()
    errors = []
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        missing = set(EXPORT_FIELDS) - set(reader.fieldnames or [])
        if missing:
            errors.append(f"Missing fields: {sorted(missing)}")
        count = 0
        for count, row in enumerate(reader, 1):
            identity = (row.get("id"), row.get("version"))
            if identity in seen:
                errors.append(f"Row {count}: duplicated source ID and version")
            seen.add(identity)
            for field in ["id", "version", "headline", "ticker", "entity", "source_url", "source"]:
                if not row.get(field):
                    errors.append(f"Row {count}: blank {field}")
            stamps = {}
            for field in ["published_at", "updated_at", "first_received_at", "available_at"]:
                if field == "updated_at" and not row.get(field):
                    continue
                try:
                    stamp = datetime.fromisoformat(row.get(field, "").replace("Z", "+00:00"))
                    if stamp.utcoffset() != timedelta(0):
                        raise ValueError("UTC timezone required")
                    stamps[field] = stamp
                except Exception:
                    errors.append(f"Row {count}: {field} must be timezone-aware UTC")
            if all(field in stamps for field in ["published_at", "first_received_at", "available_at"]) and stamps["available_at"] < max(stamps.values()):
                errors.append(f"Row {count}: available_at precedes publication, version update or historical receipt")
            if row.get("timestamp_precision") not in ["minute", "second"]:
                errors.append(f"Row {count}: minute or second timestamp precision required")
            if any(ticker not in UNIVERSE for ticker in row.get("ticker", "").split("|")):
                errors.append(f"Row {count}: unrecognized or unmapped ETF ticker")
            if str(row.get("version_history_verified", "")).lower() not in ["true", "1"]:
                errors.append(f"Row {count}: original/update version history is unverified")
    result = {"source_file": str(path), "rows": count, "schema_errors": errors, "schema_valid": not errors, "backtest_approved": False, "note": "Passing schema does not establish source completeness, rights, entity relevance, historical receipt correctness, or vintage eligibility."}
    write_json(NEWS / "export_validation.json", result)
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download-model")
    fed = commands.add_parser("fetch-fed")
    fed.add_argument("--start-year", type=int, default=2006)
    fed.add_argument("--end-year", type=int, default=2025)
    commands.add_parser("smoke-test")
    commands.add_parser("score-fed")
    validate = commands.add_parser("validate-export")
    validate.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "download-model":
        download_model()
    elif args.command == "fetch-fed":
        fetch_fed(args.start_year, args.end_year)
    elif args.command == "smoke-test":
        smoke_test()
    elif args.command == "score-fed":
        score_fed()
    elif args.command == "validate-export":
        validate_export(args.path)
