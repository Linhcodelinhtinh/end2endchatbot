"""High-scale KB crawler for HTML + PDF.

Features:
  - Breadth-first crawl with configurable depth/limits
  - Concurrent fetching via thread pool
  - Retry + exponential backoff for transient failures (especially 429/5xx)
  - Domain-aware pacing to reduce rate limiting
  - URL canonicalization + resume from existing manifest
  - Progressive checkpoints and detailed runtime stats
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
import yaml
from bs4 import BeautifulSoup

MANIFEST_FIELDS = [
    "doc_id",
    "source_id",
    "category",
    "priority",
    "url",
    "domain",
    "crawl_time",
    "status",
    "status_code",
    "content_type",
    "file_path",
    "error",
]

TRACKING_QUERY_PREFIXES = ("utm_", "fbclid", "gclid", "mc_")
NON_CONTENT_EXTENSIONS = {
    ".css",
    ".js",
    ".json",
    ".xml",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".mp3",
    ".wav",
    ".mp4",
    ".avi",
    ".mov",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
}
LOW_VALUE_URL_PATTERNS = (
    "/wp-json/",
    "/feed",
    "/comment",
    "/login",
    "/dang-nhap",
    "/register",
    "/dang-ky",
    "/search",
    "/tag/",
)
URL_PATTERN_RE = re.compile(r"""https?://[^\s"'<>]+""", flags=re.IGNORECASE)


@dataclass
class FetchResult:
    row: dict[str, str]
    links: list[str]
    is_success: bool
    is_pdf: bool


class DomainPacer:
    """Simple per-domain pacing to avoid burst requests."""

    def __init__(self, min_interval_seconds: float):
        self.min_interval_seconds = max(min_interval_seconds, 0.0)
        self._next_allowed: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, domain: str) -> None:
        if self.min_interval_seconds <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                allowed_at = self._next_allowed.get(domain, now)
                if now >= allowed_at:
                    self._next_allowed[domain] = now + self.min_interval_seconds
                    return
                sleep_for = allowed_at - now
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.25))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect raw KB files at large scale.")
    parser.add_argument("--sources", default="sources.yaml")
    parser.add_argument("--raw-dir", default="data/knowledge_base/raw")
    parser.add_argument("--manifest", default="data/knowledge_base/source_manifest.csv")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent fetch workers.")
    parser.add_argument("--progress-every", type=int, default=25, help="Progress log interval.")
    parser.add_argument("--max-runtime-seconds", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=0, help="Global fetch cap, 0=unlimited.")
    parser.add_argument("--sleep-seconds", type=float, default=0.1, help="Small post-success delay.")
    parser.add_argument("--domain-min-interval", type=float, default=0.15)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for transient errors.")
    parser.add_argument("--retry-base-delay", type=float, default=1.2)
    parser.add_argument("--retry-429", action="store_true", default=True)
    parser.add_argument("--no-retry-429", dest="retry_429", action="store_false")
    parser.add_argument("--save-manifest-every-source", action="store_true")
    parser.add_argument("--resume-from-manifest", action="store_true")
    return parser.parse_args()


def load_sources(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_hms(seconds: float) -> str:
    total = int(max(seconds, 0))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def log(message: str) -> None:
    print(message, flush=True)


def stable_name(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    cleaned_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_lower = key.lower()
        if key_lower.startswith(TRACKING_QUERY_PREFIXES) or key_lower in TRACKING_QUERY_PREFIXES:
            continue
        cleaned_query.append((key, value))
    cleaned_query.sort()

    return urlunparse((scheme, netloc, path, "", urlencode(cleaned_query), ""))


def is_allowed_url(
    url: str,
    allowed_domains: set[str],
    blocked_keywords: list[str],
    include_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False

    domain_ok = any(parsed.netloc.endswith(domain) for domain in allowed_domains)
    if not domain_ok:
        return False

    url_l = url.lower()
    path_l = parsed.path.lower()
    if any(path_l.endswith(ext) for ext in NON_CONTENT_EXTENSIONS):
        return False
    if any(pattern in url_l for pattern in LOW_VALUE_URL_PATTERNS):
        return False
    if len(parsed.query) > 220:
        return False

    if any(keyword in url_l for keyword in blocked_keywords):
        return False
    if exclude_keywords and any(keyword in url_l for keyword in exclude_keywords):
        return False
    if include_keywords and not any(keyword in url_l for keyword in include_keywords):
        return False
    return True


def extract_links(base_url: str, html_text: str) -> list[str]:
    soup = BeautifulSoup(html_text, "html.parser")
    links: set[str] = set()

    attr_candidates = ("href", "src", "data-src", "data-url", "action")
    for tag in soup.find_all(True):
        for attr_name in attr_candidates:
            value = tag.attrs.get(attr_name)
            if not value:
                continue
            if isinstance(value, list):
                for item in value:
                    if item:
                        links.add(urljoin(base_url, str(item)))
            else:
                links.add(urljoin(base_url, str(value)))

    for tag in soup.find_all(True):
        onclick = tag.attrs.get("onclick")
        if not onclick:
            continue
        match = re.search(r"""['"](https?://[^'"]+|/[^'"]+)['"]""", str(onclick))
        if match:
            links.add(urljoin(base_url, match.group(1)))

    # Parse URL-like strings embedded in scripts/inline text (helps discover hidden PDF endpoints).
    for match in URL_PATTERN_RE.finditer(html_text):
        links.add(match.group(0).strip(".,);]}>\"'"))

    cleaned: list[str] = []
    for link in links:
        if not link:
            continue
        link = link.strip().strip("\\")
        if link.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        if "\\" in link:
            continue
        cleaned.append(link)
    return cleaned


def is_pdf_candidate(url: str, content_type: str) -> bool:
    url_l = url.lower()
    ctype_l = content_type.lower()
    if "application/pdf" in ctype_l:
        return True
    if any(token in ctype_l for token in ("application/octet-stream", "application/download")) and (
        ".pdf" in url_l or "pdf" in url_l
    ):
        return True
    if ".pdf" in url_l:
        return True
    return False


def looks_like_pdf(binary_content: bytes) -> bool:
    # Some endpoints return HTML error pages while keeping a *.pdf URL.
    prefix = (binary_content or b"")[:1024].lstrip()
    return prefix.startswith(b"%PDF-")


def queue_priority(url: str) -> int:
    url_l = url.lower()
    if ".pdf" in url_l:
        return 0
    if any(token in url_l for token in ("/home/?c", "/dao-tao", "/tuyen-sinh", "/quy-che", "/van-ban")):
        return 1
    return 2


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_urls_from_manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            url = row.get("url", "").strip()
            if not url:
                continue
            seen.add(canonicalize_url(url))
    return seen


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({field: row.get(field, "") for field in MANIFEST_FIELDS})
    return rows


def fetch_with_retry(
    url: str,
    headers: dict[str, str],
    timeout_seconds: float,
    max_retries: int,
    retry_base_delay: float,
    retry_429: bool,
    pacer: DomainPacer,
) -> requests.Response:
    last_exc: Exception | None = None
    domain = urlparse(url).netloc
    for attempt in range(max_retries + 1):
        pacer.wait(domain)
        try:
            req_headers = dict(headers)
            if "cdnportal.vnu.edu.vn" in domain and "Referer" not in req_headers:
                req_headers["Referer"] = "https://vnu.edu.vn/"
            response = requests.get(url, headers=req_headers, timeout=timeout_seconds, allow_redirects=True)
            retriable_status = response.status_code in {429, 500, 502, 503, 504}
            if retriable_status and attempt < max_retries:
                if response.status_code == 429 and not retry_429:
                    return response
                delay = retry_base_delay * (2**attempt)
                time.sleep(delay)
                continue
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = retry_base_delay * (2**attempt)
            time.sleep(delay)
    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected retry state")


def build_error_row(
    doc_id: str,
    source_id: str,
    category: str,
    priority: str,
    url: str,
    status: str,
    status_code: str = "",
    content_type: str = "",
    file_path: str = "",
    error: str = "",
) -> dict[str, str]:
    return {
        "doc_id": doc_id,
        "source_id": source_id,
        "category": category,
        "priority": priority,
        "url": url,
        "domain": urlparse(url).netloc,
        "crawl_time": utc_now(),
        "status": status,
        "status_code": status_code,
        "content_type": content_type,
        "file_path": file_path,
        "error": error,
    }


def fetch_url(
    url: str,
    source_id: str,
    category: str,
    priority: str,
    headers: dict[str, str],
    timeout_seconds: float,
    max_retries: int,
    retry_base_delay: float,
    retry_429: bool,
    pacer: DomainPacer,
    html_dir: Path,
    pdf_dir: Path,
    sleep_seconds: float,
) -> FetchResult:
    doc_id = stable_name(url)
    try:
        response = fetch_with_retry(
            url=url,
            headers=headers,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_429=retry_429,
            pacer=pacer,
        )

        status_code = str(response.status_code)
        content_type = response.headers.get("Content-Type", "").lower()
        final_url = canonicalize_url(response.url or url)
        if response.status_code >= 400:
            return FetchResult(
                row=build_error_row(
                    doc_id=doc_id,
                    source_id=source_id,
                    category=category,
                    priority=priority,
                    url=final_url,
                    status="http_error",
                    status_code=status_code,
                    content_type=content_type,
                    error=f"http_{status_code}",
                ),
                links=[],
                is_success=False,
                is_pdf=False,
            )

        if is_pdf_candidate(final_url, content_type):
            if looks_like_pdf(response.content):
                target_path = pdf_dir / f"{doc_id}.pdf"
                target_path.write_bytes(response.content)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                return FetchResult(
                    row=build_error_row(
                        doc_id=doc_id,
                        source_id=source_id,
                        category=category,
                        priority=priority,
                        url=final_url,
                        status="ok",
                        status_code=status_code,
                        content_type=content_type,
                        file_path=str(target_path),
                    ),
                    links=[],
                    is_success=True,
                    is_pdf=True,
                )
            content_type = "text/html; charset=utf-8"

        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return FetchResult(
                row=build_error_row(
                    doc_id=doc_id,
                    source_id=source_id,
                    category=category,
                    priority=priority,
                    url=final_url,
                    status="ignored_non_content",
                    status_code=status_code,
                    content_type=content_type,
                    error="not_html_or_pdf",
                ),
                links=[],
                is_success=False,
                is_pdf=False,
            )

        text = response.text
        target_path = html_dir / f"{doc_id}.html"
        target_path.write_text(text, encoding="utf-8", errors="ignore")
        links = extract_links(final_url, text)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return FetchResult(
            row=build_error_row(
                doc_id=doc_id,
                source_id=source_id,
                category=category,
                priority=priority,
                url=final_url,
                status="ok",
                status_code=status_code,
                content_type=content_type,
                file_path=str(target_path),
            ),
            links=links,
            is_success=True,
            is_pdf=False,
        )
    except Exception as exc:  # pragma: no cover
        return FetchResult(
            row=build_error_row(
                doc_id=doc_id,
                source_id=source_id,
                category=category,
                priority=priority,
                url=url,
                status="exception",
                error=str(exc),
            ),
            links=[],
            is_success=False,
            is_pdf=False,
        )


def collect_for_source(
    source: dict[str, Any],
    crawl_cfg: dict[str, Any],
    raw_dir: Path,
    existing_urls: set[str],
    args: argparse.Namespace,
    started_at: float,
    source_idx: int,
    source_count: int,
    global_attempted_ref: list[int],
) -> list[dict[str, str]]:
    max_depth = int(crawl_cfg.get("max_depth", 2))
    max_pages_per_seed = int(crawl_cfg.get("max_pages_per_seed", 80))
    timeout_seconds = float(crawl_cfg.get("request_timeout_seconds", args.request_timeout))
    user_agent = str(crawl_cfg.get("user_agent", "End2EndNLPBot/2.0"))
    allowed_domains = set(crawl_cfg.get("allowed_domains", []))
    blocked_keywords = list(crawl_cfg.get("blocked_path_keywords", []))
    allow_subpages = bool(source.get("allow_subpages", False))

    include_keywords = list(source.get("include_url_keywords", []) or [])
    exclude_keywords = list(source.get("exclude_url_keywords", []) or [])

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/pdf,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi,en;q=0.8",
    }
    source_id = str(source["id"])
    category = str(source.get("category", "unknown"))
    priority = str(source.get("priority", "medium"))
    seed_urls = source.get("seed_urls", [])

    html_dir = raw_dir / "html"
    pdf_dir = raw_dir / "pdf"
    html_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict[str, str]] = []
    source_attempted = 0
    source_ok = 0
    source_errors = 0
    source_skipped = 0
    source_links_discovered = 0
    source_start = time.monotonic()
    pacer = DomainPacer(args.domain_min_interval)

    log(f"[source {source_idx}/{source_count}] {source_id} (seeds={len(seed_urls)})")
    for seed_idx, seed_url in enumerate(seed_urls, start=1):
        if args.max_runtime_seconds > 0 and (time.monotonic() - started_at) >= args.max_runtime_seconds:
            log("[runtime-limit] reached, stopping current source.")
            break
        queue: deque[tuple[str, int]] = deque([(canonicalize_url(seed_url), 0)])
        discovered: set[str] = {canonicalize_url(seed_url)}
        fetched_this_seed = 0
        seed_attempted = 0
        seed_ok = 0
        seed_errors = 0
        seed_skipped = 0
        log(
            f"  [seed {seed_idx}/{len(seed_urls)}] {seed_url} "
            f"(max_pages={max_pages_per_seed}, depth={max_depth})"
        )

        while queue and fetched_this_seed < max_pages_per_seed:
            if args.max_runtime_seconds > 0 and (time.monotonic() - started_at) >= args.max_runtime_seconds:
                log("  [runtime-limit] stopping seed due to global timeout.")
                break
            if args.max_requests > 0 and global_attempted_ref[0] >= args.max_requests:
                log("  [request-limit] stopping seed due to --max-requests.")
                break

            batch: list[tuple[str, int]] = []
            while queue and len(batch) < args.workers:
                url, depth = queue.popleft()
                if url in existing_urls:
                    seed_skipped += 1
                    source_skipped += 1
                    continue
                if not is_allowed_url(
                    url,
                    allowed_domains=allowed_domains,
                    blocked_keywords=blocked_keywords,
                    include_keywords=include_keywords,
                    exclude_keywords=exclude_keywords,
                ):
                    seed_skipped += 1
                    source_skipped += 1
                    continue
                batch.append((url, depth))

            if not batch:
                continue

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        fetch_url,
                        url=url,
                        source_id=source_id,
                        category=category,
                        priority=priority,
                        headers=headers,
                        timeout_seconds=timeout_seconds,
                        max_retries=args.max_retries,
                        retry_base_delay=args.retry_base_delay,
                        retry_429=args.retry_429,
                        pacer=pacer,
                        html_dir=html_dir,
                        pdf_dir=pdf_dir,
                        sleep_seconds=args.sleep_seconds,
                    ): (url, depth)
                    for url, depth in batch
                }

                for future in as_completed(futures):
                    url, depth = futures[future]
                    result = future.result()
                    source_rows.append(result.row)
                    existing_urls.add(canonicalize_url(url))

                    seed_attempted += 1
                    source_attempted += 1
                    global_attempted_ref[0] += 1
                    fetched_this_seed += 1
                    if result.is_success:
                        seed_ok += 1
                        source_ok += 1
                    else:
                        seed_errors += 1
                        source_errors += 1

                    if allow_subpages and depth < max_depth and result.links:
                        source_links_discovered += len(result.links)
                        for link in result.links:
                            norm = canonicalize_url(link)
                            if norm not in discovered and norm not in existing_urls:
                                discovered.add(norm)
                                if queue_priority(norm) <= 1:
                                    queue.appendleft((norm, depth + 1))
                                else:
                                    queue.append((norm, depth + 1))

                    if args.progress_every > 0 and (source_attempted % args.progress_every == 0):
                        elapsed = time.monotonic() - source_start
                        rate = source_attempted / elapsed if elapsed > 0 else 0.0
                        log(
                            "    [progress] "
                            f"ok={source_ok} err={source_errors} skipped={source_skipped} "
                            f"attempted={source_attempted} queue={len(queue)} "
                            f"global_attempted={global_attempted_ref[0]} "
                            f"rate={rate:.2f} req/s elapsed={elapsed_hms(elapsed)}"
                        )

        log(
            f"  [seed done] attempted={seed_attempted} ok={seed_ok} err={seed_errors} "
            f"skipped={seed_skipped} fetched={fetched_this_seed}/{max_pages_per_seed} "
            f"discovered={len(discovered)}"
        )

    elapsed = time.monotonic() - source_start
    log(
        f"[source done] {source_id} attempted={source_attempted} ok={source_ok} err={source_errors} "
        f"skipped={source_skipped} discovered_links={source_links_discovered} "
        f"elapsed={elapsed_hms(elapsed)}"
    )
    return source_rows


def summarize(rows: list[dict[str, str]], started_at: float) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        status = row.get("status", "unknown")
        source_id = row.get("source_id", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        if source_id not in by_source:
            by_source[source_id] = {"total": 0, "ok": 0, "errors": 0}
        by_source[source_id]["total"] += 1
        if status == "ok":
            by_source[source_id]["ok"] += 1
        else:
            by_source[source_id]["errors"] += 1

    total_rows = len(rows)
    ok_rows = by_status.get("ok", 0)
    elapsed = time.monotonic() - started_at
    return {
        "total_rows": total_rows,
        "ok_rows": ok_rows,
        "error_rows": total_rows - ok_rows,
        "ok_rate_percent": round((ok_rows / total_rows * 100.0), 2) if total_rows else 0.0,
        "status_breakdown": by_status,
        "elapsed_hms": elapsed_hms(elapsed),
        "per_source": by_source,
    }


def main() -> None:
    args = parse_args()
    cfg = load_sources(args.sources)
    raw_dir = Path(args.raw_dir)
    manifest_path = Path(args.manifest)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    existing_urls: set[str] = set()
    if args.resume_from_manifest:
        all_rows = load_manifest_rows(manifest_path)
        existing_urls = load_existing_urls_from_manifest(manifest_path)
        log(
            f"[resume] loaded {len(existing_urls)} successful URLs "
            f"and {len(all_rows)} manifest rows"
        )

    sources = cfg.get("sources", [])
    crawl_cfg = cfg.get("crawl", {})
    started_at = time.monotonic()
    global_attempted_ref = [0]

    log(
        "[start] high-scale crawl "
        f"(sources={len(sources)}, workers={args.workers}, depth={crawl_cfg.get('max_depth', 2)}, "
        f"max_pages_per_seed={crawl_cfg.get('max_pages_per_seed', 80)}, "
        f"max_runtime={args.max_runtime_seconds or 'unlimited'}s)"
    )

    for idx, source in enumerate(sources, start=1):
        if args.max_runtime_seconds > 0 and (time.monotonic() - started_at) >= args.max_runtime_seconds:
            log("[runtime-limit] reached before next source, stopping.")
            break
        if args.max_requests > 0 and global_attempted_ref[0] >= args.max_requests:
            log("[request-limit] reached before next source, stopping.")
            break

        rows = collect_for_source(
            source=source,
            crawl_cfg=crawl_cfg,
            raw_dir=raw_dir,
            existing_urls=existing_urls,
            args=args,
            started_at=started_at,
            source_idx=idx,
            source_count=len(sources),
            global_attempted_ref=global_attempted_ref,
        )
        all_rows.extend(rows)

        if args.save_manifest_every_source:
            write_manifest(manifest_path, all_rows)
            log(f"[checkpoint] manifest saved ({len(all_rows)} rows)")

    write_manifest(manifest_path, all_rows)
    summary = summarize(all_rows, started_at)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
