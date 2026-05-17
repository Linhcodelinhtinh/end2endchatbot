"""Clean raw HTML/PDF files into high-quality docs.jsonl.

Usage:
  python scripts/clean_kb.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
DATE_RE = re.compile(
    r"(20\d{2}[-/]\d{1,2}[-/]\d{1,2})|(\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2})",
    flags=re.UNICODE,
)

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(trang chủ|home)\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*(xem thêm|chi tiết)\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*(đăng nhập|login)\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*(liên hệ|contact)\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*(menu|navigation)\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*(copyright|all rights reserved)", flags=re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean KB raw files into docs.jsonl.")
    parser.add_argument("--manifest", default="data/knowledge_base/source_manifest.csv")
    parser.add_argument("--output", default="data/knowledge_base/processed/docs.jsonl")
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--min-words", type=int, default=25)
    parser.add_argument("--near-dup-threshold", type=int, default=3, help="Simhash Hamming distance.")
    parser.add_argument("--max-boilerplate-line-ratio", type=float, default=0.55)
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def normalize_pdf_lines(text: str) -> str:
    # Join line-break hyphenation artifacts from PDF extraction.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def filter_boilerplate_lines(text: str) -> tuple[str, float]:
    lines = [line.strip() for line in text.splitlines()]
    if not lines:
        return "", 0.0
    kept: list[str] = []
    removed = 0
    for line in lines:
        if not line:
            continue
        if any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS):
            removed += 1
            continue
        kept.append(line)
    ratio = removed / max(len(lines), 1)
    return normalize_whitespace("\n".join(kept)), ratio


def detect_language(text: str) -> str:
    vi_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    lowered = text.lower()
    vi_score = sum(lowered.count(ch) for ch in vi_chars)
    ascii_letters = sum("a" <= ch <= "z" for ch in lowered)
    if vi_score >= 3:
        return "vi"
    if ascii_letters > 0:
        return "en"
    return "unknown"


def extract_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    return next(group for group in match.groups() if group)


def extract_html_text(path: Path) -> tuple[str, str]:
    raw_html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw_html, "html.parser")

    title = ""
    if soup.title and soup.title.text:
        title = normalize_whitespace(soup.title.text)

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "footer", "header", "aside"]):
        tag.decompose()

    best_node = soup.find("article") or soup.find("main")
    if not best_node:
        candidates = soup.find_all(["div", "section"])
        best_node = max(candidates, key=lambda node: len(node.get_text(" ", strip=True)), default=soup)

    text = best_node.get_text("\n", strip=True)
    cleaned, _ = filter_boilerplate_lines(text)
    return title, cleaned


def extract_pdf_text(path: Path) -> str:
    try:
        text = extract_pdf_text_pypdf(path)
    except Exception:
        text = ""
    if len(text.split()) < 80:
        # Fallback for PDFs that pypdf handles poorly.
        try:
            text = extract_pdf_text_pdfplumber(path) or text
        except Exception:
            pass

    text = normalize_pdf_lines(text)
    text = remove_repeated_pdf_lines(text)
    cleaned, _ = filter_boilerplate_lines(text)
    return cleaned


def extract_pdf_text_pypdf(path: Path) -> str:
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError("Install pypdf to parse PDF files in clean_kb.py") from exc

    reader = pypdf.PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if not page_text:
            try:
                page_text = page.extract_text(extraction_mode="layout") or ""
            except TypeError:
                page_text = page.extract_text() or ""
        parts.append(page_text)
    return "\n".join(parts)


def extract_pdf_text_pdfplumber(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""

    parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def remove_repeated_pdf_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text

    freq: dict[str, int] = defaultdict(int)
    for line in lines:
        if len(line) <= 100:
            freq[line] += 1

    repeated = {
        line
        for line, count in freq.items()
        if count >= 3 and not re.fullmatch(r"\d+", line)
    }
    kept = [line for line in lines if line not in repeated]
    return "\n".join(kept)


def row_iter(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def simhash(text: str, bits: int = 64) -> int:
    features = Counter(token.lower() for token in TOKEN_RE.findall(text))
    if not features:
        return 0
    weights = [0] * bits
    for token, count in features.items():
        token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for bit in range(bits):
            if (token_hash >> bit) & 1:
                weights[bit] += count
            else:
                weights[bit] -= count
    result = 0
    for bit, weight in enumerate(weights):
        if weight > 0:
            result |= 1 << bit
    return result


def hamming_distance(a: int, b: int) -> int:
    x = a ^ b
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count("1")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    seen_simhashes: list[int] = []
    total = 0
    kept = 0
    duplicate_exact = 0
    duplicate_near = 0
    too_short = 0
    missing_file = 0
    noisy_filtered = 0

    with output_path.open("w", encoding="utf-8") as out:
        for row in row_iter(manifest_path):
            total += 1
            if row.get("status") != "ok":
                continue

            file_path = Path(row["file_path"])
            if not file_path.exists():
                missing_file += 1
                continue

            suffix = file_path.suffix.lower()
            title = ""
            text = ""
            if suffix in {".html", ".htm"}:
                title, text = extract_html_text(file_path)
            elif suffix == ".pdf":
                text = extract_pdf_text(file_path)
            else:
                text, _ = filter_boilerplate_lines(
                    file_path.read_text(encoding="utf-8", errors="ignore")
                )

            text = normalize_whitespace(text)
            if not text or len(text) < args.min_chars or len(text.split()) < args.min_words:
                too_short += 1
                continue

            text, removed_ratio = filter_boilerplate_lines(text)
            if removed_ratio > args.max_boilerplate_line_ratio:
                noisy_filtered += 1
                continue

            exact_digest = text_hash(text)
            if exact_digest in seen_hashes:
                duplicate_exact += 1
                continue

            text_simhash = simhash(text)
            if any(
                hamming_distance(text_simhash, prev) <= args.near_dup_threshold
                for prev in seen_simhashes
            ):
                duplicate_near += 1
                continue

            seen_hashes.add(exact_digest)
            seen_simhashes.append(text_simhash)

            doc = {
                "id": row["doc_id"],
                "source_id": row["source_id"],
                "source_url": row["url"],
                "source_file": row["file_path"],
                "title": title,
                "category": row.get("category") or "unknown",
                "language": detect_language(text),
                "published_at": extract_date(text[:2000]),
                "collected_at": row.get("crawl_time"),
                "text": text,
                "metadata": {
                    "domain": row.get("domain"),
                    "priority": row.get("priority"),
                    "content_type": row.get("content_type"),
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "boilerplate_removed_ratio": round(removed_ratio, 3),
                    "doc_type": "pdf" if suffix == ".pdf" else "html",
                },
            }
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

    print(
        json.dumps(
            {
                "total_manifest_rows": total,
                "docs_kept": kept,
                "filtered_too_short": too_short,
                "filtered_exact_duplicates": duplicate_exact,
                "filtered_near_duplicates": duplicate_near,
                "filtered_noisy_pages": noisy_filtered,
                "missing_raw_files": missing_file,
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
