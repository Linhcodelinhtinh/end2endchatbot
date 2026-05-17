#!/usr/bin/env python3
"""
Fetch a webpage and write all discovered links to a file.

Usage:
    python scripts/extract_links.py
    python scripts/extract_links.py --url "https://vnu.edu.vn/" --output "data/vnu_links.txt"
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urldefrag, urlsplit
from urllib.request import Request, urlopen


class LinkExtractor(HTMLParser):
    """Extract candidate URL values from common HTML attributes."""

    LINK_ATTRS = {"href", "src", "action", "data-src", "data-url"}

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag
        for attr_name, attr_value in attrs:
            if attr_name not in self.LINK_ATTRS or not attr_value:
                continue

            value = attr_value.strip()
            if not value or value.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue

            absolute = urljoin(self.base_url, value)
            normalized, _fragment = urldefrag(absolute)
            if urlsplit(normalized).scheme not in {"http", "https"}:
                continue
            self.links.add(normalized)


def fetch_html(url: str, timeout: int = 30) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            raise ValueError(f"URL does not return HTML. Content-Type: {content_type}")
        return response.read().decode("utf-8", errors="replace")


def write_links(links: set[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sorted(links)) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract links from a webpage.")
    parser.add_argument("--url", default="https://vnu.edu.vn/", help="Target page URL.")
    parser.add_argument(
        "--output",
        default="data/vnu_links.txt",
        help="Output file path to save all extracted links.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html = fetch_html(args.url)

    extractor = LinkExtractor(base_url=args.url)
    extractor.feed(html)

    output_path = Path(args.output)
    write_links(extractor.links, output_path)

    print(f"Extracted {len(extractor.links)} links to {output_path}")


if __name__ == "__main__":
    main()
