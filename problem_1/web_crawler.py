import argparse
import io
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


# Common soft-404 phrases.
SOFT_404_PATTERNS = [
    r"\b404\b",
    r"page\s+not\s+found",
    r"the\s+page\s+you\s+are\s+looking\s+for\s+(cannot|can\'t)?\s*(be\s+found|does\s+not\s+exist)",
    r"sorry,?\s+we\s+can\'t\s+find",
    r"not\s+found",
]


def normalize_url(base_url: str, href: str) -> Optional[str]:
    if not href:
        return None

    href = href.strip()
    if not href or href.startswith("#"):
        return None

    absolute = urljoin(base_url, href)
    absolute, _frag = urldefrag(absolute)

    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None

    # Keep duplicates low.
    cleaned = parsed._replace(netloc=parsed.netloc.lower()).geturl()
    return cleaned.rstrip("/") if cleaned.endswith("/") and len(cleaned) > len(parsed.scheme) + 3 else cleaned


def is_same_domain(candidate_url: str, root_domain: str) -> bool:
    return urlparse(candidate_url).netloc.lower() == root_domain.lower()


# Catch pages that look like 404 even when status is 200.
def is_soft_404(title_text: str, body_text: str) -> bool:
    combined = f"{title_text}\n{body_text[:3000]}".lower()
    return any(re.search(pattern, combined, flags=re.IGNORECASE) for pattern in SOFT_404_PATTERNS)


def extract_visible_text(soup: BeautifulSoup) -> str:
    # Strip obvious junk first.
    for noisy in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        noisy.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # Quick whitespace cleanup.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Read PDFs
def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        return ""

    text_chunks: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            text_chunks.append(page_text)

    combined = "\n".join(text_chunks)
    return re.sub(r"\s+", " ", combined).strip()


def crawl(
    start_url: str,
    max_pages: int,
    delay_seconds: float,
    timeout_seconds: int,
    same_domain_only: bool,
) -> List[Dict[str, str]]:
    parsed_start = urlparse(start_url)
    if parsed_start.scheme not in {"http", "https"}:
        raise ValueError("start_url must begin with http:// or https://")

    root_domain = parsed_start.netloc

    queue: Deque[str] = deque([start_url])
    seen: Set[str] = set()
    accepted_pages: List[Dict[str, str]] = []

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    while queue and len(accepted_pages) < max_pages:
        current_url = queue.popleft()
        if current_url in seen:
            continue
        seen.add(current_url)

        try:
            response = session.get(current_url, timeout=timeout_seconds)
        except requests.RequestException:
            continue

        if response.status_code == 404:
            continue

        # We only care about HTML/PDF.
        content_type = response.headers.get("Content-Type", "")
        lowered_content_type = content_type.lower()
        is_html = "text/html" in lowered_content_type
        is_pdf = "application/pdf" in lowered_content_type or current_url.lower().endswith(".pdf")

        if not is_html and not is_pdf:
            continue

        if is_pdf:
            text = extract_pdf_text(response.content)
            if not text:
                continue

            accepted_pages.append(
                {
                    "url": current_url,
                    "title": Path(urlparse(current_url).path).name or "PDF Document",
                    "text": text,
                }
            )
        else:
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else ""
            text = extract_visible_text(soup)

            # Soft 404.
            if is_soft_404(title, text):
                continue

            accepted_pages.append(
                {
                    "url": current_url,
                    "title": title,
                    "text": text,
                }
            )

            # Follow links from HTML pages.
            for anchor in soup.find_all("a", href=True):
                next_url = normalize_url(current_url, anchor.get("href", ""))
                if not next_url:
                    continue
                if same_domain_only and not is_same_domain(next_url, root_domain):
                    continue
                if next_url not in seen:
                    queue.append(next_url)

        # Sever overload prevention.
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return accepted_pages


def save_jsonl(records: Iterable[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_txt(records: Iterable[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records, start=1):
            f.write(f"=== PAGE {idx} ===\n")
            f.write(f"URL: {record.get('url', '')}\n")
            f.write(f"TITLE: {record.get('title', '')}\n\n")
            f.write(record.get("text", ""))
            f.write("\n\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple link-following crawler that skips hard/soft 404 pages"
    )
    parser.add_argument("start_url", help="Seed URL to start crawling from")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Maximum number of accepted pages to save (default: 30)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between requests (default: 0.5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Request timeout in seconds (default: 15)",
    )
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="If set, crawler can follow links to other domains",
    )
    parser.add_argument(
        "--output",
        default="scrapped.txt",
        help="Output file path (default: scrapped.txt)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pages = crawl(
        start_url=args.start_url,
        max_pages=args.max_pages,
        delay_seconds=args.delay,
        timeout_seconds=args.timeout,
        same_domain_only=not args.allow_external,
    )

    output_path = Path(args.output)
    if output_path.suffix.lower() == ".txt":
        save_txt(pages, output_path)
    else:
        save_jsonl(pages, output_path)

    print(f"Saved {len(pages)} pages to {output_path}")



main()
