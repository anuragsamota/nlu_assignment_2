import argparse
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse


# markdown link: [text](url)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", flags=re.IGNORECASE)

# raw urls in plain text
RAW_URL_RE = re.compile(r"https?://[^\s)\]}>\"']+", flags=re.IGNORECASE)

# html anchors, just in case
HTML_ANCHOR_RE = re.compile(r"<a\b[^>]*>(.*?)</a>", flags=re.IGNORECASE | re.DOTALL)

# URL bits that are mostly noise
URL_NOISE = {
    "http",
    "https",
    "www",
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "ac",
    "in",
    "html",
    "htm",
    "php",
    "aspx",
    "jsp",
    "pdf",
}

# Common repeated site boilerplate
BOILERPLATE_PHRASES = [
    "important links",
    "all rights reserved",
    "this portal is owned designed and developed",
    "for any comments enquiries feedback",
    "please email the wim",
    "web information manager",
    "feedback cert in help nirf internal committee intranet links",
    "copyright",
    "home people academics undergraduate programs postgraduate programs",
    "programs for working professionals doctoral programs",
    "previous next play arrow",
    "arrow downward",
    "last updated",
    "sitemap",
]

# Common grammatical words to drop in aggressive mode.
AGGRESSIVE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "so",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
}


# Break URL into useful words pieces.
def url_to_words(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return ""

    raw_parts = " ".join([parsed.netloc, parsed.path, parsed.query, parsed.fragment])
    pieces = re.split(r"[^A-Za-z]+", raw_parts)

    words = []
    for piece in pieces:
        word = piece.lower().strip()
        if len(word) <= 1:
            continue
        if word in URL_NOISE:
            continue
        words.append(word)

    return " ".join(words)


# Keep ASCII only
def normalize_ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


# Drop crawler wrappers like PAGE/URL labels.
def strip_structural_lines(raw_text: str) -> str:
    kept_lines = []
    for line in raw_text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue

        lowered = clean_line.lower()
        if lowered.startswith("=== page"):
            continue
        if lowered.startswith("url:"):
            continue
        if lowered.startswith("title:"):
            clean_line = clean_line.split(":", 1)[1].strip()

        kept_lines.append(clean_line)

    return "\n".join(kept_lines)


# Replace links with readable words.
def replace_links(text: str) -> str:
    def markdown_sub(match: re.Match[str]) -> str:
        visible_text = match.group(1) or ""
        link = match.group(2) or ""
        from_url = url_to_words(link)
        return f" {visible_text} {from_url} "

    text = MARKDOWN_LINK_RE.sub(markdown_sub, text)

    # Keep anchor text, ditch tags.
    text = HTML_ANCHOR_RE.sub(lambda m: f" {m.group(1)} ", text)

    # Replace leftover raw URLs too.
    text = RAW_URL_RE.sub(lambda m: f" {url_to_words(m.group(0))} ", text)

    return text


# Aggressive nav/footer cleanup.
def remove_boilerplate(text: str) -> str:
    working = text

    # quick phrase wipe
    for phrase in BOILERPLATE_PHRASES:
        working = re.sub(rf"\b{re.escape(phrase)}\b", " ", working, flags=re.IGNORECASE)

    # remove common legal/address footer line shapes
    working = re.sub(
        r"\b(?:nh|road|jodhpur|rajasthan|india)\b(?:\s+[a-z0-9]+){0,15}\b(?:copyright|rights|reserved)\b",
        " ",
        working,
        flags=re.IGNORECASE,
    )

    # collapse "word word"
    working = re.sub(r"\b([a-zA-Z]{2,})\s+\1\b", r"\1", working, flags=re.IGNORECASE)
    return working


# Main cleaning pipeline.
def clean_text(raw_text: str, aggressive: bool = False) -> str:
    text = strip_structural_lines(raw_text)
    text = replace_links(text)

    if aggressive:
        text = remove_boilerplate(text)

    text = normalize_ascii(text)
    text = text.lower()

    # remove known crawler junk + symbols
    text = re.sub(r"redirecttologinpage", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[#$_%&*+=~`^|\\/<>:\-]+", " ", text)

    # keep only alphabetic english tokens
    tokens = re.findall(r"[a-z]+", text)

    # remove single-char tokens
    tokens = [tok for tok in tokens if len(tok) > 1]

    if aggressive:
        tokens = [tok for tok in tokens if tok not in AGGRESSIVE_STOPWORDS]

    cleaned = " ".join(tokens)

    if aggressive:
        # one more dedup pass
        cleaned = re.sub(r"\b([a-z]{2,})\s+\1\b", r"\1", cleaned)

    return cleaned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean scrapped text into single-line cleaned text")
    parser.add_argument("--input", default="scrapped.txt", help="Input scraped text file (default: scrapped.txt)")
    parser.add_argument("--output", default="cleaned.txt", help="Output cleaned text file (default: cleaned.txt)")
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Apply stronger boilerplate/nav/footer removal",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    raw_text = input_path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw_text, aggressive=args.aggressive)

    output_path.write_text(cleaned + "\n", encoding="utf-8")
    print(f"Cleaned text written to {output_path}")



main()