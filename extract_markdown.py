#!/usr/bin/env python3
"""
Extract page-level markdown from an OCR JSON (ocr_output.json) into a single markdown file.

Usage:
  python extract_markdown.py --input /path/to/ocr_output.json --output /path/to/output.md

If not provided, defaults to:
  --input  ./ocr_output.json
  --output ./ocr_output.md
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, List, Optional


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_markdown(pages: List[dict]) -> str:
    # Sort by 'index' if present; otherwise keep order
    try:
        pages_sorted = sorted(pages, key=lambda p: p.get("index", 0))
    except Exception:
        pages_sorted = pages

    chunks: List[str] = []
    for page in pages_sorted:
        md: Optional[str] = page.get("markdown") if isinstance(page, dict) else None
        if not md:
            continue
        # Ensure trailing newline separation per page
        chunks.append(md.rstrip() + "\n\n")

    return "".join(chunks)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract markdown from OCR JSON")
    parser.add_argument("--input", "-i", type=Path, default=Path("ocr_output.json"), help="Path to ocr_output.json")
    parser.add_argument("--output", "-o", type=Path, default=Path("ocr_output.md"), help="Path to write concatenated markdown")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input JSON not found: {args.input}", file=sys.stderr)
        return 1

    data = load_json(args.input)

    # The JSON may be a dict with 'pages' or a list of pages
    if isinstance(data, dict) and "pages" in data and isinstance(data["pages"], list):
        pages = data["pages"]
    elif isinstance(data, list):
        pages = data
    else:
        print("Unrecognized JSON structure: expected a dict with 'pages' or a list", file=sys.stderr)
        return 2

    markdown = extract_markdown(pages)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Wrote {len(markdown)} characters of markdown to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
