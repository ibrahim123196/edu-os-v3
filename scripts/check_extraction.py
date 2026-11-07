"""Check how well a PDF extracted, before trusting it.

Extraction quality varies by chapter. A chapter of prose and circuit diagrams
behaves nothing like one of vector calculus, and the only way to know is to
measure. This reports four things that have each caused a real bug:

* text loss      -- did the layout pass drop content the plain reader found?
* reading order  -- did columns or blocks get interleaved?
* formulas       -- how many stacked fractions were rebuilt, and do they look sane?
* furniture      -- did any running header or footer survive into the text?

Run it on a folder before or after ingesting:

    python -m scripts.check_extraction ./ncert
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

FRACTION = re.compile(r"\([^()]{1,70}\)/\([^()]{1,70}\)")
WORDS = re.compile(r"[A-Za-z]{4,}")
FURNITURE = re.compile(r"reprint\s*\d{4}", re.I)


def report(path: Path) -> dict:
    import pymupdf

    from ai.ingestion.boilerplate import strip_boilerplate
    from ai.ingestion.extract import Page, clean_text
    from ai.ingestion.math_layout import page_text_with_math

    document = pymupdf.open(str(path))
    rebuilt, native, order_scores = [], [], []

    for index, page in enumerate(document, start=1):
        mine = page_text_with_math(page)
        plain = page.get_text("text", sort=True)
        rebuilt.append(Page(index, clean_text(mine)))
        native.append(plain)

        mine_words, plain_words = WORDS.findall(mine), WORDS.findall(plain)
        if plain_words:
            order_scores.append(
                (difflib.SequenceMatcher(None, plain_words, mine_words).ratio(), index)
            )

    pages = [p for p in rebuilt if p.text]
    pages, furniture = strip_boilerplate(pages)
    text = "\n".join(p.text for p in pages)

    mine_total = sum(len(WORDS.findall(p.text)) for p in pages)
    plain_total = sum(len(WORDS.findall(t)) for t in native)

    fractions = FRACTION.findall(text)
    suspicious = [
        f for f in fractions
        if len(f) > 55 or "∑" in f or "∫" in f or f.count("_") > 4
    ]
    order_scores.sort()

    return {
        "file": path.name,
        "pages": document.page_count,
        "words_kept": f"{100 * mine_total // max(plain_total, 1)}%",
        "fractions": len(fractions),
        "suspicious_fractions": len(suspicious),
        "furniture_removed": len(furniture),
        "furniture_surviving": len(FURNITURE.findall(text)),
        "worst_order_pages": order_scores[:3],
        "samples": fractions[:6],
        "suspicious_samples": suspicious[:4],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PDF extraction quality.")
    parser.add_argument("target", type=Path, help="a PDF or a folder of PDFs")
    args = parser.parse_args()

    paths = (
        sorted(p for p in args.target.iterdir() if p.suffix.lower() == ".pdf")
        if args.target.is_dir()
        else [args.target]
    )
    if not paths:
        print(f"No PDFs at {args.target}")
        return 1

    for path in paths:
        r = report(path)
        print(f"\n=== {r['file']} ({r['pages']} pages) ===")
        print(f"  words kept vs plain extraction : {r['words_kept']}")
        print(f"  fractions rebuilt              : {r['fractions']}"
              f"  ({r['suspicious_fractions']} look wrong)")
        print(f"  furniture lines removed        : {r['furniture_removed']}")
        print(f"  footer text still in the body  : {r['furniture_surviving']}")
        print(f"  lowest reading-order agreement : "
              + ", ".join(f"p{p} {s:.2f}" for s, p in r["worst_order_pages"]))
        if r["samples"]:
            print("  sample formulas:")
            for f in r["samples"]:
                print(f"    {f}")
        if r["suspicious_samples"]:
            print("  questionable:")
            for f in r["suspicious_samples"]:
                print(f"    {f}")

    print("\nWhat to look at:")
    print("  words kept below ~95%      -> the layout pass is dropping text")
    print("  reading-order below ~0.90  -> check that page's prose for interleaving")
    print("  sample formulas            -> read them; only you know what they should say")
    return 0


if __name__ == "__main__":
    sys.exit(main())
