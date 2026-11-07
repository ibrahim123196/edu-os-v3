"""Export every formula in a document so a human can check it against the PDF.

Automatic extraction gets most of a textbook right and quietly gets a little
of it wrong, and a wrong formula looks exactly like a correct one. For
platform content that is worth fixing by hand: the corpus is finite and
every student reads the same copy.

This writes a review file listing each formula with its page number, and a
matching correction stub. Open the PDF beside the review file, work down the
list, and for anything wrong copy the extracted line into the stub and type
what it should say.

    python -m scripts.review_formulas ./ncert/leph101.pdf

Writes:
    content/review/leph101.txt          what was extracted, page by page
    content/corrections/leph101.json    the stub to edit (never overwritten)

Re-running refreshes the review file and leaves your corrections alone.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FRACTION = re.compile(r"\([^()]{1,70}\)/\([^()]{1,70}\)")
SCRIPT = re.compile(r"[\^_][A-Za-z0-9]")
EQUALS = re.compile(r"[A-Za-z0-9\)\]]\s*=\s*[-+]?[A-Za-z0-9\(\[]")

REVIEW_DIR = Path("content/review")
CORRECTIONS_DIR = Path("content/corrections")


def interesting(line: str) -> bool:
    """Lines worth a human's attention: anything carrying notation."""
    return bool(FRACTION.search(line) or SCRIPT.search(line) or EQUALS.search(line))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export formulas for review.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--key", help="document key (defaults to the file stem)")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"No such file: {args.pdf}")
        return 1

    import pymupdf

    from ai.ingestion.boilerplate import strip_boilerplate
    from ai.ingestion.extract import Page, clean_text
    from ai.ingestion.math_layout import page_text_with_math

    key = args.key or args.pdf.stem
    document = pymupdf.open(str(args.pdf))

    pages = [
        Page(index, clean_text(page_text_with_math(page)))
        for index, page in enumerate(document, start=1)
    ]
    pages = [p for p in pages if p.text]
    pages, _ = strip_boilerplate(pages)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# Formula review — {args.pdf.name}",
        "#",
        "# Check each line against the PDF page named beside it.",
        "# For anything wrong, add an entry to "
        f"content/corrections/{key}.json:",
        '#   {"page": 7, "find": "<paste the wrong text>", '
        '"replace": "<what it should say>", "note": "why"}',
        "#",
        "",
    ]

    total = 0
    for page in pages:
        hits = [line.strip() for line in page.text.splitlines() if interesting(line)]
        if not hits:
            continue
        lines.append(f"--- page {page.number} " + "-" * 50)
        for hit in hits:
            lines.append(f"  {hit}")
            total += 1
        lines.append("")

    review_path = REVIEW_DIR / f"{key}.txt"
    review_path.write_text("\n".join(lines), encoding="utf-8")

    stub_path = CORRECTIONS_DIR / f"{key}.json"
    if not stub_path.exists():
        stub_path.write_text(
            json.dumps(
                {
                    "document": key,
                    "source": args.pdf.name,
                    "corrections": [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        created = " (created)"
    else:
        existing = json.loads(stub_path.read_text(encoding="utf-8"))
        created = f" ({len(existing.get('corrections', []))} correction(s) already in it)"

    print(f"{total} notation line(s) across {len(pages)} page(s)")
    print(f"  review:      {review_path}")
    print(f"  corrections: {stub_path}{created}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
