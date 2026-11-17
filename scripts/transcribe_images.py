"""Write out the chemistry a chapter prints as pictures, once.

    python -m scripts.transcribe_images path\\to\\lech106.pdf --pages 9,10   # try two pages
    python -m scripts.transcribe_images path\\to\\lech106.pdf                # every page

For each page, a vision model is shown the page and its
extracted text, and writes out the reactions, structures and equations that
are drawn rather than printed, each tied to the line of text it follows.
The result goes to content/transcriptions/<file name>.json -- read it, fix
anything misread -- and the next ingestion of that file puts it into the
text. Pages already in the file are kept unless they are asked for again.

The cost is printed at the end. Nothing is ingested by this script.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

from ai.generation.usage import summary
from ai.ingestion.boilerplate import strip_boilerplate
from ai.ingestion.extract import extract
from ai.ingestion.transcriptions import PAGE_START, TRANSCRIPTIONS_DIR

# The page is sent as a top and a bottom half, each sharper than the whole
# page could be: an image is scaled down to about 1,568 pixels on its long
# side, so a whole page arrives at about 130 dpi, and at that size the one
# double bond in a ring was missed -- 1-methylcyclohexene read as toluene.
# Each half at 185 dpi arrives whole, a page width of about 1,530 pixels.
DPI = 185
OVERLAP = 30      # points shared by the halves, so nothing is cut in two

# Chosen on pages 3 and 9 of Haloalkanes and Haloarenes (17 Nov 2026):
# claude-haiku-4-5 misread 3 of 11 structures on page 9 (benzene for
# cyclohexene among them); claude-sonnet-4-6 still missed the one double
# bond of a cyclohexene; claude-sonnet-5-5 read every one, at about 2 cents
# a page.
DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM = (
    "You transcribe chemistry from textbook page images into plain text. You "
    "write only what is drawn on the page, never what you expect to be there."
)

INSTRUCTION = """This is page {page} of an NCERT Chemistry chapter. Its printed text was \
extracted below, line by line, numbered. Anything drawn rather than printed -- reactions, \
structural formulas, rings, mechanisms, equations set as pictures -- is missing from it, or \
appears only as scattered fragments of its atom labels ("CH3", "Br").

Write out each such drawn item, in the order it appears on the page:
- A reaction: reactants, then "--(reagents, conditions)-->", then products. Use condensed \
formulas (CH3CH2Br, C6H5Cl) with numbers as plain digits.
- A structure with no simple condensed formula: describe it briefly and give its name if \
the page gives one, e.g. "benzene ring with Cl and NO2 para to each other (1-chloro-4-nitrobenzene)".
- A mechanism: each step as its own reaction line.
- Keep labels printed on the drawing ("(i)", "Major product", "SN2").
- If part of a drawing cannot be read with confidence, write [unclear] for that part.
- Rings: count the double bonds actually drawn. Three alternating double bonds, or a \
circle inside, is a benzene ring; one double bond is a cyclohexene (or cyclopentene); none \
is a cyclohexane. A ring with one double bond is never benzene.
- One item per reaction or structure, even when several share a line on the page.
- Names and labels already in the extracted text need not be repeated.
Skip photographs, decorative icons, and figures that have a printed "Fig." caption.

For each item give after_line: the number of the extracted line it comes right after on the \
page (0 if it comes before every line).

Reply with JSON only, no other text:
{{"items": [{{"after_line": 12, "text": "CH3CH2OH + HCl --(ZnCl2)--> CH3CH2Cl + H2O"}}]}}
If nothing on the page is drawn chemistry, reply {{"items": []}}.

Extracted text:
{lines}"""


def render_halves(pdf: Path, number: int) -> list[bytes]:
    """The page's top and bottom halves as PNGs, overlapping a little."""
    import pymupdf

    with pymupdf.open(pdf) as doc:
        page = doc[number - 1]
        r = page.rect
        middle = r.y0 + r.height / 2
        halves = [pymupdf.Rect(r.x0, r.y0, r.x1, middle + OVERLAP),
                  pymupdf.Rect(r.x0, middle - OVERLAP, r.x1, r.y1)]
        return [page.get_pixmap(dpi=DPI, clip=h).tobytes("png") for h in halves]


def parse_items(reply: str, lines: list[str], page: int) -> list[dict]:
    """The model's JSON as transcription entries, the line number turned
    into that line's text. Anything malformed is dropped, not guessed at."""
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for entry in data.get("items", []) if isinstance(data, dict) else []:
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        try:
            n = int(entry.get("after_line"))
        except (TypeError, ValueError):
            n = None                          # unreadable: the end of the page
        if n == 0:
            # Before every line: the top of the page. Read as "no line"
            # before, this sent the reactions at the top of page 9 to the
            # bottom, after the questions that follow them.
            after = PAGE_START
        else:
            after = (lines[n - 1].strip()
                     if n is not None and 1 <= n <= len(lines) and lines[n - 1].strip() else None)
        out.append({"page": page, "after": after, "text": text})
    return out


def transcribe_page(llm, pdf: Path, number: int, text: str) -> list[dict]:
    lines = text.split("\n")
    numbered = "\n".join(f"{i}: {ln}" for i, ln in enumerate(lines, 1))
    images = [base64.b64encode(png).decode() for png in render_halves(pdf, number)]
    reply = llm.complete(
        SYSTEM,
        [{"role": "user", "content": [
            {"type": "text", "text": "Top half of the page:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": images[0]}},
            {"type": "text", "text": "Bottom half of the page (the two overlap slightly; "
                                     "do not write an item twice):"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": images[1]}},
            {"type": "text", "text": INSTRUCTION.format(page=number, lines=numbered)},
        ]}],
        max_tokens=2000,
        temperature=0,
    )
    return parse_items(reply, lines, number)


def output_path(pdf: Path) -> Path:
    return TRANSCRIPTIONS_DIR / f"{pdf.name.split('.')[0]}.json"


def main(argv: list[str] | None = None, llm=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", help="comma-separated page numbers; default: every page with pictures")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"vision model to use (default {DEFAULT_MODEL})")
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        print(f"No such file: {args.pdf}")
        return 1
    if llm is None:
        from ai.generation.llm import AnthropicClient
        from apps.api.config import settings
        llm = AnthropicClient(args.model, settings.anthropic_api_key)

    pages, _ = strip_boilerplate(extract(args.pdf))     # the text ingestion will see
    texts = {p.number: p.text for p in pages}
    # Every page by default. Choosing pages by their pictures missed most of
    # them: NCERT draws benzene rings as lines, not images, and every page
    # carries a page-sized background image besides.
    wanted = [int(n) for n in args.pages.split(",")] if args.pages else sorted(texts)
    wanted = [n for n in wanted if n in texts]

    path = output_path(args.pdf)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    items = [i for i in existing.get("items", []) if i.get("page") not in wanted]

    print(f"Reading {len(wanted)} page(s) with {llm.model}: {wanted}")
    for n in wanted:
        found = transcribe_page(llm, args.pdf, n, texts[n])
        items.extend(found)
        print(f"  page {n}: {len(found)} drawn item(s)")
        for item in found:
            print(f"      {item['text']}")

    items.sort(key=lambda i: i["page"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": args.pdf.name, "model": llm.model, "items": items},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(items)} item(s) to {path}")
    print("Read it before ingesting: fix anything misread, delete anything wrong.")
    cost = summary()
    if cost:
        print(cost)
    return 0


if __name__ == "__main__":
    sys.exit(main())
