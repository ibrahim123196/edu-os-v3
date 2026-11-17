"""Chemistry drawn as pictures, written out as text.

In NCERT Chemistry the reactions and structures are images, not text: on
page 9 of Haloalkanes and Haloarenes, "Write the products of the following
reactions:" is followed by four reactions that extraction cannot see. A
student asking about them got nothing, because nothing was indexed.

A vision model reads each such page once (scripts.transcribe_images) and
writes what is drawn as text, saved to content/transcriptions/<file>.json,
where a person can read and correct it. Ingestion then puts each piece into
the page's text right after the line it follows in the book. Like the
corrections, it is paid for once per chapter, not per student, and
re-ingesting costs nothing.

Each piece is marked as transcribed from a drawing, so a reader -- or the
tutor quoting it -- knows it is a reading of the picture, not printed text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ai.ingestion.corrections import candidate_keys
from ai.ingestion.extract import Page

log = logging.getLogger(__name__)

TRANSCRIPTIONS_DIR = Path(__file__).resolve().parents[2] / "content" / "transcriptions"
MARK = "[Drawn in the book]"
# "after" for a drawing that comes before every line of text on its page.
PAGE_START = "(start of page)"


@dataclass
class Transcription:
    page: int
    after: str | None       # the line of page text it follows; PAGE_START; None: end of page
    text: str


@dataclass
class TranscriptionResult:
    pages: list[Page]
    placed: int = 0
    at_page_end: list[Transcription] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [f"{self.placed} drawing(s) written in"]
        if self.at_page_end:
            parts.append(f"{len(self.at_page_end)} placed at the end of their page "
                         "(the line they follow was not found)")
        return ", ".join(parts)


def transcriptions_path(document_key: str, directory: Path | None = None) -> Path:
    base = directory or TRANSCRIPTIONS_DIR
    for key in candidate_keys(document_key):
        candidate = base / f"{key}.json"
        if candidate.exists():
            return candidate
    return base / f"{document_key}.json"


def load_transcriptions(document_key: str, directory: Path | None = None) -> list[Transcription]:
    path = transcriptions_path(document_key, directory)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.exception("Transcription file %s is not valid JSON; ignoring it", path)
        return []
    out = []
    for entry in data.get("items", []):
        text = (entry.get("text") or "").strip()
        if text and isinstance(entry.get("page"), int):
            out.append(Transcription(page=entry["page"], after=entry.get("after") or None, text=text))
    return out


def apply_transcriptions(pages: list[Page], items: list[Transcription]) -> TranscriptionResult:
    """Each transcription into its page's text, after the line it follows.

    Matched on a whole line, after trimming, so a short "after" cannot land
    inside a sentence. Several after one line keep their order. One whose
    line is gone (the extraction changed) goes to the end of its page, which
    still keeps it with the right page and section, and is reported.
    """
    if not items:
        return TranscriptionResult(pages=pages)
    by_page: dict[int, list[Transcription]] = {}
    for item in items:
        by_page.setdefault(item.page, []).append(item)

    result = TranscriptionResult(pages=[])
    for page in pages:
        mine = by_page.get(page.number)
        if not mine:
            result.pages.append(page)
            continue
        lines = page.text.split("\n")
        inserts: dict[int, list[str]] = {}
        tail: list[str] = []
        head: list[str] = []
        for item in mine:
            block = f"{MARK} {item.text}"
            spot = None
            if item.after == PAGE_START:
                head.append(block)
                result.placed += 1
                continue
            if item.after:
                wanted = item.after.strip()
                spot = next((i for i, ln in enumerate(lines) if ln.strip() == wanted), None)
            if spot is None:
                tail.append(block)
                result.at_page_end.append(item)
            else:
                inserts.setdefault(spot, []).append(block)
            result.placed += 1
        out: list[str] = list(head)
        for i, line in enumerate(lines):
            out.append(line)
            out.extend(inserts.get(i, []))
        out.extend(tail)
        result.pages.append(Page(number=page.number, text="\n".join(out)))
    return result
