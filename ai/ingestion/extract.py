"""Turn an uploaded file into page-tagged text.

Everything downstream keys off the page number, because that is what a
citation has to point at. A block that loses its page number cannot be cited,
so extraction preserves page boundaries even when the text is messy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


@dataclass(slots=True)
class Page:
    number: int  # 1-indexed, matches what the student sees in a PDF reader
    text: str


class UnsupportedFileType(Exception):
    pass


class EmptyDocument(Exception):
    pass


_WS = re.compile(r"[ \t\u00a0]+")
_NEWLINES = re.compile(r"\n{3,}")
# Words split across a line break by hyphenation: "photo-\nsynthesis"
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def clean_text(raw: str) -> str:
    """Normalise whitespace without destroying paragraph structure."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NEWLINES.sub("\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[Page] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            # A single unreadable page should not fail the whole upload.
            raw = ""
        cleaned = clean_text(raw)
        if cleaned:
            pages.append(Page(number=index, text=cleaned))
    return pages


def extract_plaintext(path: Path) -> list[Page]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    return [Page(number=1, text=cleaned)] if cleaned else []


def extract(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileType(
            f"{suffix or 'file'} is not supported. Upload a PDF, .txt or .md file."
        )

    pages = extract_pdf(path) if suffix == ".pdf" else extract_plaintext(path)

    if not pages:
        raise EmptyDocument(
            "No readable text found. If this is a scanned PDF it needs OCR first."
        )
    return pages
