"""Hand-corrections applied to platform content at ingestion time.

Automatic extraction gets roughly 99% of a textbook right. The last
fraction resists every rule, because a PDF stores glyphs at coordinates and
never says "this is a fraction" -- so structure is always inferred, and
inference has an error rate.

For a student's own upload that is simply the ceiling. For platform content
it is not: the corpus is finite, loaded once, and read by every student, so
a formula corrected by hand today is correct for everyone from then on. That
makes the economics completely different, and it is the reason this file
exists.

Corrections live in JSON next to the content, one file per document, so they
are reviewable in a diff and travel with the repository rather than hiding
in a database.

A correction that no longer matches is reported, never skipped in silence.
Extraction improves over time, and a correction whose `find` text has been
fixed upstream would otherwise sit there looking applied while doing
nothing -- or worse, mask a regression.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ai.ingestion.extract import Page

log = logging.getLogger(__name__)

CORRECTIONS_DIR = Path("content/corrections")


@dataclass(frozen=True)
class Correction:
    page: int | None
    find: str
    replace: str
    note: str = ""


@dataclass
class CorrectionResult:
    pages: list[Page]
    applied: list[Correction] = field(default_factory=list)
    unmatched: list[Correction] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [f"{len(self.applied)} correction(s) applied"]
        if self.unmatched:
            parts.append(f"{len(self.unmatched)} NO LONGER MATCH")
        return ", ".join(parts)


def candidate_keys(document_key: str) -> list[str]:
    """Keys to try, stripping stacked extensions.

    Windows hides known extensions by default, so renaming a download to
    "leph101.pdf" in Explorer produces "leph101.pdf.pdf" on disk. The
    document is then stored under the stem "leph101.pdf", which matches no
    correction file -- and the corrections sit there doing nothing, with the
    formulas they fix still wrong in the index.
    """
    keys = [document_key]
    while "." in keys[-1]:
        keys.append(keys[-1].rsplit(".", 1)[0])
    return keys


def corrections_path(document_key: str, directory: Path | None = None) -> Path:
    base = directory or CORRECTIONS_DIR
    for key in candidate_keys(document_key):
        candidate = base / f"{key}.json"
        if candidate.exists():
            return candidate
    return base / f"{document_key}.json"


def load_corrections(document_key: str, directory: Path | None = None) -> list[Correction]:
    """Read the correction file for a document, if one exists."""
    path = corrections_path(document_key, directory)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.exception("Correction file %s is not valid JSON; ignoring it", path)
        return []

    out: list[Correction] = []
    for entry in data.get("corrections", []):
        find = entry.get("find", "")
        if not find:
            continue
        out.append(
            Correction(
                page=entry.get("page"),
                find=find,
                replace=entry.get("replace", ""),
                note=entry.get("note", ""),
            )
        )
    return out


def apply_corrections(
    pages: list[Page], corrections: list[Correction]
) -> CorrectionResult:
    """Replace corrected text, page by page.

    Matching is exact substring replacement rather than a pattern, because
    the person writing a correction is copying a string out of a review file
    and should not have to think about regex escaping. A correction may name
    a page, which limits it and stops a short string matching elsewhere.
    """
    if not corrections:
        return CorrectionResult(pages=pages)

    texts = {page.number: page.text for page in pages}
    applied: list[Correction] = []
    unmatched: list[Correction] = []

    for correction in corrections:
        targets = (
            [correction.page]
            if correction.page is not None
            else list(texts)
        )
        hit = False
        for number in targets:
            text = texts.get(number)
            if text and correction.find in text:
                texts[number] = text.replace(correction.find, correction.replace)
                hit = True
        (applied if hit else unmatched).append(correction)

    if unmatched:
        log.warning(
            "%d correction(s) no longer match the extracted text; "
            "the extractor may have changed. Re-run the review export.",
            len(unmatched),
        )
        for correction in unmatched:
            log.warning("  page %s: %r", correction.page, correction.find[:70])

    corrected = [
        Page(number=page.number, text=texts[page.number])
        for page in pages
        if texts[page.number].strip()
    ]
    return CorrectionResult(pages=corrected, applied=applied, unmatched=unmatched)