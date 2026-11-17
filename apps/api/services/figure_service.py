"""The textbook figure a lesson points to, cut from its PDF and kept on disk.

A figure is shown only when three things agree: the lesson's "picture it"
step names it, one of the passages the lesson was written from mentions it
(so it is the figure of this topic, not a number the model made up), and
the figure can be cut cleanly out of that passage's PDF. Anything less and
the step keeps its words alone.

Cut once and kept: the same figure is asked for by every student taught the
topic. A figure that could not be cut is remembered too, so a page that
defeats the cutter is not searched again on every lesson.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from ai.ingestion import figures
from apps.api.config import settings
from apps.api.models import Document, DocumentChunk

log = logging.getLogger(__name__)

LABEL = re.compile(r"^\d{1,2}\.\d{1,3}$")


def cache_dir() -> Path:
    return Path(settings.upload_dir).parent / "figures"


def _paths(document_id: uuid.UUID, label: str) -> tuple[Path, Path]:
    # Not with_suffix: "2.24" already looks like a name with a suffix.
    stem = f"{document_id}_{label}_v{figures.VERSION}"
    return cache_dir() / f"{stem}.png", cache_dir() / f"{stem}.none"


def picture(db: Session, document_id: uuid.UUID, label: str, page_hint: int | None = None) -> Path | None:
    """The figure's PNG on disk, cut now if it has not been before; None if
    it cannot be cut, or the document is not a PDF on this machine."""
    if not LABEL.match(label):
        return None
    png, none = _paths(document_id, label)
    if png.exists():
        return png
    if none.exists():
        return None
    doc = db.get(Document, document_id)
    if doc is None:
        return None
    source = Path(doc.storage_path)
    if source.suffix.lower() != ".pdf" or not source.exists():
        return None
    try:
        data = figures.render(source, label, page_hint)
    except Exception:
        # A page the cutter trips on: no figure, and the lesson goes on.
        log.warning("Cutting figure %s from %s failed", label, doc.filename, exc_info=True)
        data = None
    cache_dir().mkdir(parents=True, exist_ok=True)
    if data is None:
        none.touch()
        log.info("Figure %s could not be cut from %s", label, doc.filename)
        return None
    png.write_bytes(data)
    return png


def for_lesson(db: Session, visual_text: str, chunk_ids: list[uuid.UUID],
               cited_ids: list[uuid.UUID] | None = None) -> dict | None:
    """The figure for the lesson's visual step, as the page needs it, or
    None.

    A figure the step names: the first one a source passage also mentions
    and that can be cut. If it names one that cannot be cut, nothing -- a
    different figure would not match the words under it.

    A step that names none, and describes a drawing instead: the first
    figure mentioned in the passages that step itself cites. Those are the
    passages its words came from, so their figure shows what it describes.
    """
    if not chunk_ids:
        return None
    named = figures.labels_in(visual_text)
    pool = chunk_ids if named else [c for c in (cited_ids or []) if c in chunk_ids]
    chunks = db.query(DocumentChunk).filter(DocumentChunk.id.in_(pool)).all() if pool else []
    order = {cid: i for i, cid in enumerate(pool)}
    chunks.sort(key=lambda c: order.get(c.id, len(order)))
    if named:
        wanted = [(label, c) for label in named for c in chunks
                  if label in figures.labels_in(c.content)]
    else:
        wanted = [(label, c) for c in chunks for label in figures.labels_in(c.content)]
    tried = set()
    for label, chunk in wanted:
        if (chunk.document_id, label) in tried:
            continue
        tried.add((chunk.document_id, label))
        try:
            found = picture(db, chunk.document_id, label, chunk.page_start)
        except OSError:
            # The cache folder unwritable, say: a figure is an extra, and
            # must never cost the student the lesson.
            log.warning("Keeping figure %s failed", label, exc_info=True)
            found = None
        if found:
            return {
                "label": label,
                "page": chunk.page_start,
                "url": f"/figures/{chunk.document_id}/{label}.png",
            }
    return None
