"""Ingest files straight into the index, bypassing HTTP.

Useful for seeding a corpus, for re-running ingestion after changing chunk
settings, and for evaluation runs where uploading through the browser would be
tedious.

Usage:
    python -m scripts.ingest notes/*.pdf --subject Biology --grade 12
"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path

from ai.ingestion.extract import SUPPORTED_SUFFIXES
from apps.api.config import settings
from apps.api.db import init_db, session_scope
from apps.api.models import Document, IngestStatus
from apps.api.services.ingestion_service import compute_checksum, ingest_document


def add_file(path: Path, subject: str | None, grade: str | None) -> uuid.UUID | None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        print(f"  skipped {path.name}: {suffix} is not supported")
        return None

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    document_id = uuid.uuid4()
    destination = upload_dir / f"{document_id}{suffix}"
    shutil.copy2(path, destination)

    with session_scope() as db:
        db.add(
            Document(
                id=document_id,
                title=path.stem,
                filename=path.name,
                content_type="application/pdf" if suffix == ".pdf" else "text/plain",
                storage_path=str(destination),
                byte_size=destination.stat().st_size,
                checksum=compute_checksum(destination),
                subject=subject,
                grade=grade,
                status=IngestStatus.pending,
            )
        )
    return document_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest files into EduOS.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--subject")
    parser.add_argument("--grade")
    args = parser.parse_args()

    init_db()

    ingested = 0
    for path in args.paths:
        if not path.exists():
            print(f"  skipped {path}: not found")
            continue
        print(f"Reading {path.name}…")
        document_id = add_file(path, args.subject, args.grade)
        if document_id is None:
            continue
        ingest_document(document_id)

        with session_scope() as db:
            doc = db.get(Document, document_id)
            if doc.status == IngestStatus.ready:
                print(f"  {doc.page_count} pages, {doc.chunk_count} passages indexed")
                ingested += 1
            else:
                print(f"  failed: {doc.error}")

    print(f"\n{ingested} of {len(args.paths)} file(s) indexed.")
    return 0 if ingested else 1


if __name__ == "__main__":
    sys.exit(main())
