"""Ingest NCERT chapters into the platform corpus.
 
Platform documents differ from student uploads in two ways: every student can
search them, and they carry provenance -- publisher, licence, edition, source
URL -- so a citation can say what it is quoting and under what terms.
 
NCERT textbooks are published by the Government of India and freely
distributable, which is why they sit at the top of the README's provenance
ladder. Coaching material and private question banks do not, and must not be
loaded this way.
 
Chapters are matched to the curriculum graph by number, so run
`python -m scripts.seed_curriculum` first.
 
    python -m scripts.ingest_ncert ./ncert --subject physics --part I
 
File names must contain the chapter number: "leph101.pdf", "chapter-1.pdf"
and "1.pdf" all work.
"""
 
from __future__ import annotations
 
import argparse
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
 
from sqlalchemy import select
 
from ai.retrieval.spelling import rebuild_vocabulary
from apps.api.config import settings
from apps.api.curriculum import Chapter, License, SourceKind, Subject
from apps.api.db import init_db, session_scope
from apps.api.models import Document, IngestStatus
from apps.api.services.ingestion_service import compute_checksum, ingest_document
 
NCERT_URL = "https://ncert.nic.in/textbook.php"
 
# leph101 -> chapter 1 (NCERT's own naming), chapter-1, ch1, 1
_PATTERNS = [
    re.compile(r"leph1(\d{2})", re.I),          # leph101 = Part I, chapter 01
    re.compile(r"ch(?:apter)?[\s_-]*(\d{1,2})", re.I),
    re.compile(r"^(\d{1,2})\b"),
]
 
 
def chapter_number(path: Path) -> int | None:
    stem = path.stem
    for pattern in _PATTERNS:
        match = pattern.search(stem)
        if match:
            return int(match.group(1))
    return None
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest NCERT chapters as platform content.")
    parser.add_argument("folder", type=Path, help="folder of chapter PDFs")
    parser.add_argument("--subject", default="physics", help="subject code from the curriculum graph")
    parser.add_argument("--part", default="I", help="NCERT part (I or II)")
    parser.add_argument("--edition", default="2026-27", help="textbook edition")
    args = parser.parse_args()
 
    if not args.folder.is_dir():
        print(f"Not a folder: {args.folder}")
        return 1
 
    started = time.perf_counter()
    init_db()
 
    with session_scope() as db:
        subject = db.scalar(select(Subject).where(Subject.code == args.subject))
        if subject is None:
            print(
                f"Subject '{args.subject}' is not in the curriculum graph. "
                "Run: python -m scripts.seed_curriculum"
            )
            return 1
        subject_id = subject.id
 
    pdfs = sorted(p for p in args.folder.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"No PDFs in {args.folder}")
        return 1
 
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
 
    ingested = 0
    for path in pdfs:
        number = chapter_number(path)
        if number is None:
            print(f"  skipped {path.name}: no chapter number in the file name")
            continue
 
        with session_scope() as db:
            chapter = db.scalar(
                select(Chapter).where(
                    Chapter.subject_id == subject_id,
                    Chapter.number == number,
                    Chapter.part == args.part,
                )
            )
            if chapter is None:
                print(f"  skipped {path.name}: chapter {number} is not in the graph")
                continue
            chapter_id, chapter_name = chapter.id, chapter.name
 
        print(f"Chapter {number}: {chapter_name}")
 
        document_id = uuid.uuid4()
        destination = upload_dir / f"{document_id}.pdf"
        shutil.copy2(path, destination)
 
        with session_scope() as db:
            db.add(
                Document(
                    id=document_id,
                    title=f"NCERT Physics Part {args.part} \u2014 Chapter {number}: {chapter_name}",
                    filename=path.name,
                    content_type="application/pdf",
                    storage_path=str(destination),
                    byte_size=destination.stat().st_size,
                    checksum=compute_checksum(destination),
                    subject=args.subject,
                    grade="12",
                    chapter_id=chapter_id,
                    source_kind=SourceKind.platform.value,
                    license=License.public_curriculum.value,
                    publisher="NCERT",
                    edition=args.edition,
                    source_url=NCERT_URL,
                    status=IngestStatus.pending,
                )
            )
 
        # Vocabulary is rebuilt once after the batch, not per chapter.
        ingest_document(document_id, rebuild_vocab=False)
 
        with session_scope() as db:
            doc = db.get(Document, document_id)
            if doc.status == IngestStatus.ready:
                print(f"  {doc.page_count} pages, {doc.chunk_count} passages")
                ingested += 1
            else:
                print(f"  failed: {doc.error}")
 
    if ingested:
        print("\nRebuilding the spelling vocabulary...")
        with session_scope() as db:
            terms = rebuild_vocabulary(db)
        print(f"  {terms} distinct terms")
 
    elapsed = time.perf_counter() - started
    print(f"\n{ingested} of {len(pdfs)} chapter(s) in the corpus "
          f"({elapsed:.0f}s, {elapsed / max(ingested, 1):.0f}s per chapter)")
    return 0 if ingested else 1
 
 
if __name__ == "__main__":
    sys.exit(main())