"""Checks on an uploaded file before anything reads it.

Anyone can make an account now (24 Nov), so an upload may be broken, not
what its name says, or made to hurt the server. Each check here is cheap and
reads only the file's structure; the slow work (text, figures) happens only
after a file passes.

What is refused, and why:
- a ".pdf" that is not a PDF, or one PyMuPDF cannot open: it would fail
  later anyway, after a worker had been tied up on it;
- password-protected PDFs: nothing in them can be read;
- more than MAX_PAGES pages: a textbook chapter is 20-40; a book is not notes;
- PDFs carrying scripts, launch actions or attached files: EduOS never runs
  them, but study notes have no reason to contain them, and they are how a
  harmful PDF usually works;
- images over MAX_IMAGE_PIXELS: one giant picture is a classic way to run a
  server out of memory, and cutting a figure decodes the images on its page;
- a text file with binary bytes in it: it is not text.

Pages with enormous drawings are not refused: ingestion already skips their
drawings (heavy.py), and the rest of the page is still useful.
"""

from __future__ import annotations

from pathlib import Path

MAX_PAGES = 300
# 100 megapixels: a full-page photo at 600 dpi is about 35.
MAX_IMAGE_PIXELS = 100_000_000
# How much of a text file to look at for binary bytes.
_TEXT_SNIFF = 64 * 1024

# Names inside a PDF that mean "run something" or "carry a file".
_ACTIVE = ("/JavaScript", "/JS", "/Launch", "/EmbeddedFile", "/RichMedia")


class UploadRefused(ValueError):
    """Why the file was not accepted, in words for the student."""


def check(path: Path) -> None:
    """Raise UploadRefused if the file should not be read."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        _check_pdf(path)
    else:
        _check_text(path)


def _check_text(path: Path) -> None:
    with path.open("rb") as f:
        head = f.read(_TEXT_SNIFF)
    # Notepad's "Unicode" (UTF-16) is text with a nul in every other byte;
    # it is marked by two bytes at the start, so look past those.
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        head = head[2:len(head) - len(head) % 2].decode(
            "utf-16-le" if head[0] == 0xFF else "utf-16-be", errors="replace").encode()
    if b"\x00" in head:
        raise UploadRefused("That file is not plain text. Upload a PDF, .txt or .md file.")


def _check_pdf(path: Path) -> None:
    with path.open("rb") as f:
        head = f.read(1024)
    if b"%PDF-" not in head:
        raise UploadRefused("That file is not really a PDF, even though its name ends in .pdf.")

    import pymupdf

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise UploadRefused("That PDF is damaged and cannot be opened.") from exc
    with doc:
        if doc.needs_pass or doc.is_encrypted:
            raise UploadRefused("That PDF is password-protected. Upload a copy without a password.")
        if doc.page_count == 0:
            raise UploadRefused("That PDF has no pages.")
        if doc.page_count > MAX_PAGES:
            raise UploadRefused(
                f"That PDF has {doc.page_count} pages; the limit is {MAX_PAGES}. "
                "Upload the chapters you need one at a time.")
        if doc.embfile_count() > 0 or _has_active_content(doc):
            raise UploadRefused(
                "That PDF contains scripts or attached files, which study material does not "
                "need. Save it again as a plain PDF (Print → Save as PDF) and upload that.")
        for page in doc:
            for image in page.get_images(full=False):
                width, height = image[2], image[3]
                if width * height > MAX_IMAGE_PIXELS:
                    raise UploadRefused(
                        f"Page {page.number + 1} has an image too large to handle "
                        f"({width} × {height}). Shrink the PDF and try again.")


def _has_active_content(doc) -> bool:
    """Whether any object in the PDF names a script, launch action or
    attached file. Reads the objects' dictionaries, not their streams."""
    for xref in range(1, doc.xref_length()):
        try:
            source = doc.xref_object(xref, compressed=True)
        except Exception:
            continue
        for name in _ACTIVE:
            at = source.find(name)
            # A whole name: "/JS" must not match "/JSomething".
            while at != -1:
                end = at + len(name)
                if end == len(source) or not (source[end].isalnum() or source[end] == "_"):
                    return True
                at = source.find(name, end)
    return False
