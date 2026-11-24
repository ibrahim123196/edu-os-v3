"""The checks on an uploaded file, against PDFs made here."""

from __future__ import annotations

import pymupdf
import pytest

from ai.ingestion import scan
from ai.ingestion.scan import UploadRefused, check


def _pdf(path, pages=1, text="Capacitance is charge per unit potential."):
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page().insert_text((72, 72), text)
    doc.save(str(path))
    return path


def test_ordinary_study_material_passes(tmp_path):
    check(_pdf(tmp_path / "notes.pdf", pages=3))
    (tmp_path / "notes.txt").write_text("Ohm's law: V = IR\n", encoding="utf-8")
    check(tmp_path / "notes.txt")
    (tmp_path / "notes.md").write_text("# Résumé — ε₀ and μ₀\n", encoding="utf-8")
    check(tmp_path / "notes.md")


def test_a_file_that_only_pretends_to_be_a_pdf(tmp_path):
    fake = tmp_path / "notes.pdf"
    fake.write_bytes(b"MZ\x90\x00 this is a program, not a PDF")
    with pytest.raises(UploadRefused, match="not really a PDF"):
        check(fake)


def test_a_damaged_pdf(tmp_path):
    broken = tmp_path / "notes.pdf"
    broken.write_bytes(b"%PDF-1.7\n" + b"\x00garbage" * 50)
    with pytest.raises(UploadRefused):
        check(broken)


def test_a_password_protected_pdf(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    path = tmp_path / "locked.pdf"
    doc.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    with pytest.raises(UploadRefused, match="password"):
        check(path)


def test_too_many_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "MAX_PAGES", 5)
    with pytest.raises(UploadRefused, match="6 pages"):
        check(_pdf(tmp_path / "book.pdf", pages=6))


def test_a_pdf_that_runs_a_script(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    xref = doc.get_new_xref()
    doc.update_object(xref, "<< /Type /Action /S /JavaScript /JS (app.alert(1)) >>")
    doc.xref_set_key(doc.pdf_catalog(), "OpenAction", f"{xref} 0 R")
    path = tmp_path / "script.pdf"
    doc.save(str(path))
    with pytest.raises(UploadRefused, match="scripts"):
        check(path)


def test_a_pdf_carrying_a_file(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    doc.embfile_add("run.exe", b"MZ program")
    path = tmp_path / "carrier.pdf"
    doc.save(str(path))
    with pytest.raises(UploadRefused, match="attached files"):
        check(path)


def test_a_giant_image(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "MAX_IMAGE_PIXELS", 50 * 50)
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    pix.clear_with(200)
    page.insert_image(page.rect, pixmap=pix)
    path = tmp_path / "big.pdf"
    doc.save(str(path))
    with pytest.raises(UploadRefused, match="Page 1 has an image too large"):
        check(path)


def test_a_name_that_only_starts_like_a_script_is_fine(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    xref = doc.get_new_xref()
    doc.update_object(xref, "<< /JSansFont 1 /Launcher 2 >>")
    path = tmp_path / "ok.pdf"
    doc.save(str(path), garbage=0)
    check(path)


def test_binary_bytes_in_a_text_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"looks like text\x00\x01\x02 but is not")
    with pytest.raises(UploadRefused, match="not plain text"):
        check(path)


def test_notepad_unicode_text_is_text(tmp_path):
    """UTF-16 with its mark, as Windows Notepad saves "Unicode"."""
    from ai.ingestion.extract import extract

    for encoding in ("utf-16", "utf-16-be"):
        path = tmp_path / f"notes-{encoding}.txt"
        data = "Ohm's law: V = IR. 抵抗".encode(encoding)
        if encoding == "utf-16-be":
            data = b"\xfe\xff" + data
        path.write_bytes(data)
        check(path)
        assert "Ohm's law: V = IR" in extract(path)[0].text


def test_a_utf8_mark_is_dropped(tmp_path):
    from ai.ingestion.extract import extract

    path = tmp_path / "notes.txt"
    path.write_bytes("\ufeffCapacitance".encode("utf-8"))
    check(path)
    assert extract(path)[0].text.startswith("Capacitance")
