"""Chemistry drawn as pictures, written into the page text."""

from __future__ import annotations

import json

import pytest

from ai.ingestion.extract import Page
from ai.ingestion.transcriptions import (
    MARK,
    PAGE_START,
    Transcription,
    apply_transcriptions,
    load_transcriptions,
)

PAGE = Page(number=9, text="Example 6.4\nWrite the products of the following reactions:\nSolution\n(i) ...")


def test_a_transcription_goes_right_after_its_line():
    result = apply_transcriptions([PAGE], [
        Transcription(9, "Write the products of the following reactions:", "CH3CH2OH --(HCl)--> CH3CH2Cl"),
        Transcription(9, "Write the products of the following reactions:", "C6H6 --(Cl2, FeCl3)--> C6H5Cl"),
    ])
    lines = result.pages[0].text.split("\n")
    assert lines[1] == "Write the products of the following reactions:"
    assert lines[2] == f"{MARK} CH3CH2OH --(HCl)--> CH3CH2Cl"       # in order
    assert lines[3] == f"{MARK} C6H6 --(Cl2, FeCl3)--> C6H5Cl"
    assert lines[4] == "Solution"
    assert result.placed == 2 and not result.at_page_end


def test_a_line_must_match_whole_not_inside_a_sentence():
    """A short 'after' must not land in the middle of a longer line."""
    result = apply_transcriptions([PAGE], [Transcription(9, "Solution", "X")])
    assert result.pages[0].text.split("\n")[3] == f"{MARK} X"
    result = apply_transcriptions([PAGE], [Transcription(9, "the following", "Y")])
    assert result.pages[0].text.endswith(f"{MARK} Y")
    assert len(result.at_page_end) == 1


def test_a_line_that_is_gone_puts_it_at_the_end_of_its_page():
    result = apply_transcriptions([PAGE, Page(10, "next page")],
                                  [Transcription(9, "not on the page any more", "Z"),
                                   Transcription(9, None, "W")])
    assert result.pages[0].text.endswith(f"{MARK} Z\n{MARK} W")
    assert result.pages[1].text == "next page"                     # other pages untouched
    assert "placed at the end of their page" in result.summary


def test_a_drawing_before_every_line_goes_to_the_top_of_the_page():
    """after_line 0 is the top: the reactions heading page 9 were being sent
    to its bottom, after the questions that follow them."""
    result = apply_transcriptions([PAGE], [Transcription(9, PAGE_START, "A --> B")])
    assert result.pages[0].text.startswith(f"{MARK} A --> B\nExample 6.4")
    assert not result.at_page_end


def test_nothing_to_apply_leaves_the_pages_alone():
    assert apply_transcriptions([PAGE], []).pages == [PAGE]


def test_loading_skips_empty_and_malformed_entries(tmp_path):
    (tmp_path / "lech106.json").write_text(json.dumps({"items": [
        {"page": 9, "after": "Solution", "text": "A --> B"},
        {"page": 9, "text": "  "},
        {"page": "nine", "text": "C"},
        {"page": 10, "text": "D"},
    ]}), encoding="utf-8")
    items = load_transcriptions("lech106", tmp_path)
    assert [(i.page, i.after, i.text) for i in items] == [(9, "Solution", "A --> B"), (10, None, "D")]
    assert load_transcriptions("lech106.pdf", tmp_path)                # Windows' doubled extension
    assert load_transcriptions("other", tmp_path) == []


def test_a_broken_file_is_ignored_not_fatal(tmp_path):
    (tmp_path / "lech106.json").write_text("{not json", encoding="utf-8")
    assert load_transcriptions("lech106", tmp_path) == []


# ------------------------------------------------------------- the script

pymupdf = pytest.importorskip("pymupdf")

from scripts import transcribe_images  # noqa: E402


def test_the_models_reply_is_read_carefully():
    lines = ["Example 6.4", "Write the products:", "Solution"]
    reply = 'Here you go:\n{"items": [{"after_line": 2, "text": "A --> B"}, ' \
            '{"after_line": 99, "text": "C --> D"}, {"after_line": "x", "text": "E"}, ' \
            '{"after_line": 0, "text": "F"}, {"text": ""}]}'
    assert transcribe_images.parse_items(reply, lines, 9) == [
        {"page": 9, "after": "Write the products:", "text": "A --> B"},
        {"page": 9, "after": None, "text": "C --> D"},          # no such line: end of page
        {"page": 9, "after": None, "text": "E"},                # unreadable: end of page
        {"page": 9, "after": PAGE_START, "text": "F"},         # 0: before every line, the top
    ]
    assert transcribe_images.parse_items("I cannot read this page.", lines, 9) == []
    assert transcribe_images.parse_items('{"items": [', lines, 9) == []


class FakeVision:
    model = "fake-vision"

    def __init__(self):
        self.calls = []

    def complete(self, system, messages, max_tokens=None, temperature=None):
        content = messages[-1]["content"]
        # The page is shown, as two sharper halves, before the instructions.
        assert [c["type"] for c in content] == ["text", "image", "text", "image", "text"]
        text = content[-1]["text"]
        self.calls.append(text)
        page = int(text.split("This is page ")[1].split(" ")[0])
        return json.dumps({"items": [{"after_line": 1, "text": f"drawn on page {page}"}]})


def _chapter(tmp_path):
    doc = pymupdf.open()
    for n in range(3):
        doc.new_page().insert_text((72, 100), f"Printed text of page {n + 1}.", fontsize=11)
    path = tmp_path / "lech106.pdf"
    doc.save(path)
    return path


def test_the_script_saves_what_it_reads_and_keeps_other_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(transcribe_images, "TRANSCRIPTIONS_DIR", tmp_path / "out")
    pdf = _chapter(tmp_path)
    vision = FakeVision()
    assert transcribe_images.main([str(pdf)], llm=vision) == 0
    assert len(vision.calls) == 3
    saved = json.loads((tmp_path / "out" / "lech106.json").read_text(encoding="utf-8"))
    assert saved["model"] == "fake-vision"
    assert [i["text"] for i in saved["items"]] == [f"drawn on page {n}" for n in (1, 2, 3)]
    assert saved["items"][0]["after"] == "Printed text of page 1."

    # Asking again for one page replaces only that page's items.
    transcribe_images.main([str(pdf), "--pages", "2"], llm=FakeVision())
    saved = json.loads((tmp_path / "out" / "lech106.json").read_text(encoding="utf-8"))
    assert [i["page"] for i in saved["items"]] == [1, 2, 3]

    # And ingestion's loader reads the file back and places the items.
    items = load_transcriptions("lech106", tmp_path / "out")
    pages = apply_transcriptions([Page(1, "Printed text of page 1.")], items).pages
    assert pages[0].text == f"Printed text of page 1.\n{MARK} drawn on page 1"


def test_a_missing_file_is_reported(tmp_path, capsys):
    assert transcribe_images.main([str(tmp_path / "nope.pdf")], llm=FakeVision()) == 1
    assert "No such file" in capsys.readouterr().out
