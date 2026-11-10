"""Split page-tagged text into retrievable chunks.

Design constraints:

1. A chunk must know which pages it came from, or it cannot be cited.
2. Chunks should break on paragraph boundaries, not mid-sentence, so a
   retrieved passage reads as a coherent explanation.
3. Consecutive chunks overlap, so a definition that straddles a boundary is
   still fully present in at least one chunk.

Word counts stand in for token counts here. They are within ~30% of tokens for
English prose, and avoiding a tokenizer dependency keeps ingestion fast on the
low-spec machines this is meant to run on. Swap in tiktoken if the drift starts
to matter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ai.ingestion.extract import Page


@dataclass(slots=True)
class Chunk:
    index: int
    content: str
    page_start: int
    page_end: int
    word_count: int
    section: str | None = None


@dataclass(slots=True)
class _Block:
    """A paragraph, tagged with the page it came from."""

    text: str
    page: int
    words: int


# A heading: short, no terminal punctuation, often numbered.
_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][^.!?]{2,80}$"
)

# A real section heading carries the textbook's own number: "2.2
# ELECTROSTATIC POTENTIAL". Margin biographies and figure captions look like
# headings too, and on one page "COUNT ALESSANDRO VOLTA (1745-1827)"
# displaced the section heading sitting right beside it -- which left four
# pages of a chapter with nothing to tag them against.
_NUMBERED_HEADING = re.compile(r"^\d+\.\d+(?:\.\d+)?\s+\S")


def _is_numbered_heading(text: str) -> bool:
    return bool(_NUMBERED_HEADING.match(text.strip()))


def _looks_like_heading(text: str) -> bool:
    if len(text) > 90 or "\n" in text:
        return False
    words = text.split()
    if not (1 <= len(words) <= 12):
        return False
    return bool(_HEADING.match(text.strip()))


def _blocks_from_pages(pages: list[Page]) -> list[_Block]:
    blocks: list[_Block] = []
    for page in pages:
        for para in page.text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            blocks.append(_Block(text=para, page=page.number, words=len(para.split())))
    return blocks


def _hard_split(block: _Block, target: int) -> list[_Block]:
    """Last resort for text with no sentence punctuation at all.

    Tables, equation dumps and badly extracted PDF pages routinely arrive as
    one enormous run of words. Splitting on word count keeps them from
    swallowing an entire chunk.
    """
    words = block.text.split()
    if len(words) <= target:
        return [block]
    return [
        _Block(
            text=" ".join(words[i : i + target]),
            page=block.page,
            words=len(words[i : i + target]),
        )
        for i in range(0, len(words), target)
    ]


def _split_long_block(block: _Block, target: int) -> list[_Block]:
    """A single paragraph bigger than the target gets split on sentences."""
    if block.words <= target:
        return [block]

    sentences = re.split(r"(?<=[.!?])\s+", block.text)
    out: list[_Block] = []
    buf: list[str] = []
    count = 0
    for sentence in sentences:
        n = len(sentence.split())
        if count + n > target and buf:
            joined = " ".join(buf)
            out.append(_Block(text=joined, page=block.page, words=count))
            buf, count = [], 0
        buf.append(sentence)
        count += n
    if buf:
        out.append(_Block(text=" ".join(buf), page=block.page, words=count))

    # Sentence splitting cannot help if there were no sentence boundaries.
    final: list[_Block] = []
    for piece in out:
        final.extend(_hard_split(piece, target))
    return final


def _tail_overlap(blocks: list[_Block], overlap_words: int) -> list[_Block]:
    """Carry the trailing `overlap_words` words into the next chunk.

    Whole blocks are preferred, but a block far larger than the budget is
    trimmed to its tail -- otherwise a single big paragraph would drag the
    next chunk to double the target size.
    """
    if overlap_words <= 0 or not blocks:
        return []

    picked: list[_Block] = []
    total = 0
    for block in reversed(blocks):
        if total >= overlap_words:
            break
        remaining = overlap_words - total
        if block.words > remaining * 2:
            words = block.text.split()[-remaining:]
            picked.append(
                _Block(text=" ".join(words), page=block.page, words=len(words))
            )
            total += len(words)
        else:
            picked.append(block)
            total += block.words
    return list(reversed(picked))


def chunk_pages(
    pages: list[Page],
    target_words: int = 220,
    overlap_words: int = 45,
    min_words: int = 25,
) -> list[Chunk]:
    """Group page-tagged paragraphs into overlapping, citable chunks."""
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    if overlap_words >= target_words:
        raise ValueError("overlap_words must be smaller than target_words")

    raw_blocks = _blocks_from_pages(pages)
    blocks: list[_Block] = []
    for block in raw_blocks:
        blocks.extend(_split_long_block(block, target_words))

    chunks: list[Chunk] = []
    current: list[_Block] = []
    current_words = 0
    current_heading: str | None = None
    pending_heading: str | None = None
    # True once `current` holds text that has not already been emitted. Guards
    # against flushing carried-over overlap as a chunk in its own right.
    has_new_content = False

    def flush() -> None:
        nonlocal current, current_words, current_heading, has_new_content
        if not current or not has_new_content:
            return
        body = [b for b in current]
        content = "\n\n".join(b.text for b in body).strip()
        if not content:
            current, current_words = [], 0
            return
        chunks.append(
            Chunk(
                index=len(chunks),
                content=content,
                page_start=min(b.page for b in body),
                page_end=max(b.page for b in body),
                word_count=sum(b.words for b in body),
                section=current_heading,
            )
        )
        carry = _tail_overlap(body, overlap_words)
        current = list(carry)
        current_words = sum(b.words for b in carry)
        has_new_content = False

    for block in blocks:
        if _looks_like_heading(block.text):
            # A heading belongs with the text under it, so start a new chunk.
            if has_new_content and current_words >= min_words:
                flush()
                current, current_words = [], 0
            pending_heading = block.text

        if pending_heading and not _looks_like_heading(block.text):
            # A numbered heading always wins and holds until the next one.
            # An unnumbered candidate only applies while no numbered heading
            # has been seen, so a margin note cannot displace the section it
            # happens to sit next to.
            if _is_numbered_heading(pending_heading) or not (
                current_heading and _is_numbered_heading(current_heading)
            ):
                current_heading = pending_heading
            pending_heading = None

        # Close the current chunk first if this block would overshoot it.
        if (
            has_new_content
            and current_words >= min_words
            and current_words + block.words > target_words
        ):
            flush()

        current.append(block)
        current_words += block.words
        has_new_content = True

        if current_words >= target_words:
            flush()

    # Final partial chunk: keep it unless it is only carried-over overlap.
    if current and has_new_content and current_words >= min_words:
        content = "\n\n".join(b.text for b in current).strip()
        already_covered = bool(chunks) and content in chunks[-1].content
        if not already_covered:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    content=content,
                    page_start=min(b.page for b in current),
                    page_end=max(b.page for b in current),
                    word_count=current_words,
                    section=current_heading,
                )
            )

    # A very short document still deserves one chunk.
    if not chunks and raw_blocks:
        content = "\n\n".join(b.text for b in raw_blocks).strip()
        chunks.append(
            Chunk(
                index=0,
                content=content,
                page_start=min(b.page for b in raw_blocks),
                page_end=max(b.page for b in raw_blocks),
                word_count=sum(b.words for b in raw_blocks),
                section=None,
            )
        )

    return chunks
