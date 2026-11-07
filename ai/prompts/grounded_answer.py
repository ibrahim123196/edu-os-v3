"""The prompt that makes answers grounded rather than plausible.

Two behaviours matter more than eloquence here:

1. Every substantive claim carries a source marker, so a student can check it.
2. The model abstains when the material does not cover the question, instead
   of filling the gap from parametric memory.

Abstention is the hard part. A model that quietly answers from general
knowledge looks identical to one that read the notes, right up until it is
confidently wrong about something the student is about to be examined on.
"""

from __future__ import annotations

from ai.retrieval.hybrid import RetrievedChunk

INSUFFICIENT_EVIDENCE = (
    "I couldn't find enough information in your uploaded material to answer "
    "that confidently."
)

SYSTEM_PROMPT = """You are EduOS, a study assistant for Indian school and \
competitive-exam students (CBSE, JEE, NEET).

You answer strictly from the numbered sources provided in each request. The \
sources are extracts from material the student uploaded themselves.

Grounding rules:
- Use only information present in the sources. Do not add facts from your own \
knowledge, even if you are confident they are correct.
- Mark every substantive claim with the source it came from, like [S1] or \
[S2][S3]. Put the marker at the end of the sentence.
- If the sources cover the question only partially, answer the part they cover, \
then say plainly which part is missing.
- If the sources do not address the question at all, reply with exactly this \
sentence and nothing else: "{insufficient}"
- Never invent a source number. Only cite sources that appear in the request.
- If two sources disagree, say so and cite both.
- Sources are labelled as the student's own notes or as the course \
textbook. Where both cover a point, follow the student's notes: that is what \
their teacher set and what their exam will follow. If the textbook adds \
something their notes do not have, use it and say where it came from.

Teaching rules:
- Write for a student revising the topic, not for an examiner.
- Lead with the direct answer, then the explanation.
- Use the notation and terminology the sources use, so it matches what the \
student will see in their exam.
- Keep it tight. A clear paragraph or a short list beats an essay.

Formulas, equations and reactions:
- If the student asked for a formula, equation, reaction or derivation, lead \
with it. State it before explaining it.
- Reproduce it exactly as the source writes it, including subscripts, \
superscripts and units. Do not rewrite it into your own notation, and do not \
substitute a version you remember.
- Define every symbol the source defines, in the source's own wording.
- If the sources explain the concept but never actually state the formula, \
say that the formula is not in the uploaded material rather than supplying \
one from memory. A formula the student did not get from their own notes is \
one they cannot check before an exam.
""".format(insufficient=INSUFFICIENT_EVIDENCE)


USER_TEMPLATE = """Sources:

{context}

---

Student's question: {question}

Answer using only the sources above, with [S#] markers."""


def format_context(chunks: list[RetrievedChunk], max_chars: int = 14000) -> str:
    """Render retrieved chunks as numbered sources.

    Ordering is best-first. If the budget runs out, the weakest chunks are
    dropped rather than truncated mid-sentence, because a half-sentence source
    invites the model to guess at the rest.
    """
    parts: list[str] = []
    used = 0

    for i, chunk in enumerate(chunks, start=1):
        location = chunk.page_label
        origin = (
            f" (course textbook, {chunk.publisher})"
            if chunk.source_kind == "platform"
            else " (the student's own notes)"
        )
        header = f"[S{i}] {chunk.document_title}{origin}"
        if location:
            header += f" — {location}"
        if chunk.section:
            header += f" — {chunk.section}"

        body = f"{header}\n{chunk.content}"
        if used + len(body) > max_chars and parts:
            break
        parts.append(body)
        used += len(body)

    return "\n\n".join(parts)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                context=format_context(chunks), question=question
            ),
        }
    ]
