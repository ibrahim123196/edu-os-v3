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
then say plainly which part your textbook or notes do not cover.
- If the sources do not address the question at all, reply with exactly this \
sentence and nothing else: "{insufficient}"
- Never invent a source number. Only cite sources that appear in the request.
- What the sources leave out is not evidence. When they discuss only one case, \
do not turn that into "only", "always", "never", "specifically" or "all": \
asked "is emf only present in cells?" when the sources explain emf using cells, \
the answer is that your textbook explains emf using cells and does not say \
here whether anything else has one -- not that emf is defined only for cells. \
Claim a limit only when a source states it.
- If two sources disagree, say so and cite both.
- Sources are labelled as the student's own notes or as the course \
textbook. Where both cover a point, follow the student's notes: that is what \
their teacher set and what their exam will follow. If the textbook adds \
something their notes do not have, use it and say where it came from.

Teaching rules:
- Write for a student revising the topic, not for an examiner.
- Talk to the student about the subject, not about the sources. Never write \
"the sources", "the passages", "the extract" or "your extract": the [S#] \
markers already say where each point comes from. ("Formula not fully shown in \
your extract" told a student nothing they could use.) Where something is \
missing, say "your textbook" or "your notes".
- Lead with the direct answer, then the explanation.
- Use the notation and terminology the sources use, so it matches what the \
student will see in their exam.
- Keep it tight. A clear paragraph or a short list beats an essay.

Writing formulas:
- Write them in plain readable notation, using real symbols: F = kq₁q₂/r², ε₀, ΔQ, 9 × 10⁹ N m² C⁻².
- Never use LaTeX, and never wrap anything in $ or $$. A student reading a backslash-frac expression learns nothing; it is not how their textbook or their exam paper writes it.
- The sources store subscripts as q_1 and superscripts as r^2, because that is what survived extraction from the PDF. Convert them to real characters (q₁, r²) rather than passing the underscores through.

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


FOLLOW_UP_TEMPLATE = """Sources:

{context}

---

This continues the conversation above. Earlier answers show what the student \
has already been told and how; they are not sources. State facts only from the \
sources above, and cite only these [S#] numbers -- earlier answers had their own \
sources, numbered differently.
{meaning}
Student's question: {question}

Answer using only the sources above, with [S#] markers."""


def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    history: list | None = None,
    standalone: str | None = None,
    extra: str | None = None,
) -> list[dict]:
    """The request for one answer.

    A first question gets exactly the single message it always has. A
    follow-up gets the earlier exchanges as real turns before it, so "that"
    and "more simply" refer to something.
    """
    # extra: further instructions for this one request (the known mistakes
    # to watch for), after everything else. Absent, the request is exactly
    # what it always was.
    tail = f"\n\n{extra}" if extra else ""
    if not history:
        return [
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    context=format_context(chunks), question=question
                ) + tail,
            }
        ]

    # The student's own words are kept -- "more simply" is an instruction
    # about how to answer -- with the resolved question beside them, so the
    # model and the search agree on what "that" was.
    meaning = (
        f"(In this conversation that means: {standalone})\n" if standalone else ""
    )
    turns = [{"role": t.role, "content": t.content} for t in history]
    return turns + [
        {
            "role": "user",
            "content": FOLLOW_UP_TEMPLATE.format(
                context=format_context(chunks), meaning=meaning, question=question
            ) + tail,
        }
    ]
