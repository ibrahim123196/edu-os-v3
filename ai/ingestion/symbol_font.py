"""Text set in the Symbol font, read as the characters it shows.

Symbol is a font of Greek letters and maths signs stored under ordinary
codes: the letter "D" in it is drawn as Δ, "µ" as ∝, "®" as →. A PDF that
does not say so (NCERT's do not) extracts as the codes, so "ΔTb = Kb m"
arrives as "DTb = Kb m", "F ∝ q1q2" as "F µ q1q2" (the micro sign), and the
arrow of a reaction as "®". A student -- or a search -- reading "DT" has
lost the quantity.

The table is the Adobe Symbol encoding, for the characters a science
textbook uses. Some PDFs store Symbol glyphs in the private-use area
(U+F020-U+F0FF), offset by 0xF000; those are mapped the same way.
"""

from __future__ import annotations

_GREEK = dict(zip(
    "ABGDEZHQIKLMNXOPRSTUFCYWabgdezhqiklmnxoprstufcywjvJV",
    "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψωϕϖϑς",
))

_SIGNS = {
    0xA2: "′", 0xA3: "≤", 0xA5: "∞", 0xAB: "↔", 0xAC: "←", 0xAD: "↑", 0xAE: "→",
    0xAF: "↓", 0xB0: "°", 0xB1: "±", 0xB2: "″", 0xB3: "≥", 0xB4: "×", 0xB5: "∝",
    0xB6: "∂", 0xB7: "•", 0xB8: "÷", 0xB9: "≠", 0xBA: "≡", 0xBB: "≈", 0xBC: "…",
    0xC4: "⊗", 0xC5: "⊕", 0xC6: "∅", 0xC7: "∩", 0xC8: "∪", 0xCE: "∈", 0xCF: "∉",
    0xD0: "∠", 0xD1: "∇", 0xD6: "√", 0xD7: "·", 0xDB: "⇔", 0xDC: "⇐", 0xDE: "⇒",
    0xE5: "∑", 0xF2: "∫", 0x2D: "−",
}

TABLE = {ord(k): v for k, v in _GREEK.items()} | _SIGNS


def is_symbol_font(name: str | None) -> bool:
    return bool(name) and "symbol" in name.lower()


def decode(text: str) -> str:
    """Symbol-font text as the characters it draws."""
    out = []
    for ch in text:
        code = ord(ch)
        if 0xF020 <= code <= 0xF0FF:         # private-use copy of the encoding
            code -= 0xF000
        out.append(TABLE.get(code, chr(code)))
    return "".join(out)
