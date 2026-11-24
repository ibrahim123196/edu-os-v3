"""What the model calls in this run used, and roughly what they cost.

Every API reply says exactly how many tokens it used, and that was being
thrown away. The only way to learn what a run cost was the Billing page,
which can lag behind by hours: 87 question checks once left the balance
unchanged. Scripts now print a total at the end instead.

The cost is an estimate at list prices. The Billing page is the authority:
prices change, and discounts or tiers are not modelled here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

# US dollars per million tokens, (input, output). From Anthropic's models
# overview, 16 Nov 2026. A model missing here is reported in tokens only,
# never priced by guessing.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-5-5": (0.10, 0.50),   # listed as "from": the floor of its range
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-fable-5-2": (8.0, 40.0),
}


@dataclass
class ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost(self, model: str) -> float | None:
        price = PRICES.get(model)
        if price is None:
            return None
        return (self.input_tokens * price[0] + self.output_tokens * price[1]) / 1_000_000


_usage: dict[str, ModelUsage] = defaultdict(ModelUsage)

# Also told about every call: the server records each one against the
# student who asked (apps/api/ai_limits.py). A script has none.
_listeners: list = []


def add_listener(fn) -> None:
    if fn not in _listeners:
        _listeners.append(fn)


def record(model: str, input_tokens: int | None, output_tokens: int | None) -> None:
    u = _usage[model]
    u.calls += 1
    u.input_tokens += input_tokens or 0
    u.output_tokens += output_tokens or 0
    for fn in _listeners:
        fn(model, input_tokens, output_tokens)


def reset() -> None:
    _usage.clear()


def summary() -> str:
    """One line per model, then the total. Empty when nothing was called."""
    if not _usage:
        return ""
    lines, total, unpriced = [], 0.0, False
    for model, u in sorted(_usage.items()):
        cost = u.cost(model)
        price = f"≈ ${cost:.3f}" if cost is not None else "price not in table"
        if cost is None:
            unpriced = True
        else:
            total += cost
        lines.append(
            f"  {model}: {u.calls} call(s), {u.input_tokens:,} in / "
            f"{u.output_tokens:,} out tokens, {price}"
        )
    note = " (plus the unpriced model above)" if unpriced else ""
    lines.append(f"  total ≈ ${total:.3f}{note}, at list prices; Billing has the real figure")
    return "API use this run:\n" + "\n".join(lines)
