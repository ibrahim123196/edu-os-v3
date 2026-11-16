"""Counting what model calls cost.

The Billing page lagged by hours after 87 question checks, so scripts now
total the token counts every reply carries.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai.generation import usage
from ai.generation.llm import AnthropicClient


@pytest.fixture(autouse=True)
def fresh():
    usage.reset()
    yield
    usage.reset()


def test_cost_at_list_prices():
    # Sonnet 5.5 at $2 in / $10 out per million tokens.
    usage.record("claude-sonnet-5-5", 1_000_000, 100_000)
    assert "≈ $3.000" in usage.summary()


def test_calls_add_up_per_model():
    usage.record("claude-haiku-4-5", 1_000, 200)
    usage.record("claude-haiku-4-5", 3_000, 800)
    line = next(l for l in usage.summary().splitlines() if "haiku-4-5" in l)
    assert "2 call(s), 4,000 in / 1,000 out tokens" in line
    assert "≈ $0.009" in line  # 4000 × $1 + 1000 × $5, per million


def test_a_model_without_a_price_is_counted_not_guessed():
    usage.record("some-new-model", 500, 50)
    summary = usage.summary()
    assert "price not in table" in summary
    assert "500 in / 50 out" in summary
    assert "plus the unpriced model" in summary


def test_nothing_called_prints_nothing():
    assert usage.summary() == ""


def test_an_anthropic_reply_is_recorded(monkeypatch):
    client = AnthropicClient("claude-sonnet-5-5", api_key="test-key")
    reply = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
    )
    monkeypatch.setattr(client._client.messages, "create", lambda **kwargs: reply)
    assert client.complete("system", [{"role": "user", "content": "hi"}]) == "ok"
    assert "1 call(s), 1,200 in / 300 out tokens" in usage.summary()


def test_a_reply_without_usage_still_counts_as_a_call(monkeypatch):
    """Older SDKs or a stub may omit usage; the call is not silently lost."""
    client = AnthropicClient("claude-sonnet-5-5", api_key="test-key")
    reply = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
    monkeypatch.setattr(client._client.messages, "create", lambda **kwargs: reply)
    client.complete("system", [{"role": "user", "content": "hi"}])
    assert "1 call(s), 0 in / 0 out tokens" in usage.summary()
