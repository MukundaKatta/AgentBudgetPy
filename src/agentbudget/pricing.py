"""Default pricing table + cost helpers.

A small starter map covering the most common Anthropic / OpenAI production
models as of early 2026. **Always verify with the provider's current pricing
page before relying on this for billing-critical work.**

Format: dollars per 1,000 tokens.

Cache + batch tiers are *not* modeled here on purpose — that's a per-account
configuration choice. If you use prompt caching, pass your own pricing map
with the cached input rate baked in.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, TypedDict


class ModelRate(TypedDict):
    input_per_1k: float
    output_per_1k: float


DEFAULT_PRICING: Mapping[str, ModelRate] = MappingProxyType(
    {
        # --- Anthropic (Claude) ---
        "claude-opus-4-5": {"input_per_1k": 0.015, "output_per_1k": 0.075},
        "claude-sonnet-4-7": {"input_per_1k": 0.003, "output_per_1k": 0.015},
        "claude-haiku-4-5": {"input_per_1k": 0.0008, "output_per_1k": 0.004},
        # --- OpenAI ---
        "gpt-4o": {"input_per_1k": 0.0025, "output_per_1k": 0.01},
        "gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
        "o1": {"input_per_1k": 0.015, "output_per_1k": 0.06},
        "o1-mini": {"input_per_1k": 0.003, "output_per_1k": 0.012},
    }
)


def compute_cost(rate: ModelRate, *, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a single LLM call given a per-1k rate."""
    return (
        rate["input_per_1k"] * input_tokens / 1000
        + rate["output_per_1k"] * output_tokens / 1000
    )
