"""Tests for the Budget class. Mirrors the JS sibling's coverage."""

from __future__ import annotations

import asyncio
import math

import pytest

from agentbudget import (
    VERSION,
    Budget,
    BudgetExceededError,
    DEFAULT_PRICING,
    UnknownPricingError,
    compute_cost,
)


def test_version_is_semverish():
    assert isinstance(VERSION, str)
    assert VERSION.count(".") >= 2


# --- record_usage --------------------------------------------------------


def test_record_usage_tallies_tokens_and_calls():
    b = Budget()
    b.record_usage({"input_tokens": 100, "output_tokens": 50})
    b.record_usage({"input_tokens": 30, "output_tokens": 20})
    assert b.totals["input_tokens"] == 130
    assert b.totals["output_tokens"] == 70
    assert b.totals["total_tokens"] == 200
    assert b.totals["calls"] == 2


def test_record_usage_raises_on_input_tokens_cap():
    b = Budget(max_input_tokens=100)
    b.record_usage({"input_tokens": 80, "output_tokens": 0})
    with pytest.raises(BudgetExceededError) as exc:
        b.record_usage({"input_tokens": 30, "output_tokens": 0})
    assert exc.value.cap == "input_tokens"
    assert exc.value.attempted == 110
    assert exc.value.limit == 100


def test_record_usage_raises_on_total_tokens_cap():
    b = Budget(max_total_tokens=100)
    with pytest.raises(BudgetExceededError) as exc:
        b.record_usage({"input_tokens": 60, "output_tokens": 50})
    assert exc.value.cap == "total_tokens"


def test_cost_cap_with_default_pricing():
    b = Budget(max_cost_usd=0.01)
    # claude-sonnet-4-7: 0.003/1k in, 0.015/1k out
    # 1000 in + 500 out = 0.003 + 0.0075 = 0.0105 → over $0.01.
    with pytest.raises(BudgetExceededError) as exc:
        b.record_usage(
            {"model": "claude-sonnet-4-7", "input_tokens": 1000, "output_tokens": 500}
        )
    assert exc.value.cap == "cost_usd"


def test_totals_still_updated_when_call_trips_cap():
    # Documents the contract: by the time you have token counts, the call
    # already cost money. We tally + raise, not skip.
    b = Budget(max_input_tokens=100)
    with pytest.raises(BudgetExceededError):
        b.record_usage({"input_tokens": 200, "output_tokens": 0})
    assert b.totals["input_tokens"] == 200
    assert b.totals["calls"] == 1


def test_first_violation_wins_when_multiple_caps_tripped():
    # Cap order: input → output → total → cost. Both input and total are
    # over here; we report input (the more specific cap).
    b = Budget(max_input_tokens=50, max_total_tokens=50)
    with pytest.raises(BudgetExceededError) as exc:
        b.record_usage({"input_tokens": 100, "output_tokens": 100})
    assert exc.value.cap == "input_tokens"


# --- pricing edge cases --------------------------------------------------


def test_unknown_pricing_error_when_cost_cap_and_unknown_model():
    b = Budget(max_cost_usd=1)
    with pytest.raises(UnknownPricingError) as exc:
        b.record_usage({"model": "nope-3", "input_tokens": 1, "output_tokens": 1})
    assert exc.value.model == "nope-3"


def test_allow_unknown_pricing_charges_zero():
    b = Budget(max_cost_usd=1, allow_unknown_pricing=True)
    b.record_usage({"model": "preview-x", "input_tokens": 1000, "output_tokens": 1000})
    assert b.totals["cost_usd"] == 0


def test_no_cost_cap_means_no_pricing_needed():
    b = Budget(max_total_tokens=1_000_000)
    b.record_usage({"model": "nope-3", "input_tokens": 1, "output_tokens": 1})
    assert b.totals["calls"] == 1


def test_user_pricing_wins_over_default():
    b = Budget(
        max_cost_usd=1,
        pricing={"claude-sonnet-4-7": {"input_per_1k": 100, "output_per_1k": 100}},
    )
    # 100 / 1k * 5 in + 100 / 1k * 5 out = 1.0 → exactly at cap, just under.
    b.record_usage(
        {"model": "claude-sonnet-4-7", "input_tokens": 5, "output_tokens": 5}
    )
    assert math.isclose(b.totals["cost_usd"], 1.0)
    with pytest.raises(BudgetExceededError):
        b.record_usage(
            {"model": "claude-sonnet-4-7", "input_tokens": 1, "output_tokens": 0}
        )


# --- pre-flight ----------------------------------------------------------


def test_would_exceed_returns_cap_name_without_mutating():
    b = Budget(max_input_tokens=100)
    assert b.would_exceed({"input_tokens": 50, "output_tokens": 0}) is None
    assert b.would_exceed({"input_tokens": 200, "output_tokens": 0}) == "input_tokens"
    assert b.totals["input_tokens"] == 0


def test_assert_can_spend_raises_but_does_not_mutate():
    b = Budget(max_input_tokens=100)
    b.record_usage({"input_tokens": 50, "output_tokens": 0})
    with pytest.raises(BudgetExceededError):
        b.assert_can_spend(input_tokens=80)
    assert b.totals["input_tokens"] == 50


# --- wrap ----------------------------------------------------------------


def test_wrap_records_anthropic_shape():
    b = Budget(max_total_tokens=1000)

    async def fake_create():
        return {
            "model": "claude-sonnet-4-7",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    asyncio.run(b.wrap(fake_create)())
    assert b.totals["total_tokens"] == 15
    assert b.totals["calls"] == 1


def test_wrap_records_openai_shape():
    b = Budget()

    async def fake_create():
        return {
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

    asyncio.run(b.wrap(fake_create)())
    assert b.totals["input_tokens"] == 100
    assert b.totals["output_tokens"] == 50


def test_wrap_with_custom_extractor():
    b = Budget()

    async def fake_create():
        return {"tokens": {"in": 7, "out": 3}}

    wrapped = b.wrap(
        fake_create,
        extract_usage=lambda r: {
            "input_tokens": r["tokens"]["in"],
            "output_tokens": r["tokens"]["out"],
        },
    )
    asyncio.run(wrapped())
    assert b.totals["total_tokens"] == 10


def test_wrap_silently_noops_when_no_usage():
    b = Budget()

    async def fake():
        return {"content": "cached"}

    asyncio.run(b.wrap(fake)())
    assert b.totals["calls"] == 0


def test_wrap_raises_on_call_that_trips_cap():
    b = Budget(max_total_tokens=100)

    async def fake():
        return {"usage": {"input_tokens": 60, "output_tokens": 50}}

    with pytest.raises(BudgetExceededError):
        asyncio.run(b.wrap(fake)())


# --- input validation ----------------------------------------------------


def test_record_usage_rejects_bad_input():
    b = Budget()
    with pytest.raises(TypeError):
        b.record_usage(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        b.record_usage({"input_tokens": -1, "output_tokens": 0})
    with pytest.raises(TypeError):
        b.record_usage({"input_tokens": float("nan"), "output_tokens": 0})
    with pytest.raises(TypeError):
        b.record_usage({"input_tokens": float("inf"), "output_tokens": 0})
    with pytest.raises(TypeError):
        b.record_usage({"input_tokens": 0, "output_tokens": float("inf")})
    with pytest.raises(TypeError):
        b.record_usage({"input_tokens": "lots", "output_tokens": 0})  # type: ignore[typeddict-item]


def test_record_usage_rejects_infinity_without_mutating():
    # Regression: float("inf") used to slip past validation and blow up with a
    # low-level OverflowError inside the tally. It must raise TypeError up front
    # and leave totals untouched.
    b = Budget()
    with pytest.raises(TypeError):
        b.record_usage({"input_tokens": float("inf"), "output_tokens": 0})
    assert b.totals["input_tokens"] == 0
    assert b.totals["calls"] == 0


# --- introspection -------------------------------------------------------


def test_remaining_reports_per_cap_skipping_unset():
    b = Budget(max_input_tokens=1000, max_cost_usd=1)
    b.record_usage(
        {"model": "claude-sonnet-4-7", "input_tokens": 200, "output_tokens": 100}
    )
    r = b.remaining()
    assert r["input_tokens"] == {"used": 200, "limit": 1000, "remaining": 800}
    assert r["calls"] == 1
    # output_tokens has no cap → not surfaced.
    assert "output_tokens" not in r
    assert r["cost_usd"]["used"] > 0


def test_reset_zeroes_totals_but_preserves_caps():
    b = Budget(max_input_tokens=100)
    b.record_usage({"input_tokens": 50, "output_tokens": 0})
    b.reset()
    assert b.totals["input_tokens"] == 0
    assert b.totals["calls"] == 0
    # Caps still enforced after reset.
    with pytest.raises(BudgetExceededError):
        b.record_usage({"input_tokens": 200, "output_tokens": 0})


# --- pricing helpers -----------------------------------------------------


def test_compute_cost_matches_simple_math():
    cost = compute_cost(
        {"input_per_1k": 0.003, "output_per_1k": 0.015},
        input_tokens=1000,
        output_tokens=500,
    )
    # 1000 × 0.003/1k + 500 × 0.015/1k = 0.003 + 0.0075
    assert round(cost, 6) == 0.0105


def test_default_pricing_is_immutable():
    # MappingProxyType doesn't allow mutation.
    with pytest.raises(TypeError):
        DEFAULT_PRICING["claude-sonnet-4-7"] = {  # type: ignore[index]
            "input_per_1k": 0,
            "output_per_1k": 0,
        }
