"""The ``Budget`` class -- post-call cap enforcement for AI agents.

Mirrors the JS sibling at @mukundakatta/agentbudget. Caps are checked after
each ``record_usage`` so the offending call still happens once and then the
budget refuses any further work — that matches the most common provider SDK
pattern (you only know token counts after the response).
"""

from __future__ import annotations

import math
from typing import (
    Any,
    Awaitable,
    Callable,
    Mapping,
    Optional,
    TypedDict,
    TypeVar,
)

from .errors import BudgetExceededError, CapName, UnknownPricingError
from .pricing import DEFAULT_PRICING, ModelRate, compute_cost


class UsageInput(TypedDict, total=False):
    """Shape passed to ``record_usage`` and friends.

    ``model`` is optional — required only when a ``cost_usd`` cap is set
    (and even then, it's optional if you've passed ``allow_unknown_pricing=True``).
    """

    model: str
    input_tokens: int
    output_tokens: int


class _Totals(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    calls: int


T = TypeVar("T")


# Cap evaluation order — first violation wins. Keeps a more specific cap from
# being shadowed by a coincidental total-tokens overrun.
_CAP_ORDER: tuple[CapName, ...] = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
)


class Budget:
    """Token + dollar budget for an agent run.

    Construct one per agent run; share across the calls inside that run. All
    cap kwargs are optional — if you only want a token cap, pass only
    ``max_total_tokens``. If you only want dollars, pass only ``max_cost_usd``
    plus a ``pricing`` table (or rely on ``DEFAULT_PRICING``).
    """

    def __init__(
        self,
        *,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
        max_total_tokens: int | None = None,
        max_cost_usd: float | None = None,
        pricing: Mapping[str, ModelRate] | None = None,
        allow_unknown_pricing: bool = False,
    ) -> None:
        self.caps: dict[CapName, float | None] = {
            "input_tokens": max_input_tokens,
            "output_tokens": max_output_tokens,
            "total_tokens": max_total_tokens,
            "cost_usd": max_cost_usd,
        }
        # User-supplied pricing wins; we fall back to DEFAULT_PRICING per-lookup
        # so a partial map (just one custom model) still benefits from defaults.
        self.pricing: Mapping[str, ModelRate] = pricing or {}
        self.allow_unknown_pricing = bool(allow_unknown_pricing)

        self.totals: _Totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "calls": 0,
        }

    # --- public API ---------------------------------------------------------

    def record_usage(self, usage: UsageInput) -> _Totals:
        """Tally one LLM call's usage and raise if any cap is now breached.

        Returns a copy of the totals after the charge, mirroring the JS API.
        """
        self._validate(usage)
        cost = self._cost_for(usage)

        self.totals["input_tokens"] += int(usage["input_tokens"])
        self.totals["output_tokens"] += int(usage["output_tokens"])
        self.totals["total_tokens"] = (
            self.totals["input_tokens"] + self.totals["output_tokens"]
        )
        self.totals["cost_usd"] += cost
        self.totals["calls"] += 1

        violated = self._first_violation(self.totals)
        if violated is not None:
            raise BudgetExceededError(
                cap=violated,
                limit=float(self.caps[violated] or 0),
                attempted=float(self.totals[violated]),
                model=usage.get("model"),
            )
        return dict(self.totals)  # type: ignore[return-value]

    def would_exceed(self, usage: UsageInput) -> CapName | None:
        """Pre-flight check — returns the cap name that *would* be tripped, or
        ``None`` if the call is safe. Doesn't mutate totals.
        """
        self._validate(usage)
        projected = self._project(usage)
        return self._first_violation(projected)

    def assert_can_spend(
        self, *, model: str | None = None, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        """Raise ``BudgetExceededError`` if the projected totals would breach.

        Useful when you can split the work — call this before scheduling more.
        """
        usage: UsageInput = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if model is not None:
            usage["model"] = model
        projected = self._project(usage)
        violated = self._first_violation(projected)
        if violated is not None:
            raise BudgetExceededError(
                cap=violated,
                limit=float(self.caps[violated] or 0),
                attempted=float(projected[violated]),
                model=model,
            )

    def wrap(
        self,
        fn: Callable[..., Awaitable[T]],
        *,
        extract_usage: Optional[Callable[[T], UsageInput | None]] = None,
    ) -> Callable[..., Awaitable[T]]:
        """Wrap an async callable so its result's ``.usage`` is auto-recorded.

        The default extractor knows the Anthropic + OpenAI shapes (objects or
        dicts). For other providers, pass ``extract_usage``.
        """
        extractor = extract_usage or _default_extract_usage

        async def wrapped(*args: Any, **kwargs: Any) -> T:
            result = await fn(*args, **kwargs)
            usage = extractor(result)
            if usage is not None:
                self.record_usage(usage)
            return result

        return wrapped

    def reset(self) -> None:
        """Zero the totals. Caps + pricing preserved."""
        for k in self.totals:
            self.totals[k] = 0 if k != "cost_usd" else 0.0  # type: ignore[literal-required]

    def remaining(self) -> dict[str, Any]:
        """Per-cap snapshot of {used, limit, remaining}. Skips unset caps."""
        out: dict[str, Any] = {"calls": self.totals["calls"]}
        for cap in _CAP_ORDER:
            limit = self.caps[cap]
            if limit is None:
                continue
            used = self.totals[cap]
            out[cap] = {"used": used, "limit": limit, "remaining": limit - used}
        return out

    # --- internals ----------------------------------------------------------

    def _project(self, usage: UsageInput) -> _Totals:
        cost = self._cost_for(usage)
        input_tokens = self.totals["input_tokens"] + int(usage["input_tokens"])
        output_tokens = self.totals["output_tokens"] + int(usage["output_tokens"])
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": self.totals["cost_usd"] + cost,
            "calls": self.totals["calls"] + 1,
        }

    def _first_violation(self, totals: _Totals) -> CapName | None:
        for cap in _CAP_ORDER:
            limit = self.caps[cap]
            if limit is None:
                continue
            if totals[cap] > limit:
                return cap
        return None

    def _cost_for(self, usage: UsageInput) -> float:
        if self.caps["cost_usd"] is None:
            # No cost cap → no need to compute, no need to raise on missing rates.
            return 0.0
        model = usage.get("model")
        rate = self._lookup_rate(model)
        if rate is None:
            if self.allow_unknown_pricing:
                return 0.0
            raise UnknownPricingError(model or "<unknown>")
        return compute_cost(
            rate,
            input_tokens=int(usage["input_tokens"]),
            output_tokens=int(usage["output_tokens"]),
        )

    def _lookup_rate(self, model: str | None) -> ModelRate | None:
        if not model:
            return None
        return self.pricing.get(model) or DEFAULT_PRICING.get(model)

    @staticmethod
    def _validate(usage: UsageInput) -> None:
        if not isinstance(usage, dict):
            raise TypeError("agentbudget: usage must be a dict")
        for key in ("input_tokens", "output_tokens"):
            if key not in usage:
                raise TypeError(f"agentbudget: usage.{key} is required")
            v = usage[key]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise TypeError(
                    f"agentbudget: usage.{key} must be a non-negative finite number"
                )
            if v < 0 or not math.isfinite(v):  # rejects negatives, NaN, and ±inf
                raise TypeError(
                    f"agentbudget: usage.{key} must be a non-negative finite number"
                )


def _default_extract_usage(result: Any) -> UsageInput | None:
    """Adapt Anthropic and OpenAI response shapes (object or dict) to ``UsageInput``.

    Returns ``None`` if the shape isn't recognized — wrapped functions whose
    responses don't carry usage (mocks, cached responses) silently no-op rather
    than raising.
    """
    if result is None:
        return None
    usage = _attr(result, "usage")
    if usage is None:
        return None
    model = _attr(result, "model")
    # Anthropic: input_tokens / output_tokens
    inp = _attr(usage, "input_tokens")
    out = _attr(usage, "output_tokens")
    if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
        item: UsageInput = {"input_tokens": int(inp), "output_tokens": int(out)}
        if isinstance(model, str):
            item["model"] = model
        return item
    # OpenAI: prompt_tokens / completion_tokens
    inp = _attr(usage, "prompt_tokens")
    out = _attr(usage, "completion_tokens")
    if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
        item = {"input_tokens": int(inp), "output_tokens": int(out)}
        if isinstance(model, str):
            item["model"] = model
        return item
    return None


def _attr(obj: Any, name: str) -> Any:
    """Get ``obj.name`` or ``obj[name]``, whichever works. ``None`` if neither."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
