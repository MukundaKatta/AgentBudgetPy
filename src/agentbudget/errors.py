"""Error types raised by ``Budget``.

Kept in their own module so consumers can ``from agentbudget.errors import …``
without pulling in the whole budget machinery.
"""

from __future__ import annotations

from typing import Literal

CapName = Literal["input_tokens", "output_tokens", "total_tokens", "cost_usd"]


class BudgetExceededError(Exception):
    """Raised by ``Budget.record_usage`` / ``Budget.assert_can_spend`` when a cap is hit.

    Carries the offending cap (so callers can build human messages without
    re-checking the budget) plus the totals as of *after* the rejected charge
    was theoretically applied — that's the most useful debug output.
    """

    cap: CapName
    limit: float
    attempted: float
    overshoot: float
    model: str | None

    def __init__(
        self,
        *,
        cap: CapName,
        limit: float,
        attempted: float,
        model: str | None = None,
    ) -> None:
        overshoot = attempted - limit
        formatted = (
            f"{overshoot:.4f}" if cap == "cost_usd" else f"{int(overshoot)}"
        )
        message = (
            f"agentbudget: {cap} cap exceeded — limit {limit}, attempted {attempted} "
            f"(over by {formatted})"
        )
        if model:
            message += f' on model "{model}"'
        super().__init__(message)
        self.cap = cap
        self.limit = limit
        self.attempted = attempted
        self.overshoot = overshoot
        self.model = model


class UnknownPricingError(Exception):
    """Raised when a ``cost_usd`` cap is configured but a recorded usage names
    a model with no rate in either ``pricing`` or ``DEFAULT_PRICING``. Without
    this we'd silently drift past the dollar ceiling, so by default we raise.

    Set ``allow_unknown_pricing=True`` on the ``Budget`` to charge $0 for
    unknown models instead.
    """

    model: str

    def __init__(self, model: str) -> None:
        super().__init__(
            f'agentbudget: no pricing entry for model "{model}" but a cost_usd cap is set. '
            "Add it to the budget's `pricing` map, or remove the cost_usd cap."
        )
        self.model = model
