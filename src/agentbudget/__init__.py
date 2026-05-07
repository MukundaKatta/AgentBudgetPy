"""agentbudget -- token + dollar caps for AI agents.

Public surface (mirrors the JS sibling at @mukundakatta/agentbudget):

* ``Budget``                  -- the main class; track usage, raise on overage
* ``BudgetExceededError``     -- raised when a cap is breached
* ``UnknownPricingError``     -- raised when costUsd is set but a model has no rate
* ``DEFAULT_PRICING``         -- starter rate table (Claude + GPT, early-2026)
* ``compute_cost``            -- utility: dollars for a single (rate, usage) pair
"""

from .budget import Budget, UsageInput
from .errors import BudgetExceededError, UnknownPricingError
from .pricing import DEFAULT_PRICING, ModelRate, compute_cost

__version__ = "0.1.0"
VERSION = __version__

__all__ = [
    "VERSION",
    "Budget",
    "BudgetExceededError",
    "DEFAULT_PRICING",
    "ModelRate",
    "UnknownPricingError",
    "UsageInput",
    "compute_cost",
]
