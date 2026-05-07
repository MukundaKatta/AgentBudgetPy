# agentbudget-py

Token + dollar budget caps for AI agents. Raises `BudgetExceededError` when an LLM call would push past the ceiling. Zero deps, drop into any provider SDK.

```bash
pip install agentbudget-py
```

Python port of [`@mukundakatta/agentbudget`](https://www.npmjs.com/package/@mukundakatta/agentbudget) — same API, snake_case names.

## Why

You ship an agent. A bug in the planner makes it loop. Your `claude-opus-4-5` bill is $300 before you notice.

`agentbudget` is one class. Set caps once, record usage after each call, raise the moment any cap is breached. CI catches loops; production catches runaways.

## Quickstart

```python
from agentbudget import Budget, BudgetExceededError

budget = Budget(
    max_total_tokens=200_000,   # hard token ceiling
    max_cost_usd=5.00,          # hard dollar ceiling
)

try:
    for turn in turns:
        resp = client.messages.create(...)
        budget.record_usage({
            "model": resp.model,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        })
except BudgetExceededError as err:
    print(f"stopped — {err.cap} cap of {err.limit} hit")
    raise
```

The raised `BudgetExceededError` carries `cap`, `limit`, `attempted`, `overshoot`, and `model` so you can build human messages without re-reading the budget.

## Caps

All optional, all checked after each `record_usage`. The first violation wins, in this order:

| Argument             | Caps                                          |
| -------------------- | --------------------------------------------- |
| `max_input_tokens`   | total input tokens across all calls           |
| `max_output_tokens`  | total output tokens across all calls          |
| `max_total_tokens`   | input + output combined                       |
| `max_cost_usd`       | dollars (requires pricing — see below)        |

## Auto-record with `wrap`

`Budget.wrap` adapts the Anthropic and OpenAI response shapes (object or dict) out of the box:

```python
import anthropic
from agentbudget import Budget

client = anthropic.Anthropic()
budget = Budget(max_cost_usd=1)

create = budget.wrap(client.messages.create)

await create(model="claude-sonnet-4-7", max_tokens=1024, messages=[...])
# budget.totals is updated automatically; raises if the cap is hit
```

For other providers, pass `extract_usage`:

```python
wrapped = budget.wrap(
    my_custom_call,
    extract_usage=lambda r: {
        "model": r["model_id"],
        "input_tokens": r["tokens"]["in"],
        "output_tokens": r["tokens"]["out"],
    },
)
```

## Pre-flight checks

Don't want to make the call when you're already near the cap? Use `would_exceed` (returns the cap name or `None`) or `assert_can_spend` (raises):

```python
if budget.would_exceed({"input_tokens": 8000, "output_tokens": 2000}):
    return await fallback()  # skip the call entirely

# or, in batch flows where you can split work:
budget.assert_can_spend(input_tokens=estimated_tokens)  # raises if not
```

## Pricing

`max_cost_usd` needs per-model rates. `agentbudget` ships a starter `DEFAULT_PRICING` table (Claude + GPT, early-2026 rates) and lets you override:

```python
from agentbudget import Budget

budget = Budget(
    max_cost_usd=10,
    pricing={
        # override one model
        "claude-sonnet-4-7": {"input_per_1k": 0.0015, "output_per_1k": 0.0075},  # cached rate
        # add a model the default doesn't know
        "my-finetune-v2": {"input_per_1k": 0.001, "output_per_1k": 0.001},
    },
)
```

Always verify the default rates against the provider's current pricing page before relying on them for billing-critical work.

If you call a model not in either table:

```python
Budget(max_cost_usd=1)                              # raises UnknownPricingError
Budget(max_cost_usd=1, allow_unknown_pricing=True)  # unknown models cost $0
Budget(max_total_tokens=1_000_000)                  # no cap, no error — pricing irrelevant
```

## Introspection

```python
budget.totals
# {'input_tokens': 12_400, 'output_tokens': 3_100, 'total_tokens': 15_500,
#  'cost_usd': 0.084, 'calls': 7}

budget.remaining()
# {
#   'total_tokens': {'used': 15500, 'limit': 200000, 'remaining': 184500},
#   'cost_usd':     {'used': 0.084,  'limit': 5,      'remaining': 4.916},
#   'calls': 7,
# }
```

`budget.reset()` zeroes the totals but keeps caps + pricing — useful for re-using one Budget across runs.

## Sibling libraries

Part of the [`@mukundakatta/agent*`](https://github.com/MukundaKatta?tab=repositories&q=agent) reliability stack:

- [agentsnap-py](https://pypi.org/project/agentsnap-py/) — snapshot tests for tool-call traces
- [agentguard-firewall](https://pypi.org/project/agentguard-firewall/) — network egress allowlist
- [agentcast-py](https://pypi.org/project/agentcast-py/) — JSON output enforcer
- [agentfit-py](https://pypi.org/project/agentfit-py/) — fit messages to context window
- [agentvet-py](https://pypi.org/project/agentvet-py/) — validate tool args before execution
- **agentbudget-py** — this lib

JS sibling: [`@mukundakatta/agentbudget`](https://www.npmjs.com/package/@mukundakatta/agentbudget) on npm.

## License

[MIT](LICENSE) © Mukunda Katta
