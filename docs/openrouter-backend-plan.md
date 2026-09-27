# OpenRouter operators through the Codex agent

Run hillclimb operators on OpenRouter models — cheap, parallel, billed to
OpenRouter credits — without touching a Claude or ChatGPT subscription.

## Why

Today every operator call bills a subscription: `claude-code` against the
Claude plan, `codex` against the ChatGPT plan. A search is therefore paced by
session limits rather than by the machine, and a fleet of eight operators is
expensive in exactly the currency that runs out.

OpenRouter sells the cheap open models (`qwen/qwen3-coder`,
`deepseek/deepseek-v3`, …) at $0.02–0.15 per million input tokens, with a
`:free` tier for smoke tests. Pointing an existing agent CLI at it turns the
subscription ceiling into a dollar ceiling the engine already knows how to
enforce (`budget.max_cost_usd`).

## What the spike established

Measured on 2026-09-15 with codex 0.153.4 against
`https://openrouter.ai/api/v1`, model `cohere/north-mini-code:free`
(total spend: $0.00).

| question | result |
|---|---|
| Does codex drive a custom provider? | Yes — `-c model_providers.openrouter={…}` plus `-c model_provider=openrouter`; the full agent loop ran (tool calls, file written, `turn.completed`). |
| Does `wire_api` matter? | Yes. Codex 0.153 dropped Chat Completions (`wire_api = "chat"` is no longer supported), so the provider must be `wire_api = "responses"`. OpenRouter's `/api/v1/responses` satisfies this. |
| Does session resume survive? | Yes. OpenRouter's Responses API is stateless (it 400s on `store: true` or `previous_response_id`), but codex replays history client-side, so `codex exec resume <thread_id>` worked unchanged. Debug chains are safe. |
| Do the JSON events change? | No. The stream is byte-compatible with what `_CodexStreamReader` already normalizes — no parser work. |
| What does a call cost in tokens? | ~12.3k input per request (codex's system prompt + tool schemas); ~24.7k for a two-request turn. |
| Does personal config leak in? | Yes, and expensively. With the user's `~/.codex/config.toml` the same task cost 216k input tokens and emitted `service_tier`/skills warnings; with an isolated `CODEX_HOME` it cost 24.7k. |
| Can the preamble be trimmed further? | Not meaningfully. `include_*_instructions=false` + `tools.web_search=false` saved 3% (12,299 → 11,975). Not worth the config surface. |
| What happens when credits run out? | OpenRouter returns 402 and codex retries it five times before failing the turn. Codex always requests `max_output_tokens: 65536` and offers no knob to lower it, so a thin balance fails every call regardless of prompt size. |

Two consequences drive the design: `CODEX_HOME` isolation is a correctness
*and* cost feature, and a 402 must stop a search rather than be retried.

## Design

### 1. A third auth mode on the codex agent

`agent_auth` gains `openrouter` alongside `subscription` and `api-key`.
The model name becomes an OpenRouter id.

```yaml
agent: codex
agent_auth: openrouter
model: qwen/qwen3-coder
```

This rides the existing routing precedence (`routing.py`) with no new
machinery, because `Route.agent_auth` is already a per-operator field. A
mixed fleet is therefore config, not code:

```yaml
routing:
  draft:   {agent: claude-code, agent_auth: subscription, model: sonnet}
  improve: {models: [qwen/qwen3-coder, deepseek/deepseek-v3]}   # UCB1 bandit picks
  debug:   {model: cohere/north-mini-code:free}
```

Comparing models head-to-head needs no CLI work either — an experiment spec
with one arm per model (`arm_overrides: {model: …}`) already tags the searches
and feeds `chart` and `experiment report`.

`codex_env()`:

- `openrouter` → require `OPENROUTER_API_KEY` (raise a clear error naming the
  config knob when absent), drop `OPENAI_API_KEY` so an inherited key cannot
  silently redirect billing.
- `subscription` / `api-key` → unchanged.

`_command()` prepends, for `openrouter` only:

```
-c model_providers.openrouter={name="OpenRouter",base_url="https://openrouter.ai/api/v1",env_key="OPENROUTER_API_KEY",wire_api="responses"}
-c model_provider=openrouter
```

### 2. Isolated CODEX_HOME

Every codex call runs with `CODEX_HOME` pointed at
`~/.cache/hillclimb/codex-home/<auth>/`, created on demand:

- `subscription` copies `~/.codex/auth.json` in (the login is the point).
- `openrouter` needs no auth file at all — the env key is the credential.

The directory is machine-scoped state, so it belongs next to the other
`~/.cache/hillclimb/` entries (shared venvs, agent slots), not in a run dir.
This is what stops a search's behaviour depending on personal codex settings,
and it is worth 8× on the token bill.

### 3. Error kinds

`OperatorResult.error_kind` gains `out_of_credits`, raised when the provider
reports HTTP 402 or an insufficient-credits message. The searcher treats it
like a budget wall: stop the search with that reason rather than spending the
remaining budget on calls that cannot succeed. 429s keep flowing through the
existing `rate_limited` path (`RATE_LIMIT_MARKERS` already covers them).

### 4. Cost accounting — `pricing.py`

`_normalized_usage` currently keeps only `input_tokens` and `output_tokens`,
discarding `cached_input_tokens` and `cache_write_input_tokens`. Instead it
maps OpenRouter's usage onto the four canonical token keys the claude agent
already defines (`USAGE_TOKEN_KEYS`): OpenRouter's `input_tokens` *includes*
its cached subset, so the uncached remainder becomes `input_tokens`,
`cached_input_tokens` becomes `cache_read_input_tokens`, and
`cache_write_input_tokens` becomes `cache_creation_input_tokens`. One
vocabulary for both backends keeps `usage_total_tokens` correct (no double
counting) and lets pricing and the TUI read one shape. `reasoning_output_tokens`
is a subset of `output_tokens` and is not journaled separately.

Keeping the cache kinds is what makes the
next point measurable: the 12.3k preamble is identical across every operator
call of a search, so providers that auto-cache (DeepSeek, Anthropic, Google)
charge ~10% for those reads — and today hillclimb cannot tell you whether that
discount is landing.

A new module owns pricing, with no engine knowledge:

```python
def model_prices(model: str) -> Prices | None   # cached catalogue lookup
def cost_usd(model: str, token_usage: dict[str, int]) -> float | None
```

The catalogue comes from `/api/v1/models`, cached at
`~/.cache/hillclimb/openrouter-models.json` with a TTL, refreshed lazily and
never fetched during a test. `cost_usd` is a pure function over (model,
usage), so it tests offline against a checked-in fixture. An unknown model
yields `None`, never a guess — `budget.max_cost_usd` must not be enforced
against invented numbers.

`CodexCliAgent.invoke` fills `OperatorResult.cost_usd` from it for
`openrouter` auth only; subscription runs keep reporting `None` as today.

### 5. Key handling — `.env` beside `config.yaml`

`load_config` reads a `.env` in the hillclimb dir into the process
environment, never overriding a variable that is already set. The value
reaches exactly one place: the child process env built by `codex_env()`. It is
never journaled, never written to a status record, never echoed into an agent
stream, and `.env` is gitignored.

## Testing

- **Command construction** — `openrouter` auth produces the provider
  overrides and the right model; `subscription` produces today's argv
  unchanged (regression guard for the existing agent).
- **Env** — the key reaches the child env; `OPENAI_API_KEY` is dropped; a
  missing key raises a named error before any process spawns.
- **CODEX_HOME** — a per-auth dir is created; `auth.json` is copied for
  subscription and absent for openrouter.
- **Usage** — all token kinds survive `_normalized_usage`, including
  `cached_input_tokens`.
- **Pricing** — `cost_usd` against a fixture catalogue, including the
  cache-read discount and the unknown-model `None`.
- **402** — an OpenRouter payment-required turn yields
  `error_kind="out_of_credits"`, not `error` and not `rate_limited`.
- **Key leakage** — a search journal, status record and agent stream contain
  no `sk-or-` substring.

No test touches the network. The live smoke path stays
`hillclimb smoke --agent codex`, run manually against a `:free` model; it
takes the auth mode from `config.yaml` (`agent_auth: openrouter`) rather
than gaining a flag of its own.

## Documentation

- README's agent table gains the OpenRouter row and the three-line config.
- A short "which model" note: prefer providers with automatic prompt caching,
  because the 12.3k preamble repeats on every call.
- CLAUDE.md gains one line on the `agent_auth: openrouter` route, in the
  concurrency/agents area.

## Out of scope

- A generic `providers:` block for Groq/Together/local vLLM. The same
  mechanism generalises when a second provider is actually needed.
- An `opencode` agent. Only worthwhile if codex's Responses-only constraint
  becomes a problem; the spike says it is not.
- A `--model`-per-arm fleet CLI. Experiment arms already express it.
- Lowering codex's hard-coded 65536 output request. No config knob exists;
  the fix is credits, not code.
