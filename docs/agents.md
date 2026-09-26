# Agents

The coding agents that run operators — Claude Code, Codex, pi — how to connect one, who pays, and sampling.

## Install (from source)

```bash
uv sync
claude login   # operator calls bill your Claude subscription
```

Optional — shell tab-completion for commands, subcommands, and options:

```bash
hillclimb --install-completion   # writes into your shell config; restart the shell
```

## Supported agents

Operators are headless coding-agent processes, one per operator call, behind
the backend seam in `src/hillclimb/backends/`:

| backend | what it is |
|---|---|
| `claude-code` | Claude Code in headless mode — the production backend; bills your Claude subscription |
| `codex` | Codex CLI in non-interactive mode; uses your Codex login by default |
| `codex` + `backend_auth: openrouter` | the same Codex CLI pointed at OpenRouter: cheap open models billed to OpenRouter credits, no subscription touched |
| `pi` | pi coding agent in JSON mode; subscription login, API keys, OpenRouter or custom local providers, with per-operator sampling |
| `dummy` | no model calls: a scripted operator for exercising the engine, TUIs and run layout |
| `fake` | deterministic canned operator for the test suite |

Other agents (OpenCode, …) plug in at the same seam: a backend
implements the `OperatorBackend` protocol in `backends/base.py` — take a prompt plus a
working directory, return the agent's JSON result — and is selected with
`--backend <name>`.

## Connecting an agent

```
$ hillclimb connect
   target      billing       state
●  claude      subscription  ready       logged in (claude.ai, you@example.com) — 6% of the 5-hour window used
   codex       subscription  ready       Logged in using ChatGPT
   pi          subscription  logged-out  no provider logged in at ~/.pi/agent/auth.json
   openrouter  openrouter    no-key      OPENROUTER_API_KEY is not set
```

Every credential is read **through the same environment an operator gets**, so
what the table says is what a search will find — an `ANTHROPIC_API_KEY` left in
your shell, which would quietly rebill a "subscription" search to the API, shows
up here rather than on an invoice. `●` is the backend this config runs by
default.

`hillclimb connect claude` (or `codex`, `pi`, `openrouter`) sets one up: it runs
that agent's own login, stages the credential in the isolated per-auth home
searches read (`~/.cache/hillclimb/codex-home/…`, `pi-home/…`), pings the route
with one tool-free agent call — which is where a model the account cannot use
fails, in seconds instead of mid-search — and pins `backend`/`backend_auth` in
`config.yaml`. It leaves a config that already pins a backend alone unless you
pass `--default`. `--no-probe` skips the ping, `--auth api-key|openrouter`
picks a different bill, `--user` writes the defaults to
`~/.config/hillclimb/config.yaml`.

`hillclimb connect openrouter` is the one credential hillclimb stores itself:
the key is validated against OpenRouter (one unbilled call) and written to the
`.env` beside `config.yaml` that `hillclimb init` gitignores — never into
`config.yaml`, where it could be journaled. `--backend codex --model
qwen/qwen3-coder` pins the route in the same command.

`hillclimb smoke` is the next step up: a whole DRAFT on a real problem.

## Cheap operators through OpenRouter

```yaml
backend: codex
backend_auth: openrouter
model: qwen/qwen3-coder          # any OpenRouter model id
```

`OPENROUTER_API_KEY` comes from the environment or a `.env` beside
`config.yaml`. Routing mixes providers per operator, and a `models:` pool lets
the bandit learn which cheap model actually earns improvements (see
[operators-and-memory.md](operators-and-memory.md)):

```yaml
routing:
  draft:   {backend: claude-code, backend_auth: subscription, model: sonnet}
  improve: {models: [qwen/qwen3-coder, deepseek/deepseek-v3]}
  debug:   {model: cohere/north-mini-code:free}
```

Codex resends an identical ~12k-token preamble on every call, so prefer models
whose providers cache prompts: the journaled `cache_read_input_tokens` tells
you whether the discount is landing. Set `budget.max_cost_usd` — cheap per
token is not cheap per search, because a weaker model compensates with
volume: one measured DRAFT burned 3M tokens (~$0.53 at qwen3-coder prices)
and another spent its whole agent timeout without converging. Running out of
credits parks the search — top up, then `resume`.
To compare models head to head, give an experiment one arm per model
(`arm_overrides: {model: …}`); the chart and `experiment report` group on the
arm tags (see [experiments.md](experiments.md)).

## Sampling with pi

Install pi and select a provider-qualified model. Subscription mode copies
the credentials from pi's own `/login` (`~/.pi/agent/auth.json`); `api-key`
uses provider environment variables such as `ANTHROPIC_API_KEY`. OpenRouter
requires `OPENROUTER_API_KEY` in the environment or `.env` beside `config.yaml`:

```yaml
backend: pi
backend_auth: openrouter
model: openrouter/deepseek/deepseek-v3.2
routing:
  draft:   {sampling: {temperature: 1.0, top_p: 0.95}}
  improve: {sampling: {temperature: 0.2}}
```

| setting | meaning |
|---|---|
| `routing.<op>.sampling` | Numeric provider request parameters, e.g. `temperature`, `top_p`, `top_k`, `min_p`; only supported by pi |
| `routing.default.sampling` | Fallback for all operators; an operator's dict replaces it, and `{}` disables inherited sampling |
| `pi.models_file` | Optional pi `models.json` for custom providers, including vLLM and llama.cpp |

Sampling follows action → operator → default routing precedence and is
recorded on each candidate. Unsupported backend combinations fail config
validation. A short, tool-free preflight checks each distinct pi model and
sampling combination (including every model in a pool) before search work
starts. Provider rejection fails startup, including errors pi emits with
exit code 0. Preflight streams and accounting are under `pi-preflight/`;
their small provider cost is separate from candidate spend.

Pi runs with an isolated `PI_CODING_AGENT_DIR` under
`~/.cache/hillclimb/pi-home/<auth>/`, with discovery of personal extensions,
skills, prompt templates and context files disabled. Custom model files get
a content-hashed subdirectory so concurrent searches cannot overwrite each
other's providers. `PI_OFFLINE=1` disables startup updates and telemetry;
install/update pi explicitly to update its bundled catalogue. Tested with
pi 0.73.1. Its `update --models` flag is not supported.

Debug children fork the parent's pi session into their own candidate
directory. Raw pi events remain in `agent_stream.jsonl` and are visible in
`watch`, including token usage and pi's reported cost. For OpenRouter models
with zero reported cost, Hillclimb falls back to its pricing catalogue.

For a local OpenAI-compatible server, set `backend_auth: api-key`,
`model: vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct` and point `pi.models_file`
at a file like:

```json
{"providers": {"vllm": {"baseUrl": "http://localhost:8000/v1",
  "api": "openai-completions", "apiKey": "none",
  "models": [{"id": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "contextWindow": 131072}]}}}
```

Relative model-file paths resolve from the project root; with an explicit
config-file load they resolve beside that config. Provider/model support
determines which sampling fields are accepted. Use the preflight to check
compatibility; reasoning effort remains a separate follow-up.

`hillclimb/experiments/temperature.yaml` compares temperatures 0.2, 0.7 and
1.0 on circle-packing, with three repeats and three concurrent searches.
Run `hillclimb experiment run temperature --dry-run` to inspect the jobs;
`hillclimb experiment report temperature --json` reports the arm verdicts
after the experiment finishes.
