# pi operators with a sampling knob

## Implementation notes (2026-09-16)

Implemented with agent, routing/journal sampling, custom providers,
preflight, raw-stream viewer support, tests and the temperature experiment
spec. The reasoning-effort sibling remains a follow-up. Tests against the
installed 0.73.1 CLI and a loopback OpenAI-compatible provider confirmed
payload injection, tool execution, session history and exit-0 error parsing.
Three corrections to the original proposal below:

- `--session` restores the original session's cwd. Debug children must use
  `--fork`, which retains history and creates a child-owned session/cwd.
- This installed CLI has no `update --models` command or
  `defaultProjectTrust` setting. Isolation uses explicit `--no-*` discovery
  flags; the installed catalogue stays pinned with `PI_OFFLINE=1`. Updating
  pi is an explicit setup step, never automatic during a search.
- Different custom `models.json` contents receive separate hashed config
  dirs beneath `<auth>/`, preventing concurrent searches from replacing
  one another's providers.

The temperature seed verifier was measured five times: score 2.158 on each,
spread 0, MAD 0. That measured floor is recorded in the experiment spec.
The paid three-arm experiment still requires an OpenRouter key; no arm
verdict is claimed by the implementation or the local-provider tests.

## Original proposal

Run hillclimb operators through the pi coding agent
(`@earendil-works/pi-coding-agent`, installed 2026-09-16 as pi 0.73.1) so a
search can set the LLM's sampling parameters — `temperature`, `top_p`,
`top_k`, `min_p` — per operator. Neither `claude-code` nor `codex` can: Claude
Code exposes no sampling flag and Claude 4.7+ rejects one anyway; the Codex
CLI's Responses request struct has no such field and no body hook. pi's
request builder does, and its extension hook lets us set it from outside.

## Why

Exploration vs exploitation in the operator layer today is prompt pressure
("take a meaningfully different approach"), model pools (UCB1 over
`routing.<op>.models`) and reasoning effort. None of them is a *sampling*
dial. Temperature is the classic one, and it is the one that maps most
directly onto "hot drafts, cold improves". It only exists on the open-weight
tier (DeepSeek, Kimi, GLM, Qwen; anything on a local vLLM or llama.cpp
server), which is also where sampling studies are cheap enough to repeat.

## What the spike established

Measured 2026-09-16 with pi 0.73.1, an isolated `PI_CODING_AGENT_DIR`, a
ten-line extension loaded with `-e`, and `ANTHROPIC_API_KEY` auth. Total
spend: under $0.01.

| question | result |
|---|---|
| Can an extension inject sampling params? | Yes. A `before_provider_request` handler that returns `{...payload, temperature: 0.9}` changed the wire request; Haiku 4.5 answered with `temperature=0.9` visible in the payload log. |
| Does it work headless? | Yes. `pi -p --mode json` reads the prompt from stdin, streams JSON lines, exits 0. |
| Is there a session id for debug chains? | Yes. The first JSON line is `{"type":"session","id":"<uuid>",…}`; `--session <uuid> --session-dir <dir>` resumed the same session headlessly and the model remembered the earlier turn. |
| Where do usage and cost come from? | Every assistant `message_end` carries `usage: {input, output, cacheRead, cacheWrite, totalTokens, cost: {…, total}}` and `model` + `provider`. Cost is pi's own catalogue price. |
| What does a model that refuses temperature do? | Sonnet 5 returned Anthropic's 400 `"temperature is deprecated for this model"`. pi reported it as an assistant message with `stopReason: "error"` and **still exited 0**. The agent must read `stopReason`, never the exit code. |
| Does personal config leak in? | Not with `PI_CODING_AGENT_DIR` pointed at an isolated folder plus `--no-extensions --no-skills --no-prompt-templates --no-context-files`. The extension is still loaded via explicit `-e`. |
| Catalogue freshness | The bundled catalogue did not know `claude-sonnet-5` (ran as a custom id with a warning). `pi update --models` refreshes it; `PI_OFFLINE=1` pins it. |

Two consequences drive the design: the sampling knob is an env var read by
a shipped extension, and provider errors are detected from the event stream.

## Design

### 1. A `pi` agent next to `claude-code` and `codex`

`agents/pi_cli.py`, registered in `agents/__init__.py` as `"pi"`. One
operator call is one process:

```
pi -p --mode json
   --no-extensions -e <package-data>/hillclimb-sampling.ts
   --no-skills --no-prompt-templates --no-context-files
   --tools read,write,edit,bash,grep,find,ls
   --session-dir <search_dir>/pi-sessions [--session <uuid>]
   --model <model> [--thinking <level>]
```

- prompt on stdin, cwd = `candidate_dir`, `start_new_session=True` and the
  codex agent's `_kill_group` for timeouts/abort;
- raw stream to `<candidate_dir>/agent_stream.jsonl`, pid to `agent.pid`,
  stderr to `agent_stderr.log` (same files as the other agents, so `watch`
  needs nothing new);
- `--session-dir` is per search so a debug child can resume its parent's
  uuid from its own candidate dir (`search_dir = candidate_dir.parents[1]`,
  the `searches/<id>/candidates/<cid>/` layout; fall back to `candidate_dir`
  when the layout does not match, e.g. in tests);
- env: `single_threaded(os.environ)` plus the auth handling below.

`_PiStreamReader` normalizes the JSON lines into what `OperatorResult`
wants:

| field | source |
|---|---|
| `session_id` | `{"type":"session","id":…}` header |
| `model_id` | first assistant `message_end.message.model` (pi's id; prefix `provider/` when it is not the requested alias) |
| `token_usage` | sum over assistant `message_end.usage`: `input→input_tokens`, `output→output_tokens`, `cacheWrite→cache_creation_input_tokens`, `cacheRead→cache_read_input_tokens` |
| `cost_usd` | sum of `usage.cost.total`; when pi reports 0 for an OpenRouter model, fall back to `pricing.cost_usd` like the codex agent |
| `num_turns` | count of `turn_end` |
| `ok` / `error_kind` | last assistant `stopReason`: `"stop"`/`"toolUse"` ok; `"error"` → classify `errorMessage`: `RATE_LIMIT_MARKERS` → `rate_limited`, `402`/insufficient credits → `out_of_credits`, else `error`; killed → `timeout`/`aborted` |

### 2. Auth modes

`agent_auth` keeps its three values; the validator in `config.py` that
pins `openrouter` to `codex` widens to `codex | pi`.

- `subscription`: pi's `/login` writes `~/.pi/agent/auth.json`; copy it into
  the isolated dir exactly like `codex_home()` does (mtime-gated, atomic
  replace). Anthropic OAuth and ChatGPT-Codex logins both ride this.
- `api-key`: keep `ANTHROPIC_API_KEY` (pi reads it natively).
- `openrouter`: `OPENROUTER_API_KEY` from the env or the `.env` beside
  config.yaml, model ids `openrouter/<vendor>/<model>`. Missing key raises
  before the spawn, as in `codex_env`.

Isolation dir: `~/.cache/hillclimb/pi-home/<auth>/` with a generated
`settings.json` (`defaultProjectTrust: never`, `quietStartup: true`) and an
optional `models.json` (see §4). `PI_CODING_AGENT_DIR` points at it.

### 3. The sampling knob

One new optional field, threaded through the same layers `model` already
uses:

```yaml
routing:
  draft:   {agent: pi, agent_auth: openrouter,
            model: openrouter/deepseek/deepseek-v3.2,
            sampling: {temperature: 1.0, top_p: 0.95}}
  improve: {agent: pi, agent_auth: openrouter,
            model: openrouter/deepseek/deepseek-v3.2,
            sampling: {temperature: 0.2}}
```

- `RouteConfig.sampling: dict[str, float] | None` (config), `Route.sampling`
  (policy override), `ResolvedRoute.sampling` (routing.py `pick`),
  `OperatorRequest.sampling` (agent), `AgentInfo.sampling` (journal —
  charts and experiment reports group on it).
- The pi agent serializes it as `HILLCLIMB_SAMPLING='{"temperature":…}'`
  in the child env. The shipped extension
  (`src/hillclimb/agents/pi_ext/hillclimb-sampling.ts`, package data,
  the file used in the spike) merges the object into every provider payload
  and logs one line per request to stderr. Empty/unset = no-op.
- The config validator rejects `sampling` on `claude-code` and `codex`
  routes with the same shape of error as the `openrouter` check, so a knob
  that would be silently ignored fails at load instead.
- Model gating stays the operator's responsibility: Claude 4.7+ and OpenAI
  reasoning models 400 on any temperature. The agent turns that 400 into
  `error_kind="error"` with the provider message, and a **preflight** at
  search start (`pi -p --no-tools` "reply pong" with the route's sampling,
  ~500 tokens, one per distinct (model, sampling) route) fails the search
  before a draft burns its budget. Reuse the reader; it is the same stream.

`sampling` is a free-form dict on purpose: `min_p`, `repetition_penalty`
and friends go straight through to vLLM/llama.cpp, and seeds stay out of it
(seeds are never tuned).

### 4. Local servers (vLLM, llama.cpp)

pi's `models.json` declares custom OpenAI-compatible providers. Add
`pi.models_file: <path>` to `Config` (default: none); when set, the agent
copies it into the isolated dir as `models.json` at construction. A vLLM
entry is then ordinary config:

```json
{"providers": {"vllm": {"baseUrl": "http://localhost:8000/v1",
  "api": "openai-completions", "apiKey": "none",
  "models": [{"id": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "contextWindow": 131072}]}}}
```

and `model: vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct` routes to it. The
sampling extension covers per-operator values; a static
`samplingParams` in `models.json` remains available for per-model defaults
and is overridden per key by the extension.

### 5. Reasoning effort as the sibling knob (optional, same PR or next)

`--thinking off|minimal|low|medium|high|xhigh` is pi's effort flag. Adding
`RouteConfig.effort` alongside `sampling` and mapping it to `--thinking`
(pi), `--effort` (claude-code) and `-c model_reasoning_effort=` (codex)
gives the frontier models their exploration dial through the same routing
layer. Keep it a separate commit so the temperature study is not gated on
it.

## Steps

1. `agents/pi_cli.py` + `pi_ext/hillclimb-sampling.ts` (package data in
   `pyproject.toml`), `_PiStreamReader`, `pi_home(auth)`, `pi_env(auth)`,
   registration in `agents/__init__.py`.
2. `tests/test_pi_agent.py` on the codex-agent pattern (stub `pi`
   script that asserts stdin/env/argv and prints a session header, an
   assistant `message_end` with usage, a `turn_end`): session id, usage
   mapping, cost fallback, `stopReason: error` → not ok on exit 0,
   rate-limit and 402 classification, `HILLCLIMB_SAMPLING` present only
   when the route carries `sampling`, `--session` on resume, isolated
   `PI_CODING_AGENT_DIR`, auth copy.
3. `sampling` field through `config.py` (+ validator), `policy.Route`,
   `routing.py`, `OperatorRequest`, `search.py` (request + `AgentInfo`),
   `candidate.py`; tests in `test_config`/`test_policy`/`test_search`.
4. `Config.pi.models_file` and the isolated-dir `settings.json`/`models.json`
   writer; a test that the file lands where `PI_CODING_AGENT_DIR` points.
5. Preflight in `api.execute_search` (behind the pi agent only), logged
   like the emflow venv warm-up; a test through the stub binary.
6. `hillclimb/experiments/temperature.yaml`: control `t0.2` vs `t0.7` vs
   `t1.0` on `openrouter/deepseek/deepseek-v3.2` (or `qwen/qwen3-coder`),
   `schedule: parallel`, `max_concurrent: 3`, problem `circle-packing`
   (fast verifier, `instances` breakdown), `noise_floor` from
   `hillclimb verify --repeat 5`. The report's `verdict` per arm is the
   deliverable.
7. Docs: CLAUDE.md agent list (`pi`, `sampling`, `pi.models_file`),
   README config table, `hillclimb` skill note.

## Open questions

- **Context files.** `--no-context-files` keeps pi from reading a
  candidate's `CLAUDE.md`/`AGENTS.md`. The prompt already carries the
  contract, so parity with the other agents argues for off; flip it if a
  problem starts shipping an `AGENTS.md` on purpose.
- **Session dir location.** Per search keeps resume simple; per candidate
  would keep everything that belongs to a candidate inside its dir (the
  vocabulary rule) at the cost of passing the parent's session *path* on
  resume. Per search is proposed; both are one line.
- **Catalogue drift.** pi refreshes its model catalogue on startup unless
  `PI_OFFLINE=1`. Pinning it makes a search reproducible but ages the
  `supportsTemperature` flags; refreshing once in `pi_home()` and running
  offline afterwards is the proposed middle.
