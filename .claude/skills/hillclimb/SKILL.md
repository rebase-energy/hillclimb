---
name: hillclimb
description: Drive hillclimb searches — start, monitor, stop, prune, and resume auto-hillclimbing runs that spawn headless coding agents as operators.
---

# Driving hillclimb runs

`hillclimb` is a headless engine: it greedily hillclimbs a verifier-defined problem by spawning
headless coding agents (Claude Code, Codex or pi; draft → debug → improve → ensemble), executing each
candidate solution, and keeping the best submission in
`runs/<run-id>/searches/<search-id>/best/`. You (the interactive agent) are the
front door: start runs, watch them, and control them on the user's behalf. The
user may also have `hillclimb watch` (a live TUI) open in another terminal — it
reads the same on-disk state you do.

The hierarchy: a **Run** (one invocation) contains **Searches** (one engine
process on one problem), each exploring a tree of **Candidates** (immutable code
artifacts), each evaluated as one or more **Trials** (parameter sets) of seeded
**Replicates**. A candidate that declares `params.json` gets extra trials from
the search's tuner (`cNNN(tune/exec/tK)` in `watch`; `hillclimb show` lists
them with their params).

All commands: `uv run hillclimb <command>` from anywhere inside a hillclimb
dir (found by upward search for `hillclimb.yaml`). Search-addressing
commands accept `<run-id>/<search-id>`, a bare `<run-id>` (when the run has one
search), or `latest` (the default).

## Lifecycle

**Start** (long-running — always run in the background and poll):

```bash
uv run hillclimb run <problem> --name "Run name" --budget 2h
uv run hillclimb run problems/demo-suite.yaml --name "Demo"  # one search per suite problem
uv run hillclimb run emflow://gefcom2014:solar --budget 2h   # emflow problem (agents write Predictors)
uv run hillclimb run emflow://gefcom2014 --budget 2h         # virtual suite: all variants
uv run hillclimb run mlebench://spaceship-titanic --budget 2h  # MLE-bench comp (graded once, post-search)
uv run hillclimb run mlebench://lite --budget 4h             # virtual suite: MLE-bench Lite (22 comps)
uv run hillclimb run <problem> --backend dummy   # fast no-agent backend for testing
```

Before the first real run on a machine, `uv run hillclimb connect --json` says
which backends have a working credential (checked through the same environment
an operator gets) and which one is the default; `hillclimb connect <claude|
codex|pi|openrouter>` sets one up. A dead credential there is why a search
would otherwise fail on its first operator call.

Exit codes: `0` done, `2` parked or stopped (resumable). Rate limits park the
search automatically.

**Monitor** — two equivalent sources, poll every 30-60s while a search is live:

```bash
uv run hillclimb status <search>                       # state line + candidate tree
cat runs/<run-id>/searches/<search-id>/status.json     # machine-readable
```

`status.json` fields worth reading: `state`, `budget.remaining_s`, `candidates`
counts, `current` (the candidate being worked on right now, with
`phase: agent|exec` and `agent_pid`), `best`, `selected`, `last_error`.

The TUI hierarchy is: runs → searches → candidates. Effective search
states (derived, shown by `status` and the TUI):

In the searches view, press enter on a search to open its candidates in a
panel under the table and move the cursor into it (the panel follows the
highlighted search); enter on a candidate there gives the candidates the whole
screen with that candidate's details underneath; `o` opens the full candidate
view directly; `m` maximizes/restores the lower panel on either screen. In the candidate view, press enter or
click a candidate to open the bottom detail panel. That panel follows the highlighted candidate and shows notes,
scores, lineage, trial output, and the operator stream. Press escape to close
the panel; press escape again to go back. Drag the divider or use `+` /
`-` to resize either panel. Footers carry only enter / esc / `?` / q; `?`
slides out a panel from the right listing every key and gesture of the
current screen, and ctrl+c quits any of the TUIs.

| state | meaning |
|---|---|
| running | engine alive, heartbeat fresh |
| parked | hit a rate limit or repeated agent failures → `resume` later |
| stopped | user stop/kill → `resume` continues where it left off |
| done | budget spent or max candidates reached |
| failed | engine crashed with an exception (see `last_error`) |
| crashed | status says running but the pid is dead / heartbeat stale → `resume` |
| unknown | no status.json yet |

To see what the in-flight agent is doing live:
`tail -f runs/<run-id>/searches/<search-id>/candidates/<candidate-id>/agent_stream.jsonl`
(the `current` candidate from status.json).

**Control:**

```bash
uv run hillclimb stop <search>                    # now: aborts in-flight operators (journaled abandoned), parks; resumable
uv run hillclimb stop --graceful <search>         # lets in-flight operators finish and be scored, then parks
uv run hillclimb kill <search>                    # last resort for an unresponsive engine: SIGTERM, then SIGKILL
uv run hillclimb prune <search> <candidate-id>    # cut a candidate + its whole subtree from the search
uv run hillclimb resume <search>                  # continue a parked/stopped/crashed search
uv run hillclimb resume --all                     # resume everything resumable, each detached
```

Pause/resume flow for changing code or env under a live project:
`stop --all` → make the change → `resume --all` (each search restarts as a
fresh detached engine, so it picks up new code and environment variables).

Prune when a branch is clearly overfitting (val ≫ holdout), wasting budget, or the
user asks to cut it. Works both while the engine runs (queued, applied between
operators) and offline (applied immediately). The baseline candidate (`c000`)
cannot be pruned. If the engine is running, never prune "offline" by hand — the
CLI decides queue-vs-direct itself.

**Finish:**

```bash
uv run hillclimb tree <search>     # render the exploration tree to <search>/tree.png
```

## Cross-search memory

Finished searches feed a file-based memory under `knowledge/`:
statistical cards plus LLM-distilled claims (entities classified into a
concept ontology), all folded into a derived temporal graph index.

```bash
uv run hillclimb knowledge graph --stats        # text summary of the graph
uv run hillclimb knowledge graph                # interactive TUI (zoom/pan/click/scrub) — don't run headless (alias: hillclimb graph)
uv run hillclimb watch candidates [search]      # TUI straight on a search's candidates — don't run headless
uv run hillclimb stop --all                     # stop every running search (e.g. a demo)
uv run hillclimb reset --yes                  # kill this folder's engines AND delete what hillclimb made here (hillclimb.yaml, problems/, runs/, …; other files and folders untouched)
uv run hillclimb chart                          # live hillclimb curve TUI (best score vs time; several charts = a picker table first, enter/esc) — don't run headless
uv run hillclimb chart --detail [search]        # same, one search with its exploration tree on the curve — don't run headless
uv run hillclimb tree [search]                  # exploration tree TUI (expanded / discontinued / failed lineages) — don't run headless
uv run hillclimb knowledge rebuild              # regenerate the derived graph.json
uv run hillclimb knowledge distill [search]     # claims pass for one search (--backfill: all cards)
uv run hillclimb knowledge consolidate          # sleep phase: generalize claims + rewrite playbooks (agent calls)
uv run hillclimb knowledge query "<terms>"      # read-only memory lookup (no agent calls)
uv run hillclimb knowledge show <target>        # prior-experience block a new search would get
uv run hillclimb paper add <pdf> --problem <t>  # distill a PDF paper into claims (before a run: inspect wiring with `hillclimb graph`)
uv run hillclimb paper list                     # ingested papers with scope and claim counts
uv run hillclimb experiment run <spec> [--dry-run] [--parallel]  # a study: experiments × problems × repeats (real searches; --dry-run lists jobs)
uv run hillclimb experiment report [spec]       # compare the experiments on holdout, gap vs control judged against the noise floor (--json: gaps + verdicts as data)
uv run hillclimb climber list                     # presets (greedy | openevolve | gepa), one-file climbers under climbers/, and the registered building blocks per slot
uv run hillclimb climber show [NAME]              # a climber as the `climber:` block a run config takes (a preset, a .py file, this folder's; a pre-0.6 climber dir comes out as its block)
uv run hillclimb climber new mine --from greedy   # copy greedy's source into climbers/mine.py and print the block that runs it
uv run hillclimb climber check [SPEC.yaml] [--climber NAME] [--set climber.params.k=v] [--problem P --smoke]  # resolve every module, then replay recorded journals through the policy (no agent): resume-determinism, dangling ids, writes, prompt lint; a spec checks every entry's climber; exit 1 on a breach
uv run hillclimb run <problem> --climber climbers/mine.py  # a preset's name or one .py file (a Policy class, or POLICY=...); replaces the folder's `climber:` block. search.yaml records climber_sha256 and the block, and snapshots it
uv run hillclimb run run.yaml                     # a run spec: each entry's `climber:` block DEFINES that search's climber (policy/loop, select, operators, tuner, memory, params); a top-level `climber:` is the entries' default
uv run hillclimb run <problem> --set climber=openevolve --set climber.selector_params.num_islands=3 --study S --experiment E  # one experiment by hand (counts in the report); `climber=` names the block, `climber.<field>` edits it
uv run hillclimb run <problem> --climber greedy --climber openevolve --climber gepa --experiment-set gepa:concurrency.parallel_agents=1  # mixed fleet: one search per climber under one run; `experiment report <run-id>` compares
```

## Rules

- Pi routes support `routing.<op>.sampling: {temperature: 0.7}` and
  `pi.models_file` for local providers. OpenRouter uses
  `backend: pi`, `backend_auth: openrouter`, a provider-qualified model
  such as `openrouter/deepseek/deepseek-v3.2`, and `OPENROUTER_API_KEY` in
  the environment or `.env` beside hillclimb.yaml. Search startup runs cheap
  no-tools preflights; a failed preflight means fix that model/sampling
  combination before retrying. Pi errors can exit 0: use Hillclimb's parsed
  status and `agent_stream.jsonl`. The `temperature` experiment spec compares
  three temperatures; inspect with `experiment run temperature --dry-run`,
  then use `experiment report temperature --json` for the verdicts.

- Agents and verifier runs are sandboxed by default (`docs/sandbox.md`):
  they write only to their candidate's folder. `Not started: the sandbox …`
  means it cannot start here; the message names the fix. When this session's
  own shell is sandboxed, macOS refuses the inner one: start the run from an
  unsandboxed shell. Never set `sandbox: off` for the user without asking.

- **Never edit `journal.jsonl`, `status.json`, or `control/` by hand.**
  With `store.backend: sqlite` those records live in `store.sqlite`
  instead of the search dir — use `hillclimb status` / `store searches` rather
  than reading files.
  The engine is the single writer of search state; use the CLI commands, which
  route through the control queue when the engine is live.
- Don't start a `resume` while also issuing an offline `prune` for the same
  search — narrow race, the CLI's running-check can't see an engine that is
  still starting.
- A `crashed` state is a heuristic (dead pid or stale heartbeat); `resume` is
  always safe — the journal is append-only and replay-consistent.
- Agent operator calls bill the selected subscription or API provider; keep budgets modest
  unless the user says otherwise.
