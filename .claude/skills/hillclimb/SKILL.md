---
name: hillclimb
description: Drive hillclimb searches — start, monitor, stop, prune, and resume auto-hillclimbing runs that spawn headless coding agents as operators.
---

# Driving hillclimb runs

`hillclimb` is a headless engine: it greedily hillclimbs a verifier-defined problem by spawning
headless Claude Code agents (draft → debug → improve → ensemble), executing each
candidate solution, and keeping the best submission in
`runs/<run-id>/searches/<search-id>/best/`. You (the interactive agent) are the
front door: start runs, watch them, and control them on the user's behalf. The
user may also have `hillclimb watch` (a live TUI) open in another terminal — it
reads the same on-disk state you do.

The hierarchy: a **Run** (one invocation) contains **Searches** (one engine
process per problem), each exploring a tree of **Candidates** (immutable code
artifacts), each executed as a **Trial**.

All commands: `uv run hillclimb <command>` from anywhere inside a workspace
(found by upward search for `hillclimb/config.yaml`). Search-addressing
commands accept `<run-id>/<search-id>`, a bare `<run-id>` (when the run has one
search), or `latest` (the default).

## Lifecycle

**Start** (long-running — always run in the background and poll):

```bash
uv run hillclimb run <problem> --name "Run name" --budget 2h
uv run hillclimb run problems/demo-suite.yaml --name "Demo"  # one search per suite problem
uv run hillclimb run emflow://gefcom2014:solar --budget 2h   # emflow problem (agents write Predictors)
uv run hillclimb run emflow://gefcom2014 --budget 2h         # virtual suite: all variants
uv run hillclimb run <problem> --backend dummy   # fast no-agent backend for testing
```

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

In the candidate view, press enter or click a candidate to open the bottom
detail panel. That panel follows the highlighted candidate and shows notes,
scores, lineage, trial output, and any agent stream. Press escape to close
the detail panel; press escape again to go back. Drag the divider or use `+` /
`-` to resize the detail panel.

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
uv run hillclimb stop <search>                    # graceful: finishes current operator, then parks
uv run hillclimb kill <search>                    # SIGTERM now; state is finalized, still resumable
uv run hillclimb prune <search> <candidate-id>    # cut a candidate + its whole subtree from the search
uv run hillclimb resume <search>                  # continue a parked/stopped/crashed search
```

Prune when a branch is clearly overfitting (val ≫ holdout), wasting budget, or the
user asks to cut it. Works both while the engine runs (queued, applied between
operators) and offline (applied immediately). The baseline candidate (`c000`)
cannot be pruned. If the engine is running, never prune "offline" by hand — the
CLI decides queue-vs-direct itself.

**Finish:**

```bash
uv run hillclimb tree <search>     # render the exploration tree to <search>/tree.png
```

## Rules

- **Never edit `journal.jsonl`, `status.json`, or `control/` by hand.**
  The engine is the single writer of search state; use the CLI commands, which
  route through the control queue when the engine is live.
- Don't start a `resume` while also issuing an offline `prune` for the same
  search — narrow race, the CLI's running-check can't see an engine that is
  still starting.
- A `crashed` state is a heuristic (dead pid or stale heartbeat); `resume` is
  always safe — the journal is append-only and replay-consistent.
- Agent operator calls bill the user's Claude subscription; keep budgets modest
  unless the user says otherwise.
