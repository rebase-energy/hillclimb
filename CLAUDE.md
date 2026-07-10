# hillclimb

Auto-hillclimbing for ML tasks: a greedy search engine that spawns headless
Claude Code agents as operators (draft/debug/improve/ensemble), executes each
candidate, and keeps the best submission per search.

Hierarchy: **Run** (one invocation, `runs/<run-id>/`) → **Search** (one engine
process per problem, `searches/<search-id>/`) → **Candidate** (immutable code
artifact) → **Trial** (one execution).

- Workspaces: all data lives in a `hillclimb/` folder (config.yaml marker,
  problems/, specs/, runs/) found by upward search; this repo overrides
  runs/problems to its legacy top-level dirs in `hillclimb/config.yaml`.
  Machine-scoped state (shared venvs, emflow cache, agent slots) lives in
  `~/.cache/hillclimb/`.
- Tests: `uv run pytest`
- CLI: `uv run hillclimb --help` (engine); `uv run hillclimb watch` (live TUI)
- Driving runs from chat: use the `hillclimb` skill (`.claude/skills/hillclimb/SKILL.md`)

## Run-state rules

The engine process is the **single writer** of search state
(`runs/<run-id>/searches/<search-id>/journal.jsonl`, `status.json`, `best/`).
Never edit those files directly — control a search through
`hillclimb stop|kill|prune|resume`, which route through the search's `control/`
command queue when the engine is live. `journal.jsonl` is append-only; replay
keeps the last record per candidate.

`hillclimb/knowledge/graph.json` is a **derived index** (gitignored) rebuilt
deterministically from the knowledge YAML (cards, entities.yaml,
concepts.yaml, credit/, consolidated.yaml) — never hand-edit it;
`hillclimb knowledge rebuild` regenerates it. The YAML files are the source
of truth and are git-versioned; `knowledge/credit/` holds append-only
per-search outcome events (one file per search — never merge or rewrite
them). `knowledge/playbooks/`, `knowledge/consolidated.yaml`, and
`knowledge/skills/` are consolidation/harvest outputs — regenerate via
`hillclimb knowledge consolidate` rather than hand-editing (playbook edits
are legitimate but land as reviewed git diffs).
