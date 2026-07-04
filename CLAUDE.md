# rebase-hillclimb

Auto-hillclimbing for ML tasks: a greedy search engine that spawns headless
Claude Code agents as operators (draft/debug/improve/ensemble), executes each
candidate, and keeps the best submission per run.

- Tests: `uv run pytest`
- CLI: `uv run hillclimb --help` (engine); `uv run hillclimb watch` (live TUI)
- Driving runs from chat: use the `hillclimb` skill (`.claude/skills/hillclimb/SKILL.md`)

## Run-state rules

The engine process is the **single writer** of run state (`runs/<id>/journal.jsonl`,
`status.json`, `best/`). Never edit those files directly — control a run through
`hillclimb stop|kill|prune|resume`, which route through the `runs/<id>/control/`
command queue when the engine is live. `journal.jsonl` is append-only; replay
keeps the last record per node.
