---
name: hillclimb-terminology-refactor
description: Use when renaming a concept in hillclimb docs and website.
category: documentation
---

When renaming a concept in hillclimb, search and update these locations together, not piecemeal.

**Search scope (in order):**
1. `hillclimb/src/hillclimb/cli/` — terminal output strings (say/warn/fail messages)
2. `hillclimb/CLAUDE.md` — project instructions carry example terminology
3. `hillclimb/docs/` — local documentation (commands.md, problems.md, etc.)
4. `hillclimb-web/docs/content/docs/` — **sibling repo** with website docs; same terminology must apply
5. `hillclimb/experiments/*.yaml` — experiment specs carry example commands
6. `.claude/skills/hillclimb/SKILL.md` — skill examples must be current
7. `hillclimb/src/hillclimb/` (comments) — code comments and docstrings in harness/tui/cli

**Config key migrations:**
- Old keys remain as aliases in the config (e.g., `parallel_operators` → `parallel_agents`)
- Update all *uses* in example commands and docs to the new key
- Document the alias in CHANGELOG if the key is user-facing
- The promotion message ("the old spelling") helps users recognize they're using the old key

**Verification:**
- Run `rg` to confirm no instances remain with the old terminology
- Verify `uv run pytest` passes (no test data or goldens broke)
- Check the hillclimb-web folder exists and is up-to-date (it's a separate github.com/rebase-energy/hillclimb-web repo)
