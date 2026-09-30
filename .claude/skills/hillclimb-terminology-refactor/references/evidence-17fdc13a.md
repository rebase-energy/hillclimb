rg -ni -e 'parallel_operators|machine_max_operators|operators? (in parallel|at once|concurrently)|max.?operators|parallel_operators|operator_slots|operators running' hillclimb hillclimb-web | head -40
Found instances across:
- hillclimb/experiments/*.yaml (config keys)
- hillclimb/src/hillclimb/cli/run.py (terminal output)
- hillclimb/CLAUDE.md, hillclimb/docs/*.md
- hillclimb-web/docs/content/docs/ (website docs, separate repo)
- .claude/skills/hillclimb/SKILL.md (example commands)