# Output contract (mandatory)

Work only inside the current working directory. Before you finish, these two files MUST exist here:

1. `solution.py` — a single self-contained Python script that:
   - reads the problem files from `./problem/` and any runtime inputs from `./data/` (treat both as read-only)
   - solves the problem without modifying the problem files
   {{verifier_clause}}
   - writes `./submission.csv` matching `./problem/sample_submission.csv` exactly: same columns, same id values, same value dtypes
2. `notes.md` — first line: one sentence summarizing the approach (or the change you made); a short explanation may follow.
{{holdout_clause}}
{{report_clause}}

Rules:
- Do NOT run long training yourself. Quick sanity checks (imports, loading a few rows, a 1-minute dry run) are fine; the orchestrator executes `solution.py` for real after you finish.
- When executed by the orchestrator (`python solution.py`, cwd = this directory), the script must finish within {{exec_timeout_min}} minutes.
- Available packages: {{runtime_pkgs}}. Nothing else is installed. {{network_note}}
- Set random seeds for reproducibility.
{{tools_clause}}
- Total hillclimb time remaining for this problem: {{time_remaining}}.
