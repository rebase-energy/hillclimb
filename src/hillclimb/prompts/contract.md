# Output contract (mandatory)

Work only inside the current working directory. Before you finish, these two files MUST exist here:

1. `solution.py` — a single self-contained Python script that:
   - reads the competition data from `./data/` (treat it as read-only)
   - trains a model and evaluates it on an internal validation split carved from the training data (test labels do not exist for you — never fabricate them)
   - prints exactly one line `val_score: <float>` (your validation {{metric_name}}) as the FINAL line of stdout
   - writes `./submission.csv` matching `./data/sample_submission.csv` exactly: same columns, same id values, same value dtypes
2. `notes.md` — first line: one sentence summarizing the approach (or the change you made); a short explanation may follow.

Rules:
- Do NOT run long training yourself. Quick sanity checks (imports, loading a few rows, a 1-minute dry run) are fine; the orchestrator executes `solution.py` for real after you finish.
- When executed by the orchestrator (`python solution.py`, cwd = this directory), the script must finish within {{exec_timeout_min}} minutes.
- Available packages: {{runtime_pkgs}}. Nothing else is installed; assume no internet access at execution time.
- Set random seeds for reproducibility.
- Total hillclimb time remaining for this competition: {{time_remaining}}.
