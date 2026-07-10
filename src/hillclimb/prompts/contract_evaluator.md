# Output contract (mandatory)

Work only inside the current working directory. Before you finish, these two files MUST exist here:

1. `solution.py` — your solution, in exactly the shape the problem-specific contract below requires. The orchestrator scores it by running the problem's own evaluator in this directory:

   ```
   {{eval_command_display}}
   ```

   The evaluator's final `val_score: <float>` line is your official validation {{metric_name}}. Do NOT print a `val_score:` line yourself and do NOT write `eval_result.json` — the evaluator owns both.
2. `notes.md` — first line: one sentence summarizing the approach (or the change you made); a short explanation may follow.

## Problem-specific solution contract

{{problem_contract}}

Rules:
- You may run the evaluator yourself during development as a sanity check, but do NOT run long training; the orchestrator runs the real evaluation after you finish.
- The evaluator run (including everything it does with `solution.py`) must finish within {{exec_timeout_min}} minutes.
- Available packages: {{runtime_pkgs}}. Nothing else is installed. {{network_note}}
- Set random seeds for reproducibility (the orchestrator sets `HILLCLIMB_TRIAL_SEED` when running trials).
{{tools_clause}}
- Total hillclimb time remaining for this problem: {{time_remaining}}.
