# Output contract (mandatory)

Work only inside the current working directory. Before you finish, these two files MUST exist here:

1. `solution.py` — a **one-file hillclimb climber**: the search process that decides what an inner search tries next. The orchestrator scores it by running the problem's verifier in this directory, which starts real inner hillclimb searches with your file as their climber and reports how far they climbed:

   ```
   {{verifier_display}}
   ```

   Whatever score that verifier reports is your official validation {{metric_name}}. Do NOT write `eval_result.json` yourself — the verifier owns the score, and anything you write there is discarded.
2. `notes.md` — first line: one sentence summarizing the approach (or the change you made); a short explanation may follow.

## What a one-file climber is

`solution.py` imports from `hillclimb.sdk` (and the standard library) and nothing else, and defines:

- exactly ONE search policy: a class with `propose(self, view: PolicyInput) -> Action | None` and `observe(self, view: PolicyInput, candidate: Candidate) -> None` (or `POLICY = <class or factory>`). `propose` returns the next `Action(operator=..., target_id=..., inspiration_ids=..., args=...)` or `None` to wait for in-flight results; it must be a pure function of `view` (the holdout-blind journal, in-flight refs, the budget, `higher_is_better`, `accept_band`) so a resumed search replays to the same decisions. `observe` is where a stateful policy rebuilds its caches; it is replayed for every existing candidate on start.
- optionally, `Operator` subclasses (`name`, `role` in `create | repair | refine | combine`, `prepare(self, ctx: OperatorContext) -> Preparation(prompt, copy_parent, inherit_params, copy_inspirations, files, texts, require_change)`). An operator you define replaces the built-in of the same name; the prompt it returns is what the inner agent reads, and `{{contract}}` in it is where the inner problem's contract goes (appended if you leave it out). The built-ins `draft`, `debug`, `improve` and `ensemble` stay available to `propose` untouched.

Memory, the tuner, the agent and the model are not yours to choose: the user's config sets them, the file is refused if it tries.

## Problem-specific solution contract

{{problem_contract}}
{{interface_section}}
{{params_section}}
{{holdout_clause}}

Rules:
- A verifier run is EXPENSIVE: it runs whole inner searches with real agents. Before you finish, run the cheap check — it loads the file, finds the policy, replays recorded searches through it and rejects imports outside `hillclimb.sdk`:

   ```
   {{engine_python}} -m hillclimb.cli meta check --climber solution.py
   ```

   Do not run the verifier yourself unless the check passes and you have a specific reason; the orchestrator runs the real evaluation after you finish.
- The verifier run must finish within {{exec_timeout_min}} minutes.
- {{network_note}}
- The policy must never write to disk, open the journal or the store directly, or read anything it is not handed through `view`; a policy that does is not replayable and the search will be refused.
{{tools_clause}}
- Total hillclimb time remaining for this problem: {{time_remaining}}.
