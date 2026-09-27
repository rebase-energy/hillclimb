
## Tunable parameters (optional)

If your climber has numeric knobs (how many drafts before improving, how deep a debug chain may go, the reserve for an ensemble window, a temperature), declare them in `params.json` next to `solution.py`. The orchestrator then runs extra trials of THIS exact file with other values — each trial a full verifier run, so declare only what matters — and keeps the best.

```json
{"num_drafts":      {"type": "int", "low": 1, "high": 6, "default": 3},
 "max_debug_depth": {"type": "int", "low": 0, "high": 5, "default": 3}}
```

A trial's values reach the inner searches as the climber's `params` (`hillclimb run --set climber.params.<name>=<value>`), so your policy must read its knobs from the `params` mapping it is constructed with (`self.params.get("num_drafts", 3)`), never from a file. Every parameter needs a `default` equal to what the file uses now; 2–4 parameters. A malformed declaration is still scored on your defaults but never tuned.
{{inherited}}
