
## Tunable parameters (optional)

If your climber has numeric knobs (how many drafts before improving, how deep a debug chain may go, the reserve for an ensemble window, a temperature), declare them in `params.json` next to `solution.py`. The orchestrator then runs extra trials of THIS exact file with other values — each trial a full verifier run, so declare only what matters — and keeps the best.

```json
{"num_drafts":      {"type": "int", "low": 1, "high": 6, "default": 3},
 "max_debug_depth": {"type": "int", "low": 0, "high": 5, "default": 3}}
```

A trial's values reach the inner searches as the climber's `params` (`hillclimb run --set climber.params.<name>=<value>`): a knob the selector policy declares in its `DEFAULTS` reaches the selector policy, every other one the operator policy, so a policy must read its knobs with `self.param("num_drafts")` from the class's `DEFAULTS`, never from a file. Every parameter needs a `default` equal to the `DEFAULTS` value in the file now; 2–4 parameters. A malformed declaration is still scored on your defaults but never tuned.
{{inherited}}
