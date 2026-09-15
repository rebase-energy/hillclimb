
## Tunable parameters (optional)

If your solution has numeric knobs (restart counts, iteration budgets, step sizes, temperatures, a choice of initializer), declare them in `params.json` next to `solution.py` and read them at runtime. The orchestrator then runs extra trials of THIS exact code with other values and keeps the best — search time you do not have to spend by hand.

```json
{"restarts": {"type": "int", "low": 1, "high": 64, "log": true, "default": 8},
 "step":     {"type": "float", "low": 1e-4, "high": 0.1, "log": true, "default": 0.01},
 "init":     {"type": "categorical", "choices": ["grid", "random"], "default": "grid"}}
```

In `solution.py`:

```python
from hillclimb import spaces   # importable when the verifier runs{{shim_note}}
P = spaces.params()            # {"restarts": 8, "step": 0.01, "init": "grid"}: the trial's values, else your defaults
```

Rules: every parameter needs a `default` equal to what your code uses now; 2–6 parameters; every value in the declared ranges must finish within the time limit; read values only through `spaces.params()` (it follows `$HILLCLIMB_PARAMS` to the trial's copy). A malformed declaration is still scored on your defaults but never tuned.
{{inherited}}
