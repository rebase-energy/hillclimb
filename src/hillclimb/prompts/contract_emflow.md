# Output contract (mandatory)

Work only inside the current working directory. Before you finish, these two files MUST exist here:

1. `solution.py` — a Python module (NOT a script) defining an emflow Predictor and exposing:

   ```python
   def get_model():
       return MyModel()   # a FRESH, UNTRAINED emflow Predictor
   ```

   The orchestrator evaluates it for you: it fits the model on the official training
   view and scores it on the **validation** split of `{{emflow_problem}}`, producing
   the official `val_score`. Do NOT print your own `val_score:` line, do NOT write
   `submission.csv`, and do NOT run the full evaluation yourself.
2. `notes.md` — first line: one sentence summarizing the approach (or the change you made); a short explanation may follow.

## The Predictor API

```python
import pandas as pd
from emflow.models.predictor import Predictor, FeaturePredictor

class MyModel(Predictor):
    {{quantile_note}}

    def fit(self, train):
        # `train` is a leak-proof TimeView frozen at the training cutoff:
        #   train.history("<field>", window="120D") -> DataFrame (past actuals)
        #   train.forecasts("<field>")              -> DataFrame (issued forecasts)
        return self

    def predict(self, obs):
        # `obs` is an Observation for ONE forecast origin:
        #   obs.history("<field>")  — everything knowable at the origin
        #   obs.forecasts("<field>") — forecast fields available at the origin
        #   obs.target_index        — the timestamps you must predict
        #   obs.column              — target column/zone for multi-zone problems
        # Return a DataFrame indexed by obs.target_index with the required columns.
        ...
```

- **Strongly prefer `FeaturePredictor`** (declare `features` specs, implement
  `predict_tabular(self, X)`): it enables vectorized evaluation — orders of magnitude
  faster than per-origin calls. Plain `Predictor.predict` is evaluated origin by origin.
- Explore the problem freely during development (this is sanctioned EDA):
  ```python
  import emflow as ef
  problem = ef.load_problem("{{emflow_problem}}")
  feed_data = problem.load_dataset()   # already cached locally
  ```
- NEVER evaluate on or fit against the holdout split. The orchestrator scores holdout
  separately; touching it invalidates the search.

Rules:
- Do NOT run long training yourself. Quick sanity checks (imports, instantiating your
  model, predicting one origin) are fine; the orchestrator runs the real evaluation
  after you finish.
- The full fit + validation evaluation must finish within {{exec_timeout_min}} minutes.
- Available packages: {{runtime_pkgs}}. Nothing else is installed. {{network_note}}
- Set random seeds for reproducibility.
{{tools_clause}}
- Total hillclimb time remaining for this problem: {{time_remaining}}.
