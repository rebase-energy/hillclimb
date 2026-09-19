from __future__ import annotations

from hillclimb.backends.base import OperatorRequest, OperatorResult

# Generic tabular solver used for quota-free end-to-end testing. It infers the
# id/target columns from sample_submission.csv, so it works on spaceship-titanic
# and on tiny synthetic fixture tasks alike.
SOLVER_TEMPLATE = '''\
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = 0
data = Path("data")
train = pd.read_csv(data / "train.csv")
test = pd.read_csv(data / "test.csv")
sample = pd.read_csv(data / "sample_submission.csv")

id_col, target_col = sample.columns[0], sample.columns[1]
y = train[target_col]
X = train.drop(columns=[target_col, id_col], errors="ignore")
X_test = test.drop(columns=[id_col], errors="ignore")[X.columns.intersection(test.columns)]
X = X[X_test.columns]

for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        cats = pd.Categorical(pd.concat([X[col], X_test[col]]).astype(str)).categories
        X[col] = pd.Categorical(X[col].astype(str), categories=cats).codes
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats).codes

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=rng)
model = HistGradientBoostingClassifier(max_iter={max_iter}, random_state=rng)
model.fit(X_tr, y_tr)
val_pred = model.predict(X_val)
score = accuracy_score(y_val, val_pred)

sub = sample.copy()
sub[target_col] = pd.Series(model.predict(X_test)).astype(sample[target_col].dtype)
sub.to_csv("submission.csv", index=False)

holdout_path = data / "holdout.csv"
if holdout_path.exists():
    holdout = pd.read_csv(holdout_path)
    X_hold = holdout.drop(columns=[id_col], errors="ignore")[X.columns]
    for col in X_hold.columns:
        if not pd.api.types.is_numeric_dtype(X_hold[col]):
            cats = pd.Categorical(X_hold[col].astype(str)).categories
            X_hold[col] = pd.Categorical(X_hold[col].astype(str), categories=cats).codes
    hold_sub = pd.DataFrame({{id_col: holdout[id_col]}})
    hold_sub[target_col] = pd.Series(model.predict(X_hold)).astype(sample[target_col].dtype)
    hold_sub.to_csv("holdout_predictions.csv", index=False)
{bug}
print(f"val_score: {{score}}")
'''

BUGGY_LINE = "undefined_variable_to_trigger_debug  # noqa"

VERIFIER_PROBLEM_TEMPLATE = '''\
import shutil
from pathlib import Path

sample = Path("problem") / "sample_submission.csv"
shutil.copy(sample, "submission.csv")
'''

# Nothing to imitate: no sample submission and no training data, so the only
# thing this stand-in can do is emit a number and let the problem's verifier
# decide what to make of it. Enough to exercise the engine loop end to end on
# a freshly scaffolded problem.
BARE_TEMPLATE = '''\
import os

try:
    from hillclimb import spaces  # the runtime shim makes this importable

    P = spaces.params({"offset": 0.0})
except ImportError:
    P = {"offset": 0.0}

# a tiny bit of movement per call so successive candidates differ
print("dummy solution")
print(round(0.5 + 0.01 * int(os.environ.get("HILLCLIMB_REPLICATE_SEED", 0)) + P["offset"], 4))
'''

# one declared knob so `hillclimb run --backend dummy` exercises tune jobs
# end to end (the score moves with `offset`, so a tuner has something to find)
PARAMS_TEMPLATE = '''\
{"offset": {"type": "float", "low": 0.0, "high": 0.05, "default": 0.0}}
'''


class DummyBackend:
    """Emits canned sklearn scripts so the whole loop can run without an LLM.
    The first draft is intentionally buggy to exercise the DEBUG path."""

    name = "dummy"

    def __init__(self):
        self.calls = 0

    def invoke(self, request: OperatorRequest) -> OperatorResult:
        self.calls += 1
        if not (request.candidate_dir / "problem" / "sample_submission.csv").exists():
            # problems whose verifier drives solution.py directly have no
            # sample_submission to mimic
            (request.candidate_dir / "solution.py").write_text(BARE_TEMPLATE)
            (request.candidate_dir / "params.json").write_text(PARAMS_TEMPLATE)
            (request.candidate_dir / "notes.md").write_text(
                "dummy: prints a number for the problem's verifier to score\n"
            )
            return OperatorResult(ok=True, session_id=f"dummy-{self.calls}", duration_s=0.0)
        if not (request.candidate_dir / "data" / "train.csv").exists():
            script = VERIFIER_PROBLEM_TEMPLATE
            note = "baseline copy for verifier-defined problem"
            (request.candidate_dir / "solution.py").write_text(script)
            (request.candidate_dir / "notes.md").write_text(note + "\n")
            return OperatorResult(ok=True, session_id=f"dummy-{self.calls}", duration_s=0.0)
        # canned behaviour follows the KIND of attempt, so a climber's own
        # operators get a sensible stand-in too
        from hillclimb.operators import role_of

        role = request.role or role_of(request.operator)
        if role == "create" and self.calls == 1:
            script = SOLVER_TEMPLATE.format(max_iter=50, bug=BUGGY_LINE)
            note = "buggy first draft (HistGradientBoosting, 50 iters)"
        elif role == "repair":
            script = SOLVER_TEMPLATE.format(max_iter=50, bug="")
            note = "fix: removed undefined variable"
        elif role == "refine":
            script = SOLVER_TEMPLATE.format(max_iter=300, bug="")
            note = "improve: raise max_iter 100 -> 300"
        elif role == "combine":
            script = SOLVER_TEMPLATE.format(max_iter=500, bug="")
            note = "ensemble: blend of top candidates (canned stand-in)"
        else:
            script = SOLVER_TEMPLATE.format(max_iter=100, bug="")
            note = f"draft #{self.calls} (HistGradientBoosting, 100 iters)"
        (request.candidate_dir / "solution.py").write_text(script)
        (request.candidate_dir / "notes.md").write_text(note + "\n")
        return OperatorResult(ok=True, session_id=f"dummy-{self.calls}", duration_s=0.0)
