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


class DummyBackend:
    """Emits canned sklearn scripts so the whole loop can run without an LLM.
    The first draft is intentionally buggy to exercise the DEBUG path."""

    name = "dummy"

    def __init__(self):
        self.calls = 0

    def invoke(self, request: OperatorRequest) -> OperatorResult:
        self.calls += 1
        if request.operator == "draft" and self.calls == 1:
            script = SOLVER_TEMPLATE.format(max_iter=50, bug=BUGGY_LINE)
            note = "buggy first draft (HistGradientBoosting, 50 iters)"
        elif request.operator == "debug":
            script = SOLVER_TEMPLATE.format(max_iter=50, bug="")
            note = "fix: removed undefined variable"
        elif request.operator == "improve":
            script = SOLVER_TEMPLATE.format(max_iter=300, bug="")
            note = "improve: raise max_iter 100 -> 300"
        else:
            script = SOLVER_TEMPLATE.format(max_iter=100, bug="")
            note = f"draft #{self.calls} (HistGradientBoosting, 100 iters)"
        (request.workspace / "solution.py").write_text(script)
        (request.workspace / "notes.md").write_text(note + "\n")
        return OperatorResult(ok=True, session_id=f"dummy-{self.calls}", duration_s=0.0)
