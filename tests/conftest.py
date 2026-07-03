from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hillclimb.config import Config
from hillclimb.task import TaskSpec

OK_SCRIPT = """\
import shutil
shutil.copy("data/sample_submission.csv", "submission.csv")
print("val_score: {score}")
"""

CRASH_SCRIPT = 'raise RuntimeError("boom")\n'


@pytest.fixture
def task(tmp_path: Path) -> TaskSpec:
    data_dir = tmp_path / "public"
    data_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n")
    (data_dir / "train.csv").write_text("id,feature,target\n1,0.5,1\n2,0.1,0\n")
    (data_dir / "test.csv").write_text("id,feature\n3,0.4\n4,0.2\n")
    (data_dir / "description.md").write_text("Predict target from feature.")
    return TaskSpec(
        task_id="synthetic",
        comp_id="synthetic",
        data_dir=data_dir,
        description="Predict target from feature.",
        metric_name="accuracy",
        lower_is_better=False,
        sample_submission=data_dir / "sample_submission.csv",
        time_budget_s=3600,
    )


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.paths.runs_dir = tmp_path / "runs"
    cfg.paths.runtime_python = Path(sys.executable)
    cfg.budget.exec_timeout_s = 30
    return cfg


def ok_script(score: float) -> str:
    return OK_SCRIPT.format(score=score)
