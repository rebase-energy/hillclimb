from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hillclimb.config import Config
from hillclimb.harness.executor import CommandExecutor
from hillclimb.problem import ProblemSpec
from hillclimb.runtime import RUN_SOLUTION

# fixture problems are self-reported: the verifier just runs solution.py and
# takes the `val_score:` line it prints (what MLE-bench problems do for real)
SELF_REPORT_CMD = ["{python}", str(RUN_SOLUTION), "{solution}", "--require", "submission.csv"]


def local_executor() -> CommandExecutor:
    return CommandExecutor(Path(sys.executable), SELF_REPORT_CMD)


@pytest.fixture(autouse=True)
def _user_level_isolated(tmp_path: Path, monkeypatch):
    """The user level — `~/.config/hillclimb/` (config.yaml, .env, the intro
    marker) — is where `connect` writes by default. Point it at the test's
    tmp dir for EVERY test, so a test that runs a command can never edit the
    developer's real files (one did, once)."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


@pytest.fixture(autouse=True)
def _quota_off(monkeypatch):
    """Real agents snapshot subscription quota around every call — tests
    must never reach the network or the developer's keychain for it."""
    monkeypatch.setenv("HILLCLIMB_QUOTA", "off")


def executor_for(problem) -> CommandExecutor:
    """The problem's own verifier command, run by the dev interpreter."""
    return CommandExecutor(Path(sys.executable), problem.verifier_cmd, problem.verifier_env)

OK_SCRIPT = """\
import shutil
shutil.copy("data/sample_submission.csv", "submission.csv")
print("val_score: {score}")
"""

CRASH_SCRIPT = 'raise RuntimeError("boom")\n'


@pytest.fixture
def task(tmp_path: Path) -> ProblemSpec:
    data_dir = tmp_path / "public"
    data_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n")
    (data_dir / "train.csv").write_text("id,feature,target\n1,0.5,1\n2,0.1,0\n")
    (data_dir / "test.csv").write_text("id,feature\n3,0.4\n4,0.2\n")
    (data_dir / "description.md").write_text("Predict target from feature.")
    return ProblemSpec(
        problem_id="synthetic",
        problem_dir=data_dir,
        data_dir=data_dir,
        description="Predict target from feature.",
        metric_name="accuracy",
        higher_is_better=True,
        verifier_cmd=SELF_REPORT_CMD,
        report_trusted=False,
        contract_template="contract_submission",
        time_budget_s=3600,
    )


@pytest.fixture
def task_larger(tmp_path: Path) -> ProblemSpec:
    """Bigger fixture problem (30 rows) so a 30% holdout split is meaningful.
    target = feature > 0, so holdout accuracy is fully controllable."""
    data_dir = tmp_path / "public-large"
    data_dir.mkdir()
    n = 30
    features = [(-1.0 if i % 2 else 1.0) * (1 + i) for i in range(n)]
    rows = ["id,feature,target"] + [
        f"{i},{features[i]},{1 if features[i] > 0 else 0}" for i in range(n)
    ]
    (data_dir / "train.csv").write_text("\n".join(rows) + "\n")
    (data_dir / "test.csv").write_text("id,feature\n100,0.5\n101,-0.5\n")
    (data_dir / "sample_submission.csv").write_text("id,target\n100,0\n101,0\n")
    (data_dir / "description.md").write_text("Predict target from feature.")
    return ProblemSpec(
        problem_id="synthetic-large",
        problem_dir=data_dir,
        data_dir=data_dir,
        description="Predict target from feature.",
        metric_name="accuracy",
        higher_is_better=True,
        verifier_cmd=SELF_REPORT_CMD,
        report_trusted=False,
        contract_template="contract_submission",
        time_budget_s=3600,
    )


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.paths.runs_dir = tmp_path / "runs"
    cfg.paths.problems_dir = Path("problems")  # this repo keeps problems/ at the root
    cfg.paths.runtime_python = Path(sys.executable)
    cfg.budget.exec_timeout_s = 30
    return cfg


def ok_script(score: float) -> str:
    return OK_SCRIPT.format(score=score)
