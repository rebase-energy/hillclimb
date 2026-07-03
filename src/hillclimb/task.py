from __future__ import annotations

import subprocess
from pathlib import Path

import yaml
from pydantic import BaseModel

from hillclimb.config import Config


class TaskSpec(BaseModel):
    """Generic task interface. MLE-bench is adapter #1; future adapters
    (e.g. energy-forecasting tasks) only need to produce one of these."""

    task_id: str
    comp_id: str
    data_dir: Path
    description: str
    metric_name: str
    lower_is_better: bool
    sample_submission: Path
    time_budget_s: int


def mlebench_data_dir(config: Config) -> Path:
    """Resolve the mlebench cache root by asking the mlebench venv itself,
    so we never drift from its default."""
    if config.paths.mlebench_data_dir:
        return config.paths.mlebench_data_dir
    result = subprocess.run(
        [
            str(config.paths.mlebench_python),
            "-c",
            "from mlebench.registry import registry; print(registry.get_data_dir())",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip().splitlines()[-1])


def load_task(task_id: str, config: Config) -> TaskSpec:
    """`task_id` is either a name under the tasks dir or a path to a task YAML.
    A task YAML with an explicit `data_dir` bypasses the mlebench cache — the
    seam for non-Kaggle tasks (e.g. energy forecasting) and local fixtures."""
    as_path = Path(task_id)
    if as_path.suffix == ".yaml" and as_path.exists():
        task_yaml = as_path
        task_id = as_path.stem
    else:
        task_yaml = config.paths.tasks_dir / f"{task_id}.yaml"
    meta = yaml.safe_load(task_yaml.read_text())
    comp_id = meta["comp_id"]
    if meta.get("data_dir"):
        public_dir = (task_yaml.parent / meta["data_dir"]).resolve()
    else:
        public_dir = mlebench_data_dir(config) / comp_id / "prepared" / "public"
    if not public_dir.exists():
        raise FileNotFoundError(
            f"Prepared data not found at {public_dir}. "
            f"Run: mlebench prepare -c {comp_id} (requires Kaggle credentials "
            "and accepting the competition rules on kaggle.com)."
        )
    description = (public_dir / "description.md").read_text()
    sample = public_dir / "sample_submission.csv"
    if not sample.exists():
        # older comps vary: sampleSubmission.csv, sample_submission.csv, ...
        candidates = [
            p for p in sorted(public_dir.iterdir())
            if "ubmission" in p.name and p.suffix == ".csv"
        ]
        if not candidates:
            raise FileNotFoundError(f"No sample submission found in {public_dir}")
        sample = candidates[0]
    return TaskSpec(
        task_id=task_id,
        comp_id=comp_id,
        data_dir=public_dir,
        description=description,
        metric_name=meta["metric"],
        lower_is_better=meta["lower_is_better"],
        sample_submission=sample,
        time_budget_s=meta.get("time_budget_s", config.budget.total_s),
    )
