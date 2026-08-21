"""MLE-bench problem provider: resolves `mlebench://<comp-id-or-split>`.

Competition metadata (description, metric, score direction) is read straight
from the mle-bench checkout — located by walking up from
`config.paths.mlebench_python` — and the problem's data is the PREPARED
PUBLIC split (`<data>/<comp_id>/prepared/public`). The engine never sees the
private answers: candidates climb on the agent's own validation score, and
the official `mlebench grade-sample` run happens exactly once on the selected
candidate after the search finishes (api._mlebench_grade). That ordering is
the MLE-bench selection-integrity contract: the private test set must not
influence which candidate wins.

Split names resolve as virtual suites: `mlebench://low` (alias
`mlebench://lite`, the 22-competition MLE-bench Lite set) launches one search
per listed competition, same shape as `emflow://gefcom2014`.

No mlebench import happens here — the sidecar stays subprocess-only
(grading.py), so the core carries no new dependency.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hillclimb.config import Config
from hillclimb.problem import ProblemSpec, ResolvedTarget, SuiteSpec

SPLIT_ALIASES = {"lite": "low"}  # MLE-bench Lite = the "low complexity" split


def mlebench_repo_root(config: Config) -> Path:
    """The mle-bench checkout, found by walking up from mlebench_python (the
    venv usually lives inside the checkout). Normalization is lexical —
    resolve() would follow the venv python symlink out of the checkout."""
    import os

    start = Path(os.path.normpath(Path(config.paths.mlebench_python).absolute()))
    for parent in start.parents:
        if (parent / "mlebench" / "competitions").is_dir():
            return parent
    raise FileNotFoundError(
        "mle-bench checkout not found above paths.mlebench_python "
        f"({config.paths.mlebench_python}); point it at the venv python inside "
        "a mle-bench clone (e.g. ../mle-bench/.venv/bin/python)"
    )


def mlebench_data_dir(config: Config) -> Path:
    """Where `mlebench prepare` put the data: the configured dir, or the
    mlebench CLI's default cache locations."""
    if config.paths.mlebench_data_dir:
        return Path(config.paths.mlebench_data_dir).resolve()
    for candidate in (
        Path.home() / "Library" / "Caches" / "mle-bench" / "data",  # appdirs, darwin
        Path.home() / ".cache" / "mle-bench" / "data",  # appdirs, linux
    ):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "no mle-bench data directory found; set paths.mlebench_data_dir or run "
        "`mlebench prepare` with its default cache"
    )


def _lower_is_better(leaderboard_csv: Path) -> bool:
    """Score direction the way mlebench's Grader defines it: the leaderboard
    is ranked best-first, so top < bottom means lower is better."""
    scores = pd.read_csv(leaderboard_csv)["score"].dropna()
    if len(scores) < 2:
        raise ValueError(f"leaderboard too small to infer score direction: {leaderboard_csv}")
    return bool(scores.iloc[0] < scores.iloc[-1])


def load_mlebench_problem(comp_id: str, config: Config) -> ProblemSpec:
    import yaml

    comp_dir = mlebench_repo_root(config) / "mlebench" / "competitions" / comp_id
    if not comp_dir.is_dir():
        raise FileNotFoundError(f"unknown MLE-bench competition: {comp_id} ({comp_dir})")
    meta = yaml.safe_load((comp_dir / "config.yaml").read_text())

    public = mlebench_data_dir(config) / comp_id / "prepared" / "public"
    sample_submission = public / "sample_submission.csv"
    if not sample_submission.exists():
        raise FileNotFoundError(
            f"competition data not prepared: {public} has no sample_submission.csv — "
            f"run `mlebench prepare -c {comp_id}` in the mle-bench venv first"
        )

    from hillclimb.runtime import RUN_SOLUTION

    return ProblemSpec(
        problem_id=comp_id,
        problem_dir=public,  # data listing + workspace ./problem both serve the public split
        data_dir=public,
        description=(comp_dir / "description.md").read_text(),
        metric_name=meta["grader"]["name"],
        lower_is_better=_lower_is_better(comp_dir / "leaderboard.csv"),
        time_budget_s=config.budget.total_s,
        # the competition ships no runnable validator: the agent splits the
        # public data and reports its own score; official grading is one
        # `mlebench grade-sample` run after the search
        verifier_cmd=["{python}", str(RUN_SOLUTION), "{solution}", "--require", "submission.csv"],
        verifier_display="python solution.py   (must write ./submission.csv)",
        report_trusted=False,
        contract_template="contract_submission",
        # never selected (no trials), but keeps best/ shippable if no
        # candidate ever succeeds
        baseline_files={"submission.csv": sample_submission},
        mlebench_comp_id=comp_id,
    )


def resolve_mlebench_target(name: str, config: Config) -> ResolvedTarget:
    """A split name (`low`/`lite`/`medium`/`high`/any experiments/splits file)
    is a virtual suite — one search per competition; anything else is a single
    competition."""
    split = SPLIT_ALIASES.get(name, name)
    splits_file = mlebench_repo_root(config) / "experiments" / "splits" / f"{split}.txt"
    comp_dir = mlebench_repo_root(config) / "mlebench" / "competitions" / name
    if not comp_dir.is_dir() and splits_file.is_file():
        comps = [line.strip() for line in splits_file.read_text().splitlines() if line.strip()]
        return ResolvedTarget(
            kind="suite",
            suite=SuiteSpec(
                suite_id=f"mlebench-{split}",
                suite_path=splits_file,
                problems=[f"mlebench://{comp}" for comp in comps],
            ),
        )
    return ResolvedTarget(kind="problem", problem=load_mlebench_problem(name, config))
