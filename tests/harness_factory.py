"""Build a bare `Harness` for tests: no policy, no loop — drive it with
`harness.run(Action(...))` or `harness.execute(loop)`."""

from __future__ import annotations

from hillclimb.budget import BudgetManager
from hillclimb.dirs import create_search_dir
from hillclimb.harness import Harness
from hillclimb.journal import Journal
from tests.conftest import local_executor


def make_harness(task, config, backend, *, name: str = "test-search", **kwargs):
    """-> (harness, journal, search_dir)"""
    search_dir = create_search_dir(config.paths.runs_dir, name)
    journal = Journal(search_dir / "journal.jsonl")
    kwargs.setdefault("budget", BudgetManager(3600, stop_margin_s=1))
    harness = Harness(
        problem=task,
        config=config,
        journal=journal,
        backend=backend,
        executor=local_executor(),
        search_dir=search_dir,
        log=lambda *_: None,
        **kwargs,
    )
    return harness, journal, search_dir
