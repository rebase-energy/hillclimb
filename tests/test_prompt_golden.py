"""Byte-level goldens for every prompt the four greedy scenarios produce.

The restructure moves prompt assembly out of the searcher into Operator
classes (docs: harness + climber plan, phase 2); these goldens are the proof
that not one byte of a prompt changed on the way. Machine- and run-specific
text (temp dirs, the interpreter, the clock) is normalized before comparing.

Regenerate on purpose with `HILLCLIMB_UPDATE_GOLDENS=1 uv run pytest
tests/test_prompt_golden.py` and review the diff like any other change.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

from hillclimb.agents.fake import FakeAgent
from tests.test_parallel_search import GOLDEN_SCENARIOS, make_searcher

GOLDEN_DIR = Path(__file__).parent / "golden" / "prompts"
UPDATE = os.environ.get("HILLCLIMB_UPDATE_GOLDENS") == "1"


def normalize(text: str, tmp_path: Path) -> str:
    """Strip what differs between machines and runs, nothing else."""
    for real, token in (
        (str(tmp_path.resolve()), "<TMP>"),
        (str(tmp_path), "<TMP>"),
        (sys.executable, "<PYTHON>"),
        (str(Path.home()), "<HOME>"),
    ):
        text = text.replace(real, token)
    # the shim dir is keyed by a content hash of spaces.py
    text = re.sub(r"interface-shim[-/][0-9a-f]{12}", "interface-shim-<HASH>", text)
    # the budget clock (BudgetManager.remaining_str: "1h 05m" | "59 minutes")
    text = re.sub(
        r"(time remaining for this problem: |Remaining search budget: )(\d+h \d+m|\d+ minutes)",
        r"\1<CLOCK>", text,
    )
    # measured wall-clock / CPU seconds inside feedback JSON (gepa's ASI payload)
    text = re.sub(r'("(?:duration|cpu|wall)[a-z_]*": )\d+(?:\.\d+)?(?:e-?\d+)?', r"\1<SECONDS>", text)
    return text


def collect_prompts(search_dir: Path, tmp_path: Path) -> dict[str, str]:
    prompts = {}
    for path in sorted((search_dir / "candidates").glob("*/prompt.md")):
        prompts[path.parent.name] = normalize(path.read_text(), tmp_path)
    return prompts


@pytest.mark.parametrize("scenario_name", sorted(GOLDEN_SCENARIOS))
def test_prompts_match_golden(task, config, tmp_path, scenario_name):
    scenario = GOLDEN_SCENARIOS[scenario_name]
    agent = FakeAgent()
    scenario.queue(agent)
    searcher, _journal, search_dir = make_searcher(
        task, config, agent,
        max_candidates=scenario.max_candidates, **scenario.searcher_kwargs(),
    )
    searcher.run()

    _check(scenario_name, collect_prompts(search_dir, tmp_path))


def _check(name: str, prompts: dict[str, str]) -> None:
    assert prompts, "the scenario produced no prompts"
    golden_dir = GOLDEN_DIR / name
    if UPDATE:
        golden_dir.mkdir(parents=True, exist_ok=True)
        for stale in golden_dir.glob("*.md"):
            stale.unlink()
        for key, text in prompts.items():
            (golden_dir / f"{key}.md").write_text(text)
        return
    expected = {p.stem: p.read_text() for p in sorted(golden_dir.glob("*.md"))}
    assert sorted(prompts) == sorted(expected), "a different set of prompts was written"
    for key, text in prompts.items():
        assert text == expected[key], f"{name}/{key}: prompt bytes changed"


def test_openevolve_prompts_match_golden(task, config, tmp_path):
    """A policy's `extra_prompt_context` and its inspiration files ride the
    same assembly path; MAP-Elites sampling is seeded, so this is stable."""
    pytest.importorskip("openevolve")
    from hillclimb.harness.budget import BudgetManager
    from hillclimb.harness.dirs import create_search_dir
    from hillclimb.harness.journal import Journal
    from hillclimb.modules.policies import get_policy
    from tests.harness_factory import SearchRig
    from tests.conftest import local_executor, ok_script
    from tests.test_openevolve_policy import PARAMS

    config.climber.ref = "openevolve"
    config.climber.params = {**PARAMS, "num_drafts": 2}
    agent = FakeAgent()
    for score, note in ((0.6, "one\n"), (0.7, "two\n"), (0.8, "three\n")):
        agent.queue(script=ok_script(score), notes=note)
    search_dir = create_search_dir(config.paths.runs_dir, "test-search")
    SearchRig(
        problem=task, config=config, journal=Journal(search_dir / "journal.jsonl"),
        agent=agent, executor=local_executor(), budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir, max_candidates=4, log=lambda *_: None,
        policy=get_policy("openevolve", config.climber.params),
    ).run()
    _check("openevolve", collect_prompts(search_dir, tmp_path))


def test_gepa_proposer_prompts_match_golden(task, config, tmp_path):
    """What a `gepa-reflect` agent is told: the reflective body (a template
    like any other) with the problem's contract appended by the harness."""
    from tests.conftest import ok_script
    from tests.gepa_fakes import FakeGEPADriver, make_gepa

    agent = FakeAgent()
    agent.queue(script=ok_script(0.6))
    agent.queue(script=ok_script(0.7))
    search = make_gepa(task, config, tmp_path, agent=agent, driver=FakeGEPADriver(steps=2))
    search.run()
    _check("gepa", collect_prompts(search.search_dir, tmp_path))
