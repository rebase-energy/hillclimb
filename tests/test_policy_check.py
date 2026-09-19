"""The policy conformance check: greedy conforms; each contract breach a
broken policy can commit is named; the CLI wraps it over the store."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from hillclimb.candidate import Candidate
from hillclimb.journal import Journal
from hillclimb.policies.greedy import GreedyPolicy
from hillclimb.policy import Action, PolicyInput
from hillclimb.policy_check import JournalCase, check_policy
from tests.test_policy import add_candidate


def _cases(tmp_path: Path) -> list[JournalCase]:
    """A journal with a debug chain, scored drafts and a tunable candidate,
    living under a search dir the read-only check watches."""
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    journal = Journal(search_dir / "journal.jsonl")
    add_candidate(journal, "c000", "baseline", val_score=0.1)
    add_candidate(journal, "c001", "draft", status="buggy")
    add_candidate(journal, "c002", "debug", val_score=0.5, parent_id="c001",
                  solution="a\n", tmp_path=search_dir)
    add_candidate(journal, "c003", "draft", val_score=0.7, solution="b\n", tmp_path=search_dir)
    add_candidate(journal, "c004", "draft", val_score=0.6, solution="c\n", tmp_path=search_dir)
    return [JournalCase("t/one", journal, True, 3600, search_dir)]


def _by_check(report, check: str, journal: str | None = None):
    return [
        f for f in report.findings
        if f.check == check and (journal is None or f.journal == journal)
    ]


def test_greedy_conforms(config, tmp_path):
    report = check_policy(lambda: GreedyPolicy(), _cases(tmp_path), config)
    assert report.ok, report.render()
    assert report.policy == "greedy"
    checks = {f.check for f in report.findings}
    assert checks == {"constructs", "starts", "replay", "idempotent", "references", "read-only", "templates"}
    # the synthetic empty journal is always probed, before the recorded ones
    assert [f.journal for f in report.findings if f.check == "replay"] == ["empty", "t/one"]
    assert "draft (minimal)" in _by_check(report, "starts")[0].detail
    assert "conforms" in report.render()
    assert report.to_dict()["ok"] is True


class StallingPolicy:
    name, params = "stall", {}

    def propose(self, view):
        return None

    def observe(self, view, candidate):
        pass


class RandomPolicy:
    name, params = "random", {}

    def propose(self, view):
        return Action(operator="draft", complexity=random.choice(["minimal", "advanced"]))

    def observe(self, view, candidate):
        pass


class CountingPolicy:
    """propose consumes state: the second ask differs from the first."""

    name, params = "counting", {}

    def __init__(self):
        self.n = 0

    def propose(self, view):
        self.n += 1
        return Action(operator="draft", complexity="minimal" if self.n % 2 else "advanced")

    def observe(self, view, candidate):
        pass


class DanglingPolicy:
    name, params = "dangling", {}

    def propose(self, view):
        if view.journal.candidates:
            return Action(operator="improve", target_id="c999")
        return Action(operator="ensemble")  # no target either

    def observe(self, view, candidate):
        pass


class WrongTargetPolicy:
    name, params = "wrong-target", {}

    def propose(self, view):
        if "c003" in view.journal.candidates:
            return Action(operator="debug", target_id="c003")  # c003 is ok, not buggy
        return Action(operator="draft")

    def observe(self, view, candidate):
        pass


class MutatingPolicy:
    name, params = "mutating", {}

    def propose(self, view):
        for c in view.journal.candidates.values():
            c.pruned = True
        return Action(operator="draft")

    def observe(self, view, candidate):
        pass


class WritingPolicy:
    name, params = "writing", {}

    def propose(self, view):
        for c in view.journal.candidates.values():
            if c.candidate_dir:
                Path(c.candidate_dir, "scratch.txt").write_text("x")
        return Action(operator="draft")

    def observe(self, view, candidate):
        pass


class SingletonFactory:
    def __init__(self):
        self.instance = GreedyPolicy()

    def __call__(self):
        return self.instance


def test_stall_on_empty_journal_is_a_breach(config, tmp_path):
    report = check_policy(StallingPolicy, _cases(tmp_path), config)
    starts = _by_check(report, "starts")[0]
    assert not starts.ok and "never start" in starts.detail


def test_nondeterminism_is_a_breach(config, tmp_path):
    random.seed(0)
    report = check_policy(RandomPolicy, _cases(tmp_path), config, budget_points=tuple([1.0] * 12))
    assert not report.ok
    assert any(not f.ok and f.check in ("replay", "idempotent") for f in report.findings)


def test_consuming_propose_is_a_breach(config, tmp_path):
    report = check_policy(CountingPolicy, _cases(tmp_path), config)
    failed = [f for f in _by_check(report, "idempotent") if not f.ok]
    assert failed and "asking twice" in failed[0].detail


def test_dangling_and_missing_targets_are_breaches(config, tmp_path):
    report = check_policy(DanglingPolicy, _cases(tmp_path), config)
    details = [f.detail for f in _by_check(report, "references") if not f.ok]
    assert any("c999" in d and "not in the journal" in d for d in details)
    assert any("ensemble without a target_id" in d for d in details)


def test_operator_target_mismatch_is_a_breach(config, tmp_path):
    report = check_policy(WrongTargetPolicy, _cases(tmp_path), config)
    details = [f.detail for f in _by_check(report, "references", "t/one") if not f.ok]
    assert details and "debug targets c003 whose status is passing" in details[0]


def test_mutating_the_journal_is_a_breach(config, tmp_path):
    report = check_policy(MutatingPolicy, _cases(tmp_path), config)
    failed = [f for f in _by_check(report, "read-only", "t/one") if not f.ok]
    assert failed and "mutated" in failed[0].detail


def test_writing_files_is_a_breach(config, tmp_path):
    cases = _cases(tmp_path)
    report = check_policy(WritingPolicy, cases, config)
    failed = [f for f in _by_check(report, "read-only", "t/one") if not f.ok]
    assert failed and "files under the search dir changed" in failed[0].detail


def test_shared_instance_factory_is_a_breach(config, tmp_path):
    report = check_policy(SingletonFactory(), _cases(tmp_path), config)
    constructs = _by_check(report, "constructs")[0]
    assert not constructs.ok and "same object" in constructs.detail


def test_raising_policy_is_reported_not_raised(config, tmp_path):
    class Broken:
        name, params = "broken", {}

        def propose(self, view):
            raise KeyError("oops")

        def observe(self, view, candidate):
            pass

    report = check_policy(Broken, _cases(tmp_path), config)
    assert not report.ok
    assert any("KeyError" in f.detail for f in report.findings if not f.ok)


def test_prompt_override_lint_is_part_of_the_check(config, tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "draft.md").write_text("{{description}} and {{no_such_token}}\n")
    report = check_policy(lambda: GreedyPolicy(), [], config, prompts_dir=prompts)
    templates = _by_check(report, "templates")[0]
    assert not templates.ok and "{{no_such_token}}" in templates.detail
    (prompts / "draft.md").write_text("{{description}} only\n")
    report = check_policy(lambda: GreedyPolicy(), [], config, prompts_dir=prompts)
    assert _by_check(report, "templates")[0].ok


def test_cli_replays_the_stores_journals(tmp_path, monkeypatch, capsys):
    from hillclimb.api import create_run, create_search
    from hillclimb.cli import main as cli_main
    from hillclimb.config import Config
    from hillclimb.problem import load_problem
    from hillclimb.run import RunMeta
    from tests.test_cli import write_problem

    root = tmp_path / "hillclimb"
    root.mkdir()
    (root / "config.yaml").write_text("paths:\n  problems_dir: problems\n")
    write_problem(tmp_path / "problems", "p")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    monkeypatch.delenv("HILLCLIMB_WORKSPACE", raising=False)
    config = Config.load()
    run_dir = create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="p", problem_ids=["p"]))
    search_dir = create_search(config, load_problem("p", config), run_dir, "r1", 600)
    journal = Journal(search_dir / "journal.jsonl")
    add_candidate(journal, "c000", "baseline", val_score=0.1)
    add_candidate(journal, "c001", "draft", val_score=0.4)

    with pytest.raises(SystemExit) as exc:
        cli_main(["policy", "check", "--json", "--set", "search.policy_params.num_drafts=1"])
    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["policy"] == "greedy"
    assert payload["journals"] == ["r1/p"]
    assert payload["resolved_params"]["num_drafts"] == 1
    replay = next(f for f in payload["findings"] if f["check"] == "replay" and f["journal"] == "r1/p")
    assert "improve -> c001" in replay["detail"]  # one draft satisfied num_drafts=1

    # a climber that brings its own SearchLoop is out of scope, and says so
    with pytest.raises(SystemExit) as exc:
        cli_main(["policy", "check", "--policy", "gepa"])
    assert exc.value.code == 2
    assert "brings its own SearchLoop" in capsys.readouterr().err


def test_cli_checks_a_file_policy_relative_to_the_hillclimb_dir(tmp_path, monkeypatch, capsys):
    from hillclimb.cli import main as cli_main
    from tests.test_policy import FILE_POLICY

    root = tmp_path / "hillclimb"
    (root / "policies").mkdir(parents=True)
    (root / "config.yaml").write_text("")
    (root / "policies" / "drafts_only.py").write_text(FILE_POLICY)
    monkeypatch.chdir(tmp_path / "hillclimb")  # any subdirectory resolves the same dir
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    monkeypatch.delenv("HILLCLIMB_WORKSPACE", raising=False)

    with pytest.raises(SystemExit) as exc:
        cli_main(["policy", "check", "--policy", "hillclimb/policies/drafts_only.py", "--json"])
    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["policy"].startswith("drafts-only (") and payload["policy"].endswith("drafts_only.py)")

    (root / "policies" / "stalls.py").write_text(
        "class Stalls:\n"
        "    def propose(self, view):\n        return None\n"
        "    def observe(self, view, candidate):\n        pass\n"
    )
    with pytest.raises(SystemExit) as exc:
        cli_main(["policy", "check", "--policy", "hillclimb/policies/stalls.py"])
    assert exc.value.code == 1
    assert "never start" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        cli_main(["policy", "check", "--policy", "hillclimb/policies/missing.py"])
    assert exc.value.code == 2
    assert "policy file not found" in capsys.readouterr().err
