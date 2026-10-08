"""The policy conformance check: greedy conforms; each contract breach a
broken policy can commit is named; the CLI wraps it over the store."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from hillclimb.harness.candidate import Candidate
from hillclimb.harness.journal import Journal
from tests.catalog_fixture import GEPA, GREEDY, block, greedy_classes

Greedy, Best = greedy_classes()
from hillclimb.modules.policies.base import Action, SearchState
from hillclimb.modules.policies.check import JournalCase, check_policy
from tests.test_policy import add_candidate
from tests.catalog_fixture import GREEDY, class_ref
from tests.folder_config import write_config


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
    report = check_policy(lambda: Greedy(selector=Best()), _cases(tmp_path), config)
    assert report.ok, report.render()
    assert report.policy == "greedy"
    checks = {f.check for f in report.findings}
    assert checks == {"constructs", "starts", "replay", "idempotent", "resume", "references", "read-only", "templates"}
    # the synthetic empty journal is always probed, before the recorded ones
    assert [f.journal for f in report.findings if f.check == "replay"] == ["empty", "t/one"]
    assert "draft (minimal)" in _by_check(report, "starts")[0].detail
    assert "conforms" in report.render()
    assert report.to_dict()["ok"] is True


class StallingPolicy:
    name, params = "stall", {}

    def propose(self, view, selection):
        return None

    def observe(self, view, candidate):
        pass


class RandomPolicy:
    name, params = "random", {}

    def propose(self, view, selection):
        return Action(operator="draft", args={"complexity": random.choice(["minimal", "advanced"])})

    def observe(self, view, candidate):
        pass


class CountingPolicy:
    """propose consumes state: the second ask differs from the first."""

    name, params = "counting", {}

    def __init__(self):
        self.n = 0

    def propose(self, view, selection):
        self.n += 1
        return Action(operator="draft", args={"complexity": "minimal" if self.n % 2 else "advanced"})

    def observe(self, view, candidate):
        pass


class DanglingPolicy:
    name, params = "dangling", {}

    def propose(self, view, selection):
        if view.journal.candidates:
            return Action(operator="improve", target_id="c999")
        return Action(operator="ensemble")  # no target either

    def observe(self, view, candidate):
        pass


class WrongTargetPolicy:
    name, params = "wrong-target", {}

    def propose(self, view, selection):
        if "c003" in view.journal.candidates:
            return Action(operator="debug", target_id="c003")  # c003 is ok, not buggy
        return Action(operator="draft")

    def observe(self, view, candidate):
        pass


class MutatingPolicy:
    name, params = "mutating", {}

    def propose(self, view, selection):
        for c in view.journal.candidates.values():
            c.pruned = True
        return Action(operator="draft")

    def observe(self, view, candidate):
        pass


class WritingPolicy:
    name, params = "writing", {}

    def propose(self, view, selection):
        for c in view.journal.candidates.values():
            if c.candidate_dir:
                Path(c.candidate_dir, "scratch.txt").write_text("x")
        return Action(operator="draft")

    def observe(self, view, candidate):
        pass


class SingletonFactory:
    def __init__(self):
        self.instance = Greedy()

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


class JournalSizePolicy:
    """Keeps what the journal looked like when each result was shown — the
    shape of the OpenEvolve bug: a live search shows a journal that grows,
    a resumed one the finished journal for every candidate."""

    name, params = "journal-size", {}

    def __init__(self):
        self.seen = 0

    def propose(self, view, selection):
        return Action(operator="draft", args={"complexity": str(self.seen)})

    def observe(self, view, candidate):
        self.seen += len(view.journal.candidates)


def test_state_that_depends_on_when_a_result_was_shown_is_a_breach(config, tmp_path):
    report = check_policy(JournalSizePolicy, _cases(tmp_path), config)
    failed = [f for f in _by_check(report, "resume", "t/one") if not f.ok]
    assert failed and "a resumed search would diverge" in failed[0].detail
    assert all(f.ok for f in _by_check(report, "replay"))  # two replays agree: only `resume` sees it


class CrossoverPolicy:
    name, params = "crossover", {}

    def propose(self, view, selection):
        return Action(operator="crossover")

    def observe(self, view, candidate):
        pass


def test_a_climbers_own_operator_is_known_to_the_check(config, tmp_path):
    from hillclimb.climber import OperatorSet
    from hillclimb.modules.operators import Operator
    from hillclimb.modules.operators.base import Attempt

    class Crossover(Operator):
        name, kind = "crossover", "combine"

        def prepare(self, ctx):
            return Attempt(prompt="cross")

    cases = _cases(tmp_path)
    report = check_policy(CrossoverPolicy, cases, config)
    assert any("unknown operator 'crossover'" in f.detail for f in _by_check(report, "references") if not f.ok)
    operators = OperatorSet({"crossover": (Crossover, {})})
    report = check_policy(CrossoverPolicy, cases, config, operators=operators)
    assert report.ok, report.render()


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

        def propose(self, view, selection):
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
    report = check_policy(lambda: Greedy(), [], config, prompts_dir=prompts)
    templates = _by_check(report, "templates")[0]
    assert not templates.ok and "{{no_such_token}}" in templates.detail
    (prompts / "draft.md").write_text("{{description}} only\n")
    report = check_policy(lambda: Greedy(), [], config, prompts_dir=prompts)
    assert _by_check(report, "templates")[0].ok


def test_cli_replays_the_stores_journals(tmp_path, monkeypatch, capsys):
    from hillclimb.api import create_run, create_search
    from hillclimb.cli import main as cli_main
    from hillclimb.config import Config
    from hillclimb.problem import load_problem
    from hillclimb.harness.run import RunMeta
    from tests.test_cli import write_problem

    write_config(tmp_path, {"climber": str(GREEDY)})  # the folder names its climber: the catalog file
    write_problem(tmp_path / "problems", "p")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    config = Config.load()
    run_dir = create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="p", problem_ids=["p"]))
    search_dir = create_search(config, load_problem("p", config), run_dir, "r1", 600)
    journal = Journal(search_dir / "journal.jsonl")
    add_candidate(journal, "c000", "baseline", val_score=0.1)
    add_candidate(journal, "c001", "draft", val_score=0.4)

    with pytest.raises(SystemExit) as exc:
        cli_main(["climber", "check", "--json", "--set", "climber.params.num_drafts=1"])
    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["policy"] == "greedy"
    assert payload["journals"] == ["r1/p"]
    assert payload["resolved_params"]["num_drafts"] == 1
    replay = next(f for f in payload["findings"] if f["check"] == "replay" and f["journal"] == "r1/p")
    assert "improve -> c001" in replay["detail"]  # one draft satisfied num_drafts=1

    # a climber that brings its own Loop is out of scope, and says so
    with pytest.raises(SystemExit) as exc:
        cli_main(["climber", "check", "--climber", str(GEPA)])
    assert exc.value.code == 2
    assert "brings its own Loop" in capsys.readouterr().err


def test_cli_checks_a_file_policy_relative_to_the_hillclimb_dir(tmp_path, monkeypatch, capsys):
    from hillclimb.cli import main as cli_main
    from tests.test_policy import FILE_POLICY

    root = tmp_path
    (root / "policies").mkdir(parents=True)
    (root / "hillclimb.yaml").write_text("")
    (root / "policies" / "drafts_only.py").write_text(FILE_POLICY)
    monkeypatch.chdir(root)  # commands run from the hillclimb dir's root
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)

    with pytest.raises(SystemExit) as exc:
        cli_main(["climber", "check", "--climber", "policies/drafts_only.py", "--json"])
    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["policy"].startswith("drafts-only (") and payload["policy"].endswith("drafts_only.py)")

    (root / "policies" / "stalls.py").write_text(
        "class Stalls:\n"
        "    def propose(self, view, selection):\n        return None\n"
        "    def observe(self, view, candidate):\n        pass\n"
    )
    with pytest.raises(SystemExit) as exc:
        cli_main(["climber", "check", "--climber", "policies/stalls.py"])
    assert exc.value.code == 1
    assert "never start" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        cli_main(["climber", "check", "--climber", "policies/missing.py"])
    assert exc.value.code == 2
    assert "climber file not found" in capsys.readouterr().err

    # 0.6 removed the old spellings outright
    for gone in (["policy", "check"], ["climber", "check", "--policy", "greedy"]):
        with pytest.raises(SystemExit) as exc:
            cli_main(gone)
        assert exc.value.code == 2


def test_climber_check_takes_a_run_spec_and_checks_every_entrys_climber(tmp_path, monkeypatch, capsys):
    """The run config defines the climber, so the check reads it from there:
    every entry's block (its `set` pairs applied), each distinct one once."""
    import yaml

    from hillclimb.cli import main as cli_main
    from tests.test_cli import write_problem

    (tmp_path / "hillclimb.yaml").write_text("")
    write_problem(tmp_path / "problems", "p")
    (tmp_path / "stalls.py").write_text(
        "class Stalls:\n    name = 'stalls'\n    def propose(self, view, selection):\n        return None\n"
        "    def observe(self, view, candidate):\n        pass\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    spec = tmp_path / "run.yaml"
    spec.write_text(yaml.safe_dump({"problems": [
        {"target": "p", "climber": str(GREEDY), "set": ["climber.params.num_drafts=1"]},
        {"target": "p", "name": "again", "climber": {**block("greedy"), "params": {"num_drafts": 1}}},
    ]}))
    with pytest.raises(SystemExit) as exc:
        cli_main(["climber", "check", "run.yaml", "--json"])
    assert exc.value.code == 0
    (payload,) = json.loads(capsys.readouterr().out)  # two entries, one climber
    assert payload["ok"] and payload["resolved_params"]["num_drafts"] == 1 and payload["entry"].endswith("[1]")

    spec.write_text(yaml.safe_dump({"climber": str(GREEDY), "problems": ["p", {"target": "p", "climber": "stalls.py"}]}))
    with pytest.raises(SystemExit) as exc:
        cli_main(["climber", "check", "run.yaml"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "policy greedy" in out and "policy stalls" in out and "never start" in out

    spec.write_text(yaml.safe_dump({"problems": [{"target": "p", "climber": {"operator_policy": "nope"}}]}))
    with pytest.raises(SystemExit) as exc:
        cli_main(["climber", "check", "run.yaml"])
    assert exc.value.code == 2 and "unknown operator policy 'nope'" in capsys.readouterr().err


def test_the_config_init_writes_shows_a_block_that_loads():
    """The commented `climber:` block in a fresh runs/config.yaml is the
    documentation most people read: uncommented, it must be a valid block."""
    import re

    import yaml

    from hillclimb.config import Config
    from hillclimb.project import RUNS_CONFIG_TEMPLATE

    lines = RUNS_CONFIG_TEMPLATE.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("# climber:") and "whole block" in line)
    block = []
    for line in lines[start:]:
        if not line.startswith("#"):
            break
        block.append(re.sub(r"^# ?", "", line))
    config = Config.model_validate(yaml.safe_load("\n".join(block)))
    assert config.climber.operator_policy == "climbers/greedy/policy.py:Greedy" and config.climber.operators == ["draft", "debug", "improve", "ensemble"]
    shorthand = next(line for line in lines if line.startswith("# climber: climbers/greedy/policy.py"))
    # from this checkout the file exists, so the name expands to the file's own block
    assert Config.model_validate(yaml.safe_load(shorthand[2:])).climber.operator_policy.endswith("climbers/greedy/policy.py:Greedy")
