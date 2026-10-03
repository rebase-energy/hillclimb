"""Experiments: spec → jobs, experiment overrides, tagged searches, experiment comparison
and the report; the `experiment run` launcher and `run --set/--experiment` plumbing."""

from __future__ import annotations

from pathlib import Path

import json
import pytest
import typer
import yaml

from hillclimb.harness.candidate import AgentInfo
from hillclimb.config import Config, parse_set_overrides
from hillclimb.harness.dirs import create_run_dir, create_search_dir
from hillclimb.experiment import (
    StudyRow,
    collect_results,
    expand,
    flatten_overrides,
    load_study,
    render_report,
    resolve_study_path,
    summarize,
)
from hillclimb.harness.journal import Journal
from hillclimb.harness.run import RunMeta, SearchMeta, write_run_meta, write_search_meta
from hillclimb.harness.status import ScoreRef, SearchStatus, write_status
from tests.test_watch import make_candidate

SPEC = """
problems: [circle-packing]
repeats: 2
budget: 10m
noise_floor: 0.02
defaults: {model: sonnet}
experiments:
  greedy: {search.policy: greedy}
  nomem: {search: {policy: greedy}, learning: {enabled: false}}
  openevolve: {search.policy: openevolve, search.policy_params: {population_size: 50}}
"""


def write_spec(path: Path, text: str = SPEC) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class TestSpec:
    def test_load_expand_and_flatten(self, tmp_path):
        spec = load_study(write_spec(tmp_path / "policy-vs-memory.yaml"))
        assert spec.name == "policy-vs-memory"  # the file stem
        assert spec.control == "greedy"
        assert spec.noise_for("circle-packing") == 0.02
        # nested and dotted forms are the same overrides; policy_params stays a mapping
        assert spec.experiment_overrides("nomem") == {"model": "sonnet", "search.policy": "greedy", "learning.enabled": False}
        assert spec.experiment_overrides("openevolve") == {
            "model": "sonnet", "search.policy": "openevolve", "search.policy_params": {"population_size": 50},
        }
        jobs = expand(spec)
        # repeat-major, experiments round-robin inside: every experiment's k-th repeat starts
        # from the same shared state
        assert [(j.index, j.experiment, j.repeat) for j in jobs] == [
            (1, "greedy", 1), (2, "nomem", 1), (3, "openevolve", 1),
            (4, "greedy", 2), (5, "nomem", 2), (6, "openevolve", 2),
        ]

    def test_rejects_bad_specs(self, tmp_path):
        with pytest.raises(ValueError, match="at least two experiments"):
            load_study(write_spec(tmp_path / "one.yaml", "problems: [p]\nexperiments: {a: {}}\n"))
        with pytest.raises(ValueError, match="schedule"):
            load_study(write_spec(tmp_path / "s.yaml", "problems: [p]\nschedule: soon\nexperiments: {a: {}, b: {}}\n"))

    def test_legacy_arms_key_still_loads(self, tmp_path):
        spec = load_study(write_spec(tmp_path / "old.yaml", "problems: [p]\narms: {a: {}, b: {model: opus}}\n"))
        assert list(spec.experiments) == ["a", "b"] and spec.control == "a"

    def test_resolve_by_path_or_name(self, tmp_path):
        hillclimb_dir = tmp_path / "hillclimb"
        spec = write_spec(hillclimb_dir / "experiments" / "ab.yaml")
        assert resolve_study_path("ab", hillclimb_dir) == spec.resolve()
        assert resolve_study_path(str(spec), None) == spec.resolve()
        with pytest.raises(FileNotFoundError, match="experiments/"):
            resolve_study_path("nope", hillclimb_dir)

    def test_an_experiment_names_or_defines_its_climber(self, tmp_path):
        """`climber:` in an experiment is the climber — a preset's name or the
        whole block, never flattened into per-field overrides — and it is
        applied FIRST, so `climber.<field>` overrides (the experiment's own,
        or the study's defaults) edit the block it names."""
        spec = load_study(write_spec(
            tmp_path / "s.yaml",
            "problems: [p]\n"
            "defaults: {climber.params.num_drafts: 2, model: sonnet}\n"
            "experiments:\n"
            "  a: {climber.params.tune_budget: 0, climber: openevolve}\n"
            "  b: {climber: {operator_policy: greedy, tuner: optuna, params: {ensemble: false}}}\n",
        ))
        a, b = spec.experiment_overrides("a"), spec.experiment_overrides("b")
        assert list(a) == ["climber", "climber.params.num_drafts", "model", "climber.params.tune_budget"]
        assert b["climber"] == {"operator_policy": "greedy", "tuner": "optuna", "params": {"ensemble": False}}
        config = Config()
        config.apply_overrides(a)
        assert config.climber.selector_policy == "map-elites"
        assert config.climber.params == {"tune_budget": 0}
        assert config.climber.selector_params == {"ensemble": False, "num_drafts": 2}
        config = Config()
        config.apply_overrides(b)
        assert (config.climber.tuner, config.climber.selector_params) == ("optuna", {"ensemble": False, "num_drafts": 2})
        # and as a child engine receives it: every value a `--set KEY=<json>` pair
        from hillclimb.cli.experiment import _set_value

        child = Config()
        child.apply_overrides(parse_set_overrides([f"{key}={_set_value(value)}" for key, value in b.items()]))
        assert child.climber.block() == config.climber.block()

    def test_flatten_keeps_dict_valued_settings_whole(self):
        assert flatten_overrides({"search": {"policy_params": {"k": 1}, "n_replicates": 2}}) == {
            "search.policy_params": {"k": 1}, "search.n_replicates": 2,
        }


class TestOverrides:
    def test_apply_overrides_walks_dotted_paths_with_coercion(self):
        config = Config()
        config.apply_overrides(parse_set_overrides([
            "search.policy=openevolve", "learning.enabled=false", "search.n_replicates=3",
            "search.policy_params={population_size: 50}", "search.policy_params.seed=7", "model=opus",
        ]))
        assert config.climber.label == "openevolve"
        assert config.learning.enabled is False
        assert config.evaluation.n_replicates == 3
        assert config.climber.params == {"population_size": 50, "seed": 7}  # the dict replaces the preset's, then one key
        assert config.model == "opus"
        with pytest.raises(KeyError, match="search.nope"):
            config.apply_overrides({"search.nope": 1})
        with pytest.raises(ValueError, match="key=value"):
            parse_set_overrides(["garbage"])

    def test_create_search_records_the_tags(self, tmp_path):
        from hillclimb.api import create_search
        from hillclimb.problem import ProblemSpec
        from hillclimb.harness.run import load_search_meta

        config = Config()
        config.paths.runs_dir = tmp_path / "runs"
        config.apply_overrides({"learning.enabled": False})
        problem = ProblemSpec(
            problem_id="p", problem_dir=tmp_path, data_dir=tmp_path,
            description="", metric_name="accuracy", higher_is_better=True,
            verifier_cmd=["./verifier.sh"], time_budget_s=60,
        )
        run_dir = create_run_dir(tmp_path / "runs", "r1")
        search_dir = create_search(
            config, problem, run_dir, "r1", 60,
            study="ab", experiment="nomem", repeat=2, experiment_overrides={"learning.enabled": False},
        )
        meta = load_search_meta(search_dir)
        assert (meta.study, meta.experiment, meta.repeat) == ("ab", "nomem", 2)
        assert meta.experiment_overrides == {"learning.enabled": False}
        assert meta.learning_enabled is False


    def test_legacy_search_meta_maps_experiment_and_arm(self):
        """A search.yaml written before the study vocabulary: its `experiment`
        was the study and its `arm` the experiment."""
        from hillclimb.harness.run import SearchMeta

        base = dict(search_id="p", run_id="r", problem="p", problem_id="p", agent="claude-code", model="sonnet", metric="score")
        old = SearchMeta.model_validate({
            **base, "experiment": "ab", "arm": "nomem", "repeat": 2, "arm_overrides": {"learning.enabled": False},
        })
        assert (old.study, old.experiment, old.repeat) == ("ab", "nomem", 2)
        assert old.experiment_overrides == {"learning.enabled": False}
        # an untagged old search wrote `arm: null`: still untagged, not a study
        plain = SearchMeta.model_validate({**base, "experiment": None, "arm": None, "arm_overrides": {}})
        assert (plain.study, plain.experiment) == (None, None)
        # a current record passes through untouched
        new = SearchMeta.model_validate({**base, "study": "ab", "experiment": "nomem"})
        assert (new.study, new.experiment) == ("ab", "nomem")


def row(experiment, repeat, score, *, problem="p", study="ab", state="done", lower=False, tokens=0, started=""):
    return StudyRow(
        study=study, experiment=experiment, repeat=repeat, problem_id=problem, problem_key=problem,
        run_id="r", search_id=f"{problem}-{experiment}-{repeat}", state=state, holdout=score, val=None,
        higher_is_better=not lower, started_at=started or f"t{repeat}-{experiment}", tokens=tokens,
    )


class TestSummary:
    def test_arms_wins_and_control_comparison(self):
        rows = [
            row("greedy", 1, 0.70, started="t1"), row("openevolve", 1, 0.80, started="t2"),
            row("greedy", 2, 0.90, started="t3"), row("openevolve", 2, 0.85, started="t4"),
            row("nomem", 1, 0.70, started="t5"),  # ties the control in repeat 1, no repeat 2
        ]
        summary = summarize(rows, noise_floor={"p": 0.02})[0]
        assert [a.experiment for a in summary.experiments] == ["greedy", "openevolve", "nomem"]  # control first
        by_experiment = {a.experiment: a for a in summary.experiments}
        assert by_experiment["greedy"].mean == pytest.approx(0.8) and by_experiment["greedy"].spread == pytest.approx(0.2)
        assert by_experiment["openevolve"].median == pytest.approx(0.825)
        # best-of-repeat: openevolve takes r1, greedy r2
        assert (by_experiment["greedy"].wins, by_experiment["openevolve"].wins, by_experiment["nomem"].wins) == (1, 1, 0)
        cmp = {c.experiment: c for c in summary.comparisons}
        assert cmp["openevolve"].gap == pytest.approx(0.025)  # paired: (+0.10 - 0.05) / 2
        assert (cmp["openevolve"].wins, cmp["openevolve"].losses) == (1, 1)
        assert cmp["openevolve"].within_noise is False
        assert cmp["nomem"].gap == pytest.approx(0.0) and cmp["nomem"].ties == 1
        assert cmp["nomem"].within_noise is True

    def test_control_can_be_named_and_direction_respected(self):
        rows = [row("a", 1, 0.03, lower=True), row("b", 1, 0.02, lower=True)]
        summary = summarize(rows, control="b")[0]
        assert summary.experiments[0].experiment == "b" and summary.experiments[0].wins == 1
        cmp = summary.comparisons[0]
        assert cmp.experiment == "a" and cmp.losses == 1 and cmp.within_noise is None

    def test_render(self):
        rows = [row("greedy", 1, 0.7, started="t1"), row("openevolve", 1, 0.8, started="t2", tokens=1_500_000)]
        text = render_report(summarize(rows, noise_floor={"p": 0.02}))
        assert "## ab · p (higher is better)" in text
        assert "| greedy (control) | 1 | 0.7 |" in text
        assert "1.5M |" in text
        assert "openevolve: +0.1 vs greedy — better beyond noise (0.02); wins 1, loses 0, ties 0" in text
        assert "no finished study" in render_report([])

    def test_unfinished_rows_are_listed_not_scored(self):
        rows = [row("a", 1, 0.5), row("b", 1, None, state="crashed")]
        summary = summarize(rows)[0]
        assert [r.experiment for r in summary.unfinished] == ["b"]
        assert "not finished: r/p-b-1 (b, crashed)" in render_report([summary])


class TestCollect:
    def _search(self, runs_dir, run_id, search_id, *, study, experiment, repeat, holdout, state="done"):
        run_dir = runs_dir / run_id
        if not (run_dir / "run.yaml").exists():
            create_run_dir(runs_dir, run_id)
            write_run_meta(run_dir, RunMeta(run_id=run_id, name=run_id, kind="experiment", target="p", problem_ids=["p"]))
        search_dir = create_search_dir(run_dir, search_id)
        write_search_meta(search_dir, SearchMeta(
            search_id=search_id, run_id=run_id, problem="p", problem_id="p", agent="dummy", model="-",
            metric="accuracy", study=study, experiment=experiment, repeat=repeat,
            started_at=f"2026-08-23T0{repeat}:00:00+00:00",
        ))
        journal = Journal(search_dir / "journal.jsonl")
        c = make_candidate("c001", operator="draft", status="passing", val_score=holdout or 0.1,
                           finished_at=f"2026-08-23T0{repeat}:10:00+00:00")
        c.agent = AgentInfo(name="dummy", total_tokens=1000)
        journal.candidate_result(c)
        write_status(search_dir, SearchStatus(
            search_id=search_id, run_id=run_id, state=state, pid=0,
            selected=ScoreRef(candidate_id="c001", val_score=holdout, holdout_score=holdout),
        ))

    def test_collects_tagged_finished_searches_only(self, tmp_path):
        runs = tmp_path / "runs"
        self._search(runs, "exp", "p", study="ab", experiment="a", repeat=1, holdout=0.7)
        self._search(runs, "exp", "p-2", study="ab", experiment="b", repeat=1, holdout=0.8)
        self._search(runs, "other", "p", study="cd", experiment="x", repeat=1, holdout=0.9)
        self._search(runs, "plain", "p", study=None, experiment=None, repeat=0, holdout=0.99)
        rows = collect_results(runs)
        assert {(r.study, r.experiment) for r in rows} == {("ab", "a"), ("ab", "b"), ("cd", "x")}
        rows = collect_results(runs, study="ab")
        assert [r.experiment for r in rows] == ["a", "b"]
        assert rows[0].tokens == 1000 and rows[0].candidates == 1
        assert rows[0].minutes_to_best == pytest.approx(10.0)
        text = render_report(summarize(rows))
        assert "b: +0.1 vs a — better (no noise floor known); wins 1, loses 0, ties 0" in text


def test_chart_labels_and_colours_experiment_curves_by_experiment():
    from hillclimb.tui.chart import EXPERIMENT_PALETTE, Curve, build_plot, curve_colors, curve_label
    from hillclimb.harness.store import SearchRecord

    def record(experiment, repeat, search_id):
        meta = SearchMeta(
            search_id=search_id, run_id="exp", problem="p", problem_id="p", agent="dummy", model="-",
            metric="score", study="ab" if experiment else None, experiment=experiment, repeat=repeat,
        )
        return SearchRecord(meta=meta, run_name="exp", state="done", search_dir=Path("x"), activity_at="")

    assert curve_label(record("greedy", 2, "p-3"), {"exp": 3}) == "greedy r2"
    assert curve_label(record(None, 0, "p-3"), {"exp": 3}) == "exp/p-3"
    curves = [
        Curve("greedy r1", "done", [0, 1], [1, 2], experiment="greedy"),
        Curve("openevolve r1", "done", [0, 1], [1, 3], experiment="openevolve"),
        Curve("greedy r2", "done", [0, 1], [1, 2.5], experiment="greedy"),
    ]
    colors = curve_colors(curves + [Curve("plain", "done", [0], [1])])
    assert colors[0] == colors[2] == EXPERIMENT_PALETTE[0] and colors[1] == EXPERIMENT_PALETTE[1]
    assert colors[3] is None  # not an experiment: plotui's own palette
    build_plot(curves)  # and the plot takes the explicit colours


class TestCli:
    def test_experiment_run_dry_run_and_parallel_launch(self, config, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from hillclimb.cli import app
        from tests.test_cli import write_problem

        root = tmp_path / "problems"
        write_problem(root, "p")
        config.paths.problems_dir = root
        hillclimb_dir = tmp_path / "hillclimb"
        spec = write_spec(hillclimb_dir / "experiments" / "ab.yaml", "problems: [p]\nrepeats: 1\nexperiments:\n  a: {search.policy: greedy}\n  b: {learning.enabled: false, search.policy_params: {k: 1}}\n")
        config.hillclimb_dir = hillclimb_dir
        monkeypatch.setattr("hillclimb.cli.common.load_config", lambda **kw: config)
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["experiment", "run", "ab", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "2 experiments × 1 problem(s) × 1 repeat(s) = 2 searches, sequential" in result.output
        assert "p · b · r1  learning.enabled=False, search.policy_params={'k': 1}" in result.output

        calls = []

        class DummyProc:
            pid = 7

        monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: calls.append(cmd) or DummyProc())
        result = runner.invoke(app, ["experiment", "run", str(spec), "--parallel", "--budget", "5m"])
        assert result.exit_code == 0, result.output
        assert len(calls) == 2
        argv = calls[1]
        assert argv[argv.index("--experiment") + 1] == "b"
        assert argv[argv.index("--study") + 1] == "ab"
        assert "--set" in argv and "learning.enabled=false" in argv and 'search.policy_params={"k": 1}' in argv
        assert argv[argv.index("--budget") + 1] == "5m"
        runs = list((config.paths.runs_dir).iterdir())
        assert len(runs) == 1 and yaml.safe_load((runs[0] / "run.yaml").read_text())["kind"] == "experiment"

    def test_shared_seed_resolves_and_reaches_every_child(self, config, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from hillclimb.cli import app
        from tests.test_cli import write_problem

        root = tmp_path / "problems"
        write_problem(root, "p")
        config.paths.problems_dir = root
        hillclimb_dir = tmp_path / "hillclimb"
        seed = hillclimb_dir / "experiments" / "seeds" / "p.py"
        seed.parent.mkdir(parents=True)
        seed.write_text("print('val_score: 0.5')\n")
        write_spec(
            hillclimb_dir / "experiments" / "ab.yaml",
            "problems: [p]\nrepeats: 1\nseed_from: seeds/p.py\n"
            "experiments:\n  a: {search.policy: greedy}\n  b: {learning.enabled: false}\n",
        )
        config.hillclimb_dir = hillclimb_dir
        monkeypatch.setattr("hillclimb.cli.common.load_config", lambda **kw: config)
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["experiment", "run", "ab", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert f"shared seed: {seed.resolve()}" in result.output
        assert "sha256" in result.output

        calls = []

        class DummyProc:
            pid = 7

        monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: calls.append(cmd) or DummyProc())
        result = runner.invoke(app, ["experiment", "run", "ab", "--parallel"])
        assert result.exit_code == 0, result.output
        assert len(calls) == 2
        for argv in calls:
            assert argv[argv.index("--seed-from") + 1] == str(seed.resolve())

    def test_missing_seed_fails_before_any_run(self, config, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from hillclimb.cli import app
        from tests.test_cli import write_problem

        write_problem(tmp_path / "problems", "p")
        config.paths.problems_dir = tmp_path / "problems"
        hillclimb_dir = tmp_path / "hillclimb"
        write_spec(
            hillclimb_dir / "experiments" / "ab.yaml",
            "problems: [p]\nseed_from: seeds/nope.py\nexperiments:\n  a: {}\n  b: {}\n",
        )
        config.hillclimb_dir = hillclimb_dir
        monkeypatch.setattr("hillclimb.cli.common.load_config", lambda **kw: config)
        result = CliRunner().invoke(app, ["experiment", "run", "ab", "--dry-run"])
        assert result.exit_code != 0
        assert "seed_from not found" in result.output
        assert not (config.paths.runs_dir).exists()

    def test_resolved_seed_paths(self, tmp_path):
        from hillclimb.experiment import StudySpec, resolved_seed

        spec = StudySpec(
            name="x", problems=["p"], experiments={"a": {}, "b": {}},
            seed_from=None, spec_path=tmp_path / "x.yaml",
        )
        assert resolved_seed(spec) is None
        seed = tmp_path / "seeds" / "s.py"
        seed.parent.mkdir()
        seed.write_text("pass\n")
        relative = spec.model_copy(update={"seed_from": "seeds/s.py"})
        assert resolved_seed(relative) == seed.resolve()
        absolute = spec.model_copy(update={"seed_from": str(seed)})
        assert resolved_seed(absolute) == seed.resolve()

    def test_run_set_and_arm_flags_reach_the_search(self, config, tmp_path, monkeypatch):
        from hillclimb.cli.run import _run_problem
        from hillclimb.harness.run import load_search_meta
        from tests.test_cli import write_problem

        problem = write_problem(tmp_path / "problems", "p")
        config.paths.problems_dir = tmp_path / "problems"
        seen = {}

        def fake_execute(config_arg, problem_arg, search_dir, budget, seed_from=None, knowledge_context=None):
            seen["config"] = config_arg
            seen["search_dir"] = search_dir

        monkeypatch.setattr("hillclimb.cli.run._execute", fake_execute)
        _run_problem(
            str(problem), config, budget="1m", study="ab", experiment="b", repeat=1,
            experiment_overrides=parse_set_overrides(["learning.enabled=false", "search.policy_params={num_drafts: 1}"]),
        )
        assert seen["config"].learning.enabled is False
        assert seen["config"].climber.selector_params == {"num_drafts": 1}
        meta = load_search_meta(seen["search_dir"])
        assert (meta.study, meta.experiment, meta.repeat, meta.learning_enabled) == ("ab", "b", 1, False)
        assert meta.experiment_overrides == {"learning.enabled": False, "search.policy_params": {"num_drafts": 1}}
        with pytest.raises(typer.BadParameter, match="unknown config setting"):
            _run_problem(str(problem), config, budget="1m", experiment_overrides={"search.nope": 1})





class TestBoundedLaunch:
    """`experiment run --max-concurrent N`: detached launches in job order,
    never more than N alive, exit codes read off the Popen objects."""

    @staticmethod
    def _setup(config, tmp_path, monkeypatch, spec_text):
        from tests.test_cli import write_problem

        root = tmp_path / "problems"
        write_problem(root, "p")
        config.paths.problems_dir = root
        hillclimb_dir = tmp_path / "hillclimb"
        spec = write_spec(hillclimb_dir / "experiments" / "ab.yaml", spec_text)
        config.hillclimb_dir = hillclimb_dir
        monkeypatch.setattr("hillclimb.cli.common.load_config", lambda **kw: config)
        monkeypatch.setattr("hillclimb.cli.experiment._REAP_POLL_S", 0)
        monkeypatch.chdir(tmp_path)
        return spec

    @staticmethod
    def _fake_popen(monkeypatch, exit_codes: dict[str, int] | None = None):
        """Every launched child exits on its first poll (the experiment's code from
        `exit_codes`, else 0); records the argv and how many were alive."""
        calls: list[list[str]] = []
        alive: list[int] = []
        peak = [0]

        class DummyProc:
            def __init__(self, cmd):
                self.pid = 100 + len(calls)
                self.cmd = cmd
                alive.append(self.pid)
                peak[0] = max(peak[0], len(alive))

            def poll(self):
                alive.remove(self.pid)
                experiment = self.cmd[self.cmd.index("--experiment") + 1]
                return (exit_codes or {}).get(experiment, 0)

        monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: calls.append(cmd) or DummyProc(cmd))
        return calls, peak

    def test_spec_max_concurrent_parsed(self, tmp_path):
        spec = load_study(write_spec(tmp_path / "x.yaml", "problems: [p]\nschedule: parallel\nmax_concurrent: 3\nexperiments: {a: {}, b: {}}\n"))
        assert spec.max_concurrent == 3
        with pytest.raises(ValueError, match="max_concurrent"):
            load_study(write_spec(tmp_path / "y.yaml", "problems: [p]\nmax_concurrent: 0\nexperiments: {a: {}, b: {}}\n"))

    def test_expand_numbers_repeats_from_first(self):
        from hillclimb.experiment import StudySpec

        spec = StudySpec(name="e", problems=["p"], experiments={"a": {}, "b": {}}, repeats=2)
        assert [(j.experiment, j.repeat) for j in expand(spec, first_repeat=3)] == [
            ("a", 3), ("b", 3), ("a", 4), ("b", 4),
        ]
        assert [j.index for j in expand(spec, first_repeat=3)] == [1, 2, 3, 4]
        with pytest.raises(ValueError):
            expand(spec, first_repeat=0)

    def test_bounded_launch_keeps_order_and_bound(self, config, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from hillclimb.cli import app

        spec = self._setup(
            config, tmp_path, monkeypatch,
            "problems: [p]\nrepeats: 2\nschedule: parallel\nmax_concurrent: 2\n"
            "experiments:\n  a: {search.policy: greedy}\n  b: {learning.enabled: false}\n  c: {}\n",
        )
        calls, peak = self._fake_popen(monkeypatch)
        result = CliRunner().invoke(app, ["experiment", "run", str(spec)])
        assert result.exit_code == 0, result.output
        assert "= 6 searches, parallel (at most 2 at once)" in result.output
        assert len(calls) == 6
        assert [(c[c.index("--experiment") + 1], c[c.index("--repeat") + 1]) for c in calls] == [
            ("a", "1"), ("b", "1"), ("c", "1"), ("a", "2"), ("b", "2"), ("c", "2"),
        ]
        assert peak[0] <= 2
        assert result.output.count("finished:") == 6
        assert "6 searches finished, 0 with a non-zero exit" in result.output
        runs = list(config.paths.runs_dir.iterdir())
        assert len(runs) == 1

    def test_flag_overrides_spec_and_exit_codes(self, config, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from hillclimb.cli import app

        spec = self._setup(
            config, tmp_path, monkeypatch,
            "problems: [p]\nrepeats: 1\nmax_concurrent: 4\nexperiments:\n  a: {}\n  b: {}\n  c: {}\n",
        )
        # spec max_concurrent without schedule: parallel is ignored (sequential)
        result = CliRunner().invoke(app, ["experiment", "run", str(spec), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "3 searches, sequential" in result.output and "at most" not in result.output
        # the flag implies parallel and overrides the spec's bound
        calls, peak = self._fake_popen(monkeypatch, exit_codes={"b": 2})
        result = CliRunner().invoke(app, ["experiment", "run", str(spec), "--max-concurrent", "1"])
        assert result.exit_code == 2, result.output  # a parked child, nothing failed
        assert "(at most 1 at once)" in result.output and peak[0] == 1
        assert "p-b-r1: parked" in result.output
        calls, peak = self._fake_popen(monkeypatch, exit_codes={"b": 2, "c": 1})
        result = CliRunner().invoke(app, ["experiment", "run", str(spec), "--max-concurrent", "3"])
        assert result.exit_code == 1, result.output
        assert peak[0] == 3

    def test_sequential_and_max_concurrent_contradict(self, config, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from hillclimb.cli import app

        spec = self._setup(config, tmp_path, monkeypatch, "problems: [p]\nexperiments:\n  a: {}\n  b: {}\n")
        result = CliRunner().invoke(app, ["experiment", "run", str(spec), "--sequential", "--max-concurrent", "2"])
        assert result.exit_code != 0
        assert "contradict" in result.output

    def test_run_id_appends_repeats_to_an_existing_run(self, config, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from hillclimb.cli import app

        spec = self._setup(
            config, tmp_path, monkeypatch,
            "problems: [p]\nrepeats: 1\nschedule: parallel\nmax_concurrent: 8\nexperiments:\n  a: {}\n  b: {}\n",
        )
        runner = CliRunner()
        calls, _ = self._fake_popen(monkeypatch)
        assert runner.invoke(app, ["experiment", "run", str(spec)]).exit_code == 0
        (run_dir,) = list(config.paths.runs_dir.iterdir())
        run_id = run_dir.name
        calls, _ = self._fake_popen(monkeypatch)
        result = runner.invoke(
            app, ["experiment", "run", str(spec), "--run-id", run_id, "--first-repeat", "2", "--repeats", "2"],
        )
        assert result.exit_code == 0, result.output
        assert "repeats 2..3 = 4 searches" in result.output
        assert f"Appending to run {run_id}" in result.output
        assert [c[c.index("--repeat") + 1] for c in calls] == ["2", "2", "3", "3"]
        assert all(c[c.index("--run-id") + 1] == run_id for c in calls)
        assert [d.name for d in config.paths.runs_dir.iterdir()] == [run_id]  # no new run
        result = runner.invoke(app, ["experiment", "run", str(spec), "--run-id", "nope"])
        assert result.exit_code != 0 and "not an existing study run" in result.output


def test_run_records_the_seed_and_its_hash(config, tmp_path):
    """`hillclimb run --seed-from` lands in search.yaml as the path and the
    sha256 of the file — the identity the run-scope similarity view checks."""
    import hashlib

    from hillclimb.api import create_run, create_search
    from hillclimb.problem import load_problem
    from hillclimb.harness.run import load_search_meta
    from tests.test_cli import write_problem

    root = tmp_path / "problems"
    write_problem(root, "p")
    config.paths.problems_dir = root
    seed = tmp_path / "seed.py"
    seed.write_text("print('seed')\n")
    run_dir = create_run(config, RunMeta(run_id="r1", name="r1", kind="experiment", target="x", problem_ids=["p"]))
    search_dir = create_search(config, load_problem("p", config), run_dir, "r1", 60, seed_from=seed)
    meta = load_search_meta(search_dir)
    assert meta.seed_from == str(seed)
    assert meta.seed_sha256 == hashlib.sha256(seed.read_bytes()).hexdigest()


class TestLegacyReplicateKeys:
    """`n_trials`/`trial_mode` predate the Trial (params) → Replicate (seed)
    split; old specs, `--set` lines and config files keep working."""

    def test_config_load_maps_old_keys(self):
        config = Config.model_validate({"search": {"n_trials": 2, "trial_mode": "serial"}})
        assert config.evaluation.n_replicates == 2
        assert config.evaluation.replicate_mode == "serial"

    def test_apply_overrides_maps_old_keys(self):
        config = Config()
        config.apply_overrides(parse_set_overrides(["search.n_trials=3", "search.trial_mode=serial"]))
        assert config.evaluation.n_replicates == 3
        assert config.evaluation.replicate_mode == "serial"


class TestSummaryJson:
    def test_summary_to_dict_carries_scores_gaps_and_verdicts(self):
        from hillclimb.experiment import summaries_to_dict

        rows = [
            row("greedy", 1, 0.70, started="t1"), row("openevolve", 1, 0.80, started="t2"),
            row("greedy", 2, 0.90, started="t3"), row("openevolve", 2, 0.85, started="t4"),
            row("nomem", 1, 0.70, started="t5"), row("late", 1, None, state="running", started="t6"),
        ]
        payload = summaries_to_dict(summarize(rows, noise_floor={"p": 0.02}))
        assert len(payload) == 1
        data = payload[0]
        assert (data["study"], data["problem_key"], data["noise_floor"]) == ("ab", "p", 0.02)
        experiments = {a["experiment"]: a for a in data["experiments"]}
        assert experiments["greedy"]["control"] is True and experiments["greedy"]["scores"] == [0.7, 0.9]
        assert experiments["greedy"]["searches"] == ["r/p-greedy-1", "r/p-greedy-2"]
        assert experiments["openevolve"]["control"] is False and experiments["openevolve"]["wins"] == 1
        cmp = {c["experiment"]: c for c in data["comparisons"]}
        assert cmp["openevolve"]["gap"] == pytest.approx(0.025)
        assert cmp["openevolve"]["verdict"] == "better"
        assert cmp["nomem"]["verdict"] == "within-noise"
        assert data["unfinished"] == [{"search": "r/p-late-1", "experiment": "late", "state": "running"}]
        json.dumps(payload)  # plain data throughout

    def test_verdicts_without_a_noise_floor_and_for_lower_is_better(self):
        from hillclimb.experiment import summary_to_dict

        rows = [row("a", 1, 0.03, lower=True), row("b", 1, 0.02, lower=True)]
        data = summary_to_dict(summarize(rows, control="b")[0])
        assert data["higher_is_better"] is False
        assert data["comparisons"][0]["verdict"] == "unknown"  # no floor: direction alone is not a result
        data = summary_to_dict(summarize(rows, control="b", noise_floor={"p": 0.001})[0])
        assert data["comparisons"][0]["verdict"] == "worse"
        assert summary_to_dict(summarize([row("a", 1, None, state="crashed")])[0])["comparisons"] == []


def test_every_experiment_spec_in_the_repo_builds_its_climbers():
    """The specs under experiments/ are what people copy: every experiment of
    every one must resolve to a climber whose loop builds — a param the
    policy does not have, or a selector setting misplaced, fails here rather
    than an hour into a study."""
    import pytest

    from hillclimb.climber import resolve_climber

    specs = sorted((Path(__file__).resolve().parents[1] / "experiments").glob("*.yaml"))
    assert specs
    for path in specs:
        study = load_study(path)
        for name in study.experiments:
            config = Config()
            config.apply_overrides(study.experiment_overrides(name))
            if config.climber.selector_policy == "map-elites":
                pytest.importorskip("openevolve")
            climber = resolve_climber(config.climber, path.parent)
            climber.operator_set()
            climber.build_loop(parallelism=config.concurrency.parallel_agents, log=lambda *_: None)

