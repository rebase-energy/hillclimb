"""Experiments: spec → jobs, arm overrides, tagged searches, arm comparison
and the report; the `experiment run` launcher and `run --set/--arm` plumbing."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
import yaml

from hillclimb.candidate import BackendInfo
from hillclimb.config import Config, parse_set_overrides
from hillclimb.dirs import create_run_dir, create_search_dir
from hillclimb.experiment import (
    ExperimentRow,
    collect_results,
    expand,
    flatten_overrides,
    load_experiment,
    render_report,
    resolve_experiment_path,
    summarize,
)
from hillclimb.journal import Journal
from hillclimb.run import RunMeta, SearchMeta, write_run_meta, write_search_meta
from hillclimb.status import ScoreRef, SearchStatus, write_status
from tests.test_watch import make_candidate

SPEC = """
problems: [circle-packing]
repeats: 2
budget: 10m
noise_floor: 0.02
defaults: {model: sonnet}
arms:
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
        spec = load_experiment(write_spec(tmp_path / "policy-vs-memory.yaml"))
        assert spec.name == "policy-vs-memory"  # the file stem
        assert spec.control == "greedy"
        assert spec.noise_for("circle-packing") == 0.02
        # nested and dotted forms are the same overrides; policy_params stays a mapping
        assert spec.arm_overrides("nomem") == {"model": "sonnet", "search.policy": "greedy", "learning.enabled": False}
        assert spec.arm_overrides("openevolve") == {
            "model": "sonnet", "search.policy": "openevolve", "search.policy_params": {"population_size": 50},
        }
        jobs = expand(spec)
        # repeat-major, arms round-robin inside: every arm's k-th repeat starts
        # from the same shared state
        assert [(j.index, j.arm, j.repeat) for j in jobs] == [
            (1, "greedy", 1), (2, "nomem", 1), (3, "openevolve", 1),
            (4, "greedy", 2), (5, "nomem", 2), (6, "openevolve", 2),
        ]

    def test_rejects_bad_specs(self, tmp_path):
        with pytest.raises(ValueError, match="at least two arms"):
            load_experiment(write_spec(tmp_path / "one.yaml", "problems: [p]\narms: {a: {}}\n"))
        with pytest.raises(ValueError, match="schedule"):
            load_experiment(write_spec(tmp_path / "s.yaml", "problems: [p]\nschedule: soon\narms: {a: {}, b: {}}\n"))

    def test_resolve_by_path_or_name(self, tmp_path):
        hillclimb_dir = tmp_path / "hillclimb"
        spec = write_spec(hillclimb_dir / "experiments" / "ab.yaml")
        assert resolve_experiment_path("ab", hillclimb_dir) == spec.resolve()
        assert resolve_experiment_path(str(spec), None) == spec.resolve()
        with pytest.raises(FileNotFoundError, match="hillclimb/experiments"):
            resolve_experiment_path("nope", hillclimb_dir)

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
        assert config.search.policy == "openevolve"
        assert config.learning.enabled is False
        assert config.search.n_replicates == 3
        assert config.search.policy_params == {"population_size": 50, "seed": 7}
        assert config.model == "opus"
        with pytest.raises(KeyError, match="search.nope"):
            config.apply_overrides({"search.nope": 1})
        with pytest.raises(ValueError, match="key=value"):
            parse_set_overrides(["garbage"])

    def test_create_search_records_the_tags(self, tmp_path):
        from hillclimb.api import create_search
        from hillclimb.problem import ProblemSpec
        from hillclimb.run import load_search_meta

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
            experiment="ab", arm="nomem", repeat=2, arm_overrides={"learning.enabled": False},
        )
        meta = load_search_meta(search_dir)
        assert (meta.experiment, meta.arm, meta.repeat) == ("ab", "nomem", 2)
        assert meta.arm_overrides == {"learning.enabled": False}
        assert meta.learning_enabled is False


def row(arm, repeat, score, *, problem="p", experiment="ab", state="done", lower=False, tokens=0, started=""):
    return ExperimentRow(
        experiment=experiment, arm=arm, repeat=repeat, problem_id=problem, problem_key=problem,
        run_id="r", search_id=f"{problem}-{arm}-{repeat}", state=state, holdout=score, val=None,
        higher_is_better=not lower, started_at=started or f"t{repeat}-{arm}", tokens=tokens,
    )


class TestSummary:
    def test_arms_wins_and_control_comparison(self):
        rows = [
            row("greedy", 1, 0.70, started="t1"), row("openevolve", 1, 0.80, started="t2"),
            row("greedy", 2, 0.90, started="t3"), row("openevolve", 2, 0.85, started="t4"),
            row("nomem", 1, 0.70, started="t5"),  # ties the control in repeat 1, no repeat 2
        ]
        summary = summarize(rows, noise_floor={"p": 0.02})[0]
        assert [a.arm for a in summary.arms] == ["greedy", "openevolve", "nomem"]  # control first
        by_arm = {a.arm: a for a in summary.arms}
        assert by_arm["greedy"].mean == pytest.approx(0.8) and by_arm["greedy"].spread == pytest.approx(0.2)
        assert by_arm["openevolve"].median == pytest.approx(0.825)
        # best-of-repeat: openevolve takes r1, greedy r2
        assert (by_arm["greedy"].wins, by_arm["openevolve"].wins, by_arm["nomem"].wins) == (1, 1, 0)
        cmp = {c.arm: c for c in summary.comparisons}
        assert cmp["openevolve"].gap == pytest.approx(0.025)  # paired: (+0.10 - 0.05) / 2
        assert (cmp["openevolve"].wins, cmp["openevolve"].losses) == (1, 1)
        assert cmp["openevolve"].within_noise is False
        assert cmp["nomem"].gap == pytest.approx(0.0) and cmp["nomem"].ties == 1
        assert cmp["nomem"].within_noise is True

    def test_control_can_be_named_and_direction_respected(self):
        rows = [row("a", 1, 0.03, lower=True), row("b", 1, 0.02, lower=True)]
        summary = summarize(rows, control="b")[0]
        assert summary.arms[0].arm == "b" and summary.arms[0].wins == 1
        cmp = summary.comparisons[0]
        assert cmp.arm == "a" and cmp.losses == 1 and cmp.within_noise is None

    def test_render(self):
        rows = [row("greedy", 1, 0.7, started="t1"), row("openevolve", 1, 0.8, started="t2", tokens=1_500_000)]
        text = render_report(summarize(rows, noise_floor={"p": 0.02}))
        assert "## ab · p (higher is better)" in text
        assert "| greedy (control) | 1 | 0.7 |" in text
        assert "1.5M |" in text
        assert "openevolve: +0.1 vs greedy — better beyond noise (0.02); wins 1, loses 0, ties 0" in text
        assert "no finished experiment" in render_report([])

    def test_unfinished_rows_are_listed_not_scored(self):
        rows = [row("a", 1, 0.5), row("b", 1, None, state="crashed")]
        summary = summarize(rows)[0]
        assert [r.arm for r in summary.unfinished] == ["b"]
        assert "not finished: r/p-b-1 (b, crashed)" in render_report([summary])


class TestCollect:
    def _search(self, runs_dir, run_id, search_id, *, experiment, arm, repeat, holdout, state="done"):
        run_dir = runs_dir / run_id
        if not (run_dir / "run.yaml").exists():
            create_run_dir(runs_dir, run_id)
            write_run_meta(run_dir, RunMeta(run_id=run_id, name=run_id, kind="experiment", target="p", problem_ids=["p"]))
        search_dir = create_search_dir(run_dir, search_id)
        write_search_meta(search_dir, SearchMeta(
            search_id=search_id, run_id=run_id, problem="p", problem_id="p", backend="dummy", model="-",
            metric="accuracy", experiment=experiment, arm=arm, repeat=repeat,
            started_at=f"2026-08-23T0{repeat}:00:00+00:00",
        ))
        journal = Journal(search_dir / "journal.jsonl")
        c = make_candidate("c001", operator="draft", status="ok", val_score=holdout or 0.1,
                           finished_at=f"2026-08-23T0{repeat}:10:00+00:00")
        c.backend = BackendInfo(name="dummy", total_tokens=1000)
        journal.candidate_result(c)
        write_status(search_dir, SearchStatus(
            search_id=search_id, run_id=run_id, state=state, pid=0,
            selected=ScoreRef(candidate_id="c001", val_score=holdout, holdout_score=holdout),
        ))

    def test_collects_tagged_finished_searches_only(self, tmp_path):
        runs = tmp_path / "runs"
        self._search(runs, "exp", "p", experiment="ab", arm="a", repeat=1, holdout=0.7)
        self._search(runs, "exp", "p-2", experiment="ab", arm="b", repeat=1, holdout=0.8)
        self._search(runs, "other", "p", experiment="cd", arm="x", repeat=1, holdout=0.9)
        self._search(runs, "plain", "p", experiment=None, arm=None, repeat=0, holdout=0.99)
        rows = collect_results(runs)
        assert {(r.experiment, r.arm) for r in rows} == {("ab", "a"), ("ab", "b"), ("cd", "x")}
        rows = collect_results(runs, experiment="ab")
        assert [r.arm for r in rows] == ["a", "b"]
        assert rows[0].tokens == 1000 and rows[0].candidates == 1
        assert rows[0].minutes_to_best == pytest.approx(10.0)
        text = render_report(summarize(rows))
        assert "b: +0.1 vs a — better (no noise floor known); wins 1, loses 0, ties 0" in text


def test_chart_labels_and_colours_experiment_curves_by_arm():
    from hillclimb.chart import ARM_PALETTE, Curve, build_plot, curve_colors, curve_label
    from hillclimb.store import SearchRecord

    def record(arm, repeat, search_id):
        meta = SearchMeta(
            search_id=search_id, run_id="exp", problem="p", problem_id="p", backend="dummy", model="-",
            metric="score", experiment="ab" if arm else None, arm=arm, repeat=repeat,
        )
        return SearchRecord(meta=meta, run_name="exp", state="done", search_dir=Path("x"), activity_at="")

    assert curve_label(record("greedy", 2, "p-3"), {"exp": 3}) == "greedy r2"
    assert curve_label(record(None, 0, "p-3"), {"exp": 3}) == "exp/p-3"
    curves = [
        Curve("greedy r1", "done", [0, 1], [1, 2], arm="greedy"),
        Curve("openevolve r1", "done", [0, 1], [1, 3], arm="openevolve"),
        Curve("greedy r2", "done", [0, 1], [1, 2.5], arm="greedy"),
    ]
    colors = curve_colors(curves + [Curve("plain", "done", [0], [1])])
    assert colors[0] == colors[2] == ARM_PALETTE[0] and colors[1] == ARM_PALETTE[1]
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
        spec = write_spec(hillclimb_dir / "experiments" / "ab.yaml", "problems: [p]\nrepeats: 1\narms:\n  a: {search.policy: greedy}\n  b: {learning.enabled: false, search.policy_params: {k: 1}}\n")
        config.hillclimb_dir = hillclimb_dir
        monkeypatch.setattr("hillclimb.cli.load_config", lambda **kw: config)
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["experiment", "run", "ab", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "2 arms × 1 problem(s) × 1 repeat(s) = 2 searches, sequential" in result.output
        assert "p · b · r1  learning.enabled=False, search.policy_params={'k': 1}" in result.output

        calls = []

        class DummyProc:
            pid = 7

        monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: calls.append(cmd) or DummyProc())
        result = runner.invoke(app, ["experiment", "run", str(spec), "--parallel", "--budget", "5m"])
        assert result.exit_code == 0, result.output
        assert len(calls) == 2
        argv = calls[1]
        assert argv[argv.index("--arm") + 1] == "b"
        assert argv[argv.index("--experiment") + 1] == "ab"
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
            "arms:\n  a: {search.policy: greedy}\n  b: {learning.enabled: false}\n",
        )
        config.hillclimb_dir = hillclimb_dir
        monkeypatch.setattr("hillclimb.cli.load_config", lambda **kw: config)
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
            "problems: [p]\nseed_from: seeds/nope.py\narms:\n  a: {}\n  b: {}\n",
        )
        config.hillclimb_dir = hillclimb_dir
        monkeypatch.setattr("hillclimb.cli.load_config", lambda **kw: config)
        result = CliRunner().invoke(app, ["experiment", "run", "ab", "--dry-run"])
        assert result.exit_code != 0
        assert "seed_from not found" in result.output
        assert not (config.paths.runs_dir).exists()

    def test_resolved_seed_paths(self, tmp_path):
        from hillclimb.experiment import ExperimentSpec, resolved_seed

        spec = ExperimentSpec(
            name="x", problems=["p"], arms={"a": {}, "b": {}},
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
        from hillclimb.cli import _run_problem
        from hillclimb.run import load_search_meta
        from tests.test_cli import write_problem

        problem = write_problem(tmp_path / "problems", "p")
        config.paths.problems_dir = tmp_path / "problems"
        seen = {}

        def fake_execute(config_arg, problem_arg, search_dir, budget, seed_from=None, knowledge_context=None):
            seen["config"] = config_arg
            seen["search_dir"] = search_dir

        monkeypatch.setattr("hillclimb.cli._execute", fake_execute)
        _run_problem(
            str(problem), config, budget="1m", experiment="ab", arm="b", repeat=1,
            arm_overrides=parse_set_overrides(["learning.enabled=false", "search.policy_params={k: 1}"]),
        )
        assert seen["config"].learning.enabled is False
        assert seen["config"].search.policy_params == {"k": 1}
        meta = load_search_meta(seen["search_dir"])
        assert (meta.experiment, meta.arm, meta.repeat, meta.learning_enabled) == ("ab", "b", 1, False)
        assert meta.arm_overrides == {"learning.enabled": False, "search.policy_params": {"k": 1}}
        with pytest.raises(typer.BadParameter, match="unknown config setting"):
            _run_problem(str(problem), config, budget="1m", arm_overrides={"search.nope": 1})





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
        monkeypatch.setattr("hillclimb.cli.load_config", lambda **kw: config)
        monkeypatch.setattr("hillclimb.cli._REAP_POLL_S", 0)
        monkeypatch.chdir(tmp_path)
        return spec

    @staticmethod
    def _fake_popen(monkeypatch, exit_codes: dict[str, int] | None = None):
        """Every launched child exits on its first poll (the arm's code from
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
                arm = self.cmd[self.cmd.index("--arm") + 1]
                return (exit_codes or {}).get(arm, 0)

        monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: calls.append(cmd) or DummyProc(cmd))
        return calls, peak

    def test_spec_max_concurrent_parsed(self, tmp_path):
        spec = load_experiment(write_spec(tmp_path / "x.yaml", "problems: [p]\nschedule: parallel\nmax_concurrent: 3\narms: {a: {}, b: {}}\n"))
        assert spec.max_concurrent == 3
        with pytest.raises(ValueError, match="max_concurrent"):
            load_experiment(write_spec(tmp_path / "y.yaml", "problems: [p]\nmax_concurrent: 0\narms: {a: {}, b: {}}\n"))

    def test_expand_numbers_repeats_from_first(self):
        from hillclimb.experiment import ExperimentSpec

        spec = ExperimentSpec(name="e", problems=["p"], arms={"a": {}, "b": {}}, repeats=2)
        assert [(j.arm, j.repeat) for j in expand(spec, first_repeat=3)] == [
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
            "arms:\n  a: {search.policy: greedy}\n  b: {learning.enabled: false}\n  c: {}\n",
        )
        calls, peak = self._fake_popen(monkeypatch)
        result = CliRunner().invoke(app, ["experiment", "run", str(spec)])
        assert result.exit_code == 0, result.output
        assert "= 6 searches, parallel (at most 2 at once)" in result.output
        assert len(calls) == 6
        assert [(c[c.index("--arm") + 1], c[c.index("--repeat") + 1]) for c in calls] == [
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
            "problems: [p]\nrepeats: 1\nmax_concurrent: 4\narms:\n  a: {}\n  b: {}\n  c: {}\n",
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

        spec = self._setup(config, tmp_path, monkeypatch, "problems: [p]\narms:\n  a: {}\n  b: {}\n")
        result = CliRunner().invoke(app, ["experiment", "run", str(spec), "--sequential", "--max-concurrent", "2"])
        assert result.exit_code != 0
        assert "contradict" in result.output

    def test_run_id_appends_repeats_to_an_existing_run(self, config, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from hillclimb.cli import app

        spec = self._setup(
            config, tmp_path, monkeypatch,
            "problems: [p]\nrepeats: 1\nschedule: parallel\nmax_concurrent: 8\narms:\n  a: {}\n  b: {}\n",
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
        assert result.exit_code != 0 and "not an existing experiment run" in result.output


def test_run_records_the_seed_and_its_hash(config, tmp_path):
    """`hillclimb run --seed-from` lands in search.yaml as the path and the
    sha256 of the file — the identity the run-scope similarity view checks."""
    import hashlib

    from hillclimb.api import create_run, create_search
    from hillclimb.problem import load_problem
    from hillclimb.run import load_search_meta
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
        from hillclimb.config import SearchConfig

        search = SearchConfig.model_validate({"n_trials": 2, "trial_mode": "serial"})
        assert search.n_replicates == 2
        assert search.replicate_mode == "serial"

    def test_apply_overrides_maps_old_keys(self):
        config = Config()
        config.apply_overrides(parse_set_overrides(["search.n_trials=3", "search.trial_mode=serial"]))
        assert config.search.n_replicates == 3
        assert config.search.replicate_mode == "serial"
