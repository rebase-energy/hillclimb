"""Similarity additions: seed-first reference, the problem-supplied
fingerprint mode, the run-scope cube (every arm of a problem in an
experiment run), its rendering, and the CLI's auto-detection."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from hillclimb.harness.candidate import Candidate
from hillclimb.harness.journal import Journal
from hillclimb.problem import load_problem
from hillclimb.harness.run import RunMeta, SearchMeta, write_run_meta, write_search_meta
from hillclimb.tui.similarity import (
    FingerprintError,
    SearchInput,
    _FINGERPRINTS,
    _TOKENS,
    build_run_similarity,
    build_similarity,
    clear_caches,
    load_fingerprinter,
)
from tests.test_similarity import cand, sub_csv, write_candidate

SORTED_FINGERPRINT = (
    "import pandas as pd\n"
    "def fingerprint(candidate_dir):\n"
    "    frame = pd.read_csv(candidate_dir / 'submission.csv')\n"
    "    return sorted(frame['pred'].tolist())\n"
)


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_caches()
    yield
    clear_caches()


def write_fingerprint(tmp_path: Path, source: str = SORTED_FINGERPRINT) -> Path:
    path = tmp_path / "problem" / "fingerprint.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


# ---------------------------------------------------------------------------
# reference resolution


class TestReference:
    def test_seed_is_preferred_over_the_baseline(self, tmp_path):
        candidates = [
            cand("c000", "baseline", score=0.1, t=0),
            cand("c001", "seed", score=0.5, t=1),
            cand("c002", parent="c001", score=0.7, t=2),
        ]
        for i, cid in enumerate(["c000", "c001", "c002"]):
            write_candidate(tmp_path, cid, solution=f"x = {i}\n", submission=sub_csv([float(i), 1.0]))
        view = build_similarity(candidates, tmp_path, True)
        assert view.reference_id == "c001" and view.reference_label == "seed"
        assert {n.id: n.raw[2] for n in view.nodes}["c002"] == 1.0

    def test_baseline_without_source_is_fine_when_a_seed_exists(self, tmp_path):
        """A baseline_files problem's c000 ships only the copied submission —
        the heilbronn case that used to make the view unavailable."""
        candidates = [
            cand("c000", "baseline", t=0),
            cand("c001", "seed", score=0.5, t=1),
            cand("c002", parent="c001", score=0.7, t=2),
        ]
        (tmp_path / "candidates" / "c000").mkdir(parents=True)
        (tmp_path / "candidates" / "c000" / "submission.csv").write_text(sub_csv([0.0, 1.0]))
        write_candidate(tmp_path, "c001", solution="x = 1\n", submission=sub_csv([1.0, 2.0]))
        write_candidate(tmp_path, "c002", solution="x = 2\n", submission=sub_csv([2.0, 3.0]))
        view = build_similarity(candidates, tmp_path, True)
        assert view.unavailable is None and view.reference_id == "c001"
        assert view.n_unpositioned == 1  # c000: no solution.py to measure

    def test_no_seed_falls_back_to_baseline_then_earliest(self, tmp_path):
        candidates = [cand("c000", "draft", score=0.1, t=0), cand("c001", "draft", score=0.2, t=1)]
        for cid in ("c000", "c001"):
            write_candidate(tmp_path, cid, submission=sub_csv([1.0, 2.0]))
        view = build_similarity(candidates, tmp_path, True)
        assert view.reference_id == "c000" and view.reference_label == "earliest"


# ---------------------------------------------------------------------------
# fingerprint mode


class TestFingerprintMode:
    def make(self, tmp_path, values_by_cid: dict[str, list[float]]):
        candidates = []
        for t, (cid, values) in enumerate(values_by_cid.items()):
            candidates.append(cand(cid, "baseline" if t == 0 else "improve",
                                   parent=None if t == 0 else "c000", score=0.5, t=t))
            write_candidate(tmp_path, cid, solution=f"x = {t}\n", submission=sub_csv(values))
        return candidates

    def test_permuted_rows_are_far_in_submission_mode_and_at_zero_in_fingerprint_mode(self, tmp_path):
        candidates = self.make(tmp_path, {"c000": [1.0, 2.0, 3.0], "c001": [3.0, 2.0, 1.0]})
        plain = build_similarity(candidates, tmp_path, True)
        assert plain.mode == "submission"
        assert {n.id: n.raw[0] for n in plain.nodes}["c001"] > 0
        view = build_similarity(candidates, tmp_path, True, fingerprint_path=write_fingerprint(tmp_path))
        assert view.mode == "fingerprint"
        assert {n.id: n.raw[0] for n in view.nodes}["c001"] == 0.0

    def test_declining_or_raising_leaves_the_candidate_unpositioned(self, tmp_path):
        candidates = self.make(tmp_path, {"c000": [1.0, 2.0], "c001": [1.0, 2.0], "c002": [9.0, 9.0]})
        source = (
            "import pandas as pd\n"
            "def fingerprint(d):\n"
            "    values = pd.read_csv(d / 'submission.csv')['pred'].tolist()\n"
            "    if values[0] == 9.0:\n        raise RuntimeError('boom')\n"
            "    return values\n"
        )
        view = build_similarity(candidates, tmp_path, True, fingerprint_path=write_fingerprint(tmp_path, source))
        assert view.unavailable is None and view.mode == "fingerprint"
        assert {n.id for n in view.nodes} == {"c000", "c001"} and view.n_unpositioned == 1

    def test_length_mismatch_is_unpositioned(self, tmp_path):
        candidates = self.make(tmp_path, {"c000": [1.0, 2.0], "c001": [1.0, 2.0, 3.0]})
        view = build_similarity(candidates, tmp_path, True, fingerprint_path=write_fingerprint(tmp_path))
        assert [n.id for n in view.nodes] == ["c000"] and view.n_unpositioned == 1

    def test_reference_declining_falls_through_to_submission_mode(self, tmp_path):
        candidates = self.make(tmp_path, {"c000": [1.0, 2.0], "c001": [2.0, 3.0]})
        path = write_fingerprint(tmp_path, "def fingerprint(d):\n    return None\n")
        view = build_similarity(candidates, tmp_path, True, fingerprint_path=path)
        assert view.mode == "submission" and len(view.nodes) == 2

    def test_bad_module_makes_the_view_unavailable_and_names_the_file(self, tmp_path):
        candidates = self.make(tmp_path, {"c000": [1.0, 2.0]})
        path = write_fingerprint(tmp_path, "def other():\n    pass\n")
        view = build_similarity(candidates, tmp_path, True, fingerprint_path=path)
        assert view.unavailable is not None and "fingerprint(candidate_dir)" in view.unavailable
        assert str(path) in view.unavailable
        path.write_text("raise ValueError('nope')\n")
        with pytest.raises(FingerprintError, match="failed to import"):
            load_fingerprinter(path)
        with pytest.raises(FingerprintError, match="unreadable"):
            load_fingerprinter(tmp_path / "missing.py")

    def test_caches_follow_the_artifact_and_the_module(self, tmp_path):
        candidates = self.make(tmp_path, {"c000": [1.0, 2.0, 3.0], "c001": [3.0, 2.0, 1.0]})
        path = write_fingerprint(tmp_path)
        build_similarity(candidates, tmp_path, True, fingerprint_path=path)
        assert len(_FINGERPRINTS) == 2
        # a rewritten artifact re-fingerprints that candidate
        target = tmp_path / "candidates" / "c001" / "submission.csv"
        target.write_text(sub_csv([10.0, 20.0, 30.0]))
        os.utime(target, (target.stat().st_mtime + 5, target.stat().st_mtime + 5))
        view = build_similarity(candidates, tmp_path, True, fingerprint_path=path)
        assert {n.id: n.raw[0] for n in view.nodes}["c001"] > 0
        # an edited module drops every vector, since they came from old code
        path.write_text("def fingerprint(d):\n    return [1.0, 2.0]\n")
        os.utime(path, (path.stat().st_mtime + 5, path.stat().st_mtime + 5))
        load_fingerprinter(path)
        assert not _FINGERPRINTS


class TestProblemPickup:
    def test_fingerprint_is_picked_up_by_default(self, tmp_path, config):
        from tests.test_surface import write_problem

        problem_dir = write_problem(tmp_path)
        assert load_problem(problem_dir, config).fingerprint_path is None
        (problem_dir / "fingerprint.py").write_text(SORTED_FINGERPRINT)
        assert load_problem(problem_dir, config).fingerprint_path == (problem_dir / "fingerprint.py").resolve()

    def test_explicit_missing_module_is_an_error(self, tmp_path, config):
        from tests.test_surface import write_problem

        problem_dir = write_problem(tmp_path, extra_yaml="fingerprint: nope.py\n")
        with pytest.raises(FileNotFoundError):
            load_problem(problem_dir, config)


# ---------------------------------------------------------------------------
# run scope


def make_arm(
    root: Path, search_id: str, arm: str, *, seed_source: str = "s = 1\n",
    seed_values=(1.0, 2.0, 3.0), child_scores=(0.6, 0.8), child_offsets=(1.0, 2.0),
    seed_submission: bool = True,
) -> SearchInput:
    """One experiment-arm search: c000 baseline (copied submission, no
    source), c001 seed, then children of the seed."""
    search_dir = root / search_id
    (search_dir / "candidates" / "c000").mkdir(parents=True, exist_ok=True)
    (search_dir / "candidates" / "c000" / "submission.csv").write_text(sub_csv(list(seed_values)))
    candidates = [
        cand("c000", "baseline", t=0),
        cand("c001", "seed", score=0.5, t=1),
    ]
    write_candidate(search_dir, "c001", solution=seed_source,
                    submission=sub_csv(list(seed_values)) if seed_submission else None)
    for i, (score, offset) in enumerate(zip(child_scores, child_offsets), start=2):
        cid = f"c{i:03d}"
        candidates.append(cand(cid, parent="c001", score=score, t=i))
        write_candidate(search_dir, cid, solution=f"s = {i}\nprint({i})\n",
                        submission=sub_csv([v + offset for v in seed_values]))
    return SearchInput(search_id=search_id, search_dir=search_dir, candidates=candidates, arm=arm)


class TestRunView:
    def make_run(self, tmp_path) -> list[SearchInput]:
        return [
            make_arm(tmp_path, "p", "greedy", child_scores=(0.6, 0.8)),
            make_arm(tmp_path, "p-2", "openevolve", child_scores=(0.7, 0.95)),
            make_arm(tmp_path, "p-3", "greedy", child_scores=(0.65, 0.85)),
            make_arm(tmp_path, "p-4", "openevolve", child_scores=(0.55, 0.9)),
        ]

    def test_every_arm_measured_from_its_own_seed(self, tmp_path):
        view = build_run_similarity(self.make_run(tmp_path), True)
        assert view.unavailable is None and view.scope == "run"
        assert view.arms == ("greedy", "openevolve") and view.n_searches == 4
        assert view.reference_label == "seed" and view.mode == "submission"
        assert set(view.reference_ids) == {"p/c001", "p-2/c001", "p-3/c001", "p-4/c001"}
        by_id = {n.id: n for n in view.nodes}
        for ref in view.reference_ids:
            assert by_id[ref].raw == (0.0, 0.0, 0.0)
        assert by_id["p-2/c003"].raw[2] == 1.0 and by_id["p-2/c003"].arm == "openevolve"
        assert by_id["p-2/c003"].search_id == "p-2"
        assert view.n_unpositioned == 4  # the four sourceless baselines

    def test_bins_and_champion_span_the_union(self, tmp_path):
        view = build_run_similarity(self.make_run(tmp_path), True)
        by_id = {n.id: n for n in view.nodes}
        assert [n.id for n in view.nodes if n.best] == ["p-2/c003"]  # 0.95, the run's best
        assert by_id["p-2/c003"].bin == max(n.bin for n in view.nodes if n.bin is not None)
        assert by_id["p-4/c002"].bin <= by_id["p/c002"].bin  # 0.55 ranks at or below 0.6 across searches
        assert by_id["p/c001"].bin < by_id["p-2/c003"].bin  # a seed (0.5) well below the run's best

    def test_champion_reference_keeps_lineage_from_own_seed(self, tmp_path):
        view = build_run_similarity(self.make_run(tmp_path), True, reference="champion")
        assert view.unavailable is None and view.reference_ids == ("p-2/c003",)
        assert view.lineage_note == "lineage from own seed"
        by_id = {n.id: n for n in view.nodes}
        assert by_id["p-2/c003"].raw[:2] == (0.0, 0.0) and by_id["p-2/c003"].raw[2] == 1.0
        assert by_id["p/c001"].raw[2] == 0.0 and by_id["p/c001"].raw[0] > 0  # a seed: far from the champion

    def test_refusals_name_the_search(self, tmp_path):
        searches = self.make_run(tmp_path)
        other = make_arm(tmp_path, "q", "gepa", seed_source="s = 2\n")
        view = build_run_similarity([*searches, other], True)
        assert view.unavailable is not None and "seeds differ: p vs q" in view.unavailable
        seedless = SearchInput("r", tmp_path / "r", [cand("c000", "baseline", score=0.1)], arm="x")
        write_candidate(tmp_path / "r", "c000", submission=sub_csv([1.0]))
        view = build_run_similarity([*searches, seedless], True)
        assert "r has no seed candidate" in (view.unavailable or "")
        assert build_run_similarity([], True, problem_key="p").unavailable == "no searches for p in the run"

    def test_mode_mixing_is_refused(self, tmp_path):
        searches = self.make_run(tmp_path)
        report_only = make_arm(tmp_path, "s", "gepa", seed_submission=False)
        # a seed with no submission but a report would measure in report mode
        seed = report_only.candidates[1]
        seed.trials[0].replicates[0].report = {"horizons": [{"bucket": f"h{i}", "score": 1.0} for i in range(3)]}
        view = build_run_similarity([*searches, report_only], True)
        assert view.unavailable is not None and "modes differ" in view.unavailable

    def test_prune_keeps_every_searchs_entries(self, tmp_path):
        searches = self.make_run(tmp_path)
        build_run_similarity(searches, True)
        assert len(_TOKENS) == 12  # 3 sourced candidates x 4 searches survive the prune
        build_run_similarity(searches, True)
        assert len(_TOKENS) == 12

    def test_fingerprint_mode_in_run_scope(self, tmp_path):
        searches = self.make_run(tmp_path)
        view = build_run_similarity(searches, True, fingerprint_path=write_fingerprint(tmp_path))
        assert view.mode == "fingerprint" and len(view.nodes) == 12


# ---------------------------------------------------------------------------
# rendering


class TestRendering:
    def test_run_plot_groups_by_arm_and_rank(self, tmp_path):
        from hillclimb.tui.chart import ARM_PALETTE
        from hillclimb.tui.similarityview import (
            BEST_RGB, REFERENCE_RGB, arm_colour, build_similarity_plot, rank_size, statusline,
        )

        view = build_run_similarity(TestRunView().make_run(tmp_path), True)
        assert arm_colour(view, "greedy") == ARM_PALETTE[0]
        assert arm_colour(view, "openevolve") == ARM_PALETTE[1]
        assert arm_colour(view, "nope") != ARM_PALETTE[0]
        assert rank_size(None) < rank_size(0) < rank_size(5) < 7.0
        plot = build_similarity_plot(view)
        assert plot is not None
        line = statusline("run p", "done", view, "score", True, position=(0, 3))
        assert "greedy" in line and "openevolve" in line and "(1/3)" in line
        assert "12 placed over 4 searches" in line and "vs [bold]seed[/] (submission)" in line
        champion = build_run_similarity(TestRunView().make_run(tmp_path), True, reference="champion")
        assert "lineage from own seed" in statusline("run p", "done", champion, "score", True)
        assert REFERENCE_RGB != BEST_RGB

    def test_search_statusline_names_the_resolved_reference(self, tmp_path):
        from hillclimb.tui.similarityview import statusline

        candidates = [cand("c000", "seed", score=0.5, t=0), cand("c001", parent="c000", score=0.7, t=1)]
        for cid in ("c000", "c001"):
            write_candidate(tmp_path, cid, submission=sub_csv([1.0, 2.0]))
        view = build_similarity(candidates, tmp_path, True)
        assert "vs [bold]seed[/] c000 (submission)" in statusline("r", "done", view, "score", True)


# ---------------------------------------------------------------------------
# CLI auto-detection


def _experiment_search(runs_dir: Path, run_id: str, search_id: str, arm: str, *, seed: bool) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_meta(run_dir, RunMeta(run_id=run_id, name="exp", kind="experiment", target="exp", problem_ids=["p"]))
    search_dir = run_dir / "searches" / search_id
    search_dir.mkdir(parents=True)
    write_search_meta(search_dir, SearchMeta(
        search_id=search_id, run_id=run_id, problem="p", problem_id="p", problem_key="p",
        backend="dummy", model="", metric="score", higher_is_better=True,
        experiment="exp", arm=arm, repeat=1, started_at=f"2026-09-04T10:0{len(search_id)}:00+00:00",
    ))
    journal = Journal(search_dir / "journal.jsonl")
    candidates = [cand("c000", "baseline", score=0.1, t=0)]
    write_candidate(search_dir, "c000", submission=sub_csv([1.0, 2.0]))
    if seed:
        candidates.append(cand("c001", "seed", score=0.5, t=1))
        write_candidate(search_dir, "c001", solution="s = 1\n", submission=sub_csv([1.0, 2.0]))
    for candidate in candidates:
        journal.candidate_result(candidate)


class TestCliAutoDetect:
    @staticmethod
    def _capture(monkeypatch):
        launched = []

        class FakeApp:
            def __init__(self, config=None, search=None, reference="baseline", run=None,
                         view="reference", metric="behavioral"):
                launched.append({"search": search, "reference": reference, "run": run, "view": view})

            def run(self):
                pass

        monkeypatch.setattr("hillclimb.tui.similarityview.SimilarityApp", FakeApp)
        return launched

    def test_experiment_arm_opens_the_run_view(self, config, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from hillclimb.cli import app

        for search_id, arm in (("p", "greedy"), ("p-2", "gepa")):
            _experiment_search(config.paths.runs_dir, "r1", search_id, arm, seed=True)
        monkeypatch.setattr("hillclimb.cli.load_config", lambda **kw: config)
        launched = self._capture(monkeypatch)
        result = CliRunner().invoke(app, ["similarity", "r1/p"])
        assert result.exit_code == 0, result.output
        assert launched == [{"search": None, "reference": "seed", "run": ("r1", "p"), "view": "map"}]
        # a bare run id with several searches anchors on the latest, no "pick one"
        result = CliRunner().invoke(app, ["similarity", "r1"])
        assert result.exit_code == 0, result.output
        assert launched[-1]["run"] == ("r1", "p")
        result = CliRunner().invoke(app, ["similarity", "map", "r1/p", "--single"])
        assert result.exit_code == 0, result.output
        assert launched[-1] == {"search": "r1/p", "reference": "baseline", "run": None, "view": "map"}
        # the reference cube is its own subcommand, same auto-detection
        result = CliRunner().invoke(app, ["similarity", "reference", "r1/p"])
        assert result.exit_code == 0, result.output
        assert launched[-1] == {"search": None, "reference": "seed", "run": ("r1", "p"), "view": "reference"}
        result = CliRunner().invoke(app, ["similarity", "reference", "r1/p", "--single"])
        assert launched[-1] == {"search": "r1/p", "reference": "baseline", "run": None, "view": "reference"}

    def test_seedless_experiment_falls_back_to_the_single_view(self, config, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from hillclimb.cli import app

        for search_id, arm in (("p", "greedy"), ("p-2", "gepa")):
            _experiment_search(config.paths.runs_dir, "r1", search_id, arm, seed=False)
        monkeypatch.setattr("hillclimb.cli.load_config", lambda **kw: config)
        launched = self._capture(monkeypatch)
        for argv in (["similarity", "r1/p-2"], ["similarity", "reference", "r1/p-2"]):
            result = CliRunner().invoke(app, argv)
            assert result.exit_code == 0, result.output
            assert "run view unavailable (p has no seed candidate" in result.output
        assert [(l["search"], l["reference"], l["run"]) for l in launched] == [("r1/p-2", "baseline", None)] * 2
        assert [l["view"] for l in launched] == ["map", "reference"]
