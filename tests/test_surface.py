"""The surface view's pure layer: journal positions × landscape terrain."""

from __future__ import annotations

from tests.factories import trial as mk_trial

from pathlib import Path

import pytest

from hillclimb.candidate import Candidate
from hillclimb.problem import load_problem
from hillclimb.surface import LIFT, LandscapeError, build_surface, load_landscape
from hillclimb.tree import build_tree

# A tiny analytic terrain: one bowl-shaped hill, summit at the origin with
# height 4.0. grid(5) spans [-2, 2] so the argmax cell is exactly (0, 0).
LANDSCAPE = """\
def elevation(x, y):
    try:
        import numpy as np
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    except ImportError:
        pass
    return 4.0 - (x ** 2 + y ** 2)


def grid(n=5):
    import numpy as np
    xs = np.linspace(-2.0, 2.0, n)
    ys = np.linspace(-2.0, 2.0, n)
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, elevation(X, Y)
"""


@pytest.fixture
def landscape(tmp_path: Path):
    path = tmp_path / "landscape.py"
    path.write_text(LANDSCAPE)
    return load_landscape(path)


def cand(
    cid: str, operator: str = "improve", parent: str | None = None, score: float | None = None,
    pos: tuple[float, float] | None = None, status: str = "passing", t: int = 0, **kwargs,
) -> Candidate:
    trials = []
    if score is not None:
        metrics = {"x": pos[0], "y": pos[1]} if pos is not None else {}
        trials = [mk_trial(val_score=score, metrics=metrics)]
    return Candidate(
        candidate_id=cid, operator=operator, parent_id=parent, status=status, trials=trials,
        created_at=f"2026-08-28T10:{t:02d}:00+00:00",
        finished_at=f"2026-08-28T10:{t + 1:02d}:00+00:00",
        **kwargs,
    )


def climb() -> list[Candidate]:
    """draft at the rim, two improves walking up the hill, one failed probe,
    one scored candidate with no position metrics (declared-score baseline)."""
    return [
        cand("c000", "baseline", score=0.0, t=0),  # no position journaled
        cand("c001", "draft", score=1.0, pos=(2.0, 0.0), t=1),
        cand("c002", parent="c001", score=3.0, pos=(1.0, 0.0), t=2),
        cand("c003", parent="c002", score=4.0, pos=(0.0, 0.0), t=3),
        cand("c004", parent="c002", status="buggy", t=4),
    ]


class TestBuildSurface:
    def test_nodes_sit_on_the_terrain(self, landscape):
        view = build_surface(climb(), True, landscape)
        lift = LIFT * 8.0  # grid range is [-4, 4]
        by_id = {n.id: n for n in view.nodes}
        assert set(by_id) == {"c001", "c002", "c003"}  # only positioned candidates
        assert by_id["c003"].z == pytest.approx(4.0 + lift)
        assert by_id["c001"].z == pytest.approx(0.0 + lift)
        assert view.n_unpositioned == 2  # the baseline and the buggy probe

    def test_fates_and_path_match_the_tree(self, landscape):
        candidates = climb()
        view = build_surface(candidates, True, landscape)
        tree_fates = {n.id: n.fate for n in build_tree(candidates, True).nodes}
        for node in view.nodes:
            assert node.fate == tree_fates[node.id]
        assert {n.id for n in view.nodes if n.on_path} == {"c001", "c002", "c003"}

    def test_position_is_the_median_over_trials(self, landscape):
        noisy = cand("c001", "draft", score=1.0, pos=(9.0, 9.0), t=1)
        noisy = noisy.model_copy(
            update={
                "trials": [
                    mk_trial(val_score=1.0, metrics={"x": 0.0, "y": 1.0}),
                    mk_trial(val_score=1.1, metrics={"x": 2.0, "y": 1.0}),
                    mk_trial(val_score=0.9, metrics={"x": 1.0, "y": 1.0}),
                ]
            }
        )
        view = build_surface([noisy], True, landscape)
        assert (view.nodes[0].x, view.nodes[0].y) == (1.0, 1.0)

    def test_lineage_is_draped_on_the_terrain(self, landscape):
        view = build_surface(climb(), True, landscape)
        lift = LIFT * 8.0
        assert len(view.lineage.xs) > 3  # subdivided, not two straight hops
        for x, y, z in zip(view.lineage.xs, view.lineage.ys, view.lineage.zs):
            assert z == pytest.approx(4.0 - (x**2 + y**2) + lift)

    def test_unpositioned_accepted_candidates_break_no_hops(self, landscape):
        # the baseline is accepted first but has no position: the drape must
        # start at the first positioned accepted candidate, not crash
        view = build_surface(climb(), True, landscape)
        assert view.lineage.xs[0] == pytest.approx(2.0)
        assert view.lineage.xs[-1] == pytest.approx(0.0)

    def test_peak_is_the_grid_argmax(self, landscape):
        view = build_surface([], True, landscape)
        px, py, pz = view.peak
        assert (px, py) == (0.0, 0.0)
        assert pz == pytest.approx(4.0 + LIFT * 8.0)
        assert view.nodes == ()

    def test_metric_keys_are_configurable(self, landscape):
        c = cand("c001", "draft", score=1.0, t=1)
        c = c.model_copy(update={"trials": [mk_trial(val_score=1.0, metrics={"lat": 1.0, "lon": 2.0})]})
        view = build_surface([c], True, landscape, metric_keys=("lat", "lon"))
        assert (view.nodes[0].x, view.nodes[0].y) == (1.0, 2.0)


class TestLoadLandscape:
    def test_missing_functions_fail_with_the_file_named(self, tmp_path):
        path = tmp_path / "landscape.py"
        path.write_text("def elevation(x, y):\n    return 0.0\n")
        with pytest.raises(LandscapeError, match="grid"):
            load_landscape(path)

    def test_import_errors_fail_with_the_file_named(self, tmp_path):
        path = tmp_path / "landscape.py"
        path.write_text("raise RuntimeError('boom')\n")
        with pytest.raises(LandscapeError, match="landscape.py"):
            load_landscape(path)

    def test_edited_terrain_is_reloaded(self, tmp_path):
        import os

        path = tmp_path / "landscape.py"
        path.write_text(LANDSCAPE)
        first = load_landscape(path)
        assert float(first.elevation(0.0, 0.0)) == 4.0
        path.write_text(LANDSCAPE.replace("4.0 -", "5.0 -"))
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 2))
        assert float(load_landscape(path).elevation(0.0, 0.0)) == 5.0


def write_problem(tmp_path: Path, extra_yaml: str = "", landscape: bool = True) -> Path:
    problem_dir = tmp_path / "hilly"
    problem_dir.mkdir()
    (problem_dir / "problem.yaml").write_text("metric: elevation\nhigher_is_better: true\n" + extra_yaml)
    (problem_dir / "description.md").write_text("Climb the hill.\n")
    verifier = problem_dir / "verifier.sh"
    verifier.write_text("#!/bin/sh\nexit 0\n")
    verifier.chmod(0o755)
    if landscape:
        (problem_dir / "landscape.py").write_text(LANDSCAPE)
    return problem_dir


class TestProblemPickup:
    def test_landscape_is_picked_up_by_default(self, tmp_path, config):
        problem = load_problem(write_problem(tmp_path), config)
        assert problem.landscape_path == (tmp_path / "hilly" / "landscape.py").resolve()
        assert problem.surface_metrics == ["x", "y"]

    def test_no_landscape_means_no_surface(self, tmp_path, config):
        problem = load_problem(write_problem(tmp_path, landscape=False), config)
        assert problem.landscape_path is None

    def test_surface_metrics_override(self, tmp_path, config):
        problem_dir = write_problem(tmp_path, extra_yaml="surface_metrics: [lat, lon]\n")
        assert load_problem(problem_dir, config).surface_metrics == ["lat", "lon"]

    def test_fitness_landscape_problem_defines_a_surface(self, config):
        problem = load_problem("fitness-landscape", config)
        assert problem.landscape_path is not None
