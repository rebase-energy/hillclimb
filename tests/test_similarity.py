"""The similarity view's pure layer: fingerprints, distances, the built view."""

from __future__ import annotations

from tests.factories import trial as mk_trial

import json
import os
from pathlib import Path

import numpy as np
import pytest

from hillclimb.harness.candidate import Candidate
from hillclimb.tui.similarity import (
    N_BINS,
    build_similarity,
    clear_caches,
    lineage_distance,
    report_distance,
    report_fingerprint,
    structural_distance,
    submission_distance,
    token_set,
)


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_caches()
    yield
    clear_caches()


def cand(
    cid: str, operator: str = "improve", parent: str | None = None, score: float | None = None,
    status: str = "passing", t: int = 0, submission_ok: bool = True, report: dict | None = None,
    metrics: dict | None = None, **kwargs,
) -> Candidate:
    trials = []
    if score is not None or report is not None:
        trials = [mk_trial(val_score=score, submission_ok=submission_ok,
                        report=report, metrics=metrics or {})]
    return Candidate(
        candidate_id=cid, operator=operator, parent_id=parent, status=status, trials=trials,
        created_at=f"2026-08-28T10:{t:02d}:00+00:00",
        finished_at=f"2026-08-28T10:{t + 1:02d}:00+00:00",
        **kwargs,
    )


def write_candidate(search_dir: Path, cid: str, solution: str = "x = 1\n",
                    submission: str | None = None) -> Path:
    cdir = search_dir / "candidates" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "solution.py").write_text(solution)
    if submission is not None:
        (cdir / "submission.csv").write_text(submission)
    return cdir


def sub_csv(values: list[float], extra_col: bool = False) -> str:
    header = "id,pred" + (",note" if extra_col else "")
    rows = [f"{i},{v}" + (",n" if extra_col else "") for i, v in enumerate(values)]
    return "\n".join([header, *rows]) + "\n"


class TestSubmissionDistance:
    def test_identical_submissions_are_at_zero(self, tmp_path):
        write_candidate(tmp_path, "c000", submission=sub_csv([1.0, 2.0, 3.0]))
        write_candidate(tmp_path, "c001", submission=sub_csv([1.0, 2.0, 3.0]))
        view = build_similarity(
            [cand("c000", "baseline", score=0.5, t=0), cand("c001", "draft", score=0.5, t=1)],
            tmp_path, True,
        )
        assert view.mode == "submission"
        node = next(n for n in view.nodes if n.id == "c001")
        assert node.raw[0] == pytest.approx(0.0)

    def test_row_count_mismatch_is_unpositionable(self, tmp_path):
        write_candidate(tmp_path, "c000", submission=sub_csv([1.0, 2.0, 3.0]))
        write_candidate(tmp_path, "c001", submission=sub_csv([1.0, 2.0]))
        view = build_similarity(
            [cand("c000", "baseline", score=0.5, t=0), cand("c001", "draft", score=0.5, t=1)],
            tmp_path, True,
        )
        assert {n.id for n in view.nodes} == {"c000"}
        assert view.n_unpositioned == 1

    def test_extra_columns_are_ignored(self, tmp_path):
        write_candidate(tmp_path, "c000", submission=sub_csv([1.0, 2.0]))
        write_candidate(tmp_path, "c001", submission=sub_csv([1.0, 2.0], extra_col=True))
        view = build_similarity(
            [cand("c000", "baseline", score=0.5, t=0), cand("c001", "draft", score=0.5, t=1)],
            tmp_path, True,
        )
        assert {n.id for n in view.nodes} == {"c000", "c001"}

    def test_submission_not_ok_is_skipped_unparsed(self, tmp_path):
        write_candidate(tmp_path, "c000", submission=sub_csv([1.0, 2.0]))
        write_candidate(tmp_path, "c001", submission="utter garbage\x00")
        view = build_similarity(
            [cand("c000", "baseline", score=0.5, t=0),
             cand("c001", "draft", score=None, status="buggy", submission_ok=False, t=1)],
            tmp_path, True,
        )
        assert view.n_unpositioned == 1

    def test_distance_is_reference_scaled_rms(self):
        ref = (("pred",), 4, {"pred": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)})
        cnd = (("pred",), 4, {"pred": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)})
        # MAD of ref = 1.0; every dim shifted by 1 -> distance exactly 1.0
        assert submission_distance(ref, cnd) == pytest.approx(1.0)


REPORT = {
    "horizons": [{"bucket": "1h", "score": 1.0}, {"bucket": "2h", "score": 2.0},
                 {"bucket": "3h", "score": 4.0}],
    "quantiles": [{"q": 0.5, "pinball": 0.5}],
}


class TestReportDistance:
    def test_report_mode_when_reference_has_no_submission(self, tmp_path):
        write_candidate(tmp_path, "c000")
        write_candidate(tmp_path, "c001")
        view = build_similarity(
            [cand("c000", "baseline", score=1.0, submission_ok=False, report=REPORT, t=0),
             cand("c001", "draft", score=0.9, submission_ok=False, report=REPORT, t=1)],
            tmp_path, True,
        )
        assert view.mode == "report"
        node = next(n for n in view.nodes if n.id == "c001")
        assert node.raw[0] == pytest.approx(0.0)

    def test_fingerprint_priority_and_labels(self):
        fp = report_fingerprint(
            cand("c", score=1.0, report={**REPORT, "zones": [{"zone": "z1", "score": 3.0}]},
                 metrics={"x": 7.0})
        )
        assert fp == {
            "horizon:1h": 1.0, "horizon:2h": 2.0, "horizon:3h": 4.0,
            "q:0.5": 0.5, "metric:x": 7.0, "zone:z1": 3.0,
        }

    def test_intersection_below_threshold_is_none(self):
        ref = {f"horizon:{i}h": float(i) for i in range(1, 7)}  # 6 keys
        assert report_distance(ref, {"horizon:1h": 1.0, "horizon:2h": 2.0}) is None  # <3 shared
        two_of_six = {"horizon:1h": 1.0, "horizon:2h": 2.0, "other:a": 1.0, "other:b": 1.0}
        assert report_distance(ref, two_of_six) is None  # <50% of ref keys
        assert report_distance(ref, ref) == pytest.approx(0.0)

    def test_exploded_dimension_is_clipped(self):
        ref = {"a": 1.0, "b": 1.0, "c": 1.0}
        d = report_distance(ref, {"a": 1.0, "b": 1.0, "c": 1e9})
        assert d == pytest.approx(np.sqrt((0 + 0 + 10.0**2) / 3))


class TestStructural:
    def test_token_jaccard(self, tmp_path):
        a_dir = write_candidate(tmp_path, "a", solution="x = 1\ny = 2\n")
        b_dir = write_candidate(tmp_path, "b", solution="x = 1\ny = 3\n")
        a = token_set(tmp_path, cand("a"))
        b = token_set(tmp_path, cand("b"))
        assert structural_distance(a, a) == 0.0
        assert 0.0 < structural_distance(a, b) < 1.0
        assert a_dir != b_dir

    def test_unparsable_source_degrades_to_word_set(self, tmp_path):
        write_candidate(tmp_path, "a", solution="def broken(:\n    x =\n")
        assert token_set(tmp_path, cand("a")) is not None

    def test_missing_solution_is_none(self, tmp_path):
        (tmp_path / "candidates" / "a").mkdir(parents=True)
        assert token_set(tmp_path, cand("a")) is None

    def test_cache_reloads_on_mtime_change(self, tmp_path):
        write_candidate(tmp_path, "a", solution="x = 1\n")
        first = token_set(tmp_path, cand("a"))
        path = tmp_path / "candidates" / "a" / "solution.py"
        path.write_text("y = 2\n")
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 2))
        assert token_set(tmp_path, cand("a")) != first


class TestLineage:
    CANDS = {
        c.candidate_id: c
        for c in [
            cand("c000", "baseline", t=0),
            cand("c001", "draft", t=1),
            cand("c002", parent="c001", t=2),
            cand("c003", parent="c002", t=3),
            cand("c004", parent="c001", t=4),
            cand("c005", parent="ghost", t=5),  # parent missing from the journal
        ]
    }

    def test_ancestor_hops(self):
        assert lineage_distance(self.CANDS, "c003", "c001") == 2.0
        assert lineage_distance(self.CANDS, "c001", "c003") == 2.0
        assert lineage_distance(self.CANDS, "c003", "c003") == 0.0

    def test_through_the_lowest_common_ancestor(self):
        assert lineage_distance(self.CANDS, "c003", "c004") == 3.0  # via c001

    def test_disjoint_components_use_a_virtual_super_root(self):
        # c000 (root) and c005 (root after its missing parent): 0 + 0 + 2
        assert lineage_distance(self.CANDS, "c000", "c005") == 2.0


class TestBuildView:
    def make_search(self, tmp_path) -> list[Candidate]:
        candidates = [
            cand("c000", "baseline", score=0.1, t=0),
            cand("c001", "draft", score=0.5, t=1),
            cand("c002", parent="c001", score=0.7, t=2),
            cand("c003", parent="c002", score=0.9, t=3),
        ]
        for i, cid in enumerate(["c000", "c001", "c002", "c003"]):
            write_candidate(tmp_path, cid, solution=f"x = {i}\n",
                            submission=sub_csv([float(i), float(i) + 1]))
        return candidates

    def test_reference_sits_at_the_origin(self, tmp_path):
        view = build_similarity(self.make_search(tmp_path), tmp_path, True)
        ref = next(n for n in view.nodes if n.id == view.reference_id)
        assert view.reference_id == "c000"
        assert ref.raw[:2] == (0.0, 0.0) and ref.raw[2] == 0.0

    def test_champion_toggle_changes_reference_not_availability(self, tmp_path):
        candidates = self.make_search(tmp_path)
        view = build_similarity(candidates, tmp_path, True, reference="champion")
        assert view.reference_id == "c003"
        assert view.unavailable is None
        assert len(view.nodes) == 4

    def test_axes_are_normalized_to_the_unit_cube(self, tmp_path):
        view = build_similarity(self.make_search(tmp_path), tmp_path, True)
        for node in view.nodes:
            assert 0.0 <= node.x <= 1.0 and 0.0 <= node.y <= 1.0 and 0.0 <= node.z <= 1.0
        assert all(s > 0 for s in view.scales)

    def test_rank_bins_and_best(self, tmp_path):
        view = build_similarity(self.make_search(tmp_path), tmp_path, True)
        by_id = {n.id: n for n in view.nodes}
        assert by_id["c003"].best
        assert by_id["c000"].bin == 0  # worst
        assert by_id["c000"].bin < by_id["c001"].bin < by_id["c002"].bin < by_id["c003"].bin
        assert by_id["c003"].bin < N_BINS

    def test_fates_match_the_tree(self, tmp_path):
        from hillclimb.tui.tree import build_tree

        candidates = self.make_search(tmp_path)
        view = build_similarity(candidates, tmp_path, True)
        fates = {n.id: n.fate for n in build_tree(candidates, True).nodes}
        for node in view.nodes:
            assert node.fate == fates[node.id]

    def test_no_candidates_is_unavailable(self, tmp_path):
        view = build_similarity([], tmp_path, True)
        assert view.unavailable == "no candidates yet"

    def test_reference_without_artifacts_is_unavailable_with_reason(self, tmp_path):
        (tmp_path / "candidates" / "c000").mkdir(parents=True)
        view = build_similarity([cand("c000", "baseline", score=0.1, t=0)], tmp_path, True)
        assert view.unavailable is not None
        assert "solution.py" in view.unavailable

    def test_no_champion_yet_is_unavailable(self, tmp_path):
        write_candidate(tmp_path, "c000")
        view = build_similarity(
            [cand("c000", "baseline", score=None, status="buggy", submission_ok=False, t=0)],
            tmp_path, True, reference="champion",
        )
        assert view.unavailable is not None


class TestJsonSubmissions:
    """JSON-native problems (Einstein Arena) declare submission.json as their
    output artifact; the behavioral axis must read it, not fall back to
    report mode."""

    JSON_ARTIFACTS = ["submission.json"]

    def write_json(self, search_dir: Path, cid: str, payload: dict, solution: str = "x = 1\n") -> Path:
        cdir = search_dir / "candidates" / cid
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "solution.py").write_text(solution)
        (cdir / "submission.json").write_text(json.dumps(payload))
        return cdir

    def build(self, tmp_path, candidates):
        return build_similarity(
            candidates, tmp_path, True, output_artifacts=self.JSON_ARTIFACTS
        )

    def test_json_reference_selects_submission_mode(self, tmp_path):
        self.write_json(tmp_path, "c000", {"circles": [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]})
        self.write_json(tmp_path, "c001", {"circles": [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]},
                        solution="x = 2\n")
        view = self.build(tmp_path, [
            cand("c000", "baseline", score=0.5, t=0),
            cand("c001", "draft", score=0.6, t=1),
        ])
        assert view.unavailable is None
        assert view.mode == "submission"
        # identical geometry -> zero behavioral distance, nonzero structural
        moved = next(n for n in view.nodes if n.id == "c001")
        assert moved.raw[0] == pytest.approx(0.0)
        assert moved.raw[1] > 0.0

    def test_moved_circle_is_behaviorally_distant(self, tmp_path):
        self.write_json(tmp_path, "c000", {"circles": [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]})
        self.write_json(tmp_path, "c001", {"circles": [[0.0, 0.0, 1.0], [9.0, 0.0, 1.0]]})
        view = self.build(tmp_path, [
            cand("c000", "baseline", score=0.5, t=0),
            cand("c001", "draft", score=0.6, t=1),
        ])
        assert view.mode == "submission"
        assert next(n for n in view.nodes if n.id == "c001").raw[0] > 0.0

    def test_different_shape_is_unalignable_not_wrongly_compared(self, tmp_path):
        self.write_json(tmp_path, "c000", {"circles": [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]})
        self.write_json(tmp_path, "c001", {"circles": [[0.0, 0.0, 1.0]]})
        view = self.build(tmp_path, [
            cand("c000", "baseline", score=0.5, t=0),
            cand("c001", "draft", score=0.6, t=1),
        ])
        assert [n.id for n in view.nodes] == ["c000"]
        assert view.n_unpositioned == 1

    def test_bools_and_non_finite_are_not_measurements(self, tmp_path):
        self.write_json(tmp_path, "c000", {"ok": True, "n": 3.0, "bad": float("nan")})
        payload = {"ok": False, "n": 3.0, "bad": float("nan")}
        self.write_json(tmp_path, "c001", payload)
        view = self.build(tmp_path, [
            cand("c000", "baseline", score=0.5, t=0),
            cand("c001", "draft", score=0.6, t=1),
        ])
        # only "n" survives flattening, and it matches
        assert view.mode == "submission"
        assert next(n for n in view.nodes if n.id == "c001").raw[0] == pytest.approx(0.0)

    def test_unparseable_json_falls_back_to_report_mode(self, tmp_path):
        cdir = tmp_path / "candidates" / "c000"
        cdir.mkdir(parents=True)
        (cdir / "solution.py").write_text("x = 1\n")
        (cdir / "submission.json").write_text("{not json")
        view = self.build(tmp_path, [cand("c000", "baseline", score=0.5, t=0)])
        assert view.mode == "report" or view.unavailable is not None

    def test_csv_problems_are_unaffected(self, tmp_path):
        write_candidate(tmp_path, "c000", submission=sub_csv([1.0, 2.0, 3.0]))
        write_candidate(tmp_path, "c001", submission=sub_csv([1.0, 2.0, 3.0]))
        view = build_similarity(
            [cand("c000", "baseline", score=0.5, t=0), cand("c001", "draft", score=0.5, t=1)],
            tmp_path, True, output_artifacts=["submission.csv"],
        )
        assert view.mode == "submission"


class TestSiteLook:
    """The reference cube drawn as hillclimb.sh advertises it."""

    def _view(self, tmp_path):
        write_candidate(tmp_path, "c000", solution="x = 1\n", submission=sub_csv([1.0, 2.0, 3.0]))
        write_candidate(tmp_path, "c001", solution="x = 1\ny = 2\n", submission=sub_csv([2.0, 3.0, 4.0]))
        write_candidate(tmp_path, "c002", solution="z = 9\n", submission=sub_csv([5.0, 6.0, 7.0]))
        write_candidate(tmp_path, "c003", solution="q = 4\nw = 5\n", submission=sub_csv([2.0, 3.5, 4.0]))
        return build_similarity(
            [
                cand("c000", "baseline", score=0.5, t=0),
                cand("c001", "draft", parent="c000", score=0.6, t=1),   # built on
                cand("c002", "draft", parent="c000", score=0.4, t=2),   # left
                cand("c003", "improve", parent="c001", score=0.7, t=3),  # the best
            ],
            tmp_path, True,
        )

    def test_marks_say_what_a_candidate_is(self, tmp_path):
        from hillclimb.tui import similarityview as sv

        calls = {}

        class Recorder:
            legend_visible = True

            def __getattr__(self, name):
                return lambda *a, **k: calls.setdefault(name, []).append((a, k))

        import hillclimb.tui.similarityview as module

        original = module.boxed_plot
        module.boxed_plot = Recorder
        try:
            sv.build_similarity_plot(self._view(tmp_path))
        finally:
            module.boxed_plot = original
        assert calls["set_axis_titles3d"] == [(("behaviour", "code", "lineage"), {})]
        (_, graph), = calls["add_graph3d"]
        # built on = disc, left = circle; the origin and then the best are drawn last
        assert graph["node_shapes"] == ["disc", "circle", "diamond-open", "diamond"]
        assert graph["node_colors"][-2:] == [sv.REFERENCE_RGB, sv.BEST_RGB]
        # the best's lineage, from the corner out, in white
        (args, line), = calls["add_line3d"]
        assert line["color"] == sv.LINEAGE_RGB and args[0][0] == 0.0 and len(args[0]) == 3

    def test_fates_and_the_legend_row(self):
        from hillclimb.tui.similarityview import SCORE_RGB, fate_shape, legend_markup

        assert fate_shape("expanded") == "disc" and fate_shape("discontinued") == "circle"
        assert fate_shape("failed") == "dot" and fate_shape("expanded", scored=False) == "dot"
        legend = legend_markup()
        for word in ("origin, the corner", "best", "built on", "left", "failed", "the best's lineage", "score, low → high"):
            assert word in legend
        assert SCORE_RGB[0] == (59, 63, 153) and SCORE_RGB[-1] == (255, 122, 44)  # the site's ramp, end to end

    def test_the_legend_is_in_the_plot_like_the_trees(self):
        """The marks are plotui legend rows at the top left — each with the
        swatch the plot draws that mark with — and the score ramp a column
        of cells at the top right; no row under the plot."""
        import hillclimb.tui.similarityview as sv
        from hillclimb.tui.theme import themed_plot

        rows = sv.legend_rows()
        assert [(label, swatch) for label, swatch, *_ in rows] == [
            ("origin, the corner", "diamond-open"), ("best", "diamond"), ("built on", "disc"),
            ("left", "ring"), ("failed", "dot"), ("the best's lineage", "line"),
        ]
        plot = themed_plot()
        assert sv.apply_legend(plot, rows) and plot.legend_visible
        assert [entry[0] for entry in plot.legend_entries()] == [row[0] for row in rows]
        assert sv.PLOT_LEGEND  # so the screens leave out the text row

        spans = sv.ramp_spans(80)
        blocks = [(row, col, style) for row, col, text, style in spans if text == "█"]
        assert len(blocks) == len(sv.SCORE_RGB) and {col for _, col, _ in blocks} == {78}
        r, g, b = sv.SCORE_RGB[-1]
        assert blocks[0][2] == f"rgb({r},{g},{b})"  # best at the top
        assert [text.strip() for _, _, text, _ in spans if text.strip() in ("score", "high", "low")] == ["score", "high", "low"]

    def test_an_older_plotui_keeps_the_text_row(self):
        import hillclimb.tui.similarityview as sv

        class Old:  # no host legend rows
            legend_visible = True

        assert sv.apply_legend(Old(), sv.legend_rows()) is False
