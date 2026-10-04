"""`similarity map`'s pure layer: pairwise matrices, the MDS layout, its
alignment across rebuilds, and the built view in both scopes."""

from __future__ import annotations

import numpy as np
import pytest

from hillclimb.tui.similarity import build_similarity, clear_caches
from hillclimb.tui.similarity_map import (
    METRICS,
    MapView,
    blend_matrix,
    build_map,
    build_run_map,
    classical_mds,
    lineage_matrix,
    procrustes_align,
    structural_matrix,
)
from tests.test_similarity import REPORT, cand, sub_csv, write_candidate
from tests.test_similarity_run import SORTED_FINGERPRINT, make_arm, write_fingerprint


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_caches()
    yield
    clear_caches()


def pairwise(coords: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(coords[:, None, :] - coords[None, :, :]), axis=2))


# ---------------------------------------------------------------------------
# matrices


class TestMatrices:
    def test_lineage_hops_through_the_lowest_common_ancestor(self):
        parents = {"r": None, "a": "r", "b": "r", "aa": "a", "x": None}
        ids = ["r", "a", "b", "aa", "x"]
        m = lineage_matrix(ids, parents)
        assert m[ids.index("a"), ids.index("b")] == 2
        assert m[ids.index("aa"), ids.index("b")] == 3
        assert m[ids.index("r"), ids.index("aa")] == 2
        assert m[ids.index("x"), ids.index("aa")] == 0 + 2 + 2  # disjoint: via the super-root
        assert np.allclose(m, m.T) and np.all(np.diag(m) == 0)

    def test_unknown_parent_is_a_root(self):
        m = lineage_matrix(["a", "b"], {"a": "gone", "b": "gone"})
        assert m[0, 1] == 2  # two roots, joined at the super-root

    def test_structural_is_jaccard(self):
        m = structural_matrix([frozenset("ab"), frozenset("bc"), frozenset("ab")])
        assert m[0, 1] == pytest.approx(1 - 1 / 3)
        assert m[0, 2] == 0.0
        assert np.allclose(m, m.T)

    def test_blend_normalizes_each_matrix(self):
        a = np.array([[0.0, 2.0], [2.0, 0.0]])
        b = np.array([[0.0, 200.0], [200.0, 0.0]])
        m = blend_matrix((a, b), (2.0, 200.0))
        assert m[0, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# layout


class TestClassicalMds:
    def test_recovers_a_planar_configuration(self):
        points = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]])
        d = pairwise(points)
        coords, stress = classical_mds(d)
        assert coords.shape == (4, 3)
        assert np.allclose(pairwise(coords), d, atol=1e-9)
        assert stress == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(coords[:, 2], 0.0, atol=1e-6)  # a plane needs two axes, the third is idle

    def test_stress_reports_what_three_axes_cannot_show(self):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(12, 8))
        _coords, stress = classical_mds(pairwise(points))
        assert 0.0 < stress < 1.0

    def test_signs_are_pinned(self):
        d = pairwise(np.array([[0.0], [1.0], [5.0]]))
        first, _ = classical_mds(d)
        again, _ = classical_mds(d)
        assert np.array_equal(first, again)
        column = first[:, 0]
        assert column[np.argmax(np.abs(column))] > 0

    def test_degenerate_sizes(self):
        coords, stress = classical_mds(np.zeros((0, 0)))
        assert coords.shape == (0, 3) and stress == 0.0
        coords, stress = classical_mds(np.zeros((1, 1)))
        assert coords.shape == (1, 3) and stress == 0.0
        coords, _ = classical_mds(np.array([[0.0, 2.0], [2.0, 0.0]]))
        assert pairwise(coords)[0, 1] == pytest.approx(2.0)


class TestProcrustes:
    def test_reflected_layout_is_mapped_back(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        ids = ["a", "b", "c", "d"]
        previous = {cid: tuple(-coords[i]) for i, cid in enumerate(ids)}
        aligned = procrustes_align(coords, ids, previous)
        assert np.allclose(aligned, -coords, atol=1e-9)

    def test_only_shared_ids_anchor_and_distances_survive(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0]])
        ids = ["a", "b", "c", "new"]
        previous = {"a": (10.0, 0.0, 0.0), "b": (10.0, 1.0, 0.0), "c": (9.0, 0.0, 0.0)}  # rotated + shifted
        aligned = procrustes_align(coords, ids, previous)
        for i, cid in enumerate(ids[:3]):
            assert np.allclose(aligned[i], previous[cid], atol=1e-9)
        assert np.allclose(pairwise(aligned), pairwise(coords), atol=1e-9)

    def test_nothing_shared_returns_as_is(self):
        coords = np.array([[1.0, 2.0, 3.0]])
        assert procrustes_align(coords, ["a"], {"b": (0.0, 0.0, 0.0)}) is coords


# ---------------------------------------------------------------------------
# the view, search scope


def make_search(tmp_path):
    """Seedless search: c000 baseline at (1,2,3), two children of it, and a
    grandchild that duplicates c001's output but not its source."""
    write_candidate(tmp_path, "c000", solution="x = 1\n", submission=sub_csv([1.0, 2.0, 3.0]))
    write_candidate(tmp_path, "c001", solution="x = 1\ny = 2\n", submission=sub_csv([2.0, 3.0, 4.0]))
    write_candidate(tmp_path, "c002", solution="z = 9\n", submission=sub_csv([11.0, 12.0, 13.0]))
    write_candidate(tmp_path, "c003", solution="q = 4\nw = 5\n", submission=sub_csv([2.0, 3.0, 4.0]))
    return [
        cand("c000", "baseline", score=0.5, t=0),
        cand("c001", "draft", parent="c000", score=0.6, t=1),
        cand("c002", "draft", parent="c000", score=0.4, t=2),
        cand("c003", "improve", parent="c001", score=0.7, t=3),
    ]


class TestBuildMap:
    def test_layout_reproduces_behavioral_distances(self, tmp_path):
        view = build_map(make_search(tmp_path), tmp_path, True)
        assert view.unavailable is None and view.mode == "submission" and view.metric == "behavioral"
        ids = [n.id for n in view.nodes]
        assert ids == ["c000", "c001", "c002", "c003"]
        coords = np.array([[n.x, n.y, n.z] for n in view.nodes])
        assert view.behavioral is not None
        # the same unit as the cube: from the origin, the map's row IS the cube's axis
        cube = build_similarity(make_search(tmp_path), tmp_path, True)
        for node in cube.nodes:
            assert view.behavioral[0, view.index(node.id)] == pytest.approx(node.raw[0])
        assert view.behavioral[0, 2] == pytest.approx(10 * view.behavioral[0, 1])
        assert view.behavioral[1, 3] == pytest.approx(0.0)  # same output
        assert np.allclose(pairwise(coords), view.behavioral, atol=1e-6)
        assert view.stress == pytest.approx(0.0, abs=1e-6)

    def test_edges_trail_flags_and_bins(self, tmp_path):
        view = build_map(make_search(tmp_path), tmp_path, True)
        ids = [n.id for n in view.nodes]
        assert set(view.edges) == {(0, 1), (0, 2), (1, 3)}
        assert [ids[i] for i in view.trail] == ["c000", "c001", "c003"]
        by_id = {n.id: n for n in view.nodes}
        assert by_id["c000"].origin and not by_id["c000"].best
        assert by_id["c003"].best and by_id["c003"].on_path
        assert by_id["c003"].bin == max(n.bin for n in view.nodes)
        assert by_id["c002"].bin == 0
        assert by_id["c003"].parent_id == "c001"
        assert view.n_unpositioned == 0

    def test_structural_metric_separates_same_output_candidates(self, tmp_path):
        view = build_map(make_search(tmp_path), tmp_path, True, metric="structural")
        assert view.structural is not None
        i, j = view.index("c001"), view.index("c003")
        assert view.structural[i, j] > 0.5
        coords = np.array([[n.x, n.y, n.z] for n in view.nodes])
        assert np.allclose(pairwise(coords), view.structural, atol=1e-6)

    def test_blend_uses_all_three(self, tmp_path):
        view = build_map(make_search(tmp_path), tmp_path, True, metric="blend")
        assert view.unavailable is None
        assert all(s > 0 for s in view.scales)
        assert view.lineage is not None and view.lineage[view.index("c002"), view.index("c003")] == 3

    def test_unknown_metric_is_an_error(self, tmp_path):
        with pytest.raises(ValueError):
            build_map(make_search(tmp_path), tmp_path, True, metric="vibes")
        assert METRICS == ("behavioral", "structural", "blend")

    def test_previous_layout_keeps_the_picture_in_place(self, tmp_path):
        candidates = make_search(tmp_path)
        first = build_map(candidates[:3], tmp_path, True)
        previous = first.positions()
        # flip the remembered layout: the rebuild must follow it, not the pinned signs
        flipped = {cid: (-x, -y, -z) for cid, (x, y, z) in previous.items()}
        second = build_map(candidates, tmp_path, True, previous=flipped)
        for node in second.nodes:
            if node.id in flipped:
                assert np.allclose((node.x, node.y, node.z), flipped[node.id], atol=1e-6)
        assert second.index("c003") is not None

    def test_unpositioned_candidates_are_counted(self, tmp_path):
        candidates = make_search(tmp_path)
        write_candidate(tmp_path, "c004", submission=sub_csv([1.0, 2.0]))  # row count off
        candidates.append(cand("c004", "draft", parent="c000", score=0.1, t=4))
        view = build_map(candidates, tmp_path, True)
        assert view.index("c004") is None and view.n_unpositioned == 1

    def test_no_candidates(self, tmp_path):
        view = build_map([], tmp_path, True)
        assert view.unavailable == "no candidates yet" and view.nodes == ()
        assert isinstance(view, MapView)

    def test_origin_without_prints_is_unavailable(self, tmp_path):
        view = build_map([cand("c000", "baseline", score=0.5)], tmp_path, True)
        assert view.unavailable is not None and "solution.py" in view.unavailable


class TestModes:
    def test_report_mode(self, tmp_path):
        far = {"horizons": [{"bucket": "1h", "score": 2.0}, {"bucket": "2h", "score": 4.0},
                            {"bucket": "3h", "score": 8.0}], "quantiles": [{"q": 0.5, "pinball": 1.0}]}
        for cid in ("c000", "c001", "c002"):
            write_candidate(tmp_path, cid)
        view = build_map(
            [cand("c000", "baseline", score=1.0, submission_ok=False, report=REPORT, t=0),
             cand("c001", "draft", parent="c000", score=0.9, submission_ok=False, report=REPORT, t=1),
             cand("c002", "draft", parent="c000", score=0.8, submission_ok=False, report=far, t=2)],
            tmp_path, True,
        )
        assert view.mode == "report" and view.behavioral is not None
        assert view.behavioral[0, 1] == pytest.approx(0.0)
        assert view.behavioral[0, 2] == pytest.approx(1.0)  # every key doubled = +100% each
        assert view.behavioral[1, 2] == pytest.approx(1.0)

    def test_fingerprint_mode_sees_through_row_order(self, tmp_path):
        write_candidate(tmp_path, "c000", submission=sub_csv([1.0, 2.0, 3.0]))
        write_candidate(tmp_path, "c001", submission=sub_csv([3.0, 2.0, 1.0]))
        write_candidate(tmp_path, "c002", submission=sub_csv([1.0, 2.0, 4.0]))
        view = build_map(
            [cand("c000", "baseline", score=0.5, t=0),
             cand("c001", "draft", parent="c000", score=0.6, t=1),
             cand("c002", "draft", parent="c000", score=0.6, t=2)],
            tmp_path, True, fingerprint_path=write_fingerprint(tmp_path, SORTED_FINGERPRINT),
        )
        assert view.mode == "fingerprint" and view.behavioral is not None
        assert view.behavioral[0, 1] == pytest.approx(0.0)
        assert view.behavioral[0, 2] > 0.0


# ---------------------------------------------------------------------------
# run scope


class TestRunMap:
    def make_run(self, tmp_path):
        return [
            make_arm(tmp_path, "p", "greedy", child_scores=(0.6, 0.8)),
            make_arm(tmp_path, "p-2", "openevolve", child_scores=(0.7, 0.95)),
        ]

    def test_every_search_in_one_map(self, tmp_path):
        view = build_run_map(self.make_run(tmp_path), True, problem_key="p")
        assert view.unavailable is None and view.scope == "run"
        assert view.experiments == ("greedy", "openevolve") and view.n_searches == 2
        ids = [n.id for n in view.nodes]
        assert "p/c001" in ids and "p-2/c003" in ids
        assert view.n_unpositioned == 2  # the sourceless baselines
        by_id = {n.id: n for n in view.nodes}
        assert by_id["p/c001"].origin and by_id["p-2/c001"].origin
        assert [n.id for n in view.nodes if n.best] == ["p-2/c003"]
        assert by_id["p-2/c003"].experiment == "openevolve" and by_id["p-2/c003"].search_id == "p-2"
        # the two seeds share one file: zero apart on every matrix that measures output
        i, j = ids.index("p/c001"), ids.index("p-2/c001")
        assert view.behavioral[i, j] == pytest.approx(0.0)
        assert view.structural[i, j] == pytest.approx(0.0)
        assert view.lineage[i, j] == 2.0  # separate trees, joined at the super-root

    def test_edges_stay_inside_their_search_and_trail_follows_the_champion(self, tmp_path):
        view = build_run_map(self.make_run(tmp_path), True)
        ids = [n.id for n in view.nodes]
        for a, b in view.edges:
            assert ids[a].split("/")[0] == ids[b].split("/")[0]
        assert [ids[i] for i in view.trail] == ["p-2/c001", "p-2/c002", "p-2/c003"]

    def test_seed_mismatch_is_unavailable(self, tmp_path):
        searches = [
            make_arm(tmp_path, "p", "greedy"),
            make_arm(tmp_path, "p-2", "openevolve", seed_source="s = 2\n"),
        ]
        view = build_run_map(searches, True)
        assert view.unavailable is not None and "seeds differ" in view.unavailable

    def test_no_searches(self):
        view = build_run_map([], True, problem_key="p")
        assert view.unavailable == "no searches for p in the run"


def test_a_baseline_without_code_gives_way_to_the_earliest_that_has_some(tmp_path):
    """A `baseline_files` problem's baseline (heilbronn) is a sample
    submission with no solution.py: the views anchor on the earliest
    candidate that can be compared instead of refusing to draw."""
    (tmp_path / "candidates" / "c000").mkdir(parents=True)
    (tmp_path / "candidates" / "c000" / "submission.csv").write_text(sub_csv([0.0, 0.0, 0.0]))
    write_candidate(tmp_path, "c001", solution="x = 1\n", submission=sub_csv([1.0, 2.0, 3.0]))
    write_candidate(tmp_path, "c002", solution="y = 2\n", submission=sub_csv([2.0, 3.0, 4.0]))
    candidates = [
        cand("c000", "baseline", t=0),
        cand("c001", "draft", score=0.5, t=1),
        cand("c002", "draft", score=0.6, t=2),
    ]
    view = build_map(candidates, tmp_path, True)
    assert view.unavailable is None and {n.id for n in view.nodes} == {"c001", "c002"}
    cube = build_similarity(candidates, tmp_path, True)
    assert cube.unavailable is None and cube.reference_label == "earliest"
    # a champion asked for by name is not swapped for another candidate
    assert build_similarity(candidates, tmp_path, True, reference="champion").unavailable is None
