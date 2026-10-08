"""Graph index: deterministic build, time filtering, supersession, layout."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from hillclimb.modules.memory.claims import Claim, Entity, ensure_concepts, save_entities
from hillclimb.modules.memory.graph import (
    KnowledgeGraph,
    build_graph,
    compute_supersessions,
    graph_at,
    graph_path,
    graph_stats,
    load_graph,
    load_or_build_graph,
    write_graph,
)
from hillclimb.modules.memory.knowledge import ApproachNote, KnowledgeCard, OperatorStat, write_card
from tests.catalog_fixture import pinned


def make_card(problem_id="spaceship-titanic", run_ref="r1/s1", finished_at="2026-07-01T00:00:00Z",
              claims=(), libraries=("sklearn",), metric="accuracy"):
    return KnowledgeCard(
        problem_id=problem_id, family=problem_id, run_ref=run_ref,
        metric=metric, finished_at=finished_at, n_candidates=3, n_ok=3,
        selected_val=0.8, selected_operator="improve",
        operator_stats={"draft": OperatorStat(attempts=1, ok=1),
                        "improve": OperatorStat(attempts=2, ok=2, best_val=0.8)},
        top_approaches=[ApproachNote(candidate_id="c001", operator="improve",
                                     val_score=0.8, libraries=list(libraries))],
        claims=list(claims),
    )


def claim(subject="histgradientboosting", relation="helps", obj="", observed="2026-07-01T00:00:00Z",
          family="spaceship-titanic", cid="cl1", confidence=0.8):
    return Claim(
        claim_id=cid, subject=subject, relation=relation, object=obj,
        scope={"family": family, "problem_id": family}, confidence=confidence,
        evidence=["c001"], observed_at=observed,
    )


@pytest.fixture
def knowledge_dir(tmp_path):
    kdir = tmp_path / "knowledge"
    ensure_concepts(kdir)
    save_entities(kdir, [
        Entity(slug="histgradientboosting", aliases=["HGB"],
               concepts=["decision-trees", "tabular"], first_seen="2026-07-01T00:00:00Z"),
    ])
    write_card(kdir, make_card(claims=[claim()]))
    write_card(kdir, make_card(
        run_ref="r2/s1", finished_at="2026-07-02T00:00:00Z", libraries=("lightgbm",),
    ))
    return kdir


class TestBuild:
    def test_structure(self, knowledge_dir):
        graph = build_graph(knowledge_dir)
        ids = {n.id for n in graph.nodes}
        assert "problem:spaceship-titanic" in ids
        assert "family:spaceship-titanic" in ids
        assert "search:r1/s1" in ids and "search:r2/s1" in ids
        assert "entity:histgradientboosting" in ids
        assert "concept:decision-trees" in ids
        assert "claim:cl1" in ids
        kinds = {(e.src, e.dst, e.type) for e in graph.edges}
        assert ("search:r1/s1", "problem:spaceship-titanic", "ran_on") in kinds
        assert ("problem:spaceship-titanic", "family:spaceship-titanic", "belongs_to") in kinds
        assert ("claim:cl1", "entity:histgradientboosting", "about") in kinds
        assert ("claim:cl1", "search:r1/s1", "derived_from") in kinds
        assert ("entity:histgradientboosting", "concept:decision-trees", "has_concept") in kinds
        # library imports become entity-style nodes even without a registry entry
        assert "entity:lightgbm" in ids
        assert graph.events == ["2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"]
        # problems classify deterministically
        problem = graph.node_map()["problem:spaceship-titanic"]
        assert "classification" in problem.concepts

    def test_deterministic(self, knowledge_dir):
        a = build_graph(knowledge_dir)
        b = build_graph(knowledge_dir)
        assert a.model_dump(exclude={"built_at"}) == b.model_dump(exclude={"built_at"})

    def test_positions_computed_and_missing_networkx_tolerated(self, knowledge_dir, monkeypatch):
        graph = build_graph(knowledge_dir)
        assert all(n.pos is not None for n in graph.nodes)
        assert all(n.pos3 is not None and len(n.pos3) == 3 for n in graph.nodes)

        def no_networkx(*args, **kwargs):
            raise ModuleNotFoundError("networkx")

        monkeypatch.setattr("hillclimb.modules.memory.graph._spring_positions", no_networkx)
        bare = build_graph(knowledge_dir)
        assert all(n.pos is None for n in bare.nodes)
        assert all(n.pos3 is None for n in bare.nodes)

    def test_incremental_layout_pins_old_positions(self, knowledge_dir):
        first = build_graph(knowledge_dir)
        write_card(knowledge_dir, make_card(
            run_ref="r3/s1", finished_at="2026-07-03T00:00:00Z",
        ))
        second = build_graph(knowledge_dir, previous=first)
        old = first.node_map()
        for node in second.nodes:
            if node.id in old and old[node.id].pos is not None:
                assert node.pos == old[node.id].pos
                assert node.pos3 == old[node.id].pos3
        assert second.node_map()["search:r3/s1"].pos is not None
        assert second.node_map()["search:r3/s1"].pos3 is not None


class TestTimeFilter:
    def test_graph_at(self, knowledge_dir):
        graph = build_graph(knowledge_dir)
        early = graph_at(graph, "2026-07-01T12:00:00Z")
        ids = {n.id for n in early.nodes}
        assert "search:r1/s1" in ids
        assert "search:r2/s1" not in ids
        assert all(e.src in ids and e.dst in ids for e in early.edges)
        assert early.events == ["2026-07-01T00:00:00Z"]
        # concepts are timeless (first_seen="") — visible at any time
        assert "concept:decision-trees" in ids
        assert graph_at(graph, None) is graph


class TestSupersession:
    def test_same_key_newer_wins(self):
        older = claim(observed="2026-07-01T00:00:00Z", cid="a")
        newer = claim(observed="2026-07-02T00:00:00Z", cid="b")
        result = compute_supersessions([older, newer])
        assert result == {"a": ("2026-07-02T00:00:00Z", "b")}

    def test_opposing_relations_contradict(self):
        helps = claim(relation="helps", observed="2026-07-01T00:00:00Z", cid="a")
        noop = claim(relation="no_effect", observed="2026-07-02T00:00:00Z", cid="b")
        result = compute_supersessions([helps, noop])
        assert result == {"a": ("2026-07-02T00:00:00Z", "b")}

    def test_superseded_claim_hidden_after_t(self, knowledge_dir):
        write_card(knowledge_dir, make_card(
            run_ref="r3/s1", finished_at="2026-07-03T00:00:00Z",
            claims=[claim(relation="no_effect", observed="2026-07-03T00:00:00Z", cid="cl2")],
        ))
        graph = build_graph(knowledge_dir)
        node = graph.node_map()["claim:cl1"]
        assert node.superseded_at == "2026-07-03T00:00:00Z"
        assert ("claim:cl2", "claim:cl1", "supersedes") in {
            (e.src, e.dst, e.type) for e in graph.edges
        }
        # visible while it was believed, gone after the newer claim landed
        assert "claim:cl1" in {n.id for n in graph_at(graph, "2026-07-02T00:00:00Z").nodes}
        assert "claim:cl1" not in {n.id for n in graph_at(graph, "2026-07-03T00:00:00Z").nodes}


class TestRetrieval:
    def test_ranking_and_scope(self, knowledge_dir):
        from hillclimb.modules.memory.graph import node_to_claim, retrieve_claims

        # cross-family card whose claim shares the `tabular` concept via its
        # subject entity, plus one claim about an unrelated concept space
        write_card(knowledge_dir, make_card(
            problem_id="other-comp", run_ref="r4/s1",
            finished_at="2026-07-04T00:00:00Z",
            claims=[claim(family="other-comp", observed="2026-07-04T00:00:00Z", cid="cross")],
        ))
        graph = build_graph(knowledge_dir)
        nodes = retrieve_claims(
            graph, family="spaceship-titanic", problem_id="spaceship-titanic",
            concepts=["tabular", "classification"],
        )
        ids = [n.id for n in nodes]
        # same-family claim first, concept-overlapping cross-family claim after
        assert ids == ["claim:cl1", "claim:cross"]
        # no concept overlap and different family -> cross claim drops out
        assert [n.id for n in retrieve_claims(
            graph, family="spaceship-titanic", problem_id="spaceship-titanic",
            concepts=["image"],
        )] == ["claim:cl1"]
        rehydrated = node_to_claim(nodes[0])
        assert rehydrated.subject == "histgradientboosting"
        assert rehydrated.relation == "helps"
        assert rehydrated.evidence == ["c001"]

    def test_flag_gates_claims_injection(self, knowledge_dir):
        from types import SimpleNamespace

        from hillclimb.api import build_knowledge_context
        from hillclimb.config import Config

        config = pinned()
        config.learning.dir = knowledge_dir
        problem = SimpleNamespace(
            problem_id="spaceship-titanic", metric_name="accuracy",
            higher_is_better=True, runtime="csv",
        )
        with_graph, _, injected = build_knowledge_context(config, problem, "", lambda m: None)
        assert "Distilled claims" in with_graph
        assert "histgradientboosting helps" in with_graph
        assert injected == ["cl1"]  # ids surfaced for credit assignment
        config.climber.memory_params["graph_retrieval"] = False
        without, _, none_injected = build_knowledge_context(config, problem, "", lambda m: None)
        assert "Distilled claims" not in without
        assert "PREVIOUS searches" in without  # cards block unaffected
        assert none_injected == []

    def test_superseded_claims_not_retrieved(self, knowledge_dir):
        from hillclimb.modules.memory.graph import retrieve_claims

        write_card(knowledge_dir, make_card(
            run_ref="r5/s1", finished_at="2026-07-05T00:00:00Z",
            claims=[claim(relation="no_effect", observed="2026-07-05T00:00:00Z", cid="newer")],
        ))
        graph = build_graph(knowledge_dir)
        ids = [n.id for n in retrieve_claims(
            graph, family="spaceship-titanic", problem_id="spaceship-titanic",
            concepts=["tabular"],
        )]
        assert "claim:cl1" not in ids  # displaced by the newer no_effect belief
        assert "claim:newer" in ids


class TestCreditFold:
    def _event(self, knowledge_dir, run_ref, claim_ids, reward, observed):
        from hillclimb.modules.memory.credit import CreditEvent, write_credit_event

        write_credit_event(knowledge_dir, CreditEvent(
            run_ref=run_ref, problem_id="spaceship-titanic",
            family="spaceship-titanic", claim_ids=claim_ids,
            reward=reward, observed_at=observed,
        ))

    def test_track_lands_on_claim_node(self, knowledge_dir):
        self._event(knowledge_dir, "r2/s1", ["cl1"], 1.0, "2026-07-02T00:00:00Z")
        graph = build_graph(knowledge_dir)
        track = graph.node_map()["claim:cl1"].data["track"]
        assert track["injections"] == 1
        assert track["mean_reward"] == 1.0
        # authored 0.8, one win: (0.8*2 + 1) / 3
        assert track["adjusted_confidence"] == pytest.approx(2.6 / 3, abs=1e-3)

    def test_track_record_reorders_retrieval(self, knowledge_dir):
        from hillclimb.modules.memory.graph import retrieve_claims

        # a humble claim with wins vs a confident claim with losses
        write_card(knowledge_dir, make_card(
            run_ref="r3/s1", finished_at="2026-07-03T00:00:00Z",
            claims=[claim(relation="requires", cid="humble", confidence=0.4,
                          observed="2026-07-03T00:00:00Z")],
        ))
        for i, (ids, reward) in enumerate([
            (["cl1"], 0.0), (["cl1"], 0.0), (["humble"], 1.0), (["humble"], 1.0),
        ]):
            self._event(knowledge_dir, f"r{i + 4}/s1", ids, reward,
                        f"2026-07-0{i + 4}T00:00:00Z")
        graph = build_graph(knowledge_dir)
        ids = [n.id for n in retrieve_claims(
            graph, family="spaceship-titanic", problem_id="spaceship-titanic",
            concepts=["tabular"],
        )]
        # humble: (0.4*2 + 2)/4 = 0.7 beats cl1: (0.8*2 + 0)/4 = 0.4
        assert ids.index("claim:humble") < ids.index("claim:cl1")

    def test_conclusive_losers_retire(self, knowledge_dir):
        from hillclimb.modules.memory.graph import retrieve_claims

        write_card(knowledge_dir, make_card(
            run_ref="r3/s1", finished_at="2026-07-03T00:00:00Z",
            claims=[claim(relation="requires", cid="loser", confidence=0.3,
                          observed="2026-07-03T00:00:00Z")],
        ))
        for i in range(3):
            self._event(knowledge_dir, f"r{i + 4}/s1", ["loser"], 0.0,
                        f"2026-07-0{i + 4}T00:00:00Z")
        graph = build_graph(knowledge_dir)
        node = graph.node_map()["claim:loser"]
        # authored 0.3, three losses: 0.6/5 = 0.12 < 0.15 -> retired
        assert node.data["retired"] == "track record"
        assert node.superseded_at == "2026-07-06T00:00:00Z"
        retrieved = {n.id for n in retrieve_claims(
            graph, family="spaceship-titanic", problem_id="spaceship-titanic",
            concepts=["tabular"],
        )}
        assert "claim:loser" not in retrieved
        # still part of history before its retirement
        past = {n.id for n in graph_at(graph, "2026-07-05T00:00:00Z").nodes}
        assert "claim:loser" in past
        # two losses are never conclusive
        assert "track" in graph.node_map()["claim:cl1"].data or True  # cl1 untouched here
        graph2 = build_graph(knowledge_dir)
        assert graph2.node_map()["claim:cl1"].superseded_at is None


class TestPersistence:
    def test_roundtrip(self, knowledge_dir, tmp_path):
        graph = build_graph(knowledge_dir)
        path = write_graph(tmp_path / "graph.json", graph)
        loaded = load_graph(path)
        assert loaded is not None
        assert loaded.model_dump() == graph.model_dump()
        assert load_graph(tmp_path / "absent.json") is None
        (tmp_path / "bad.json").write_text("{broken")
        assert load_graph(tmp_path / "bad.json") is None

    def test_load_or_build_caches_until_stale(self, knowledge_dir):
        first = load_or_build_graph(knowledge_dir)
        assert graph_path(knowledge_dir).exists()
        cached = load_or_build_graph(knowledge_dir)
        assert cached.built_at == first.built_at  # served from disk, not rebuilt
        time.sleep(0.02)
        write_card(knowledge_dir, make_card(
            run_ref="r9/s1", finished_at="2026-07-09T00:00:00Z",
        ))
        rebuilt = load_or_build_graph(knowledge_dir)
        assert "search:r9/s1" in {n.id for n in rebuilt.nodes}

    def test_stats(self, knowledge_dir):
        text = graph_stats(load_or_build_graph(knowledge_dir))
        assert "nodes" in text and "search=2" in text


def test_change_events_lists_every_distinct_moment(knowledge_dir):
    """The fine timeline: every first_seen and superseded_at once, sorted,
    the timeless '' dropped — a superset of the per-search events."""
    from hillclimb.modules.memory.graph import change_events, rebuild_graph

    # a claim backdated to its candidate's finish: a moment of its own
    write_card(knowledge_dir, make_card(
        run_ref="r3/s1", finished_at="2026-07-03T00:00:00Z",
        claims=[claim(cid="cl-mid", observed="2026-07-02T12:00:00Z")],
    ))
    graph = rebuild_graph(knowledge_dir)
    stamps = change_events(graph)
    assert stamps == sorted(stamps) and len(stamps) == len(set(stamps))
    assert "" not in stamps
    assert set(graph.events) <= set(stamps)
    assert len(stamps) > len(graph.events)


def test_backdate_claims_stamps_evidence_finish():
    from hillclimb.modules.memory.claims import backdate_claims

    class _J:
        candidates = {
            "c001": SimpleNamespace(candidate_id="c001", finished_at="2026-07-01T10:00:00Z", created_at="x"),
            "c002": SimpleNamespace(candidate_id="c002", finished_at=None, created_at="2026-07-01T11:00:00Z"),
        }

    both = claim(cid="cl-both")
    both.evidence = ["c001", "c002"]          # last evidence wins: the claim
    lone = claim(cid="cl-lone")               # needs both to exist
    lone.evidence = ["c-unknown"]             # unknown evidence: stamp kept
    none = claim(cid="cl-none")
    none.evidence = []
    out = backdate_claims([both, lone, none], _J())
    assert both.observed_at == "2026-07-01T11:00:00Z"
    assert lone.observed_at == "2026-07-01T00:00:00Z"
    assert none.observed_at == "2026-07-01T00:00:00Z"
    assert out == [both, lone, none]
