"""Consolidation: mechanical generalization, playbooks, injection replace rule."""

from __future__ import annotations

import pytest

from hillclimb.agents.fake import FakeAgent
from hillclimb.modules.memory.claims import Entity, ensure_concepts, load_consolidated_claims, save_entities
from hillclimb.config import Config
from hillclimb.modules.memory.consolidate import (
    GENERALIZE_MIN_FAMILIES,
    Playbook,
    consolidate,
    generalize_claims,
    install_playbook,
    load_playbooks,
    render_playbooks,
)
from hillclimb.modules.memory.graph import build_graph, rebuild_graph
from hillclimb.modules.memory.knowledge import write_card
from tests.test_graph import claim, make_card


@pytest.fixture
def knowledge_dir(tmp_path):
    kdir = tmp_path / "knowledge"
    ensure_concepts(kdir)
    save_entities(kdir, [
        Entity(slug="histgradientboosting", concepts=["decision-trees", "tabular"],
               first_seen="2026-07-01T00:00:00Z"),
    ])
    # the same claim asserted independently on two tabular families
    write_card(kdir, make_card(
        problem_id="comp-a", run_ref="r1/s1", finished_at="2026-07-01T00:00:00Z",
        claims=[claim(family="comp-a", cid="a1")],
    ))
    write_card(kdir, make_card(
        problem_id="comp-b", run_ref="r2/s1", finished_at="2026-07-02T00:00:00Z",
        claims=[claim(family="comp-b", cid="b1", observed="2026-07-02T00:00:00Z")],
    ))
    return kdir


class TestGeneralize:
    def test_multi_family_claim_lifts_to_concept(self, knowledge_dir):
        graph = build_graph(knowledge_dir)
        generalized = generalize_claims(graph, now="2026-07-10T00:00:00Z")
        assert len(generalized) == 1
        lifted = generalized[0]
        assert lifted.subject == "histgradientboosting"
        assert lifted.scope["concept"] == "decision-trees"  # primary concept
        assert lifted.scope["families"] == ["comp-a", "comp-b"]
        assert set(lifted.scope["sources"]) == {"a1", "b1"}
        assert lifted.evidence == ["c001"]  # union, deduped
        assert lifted.confidence == pytest.approx(0.8)
        # deterministic id -> re-running is idempotent
        again = generalize_claims(graph, now="2026-07-11T00:00:00Z")
        assert again[0].claim_id == lifted.claim_id

    def test_single_family_never_generalizes(self, tmp_path):
        kdir = tmp_path / "knowledge"
        ensure_concepts(kdir)
        save_entities(kdir, [Entity(slug="histgradientboosting", concepts=["tabular"])])
        write_card(kdir, make_card(claims=[claim(cid="only")]))
        assert generalize_claims(build_graph(kdir)) == []

    def test_measured_confidence_preferred(self, knowledge_dir):
        from hillclimb.modules.memory.credit import CreditEvent, write_credit_event

        # a1 has a poor measured record; the mean should use it, not 0.8
        write_credit_event(knowledge_dir, CreditEvent(
            run_ref="r3/s1", problem_id="comp-a", family="comp-a",
            claim_ids=["a1"], reward=0.0, observed_at="2026-07-03T00:00:00Z",
        ))
        generalized = generalize_claims(rebuild_graph(knowledge_dir))
        # a1 adjusted: (0.8*2 + 0)/3 = 0.533; b1 authored 0.8 -> mean 0.667
        assert generalized[0].confidence == pytest.approx(0.667, abs=1e-3)

    def test_consolidated_claims_join_graph_with_generalizes_edges(self, knowledge_dir):
        from hillclimb.modules.memory.claims import save_consolidated_claims

        generalized = generalize_claims(build_graph(knowledge_dir))
        save_consolidated_claims(knowledge_dir, generalized)
        graph = rebuild_graph(knowledge_dir)
        node = graph.node_map()[f"claim:{generalized[0].claim_id}"]
        assert node.data["consolidated"] is True
        assert "decision-trees" in node.concepts
        edges = {(e.src, e.dst, e.type) for e in graph.edges}
        assert (node.id, "claim:a1", "generalizes") in edges
        assert (node.id, "claim:b1", "generalizes") in edges
        assert (node.id, "concept:decision-trees", "applies_to") in edges
        # generalized claims never re-generalize
        assert generalize_claims(graph) == generalized or all(
            g.claim_id == generalized[0].claim_id for g in generalize_claims(graph)
        )


class TestPlaybooks:
    def test_install_and_load_roundtrip(self, tmp_path):
        path = install_playbook(tmp_path, Playbook(
            concept="tabular", body="Start with gradient boosting.",
            built_at="2026-07-10T00:00:00Z", source_claims=["a1", "b1"],
            source_searches=["r1/s1"],
        ))
        assert path.name == "tabular.md"
        loaded = load_playbooks(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].concept == "tabular"
        assert loaded[0].source_claims == ["a1", "b1"]
        assert "gradient boosting" in loaded[0].body
        # concept filter
        assert load_playbooks(tmp_path, ["image"]) == []
        assert len(load_playbooks(tmp_path, ["tabular", "image"])) == 1

    def test_corrupt_playbook_skipped(self, tmp_path):
        install_playbook(tmp_path, Playbook(concept="tabular", body="ok"))
        (tmp_path / "playbooks" / "broken.md").write_text("---\n][ nope\n---\nbody")
        loaded = load_playbooks(tmp_path)
        assert [p.concept for p in loaded] == ["tabular"]

    def test_render(self):
        text = render_playbooks([Playbook(concept="tabular", body="Do X.")])
        assert "## Playbook: tabular" in text and "Do X." in text
        assert render_playbooks([]) == ""

    def test_consolidate_end_to_end_with_fake_agent(self, knowledge_dir, monkeypatch):
        # third claim so the `tabular` concept crosses PLAYBOOK_MIN_CLAIMS
        write_card(knowledge_dir, make_card(
            problem_id="comp-c", run_ref="r3/s1", finished_at="2026-07-03T00:00:00Z",
            claims=[claim(family="comp-c", cid="c1", relation="requires",
                          observed="2026-07-03T00:00:00Z")],
        ))
        agent = FakeAgent()
        for _ in range(4):  # one playbook agent call per qualifying concept
            agent.queue(operator="consolidate",
                          files={"playbook.md": "Start with HGB; avoid physics sims."})
        monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
        summary = consolidate(knowledge_dir, Config(), lambda m: None)
        assert summary["generalized"]  # the 2-family lift happened
        assert load_consolidated_claims(knowledge_dir)
        assert summary["playbooks_written"]
        playbooks = load_playbooks(knowledge_dir)
        assert playbooks and all("HGB" in p.body for p in playbooks)
        assert all(p.source_claims for p in playbooks)

    def test_dry_run_writes_nothing(self, knowledge_dir, monkeypatch):
        monkeypatch.setattr(
            "hillclimb.api.get_agent",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no agent calls in dry run")),
        )
        summary = consolidate(knowledge_dir, Config(), lambda m: None, dry_run=True)
        assert summary["generalized"]
        assert load_consolidated_claims(knowledge_dir) == []
        assert load_playbooks(knowledge_dir) == []


class TestInjectionReplaceRule:
    def _problem(self):
        from types import SimpleNamespace

        return SimpleNamespace(problem_id="comp-a", metric_name="accuracy",
                               higher_is_better=True, runtime="csv")

    def test_playbook_replaces_claims_and_carries_source_credit(self, knowledge_dir):
        from hillclimb.api import build_knowledge_context

        config = Config()
        config.learning.dir = knowledge_dir
        # no playbook yet -> raw claims block
        text, _, ids = build_knowledge_context(config, self._problem(), "", lambda m: None)
        assert "Distilled claims" in text
        assert "a1" in ids
        # playbook covering the problem's `tabular` concept -> replace
        install_playbook(knowledge_dir, Playbook(
            concept="tabular", body="Always start with gradient boosting.",
            source_claims=["a1", "b1"],
        ))
        text, _, ids = build_knowledge_context(config, self._problem(), "", lambda m: None)
        assert "## Playbook: tabular" in text
        assert "Distilled claims" not in text
        assert ids == ["a1", "b1"]  # credit flows to the playbook's sources
        # flag off -> claims block again
        config.learning.playbooks = False
        text, _, ids = build_knowledge_context(config, self._problem(), "", lambda m: None)
        assert "Distilled claims" in text and "## Playbook" not in text
