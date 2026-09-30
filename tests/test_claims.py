"""Semantic layer: claim parsing, entity/concept registries, distill pass."""

from __future__ import annotations

from tests.factories import trial as mk_trial

import yaml

from hillclimb.agents.fake import FakeAgent
from hillclimb.harness.candidate import Candidate
from hillclimb.modules.memory.claims import (
    Claim,
    Concept,
    Entity,
    SEED_CONCEPTS,
    distill_claims,
    distill_claims_from_card,
    ensure_concepts,
    load_concepts,
    load_entities,
    make_claim_id,
    merge_concepts,
    merge_entities,
    parse_claims_file,
    problem_concepts,
    render_claims,
    save_entities,
    slugify,
)
from hillclimb.config import Config
from hillclimb.harness.journal import Journal
from hillclimb.modules.memory.knowledge import KnowledgeCard, distill_card, load_cards, write_card


def make_journal(tmp_path, entries) -> Journal:
    journal = Journal(tmp_path / "journal.jsonl")
    for e in entries:
        journal.candidate_result(Candidate(**e))
    return journal


class FakeProblem:
    problem_id = "spaceship-titanic"
    metric_name = "accuracy"
    higher_is_better = True


def scored(cid, op, val, summary="", candidate_dir="w"):
    return dict(
        candidate_id=cid, operator=op, status="passing", candidate_dir=candidate_dir,
        summary=summary, trials=[mk_trial(val_score=val)],
    )


CLAIMS_YAML = """
claims:
  - subject: HistGradientBoosting
    relation: helps
    confidence: 0.8
    evidence: [c002]
  - subject: feature-scaling
    relation: no_effect
    confidence: 0.6
    evidence: [c003]
  - subject: mystery
    relation: not-a-relation
  - subject: ""
    relation: helps
entities:
  - slug: histgradientboosting
    kind: technique
    aliases: [HistGradientBoosting, HGB]
    concepts: [decision-trees, tabular, not-a-concept]
  - slug: feature-scaling
    kind: practice
    concepts: [preprocessing]
proposed_concepts: []
"""


class TestParsing:
    def test_valid_claims_kept_invalid_dropped(self, tmp_path):
        path = tmp_path / "claims.yaml"
        path.write_text(CLAIMS_YAML)
        claims, entities, proposed = parse_claims_file(
            path, run_ref="r1/s1", family="spaceship-titanic",
            problem_id="spaceship-titanic", budget_s=600,
        )
        assert [c.subject for c in claims] == ["histgradientboosting", "feature-scaling"]
        assert claims[0].scope["family"] == "spaceship-titanic"
        assert claims[0].scope["budget_s"] == 600
        assert claims[0].claim_id  # deterministic id filled in
        assert len(entities) == 2
        assert proposed == []

    def test_missing_or_corrupt_file(self, tmp_path):
        assert parse_claims_file(
            tmp_path / "absent.yaml", run_ref="r", family="f", problem_id="p", budget_s=0
        ) == ([], [], [])
        bad = tmp_path / "claims.yaml"
        bad.write_text("{ not yaml: [")
        assert parse_claims_file(
            bad, run_ref="r", family="f", problem_id="p", budget_s=0
        ) == ([], [], [])

    def test_claim_id_deterministic(self):
        a = make_claim_id("x", "helps", "", {"family": "f"}, "r1/s1")
        b = make_claim_id("x", "helps", "", {"family": "f"}, "r1/s1")
        c = make_claim_id("x", "helps", "", {"family": "f"}, "r2/s1")
        assert a == b != c

    def test_slugify(self):
        assert slugify("  Hist Gradient Boosting! ") == "hist-gradient-boosting"
        assert slugify("XGBoost 2.0") == "xgboost-2.0"


class TestRegistries:
    def test_ensure_concepts_seeds_once(self, tmp_path):
        concepts = ensure_concepts(tmp_path)
        assert {c.slug for c in concepts} == {c.slug for c in SEED_CONCEPTS}
        # user edits survive: drop everything but one, re-ensure keeps it
        (tmp_path / "concepts.yaml").write_text(
            yaml.safe_dump({"schema_version": 1, "concepts": [
                {"slug": "tabular", "dimension": "modality"}]})
        )
        assert [c.slug for c in ensure_concepts(tmp_path)] == ["tabular"]

    def test_merge_entities_alias_dedup(self):
        existing = [Entity(slug="lightgbm", aliases=["LGBM"], concepts=["decision-trees"])]
        new = [
            Entity(slug="LGBM", aliases=["lightgbm-classifier"], concepts=["ensemble", "bogus"]),
            Entity(slug="torch", kind="library", concepts=["neural-networks"]),
        ]
        merged = merge_entities(existing, new, known_concepts={"decision-trees", "ensemble", "neural-networks"})
        assert [e.slug for e in merged] == ["lightgbm", "torch"]
        lgbm = merged[0]
        assert set(lgbm.concepts) == {"decision-trees", "ensemble"}  # bogus filtered
        assert "lightgbm-classifier" in lgbm.aliases
        assert merged[1].kind == "library"

    def test_merge_concepts_only_appends_new_as_proposed(self):
        existing = list(SEED_CONCEPTS)
        merged = merge_concepts(existing, [
            Concept(slug="tabular", dimension="modality"),         # dup — ignored
            Concept(slug="graph-neural-networks", dimension="model-family"),
        ])
        assert len(merged) == len(existing) + 1
        assert merged[-1].slug == "graph-neural-networks"
        assert merged[-1].proposed is True

    def test_registry_roundtrip_and_corrupt_skip(self, tmp_path):
        save_entities(tmp_path, [Entity(slug="a"), Entity(slug="b")])
        assert [e.slug for e in load_entities(tmp_path)] == ["a", "b"]
        (tmp_path / "entities.yaml").write_text("][ broken")
        assert load_entities(tmp_path) == []

    def test_problem_concepts(self):
        assert problem_concepts("csv", "accuracy") == ["tabular", "classification"]
        assert problem_concepts("emflow", "pinball") == ["time-series", "forecasting"]
        assert problem_concepts("csv", "rmse") == ["tabular", "regression"]


class TestDistillPass:
    def _card(self, journal):
        return distill_card(
            journal, problem=FakeProblem(), run_ref="r1/s1", budget_s=600
        )

    def test_distill_claims_end_to_end(self, tmp_path, monkeypatch):
        ws = tmp_path / "cand"
        ws.mkdir()
        (ws / "solution.py").write_text("from sklearn.ensemble import HistGradientBoostingClassifier\n")
        journal = make_journal(tmp_path, [
            scored("c001", "draft", 0.74, "baseline GBM", str(ws)),
            scored("c002", "improve", 0.80, "HGB tuned", str(ws)),
        ])
        card = self._card(journal)
        agent = FakeAgent()
        agent.queue(operator="distill", files={"claims.yaml": CLAIMS_YAML})
        monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)

        knowledge_dir = tmp_path / "knowledge"
        search_dir = tmp_path / "search"
        search_dir.mkdir()
        claims = distill_claims(
            journal, problem=FakeProblem(), card=card, search_dir=search_dir,
            knowledge_dir=knowledge_dir, config=Config(), log=lambda m: None,
        )
        assert [c.subject for c in claims] == ["histgradientboosting", "feature-scaling"]
        # the distill request carried the digest and the solution excerpt
        request = agent.requests[0]
        assert request.operator == "distill"
        assert request.model == "haiku"  # default distill route
        assert "HGB tuned" in request.prompt
        assert "HistGradientBoostingClassifier" in request.prompt
        # registries were created and merged
        assert (knowledge_dir / "concepts.yaml").exists()
        assert {e.slug for e in load_entities(knowledge_dir)} == {
            "histgradientboosting", "feature-scaling",
        }
        # prompt written next to the pass's artifacts
        assert (search_dir / "distill" / "prompt.md").exists()
        # a re-distill from the CLI never writes to the journal
        assert not [r for r in journal.backend.records() if r.get("event") == "memory_agent_call"]

    def test_the_engines_distill_pass_records_what_it_spent(self, tmp_path, monkeypatch):
        """The pass runs outside the harness, so its cost is in no candidate:
        it leaves an audit line instead — memory's spend, never the search's."""
        from hillclimb.harness.budget import journal_spend

        journal = make_journal(tmp_path, [scored("c001", "draft", 0.74, "baseline GBM", str(tmp_path))])
        card = self._card(journal)
        agent = FakeAgent()
        agent.queue(operator="distill", files={"claims.yaml": CLAIMS_YAML}, cost_usd=0.07)
        monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
        search_dir = tmp_path / "search"
        search_dir.mkdir()
        before = journal_spend(journal)
        distill_claims(
            journal, problem=FakeProblem(), card=card, search_dir=search_dir,
            knowledge_dir=tmp_path / "knowledge", config=Config(), log=lambda m: None,
            record_cost=True,
        )
        [line] = [r for r in journal.backend.records() if r.get("event") == "memory_agent_call"]
        assert line["memory_pass"] == "distill" and line["ok"] is True
        assert line["cost_usd"] == 0.07
        assert journal_spend(journal) == before  # budgets count candidates only
        assert list(Journal(journal.path).candidates) == ["c001"]  # replay skips the line

    def test_distill_route_override(self, tmp_path, monkeypatch):
        journal = make_journal(tmp_path, [scored("c001", "draft", 0.7)])
        card = self._card(journal)
        agent = FakeAgent()
        agent.queue(operator="distill", files={"claims.yaml": "claims: []"})
        monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
        config = Config.model_validate({"routing": {"distill": {"model": "opus"}}})
        distill_claims(
            journal, problem=FakeProblem(), card=card, search_dir=tmp_path / "s",
            knowledge_dir=tmp_path / "k", config=config, log=lambda m: None,
        )
        assert agent.requests[0].model == "opus"

    def test_failed_agent_yields_no_claims(self, tmp_path, monkeypatch):
        journal = make_journal(tmp_path, [scored("c001", "draft", 0.7)])
        card = self._card(journal)
        agent = FakeAgent()
        agent.queue(operator="distill", result={"ok": False, "error_kind": "timeout"})
        monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
        claims = distill_claims(
            journal, problem=FakeProblem(), card=card, search_dir=tmp_path / "s",
            knowledge_dir=tmp_path / "k", config=Config(), log=lambda m: None,
        )
        assert claims == []

    def test_unregistered_subject_dropped(self, tmp_path, monkeypatch):
        journal = make_journal(tmp_path, [scored("c001", "draft", 0.7)])
        card = self._card(journal)
        agent = FakeAgent()
        # claim about a subject never declared in entities -> dropped
        agent.queue(operator="distill", files={"claims.yaml": (
            "claims:\n  - subject: phantom\n    relation: helps\nentities: []\n"
        )})
        monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
        claims = distill_claims(
            journal, problem=FakeProblem(), card=card, search_dir=tmp_path / "s",
            knowledge_dir=tmp_path / "k", config=Config(), log=lambda m: None,
        )
        assert claims == []

    def test_backfill_from_card(self, tmp_path, monkeypatch):
        card = KnowledgeCard(problem_id="p", family="p", metric="rmse", run_ref="r1/s1")
        agent = FakeAgent()
        agent.queue(operator="distill", files={"claims.yaml": CLAIMS_YAML})
        monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
        claims = distill_claims_from_card(
            card, work_dir=tmp_path / "w", knowledge_dir=tmp_path / "k",
            config=Config(), log=lambda m: None,
        )
        assert len(claims) == 2
        assert "card data only" in agent.requests[0].prompt


class TestCardIntegration:
    def test_card_with_claims_roundtrips(self, tmp_path):
        card = KnowledgeCard(
            problem_id="p", family="fam", run_ref="r1/s1",
            claims=[Claim(subject="x", relation="helps", claim_id="abc")],
        )
        write_card(tmp_path, card)
        loaded = load_cards(tmp_path, problem_id="p", family="fam")
        assert len(loaded) == 1
        assert loaded[0].claims[0].subject == "x"

    def test_old_card_without_claims_loads(self, tmp_path):
        # a card written before the claims field existed has no `claims` key
        card = KnowledgeCard(problem_id="p", family="fam", run_ref="r1/s1")
        path = write_card(tmp_path, card)
        data = yaml.safe_load(path.read_text())
        data.pop("claims", None)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        loaded = load_cards(tmp_path, problem_id="p", family="fam")
        assert loaded[0].claims == []


class TestRender:
    def test_render_claims(self):
        claims = [
            Claim(subject="hgb", relation="helps", confidence=0.8,
                  scope={"problem_id": "spaceship-titanic"}),
            Claim(subject="scaling", relation="no_effect", confidence=0.6),
        ]
        text = render_claims(claims)
        assert "hgb helps" in text
        assert "spaceship-titanic" in text
        assert "80%" in text
        assert render_claims([]) == ""
