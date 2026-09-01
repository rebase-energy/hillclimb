"""Paper ingestion: PDF -> distilled claims -> graph -> retrieval."""

from __future__ import annotations

from hillclimb.backends.fake import FakeBackend
from hillclimb.config import Config
from hillclimb.papers import distill_paper, load_papers, paper_scope

PAPER_CLAIMS_YAML = """\
paper_title: "Probabilistic wind power forecasting with analog ensembles"
claims:
  - subject: analog-ensemble
    relation: helps
    object: ""
    confidence: 0.7
    evidence: [p4, p7]
  - subject: analog-ensemble
    relation: outperforms
    object: quantile-regression
    confidence: 0.6
    evidence: [p9]
entities:
  - slug: analog-ensemble
    kind: technique
    aliases: [AnEn]
    concepts: [probabilistic, forecasting]
  - slug: quantile-regression
    kind: technique
    concepts: [probabilistic]
proposed_concepts: []
"""


def test_paper_scope_parses_targets():
    assert paper_scope("emflow://gefcom2014:wind") == (
        "gefcom2014", "gefcom2014-wind", "emflow://gefcom2014:wind",
    )
    assert paper_scope("spaceship-titanic") == ("spaceship-titanic", "spaceship-titanic", "")
    assert paper_scope(None) == ("", "", "")


def _fake_pdf(tmp_path, name="anen_wind.pdf", content=b"%PDF-1.4 fake"):
    pdf = tmp_path / name
    pdf.write_bytes(content)
    return pdf


def _ingest(tmp_path, monkeypatch, *, backend=None, force=False):
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "cache"))
    if backend is None:
        backend = FakeBackend()
        backend.queue(operator="paper", files={"claims.yaml": PAPER_CLAIMS_YAML})
    monkeypatch.setattr("hillclimb.api.get_backend", lambda *a, **k: backend)
    knowledge_dir = tmp_path / "knowledge"
    record = distill_paper(
        Config(), knowledge_dir, _fake_pdf(tmp_path),
        problem="emflow://gefcom2014:wind", force=force, log=lambda m: None,
    )
    return record, knowledge_dir, backend


def test_distill_paper_end_to_end(tmp_path, monkeypatch):
    record, knowledge_dir, backend = _ingest(tmp_path, monkeypatch)

    assert record is not None
    assert record.slug == "anen_wind"
    assert record.title.startswith("Probabilistic wind")
    assert record.family == "gefcom2014" and record.problem_id == "gefcom2014-wind"
    assert [c.subject for c in record.claims] == ["analog-ensemble", "analog-ensemble"]
    assert record.claims[0].scope["family"] == "gefcom2014"
    assert record.claims[0].evidence == ["p4", "p7"]  # page provenance
    # the PDF was staged into the agent's work dir and the prompt names it
    request = backend.requests[0]
    assert request.operator == "paper"
    assert request.model == "sonnet"  # comprehension work, not the distill small model
    assert "anen_wind.pdf" in request.prompt
    assert (request.candidate_dir / "anen_wind.pdf").exists()
    # persisted record and merged registries
    assert (knowledge_dir / "papers" / "anen_wind.yaml").exists()
    from hillclimb.claims import load_entities
    assert {e.slug for e in load_entities(knowledge_dir)} >= {"analog-ensemble", "quantile-regression"}
    assert load_papers(knowledge_dir)[0].sha256 == record.sha256


def test_reingest_is_cached_by_content_hash(tmp_path, monkeypatch):
    record, knowledge_dir, backend = _ingest(tmp_path, monkeypatch)
    again = distill_paper(
        Config(), knowledge_dir, _fake_pdf(tmp_path),
        problem="emflow://gefcom2014:wind", log=lambda m: None,
    )
    assert again is not None and again.sha256 == record.sha256
    assert len(backend.requests) == 1  # no second agent call


def test_failed_agent_writes_nothing(tmp_path, monkeypatch):
    backend = FakeBackend()
    backend.queue(operator="paper", result={"ok": False, "error_kind": "error"})
    record, knowledge_dir, _ = _ingest(tmp_path, monkeypatch, backend=backend)
    assert record is None
    assert not (knowledge_dir / "papers").exists()


def test_paper_claims_enter_graph_and_retrieval(tmp_path, monkeypatch):
    record, knowledge_dir, _ = _ingest(tmp_path, monkeypatch)
    from hillclimb.graph import build_graph, retrieve_claims

    graph = build_graph(knowledge_dir)
    nodes = graph.node_map()
    paper_node = nodes["paper:anen_wind"]
    assert paper_node.type == "paper"
    assert paper_node.data["n_claims"] == 2
    # scoped anchors exist BEFORE any search ran on the problem
    assert "problem:gefcom2014-wind" in nodes and "family:gefcom2014" in nodes
    kinds = {(e.src, e.type, e.dst) for e in graph.edges}
    claim_id = f"claim:{record.claims[0].claim_id}"
    assert (claim_id, "derived_from", "paper:anen_wind") in kinds
    assert (claim_id, "applies_to", "problem:gefcom2014-wind") in kinds
    # retrieval ranks paper claims like any other family-scoped claim
    hits = retrieve_claims(
        graph, family="gefcom2014", problem_id="gefcom2014-wind",
        concepts=["forecasting"],
    )
    assert claim_id in {n.id for n in hits}
