"""Papers as a knowledge source: distill a PDF into typed claims that ride
the existing pipeline (claims -> graph -> retrieval -> credit).

`hillclimb paper add <pdf>` runs a one-shot distiller agent over the PDF and
writes `knowledge/papers/<slug>.yaml` — git-versioned YAML like the rest of
the knowledge dir. The graph links each claim to its `paper:<slug>` node the
way search claims link to their search node, so retrieval ranks and the
credit economy judges paper-derived claims exactly like search-learned ones:
papers have to earn their place in prompts too.

The record's schema key is `paper_schema_version` (not `schema_version`) so
`graph.load_all_cards`, which globs every family subdir, naturally skips the
papers dir."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from hillclimb.candidate import utcnow
from hillclimb.claims import (
    CLAIMS_FILENAME,
    Claim,
    absorb_parsed,
    invoke_knowledge_agent,
    parse_claims_file,
    slugify,
)
from hillclimb.config import Config
from hillclimb.knowledge import problem_family

PAPERS_DIRNAME = "papers"
PAPER_SCHEMA_VERSION = 1
# reading a full PDF and extracting method structure is real comprehension
# work — do not down-route it to the distill pass's small model
DEFAULT_PAPER_MODEL = "sonnet"
# a long paper takes many Read calls before any claim is written
PAPER_TIMEOUT_S = 900


class PaperRecord(BaseModel):
    paper_schema_version: int = PAPER_SCHEMA_VERSION
    slug: str
    title: str = ""
    file: str = ""  # the ingested PDF's filename (the PDF itself is not required after distillation)
    sha256: str = ""
    family: str = ""  # scope the claims apply to; empty = global, matched by concept
    problem_id: str = ""
    target: str = ""
    added_at: str = Field(default_factory=utcnow)
    claims: list[Claim] = Field(default_factory=list)


def papers_dir(knowledge_dir: Path) -> Path:
    return knowledge_dir / PAPERS_DIRNAME


def paper_path(knowledge_dir: Path, slug: str) -> Path:
    return papers_dir(knowledge_dir) / f"{slug}.yaml"


def load_papers(knowledge_dir: Path) -> list[PaperRecord]:
    """Every ingested paper, oldest first. Corrupt or version-mismatched
    records are skipped, as everywhere in the knowledge dir."""
    records = []
    directory = papers_dir(knowledge_dir)
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
            if data.get("paper_schema_version") != PAPER_SCHEMA_VERSION:
                continue
            records.append(PaperRecord.model_validate(data))
        except Exception:  # noqa: BLE001
            continue
    records.sort(key=lambda r: r.added_at)
    return records


def write_paper(knowledge_dir: Path, record: PaperRecord) -> Path:
    path = paper_path(knowledge_dir, record.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(record.model_dump(), sort_keys=False, allow_unicode=True))
    return path


def paper_scope(problem: str | None) -> tuple[str, str, str]:
    """(family, problem_id, target) for a --problem argument: an
    `emflow://pkg:name` target, a bare local problem id, or None (global —
    the claims then reach searches through concept overlap only)."""
    if not problem:
        return "", "", ""
    if problem.startswith("emflow://"):
        rest = problem.removeprefix("emflow://")
        pkg, _, name = rest.partition(":")
        problem_id = f"{pkg}-{name}" if name else pkg
        return problem_family(problem_id, problem), problem_id, problem
    return problem_family(problem), problem, ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def distill_paper(
    config: Config,
    knowledge_dir: Path,
    pdf: Path,
    *,
    problem: str | None = None,
    force: bool = False,
    log=print,
) -> PaperRecord | None:
    """Ingest one PDF: a one-shot agent reads it and writes claims, which are
    absorbed into the registries and saved as `papers/<slug>.yaml`. Cached by
    content hash — re-adding the same PDF is a no-op without --force. Returns
    None when the distill agent failed (nothing is written then)."""
    from hillclimb.claims import _concepts_block, _entities_block, ensure_concepts, load_entities
    from hillclimb.claims import CLAIM_RELATIONS
    from hillclimb.project import machine_cache_dir
    from hillclimb.prompts.render import render

    pdf = pdf.absolute()
    slug = slugify(pdf.stem)
    sha = _sha256(pdf)
    existing_path = paper_path(knowledge_dir, slug)
    if existing_path.exists() and not force:
        try:
            existing = PaperRecord.model_validate(yaml.safe_load(existing_path.read_text()) or {})
        except Exception:  # noqa: BLE001
            existing = None
        if existing is not None and existing.sha256 == sha:
            log(f"paper: {pdf.name} already ingested as {slug} ({len(existing.claims)} claims) — use --force to re-distill")
            return existing

    family, problem_id, target = paper_scope(problem)
    work_dir = machine_cache_dir() / "paper-distill" / slug
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, work_dir / pdf.name)

    scope_line = (
        f", specifically searches on **{problem_id or family}**" if (family or problem_id) else ""
    )
    prompt = render(
        "paper",
        pdf_name=pdf.name,
        scope_line=scope_line,
        entities_block=_entities_block(load_entities(knowledge_dir)),
        concepts_block=_concepts_block(ensure_concepts(knowledge_dir)),
        relations=" | ".join(CLAIM_RELATIONS),
        claims_filename=CLAIMS_FILENAME,
    )
    log(f"paper: distilling {pdf.name} ...")
    result = invoke_knowledge_agent(
        config,
        operator="paper",
        prompt=prompt,
        work_dir=work_dir,
        timeout_s=PAPER_TIMEOUT_S,
        default_model=DEFAULT_PAPER_MODEL,
    )
    if not result.ok:
        log(f"paper: distill agent failed ({result.error_kind}): {result.error_message}")
        return None

    claims_file = work_dir / CLAIMS_FILENAME
    claims, new_entities, proposed = parse_claims_file(
        claims_file,
        run_ref=f"paper/{slug}",
        family=family,
        problem_id=problem_id,
        budget_s=0,
    )
    kept = absorb_parsed(knowledge_dir, claims, new_entities, proposed, log)
    title = ""
    try:
        title = str((yaml.safe_load(claims_file.read_text()) or {}).get("paper_title") or "")
    except Exception:  # noqa: BLE001
        pass
    record = PaperRecord(
        slug=slug, title=title, file=pdf.name, sha256=sha,
        family=family, problem_id=problem_id, target=target,
        claims=kept,
    )
    path = write_paper(knowledge_dir, record)
    log(f"paper: {slug} -> {len(kept)} claim(s) in {path}")
    return record
