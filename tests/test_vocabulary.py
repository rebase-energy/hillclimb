"""The word "workspace" is banned: a candidate's directory is its
`candidate_dir` (every level of the hierarchy is `<level>_dir`). The only
sanctioned survivor is the legacy `HILLCLIMB_WORKSPACE` env var and the
compat shims that map the old on-disk key onto `candidate_dir`. Third-party
CLI flag values we merely pass through (Codex's `--sandbox workspace-write`)
are their vocabulary, not ours, and are exempt."""

from __future__ import annotations

import re
from pathlib import Path

from hillclimb.harness.candidate import Candidate
from hillclimb.harness.status import CurrentCandidate

SRC = Path(__file__).resolve().parents[1] / "src" / "hillclimb"
ALLOWED = re.compile(
    r"HILLCLIMB_WORKSPACE|_legacy_workspace_key|\"workspace\"|`workspace`"
    r"|\"workspace-write\""  # codex --sandbox value
)


def test_no_workspace_in_source():
    offenders = []
    for path in SRC.rglob("*"):
        if path.suffix not in {".py", ".md", ".yaml"}:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"workspace", line, re.IGNORECASE) and not ALLOWED.search(line):
                offenders.append(f"{path.relative_to(SRC)}:{n}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_legacy_workspace_key_loads_as_candidate_dir():
    cand = Candidate.model_validate(
        {"candidate_id": "c001", "operator": "draft", "workspace": "/old/path"}
    )
    assert cand.candidate_dir == "/old/path"
    assert "workspace" not in cand.model_dump()
    cur = CurrentCandidate.model_validate(
        {"candidate_id": "c001", "operator": "draft", "phase": "agent", "workspace": "/old"}
    )
    assert cur.candidate_dir == "/old"


# Score direction is `higher_is_better` (hillclimb climbs). `lower_is_better`
# survives only in the load-time shim and where an external library's own
# field is read — those lines carry a `# legacy-key` marker.
LEGACY_DIRECTION_OK = re.compile(r"legacy-key|legacy_direction_key")


def test_no_lower_is_better_in_source():
    offenders = []
    for path in SRC.rglob("*"):
        if path.suffix not in {".py", ".md", ".yaml"}:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if "lower_is_better" in line and not LEGACY_DIRECTION_OK.search(line):
                offenders.append(f"{path.relative_to(SRC)}:{n}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_legacy_lower_is_better_key_loads_inverted():
    from hillclimb.modules.memory.knowledge import KnowledgeCard
    from hillclimb.harness.run import SearchMeta
    from hillclimb.harness.report import compact_report

    meta = SearchMeta.model_validate(
        {"search_id": "s", "run_id": "r", "problem": "p", "problem_id": "p", "backend": "dummy",
         "model": "m", "metric": "rmse", "lower_is_better": True}
    )
    assert meta.higher_is_better is False
    card = KnowledgeCard.model_validate({"problem_id": "p", "family": "f", "lower_is_better": False})
    assert card.higher_is_better is True
    assert compact_report({"version": 1, "lower_is_better": True})["higher_is_better"] is False
