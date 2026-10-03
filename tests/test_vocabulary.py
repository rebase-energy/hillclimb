"""The word "workspace" is banned: a candidate's directory is its
`candidate_dir` (every level of the hierarchy is `<level>_dir`). The only
sanctioned survivors are the compat shims that map the old on-disk key onto
`candidate_dir`. Third-party
CLI flag values we merely pass through (Codex's `--sandbox workspace-write`)
are their vocabulary, not ours, and are exempt."""

from __future__ import annotations

import re
from pathlib import Path

from hillclimb.harness.candidate import Candidate
from hillclimb.harness.status import CurrentCandidate

SRC = Path(__file__).resolve().parents[1] / "src" / "hillclimb"
ALLOWED = re.compile(
    r"_legacy_workspace_key|\"workspace\"|`workspace`"
    r"|\"workspace-write\""  # codex --sandbox value
    r"|sandbox_workspace_write\."  # codex config key
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


# 0.6 renamed these, with no aliases. They survive only where an old spelling
# is read (the journal key, the sdk's "renamed" table).
RENAMED_IN_06 = re.compile(
    r"\b(SearchPolicy|SearchLoop|PolicyInput|PolicyJournal|Preparation|OperatorRequest|OperatorResult|policy_meta)\b"
)
RENAMED_OK = re.compile(r"_legacy_policy_meta_key|\"policy_meta\"|\"(SearchPolicy|SearchLoop|PolicyInput|PolicyJournal|Preparation)\": \(\"")
META_PROBLEM = SRC.parents[1] / "problems" / "meta-heilbronn"


def test_no_pre_06_names_in_source():
    offenders = []
    for root in (SRC, META_PROBLEM):
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".md", ".yaml"}:
                continue
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if RENAMED_IN_06.search(line) and not RENAMED_OK.search(line):
                    offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_legacy_policy_meta_key_loads_as_climber_meta():
    cand = Candidate.model_validate(
        {"candidate_id": "c001", "operator": "draft", "policy_meta": {"island": 1}}
    )
    assert cand.climber_meta == {"island": 1}
    assert "policy_meta" not in cand.model_dump()


def test_a_renamed_sdk_name_says_where_it_went():
    import pytest

    with pytest.raises(ImportError, match="PolicyInput was renamed SearchState in hillclimb 0.6"):
        from hillclimb.sdk import PolicyInput  # noqa: F401
    with pytest.raises(ImportError, match="ROLES was renamed OPERATOR_KINDS in hillclimb 0.7"):
        from hillclimb.sdk import ROLES  # noqa: F401


# 0.7: what an operator's candidates are is its `kind` (`role` is what a
# climber plays in a search: solver | improver). The old spelling survives
# only where it is read.
OPERATOR_ROLE_NAMES = re.compile(r"\b(ROLES|RESERVED_ROLES|role_of)\b")
OPERATOR_ROLE_OK = re.compile(r"\"ROLES\": \(\"")


def test_no_operator_role_names_in_source():
    offenders = []
    for root in (SRC, META_PROBLEM):
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".md", ".yaml"}:
                continue
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if OPERATOR_ROLE_NAMES.search(line) and not OPERATOR_ROLE_OK.search(line):
                    offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_legacy_role_key_loads_as_kind():
    cand = Candidate.model_validate({"candidate_id": "c001", "operator": "improve", "role": "refine"})
    assert cand.kind == "refine"
    assert "role" not in cand.model_dump()


def test_an_operator_written_with_role_still_loads():
    from hillclimb.sdk import Attempt, Operator

    class Old(Operator):
        name, role = "old", "refine"

        def prepare(self, ctx):
            return Attempt(prompt="")

    assert Old.kind == "refine"


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
        {"search_id": "s", "run_id": "r", "problem": "p", "problem_id": "p", "agent": "dummy",
         "model": "m", "metric": "rmse", "lower_is_better": True}
    )
    assert meta.higher_is_better is False
    card = KnowledgeCard.model_validate({"problem_id": "p", "family": "f", "lower_is_better": False})
    assert card.higher_is_better is True
    assert compact_report({"version": 1, "lower_is_better": True})["higher_is_better"] is False
