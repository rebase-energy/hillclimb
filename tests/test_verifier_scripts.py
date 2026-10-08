"""The starter verifiers must not `exec` their scorer: on macOS an exec
drops the CPU time of the children the shell had already waited for — the
solution run, which is where the work is — so the journal would record the
scorer's fraction of a second as the trial's whole cost."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STARTERS = sorted((REPO / "problems").glob("*/verifier.sh"))  # the catalog: one copy


def test_no_starter_verifier_execs_its_scorer():
    assert STARTERS
    offenders = [
        str(path.relative_to(REPO))
        for path in STARTERS
        if any(line.startswith("exec ") for line in path.read_text().splitlines())
    ]
    assert offenders == [], f"call the scorer plainly, never exec it: {offenders}"
