"""The vocabulary gepa's adapter and proposer share with the loop."""

from __future__ import annotations

import json

from hillclimb.sdk import source_hash  # noqa: F401 — re-exported: the identity of a proposal

COMPONENT = "solution.py"
FEEDBACK_CAP = 16_384  # bytes of serialized reflective feedback


class ProposerError(Exception):
    """This round produced no proposal. gepa logs it and moves on; whether the
    search should stop is the harness's call (its stop callback says so)."""


def feedback_json(reflective_dataset) -> str:
    try:
        text = json.dumps(reflective_dataset, indent=2, default=str)
    except (TypeError, ValueError):
        text = json.dumps({"feedback": str(reflective_dataset)[:FEEDBACK_CAP]})
    return text[:FEEDBACK_CAP]
