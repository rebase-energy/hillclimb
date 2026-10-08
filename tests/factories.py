"""Shared test factories for the Candidate → Trial → Replicate hierarchy.

`trial(...)` takes the FLAT keyword shape tests used before trials gained
parameters (`val_score`, `seed`, `metrics`, `holdout_score`, ...) and builds
one Trial holding one Replicate — replicate-level fields land on the
replicate, holdout fields and `params`/`index`/`is_best` on the trial. Pass
`replicates=[...]` (or `*scores`) for a trial with several seeded runs.
"""

from __future__ import annotations

from hillclimb.harness.candidate import Candidate, Replicate, Trial

TRIAL_FIELDS = {
    "index", "params", "is_best", "verdict", "unit_tests",
    "holdout_score", "holdout_error", "holdout_cpu_s",
}
REPLICATE_FIELDS = set(Replicate.model_fields)


def stamp(minute: float, day: str = "2026-08-22") -> str:
    """`10:MM` on a fixed day — the timestamp convention of the view tests."""
    whole = int(minute)
    seconds = int(round((minute - whole) * 60))
    return f"{day}T10:{whole:02d}:{seconds:02d}+00:00"


def replicate(score: float | None = None, **fields) -> Replicate:
    if score is not None:
        fields.setdefault("val_score", score)
    return Replicate(**fields)


def trial(*scores: float | None, replicates: list[Replicate] | None = None, **fields) -> Trial:
    """One trial. `*scores` → one replicate per score (seeds 0..n-1 when
    several); flat replicate fields (`val_score`, `seed`, `metrics`, ...)
    describe a single replicate; `started_at`/`finished_at` stamp both."""
    trial_kw = {k: v for k, v in fields.items() if k in TRIAL_FIELDS}
    rep_kw = {k: v for k, v in fields.items() if k in REPLICATE_FIELDS}
    unknown = set(fields) - TRIAL_FIELDS - REPLICATE_FIELDS
    if unknown:
        raise TypeError(f"trial(): unknown fields {sorted(unknown)}")
    if replicates is None:
        if scores:
            replicates = [
                Replicate(**{**rep_kw, "val_score": s, **({"seed": i} if len(scores) > 1 else {})})
                for i, s in enumerate(scores)
            ]
        else:
            replicates = [Replicate(**rep_kw)]
    for key in ("started_at", "finished_at"):
        if key in rep_kw:
            trial_kw[key] = rep_kw[key]
    trial_kw.setdefault("is_best", True)
    return Trial(replicates=replicates, **trial_kw)


def candidate(candidate_id: str, *scores: float | None, trials: list[Trial] | None = None, **fields) -> Candidate:
    """A candidate with one trial of `*scores` replicates (or the given
    `trials`); every other keyword is a Candidate field."""
    fields.setdefault("operator", "draft")
    fields.setdefault("status", "passing")
    if trials is None:
        trials = [trial(*scores)] if scores else []
    return Candidate(candidate_id=candidate_id, trials=trials, **fields)


def make_policy(ref, params: dict | None = None, *, priors: dict | None = None, base_dir=None):
    """A policy built the way a search builds it: through the climber `ref`
    names (a preset, a .py file, or a block), with `params` laid over its
    own and `priors` (what memory learned) under them."""
    from hillclimb.climber import load_climber, resolve_climber

    climber = load_climber(ref, base_dir) if isinstance(ref, str) else resolve_climber(ref, base_dir)
    return climber.build_loop(params=params or {}, priors=priors, log=lambda *_: None).policy


def name_climber(config, ref: str) -> None:
    """Name the config's policy or loop by a catalog file, a preset or one
    file, keeping the rest of its `climber:` block (params set before or
    after still apply). A composed file names its selector and prompts too,
    so those follow the brain (a block may not keep another's)."""
    from hillclimb.modules.spec import ClimberSpec, expand_name

    block = expand_name(ref)
    moved = {key: block.get(key) for key in ("name", "operator_policy", "loop", "selector_policy", "prompts", "operators")}
    moved["operator_params"] = block.get("operator_params") or {}
    config.climber = ClimberSpec.model_validate({**config.climber.block(), **moved})
