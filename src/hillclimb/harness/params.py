"""Engine-side view of a candidate's declared parameter space (the
`params.json` contract lives in `spaces.py`, stdlib-only, shared with the
runtime; this module is what the engine and tuners import).

The engine never imports agent code: it reads the declaration file, writes a
values document per trial, and hands typed `ParamSpec`s to the tuner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hillclimb import spaces
from hillclimb.spaces import PARAMS_FILE, ParamsError, fold_defaults, with_values

REASON_CHARS = 300


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str  # float | int | categorical
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: float | None = None
    choices: tuple = ()
    default: object = None


ParamSpace = dict[str, ParamSpec]


@dataclass(frozen=True)
class ParamsFile:
    """A validated declaration: the raw document (written back into trial
    dirs) and its typed view (what tuners sample over)."""

    raw: dict
    space: ParamSpace

    @property
    def defaults(self) -> dict:
        return {name: spec.default for name, spec in self.space.items()}


def parse_space(raw: dict) -> ParamSpace:
    violations = spaces.check_params_space(raw)
    if violations:
        raise ParamsError("; ".join(str(v) for v in violations))
    space: ParamSpace = {}
    for name, spec in raw.items():
        space[name] = ParamSpec(
            name=name,
            type=spec["type"],
            low=spec.get("low"),
            high=spec.get("high"),
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
            choices=tuple(spec.get("choices", ())),
            default=spec["default"],
        )
    return space


def read_candidate_space(candidate_dir: Path) -> tuple[ParamsFile | None, str | None]:
    """(declaration, None) for a valid params.json, (None, None) when the
    candidate declares nothing, (None, reason) when the file is malformed —
    never raises: a broken declaration makes the candidate untunable, not
    the search crash."""
    path = Path(candidate_dir) / PARAMS_FILE
    if not path.exists():
        return None, None
    try:
        raw = json.loads(path.read_text())
        space = parse_space(raw)
    except (OSError, ValueError, ParamsError) as exc:
        reason = " ".join(str(exc).split())
        return None, reason[:REASON_CHARS]
    return ParamsFile(raw=raw, space=space), None


def coerce(space: ParamSpace, values: dict) -> dict:
    """JSON round-trip hygiene: ints stay ints, floats floats, categoricals
    the declared object; unknown names are dropped, missing ones take the
    default."""
    out = {}
    for name, spec in space.items():
        value = values.get(name, spec.default)
        if spec.type == "int":
            out[name] = int(round(float(value)))
        elif spec.type == "float":
            out[name] = float(value)
        else:
            match = next((c for c in spec.choices if c == value and type(c) is type(value)), None)
            if match is None:
                match = next((c for c in spec.choices if c == value), spec.default)
            out[name] = match
    return out


def write_trial_params(trial_dir: Path, declaration: ParamsFile, values: dict) -> Path:
    path = Path(trial_dir) / PARAMS_FILE
    path.write_text(json.dumps(with_values(declaration.raw, values), indent=2, sort_keys=True) + "\n")
    return path


def write_inherited_params(candidate_dir: Path, declaration: ParamsFile, values: dict) -> Path:
    """A child candidate starts from its parent's best-found values as the
    new defaults."""
    path = Path(candidate_dir) / PARAMS_FILE
    path.write_text(json.dumps(fold_defaults(declaration.raw, values), indent=2, sort_keys=True) + "\n")
    return path
