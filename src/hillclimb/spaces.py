"""Gymnasium-inspired I/O spaces: a problem's optional, machine-checkable
declaration of what `solution.py` must produce (and, descriptively, what it
is given).

A problem dir may ship an `interface.py` exposing module-level `output`
(and optionally `inputs`) built from this vocabulary:

    from hillclimb import spaces

    output = spaces.Table(
        "submission.csv",
        columns={
            "id": spaces.Int(low=0, high=25, unique=True),
            "x": spaces.Float(low=0.0, high=1.0),
            "y": spaces.Float(low=0.0, high=1.0),
            "r": spaces.Float(low=0.0),
        },
        n_rows=26,
    )

    if __name__ == "__main__":
        raise SystemExit(spaces.main(output))

One declaration feeds three consumers: `describe()` renders the contract
into agent prompts, `check()`/`check_value()` give verifiers and agents
located format violations instead of stack traces, and `sample()` writes a
format-valid random artifact (Gym's `space.sample()`) for authoring lints.
The engine never runs a check itself — a problem is still its verifier;
this module is a library the verifier and the agent call voluntarily.

This file is copied verbatim into the runtime-venv import shim
(`runtime.ensure_interface_shim`), so it must stay self-contained:
stdlib-only top-level imports, numpy/pandas lazy-imported inside methods
(per-problem venvs may lack them), dataclasses rather than pydantic
(the runtime venvs ship no pydantic), and no other hillclimb imports.
"""

from __future__ import annotations

import csv as _csv
import hashlib
import importlib.util
import inspect
import json as _json
import os as _os
import random as _random
from dataclasses import dataclass, field, replace
from pathlib import Path


class InterfaceError(RuntimeError):
    """An interface.py that cannot be loaded or is malformed."""


@dataclass(frozen=True)
class Violation:
    """One located format error. `path` pins where ("submission.csv[r]",
    "solution.pack"), `message` says what, `expected` restates the rule."""

    path: str
    message: str
    expected: str
    actual: str | None = None

    def __str__(self) -> str:
        suffix = f" — got {self.actual}" if self.actual else ""
        return f"{self.path}: {self.message} (expected {self.expected}){suffix}"


def _fmt_bound(low, high) -> str:
    if low is not None and high is not None:
        return f"in [{low:g}, {high:g}]"
    if low is not None:
        return f">= {low:g}"
    if high is not None:
        return f"<= {high:g}"
    return "unbounded"


def _fmt_values(values) -> str:
    if isinstance(values, range):
        return f"{values.start}..{values.stop - values.step}" if len(values) else "(empty)"
    items = sorted(values)
    if len(items) > 8:
        return f"{{{', '.join(str(v) for v in items[:8])}, …}} ({len(items)} values)"
    return f"{{{', '.join(str(v) for v in items)}}}"


@dataclass(frozen=True)
class Int:
    """Integer column: optional bounds, an explicit value set, uniqueness."""

    low: int | None = None
    high: int | None = None
    values: object = None  # any container with __contains__/__iter__, e.g. range(26)
    unique: bool = False

    def describe(self) -> str:
        parts = ["int"]
        if self.values is not None:
            parts.append(f"values {_fmt_values(self.values)}")
        elif self.low is not None or self.high is not None:
            parts.append(_fmt_bound(self.low, self.high))
        if self.unique:
            parts.append("unique")
        return ", ".join(parts)

    def check_series(self, name: str, series) -> list[Violation]:
        import pandas as pd

        expected = self.describe()
        out: list[Violation] = []
        if not pd.api.types.is_integer_dtype(series):
            # a float column of whole numbers still violates: verifiers
            # downstream index/join on it, and 3.0 != 3 across tools
            return [Violation(name, "column is not integer-typed", expected, str(series.dtype))]
        if self.values is not None:
            bad = series[~series.isin(list(self.values))]
            if len(bad):
                out.append(Violation(
                    name, f"{len(bad)} values outside the allowed set "
                    f"(first at row {bad.index[0]}: {bad.iloc[0]})", expected,
                ))
        if self.low is not None and (series < self.low).any():
            bad = series[series < self.low]
            out.append(Violation(
                name, f"{len(bad)} values below low={self.low:g} "
                f"(first at row {bad.index[0]}: {bad.iloc[0]})", expected,
            ))
        if self.high is not None and (series > self.high).any():
            bad = series[series > self.high]
            out.append(Violation(
                name, f"{len(bad)} values above high={self.high:g} "
                f"(first at row {bad.index[0]}: {bad.iloc[0]})", expected,
            ))
        if self.unique and series.duplicated().any():
            dupes = series[series.duplicated()]
            out.append(Violation(
                name, f"{len(dupes)} duplicate values "
                f"(first at row {dupes.index[0]}: {dupes.iloc[0]})", expected,
            ))
        return out

    def sample_values(self, n: int, rng: _random.Random) -> list[int]:
        if self.values is not None:
            pool = list(self.values)
            if self.unique:
                if n > len(pool):
                    raise InterfaceError(
                        f"cannot sample {n} unique values from a set of {len(pool)}"
                    )
                return rng.sample(pool, n)
            return [rng.choice(pool) for _ in range(n)]
        low = self.low if self.low is not None else 0
        high = self.high if self.high is not None else low + max(n, 1)
        if self.unique:
            if n > high - low + 1:
                raise InterfaceError(f"cannot sample {n} unique ints from [{low}, {high}]")
            return rng.sample(range(low, high + 1), n)
        return [rng.randint(low, high) for _ in range(n)]


@dataclass(frozen=True)
class Float:
    """Float column: optional bounds; NaN is a violation unless allowed."""

    low: float | None = None
    high: float | None = None
    allow_nan: bool = False

    def describe(self) -> str:
        parts = ["float"]
        if self.low is not None or self.high is not None:
            parts.append(_fmt_bound(self.low, self.high))
        if self.allow_nan:
            parts.append("NaN allowed")
        return ", ".join(parts)

    def check_series(self, name: str, series) -> list[Violation]:
        import pandas as pd

        expected = self.describe()
        if not pd.api.types.is_numeric_dtype(series):
            return [Violation(name, "column is not numeric", expected, str(series.dtype))]
        out: list[Violation] = []
        nan = series[series.isna()]
        if len(nan) and not self.allow_nan:
            out.append(Violation(
                name, f"{len(nan)} NaN values (first at row {nan.index[0]})", expected,
            ))
        finite = series.dropna()
        if self.low is not None and (finite < self.low).any():
            bad = finite[finite < self.low]
            out.append(Violation(
                name, f"{len(bad)} values below low={self.low:g} "
                f"(first at row {bad.index[0]}: {bad.iloc[0]:g})", expected,
            ))
        if self.high is not None and (finite > self.high).any():
            bad = finite[finite > self.high]
            out.append(Violation(
                name, f"{len(bad)} values above high={self.high:g} "
                f"(first at row {bad.index[0]}: {bad.iloc[0]:g})", expected,
            ))
        return out

    def sample_values(self, n: int, rng: _random.Random) -> list[float]:
        low = self.low if self.low is not None else (min(self.high, 0.0) - 1.0 if self.high is not None else 0.0)
        high = self.high if self.high is not None else low + 1.0
        return [rng.uniform(low, high) for _ in range(n)]


class Space:
    """Abstract base. `check` reads the artifact off a candidate_dir;
    `check_value` validates an in-memory object; `describe` renders the
    contract deterministically; `sample` writes a format-valid artifact."""

    def check(self, candidate_dir: Path | str) -> list[Violation]:
        raise NotImplementedError

    def check_value(self, obj) -> list[Violation]:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def sample(self, candidate_dir: Path | str, seed: int | None = None) -> None:
        raise NotImplementedError


def _pandas_or_violation(where: str):
    """(pandas module, None) or (None, [Violation]) when this venv lacks it."""
    try:
        import pandas as pd
    except ImportError:
        return None, [Violation(
            where, "pandas is not available in this environment, cannot check",
            "pandas importable",
        )]
    return pd, None


def _rows_label(n_rows) -> str:
    if n_rows is None:
        return "any number of rows"
    if isinstance(n_rows, int):
        return f"exactly {n_rows} row" + ("s" if n_rows != 1 else "")
    low, high = n_rows
    if low is not None and high is not None:
        return f"{low}–{high} rows"
    if low is not None:
        return f"at least {low} rows"
    return f"at most {high} rows"


@dataclass(frozen=True)
class Table(Space):
    """A tabular artifact (CSV) with typed, optionally bounded columns.

    `file=None` declares an in-memory table (checked via `check_value`, e.g.
    a DataFrame a verifier holds) — `check(dir)` then reports that nothing
    on disk is declared rather than guessing a filename.
    `n_rows`: exact int, `(min, max)` with None open ends, or None."""

    file: str | None
    columns: dict = field(default_factory=dict)  # {name: Int | Float}
    n_rows: object = None
    extra_columns: str = "forbid"  # or "ignore"

    def describe(self) -> str:
        where = f"File `{self.file}` (CSV)" if self.file else "An in-memory table"
        lines = [f"{where}, {_rows_label(self.n_rows)}, columns:"]
        for name, col in self.columns.items():
            lines.append(f"- `{name}`: {col.describe()}")
        if self.extra_columns == "forbid":
            lines.append("No other columns.")
        return "\n".join(lines)

    def check(self, candidate_dir: Path | str) -> list[Violation]:
        if self.file is None:
            return [Violation(
                "(table)", "this space declares no file; check the in-memory "
                "value with check_value()", "file=... or check_value(df)",
            )]
        pd, missing = _pandas_or_violation(self.file)
        if missing:
            return missing
        path = Path(candidate_dir) / self.file
        if not path.exists():
            return [Violation(self.file, "file not found", f"a CSV file named {self.file}")]
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 — any parse failure is the finding
            return [Violation(self.file, f"cannot read as CSV: {exc}", "a parseable CSV file")]
        return self.check_value(frame)

    def check_value(self, frame) -> list[Violation]:
        pd, missing = _pandas_or_violation(self.file or "(table)")
        if missing:
            return missing
        label = self.file or "(table)"
        if not isinstance(frame, pd.DataFrame):
            return [Violation(label, "value is not a DataFrame", "a pandas DataFrame",
                              type(frame).__name__)]
        out: list[Violation] = []
        for name in self.columns:
            if name not in frame.columns:
                out.append(Violation(f"{label}[{name}]", "missing column",
                                     f"a column named {name!r}"))
        if self.extra_columns == "forbid":
            for name in frame.columns:
                if name not in self.columns:
                    out.append(Violation(f"{label}[{name}]", "unexpected extra column",
                                         f"only {list(self.columns)}"))
        if self.n_rows is not None:
            n = len(frame)
            if isinstance(self.n_rows, int):
                if n != self.n_rows:
                    out.append(Violation(label, f"has {n} rows", _rows_label(self.n_rows)))
            else:
                low, high = self.n_rows
                if (low is not None and n < low) or (high is not None and n > high):
                    out.append(Violation(label, f"has {n} rows", _rows_label(self.n_rows)))
        for name, col in self.columns.items():
            if name in frame.columns:
                out.extend(
                    replace(v, path=f"{label}[{name}]")
                    for v in col.check_series(name, frame[name])
                )
        return out

    def sample(self, candidate_dir: Path | str, seed: int | None = None) -> None:
        if self.file is None:
            raise InterfaceError("cannot sample a Table with file=None onto disk")
        rng = _random.Random(seed)
        if isinstance(self.n_rows, int):
            n = self.n_rows
        elif self.n_rows is not None and self.n_rows[0] is not None:
            n = self.n_rows[0]
        else:
            n = 1
        data = {name: col.sample_values(n, rng) for name, col in self.columns.items()}
        path = Path(candidate_dir) / self.file
        with path.open("w", newline="") as handle:
            writer = _csv.writer(handle)
            writer.writerow(list(self.columns))
            for row in range(n):
                writer.writerow([data[name][row] for name in self.columns])


@dataclass(frozen=True)
class Array(Space):
    """A `.npy` array artifact: dtype kind, shape (None = wildcard dim), bounds."""

    file: str
    dtype: str = "float"  # numpy dtype kind by name: "float" | "int" | exact dtype str
    shape: tuple | None = None
    low: float | None = None
    high: float | None = None

    def describe(self) -> str:
        shape = "any shape" if self.shape is None else (
            "shape (" + ", ".join("?" if d is None else str(d) for d in self.shape) + ")"
        )
        bound = _fmt_bound(self.low, self.high)
        tail = "" if bound == "unbounded" else f", values {bound}"
        return f"File `{self.file}` (.npy), dtype {self.dtype}, {shape}{tail}"

    def _dtype_ok(self, np, dtype) -> bool:
        if self.dtype in ("float", "int"):
            return dtype.kind == self.dtype[0]
        return dtype == np.dtype(self.dtype)

    def check(self, candidate_dir: Path | str) -> list[Violation]:
        try:
            import numpy as np
        except ImportError:
            return [Violation(self.file, "numpy is not available in this environment, "
                              "cannot check", "numpy importable")]
        path = Path(candidate_dir) / self.file
        if not path.exists():
            return [Violation(self.file, "file not found", f"a .npy file named {self.file}")]
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            return [Violation(self.file, f"cannot load as .npy: {exc}", "a valid .npy file")]
        return self.check_value(array)

    def check_value(self, array) -> list[Violation]:
        import numpy as np

        expected = self.describe()
        out: list[Violation] = []
        if not self._dtype_ok(np, array.dtype):
            out.append(Violation(self.file, "wrong dtype", expected, str(array.dtype)))
        if self.shape is not None:
            ok = len(array.shape) == len(self.shape) and all(
                want is None or got == want for got, want in zip(array.shape, self.shape)
            )
            if not ok:
                out.append(Violation(self.file, "wrong shape", expected, str(array.shape)))
        if array.dtype.kind in "if":
            if self.low is not None and np.nanmin(array) < self.low:
                out.append(Violation(self.file, f"values below low={self.low:g}", expected))
            if self.high is not None and np.nanmax(array) > self.high:
                out.append(Violation(self.file, f"values above high={self.high:g}", expected))
        return out

    def sample(self, candidate_dir: Path | str, seed: int | None = None) -> None:
        import numpy as np

        rng = np.random.default_rng(seed)
        shape = tuple(1 if d is None else d for d in (self.shape or (1,)))
        low = self.low if self.low is not None else 0.0
        high = self.high if self.high is not None else low + 1.0
        if self.dtype == "int" or (self.dtype not in ("float",) and np.dtype(self.dtype).kind == "i"):
            array = rng.integers(int(low), int(high) + 1, size=shape,
                                 dtype=self.dtype if self.dtype != "int" else "int64")
        else:
            array = rng.uniform(low, high, size=shape).astype(
                self.dtype if self.dtype != "float" else "float64"
            )
        np.save(Path(candidate_dir) / self.file, array)


@dataclass(frozen=True)
class Callable(Space):
    """The solution must expose a callable: `<module>.py` defines `<name>`
    taking exactly `params` (positional order checked when given)."""

    name: str
    module: str = "solution"
    params: tuple | None = None

    def _label(self) -> str:
        return f"{self.module}.{self.name}"

    def describe(self) -> str:
        args = ", ".join(self.params) if self.params is not None else "…"
        return f"`{self.module}.py` must define `{self.name}({args})`"

    def check(self, candidate_dir: Path | str) -> list[Violation]:
        path = Path(candidate_dir) / f"{self.module}.py"
        if not path.exists():
            return [Violation(self._label(), f"{self.module}.py not found", self.describe())]
        digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
        spec = importlib.util.spec_from_file_location(f"_hillclimb_iface_{digest}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 — the import failure is the finding
            return [Violation(self._label(), f"importing {self.module}.py failed: {exc}",
                              "an importable module")]
        return self.check_value(module)

    def check_value(self, module) -> list[Violation]:
        expected = self.describe()
        target = getattr(module, self.name, None)
        if target is None:
            return [Violation(self._label(), f"no attribute {self.name!r}", expected)]
        if not callable(target):
            return [Violation(self._label(), f"{self.name!r} is not callable", expected,
                              type(target).__name__)]
        if self.params is not None:
            try:
                got = tuple(inspect.signature(target).parameters)
            except (TypeError, ValueError):
                return []  # signature not introspectable (C callable): existence is enough
            if got != tuple(self.params):
                return [Violation(self._label(), "wrong parameters", expected,
                                  f"({', '.join(got)})")]
        return []

    def sample(self, candidate_dir: Path | str, seed: int | None = None) -> None:
        raise NotImplementedError("Callable spaces cannot be sampled (v2)")


@dataclass(frozen=True)
class Dict(Space):
    """Named composite: every child space must hold."""

    spaces: dict = field(default_factory=dict)  # {name: Space}

    def describe(self) -> str:
        return "\n\n".join(f"**{name}**: {space.describe()}"
                           for name, space in self.spaces.items())

    def check(self, candidate_dir: Path | str) -> list[Violation]:
        return [
            replace(v, path=f"{name}.{v.path}")
            for name, space in self.spaces.items()
            for v in space.check(candidate_dir)
        ]

    def check_value(self, obj) -> list[Violation]:
        out: list[Violation] = []
        for name, space in self.spaces.items():
            if name not in obj:
                out.append(Violation(name, "missing entry", f"an entry named {name!r}"))
                continue
            out.extend(replace(v, path=f"{name}.{v.path}")
                       for v in space.check_value(obj[name]))
        return out

    def sample(self, candidate_dir: Path | str, seed: int | None = None) -> None:
        for offset, space in enumerate(self.spaces.values()):
            space.sample(candidate_dir, seed=None if seed is None else seed + offset)


# --- interface.py loading and rendering ---


def load_interface(path: Path | str):
    """Import a problem's interface.py as a module. Raises InterfaceError
    with the file named on any failure — an unloadable interface is a
    problem-authoring bug, caught at load time, not inside a search."""
    path = Path(path)
    if not path.exists():
        raise InterfaceError(f"interface file not found: {path}")
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    spec = importlib.util.spec_from_file_location(f"_hillclimb_interface_{digest}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise InterfaceError(f"cannot import {path}: {exc}") from exc
    if not isinstance(getattr(module, "output", None), Space) and not _inputs_of(module):
        raise InterfaceError(
            f"{path} defines neither `output` nor `inputs` built from hillclimb.spaces"
        )
    return module


def _inputs_of(module) -> dict:
    inputs = getattr(module, "inputs", None)
    if isinstance(inputs, Space):
        return {"inputs": inputs}
    if isinstance(inputs, dict):
        return {name: space for name, space in inputs.items() if isinstance(space, Space)}
    return {}


def describe_interface(module) -> str:
    """Deterministic prompt-facing rendering of an interface module."""
    sections = []
    output = getattr(module, "output", None)
    if isinstance(output, Space):
        sections.append(f"### Output\n\n{output.describe()}")
    inputs = _inputs_of(module)
    if inputs:
        body = "\n\n".join(f"**{name}**: {space.describe()}" if name != "inputs"
                           else space.describe()
                           for name, space in inputs.items())
        sections.append(f"### Inputs (provided, for reference)\n\n{body}")
    return "\n\n".join(sections)


def main(space: Space, argv: list[str] | None = None) -> int:
    """Author-footer entry point: check a directory (argv[0], default cwd)
    against `space`, print violations, return a shell exit code."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    candidate_dir = Path(args[0]) if args else Path.cwd()
    violations = space.check(candidate_dir)
    if not violations:
        print("interface: OK")
        return 0
    for violation in violations:
        print(f"interface: {violation}")
    return 1


# ---------------------------------------------------------------------------
# Tunable parameters: the `params.json` contract
#
# A solution may declare numeric knobs next to itself in `params.json`
# (flat `name -> spec`, see PARAM_TYPES); the engine then runs extra trials of
# the SAME code with other values and keeps the best. The candidate-root copy
# carries each param's `default`; the copy the engine writes into a trial dir
# adds a `value` per param. `params()` is the agent-facing reader — it works
# with no file at all (the solution's own defaults), so declaring is optional.
#
# Stdlib-only on purpose: this file is byte-copied into the runtime shim.

PARAMS_FILE = "params.json"
PARAMS_ENV = "HILLCLIMB_PARAMS"
PARAM_TYPES = ("float", "int", "categorical")
_SPEC_KEYS = {"type", "low", "high", "log", "step", "choices", "default", "value"}
_SCALARS = (str, int, float, bool)


class ParamsError(ValueError):
    """A params.json that cannot be read or violates the declaration rules."""


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_params_space(raw: object) -> list[Violation]:
    """Every rule the declaration must honor, as located violations (never
    raises): a flat object of identifier-named specs; `type` in PARAM_TYPES;
    float/int carry numeric `low < high` (ints for int), optional `log`
    (needs low > 0) and positive `step`; categorical carries unique
    same-typed scalar `choices`; every spec has a `default` in range."""
    violations: list[Violation] = []

    def bad(path: str, message: str, expected: str, actual: object = None) -> None:
        violations.append(Violation(path, message, expected, None if actual is None else repr(actual)))

    if not isinstance(raw, dict):
        bad(PARAMS_FILE, "not a JSON object", "{name: spec, ...}", type(raw).__name__)
        return violations
    if not raw:
        bad(PARAMS_FILE, "declares no parameters", "at least one entry")
    for name, spec in raw.items():
        path = f"{PARAMS_FILE}[{name}]"
        if not isinstance(name, str) or not name.isidentifier():
            bad(path, "name is not a Python identifier", "e.g. learning_rate", name)
        if not isinstance(spec, dict):
            bad(path, "spec is not an object", '{"type": ..., "default": ...}', spec)
            continue
        unknown = sorted(set(spec) - _SPEC_KEYS)
        if unknown:
            bad(path, f"unknown keys {unknown}", f"only {sorted(_SPEC_KEYS)}")
        kind = spec.get("type")
        if kind not in PARAM_TYPES:
            bad(f"{path}.type", "unknown type", "|".join(PARAM_TYPES), kind)
            continue
        if "default" not in spec:
            bad(f"{path}.default", "missing", "the value the code uses today")
        if kind == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                bad(f"{path}.choices", "missing or empty", "a non-empty list", choices)
                continue
            if any(not isinstance(c, _SCALARS) for c in choices):
                bad(f"{path}.choices", "non-scalar choice", "strings, numbers or booleans")
            if len({type(c) for c in choices}) > 1:
                bad(f"{path}.choices", "mixed types", "choices of one type")
            if len(set(map(repr, choices))) != len(choices):
                bad(f"{path}.choices", "duplicate choice", "unique choices")
            for key in ("low", "high", "log", "step"):
                if key in spec:
                    bad(f"{path}.{key}", "not allowed on a categorical", "choices only")
            if "default" in spec and spec["default"] not in choices:
                bad(f"{path}.default", "not one of the choices", str(choices), spec["default"])
            continue
        low, high = spec.get("low"), spec.get("high")
        numeric = _is_number
        if kind == "int":
            numeric = lambda v: isinstance(v, int) and not isinstance(v, bool)  # noqa: E731
        if not numeric(low) or not numeric(high):
            bad(f"{path}.low/high", "missing or not numeric", f"{kind} bounds", (low, high))
            continue
        if not low < high:
            bad(f"{path}.low/high", "low is not below high", "low < high", (low, high))
        if spec.get("log", False) not in (True, False):
            bad(f"{path}.log", "not a boolean", "true|false", spec.get("log"))
        elif spec.get("log") and low <= 0:
            bad(f"{path}.log", "log scale needs low > 0", "low > 0", low)
        if "step" in spec and (not numeric(spec["step"]) or spec["step"] <= 0):
            bad(f"{path}.step", "not a positive number", f"positive {kind}", spec["step"])
        if "choices" in spec:
            bad(f"{path}.choices", "not allowed on a numeric parameter", "low/high")
        default = spec.get("default")
        if "default" in spec:
            if not numeric(default):
                bad(f"{path}.default", "not numeric", kind, default)
            elif not low <= default <= high:
                bad(f"{path}.default", "outside [low, high]", f"[{low}, {high}]", default)
    return violations


def load_params_file(path) -> dict:
    """Parse and validate a params.json; raises ParamsError naming every
    violation (the runtime half: a broken declaration must fail visibly)."""
    try:
        raw = _json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ParamsError(f"{path}: {exc}") from exc
    violations = check_params_space(raw)
    if violations:
        raise ParamsError("; ".join(str(v) for v in violations))
    return raw


def _coerce_param(spec: dict, value):
    kind = spec["type"]
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    for choice in spec["choices"]:  # categorical: the declared object, not a lookalike
        if choice == value and type(choice) is type(value):
            return choice
    for choice in spec["choices"]:
        if choice == value:
            return choice
    raise ParamsError(f"{value!r} is not one of {spec['choices']}")


def params_values(raw: dict) -> dict:
    """`{name: value-or-default}` for a validated declaration."""
    return {name: _coerce_param(spec, spec.get("value", spec["default"])) for name, spec in raw.items()}


def with_values(raw: dict, values: dict) -> dict:
    """The trial-dir document: the declaration with a `value` per param
    (missing names fall back to the default)."""
    doc = {}
    for name, spec in raw.items():
        entry = {k: v for k, v in spec.items() if k != "value"}
        entry["value"] = values.get(name, spec.get("default"))
        doc[name] = entry
    return doc


def fold_defaults(raw: dict, values: dict) -> dict:
    """The inheritance document for a child candidate: `values` become the
    new defaults, `value` is dropped."""
    doc = {}
    for name, spec in raw.items():
        entry = {k: v for k, v in spec.items() if k != "value"}
        if name in values:
            entry["default"] = values[name]
        doc[name] = entry
    return doc


def params(defaults: dict | None = None, path=None) -> dict:
    """The agent-facing reader: `{name: value}` for this run. Resolution:
    `path` → `$HILLCLIMB_PARAMS` (the engine points it at the trial's copy)
    → `./params.json` → no file, in which case `defaults` is returned as is.
    With a file, every declared param gets its `value` (else `default`),
    type-coerced; `defaults` fills any name the file does not declare. A
    malformed file raises ParamsError so the failure is visible in the run's
    stderr rather than silently scored on wrong values."""
    location = path or _os.environ.get(PARAMS_ENV) or PARAMS_FILE
    location = Path(location)
    if not location.exists():
        return dict(defaults or {})
    raw = load_params_file(location)
    values = dict(defaults or {})
    values.update(params_values(raw))
    return values


def describe_params(raw: dict, values: dict | None = None) -> str:
    """Deterministic one-line-per-param rendering for prompts and `show`."""
    lines = []
    for name in sorted(raw):
        spec = raw[name]
        kind = spec.get("type")
        if kind == "categorical":
            domain = "one of " + ", ".join(repr(c) for c in spec.get("choices", []))
        else:
            domain = f"{kind} in [{spec.get('low')}, {spec.get('high')}]"
            if spec.get("log"):
                domain += " (log)"
            if spec.get("step") is not None:
                domain += f" step {spec['step']}"
        line = f"{name}: {domain}, default {spec.get('default')!r}"
        if values is not None and name in values:
            line += f" = {values[name]!r}"
        lines.append(line)
    return "\n".join(lines)
