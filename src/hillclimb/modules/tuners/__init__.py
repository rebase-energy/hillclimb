"""Tuners: WHICH parameter values next. Named like every other module
(`modules/refs.py`): a registry name (`random`, `optuna`), a `.py` file
(`anneal.py` or `anneal.py:Anneal`; the file sets `TUNER = <class or
factory>` or defines exactly one class with `ask`), or `package.module:Class`.
"""

from __future__ import annotations

from pathlib import Path

from hillclimb.modules import refs
from hillclimb.modules.tuners.base import Tuner

KIND = "tuner"

refs.register(KIND, "random", "hillclimb.modules.tuners.random_search:RandomSearch")
refs.register(KIND, "optuna", "hillclimb.modules.tuners.optuna:Optuna")  # optional extra


def get_tuner(
    name: str,
    params: dict | None = None,
    *,
    base_dir: Path | None = None,
    scope: refs.FileScope | None = None,
) -> Tuner:
    """A tuner instance. It gets `params` when its constructor takes them,
    and a `name` and `params` attribute when it set none."""
    resolved = refs.resolve_ref(name, KIND, base_dir=base_dir, scope=scope)
    params = dict(params or {})
    tuner = refs.construct(resolved.target, {"params": params}, name)
    if not callable(getattr(tuner, "ask", None)):
        raise refs.ClimberLoadError(f"tuner {name!r} has no ask() — not a Tuner")
    if not getattr(tuner, "name", None):
        tuner.name = resolved.label
    if getattr(tuner, "params", None) is None:
        tuner.params = params
    return tuner
