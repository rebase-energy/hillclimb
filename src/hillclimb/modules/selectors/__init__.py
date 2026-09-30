"""Selectors: WHICH candidate to expand next. Named like every other module
(`modules/refs.py`): a registry name (`best`, `map-elites`), a `.py` file
(the file sets `SELECTOR = <class>` or defines exactly one `Selector`
subclass), or `package.module:Class`. `base.py` is the contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from hillclimb.modules import refs
from hillclimb.modules.selectors.base import Selection, Selector, improvable

KIND = "select"

refs.register(KIND, "best", "hillclimb.modules.selectors.best:Best")
refs.register(KIND, "map-elites", "hillclimb.modules.selectors.map_elites:MapElites")  # optional extra


def get_selector(
    name: str,
    params: Mapping | None = None,
    *,
    base_dir: Path | None = None,
    scope: refs.FileScope | None = None,
) -> Selector:
    """A selector instance, built with `params`. A setting it does not take
    (or a missing extra) is a `ClimberLoadError` naming the selector."""
    resolved = refs.resolve_ref(name, KIND, base_dir=base_dir, scope=scope)
    try:
        selector = refs.construct(resolved.target, {"params": dict(params or {})}, name)
    except (ValueError, ImportError) as exc:
        raise refs.ClimberLoadError(f"select: {name}: {exc}") from exc
    if not selector.name:
        selector.name = resolved.label  # type: ignore[misc]
    return selector


__all__ = ["Selection", "Selector", "get_selector", "improvable"]
