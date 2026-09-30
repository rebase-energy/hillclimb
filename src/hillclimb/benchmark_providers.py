"""Lazy registry for benchmark-backed problem providers.

Providers turn a URI target (``scheme://name``) into the same ``ProblemSpec``
local verifier problems use.  The registry deliberately owns only problem
resolution and chart references: search engines, evaluation, journaling, and
post-run behavior stay in hillclimb's generic host.
"""

from __future__ import annotations

from importlib import import_module
import re
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from hillclimb.config import Config
    from hillclimb.problem import ProblemSpec, ResolvedTarget


class BenchmarkProvider(Protocol):
    """Minimum interface implemented by a benchmark integration."""

    def load_problem(self, name: str, config: Config) -> ProblemSpec: ...

    def resolve_target(self, name: str, config: Config) -> ResolvedTarget: ...


ProviderLoader = Callable[[], BenchmarkProvider]
SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*$")


def _einsteinarena_provider() -> BenchmarkProvider:
    return import_module("hillclimb.providers.einsteinarena.provider").provider


_PROVIDERS: dict[str, ProviderLoader] = {
    "einsteinarena": _einsteinarena_provider,
}


def register_benchmark_provider(
    scheme: str, loader: ProviderLoader, *, replace: bool = False
) -> None:
    """Register a lazy provider loader.

    Third-party embedders may call this before resolving a target.  Built-in
    providers use the same table, so a fake provider can exercise the entire
    target-resolution seam without importing an optional benchmark package.
    """
    normalized = scheme.strip().lower()
    if not SCHEME_RE.fullmatch(normalized):
        raise ValueError(f"invalid benchmark provider scheme: {scheme!r}")
    if normalized in _PROVIDERS and not replace:
        raise ValueError(f"benchmark provider already registered: {normalized}")
    _PROVIDERS[normalized] = loader


def benchmark_provider_schemes() -> tuple[str, ...]:
    return tuple(sorted(_PROVIDERS))


def has_benchmark_provider(scheme: str) -> bool:
    return scheme.lower() in _PROVIDERS


def get_benchmark_provider(scheme: str) -> BenchmarkProvider:
    try:
        loader = _PROVIDERS[scheme.lower()]
    except KeyError as exc:
        known = ", ".join(benchmark_provider_schemes()) or "(none)"
        raise ValueError(
            f"unknown benchmark provider {scheme!r}; registered providers: {known}"
        ) from exc
    return loader()
