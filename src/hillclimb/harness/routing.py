"""Per-operator backend/model routing: which agent runs which operator.

Field-level precedence, first non-None wins:
action.route > config.routing[operator] > config.routing["default"] >
the global `backend`/`model`/`backend_auth` scalars. An absent `routing:`
block resolves to exactly the global scalars — today's behavior.

Model choice has one extra form: a layer may carry a `models:` POOL instead
of a scalar, and the first layer that says anything about the model (pool or
scalar) wins. A 2+ entry pool routes through a UCB1 bandit (bandit.py) that
learns which model earns improvements per operator; a 1-entry pool degrades
to a scalar. The searcher feeds results back through `observe()` — both live
(at commit) and from journal replay (on construct), so bandit state survives
`resume` without extra persistence.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from hillclimb.backends import OperatorBackend, get_backend
from hillclimb.harness.bandit import OperatorBandits
from hillclimb.config import Config
from hillclimb.modules.policies.base import Route


@dataclass(frozen=True)
class ResolvedRoute:
    backend: str
    model: str
    backend_auth: str
    sampling: dict[str, int | float] | None = None


class Router:
    def __init__(self, config: Config, bandits: OperatorBandits | None = None):
        self.config = config
        if bandits is None and any(
            route.models and len(route.models) > 1 for route in config.routing.values()
        ):
            bandits = OperatorBandits()
        self.bandits = bandits

    def resolve(self, operator: str, override: Route | None = None) -> ResolvedRoute:
        config = self.config
        layers = [
            override,
            config.routing.get(operator),
            config.routing.get("default"),
        ]

        def pick(field: str, fallback: str) -> str:
            for layer in layers:
                value = getattr(layer, field, None) if layer is not None else None
                if value:
                    return value
            return fallback

        def pick_optional(field: str):
            for layer in layers:
                value = getattr(layer, field, None) if layer is not None else None
                if value is not None:
                    return dict(value) if isinstance(value, dict) else value
            return None

        route = ResolvedRoute(
            backend=pick("backend", config.backend),
            model=self._pick_model(operator, layers),
            backend_auth=pick("backend_auth", config.backend_auth),
            sampling=pick_optional("sampling"),
        )
        if route.sampling and route.backend != "pi":
            raise ValueError(f"routing.{operator}: sampling needs backend: pi, not {route.backend!r}")
        return route

    def _pick_model(self, operator: str, layers: list) -> str:
        """First layer that says anything about the model wins; within a
        layer a pool beats the scalar. Only 2+ entry pools consult the
        bandit."""
        for layer in layers:
            if layer is None:
                continue
            pool = getattr(layer, "models", None)
            if pool:
                pool = tuple(pool)
                if len(pool) == 1 or self.bandits is None:
                    return pool[0]
                return self.bandits.select(operator, pool)
            value = getattr(layer, "model", None)
            if value:
                return value
        return self.config.model

    def _pool_for(self, operator: str) -> tuple[str, ...] | None:
        """The bandit pool governing an operator's model choice, or None when
        a scalar wins first (mirrors _pick_model over the config layers —
        per-action overrides don't reach the bandit)."""
        for layer in (self.config.routing.get(operator), self.config.routing.get("default")):
            if layer is None:
                continue
            if layer.models:
                return tuple(layer.models) if len(layer.models) > 1 else None
            if layer.model:
                return None
        return None

    def observe(self, operator: str, model: str | None, reward: float) -> None:
        """Credit a journaled outcome to the model arm that produced it. A
        no-op without a bandit, outside a pool-governed operator, or for a
        model that isn't in the pool (per-action overrides, config changes)."""
        if self.bandits is None or not model:
            return
        pool = self._pool_for(operator)
        if pool is not None and model in pool:
            self.bandits.update(operator, pool, model, reward)


class BackendPool:
    """Lazy name->instance cache, one instance per (backend, auth) per search
    — backends are stateful (call counters, abort wiring), so a route must
    resolve to the same instance every time."""

    def __init__(
        self,
        abort: threading.Event | None = None,
        pi_models_file: Path | None = None,
    ):
        self._abort = abort
        self._pi_models_file = pi_models_file
        self._instances: dict[tuple[str, str], OperatorBackend] = {}
        self._lock = threading.Lock()

    def seed(self, name: str, auth: str, backend: OperatorBackend) -> None:
        """Register an existing instance (the default backend the harness
        already constructed) so the default route reuses it."""
        self._instances[(name, auth)] = backend

    def get(self, name: str, auth: str) -> OperatorBackend:
        key = (name, auth)
        with self._lock:
            if key not in self._instances:
                backend = get_backend(
                    name, auth=auth, pi_models_file=self._pi_models_file
                )
                if self._abort is not None and hasattr(backend, "abort"):
                    backend.abort = self._abort
                self._instances[key] = backend
            return self._instances[key]
