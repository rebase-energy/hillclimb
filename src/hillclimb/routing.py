"""Per-operator backend/model routing: which agent runs which operator.

Field-level precedence, first non-None wins:
action.route > config.routing[operator] > config.routing["default"] >
the global `backend`/`model`/`backend_auth` scalars. An absent `routing:`
block resolves to exactly the global scalars — today's behavior.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from hillclimb.backends import OperatorBackend, get_backend
from hillclimb.config import Config
from hillclimb.policy import Route


@dataclass(frozen=True)
class ResolvedRoute:
    backend: str
    model: str
    backend_auth: str


class Router:
    def __init__(self, config: Config):
        self.config = config

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

        return ResolvedRoute(
            backend=pick("backend", config.backend),
            model=pick("model", config.model),
            backend_auth=pick("backend_auth", config.backend_auth),
        )


class BackendPool:
    """Lazy name->instance cache, one instance per (backend, auth) per search
    — backends are stateful (call counters, abort wiring), so a route must
    resolve to the same instance every time."""

    def __init__(self, abort: threading.Event | None = None):
        self._abort = abort
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
                backend = get_backend(name, auth=auth)
                if self._abort is not None and hasattr(backend, "abort"):
                    backend.abort = self._abort
                self._instances[key] = backend
            return self._instances[key]
