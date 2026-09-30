"""Graph modules: name -> GraphModule, mirroring similarity.get_score.

`graph:` in a climber manifest (or `climber.graph` in config.yaml) names the
module that builds `knowledge/graph.json`, ranks the claims a search is
shown and answers `hillclimb knowledge query`:

- a registry name: `knowledge-graph` (the built-in, in graph.py);
- a Python file, any name ending in `.py` (in a manifest: relative to the
  climber dir; in config: relative to the folder holding the hillclimb dir,
  like `climber.ref`): the file sets `KNOWLEDGE_GRAPH = <class>` or defines
  exactly one `GraphModule` subclass;
- `package.module:ClassName` for modules shipped in an installed package;
- or `register_graph(cls)` from code that embeds hillclimb.

Every module gets a `key` — the registry name, `file.py#<sha12>` for a file
(so editing the file invalidates the index it built), or the dotted spec —
which `graph.json` records as `builder`; an index built under another key is
stale however fresh its mtime.

The registry cannot live in `memory/__init__.py`, which imports nothing so
that `hillclimb.modules.memory.x` never loads the whole memory package.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from hillclimb._moved import modernize
from hillclimb.modules.memory.base import DEFAULT_GRAPH, GraphModule
from hillclimb.modules.memory.graph import KnowledgeGraphBuilder

__all__ = [
    "DEFAULT_GRAPH",
    "GRAPH_ATTR",
    "GraphModule",
    "get_graph",
    "graph_key",
    "load_graph_file",
    "register_graph",
    "registered_graphs",
]

GRAPH_FILE_SUFFIX = ".py"
GRAPH_ATTR = "KNOWLEDGE_GRAPH"

_GRAPHS: dict[str, type[GraphModule]] = {}


def register_graph(cls: type[GraphModule], name: str | None = None) -> type[GraphModule]:
    """Add a graph module class to the registry (usable as a decorator). A
    registry name never looks like a file or a dotted path, so the three
    forms stay unambiguous."""
    key = name or cls.name
    if not key:
        raise ValueError(f"{cls.__name__} has no name; set `name = ...` or pass one")
    if key.endswith(GRAPH_FILE_SUFFIX) or ":" in key:
        raise ValueError(f"graph module name {key!r} would read as a file or module:Class")
    _GRAPHS[key] = cls
    return cls


def registered_graphs() -> dict[str, type[GraphModule]]:
    return dict(_GRAPHS)


register_graph(KnowledgeGraphBuilder)


def _is_graph_class(obj: Any, module_name: str) -> bool:
    return (
        inspect.isclass(obj)
        and issubclass(obj, GraphModule)
        and obj is not GraphModule
        and obj.__module__ == module_name
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def graph_key(ref: str, path: Path | None = None) -> str:
    """The cache key a module resolved from `ref` carries: the file's ref
    plus its content hash, else the ref itself (registry name or dotted)."""
    return f"{ref}#{_file_digest(path)}" if path is not None else ref


def load_graph_file(path: Path) -> type[GraphModule]:
    """The graph module class a `.py` file exposes. Import errors and a
    missing or ambiguous class surface as ValueError naming the file."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"graph module file not found: {path}")
    module_name = f"hillclimb_graph_{path.stem}_{_file_digest(path)}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import graph module file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 — user code
            del sys.modules[module_name]
            raise ValueError(f"graph module file {path} failed to import: {type(exc).__name__}: {exc}") from exc
    target = getattr(module, GRAPH_ATTR, None)
    if target is None:
        classes = [obj for obj in vars(module).values() if _is_graph_class(obj, module_name)]
        if len(classes) != 1:
            raise ValueError(
                f"graph module file {path} must define exactly one GraphModule subclass "
                f"(found {[c.__name__ for c in classes]}) or set {GRAPH_ATTR} = <class>"
            )
        target = classes[0]
    if not (inspect.isclass(target) and issubclass(target, GraphModule)):
        raise ValueError(f"{GRAPH_ATTR} in {path} is not a GraphModule subclass: {target!r}")
    return target


def _load_import_path(spec: str) -> type[GraphModule]:
    module_name, _, attr = modernize(spec).partition(":")  # a ref written before a move
    try:
        target = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot import graph module {spec!r}: {exc}") from exc
    if not (inspect.isclass(target) and issubclass(target, GraphModule)):
        raise ValueError(f"{spec!r} is not a GraphModule subclass")
    return target


def get_graph(name: str, *, base_dir: Path | None = None) -> GraphModule:
    """A graph module instance by registry name, `.py` path (`base_dir`
    anchors a relative one — pass `climber.climber_base_dir(config)`), or
    `module:Class`. A class without a `name` is named after the file stem or
    the class, so logs and stats stay labelled."""
    path: Path | None = None
    if name.endswith(GRAPH_FILE_SUFFIX):
        path = Path(name).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        cls = load_graph_file(path)
        fallback = path.stem
    elif ":" in name:
        cls = _load_import_path(name)
        fallback = cls.__name__
    elif name in _GRAPHS:
        cls = _GRAPHS[name]
        fallback = name
    else:
        raise ValueError(
            f"unknown graph module {name!r} (available: {', '.join(sorted(_GRAPHS))}, "
            "a path to a .py file, or module:Class)"
        )
    module = cls()
    module.key = graph_key(name, path)
    if not module.name:
        module.name = fallback  # type: ignore[misc] — instance attribute shadows the ClassVar
    return module
