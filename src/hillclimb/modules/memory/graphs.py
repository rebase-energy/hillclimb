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
from pathlib import Path

from hillclimb.modules import refs
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

KIND = "graph"
GRAPH_ATTR = refs.KINDS[KIND].attr  # KNOWLEDGE_GRAPH

# the registry itself lives with every other kind's, in modules/refs.py
_GRAPHS: dict[str, type[GraphModule]] = refs.KINDS[KIND].registry


def register_graph(cls: type[GraphModule], name: str | None = None) -> type[GraphModule]:
    """Add a graph module class to the registry (usable as a decorator). A
    registry name never looks like a file or a dotted path, so the three
    forms stay unambiguous."""
    key = name or cls.name
    if not key:
        raise ValueError(f"{cls.__name__} has no name; set `name = ...` or pass one")
    refs.register(KIND, key, cls)
    return cls


def registered_graphs() -> dict[str, type[GraphModule]]:
    return refs.registered(KIND)


register_graph(KnowledgeGraphBuilder)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def graph_key(ref: str, path: Path | None = None) -> str:
    """The cache key a module resolved from `ref` carries: the file's ref
    plus its content hash, else the ref itself (registry name or dotted)."""
    return f"{ref}#{_file_digest(path)}" if path is not None else ref


def load_graph_file(path: Path) -> type[GraphModule]:
    """The graph module class a `.py` file exposes. Import errors and a
    missing or ambiguous class surface as ValueError naming the file."""
    return refs.resolve_ref(str(path), KIND).target


def get_graph(name: str, *, base_dir: Path | None = None, scope: refs.FileScope | None = None) -> GraphModule:
    """A graph module instance by registry name, `.py` path (`base_dir`
    anchors a relative one — pass `climber.climber_base_dir(config)`), or
    `module:Class`. A class without a `name` is named after the file stem or
    the class, so logs and stats stay labelled."""
    resolved = refs.resolve_ref(name, KIND, base_dir=base_dir, scope=scope)
    module = resolved.target()
    module.key = graph_key(name, resolved.path)
    if not module.name:
        module.name = resolved.label  # type: ignore[misc] — instance attribute shadows the ClassVar
    return module
