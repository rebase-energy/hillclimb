"""Operator prompt templates: `{{token}}` files, looked up in an optional
override dir first, then this package.

The override dir (`config.paths.prompts_dir`, default `<hillclimb dir>/
prompts/`) is how the exploration process's *prompts* become data an
agent or an experiment arm can edit without touching the installed
package. The engine activates it once per process (`set_override_dir` in
`api.execute_search`); `templates_digest` hashes the effective template
set so `SearchMeta.templates_sha256` pins exactly which prompts a search
ran with. Override files carry the same `{{token}}` vocabulary as the
package template they shadow — `lint_overrides` rejects tokens the engine
would never fill.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_SUFFIX = ".md"

_override_dir: Path | None = None

COMPLEXITY_CUES = {
    "minimal": (
        "Produce a MINIMAL, reliable first solution: the simplest sensible model "
        "for this data type (e.g., gradient-boosted trees for tabular data, "
        "TF-IDF + linear model for simple text). No ensembles, no neural networks, "
        "no heavy tuning. Prioritize a correct end-to-end pipeline over cleverness."
    ),
    "moderate": (
        "Produce a MODERATE solution: a solid model with sensible feature "
        "engineering and light hyperparameter tuning. Still no large ensembles."
    ),
    "advanced": (
        "You may produce an ADVANCED solution: stronger models, richer feature "
        "engineering, careful tuning — but it must still run reliably within the "
        "execution time limit."
    ),
}

_TOKEN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


def set_override_dir(path: Path | None) -> Path | None:
    """Activate `path` as the template override dir (None = package only)
    for this process; returns the previous setting so callers can restore."""
    global _override_dir
    previous = _override_dir
    _override_dir = Path(path) if path is not None else None
    return previous


def override_dir() -> Path | None:
    return _override_dir


def template_path(template_name: str, override: Path | None = None) -> Path:
    """The file `render` would read: the override when one exists, else the
    package template. `override` defaults to the active override dir."""
    base = _override_dir if override is None else override
    if base is not None:
        candidate = base / f"{template_name}{TEMPLATE_SUFFIX}"
        if candidate.is_file():
            return candidate
    return TEMPLATE_DIR / f"{template_name}{TEMPLATE_SUFFIX}"


def render(template_name: str, _override: Path | None = None, **context) -> str:
    """Replace {{key}} tokens. Deliberately not str.format: competition
    descriptions routinely contain literal braces. `_override` is a prompts
    dir bound to ONE search (its climber's); without one the process-wide
    override dir applies."""
    text = template_path(template_name, _override).read_text()
    for key, value in context.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def template_names(override: Path | None = None) -> list[str]:
    """Every template name the engine can render: the package set plus any
    extra names the override dir adds (a problem may name a custom
    `contract_template`)."""
    names = {p.stem for p in TEMPLATE_DIR.glob(f"*{TEMPLATE_SUFFIX}")}
    base = _override_dir if override is None else override
    if base is not None and base.is_dir():
        names.update(p.stem for p in base.glob(f"*{TEMPLATE_SUFFIX}"))
    return sorted(names)


def tokens_in(text: str) -> set[str]:
    return set(_TOKEN.findall(text))


@dataclass(frozen=True)
class TemplatesDigest:
    """What a search's prompts were: a sha256 over every effective template
    (name + content, sorted) and the names the override dir shadows."""

    sha256: str
    overridden: list[str] = field(default_factory=list)


def templates_digest(override: Path | None = None) -> TemplatesDigest:
    """Hash the effective template set. `override` defaults to the active
    override dir; pass a dir explicitly to digest before activation (the
    search record is written before the engine starts)."""
    base = _override_dir if override is None else override
    digest = hashlib.sha256()
    overridden: list[str] = []
    for name in template_names(base):
        path = template_path(name, base)
        # `overridden` names shadows of package templates; a new name the
        # dir adds is hashed but is an addition, not an override
        if path.parent != TEMPLATE_DIR and (TEMPLATE_DIR / path.name).is_file():
            overridden.append(name)
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return TemplatesDigest(sha256=digest.hexdigest(), overridden=overridden)


def lint_overrides(override: Path | None = None) -> list[str]:
    """Problems with the override dir's templates, one line each, empty
    when clean. An override may drop tokens (a leaner prompt) but never
    introduce one the engine does not fill — it would reach the agent as a
    literal `{{name}}`. A template with no package counterpart is checked
    only for being non-empty."""
    base = _override_dir if override is None else override
    if base is None or not base.is_dir():
        return []
    problems: list[str] = []
    for path in sorted(base.glob(f"*{TEMPLATE_SUFFIX}")):
        name = path.stem
        text = path.read_text()
        if not text.strip():
            problems.append(f"{path.name}: empty template")
            continue
        package = TEMPLATE_DIR / path.name
        if not package.is_file():
            continue
        unknown = tokens_in(text) - tokens_in(package.read_text())
        if unknown:
            problems.append(
                f"{path.name}: unknown token(s) {', '.join(sorted('{{' + t + '}}' for t in unknown))} "
                f"— the engine fills only {', '.join(sorted(tokens_in(package.read_text()))) or 'none'}"
            )
    return problems
