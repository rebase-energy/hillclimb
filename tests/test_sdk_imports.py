"""Climber code meets the harness through `hillclimb.sdk` and nothing else.

The rule is the proof that the sdk is sufficient: the bundled policies,
tuners and similarity scores are written as if they were third-party. Every
exception is listed in ALLOWED with the phase of the harness + climber plan
that removes it — the dict only ever shrinks.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "hillclimb"

# modules that are (or become) part of a bundled climber
CLIMBER_MODULES = [
    # gepa: the loop, its operator, its scoring view (config.py / __init__.py
    # are the harness-side glue that reads Config until the manifest exists)
    "integrations/gepa/loop.py",
    "integrations/gepa/operator.py",
    "integrations/gepa/evaluator.py",
    "integrations/gepa/proposer.py",
    "operators/builtin.py",
    "policies/greedy.py",
    "policies/openevolve.py",
    "tuners/random_search.py",
    "tuners/optuna.py",
    "similarity_scores/builtin.py",
    "similarity_scores/solution_card.py",
]

# (module, imported name) -> why it is still allowed, and until when
ALLOWED = {
    ("policies/openevolve.py", "hillclimb.policies.greedy"):
        "reuses greedy's draft/debug decisions; becomes an sdk helper with the climber bundle (7a)",
    ("similarity_scores/solution_card.py", "hillclimb.openrouter"):
        "direct LLM + embedding client; moves behind an sdk completion service (7a)",
    ("similarity_scores/solution_card.py", "hillclimb.similarity_scores.base"):
        "pre-sdk import; switched together with the completion service (7a)",
    ("similarity_scores/solution_card.py", "hillclimb.similarity_scores.compute"):
        "DiskCache for cards and vectors; exported by the sdk with the completion service (7a)",
}


def hillclimb_imports(path: Path) -> set[str]:
    """Every `hillclimb...` module a file imports, at any nesting depth."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "hillclimb":
                found.update(f"hillclimb.{alias.name}" for alias in node.names)
            elif node.module.startswith("hillclimb."):
                found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.split(".")[0] == "hillclimb")
    return found


def own_package(module: str) -> str:
    """`integrations/gepa/loop.py` -> `hillclimb.integrations.gepa`: a
    climber's modules may import each other."""
    return "hillclimb." + ".".join(Path(module).parent.parts)


def test_climber_modules_import_only_the_sdk():
    offenders = []
    for module in CLIMBER_MODULES:
        siblings = own_package(module) + "."
        for name in sorted(hillclimb_imports(SRC / module)):
            if name == "hillclimb.sdk" or (module, name) in ALLOWED:
                continue
            if module.startswith("integrations/") and name.startswith(siblings):
                continue
            offenders.append(f"{module}: imports {name} — export it from hillclimb.sdk instead")
    assert not offenders, "\n".join(offenders)


def test_allowed_exceptions_are_all_still_needed():
    """A stale exception is a hole: drop the entry once the import is gone."""
    stale = [
        f"{module}: no longer imports {name}"
        for (module, name) in ALLOWED
        if name not in hillclimb_imports(SRC / module)
    ]
    assert not stale, "\n".join(stale)


def test_sdk_exports_resolve():
    import hillclimb.sdk as sdk

    for name in sdk.__all__:
        assert getattr(sdk, name) is not None, name


def test_sdk_is_lazy():
    """Importing the sdk must not drag the engine in (climber authors, and
    modules the sdk re-exports, import it at the top of the file)."""
    import subprocess
    import sys

    code = "import sys, hillclimb.sdk; print(sorted(m for m in sys.modules if m.startswith('hillclimb.')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "['hillclimb.sdk']", out
