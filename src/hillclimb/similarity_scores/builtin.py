"""Built-in scores that need nothing but the source file."""

from __future__ import annotations

import ast
import io
import tokenize
from collections import Counter

from hillclimb.similarity_scores.base import SimilarityScore, Solution

_CODE_TOKENS = frozenset({tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING})


class CodeTokens(SimilarityScore):
    """Jaccard overlap of the source's token sets — the `structural`
    distance of `similarity map`. Sensitive to naming: renaming a function
    changes the score."""

    name = "code-tokens"
    description = "Jaccard overlap of solution.py tokens (name-sensitive)"

    def represent(self, solution: Solution) -> frozenset[str] | None:
        source = solution.source
        try:
            tokens = {
                t.string for t in tokenize.generate_tokens(io.StringIO(source).readline)
                if t.type in _CODE_TOKENS
            }
        except (tokenize.TokenError, IndentationError, SyntaxError):
            tokens = set(source.split())
        return frozenset(tokens) or None


class ApiCalls(SimilarityScore):
    """Cosine over what the solution imports and which library functions it
    calls, resolved through import aliases (`np.linalg.norm` and
    `from numpy.linalg import norm` count as one feature), plus string
    `method=` choices (`minimize(..., method="SLSQP")`). Names the solution
    defines itself never appear, so renaming changes nothing; hand-written
    algorithms that call no library are invisible to it."""

    name = "api-calls"
    description = "cosine over imports + library calls + method= choices (rename-invariant)"

    def represent(self, solution: Solution) -> dict[str, float] | None:
        try:
            tree = ast.parse(solution.source)
        except SyntaxError:
            return None
        alias: dict[str, str] = {}
        bag: Counter[str] = Counter()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                    bag[f"import:{a.name}"] += 1
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for a in node.names:
                    alias[a.asname or a.name] = f"{node.module}.{a.name}"
                    bag[f"import:{node.module}.{a.name}"] += 1

        def dotted(expr: ast.expr) -> str | None:
            if isinstance(expr, ast.Name):
                return alias.get(expr.id)
            if isinstance(expr, ast.Attribute):
                base = dotted(expr.value)
                return f"{base}.{expr.attr}" if base else None
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = dotted(node.func)
                if target:
                    bag[f"call:{target}"] += 1
                for kw in node.keywords:
                    if kw.arg == "method" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        bag[f"method:{kw.value.value.lower()}"] += 1
        return {k: float(v) for k, v in bag.items()} or None
