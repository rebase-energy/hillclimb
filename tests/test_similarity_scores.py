"""Pluggable similarity scores: contract, registry/loading, cache, built-ins, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from hillclimb import cli
from hillclimb.agents import openrouter
from hillclimb.config import Config
from hillclimb.modules.similarity import (
    SimilarityScore,
    SimilarityUnavailable,
    Solution,
    default_compare,
    get_score,
    register_score,
    registered_scores,
    similarity_matrix,
)
from hillclimb.modules.similarity import compute

SLSQP = """\
import numpy as np
from scipy.optimize import minimize

def solve_radii(centers):
    return minimize(lambda x: -np.sum(x), np.zeros(3), method="SLSQP")

print(solve_radii(np.ones((3, 2))))
"""

# the same program with every self-defined name changed
SLSQP_RENAMED = """\
import numpy as np
from scipy.optimize import minimize

def f0(v1):
    return minimize(lambda v2: -np.sum(v2), np.zeros(3), method="SLSQP")

print(f0(np.ones((3, 2))))
"""

LP = """\
import scipy.optimize as so

def radii(c):
    return so.linprog([1, 1], bounds=[(0, 1), (0, 1)])
"""


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setattr(compute, "cache_root", lambda: root)
    yield root


def write(tmp_path: Path, name: str, source: str) -> Solution:
    path = tmp_path / name / "solution.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return Solution(id=name, dir=path.parent)


# ---------------------------------------------------------------------------
# contract


def test_default_compare_by_shape():
    assert default_compare([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert default_compare([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert default_compare({"a": 1.0}, {"a": 2.0, "b": 0.0}) == pytest.approx(1.0)
    assert default_compare(frozenset("ab"), frozenset("bc")) == pytest.approx(1 / 3)
    assert default_compare(np.array([3.0, 4.0]), np.array([3.0, 4.0])) == pytest.approx(1.0)
    with pytest.raises(TypeError, match="override compare"):
        default_compare(object(), object())


def test_unknown_params_rejected_when_defaults_declared():
    class WithDefaults(SimilarityScore):
        name = "with-defaults"
        defaults = {"k": 1}

        def represent(self, solution):
            return [1.0]

    assert WithDefaults({"k": 2}).params == {"k": 2}
    with pytest.raises(ValueError, match="unknown params"):
        WithDefaults({"nope": 1})


def test_matrix_symmetric_with_unrepresented_rows(tmp_path):
    class Length(SimilarityScore):
        name = "length"

        def represent(self, solution):
            if "skip" in solution.source:
                return None
            return len(solution.source)

        def compare(self, a, b):
            return min(a, b) / max(a, b)

    sols = [write(tmp_path, "a", "x" * 10), write(tmp_path, "b", "x" * 5), write(tmp_path, "c", "skip")]
    m = similarity_matrix(Length(), sols)
    assert m.ids == ("a", "b", "c")
    assert m.get("a", "b") == m.get("b", "a") == pytest.approx(0.5)
    assert m.get("a", "a") == 1.0
    assert np.isnan(m.values[2]).all() and np.isnan(m.values[:, 2]).all()
    assert m.unrepresented == ("c",)
    assert m.to_dict()["similarity"][0][2] is None
    assert m.distances()[0, 1] == pytest.approx(0.5)


def test_one_failing_solution_is_unrepresented_but_unavailable_aborts(tmp_path):
    class Picky(SimilarityScore):
        name = "picky"

        def represent(self, solution):
            if solution.id == "bad":
                raise RuntimeError("boom")
            return [1.0]

    class NoKey(SimilarityScore):
        name = "no-key"

        def represent(self, solution):
            raise SimilarityUnavailable("no key")

    sols = [write(tmp_path, "good", "1"), write(tmp_path, "bad", "2")]
    assert similarity_matrix(Picky(), sols).unrepresented == ("bad",)
    with pytest.raises(SimilarityUnavailable):
        similarity_matrix(NoKey(), sols)


def test_cache_reuses_representations_until_content_or_version_changes(tmp_path):
    calls: list[str] = []

    class Counted(SimilarityScore):
        name = "counted"
        cache = True

        def represent(self, solution):
            calls.append(solution.id)
            return [float(len(solution.source)), 1.0]

    a, b = write(tmp_path, "a", "one"), write(tmp_path, "b", "three")
    similarity_matrix(Counted(), [a, b])
    similarity_matrix(Counted(), [write(tmp_path, "a", "one"), write(tmp_path, "b", "three")])
    assert calls == ["a", "b"]
    similarity_matrix(Counted(), [write(tmp_path, "a", "one"), write(tmp_path, "b", "changed")])
    assert calls == ["a", "b", "b"]

    class CountedV2(Counted):
        version = "2"

    similarity_matrix(CountedV2(), [write(tmp_path, "a", "one")] * 1 + [write(tmp_path, "b", "changed")])
    assert calls[-2:] == ["a", "b"]


# ---------------------------------------------------------------------------
# registry and loading


def test_builtins_registered():
    assert {"solution-card", "api-calls", "code-tokens"} <= set(registered_scores())


def test_unknown_name_lists_options():
    with pytest.raises(ValueError, match="api-calls"):
        get_score("nope")


def test_file_score_single_class_named_after_stem(tmp_path):
    path = tmp_path / "my_score.py"
    path.write_text(
        "from hillclimb.modules.similarity import SimilarityScore\n"
        "class Mine(SimilarityScore):\n"
        "    def represent(self, solution):\n"
        "        return {'n': float(len(solution.source))}\n"
    )
    score = get_score("my_score.py", {"anything": 1}, base_dir=tmp_path)
    assert score.name == "my_score"
    assert score.params == {"anything": 1}
    sols = [write(tmp_path, "a", "abc"), write(tmp_path, "b", "abcdef")]
    assert similarity_matrix(score, sols).get("a", "b") == pytest.approx(1.0)


def test_file_score_attr_wins_and_errors_name_the_file(tmp_path):
    two = tmp_path / "two.py"
    two.write_text(
        "from hillclimb.modules.similarity import SimilarityScore\n"
        "class A(SimilarityScore):\n    name = 'a'\n    def represent(self, s): return [1.0]\n"
        "class B(SimilarityScore):\n    name = 'b'\n    def represent(self, s): return [1.0]\n"
    )
    with pytest.raises(ValueError, match="exactly one"):
        get_score(str(two))
    two.write_text(two.read_text() + "SIMILARITY_SCORE = B\n")
    assert get_score(str(two)).name == "b"

    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('nope')\n")
    with pytest.raises(ValueError, match="broken.py failed to import"):
        get_score(str(broken))
    with pytest.raises(ValueError, match="not found"):
        get_score(str(tmp_path / "missing.py"))


def test_import_path_and_register(tmp_path):
    assert get_score("hillclimb.modules.similarity.builtin:ApiCalls").name == "api-calls"
    assert get_score("hillclimb.similarity_scores.builtin:ApiCalls").name == "api-calls"  # a config written before the move
    with pytest.raises(ValueError, match="not a SimilarityScore"):
        get_score("hillclimb.config:Config")

    class Registered(SimilarityScore):
        name = "test-registered"

        def represent(self, solution):
            return [1.0]

    register_score(Registered)
    try:
        assert isinstance(get_score("test-registered"), Registered)
    finally:
        from hillclimb.modules import similarity as similarity_scores

        similarity_scores._SCORES.pop("test-registered")


# ---------------------------------------------------------------------------
# built-ins


def test_api_calls_is_rename_invariant_and_resolves_aliases(tmp_path):
    score = get_score("api-calls")
    a, renamed, lp = write(tmp_path, "a", SLSQP), write(tmp_path, "r", SLSQP_RENAMED), write(tmp_path, "lp", LP)
    rep = score.represent(a)
    assert rep["call:scipy.optimize.minimize"] == 1
    assert rep["call:numpy.sum"] == 1
    assert rep["method:slsqp"] == 1
    assert "call:scipy.optimize.linprog" in score.represent(lp)
    m = similarity_matrix(score, [a, renamed, lp])
    assert m.get("a", "r") == pytest.approx(1.0)
    assert m.get("a", "lp") < 0.5
    assert score.represent(write(tmp_path, "bad", "def (:\n")) is None


def test_code_tokens_sees_renames(tmp_path):
    m = similarity_matrix(get_score("code-tokens"), [write(tmp_path, "a", SLSQP), write(tmp_path, "r", SLSQP_RENAMED)])
    assert 0.0 < m.get("a", "r") < 1.0


def test_solution_card_caches_cards_and_vectors(tmp_path, monkeypatch):
    chats: list[str] = []
    embeds: list[list[str]] = []

    def fake_chat(model, prompt, **_):
        chats.append(model)
        assert "Program:" in prompt and "{{source}}" not in prompt
        return "Algorithm family: SLSQP" if "minimize" in prompt else "Algorithm family: LP"

    def fake_embed(model, texts, **_):
        embeds.append(list(texts))
        return [[1.0, 0.0] if "SLSQP" in t else [0.0, 1.0] for t in texts]

    monkeypatch.setattr(openrouter, "chat", fake_chat)
    monkeypatch.setattr(openrouter, "embed", fake_embed)
    sols = lambda: [write(tmp_path, "a", SLSQP), write(tmp_path, "r", SLSQP_RENAMED), write(tmp_path, "lp", LP)]

    m = similarity_matrix(get_score("solution-card"), sols())
    assert m.get("a", "r") == pytest.approx(1.0) and m.get("a", "lp") == pytest.approx(0.0)
    assert len(chats) == 3 and len(embeds) == 1

    similarity_matrix(get_score("solution-card"), sols())
    assert len(chats) == 3 and len(embeds) == 1  # both layers cached

    # a new embedding model re-embeds but reuses every card
    similarity_matrix(get_score("solution-card", {"embedding_model": "other/model"}), sols())
    assert len(chats) == 3 and len(embeds) == 2
    assert get_score("solution-card").explain(sols()[2]) == "Algorithm family: LP"


def test_solution_card_without_key_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SimilarityUnavailable, match="OPENROUTER_API_KEY"):
        similarity_matrix(get_score("solution-card"), [write(tmp_path, "a", SLSQP), write(tmp_path, "b", LP)])


def test_solution_card_prompt_must_carry_source_token(tmp_path):
    prompt = tmp_path / "card.md"
    prompt.write_text("describe it")
    with pytest.raises(ValueError, match="source"):
        get_score("solution-card", {"prompt": str(prompt)})


def test_config_default_scores():
    assert list(Config().similarity.scores) == ["solution-card", "api-calls"]
    assert Config.model_validate({"similarity": {"scores": {"mine.py": {"k": 1}}}}).similarity.scores == {"mine.py": {"k": 1}}


# ---------------------------------------------------------------------------
# CLI


def test_cli_scores_on_files_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a, r, lp = write(tmp_path, "a", SLSQP), write(tmp_path, "r", SLSQP_RENAMED), write(tmp_path, "lp", LP)
    result = CliRunner().invoke(cli.app, [
        "similarity", "scores", "-s", "api-calls", "-s", "code-tokens", "--json",
        "-f", str(a.path), "-f", str(r.path), "-f", str(lp.path),
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [d["score"] for d in data] == ["api-calls", "code-tokens"]
    assert data[0]["ids"] == ["a/solution.py", "r/solution.py", "lp/solution.py"]
    assert data[0]["similarity"][0][1] == pytest.approx(1.0)


def test_cli_scores_reports_unavailable_and_continues(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    a, lp = write(tmp_path, "a", SLSQP), write(tmp_path, "lp", LP)
    result = CliRunner().invoke(cli.app, [
        "similarity", "scores", "-s", "solution-card", "-s", "api-calls", "-f", str(a.path), "-f", str(lp.path),
    ])
    assert result.exit_code == 0, result.output
    assert "solution-card: unavailable" in result.output
    assert "== api-calls" in result.output
