from __future__ import annotations

from types import SimpleNamespace

import pytest

from hillclimb import benchmark_providers
from hillclimb.problem import ProblemSpec, ResolvedTarget, resolve_target


def test_registry_rejects_duplicate_and_invalid_schemes():
    with pytest.raises(ValueError, match="already registered"):
        benchmark_providers.register_benchmark_provider("einsteinarena", lambda: object())
    with pytest.raises(ValueError, match="invalid"):
        benchmark_providers.register_benchmark_provider("bad://scheme", lambda: object())


def test_registered_provider_uses_generic_target_seam(tmp_path, config, monkeypatch):
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    calls = []

    class FakeProvider:
        def load_problem(self, name, config_arg):
            calls.append(("load", name, config_arg))
            return ProblemSpec(
                problem_id=name,
                problem_dir=problem_dir,
                data_dir=problem_dir,
                description="fake benchmark",
                metric_name="score",
                higher_is_better=True,
                time_budget_s=60,
                verifier_cmd=["true"],
                provider_target=f"fakebench://{name}@v1",
                problem_key_override=f"fakebench://{name}",
            )

        def resolve_target(self, name, config_arg):
            calls.append(("resolve", name, config_arg))
            return ResolvedTarget(kind="problem", problem=self.load_problem(name, config_arg))

    monkeypatch.setitem(benchmark_providers._PROVIDERS, "fakebench", lambda: FakeProvider())

    resolved = resolve_target("fakebench://toy", config)

    assert resolved.kind == "problem"
    assert resolved.problem.target == "fakebench://toy@v1"
    assert resolved.problem.problem_key == "fakebench://toy"
    assert calls[0][:2] == ("resolve", "toy")


def test_provider_chart_baselines_are_optional(monkeypatch):
    class FakeProvider:
        pass

    monkeypatch.setitem(benchmark_providers._PROVIDERS, "nochart", lambda: FakeProvider())

    from hillclimb.problem import provider_chart_baselines

    assert provider_chart_baselines("nochart://toy") == {}


def test_unknown_provider_error_names_registered_providers():
    with pytest.raises(ValueError, match="einsteinarena"):
        benchmark_providers.get_benchmark_provider("missing")
