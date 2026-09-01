from __future__ import annotations

import json
import importlib
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

from hillclimb.executor import CommandExecutor
from hillclimb.integrations.einsteinarena.baselines import BASELINES
from hillclimb.integrations.einsteinarena.eval_runner import __file__ as eval_runner_file
from hillclimb.problem import load_problem, resolve_target
from hillclimb.api import create_search
from hillclimb.run import load_search_meta

provider_module = importlib.import_module("hillclimb.integrations.einsteinarena.provider")


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def problem_payload(slug="toy", verifier=None, scoring="maximize"):
    return {
        "id": 17,
        "title": f"Problem {slug}",
        "description": "Optimize a public construction.",
        "scoring": scoring,
        "minImprovement": 1e-10,
        "verifier": verifier
        or "def evaluate(data):\n    return float(data['value'])\n",
        "solutionSchema": {"value": "number"},
    }


def install_fake_api(monkeypatch, detail, leaderboard=None, calls=None):
    calls = calls if calls is not None else []

    def fake_urlopen(request, timeout):
        calls.append((request.get_method(), request.full_url, timeout, dict(request.headers)))
        assert request.get_method() == "GET"
        if urlparse(request.full_url).path == "/api/leaderboard":
            return FakeResponse(
                leaderboard
                if leaderboard is not None
                else [{"rank": 1, "bestScore": 3.25, "agentName": "ReferenceAgent"}]
            )
        return FakeResponse(detail)

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    return calls


def test_smoke_target_is_a_fixed_three_problem_suite(config):
    resolved = resolve_target("einsteinarena://smoke", config)
    assert resolved.kind == "suite"
    assert [entry.target for entry in resolved.suite.problems] == [
        "einsteinarena://circle-packing",
        "einsteinarena://difference-bases",
        "einsteinarena://heilbronn-triangles",
    ]


def test_load_materializes_pinned_read_only_problem(tmp_path, config, monkeypatch):
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "cache"))
    calls = install_fake_api(monkeypatch, problem_payload())

    spec = load_problem("einsteinarena://toy", config)

    assert spec.provider_revision and len(spec.provider_revision) == 64
    assert spec.target == f"einsteinarena://toy@sha256:{spec.provider_revision}"
    assert spec.problem_key == "einsteinarena://toy"
    assert spec.output_artifacts == ["submission.json"]
    assert spec.chart_baselines == {"#1 ReferenceAgent": 3.25}
    assert (spec.problem_dir / "problem.json").is_file()
    assert (spec.problem_dir / "verifier.py").read_text().startswith("def evaluate")
    assert (spec.problem_dir / "solution-schema.json").is_file()
    assert "submission.json" in (spec.problem_dir / "description.md").read_text()
    assert all(method == "GET" for method, *_ in calls)
    assert all("Authorization" not in headers for *_, headers in calls)

    search_dir = create_search(config, spec, config.paths.runs_dir / "r", "r", 60)
    meta = load_search_meta(search_dir)
    assert meta.problem == spec.target
    assert meta.problem_key == "einsteinarena://toy"
    assert meta.provider_revision == spec.provider_revision
    assert meta.output_artifacts == ["submission.json"]


def test_pinned_target_reuses_snapshot_without_refetching_problem(tmp_path, config, monkeypatch):
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "cache"))
    install_fake_api(monkeypatch, problem_payload())
    first = load_problem("einsteinarena://toy", config)

    def offline(request, timeout):
        raise OSError("offline")

    monkeypatch.setattr(provider_module, "urlopen", offline)
    resumed = load_problem(first.target, config)

    assert resumed.provider_revision == first.provider_revision
    assert resumed.problem_dir == first.problem_dir


def test_pinned_target_rejects_live_revision_drift(tmp_path, config, monkeypatch):
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "cache"))
    install_fake_api(monkeypatch, problem_payload())
    wrong = "0" * 64
    with pytest.raises(RuntimeError, match="revision drift"):
        load_problem(f"einsteinarena://toy@sha256:{wrong}", config)


def test_problem_validation_rejects_malformed_contract(tmp_path, config, monkeypatch):
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "cache"))
    payload = problem_payload()
    del payload["verifier"]
    install_fake_api(monkeypatch, payload)
    with pytest.raises(ValueError, match="missing fields: verifier"):
        load_problem("einsteinarena://toy", config)


def test_scoring_direction_maps_minimize(tmp_path, config, monkeypatch):
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "cache"))
    install_fake_api(monkeypatch, problem_payload(scoring="minimize"), leaderboard=[])
    assert not load_problem("einsteinarena://toy", config).higher_is_better


def test_local_runner_preserves_full_precision(tmp_path, config, monkeypatch):
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "cache"))
    score = 0.12345678901234566
    install_fake_api(
        monkeypatch,
        problem_payload(verifier=f"def evaluate(data):\n    return {score!r}\n"),
    )
    spec = load_problem("einsteinarena://toy", config)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    solution = candidate / "solution.py"
    solution.write_text(
        'import json\nfrom pathlib import Path\nPath("submission.json").write_text(json.dumps({"value": 1}))\n'
    )

    result = CommandExecutor(Path(sys.executable), spec.verifier_cmd).execute(
        solution, candidate, 30
    )

    assert result.ok
    assert result.val_score == score
    assert json.loads((candidate / "submission.json").read_text()) == {"value": 1}


def test_local_runner_rejects_non_finite_verifier_score(tmp_path, config, monkeypatch):
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "cache"))
    install_fake_api(
        monkeypatch,
        problem_payload(verifier="def evaluate(data):\n    return float('inf')\n"),
    )
    spec = load_problem("einsteinarena://toy", config)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    solution = candidate / "solution.py"
    solution.write_text(
        'from pathlib import Path\nPath("submission.json").write_text("{}")\n'
    )

    result = CommandExecutor(Path(sys.executable), spec.verifier_cmd).execute(
        solution, candidate, 30
    )

    assert not result.ok
    assert result.val_score is None


@pytest.mark.parametrize(
    ("slug", "verifier"),
    [
        (
            "circle-packing",
            "def evaluate(data):\n"
            "    assert len(data['circles']) == 26\n"
            "    return sum(row[2] for row in data['circles'])\n",
        ),
        (
            "difference-bases",
            "def evaluate(data):\n"
            "    assert data['set'] == [0, 1]\n"
            "    return 4.0\n",
        ),
        (
            "heilbronn-triangles",
            "def evaluate(data):\n"
            "    assert len(data['points']) == 11\n"
            "    return 0.0\n",
        ),
    ],
)
def test_smoke_baselines_are_executable_and_valid(tmp_path, slug, verifier):
    work = tmp_path / slug
    work.mkdir()
    solution = work / "solution.py"
    solution.write_text(BASELINES[slug])
    verifier_path = work / "verifier.py"
    verifier_path.write_text(verifier)
    result = work / "eval_result.json"

    completed = subprocess.run(
        [
            sys.executable,
            eval_runner_file,
            str(solution),
            "--verifier",
            str(verifier_path),
            "--result",
            str(result),
        ],
        cwd=work,
        check=False,
    )

    assert completed.returncode == 0
    assert isinstance(json.loads((work / "submission.json").read_text()), dict)
    assert isinstance(json.loads(result.read_text())["score"], (int, float))
