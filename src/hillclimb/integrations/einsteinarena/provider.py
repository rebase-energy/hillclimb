"""Einstein Arena problem provider (public GETs and local evaluation only)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from hillclimb.config import Config
from hillclimb.problem import ProblemSpec, ResolvedTarget, SuiteSpec
from hillclimb.project import machine_cache_dir

from .baselines import BASELINES

SCHEME = "einsteinarena"
CACHE_DIRNAME = "benchmark-problems/einsteinarena"
SMOKE_PROBLEMS = ("circle-packing", "difference-bases", "heilbronn-triangles")
PIN_MARKER = "@sha256:"
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
REVISION_RE = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_FIELDS = (
    "id",
    "title",
    "description",
    "scoring",
    "minImprovement",
    "verifier",
    "solutionSchema",
)


def _base_url(config: Config) -> str:
    return os.environ.get("EINSTEIN_ARENA_BASE_URL", config.einsteinarena.base_url).rstrip("/")


def _get_json(path: str, config: Config, params: dict[str, object] | None = None) -> Any:
    url = urljoin(_base_url(config) + "/", path.lstrip("/"))
    if params:
        url += "?" + urlencode(params)
    request = Request(
        url,
        method="GET",
        headers={"Accept": "application/json", "User-Agent": "hillclimb/0.2"},
    )
    try:
        with urlopen(request, timeout=config.einsteinarena.request_timeout_s) as response:
            body = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Einstein Arena GET failed for {url}: {exc}") from exc
    try:
        return json.loads(body)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Einstein Arena returned invalid JSON for {url}") from exc


def _parse_name(name: str) -> tuple[str, str | None]:
    slug, marker, revision = name.rpartition(PIN_MARKER)
    if not marker:
        slug, revision = name, None
    if not SLUG_RE.fullmatch(slug):
        raise ValueError(f"invalid Einstein Arena problem slug: {slug!r}")
    if revision is not None and not REVISION_RE.fullmatch(revision):
        raise ValueError(f"invalid Einstein Arena sha256 revision: {revision!r}")
    return slug, revision


def _canonical_problem(payload: Any, slug: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Einstein Arena problem {slug!r} must be a JSON object")
    missing = [field for field in CANONICAL_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Einstein Arena problem {slug!r} is missing fields: {', '.join(missing)}")
    canonical = {field: payload[field] for field in CANONICAL_FIELDS}
    for field in ("title", "description", "verifier"):
        if not isinstance(canonical[field], str) or not canonical[field].strip():
            raise ValueError(f"Einstein Arena problem {slug!r} has invalid {field}")
    if canonical["scoring"] not in {"maximize", "minimize"}:
        raise ValueError(f"Einstein Arena problem {slug!r} has invalid scoring direction")
    threshold = canonical["minImprovement"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"Einstein Arena problem {slug!r} has invalid minImprovement")
    if not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError(f"Einstein Arena problem {slug!r} has invalid minImprovement")
    if not isinstance(canonical["solutionSchema"], dict):
        raise ValueError(f"Einstein Arena problem {slug!r} has invalid solutionSchema")
    return canonical


def _revision(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_dir(slug: str, revision: str) -> Path:
    return machine_cache_dir() / CACHE_DIRNAME / slug / revision


def _read_snapshot(slug: str, revision: str) -> tuple[dict[str, Any], Path] | None:
    directory = _snapshot_dir(slug, revision)
    path = directory / "problem.json"
    if not path.is_file():
        return None
    try:
        payload = _canonical_problem(json.loads(path.read_text()), slug)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"corrupt Einstein Arena cache snapshot: {directory}") from exc
    actual = _revision(payload)
    if actual != revision:
        raise RuntimeError(
            f"corrupt Einstein Arena cache snapshot {directory}: expected {revision}, got {actual}"
        )
    return payload, directory


def _materialize(slug: str, payload: dict[str, Any], revision: str, base_url: str) -> Path:
    final = _snapshot_dir(slug, revision)
    if (final / "problem.json").is_file():
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=".build-", dir=final.parent))
    try:
        (build / "problem.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        (build / "verifier.py").write_text(payload["verifier"].rstrip() + "\n")
        (build / "solution-schema.json").write_text(
            json.dumps(payload["solutionSchema"], indent=2, sort_keys=True) + "\n"
        )
        # GEPA's proposer reads this file directly instead of Greedy's fully
        # rendered prompt, so it must carry the JSON output contract too.
        (build / "description.md").write_text(_description(payload, slug, revision))
        (build / "provenance.json").write_text(
            json.dumps(
                {
                    "provider": SCHEME,
                    "slug": slug,
                    "revision": revision,
                    "source": f"{base_url}/api/problems/{slug}",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
        try:
            build.rename(final)
        except FileExistsError:
            pass
    finally:
        if build.exists():
            shutil.rmtree(build)
    return final


def _problem_snapshot(name: str, config: Config) -> tuple[str, str, dict[str, Any], Path]:
    slug, pinned = _parse_name(name)
    if pinned:
        cached = _read_snapshot(slug, pinned)
        if cached is not None:
            payload, directory = cached
            return slug, pinned, payload, directory
    payload = _canonical_problem(_get_json(f"/api/problems/{slug}", config), slug)
    revision = _revision(payload)
    if pinned and pinned != revision:
        raise RuntimeError(
            f"Einstein Arena revision drift for {slug}: pinned {pinned}, live response is {revision}"
        )
    directory = _materialize(slug, payload, revision, _base_url(config))
    return slug, revision, payload, directory


def _leaderboard_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("leaderboard", "items", "results", "data"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _leaderboard_baselines(problem_id: object, config: Config) -> dict[str, float]:
    try:
        payload = _get_json(
            "/api/leaderboard", config, {"problem_id": problem_id, "limit": 10}
        )
    except (RuntimeError, ValueError):
        return {}
    baselines: dict[str, float] = {}
    for rank, row in enumerate(_leaderboard_rows(payload), start=1):
        # The current public endpoint calls these fields bestScore/agentName;
        # tolerate the older/generic spellings so mirrors remain usable.
        score = row.get("bestScore", row.get("score"))
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        if not math.isfinite(float(score)):
            continue
        agent = row.get("agent")
        if isinstance(agent, dict):
            agent = agent.get("name")
        if not isinstance(agent, str):
            agent = row.get("agent_name") or row.get("agentName") or row.get("name")
        label = f"#{rank} {agent}" if isinstance(agent, str) and agent else f"#{rank} Arena"
        baselines[label] = float(score)
    return baselines


def _description(payload: dict[str, Any], slug: str, revision: str) -> str:
    schema = json.dumps(payload["solutionSchema"], indent=2, sort_keys=True)
    direction = "higher" if payload["scoring"] == "maximize" else "lower"
    return (
        f"# {payload['title']}\n\n"
        f"{payload['description'].strip()}\n\n"
        "## Hillclimb / Einstein Arena contract\n\n"
        "Your `solution.py` must write a JSON object to `submission.json`. "
        "The pinned public Arena verifier evaluates that object locally.\n\n"
        f"Solution schema:\n\n```json\n{schema}\n```\n\n"
        f"The Arena score is optimized directly ({direction} is better). "
        f"Arena's rank-1 threshold is {payload['minImprovement']!r}; it is leaderboard "
        "context only and is not Hillclimb's candidate acceptance band.\n\n"
        f"Verifier snapshot: `{slug}@sha256:{revision}`. Downloaded verifier code executes "
        "locally in the managed runtime; the content hash provides reproducibility, not E2B isolation.\n"
    )


class EinsteinArenaProvider:
    def load_problem(self, name: str, config: Config) -> ProblemSpec:
        slug, revision, payload, problem_dir = _problem_snapshot(name, config)
        eval_runner = Path(__file__).with_name("eval_runner.py")
        baselines = _leaderboard_baselines(payload["id"], config)
        return ProblemSpec(
            problem_id=slug,
            problem_dir=problem_dir.resolve(),
            data_dir=problem_dir.resolve(),
            description=_description(payload, slug, revision),
            metric_name="arena_score",
            higher_is_better=payload["scoring"] == "maximize",
            time_budget_s=config.budget.total_s,
            chart_baselines=baselines,
            verifier_cmd=[
                "{python}",
                str(eval_runner),
                "{solution}",
                "--verifier",
                str(problem_dir / "verifier.py"),
                "--result",
                "{result}",
            ],
            verifier_display=(
                f"the pinned Einstein Arena verifier for {slug} (must write ./submission.json)"
            ),
            report_trusted=True,
            runtime="csv",
            contract=(
                "`solution.py` must write `submission.json`, containing one JSON object that "
                "matches the schema in the problem description. Do not write `eval_result.json`; "
                "the pinned Einstein Arena verifier owns the score."
            ),
            baseline_text=BASELINES.get(slug),
            baseline_summary=(
                "baseline: valid-by-construction Einstein Arena seed"
                if slug in BASELINES
                else "baseline"
            ),
            output_artifacts=["submission.json"],
            provider_target=f"{SCHEME}://{slug}{PIN_MARKER}{revision}",
            provider_revision=revision,
            problem_key_override=f"{SCHEME}://{slug}",
        )

    def resolve_target(self, name: str, config: Config) -> ResolvedTarget:
        if name == "smoke":
            return ResolvedTarget(
                kind="suite",
                suite=SuiteSpec(
                    suite_id="einsteinarena-smoke",
                    suite_path=Path("."),
                    problems=[f"{SCHEME}://{slug}" for slug in SMOKE_PROBLEMS],
                ),
            )
        return ResolvedTarget(kind="problem", problem=self.load_problem(name, config))

    def chart_baselines(self, name: str) -> dict[str, float]:
        config = Config()
        _slug, _revision_value, payload, _directory = _problem_snapshot(name, config)
        return _leaderboard_baselines(payload["id"], config)


provider = EinsteinArenaProvider()
