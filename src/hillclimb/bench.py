"""The learning A/B benchmark: does cross-search memory actually help?

Every feature in the memory stack claims to improve search outcomes; this is
the measuring stick. `hillclimb bench run` executes paired searches on one
problem — a memory-blind arm (`--no-learning`) and a memory-full arm —
SEQUENTIALLY, off-arm first within each pair, so the on-arm learns between
pairs exactly as it would in production while the off-arm never sees a card.
`hillclimb bench report` groups any finished searches by
`SearchMeta.learning_enabled` and compares the arms on the selected
candidate's holdout score (`status.json` — always written, independent of
learning), falling back to val when holdout is off.

Pure functions here; orchestration and printing live in the CLI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from hillclimb.run import iter_run_dirs, iter_search_dirs, load_run_meta, load_search_meta
from hillclimb.status import effective_state, read_status

BENCH_PREFIX = "bench-"


@dataclass(frozen=True)
class BenchRow:
    run_id: str
    run_name: str
    problem_id: str
    learning: bool
    state: str
    holdout: float | None
    val: float | None
    lower_is_better: bool
    started_at: str

    @property
    def score(self) -> float | None:
        """Comparison score: holdout when available (the honest metric),
        else val."""
        return self.holdout if self.holdout is not None else self.val


@dataclass(frozen=True)
class BenchSummary:
    problem_id: str
    pairs: list[tuple[BenchRow, BenchRow]]  # (off, on)
    unpaired: list[BenchRow]
    on_wins: int
    off_wins: int
    ties: int


def bench_run_name(problem_slug: str, pair: int, learning: bool) -> str:
    return f"{BENCH_PREFIX}{problem_slug}-p{pair}-{'on' if learning else 'off'}"


def collect_bench_results(
    runs_dir: Path, *, problem_id: str = "", include_all: bool = False
) -> list[BenchRow]:
    """One row per finished search. Default: only `bench-*` runs (the
    orchestrated pairs); `include_all` groups every finished search by its
    recorded learning flag instead."""
    rows: list[BenchRow] = []
    for run_dir in iter_run_dirs(runs_dir):
        run_meta = load_run_meta(run_dir)
        run_name = run_meta.name if run_meta else run_dir.name
        if not include_all and not run_name.startswith(BENCH_PREFIX):
            continue
        for search_dir in iter_search_dirs(run_dir):
            meta = load_search_meta(search_dir)
            status = read_status(search_dir)
            if meta is None or status is None:
                continue
            if problem_id and meta.problem_id != problem_id:
                continue
            state = effective_state(search_dir)
            if state == "running":
                continue
            selected = status.selected
            rows.append(BenchRow(
                run_id=run_dir.name,
                run_name=run_name,
                problem_id=meta.problem_id,
                learning=meta.learning_enabled,
                state=state,
                holdout=selected.holdout_score if selected else None,
                val=selected.val_score if selected else None,
                lower_is_better=meta.lower_is_better,
                started_at=meta.started_at,
            ))
    rows.sort(key=lambda r: r.started_at)
    return rows


def _better(a: float, b: float, lower_is_better: bool) -> bool:
    return a < b if lower_is_better else a > b


def pair_and_summarize(rows: list[BenchRow]) -> list[BenchSummary]:
    """Chronological off/on pairing per problem. Rows that never found a
    partner are reported, not silently dropped."""
    by_problem: dict[str, list[BenchRow]] = {}
    for row in rows:
        by_problem.setdefault(row.problem_id, []).append(row)
    summaries: list[BenchSummary] = []
    for problem_id, problem_rows in sorted(by_problem.items()):
        off_queue = [r for r in problem_rows if not r.learning]
        on_queue = [r for r in problem_rows if r.learning]
        pairs = list(zip(off_queue, on_queue))
        unpaired = off_queue[len(pairs):] + on_queue[len(pairs):]
        on_wins = off_wins = ties = 0
        for off, on in pairs:
            if off.score is None or on.score is None:
                ties += 1
            elif _better(on.score, off.score, on.lower_is_better):
                on_wins += 1
            elif _better(off.score, on.score, on.lower_is_better):
                off_wins += 1
            else:
                ties += 1
        summaries.append(BenchSummary(
            problem_id=problem_id, pairs=pairs, unpaired=unpaired,
            on_wins=on_wins, off_wins=off_wins, ties=ties,
        ))
    return summaries


def _fmt(value: float | None) -> str:
    return f"{value:.5g}" if value is not None else "-"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def render_bench_report(summaries: list[BenchSummary]) -> str:
    """Markdown-ish text report: per-pair rows, per-arm means, verdict."""
    if not summaries:
        return "no finished benchmark searches found (run `hillclimb bench run <target>` first)"
    lines: list[str] = []
    for summary in summaries:
        direction = (
            "lower is better" if (summary.pairs and summary.pairs[0][0].lower_is_better)
            or (summary.unpaired and summary.unpaired[0].lower_is_better)
            else "higher is better"
        )
        lines.append(f"## {summary.problem_id} ({direction})")
        lines.append("")
        lines.append("| pair | off (no memory) | on (memory) | winner |")
        lines.append("|---|---|---|---|")
        for index, (off, on) in enumerate(summary.pairs, 1):
            if off.score is None or on.score is None:
                winner = "-"
            elif _better(on.score, off.score, on.lower_is_better):
                winner = "on"
            elif _better(off.score, on.score, on.lower_is_better):
                winner = "off"
            else:
                winner = "tie"
            lines.append(f"| {index} | {_fmt(off.score)} | {_fmt(on.score)} | {winner} |")
        off_scores = [off.score for off, _ in summary.pairs if off.score is not None]
        on_scores = [on.score for _, on in summary.pairs if on.score is not None]
        lines.append("")
        lines.append(
            f"arm means: off {_fmt(_mean(off_scores))} vs on {_fmt(_mean(on_scores))}  "
            f"| learning arm wins {summary.on_wins}/{len(summary.pairs)} pair(s), "
            f"loses {summary.off_wins}, ties {summary.ties}"
        )
        if summary.unpaired:
            arms = ", ".join(
                f"{row.run_name} ({'on' if row.learning else 'off'}, {_fmt(row.score)})"
                for row in summary.unpaired
            )
            lines.append(f"unpaired: {arms}")
        lines.append("")
    return "\n".join(lines).rstrip()


def slugify_target(target: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", target).strip("-") or "target"
