from __future__ import annotations

from tests.factories import trial as mk_trial

import json
import os
from pathlib import Path

import pytest

from hillclimb.harness.candidate import AgentInfo, Candidate
from hillclimb.config import Config
from hillclimb.harness.control import read_commands
from hillclimb.harness.store import FileDataStore, key_for
from hillclimb.harness.journal import Journal
from hillclimb.harness.run import RunMeta, SearchMeta, load_search_meta, write_run_meta, write_search_meta
from hillclimb.harness.status import SearchStatus, write_status
from hillclimb.tui.watch import (
    DETAIL_MIN_HEIGHT,
    DETAIL_STEP,
    WatchApp,
    candidate_detail_lines,
    candidate_detail_renderables,
    candidate_rows,
    render_stream_line,
    scan_runs,
    scan_searches,
    stream_tail,
)

DEAD_PID = 2**22


def _record(search_dir):
    """The search's store record, the way the TUI hands it to the detail renderers."""
    return FileDataStore(search_dir.parents[2]).search(key_for(search_dir))


def make_candidate(candidate_id: str, **kwargs) -> Candidate:
    val_score = kwargs.pop("val_score", None)
    if val_score is not None:
        kwargs["trials"] = [mk_trial(val_score=val_score)]
    return Candidate(candidate_id=candidate_id, operator=kwargs.pop("operator", "draft"), **kwargs)


def make_run_with_search(
    runs_dir: Path,
    run_id: str,
    status: SearchStatus | None = None,
    run_name: str = "Demo",
    search_id: str = "circle-packing",
) -> Path:
    """Create a v2 run with one search; returns the search dir."""
    run_dir = runs_dir / run_id
    write_run_meta(
        run_dir,
        RunMeta(run_id=run_id, name=run_name, target="demo", problem_ids=[search_id]),
    )
    search_dir = run_dir / "searches" / search_id
    (search_dir / "candidates").mkdir(parents=True)
    (search_dir / "best").mkdir()
    write_search_meta(
        search_dir,
        SearchMeta(
            search_id=search_id,
            run_id=run_id,
            problem="/tmp/problem",
            problem_id=search_id,
            agent="claude-code",
            model="sonnet",
            metric="score",
            higher_is_better=True,
            budget_s=3600,
        ),
    )
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_result(make_candidate("c000", operator="baseline", status="passing"))
    c001 = make_candidate("c001", operator="draft", status="passing", val_score=0.7)
    c001.agent = AgentInfo(name="claude-code", total_tokens=240_000, cost_usd=0.5)
    journal.candidate_result(c001)
    c002 = make_candidate("c002", operator="improve", parent_id="c001", status="buggy", pruned=True)
    c002.agent = AgentInfo(name="claude-code", total_tokens=1_000_000, cost_usd=1.75)
    journal.candidate_result(c002)
    if status is not None:
        write_status(search_dir, status)
    return search_dir


def make_demo_search(
    tmp_path: Path,
    run_id: str,
    status: SearchStatus | None = None,
    *,
    wide: bool = False,
) -> tuple[Path, Config]:
    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(
        runs_dir, run_id, status or SearchStatus(search_id="circle-packing", run_id=run_id, state="done")
    )
    if wide:
        journal = Journal(search_dir / "journal.jsonl")
        for i in range(3, 12):
            journal.candidate_result(
                make_candidate(
                    f"c{i:03d}",
                    status="passing",
                    val_score=float(i),
                    summary="wide summary " * 20,
                )
            )
    config = Config()
    config.paths.runs_dir = runs_dir
    return search_dir, config


DETAIL_MIN_HEIGHT_FOR_TESTS = 6  # == watch.DETAIL_MIN_HEIGHT


async def open_candidate_detail(pilot) -> None:
    await pilot.press("enter")  # runs -> searches
    await pilot.press("o")  # searches -> full candidate screen
    await pilot.press("enter")  # candidate -> detail
    await pilot.pause()


def test_scan_runs_and_searches_with_status(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    make_run_with_search(
        runs_dir,
        "20260701-run",
        SearchStatus(search_id="circle-packing", run_id="20260701-run", state="done"),
    )
    rows = scan_runs(runs_dir)
    assert len(rows) == 1
    assert rows[0].state == "done"
    assert rows[0].name == "Demo"
    assert rows[0].searches == "1"
    assert rows[0].candidates == "3"
    assert rows[0].tokens == "1.24M"
    assert rows[0].spend == "$2.25"

    search_rows = scan_searches(runs_dir, "20260701-run")
    assert len(search_rows) == 1
    assert search_rows[0].problem == "circle-packing"
    assert search_rows[0].climber == "greedy"  # the default optimizer
    assert search_rows[0].agent == "claude-code"
    assert search_rows[0].candidates == "3 (2 passing)"
    assert search_rows[0].buggy == 1  # c002 crashed: the cell goes red
    assert search_rows[0].tokens == "1.24M"  # 240k + 1.0M, summed across candidates
    assert search_rows[0].spend == "$2.25"  # 0.5 + 1.75, summed the same way


def test_run_row_sums_usage_across_searches(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    make_run_with_search(runs_dir, "run", search_id="p1")
    make_run_with_search(runs_dir, "run", search_id="p2")

    row = scan_runs(runs_dir)[0]
    assert row.searches == "2"
    assert row.tokens == "2.48M"
    assert row.spend == "$4.50"


def test_fmt_tokens():
    from hillclimb.tui.watch import _fmt_tokens

    assert _fmt_tokens(None) == "-"
    assert _fmt_tokens(0) == "-"
    assert _fmt_tokens(812) == "812"
    assert _fmt_tokens(24_500) == "24.5k"
    assert _fmt_tokens(1_240_000) == "1.24M"


def test_fmt_cost():
    from hillclimb.tui.watch import _fmt_cost

    assert _fmt_cost(None) == "-"
    assert _fmt_cost(0.0) == "-"
    assert _fmt_cost(0.004) == "$0.00"
    assert _fmt_cost(12.345) == "$12.35"
    assert _fmt_cost(1234.5) == "$1234.50"


def test_estimate_cost_usd_prices_each_token_kind():
    from hillclimb.agents.claude_code import estimate_cost_usd

    usage = {
        "input_tokens": 1_000_000, "output_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
    }
    # opus: $5 in, $25 out, cache write 1.25x in, cache read 0.1x in
    assert estimate_cost_usd(usage, "claude-opus-5") == pytest.approx(5 + 25 + 6.25 + 0.5)
    # sonnet-4-6 keeps its own rate; sonnet-5 the cheaper one
    assert estimate_cost_usd({"output_tokens": 1_000_000}, "claude-sonnet-4-6") == pytest.approx(15)
    assert estimate_cost_usd({"output_tokens": 1_000_000}, "claude-sonnet-5") == pytest.approx(10)
    assert estimate_cost_usd({"output_tokens": 1_000_000}, "claude-haiku-4-5") == pytest.approx(5)
    # missing kinds count as zero; an unknown model is None, not a guess
    assert estimate_cost_usd({}, "claude-opus-5") == 0.0
    assert estimate_cost_usd(usage, None) is None
    assert estimate_cost_usd(usage, "gpt-5") is None


def test_stream_cost_live_estimate_then_final(tmp_path: Path):
    import json

    from hillclimb.tui.watch import _stream_cost_usd

    ws = tmp_path / "cand"
    ws.mkdir()
    assert _stream_cost_usd(ws) == 0.0  # no stream yet
    stream = ws / "agent_stream.jsonl"
    lines = [json.dumps({"type": "system", "subtype": "init", "model": "claude-opus-5"})]
    # two turns, each streamed twice under one id (partial then final) — the
    # newest version of each turn is what gets priced
    for mid, out in (("m1", 100), ("m1", 1000), ("m2", 2000)):
        lines.append(json.dumps({"type": "assistant", "message": {
            "id": mid, "model": "claude-opus-5",
            "usage": {"input_tokens": 0, "output_tokens": out,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1_000_000}}}))
    stream.write_text("\n".join(lines) + "\n")
    # m1: 1000 out + 1M cache read; m2: 2000 out + 1M cache read, at opus rates
    expected = (1000 + 2000) * 25 / 1e6 + 2 * 1_000_000 * 5 / 1e6 * 0.1
    assert _stream_cost_usd(ws) == pytest.approx(expected)

    # the final result's own cost wins over the estimate once the call ends
    with stream.open("a") as fh:
        fh.write(json.dumps({"type": "result", "subtype": "success", "total_cost_usd": 3.21,
                             "usage": {"input_tokens": 1, "output_tokens": 1}}) + "\n")
    assert _stream_cost_usd(ws) == pytest.approx(3.21)


def test_stream_cost_unknown_model_contributes_nothing(tmp_path: Path):
    import json

    from hillclimb.tui.watch import _stream_cost_usd

    ws = tmp_path / "cand"
    ws.mkdir()
    (ws / "agent_stream.jsonl").write_text(json.dumps({"type": "assistant", "message": {
        "id": "m1", "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}}}) + "\n")
    assert _stream_cost_usd(ws) == 0.0


def test_stream_tokens_live_and_final(tmp_path: Path):
    import json

    from hillclimb.tui.watch import _stream_tokens

    ws = tmp_path / "cand"
    ws.mkdir()
    assert _stream_tokens(ws) == 0  # no stream yet

    def turn(mid, out, cc, cr):
        return json.dumps({"type": "assistant", "message": {"id": mid, "usage": {
            "input_tokens": 1, "output_tokens": out,
            "cache_creation_input_tokens": cc, "cache_read_input_tokens": cr}}})

    # each turn streams twice under one id (partial then final) -> deduped
    lines = [turn("m1", 2, 100, 500), turn("m1", 2, 100, 500), turn("m2", 3, 50, 900)]
    (ws / "agent_stream.jsonl").write_text("\n".join(lines) + "\n")
    # unique turns: (1+2+100+500) + (1+3+50+900)
    assert _stream_tokens(ws) == 603 + 954

    # once the result lands it is authoritative, replacing the per-turn estimate
    result = json.dumps({"type": "result", "usage": {
        "input_tokens": 10, "output_tokens": 8209,
        "cache_creation_input_tokens": 29297, "cache_read_input_tokens": 199902}})
    (ws / "agent_stream.jsonl").write_text("\n".join(lines + [result]) + "\n")
    assert _stream_tokens(ws) == 10 + 8209 + 29297 + 199902

    # a half-written trailing line (agent still streaming) is ignored
    (ws / "agent_stream.jsonl").write_text("\n".join(lines) + "\n{\"type\": \"assis")
    assert _stream_tokens(ws) == 603 + 954


def test_search_row_counts_in_flight_tokens(tmp_path: Path):
    import json

    from hillclimb.harness.status import CurrentCandidate, SearchStatus
    from hillclimb.tui.watch import _search_row

    runs_dir = tmp_path / "runs"
    make_run_with_search(
        runs_dir, "20260701-run",
        SearchStatus(search_id="circle-packing", run_id="20260701-run", state="running"),
    )
    search_dir = runs_dir / "20260701-run" / "searches" / "circle-packing"
    # an in-flight candidate the journal does not know about yet
    live = search_dir / "candidates" / "c003"
    live.mkdir(parents=True)
    (live / "agent_stream.jsonl").write_text(json.dumps({"type": "result", "total_cost_usd": 0.25, "usage": {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 500_000}}) + "\n")
    status = SearchStatus(
        search_id="circle-packing", run_id="20260701-run", state="running",
        current=[CurrentCandidate(candidate_id="c003", operator="improve", phase="agent",
                                  candidate_dir=str(live))],
    )
    from hillclimb.harness.status import write_status
    write_status(search_dir, status)
    store = FileDataStore(search_dir.parents[2])
    row = _search_row(store, store.search(key_for(search_dir)))
    # 240k + 1.0M finished (from make_run_with_search) + 500k in-flight = 1.74M
    assert row.tokens == "1.74M"
    # $0.50 + $1.75 finished + $0.25 in-flight
    assert row.spend == "$2.50"
    run_row = scan_runs(runs_dir)[0]
    assert run_row.tokens == "1.74M"
    assert run_row.spend == "$2.50"


def test_search_row_shows_resolved_model_id(tmp_path: Path):
    """The model cell upgrades from the route alias to the fully-qualified
    id once a journaled candidate reports one (vendor prefix stripped)."""
    from hillclimb.tui.watch import _search_row

    import json

    from hillclimb.harness.status import CurrentCandidate, SearchStatus, write_status

    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(runs_dir, "20260701-run")
    store = FileDataStore(runs_dir)
    row = _search_row(store, store.search(key_for(search_dir)))
    assert row.model == "sonnet"  # nothing resolved yet -> alias

    # an in-flight operator whose live stream has announced the model
    live = search_dir / "candidates" / "c003"
    live.mkdir(parents=True)
    (live / "agent_stream.jsonl").write_text(
        json.dumps({"type": "system", "subtype": "init", "model": "claude-sonnet-4-5-20250929"}) + "\n"
    )
    write_status(search_dir, SearchStatus(
        search_id="circle-packing", run_id="20260701-run", state="running",
        current=[CurrentCandidate(candidate_id="c003", operator="draft", phase="agent",
                                  candidate_dir=str(live))],
    ))
    row = _search_row(store, store.search(key_for(search_dir)))
    assert row.model == "sonnet-4-5-20250929"  # live stream, journal not yet

    journal = Journal(search_dir / "journal.jsonl")
    c003 = make_candidate("c003", operator="draft", status="passing")
    c003.agent = AgentInfo(name="claude-code", model="sonnet",
                               model_id="claude-sonnet-4-5-20250929")
    journal.candidate_result(c003)
    row = _search_row(store, store.search(key_for(search_dir)))
    assert row.model == "sonnet-4-5-20250929"


def test_display_model_strips_vendor_prefix_from_alias_too():
    """A search whose agent has not reported a model yet (GEPA's proposer
    works outside the candidate dirs) falls back to the configured alias —
    which must read like its resolved neighbours, not `claude-opus-5` next
    to `opus-5`."""
    from hillclimb.tui.watch import _display_model

    assert _display_model("claude-opus-5", None) == "opus-5"
    assert _display_model("claude-opus-5", "claude-opus-5") == "opus-5"
    assert _display_model("claude-opus-5", "<synthetic>") == "opus-5"
    assert _display_model("sonnet", None) == "sonnet"
    assert _display_model(None, None) == "-"


def test_search_row_shows_requested_model_after_synthetic_error(tmp_path: Path):
    """A persisted Claude Code API-error marker is not a model identity."""
    from hillclimb.tui.watch import _search_row

    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(runs_dir, "20260701-run")
    failed = make_candidate("c003", operator="draft", status="abandoned")
    failed.agent = AgentInfo(
        name="claude-code",
        model="claude-opus-5",
        model_id="<synthetic>",
        error_kind="rate_limited",
    )
    Journal(search_dir / "journal.jsonl").candidate_result(failed)

    store = FileDataStore(runs_dir)
    row = _search_row(store, store.search(key_for(search_dir)))
    assert row.model == "opus-5"


def test_scan_searches_shows_the_policy_and_the_arm_when_it_differs(tmp_path: Path):
    """An optimizer comparison names its experiments after the policies, so the
    policy column carries it and the problem cell stays bare; an experiment that
    means something else (a model comparison) still tags the problem."""
    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(runs_dir, "exp-run")
    meta = load_search_meta(search_dir)
    meta.climber, meta.study, meta.experiment = "gepa", "optimizers", "gepa"
    write_search_meta(search_dir, meta)
    row = scan_searches(runs_dir, "exp-run")[0]
    assert (row.problem, row.climber) == ("circle-packing", "gepa")

    meta.climber, meta.experiment = "greedy", "opus"
    write_search_meta(search_dir, meta)
    row = scan_searches(runs_dir, "exp-run")[0]
    assert (row.problem, row.climber) == ("circle-packing [opus]", "greedy")


def test_scan_searches_agent_lists_every_harness_a_route_used(tmp_path: Path):
    """The configured harness first, then any other one a per-operator
    route authored a candidate in (candidates without a agent name,
    such as the baseline, add nothing)."""
    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(runs_dir, "routed-run")
    journal = Journal(search_dir / "journal.jsonl")
    c003 = make_candidate("c003", operator="improve", parent_id="c001", status="passing", val_score=0.8)
    c003.agent = AgentInfo(name="codex", model="gpt-5")
    journal.candidate_result(c003)
    assert scan_searches(runs_dir, "routed-run")[0].agent == "claude-code+codex"


def test_sort_search_rows_best_first_within_each_problem():
    from hillclimb.tui.watch import SearchRow, sort_search_rows

    def row(search_id: str, problem: str, score: float | None, higher: bool = True) -> SearchRow:
        return SearchRow(
            search_id=search_id, problem=problem, climber="greedy", agent="dummy", model="m",
            tokens="-", spend="-", state="done", candidates="-", best_val="-", selected="-",
            duration="-", best_score=score, higher_is_better=higher,
        )

    rows = [row("a", "p", 0.3), row("b", "p", None), row("c", "q", 1.0), row("d", "p", 0.9), row("e", "q", 2.0)]
    assert sort_search_rows(rows, by_best=False) is rows
    # problems keep the run's order; inside one, best first, unscored last
    assert [r.search_id for r in sort_search_rows(rows, by_best=True)] == ["d", "a", "b", "e", "c"]
    lower = [row("x", "p", 0.5, higher=False), row("y", "p", 0.1, higher=False)]
    assert [r.search_id for r in sort_search_rows(lower, by_best=True)] == ["y", "x"]


def test_candidates_cell_is_red_on_any_crash_else_green(tmp_path: Path):
    from hillclimb.tui.watch import candidates_style

    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(runs_dir, "r")  # c002 is buggy
    assert candidates_style(scan_searches(runs_dir, "r")[0]) == "red"
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_result(make_candidate("c002", operator="improve", parent_id="c001", status="passing", val_score=0.8))
    row = scan_searches(runs_dir, "r")[0]  # replay keeps the last record: healed
    assert (row.buggy, candidates_style(row)) == (0, "green")
    # a candidate the clock cut off is a warning, not a bug
    journal.candidate_result(make_candidate("c003", operator="draft", status="abandoned"))
    row = scan_searches(runs_dir, "r")[0]
    assert (row.abandoned, candidates_style(row)) == (1, "yellow")
    journal.candidate_result(make_candidate("c004", operator="draft", status="buggy"))
    assert candidates_style(scan_searches(runs_dir, "r")[0]) == "red"  # a crash outranks a cut


def test_scan_searches_detects_crash(tmp_path: Path):
    make_run_with_search(
        tmp_path / "runs",
        "crashed-run",
        SearchStatus(search_id="circle-packing", run_id="crashed-run", state="running", pid=DEAD_PID),
    )
    assert scan_searches(tmp_path / "runs", "crashed-run")[0].state == "crashed"


def test_scan_runs_skips_v1_layout(tmp_path: Path):
    """Old flat-layout dirs (run.yaml without schema_version) are invisible
    and must not crash the scanner."""
    runs_dir = tmp_path / "runs"
    old = runs_dir / "20260101-000000-legacy"
    (old / "nodes").mkdir(parents=True)
    (old / "run.yaml").write_text("run_id: legacy\nproblem_id: x\nbudget_s: 60\n")
    make_run_with_search(runs_dir, "20260701-run")

    rows = scan_runs(runs_dir)
    assert [row.run_id for row in rows] == ["20260701-run"]


def test_candidate_rows_tree_order_and_pruned(tmp_path: Path):
    search_dir = make_run_with_search(tmp_path / "runs", "r")
    rows = candidate_rows(Journal(search_dir / "journal.jsonl"))
    assert [r.candidate_id for r in rows] == ["c000", "c001", "c002"]
    assert rows[2].label == "└─ c002"  # child connected under c001
    assert rows[2].guide == "└─ "
    assert rows[0].guide == "" and rows[1].guide == ""  # roots stay flush
    assert "PRUNED" in rows[2].marks
    assert "strike" in rows[2].style


def test_candidate_rows_star_the_current_selection(tmp_path: Path):
    """With the metric's direction the search's current selection wears a
    star and is the only SELECTED row — records keep `is_selected` from the
    moment they landed, so history alone would mark several."""
    from hillclimb.tui.watch import BEST_STAR

    search_dir = make_run_with_search(tmp_path / "runs", "r")
    journal = Journal(search_dir / "journal.jsonl")
    c003 = make_candidate("c003", operator="improve", parent_id="c001", status="passing", val_score=0.9)
    c003.is_selected = True
    journal.candidate_result(c003)
    c001 = journal.candidates["c001"]
    c001.is_selected = True  # selected when it landed, never cleared
    journal.candidate_result(c001)

    rows = candidate_rows(Journal(search_dir / "journal.jsonl"), higher_is_better=True)
    starred = [r.candidate_id for r in rows if r.label.endswith(BEST_STAR)]
    assert starred == ["c003"]
    # a child row keeps the star through the cell styling (the guide is
    # restyled dim, the id and star keep the row's style)
    from hillclimb.tui.watch import _styled_candidate_cells

    (c003_row,) = [r for r in rows if r.candidate_id == "c003"]
    assert c003_row.guide  # c003 is a child of c001
    assert str(_styled_candidate_cells(c003_row, holdout=False)[0]) == c003_row.label
    assert [r.candidate_id for r in rows if "SELECTED" in r.marks] == ["c003"]
    # without the direction the rows fall back to the records' own flags
    rows = candidate_rows(Journal(search_dir / "journal.jsonl"))
    assert not any(BEST_STAR in r.label for r in rows)
    assert {r.candidate_id for r in rows if "SELECTED" in r.marks} == {"c001", "c003"}


def test_candidate_rows_branch_guides(tmp_path: Path):
    """Siblings get ├─/└─ and a │ continuation runs past an open branch."""
    search_dir = make_run_with_search(tmp_path / "runs", "r")
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_created(make_candidate("c003", parent_id="c001", status="passing"))
    journal.candidate_created(make_candidate("c004", parent_id="c002", status="passing"))
    journal = Journal(search_dir / "journal.jsonl")
    labels = {r.candidate_id: r.label for r in candidate_rows(journal)}
    assert labels["c002"] == "├─ c002"  # no longer the last child of c001
    assert labels["c004"] == "│  └─ c004"  # under c002, with c003 still below
    assert labels["c003"] == "└─ c003"


def test_candidate_rows_show_pending_as_running_or_stale(tmp_path: Path):
    from hillclimb.tui.watch import display_status

    search_dir = make_run_with_search(tmp_path / "runs", "r")
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_created(make_candidate("c003", operator="improve", parent_id="c001", status="pending"))
    journal = Journal(search_dir / "journal.jsonl")
    live = {r.candidate_id: r.status for r in candidate_rows(journal, live=True)}
    dead = {r.candidate_id: r.status for r in candidate_rows(journal, live=False)}
    assert live["c003"] == "running" and dead["c003"] == "stale"
    assert live["c001"] == dead["c001"] == "passing"  # only pending is remapped
    assert display_status("buggy", True) == "buggy"


def test_candidate_detail_lines_include_scores_lineage_and_notes(tmp_path: Path):
    search_dir = make_run_with_search(tmp_path / "runs", "r")
    candidate_dir = search_dir / "candidates" / "c001"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "notes.md").write_text("tried nearest-neighbor seed\nkept deterministic order\n")
    (candidate_dir / "exec_stdout.log").write_text("val_score: 0.7\n")
    (candidate_dir / "exec_stderr.log").write_text("warning: local search plateau\n")
    (candidate_dir / "agent_stream.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "I will try a constructive heuristic."},
        ]}})
        + "\n"
    )

    detail = "\n".join(
        candidate_detail_lines(_record(search_dir), Journal(search_dir / "journal.jsonl"), "c001")
    )

    assert "Candidate c001 | draft | passing" in detail
    assert "Score: val=0.7  holdout=-  metric=score (higher is better)" in detail
    assert "Parent: root  Children: 1  Path: c001" in detail
    assert "c002  improve  buggy  val=-  PRUNED" in detail
    assert "Trial: returncode=" in detail
    assert "Notes:" in detail
    assert "tried nearest-neighbor seed" in detail
    assert "Stderr:" in detail
    assert "warning: local search plateau" in detail
    assert "Stdout:" in detail
    assert "val_score: 0.7" in detail
    assert "Operator stream:" in detail
    assert "I will try a constructive heuristic." in detail


def test_candidate_detail_lines_baseline_without_trial(tmp_path: Path):
    """The baseline has no trials; the detail panel must not crash."""
    search_dir = make_run_with_search(tmp_path / "runs", "r")
    detail = "\n".join(
        candidate_detail_lines(_record(search_dir), Journal(search_dir / "journal.jsonl"), "c000")
    )
    assert "Candidate c000 | baseline | passing" in detail
    assert "Trial: (not executed)" in detail


def test_candidate_detail_renderables_are_sectioned(tmp_path: Path):
    from rich.console import Console

    search_dir = make_run_with_search(tmp_path / "runs", "r")
    candidate_dir = search_dir / "candidates" / "c001"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "notes.md").write_text("tried nearest-neighbor seed\n")
    (candidate_dir / "exec_stdout.log").write_text("val_score: 0.7\n")
    (candidate_dir / "exec_stderr.log").write_text("warning: local search plateau\n")

    renderables = candidate_detail_renderables(
        _record(search_dir),
        Journal(search_dir / "journal.jsonl"),
        "c001",
    )
    console = Console(record=True, width=100)
    for renderable in renderables:
        console.print(renderable)
    rendered = console.export_text()

    assert len(renderables) >= 4
    assert "Candidate c001" in rendered
    assert "val" in rendered
    assert "0.7" in rendered
    assert "Children" in rendered
    assert "c002" in rendered
    assert "Notes" in rendered
    assert "tried nearest-neighbor seed" in rendered
    assert "Stderr" in rendered
    assert "Stdout" in rendered


def test_render_stream_line_shapes():
    assistant = json.dumps(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Let me look at the data."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "head train.csv"}},
        ]}}
    )
    rendered = render_stream_line(assistant)
    assert "Let me look at the data." in rendered
    assert "→ Bash(head train.csv)" in rendered

    result = json.dumps({"type": "result", "subtype": "success", "num_turns": 5, "total_cost_usd": 1.25})
    assert render_stream_line(result) == "[result] success turns=5 cost=$1.25"

    system = json.dumps({"type": "system", "subtype": "init", "session_id": "s1"})
    assert "init" in render_stream_line(system)

    assert render_stream_line("not json at all") == "not json at all"
    assert render_stream_line(json.dumps({"type": "user"})) is None


def test_parse_stream_line_timestamps_kinds_and_noise():
    from hillclimb.tui.watch import parse_stream_line

    stamped = json.dumps({"type": "assistant", "ts": "2026-08-23T05:33:01+00:00", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
    ]}})
    entry = parse_stream_line(stamped)
    assert entry.kind == "tool" and entry.text == "→ Bash(ls)"
    assert len(entry.ts) == 8 and entry.ts.count(":") == 2  # local HH:MM:SS
    assert render_stream_line(stamped) == f"{entry.ts}  → Bash(ls)"

    # per-turn bookkeeping is noise; only init is shown, with its model
    assert parse_stream_line(json.dumps({"type": "system", "subtype": "thinking_tokens", "session_id": "s"})) is None
    init = parse_stream_line(json.dumps({"type": "system", "subtype": "init", "session_id": "s", "model": "opus"}))
    assert init.kind == "system" and "model=opus" in init.text

    failed = parse_stream_line(json.dumps({"type": "result", "subtype": "error", "is_error": True, "num_turns": 2}))
    assert failed.kind == "error"
    assert parse_stream_line("banner").kind == "raw"

    # the argument that matters, on one line, not Edit's leading boolean flag
    edit = json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Edit", "input": {"replace_all": False, "file_path": "solution.py", "old_string": "x"}},
    ]}})
    assert parse_stream_line(edit).text == "→ Edit(solution.py)"
    multi = json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "python3 -c \"\nimport re\n  x = 1\n\""}},
    ]}})
    assert parse_stream_line(multi).text == '→ Bash(python3 -c " import re x = 1 ")'


def test_running_candidate_detail_shows_agent_tokens_and_elapsed(tmp_path: Path):
    from rich.console import Console

    from hillclimb.harness.candidate import AgentInfo, Candidate, utcnow

    search_dir = make_run_with_search(tmp_path / "runs", "r")
    journal = Journal(search_dir / "journal.jsonl")
    candidate_dir = search_dir / "candidates" / "c009"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "agent_stream.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"id": "m1", "usage": {"input_tokens": 1500, "output_tokens": 500}, "content": []}}) + "\n"
    )
    journal.candidate_created(
        Candidate(
            candidate_id="c009", operator="draft", status="running", candidate_dir=str(candidate_dir),
            agent=AgentInfo(name="claude-code", model="sonnet"), created_at=utcnow(),
        )
    )
    journal = Journal(search_dir / "journal.jsonl")
    console = Console(record=True, width=120)
    for renderable in candidate_detail_renderables(_record(search_dir), journal, "c009", live=True):
        console.print(renderable)
    rendered = console.export_text()
    assert "claude-code" in rendered and "model=sonnet" in rendered
    assert "2.0k so far" in rendered
    assert "elapsed" in rendered and "0s" in rendered
    assert "path" in rendered and "candidates/c009" in rendered  # the full working dir (may wrap)
    assert "lineage" in rendered


def test_path_link_is_short_label_with_file_uri(tmp_path: Path):
    from hillclimb.tui.watch import _path_label, _path_link

    path = tmp_path / "runs" / "r1" / "searches" / "cp" / "candidates" / "c003"
    assert _path_label(path) == "runs/…/candidates/c003"
    link = _path_link(path)
    assert link.plain == "runs/…/candidates/c003"
    assert f"link {path.resolve().as_uri()}" in str(link.spans[0].style)
    assert not link.style  # no base style: cell padding must not carry the link
    assert _path_label(Path("/odd/place")) == "/odd/place"  # no runs/ layout: full path


def test_open_in_file_manager_uses_the_desktop_opener(tmp_path: Path, monkeypatch):
    from hillclimb.tui.watch import open_in_file_manager

    calls = []
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: calls.append(cmd))
    open_in_file_manager(tmp_path)
    assert calls[0][-1] == str(tmp_path)
    assert calls[0][0] in ("open", "xdg-open")


def test_scrollbars_are_whole_cell_and_one_wide():
    from textual.scrollbar import ScrollBar

    from hillclimb.tui.theme import WholeCellScrollBarRender

    assert ScrollBar.renderer is WholeCellScrollBarRender
    # a fractional position: the stock renderer would draw ▁/▃ partial cells
    segments = WholeCellScrollBarRender.render_bar(
        size=10, virtual_size=37, window_size=10, position=7.3, thickness=1, vertical=True
    ).segments
    assert {s.text for s in segments} == {" "}
    thumb = [s for s in segments if s.style.reverse]
    assert 1 <= len(thumb) < 10  # a real thumb, not the whole track


@pytest.mark.asyncio
async def test_detail_and_table_scrollbars_are_one_cell(tmp_path: Path):
    search_dir, config = make_demo_search(tmp_path, "bar-run")
    app = WatchApp(config)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.press("enter")
        await pilot.press("o")
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.query_one("#candidates").styles.scrollbar_size_vertical == 1
        assert app.screen.query_one("#candidate-detail").styles.scrollbar_size_vertical == 1


@pytest.mark.asyncio
async def test_searches_scrollbar_is_wide_and_drag_selection_is_disabled(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "bar-run")
    app = WatchApp(config)

    async with app.run_test(size=(80, 20)) as pilot:
        assert app.ALLOW_SELECT is False
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.query_one("#searches").styles.scrollbar_size_vertical == 2


def test_stream_tail(tmp_path: Path):
    (tmp_path / "agent_stream.jsonl").write_text(
        json.dumps({"type": "system", "subtype": "init", "session_id": "s"}) + "\n"
        + json.dumps({"type": "result", "subtype": "success", "num_turns": 1, "total_cost_usd": 0.1}) + "\n"
    )
    lines = stream_tail(tmp_path)
    assert len(lines) == 2
    assert lines[-1].startswith("[result]")
    assert stream_tail(tmp_path / "missing") == []


@pytest.mark.asyncio
async def test_watch_app_lists_searches_and_stops(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(
        runs_dir,
        "live-run",
        SearchStatus(
            search_id="circle-packing", run_id="live-run", state="running", pid=os.getpid()
        ),
    )
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test() as pilot:
        table = app.screen.query_one("#runs")
        assert table.row_count == 1
        await pilot.press("enter")
        await pilot.pause()
        table = app.screen.query_one("#searches")
        assert table.row_count == 1
        await pilot.press("s")  # stop highlighted search…
        await pilot.press("y")  # …confirm
        await pilot.pause()
    commands = read_commands(search_dir)
    assert len(commands) == 1
    assert commands[0][1].action == "stop"
    assert commands[0][1].source == "tui"


@pytest.mark.asyncio
async def test_double_clicking_run_opens_its_searches(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    make_run_with_search(runs_dir, "run")
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test() as pilot:
        runs_screen = app.screen
        await pilot.click("#runs", offset=(2, 1))
        await pilot.pause()
        assert app.screen is runs_screen  # one click only highlights

        await pilot.click("#runs", offset=(2, 1), times=2)
        await pilot.pause()
        assert app.screen is not runs_screen
        assert app.screen.query_one("#searches").row_count == 1


@pytest.mark.asyncio
async def test_hover_lightly_highlights_row_without_moving_cursor(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    make_run_with_search(runs_dir, "run-1")
    make_run_with_search(runs_dir, "run-2")
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test() as pilot:
        table = app.screen.query_one("#runs")
        cursor_before = table.cursor_coordinate

        await pilot.hover("#runs", offset=(2, 2))  # header, row 0, then row 1
        await pilot.pause()

        assert table.hover_coordinate.row == 1
        assert table.cursor_coordinate == cursor_before
        hover = table.get_component_rich_style("datatable--hover")
        assert hover.bgcolor is not None
        assert hover.bgcolor.triplet == (26, 32, 36)  # theme $panel


@pytest.mark.asyncio
async def test_double_clicking_search_opens_same_candidate_panel_as_enter(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "run")

    app = WatchApp(config)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        searches_screen = app.screen

        await pilot.click("#searches", offset=(2, 1))
        await pilot.pause()
        assert app.screen is searches_screen  # one click only highlights

        await pilot.click("#searches", offset=(2, 1), times=2)
        await pilot.pause()
        assert app.screen is searches_screen
        panel = app.screen.query_one("#search-candidates")
        assert panel.row_count == 3
        assert app.screen._panel_search_id == "circle-packing"
        assert app.focused is panel


@pytest.mark.asyncio
async def test_watch_has_no_command_palette_or_header_trigger(tmp_path: Path):
    from textual.widgets._header import HeaderIcon

    runs_dir = tmp_path / "runs"
    make_run_with_search(runs_dir, "run")
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test() as pilot:
        assert not app.query(HeaderIcon)
        screen = app.screen
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert app.screen is screen


@pytest.mark.asyncio
async def test_searches_table_refresh_preserves_scroll_offsets(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    run_id = "20260704-many-searches"
    for i in range(30):
        make_run_with_search(
            runs_dir,
            run_id,
            SearchStatus(search_id=f"problem-{i:02d}", run_id=run_id, state="done"),
            search_id=f"problem-{i:02d}-with-a-very-wide-name-for-columns",
        )
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test(size=(50, 10)) as pilot:
        await pilot.press("enter")
        await pilot.pause()
        table = app.screen.query_one("#searches")
        await pilot.pause()
        if table.max_scroll_x == 0 or table.max_scroll_y == 0:
            pytest.skip("headless table did not overflow in both axes")

        table.scroll_to(x=table.max_scroll_x, y=8, immediate=True, force=True)
        await pilot.pause()
        before = (table.scroll_x, table.scroll_y)

        app.screen.refresh_data()
        await pilot.pause()

        assert (table.scroll_x, table.scroll_y) == before


@pytest.mark.asyncio
async def test_runs_table_refresh_preserves_scroll_offsets(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    for i in range(30):
        make_run_with_search(
            runs_dir,
            f"20260704-run-{i:02d}",
            SearchStatus(search_id="circle-packing", run_id=f"20260704-run-{i:02d}", state="done"),
            run_name=f"Long run name {i:02d} with wide columns",
        )
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test(size=(50, 10)) as pilot:
        table = app.screen.query_one("#runs")
        await pilot.pause()
        if table.max_scroll_x == 0 or table.max_scroll_y == 0:
            pytest.skip("headless runs table did not overflow in both axes")

        table.scroll_to(x=table.max_scroll_x, y=8, immediate=True, force=True)
        await pilot.pause()
        before = (table.scroll_x, table.scroll_y)

        app.screen.refresh_data()
        await pilot.pause()

        assert (table.scroll_x, table.scroll_y) == before


@pytest.mark.asyncio
async def test_trackpad_scrolls_focused_table_from_screen_chrome(tmp_path: Path):
    from textual import events

    runs_dir = tmp_path / "runs"
    for i in range(20):
        make_run_with_search(
            runs_dir,
            f"run-{i:02d}",
            run_name=f"Long run {i:02d} with enough text to overflow horizontally",
        )
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test(size=(50, 10)) as pilot:
        table = app.screen.query_one("#runs")
        assert table.max_scroll_x > 0 and table.max_scroll_y > 0

        # The header is outside the table. Gestures there still control the
        # focused run list, which makes the large blank area useful as well.
        await pilot._post_mouse_events([events.MouseScrollRight], offset=(10, 0))
        await pilot._post_mouse_events([events.MouseScrollDown], offset=(10, 0))
        await pilot.pause()
        assert table.scroll_x > 0
        assert table.scroll_y > 0


@pytest.mark.asyncio
async def test_candidate_table_refresh_preserves_scroll_offsets(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(
        runs_dir,
        "wide-candidate-run",
        SearchStatus(search_id="circle-packing", run_id="wide-candidate-run", state="done"),
    )
    journal = Journal(search_dir / "journal.jsonl")
    for i in range(3, 40):
        journal.candidate_result(
            make_candidate(
                f"c{i:03d}",
                status="passing",
                val_score=float(i),
                summary="wide summary " * 20,
            )
        )
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test(size=(50, 10)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.press("o")  # -> full candidate screen
        await pilot.pause()
        table = app.screen.query_one("#candidates")
        if table.max_scroll_x == 0 or table.max_scroll_y == 0:
            pytest.skip("headless candidate table did not overflow in both axes")

        table.scroll_to(x=table.max_scroll_x, y=8, immediate=True, force=True)
        await pilot.pause()
        before = (table.scroll_x, table.scroll_y)

        app.screen.refresh_data()
        await pilot.pause()

        assert (table.scroll_x, table.scroll_y) == before


@pytest.mark.asyncio
async def test_t_opens_the_tree_panel_and_follows_the_cursor(tmp_path: Path):
    search_dir, config = make_demo_search(tmp_path, "tree-run")
    # a second search in the same run, with a distinct extra candidate
    second = make_run_with_search(
        tmp_path / "runs", "tree-run", SearchStatus(search_id="cp-2", run_id="tree-run", state="done"),
        search_id="cp-2",
    )
    journal2 = Journal(second / "journal.jsonl")
    # scrubber ticks come from finished_at (real searches always stamp it)
    for i, (cid, kwargs) in enumerate([
        ("c000", dict(operator="baseline", status="passing")),
        ("c001", dict(operator="draft", status="passing", val_score=0.7)),
        ("c777", dict(operator="improve", parent_id="c001", status="passing", val_score=0.9)),
    ]):
        candidate = make_candidate(cid, **kwargs)
        candidate.finished_at = f"2026-08-23T10:0{i}:00+00:00"
        journal2.candidate_result(candidate)

    app = WatchApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.press("t")
        await pilot.pause()
        tree = app.screen.query_one("#search-tree")
        assert str(tree.styles.display) != "none"
        # opens as tall as the searches allow (2 searches: fit under both)
        assert app.screen._detail_height == app.screen._fit_detail_height(max_table_rows=8)
        first_ids = {n.id for n in tree._tree.nodes}
        await pilot.press("down")  # cursor to the second search: the tree follows
        await pilot.pause()
        second_ids = {n.id for n in tree._tree.nodes}
        assert first_ids != second_ids and "c777" in second_ids
        # j scrubs back in time: the newest result drops out of the tree
        scrubber = app.screen.query_one("#search-scrubber")
        assert str(scrubber.styles.display) != "none"
        n_live = len(tree._tree.nodes)
        await pilot.press("j")
        await pilot.pause()
        assert scrubber.index is not None
        assert len(tree._tree.nodes) < n_live
        await pilot.press("k")  # forward again
        await pilot.pause()
        assert len(tree._tree.nodes) == n_live

        # selecting a node slides the candidate detail out on the right
        detail = app.screen.query_one("#search-node-detail")
        assert str(detail.styles.display) == "none"
        tree.selected = "c001"
        app.screen._show_node_detail("c001")
        assert str(detail.styles.display) == "block"
        await pilot.pause()
        assert detail.max_scroll_x == 0  # content fits the dock: no horizontal scroll
        # stream entries stay one row each: the long tool line scrolls, not wraps
        cdir = second / "candidates" / "c777"
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "agent_stream.jsonl").write_text(json.dumps({
            "type": "assistant", "ts": "2026-08-24T04:00:00+00:00",
            "message": {"content": [{"type": "tool_use", "name": "Bash",
                                     "input": {"command": "x" * 200}}]},
        }) + "\n")
        app.screen._show_node_detail("c777")
        await pilot.pause()
        assert detail.max_scroll_x > 0  # horizontal overflow, no line break

        app.screen._show_node_detail("c001")
        tree.selected = "c001"
        await pilot.pause()
        await pilot.press("escape")  # first escape: deselect/hide the detail, tree stays
        await pilot.pause()
        assert str(detail.styles.display) == "none"
        assert str(tree.styles.display) != "none" and tree.selected is None

        # drilling into a candidate and coming back closes the tree panel
        class _Msg:
            node_id = "c001"

        app.screen.on_tree_plot_widget_node_activated(_Msg())
        await pilot.pause()
        assert app.screen is not app.screen_stack[1]  # candidate screen pushed
        await pilot.press("b")  # `b` is back like esc: closes the candidate screen's own detail
        await pilot.press("b")  # pops back to the searches screen
        await pilot.pause()
        searches_screen = app.screen
        assert searches_screen._tree_open is False
        assert str(tree.styles.display) == "none"

        await pilot.press("t")  # reopen, then toggle off
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        assert str(tree.styles.display) == "none"
        assert str(scrubber.styles.display) == "none"
        assert str(detail.styles.display) == "none"


@pytest.mark.asyncio
async def test_a_opens_the_gantt_panel(tmp_path: Path):
    search_dir, config = make_demo_search(tmp_path, "gantt-run")
    second = make_run_with_search(
        tmp_path / "runs", "gantt-run",
        SearchStatus(search_id="cp-2", run_id="gantt-run", state="done"),
        search_id="cp-2",
    )
    journal2 = Journal(second / "journal.jsonl")
    for i, (cid, kwargs) in enumerate([
        ("c000", dict(operator="baseline", status="passing")),
        ("c777", dict(operator="improve", status="passing", val_score=0.9)),
    ]):
        candidate = make_candidate(cid, **kwargs)
        candidate.created_at = f"2026-08-23T10:0{i}:00+00:00"
        candidate.finished_at = f"2026-08-23T10:0{i + 2}:00+00:00"
        journal2.candidate_result(candidate)

    app = WatchApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.press("a")
        await pilot.pause()
        gantt = app.screen.query_one("#search-gantt")
        assert str(gantt.styles.display) != "none"
        assert app.screen._detail_height == app.screen._fit_detail_height(max_table_rows=8)
        assert app.screen._gantt_fingerprint[0] == "circle-packing"
        await pilot.press("down")  # cursor to the second search: the panel follows
        await pilot.pause()
        assert app.screen._gantt_fingerprint[0] == "cp-2"

        await pilot.press("t")  # the tree takes the shared lower panel over
        await pilot.pause()
        assert app.screen._gantt_open is False
        assert str(gantt.styles.display) == "none"
        assert str(app.screen.query_one("#search-tree").styles.display) != "none"
        await pilot.press("a")  # and the gantt takes it back
        await pilot.pause()
        assert app.screen._tree_open is False
        assert str(gantt.styles.display) != "none"

        await pilot.press("escape")  # closes the panel, focus back on the table
        await pilot.pause()
        assert app.screen._gantt_open is False
        assert str(gantt.styles.display) == "none"
        assert app.screen.focused is app.screen.query_one("#searches")


@pytest.mark.asyncio
async def test_tree_panel_caps_the_searches_table_at_eight_rows(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "cap-run")
    for i in range(11):  # 12 searches in all
        make_run_with_search(
            tmp_path / "runs", "cap-run",
            SearchStatus(search_id=f"cp-{i:02d}", run_id="cap-run", state="done"),
            search_id=f"cp-{i:02d}",
        )
    app = WatchApp(config)
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.press("enter")
        await pilot.press("t")
        await pilot.pause()
        screen = app.screen
        table = screen.query_one("#searches")
        assert table.row_count == 12
        hbar = 1 if table.show_horizontal_scrollbar else 0
        # the tree takes everything past 8 table rows (+ header + chrome)
        assert screen._detail_height == 32 - 4 - (1 + 8 + hbar)


@pytest.mark.asyncio
async def test_hold_column_only_for_holdout_searches(tmp_path: Path):
    from hillclimb.harness.candidate import Candidate

    search_dir, config = make_demo_search(tmp_path, "hold-run")
    app = WatchApp(config)
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.press("enter")  # open the candidate panel for the search
        await pilot.pause()
        panel = app.screen.query_one("#search-candidates")
        labels = [str(c.label) for c in panel.columns.values()]
        assert "hold" not in labels and "val" in labels  # the demo search has no holdout

        # a holdout score on any candidate brings the column back
        Journal(search_dir / "journal.jsonl").candidate_result(
            Candidate(candidate_id="c009", operator="draft", status="passing",
                      trials=[mk_trial(val_score=0.9, holdout_score=0.8)])
        )
        app.screen.refresh_data()
        await pilot.pause()
        labels = [str(c.label) for c in panel.columns.values()]
        assert "hold" in labels

        await pilot.press("o")  # full candidate screen follows the same rule
        await pilot.pause()
        labels = [str(c.label) for c in app.screen.query_one("#candidates").columns.values()]
        assert "hold" in labels


@pytest.mark.asyncio
async def test_o_opens_the_selected_candidate_dir(tmp_path: Path, monkeypatch):
    search_dir, config = make_demo_search(tmp_path, "open-run")
    (search_dir / "candidates" / "c000").mkdir(parents=True, exist_ok=True)
    opened = []
    monkeypatch.setattr("hillclimb.tui.watch.open_in_file_manager", lambda path: opened.append(path))

    app = WatchApp(config)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.press("o")  # -> full candidate screen
        await pilot.pause()
        await pilot.press("o")  # on the candidate screen: reveal the selected candidate's dir
        await pilot.pause()
    assert opened == [search_dir / "candidates" / "c000"]


@pytest.mark.asyncio
async def test_candidate_detail_panel_opens_updates_and_closes(tmp_path: Path):
    search_dir, config = make_demo_search(tmp_path, "detail-run")
    (search_dir / "candidates" / "c000").mkdir(parents=True, exist_ok=True)
    (search_dir / "candidates" / "c001").mkdir(parents=True, exist_ok=True)
    (search_dir / "candidates" / "c000" / "notes.md").write_text("baseline copy\n")
    (search_dir / "candidates" / "c001" / "notes.md").write_text("draft heuristic\n")

    app = WatchApp(config)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.press("o")  # -> full candidate screen
        await pilot.pause()
        detail = app.screen.query_one("#candidate-detail")
        divider = app.screen.query_one("#detail-divider")
        assert str(detail.styles.display) == "none"
        assert str(divider.styles.display) == "none"

        await pilot.press("enter")
        await pilot.pause()
        assert app.screen._detail_candidate_id == "c000"
        assert str(detail.styles.display) != "none"
        assert str(divider.styles.display) != "none"

        await pilot.press("down")
        await pilot.pause()
        assert app.screen._detail_candidate_id == "c001"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen._detail_candidate_id is None
        assert str(detail.styles.display) == "none"
        assert str(divider.styles.display) == "none"
        assert app.screen.__class__.__name__ == "CandidateScreen"


@pytest.mark.asyncio
async def test_candidate_detail_panel_resizes_and_clamps(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "resizable-run")

    app = WatchApp(config)
    async with app.run_test(size=(80, 24)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        screen._set_detail_height(DETAIL_MIN_HEIGHT_FOR_TESTS)  # leave room to grow
        await pilot.pause()
        initial = screen._detail_height

        await pilot.press("+")
        await pilot.pause()
        assert screen._detail_height == initial + DETAIL_STEP

        await pilot.press("-")
        await pilot.pause()
        assert screen._detail_height == initial

        screen._set_detail_height(1)
        assert screen._detail_height == DETAIL_MIN_HEIGHT

        screen._set_detail_height(999)
        assert screen._detail_height == screen._max_detail_height()


@pytest.mark.asyncio
async def test_candidate_detail_divider_drag_resizes(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "drag-run")

    app = WatchApp(config)
    async with app.run_test(size=(80, 24)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        divider = screen.query_one("#detail-divider")
        screen._set_detail_height(DETAIL_MIN_HEIGHT_FOR_TESTS)  # leave room to grow
        await pilot.pause()
        initial = screen._detail_height

        assert await pilot.mouse_down("#detail-divider", offset=(1, 0))
        await pilot.hover(offset=(divider.region.x + 1, divider.region.y - 3))
        await pilot.pause()
        assert screen._detail_height == initial + 3

        await pilot.mouse_up(offset=(divider.region.x + 1, divider.region.y - 3))
        assert not screen._dragging_detail


@pytest.mark.asyncio
async def test_candidate_table_scrollbar_row_drag_resizes(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "table-bottom-drag-run", wide=True)

    app = WatchApp(config)
    async with app.run_test(size=(50, 24)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        table = screen.query_one("#candidates")
        hbar = table.horizontal_scrollbar
        screen._set_detail_height(DETAIL_MIN_HEIGHT_FOR_TESTS)  # leave room to grow
        await pilot.pause()
        initial = screen._detail_height

        assert hbar.__class__.__name__ == "CandidateHorizontalScrollBar"
        assert table.max_scroll_x > 0
        assert await pilot.mouse_down(offset=(hbar.region.x + 1, hbar.region.y))
        await pilot.hover(offset=(hbar.region.x + 1, hbar.region.y - 3))
        await pilot.pause()
        assert screen._detail_height == initial + 3

        await pilot.mouse_up(offset=(hbar.region.x + 1, hbar.region.y - 3))
        assert not screen._dragging_detail


@pytest.mark.asyncio
async def test_candidate_table_scrollbar_horizontal_drag_scrolls(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "table-horizontal-scroll-run", wide=True)

    app = WatchApp(config)
    async with app.run_test(size=(50, 24)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        table = screen.query_one("#candidates")
        hbar = table.horizontal_scrollbar
        initial_height = screen._detail_height
        initial_scroll = table.scroll_x

        assert hbar.__class__.__name__ == "CandidateHorizontalScrollBar"
        assert table.max_scroll_x > 0
        assert await pilot.mouse_down(offset=(hbar.region.x + 1, hbar.region.y))
        await pilot.hover(offset=(hbar.region.x + 12, hbar.region.y))
        await pilot.pause()

        assert table.scroll_x > initial_scroll
        assert screen._detail_height == initial_height

        await pilot.mouse_up(offset=(hbar.region.x + 12, hbar.region.y))


@pytest.mark.asyncio
async def test_candidate_table_scrollbar_switches_scroll_then_resize_in_one_drag(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "scroll-then-resize-run", wide=True)

    app = WatchApp(config)
    async with app.run_test(size=(50, 24)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        table = screen.query_one("#candidates")
        hbar = table.horizontal_scrollbar
        initial_height = screen._detail_height
        initial_scroll = table.scroll_x

        assert await pilot.mouse_down(offset=(hbar.region.x + 1, hbar.region.y))
        await pilot.hover(offset=(hbar.region.x + 12, hbar.region.y))
        await pilot.pause()
        assert table.scroll_x > initial_scroll
        assert screen._detail_height == initial_height

        after_scroll = table.scroll_x
        await pilot.hover(offset=(hbar.region.x + 12, hbar.region.y - 3))
        await pilot.pause()
        assert table.scroll_x == after_scroll
        assert screen._detail_height == initial_height + 3

        await pilot.mouse_up(offset=(hbar.region.x + 12, hbar.region.y - 3))


@pytest.mark.asyncio
async def test_candidate_table_scrollbar_switches_resize_then_scroll_in_one_drag(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "resize-then-scroll-run", wide=True)

    app = WatchApp(config)
    async with app.run_test(size=(50, 24)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        table = screen.query_one("#candidates")
        hbar = table.horizontal_scrollbar
        initial_height = screen._detail_height
        initial_scroll = table.scroll_x

        assert await pilot.mouse_down(offset=(hbar.region.x + 1, hbar.region.y))
        await pilot.hover(offset=(hbar.region.x + 1, hbar.region.y - 3))
        await pilot.pause()
        assert screen._detail_height == initial_height + 3
        assert table.scroll_x == initial_scroll

        after_resize = screen._detail_height
        await pilot.hover(offset=(hbar.region.x + 12, hbar.region.y - 3))
        await pilot.pause()
        assert screen._detail_height == after_resize
        assert table.scroll_x > initial_scroll

        await pilot.mouse_up(offset=(hbar.region.x + 12, hbar.region.y - 3))


def test_search_duration_counts_up_with_the_budget_alongside():
    from hillclimb.tui.watch import _format_budget_total, _format_duration

    assert _format_budget_total(600) == "10m"
    assert _format_budget_total(5400) == "1h 30m"
    assert _format_budget_total(90) == "1m 30s"
    assert _format_duration(247.9, 600) == "4m 07s (budget: 10m)"
    assert _format_duration(600, 600) == "10m 00s (budget: 10m)"
    # a graceful deadline lets the last operator finish: the tail is shown, not clipped
    assert _format_duration(856, 600) == "14m 16s (budget: 10m, 4m 16s over)"
    assert _format_duration(3856, 3600) == "1h 04m 16s (budget: 1h, 4m 16s over)"
    assert _format_duration(12, 0) == "0m 12s"  # no budget declared
    assert _format_duration(None, 600) == "-"


def test_budget_left_shows_seconds():
    from hillclimb.tui.watch import _format_budget_left

    assert _format_budget_left(None) == "-"
    assert _format_budget_left(247.9) == "4m 07s"
    assert _format_budget_left(3727) == "1h 02m 07s"
    assert _format_budget_left(-5) == "0m 00s"


def test_live_remaining_counts_down_between_heartbeats():
    from datetime import datetime, timedelta, timezone

    from hillclimb.harness.status import BudgetStatus, SearchStatus, live_remaining_s

    written = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    status = SearchStatus(search_id="p", state="running", budget=BudgetStatus(remaining_s=100), updated_at=written)
    assert 89 <= live_remaining_s(status, "running") <= 91
    # a finished search's clock stopped at the last write
    assert live_remaining_s(status, "done") == 100
    status.budget.remaining_s = 3
    assert live_remaining_s(status, "running") == 0.0


def test_live_spent_s_keeps_counting_past_the_budget():
    from datetime import datetime, timedelta, timezone

    from hillclimb.harness.status import BudgetStatus, SearchStatus, live_spent_s

    written = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    # over budget and still running: remaining is floored at 0 but spent is not
    status = SearchStatus(
        search_id="p", state="running",
        budget=BudgetStatus(total_s=100, spent_s=104, remaining_s=0), updated_at=written,
    )
    assert 113 <= live_spent_s(status, "running") <= 115
    assert live_spent_s(status, "done") == 104
    # a record from before spent_s was written: total minus remaining
    status.budget.spent_s = 0
    status.budget.remaining_s = 40
    assert live_spent_s(status, "done") == 60


@pytest.mark.asyncio
async def test_searches_screen_inline_candidates_panel(tmp_path: Path):
    from hillclimb.tui.watch import DETAIL_STEP

    _, config = make_demo_search(tmp_path, "panel-run")

    app = WatchApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.pause()
        screen = app.screen
        panel = screen.query_one("#search-candidates")
        divider = screen.query_one("#detail-divider")
        assert str(panel.styles.display) == "none" and str(divider.styles.display) == "none"

        await pilot.press("enter")  # open the panel for the highlighted search
        await pilot.pause()
        assert app.screen is screen  # same screen, no push
        assert screen._panel_search_id == "circle-packing"
        assert str(panel.styles.display) == "block"
        assert panel.row_count == 3
        # c001 is the search's current selection: it wears the star
        assert [str(panel.get_cell_at((i, 0))).strip() for i in range(3)] == ["c000", "c001 ★", "└─ c002"]

        screen._set_detail_height(DETAIL_MIN_HEIGHT_FOR_TESTS)  # leave room to grow
        await pilot.pause()
        initial = screen._detail_height
        await pilot.press("+")
        assert screen._detail_height == initial + DETAIL_STEP
        await pilot.press("-")
        assert screen._detail_height == initial

        assert await pilot.mouse_down("#detail-divider", offset=(1, 0))
        await pilot.hover(offset=(divider.region.x + 1, divider.region.y - 3))
        await pilot.pause()
        assert screen._detail_height == initial + 3
        await pilot.mouse_up(offset=(divider.region.x + 1, divider.region.y - 3))
        assert not screen._dragging_detail

        await pilot.press("escape")  # closes the panel, stays on the screen
        await pilot.pause()
        assert app.screen is screen and screen._panel_search_id is None
        assert str(panel.styles.display) == "none"
        assert app.focused.id == "searches"

        await pilot.press("enter")  # opens the panel with the cursor in it
        await pilot.pause()
        assert app.focused.id == "search-candidates"
        await pilot.press("down")
        await pilot.press("enter")  # enter on a candidate: full candidates + its details
        await pilot.pause()
        assert app.screen is not screen
        full = app.screen
        assert full.query_one("#candidates").row_count == 3
        assert full._detail_candidate_id == "c001"
        assert full.query_one("#candidates").cursor_row == 1
        assert str(full.query_one("#candidate-detail").styles.display) == "block"


@pytest.mark.asyncio
async def test_searches_panel_cursor_survives_refresh_with_several_searches(tmp_path: Path):
    import asyncio

    runs_dir = tmp_path / "runs"
    make_run_with_search(runs_dir, "multi", search_id="a")  # 3 candidates
    big = make_run_with_search(runs_dir, "multi", search_id="b")
    journal = Journal(big / "journal.jsonl")
    for i in range(3, 8):
        journal.candidate_result(make_candidate(f"c00{i}", operator="improve", parent_id="c001", status="buggy"))
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.press("down")  # highlight "b"
        await pilot.press("enter")  # open its panel
        await pilot.pause()
        screen = app.screen
        assert screen._panel_search_id == "b"
        panel = screen.query_one("#search-candidates")
        screen._set_detail_height(20)
        await pilot.pause()
        rows = [str(panel.get_cell_at((i, 0))).strip("│├└─ ") for i in range(panel.row_count)]
        target = rows.index("c007")  # beyond search a's row count
        await pilot.click(offset=(panel.region.x + 3, panel.region.y + 1 + target))
        await pilot.pause()
        assert panel.cursor_row == target
        await asyncio.sleep(1.2)  # a live refresh tick
        await pilot.pause()
        assert screen._panel_search_id == "b"
        assert panel.cursor_row == target


@pytest.mark.asyncio
async def test_m_maximizes_and_restores_the_panel(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "max-run")

    app = WatchApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("enter")
        await pilot.press("m")  # nothing open: no-op
        await pilot.pause()
        screen = app.screen
        assert not screen._detail_maximized
        await pilot.press("enter")  # open panel
        await pilot.pause()
        height = screen._detail_height
        table, panel = screen.query_one("#searches"), screen.query_one("#search-candidates")
        await pilot.press("m")
        await pilot.pause()
        assert screen._detail_maximized
        assert str(table.styles.display) == "none" and str(panel.styles.height) == "1fr"
        await pilot.press("m")
        await pilot.pause()
        assert not screen._detail_maximized
        assert str(table.styles.display) == "block" and panel.styles.height.value == height
        await pilot.press("m")
        await pilot.press("escape")  # closing the panel also un-maximizes
        await pilot.pause()
        assert not screen._detail_maximized and str(table.styles.display) == "block"

        await pilot.press("o")  # the candidate screen has the same key for its detail
        await pilot.press("enter")
        await pilot.press("m")
        await pilot.pause()
        cs = app.screen
        assert cs._detail_maximized and str(cs.query_one("#candidates").styles.display) == "none"


@pytest.mark.asyncio
async def test_detail_renders_full_width_on_first_paint(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "width-run")

    app = WatchApp(config)
    async with app.run_test(size=(120, 30)) as pilot:
        await open_candidate_detail(pilot)
        detail = app.screen.query_one("#candidate-detail")
        # the first write happened before the log had a layout width; the
        # rendered lines must still span the screen, not a default 80 columns
        widths = {len(strip.text.rstrip()) for strip in detail.lines if strip.text.strip()}
        assert max(widths) >= 110, widths


@pytest.mark.asyncio
async def test_detail_is_not_rewritten_when_nothing_changed(tmp_path: Path):
    import asyncio

    from textual.widgets import RichLog

    search_dir, config = make_demo_search(tmp_path, "flicker-run")
    clears = []
    original_clear = RichLog.clear

    def counting_clear(self):
        clears.append(self.id)
        return original_clear(self)

    RichLog.clear = counting_clear
    try:
        app = WatchApp(config)
        async with app.run_test(size=(120, 30)) as pilot:
            await open_candidate_detail(pilot)
            detail = app.screen.query_one("#candidate-detail")
            detail.scroll_to(y=0, animate=False)
            baseline = len(clears)
            await asyncio.sleep(2.3)  # two live ticks with identical content
            await pilot.pause()
            assert len(clears) == baseline, "detail was cleared and rewritten on a no-change tick"

            # a change in the journal is still picked up
            journal = Journal(search_dir / "journal.jsonl")
            c000 = journal.candidates["c000"]
            c000.summary = "baseline, now with a new summary"
            journal.candidate_result(c000)
            await asyncio.sleep(1.2)
            await pilot.pause()
            assert len(clears) == baseline + 1
            assert any("new summary" in strip.text for strip in detail.lines)
    finally:
        RichLog.clear = original_clear


@pytest.mark.asyncio
async def test_detail_opens_fitted_under_the_whole_table_and_maximize_hides_divider(tmp_path: Path):
    from hillclimb.tui.watch import DETAIL_CHROME_ROWS

    _, config = make_demo_search(tmp_path, "fit-run")

    app = WatchApp(config)
    async with app.run_test(size=(120, 40)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        table = screen.query_one("#candidates")
        # 3 rows + header; the detail takes everything else
        expected = 40 - DETAIL_CHROME_ROWS - (1 + 3 + (1 if table.show_horizontal_scrollbar else 0))
        assert screen._detail_height == expected
        assert table.region.height >= 1 + 3  # header and every row visible

        divider = screen.query_one("#detail-divider")
        await pilot.press("m")
        await pilot.pause()
        assert str(divider.styles.display) == "none"
        await asyncio_tick(pilot)  # a live refresh re-renders; divider must stay hidden
        assert str(divider.styles.display) == "none"
        await pilot.press("m")
        await pilot.pause()
        assert str(divider.styles.display) == "block"


async def asyncio_tick(pilot) -> None:
    import asyncio

    await asyncio.sleep(1.2)
    await pilot.pause()


@pytest.mark.asyncio
async def test_header_clock_names_its_zone_and_choice_persists(tmp_path: Path, monkeypatch):
    import re
    from zoneinfo import ZoneInfo

    from textual.widgets import Input

    from hillclimb.tui.header import HillclimbClock, TimezoneChoiceScreen, load_display_timezone

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _, config = make_demo_search(tmp_path, "tz-run")

    app = WatchApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        assert app.title == "hillclimb"
        clock = app.screen.query_one(HillclimbClock)
        assert re.fullmatch(r"\d\d:\d\d:\d\d \S+", str(clock.render()))
        # flush with the right edge: no padding past the text
        assert clock.region.x + clock.region.width == app.size.width
        await pilot.press("t")
        await pilot.pause()
        assert isinstance(app.screen, TimezoneChoiceScreen)
        app.screen.query_one("#timezone-filter", Input).value = "tokyo"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.display_timezone == ZoneInfo("Asia/Tokyo")
        assert str(clock.render()).endswith("JST")
    assert load_display_timezone() == ZoneInfo("Asia/Tokyo")


@pytest.mark.asyncio
async def test_searches_runline_says_run(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "runline-run")
    app = WatchApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert str(app.screen.query_one("#runline").content).startswith("run: Demo")


@pytest.mark.asyncio
async def test_ctrl_c_quits_and_question_mark_lists_every_key(tmp_path: Path):
    from hillclimb.tui.keys import KeysPanel

    _, config = make_demo_search(tmp_path, "keys-run")
    app = WatchApp(config)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("enter")  # searches
        await pilot.pause()
        screen = app.screen
        footer_keys = [b.key for b in screen.BINDINGS if b.show]
        assert footer_keys == ["enter", "escape,b", "question_mark", "q"]  # lean footer; esc/b is one way back
        assert not screen.query(KeysPanel)
        await pilot.press("question_mark")
        await pilot.pause()
        rows = dict(screen.query_one(KeysPanel).rows())
        assert rows["o"] == "full candidate view" and rows["m"] == "maximize panel"
        assert rows["g"] == "knowledge graph" and rows["esc/b"] == "back"
        assert rows["drag divider"] == "resize panel"
        assert rows["t"] == "tree panel"  # the screen key shadows the app-level time zone here
        assert screen.query_one("#searches").size.width < 120  # split, not overlay
        await pilot.press("question_mark")
        await pilot.pause()
        assert not screen.query(KeysPanel)

        await pilot.press("ctrl+c")
        await pilot.pause()
    assert app.return_code is not None or not app.is_running


# --- candidate paths recorded elsewhere (mirrors of hosted runs) ---


def _running_status(search_id: str, run_id: str, current=()) -> SearchStatus:
    return SearchStatus(search_id=search_id, run_id=run_id, state="running", pid=os.getpid(), current=list(current))


def _stream_message(text: str, tokens: int = 1000) -> str:
    return json.dumps({
        "type": "assistant",
        "message": {"id": f"m{tokens}", "usage": {"input_tokens": tokens, "output_tokens": 0},
                    "content": [{"type": "text", "text": text}]},
    }) + "\n"


def test_candidate_paths_fall_back_to_the_search_dir_when_recorded_elsewhere(tmp_path: Path):
    """A mirror of a hosted run keeps the layout but not the container's
    absolute paths: the detail, the token counts and the model all read the
    candidate dir under the search dir instead."""
    from hillclimb.harness.status import CurrentCandidate
    from hillclimb.tui.watch import _search_row, resolve_candidate_dir

    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(runs_dir, "r")
    foreign = "/hillclimb/hillclimb/runs/r/searches/circle-packing/candidates/c003"
    local = search_dir / "candidates" / "c003"
    local.mkdir()
    (local / "agent_stream.jsonl").write_text(_stream_message("reading the contract", tokens=500_000))
    (local / "exec_stdout.log").write_text("epoch 1 done\n")
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_created(
        make_candidate("c003", operator="draft", status="pending", candidate_dir=foreign,
                       agent=AgentInfo(name="claude-code", model="sonnet"))
    )
    write_status(search_dir, _running_status("circle-packing", "r", [
        CurrentCandidate(candidate_id="c003", operator="draft", phase="agent", candidate_dir=foreign)]))

    assert resolve_candidate_dir(search_dir, "c003", foreign) == local
    assert resolve_candidate_dir(search_dir, "c003", str(local)) == local  # an existing path is kept
    assert resolve_candidate_dir(search_dir, "c003", None) == local

    store = FileDataStore(search_dir.parents[2])
    row = _search_row(store, store.search(key_for(search_dir)))
    assert row.tokens == "1.74M"  # 240k + 1.0M journaled + 500k from the in-flight stream

    from rich.console import Console

    console = Console(record=True, width=120)
    for renderable in candidate_detail_renderables(_record(search_dir), Journal(search_dir / "journal.jsonl"), "c003"):
        console.print(renderable)
    rendered = console.export_text()
    assert "500.0k so far" in rendered
    assert "reading the contract" in rendered
    assert "epoch 1 done" in rendered
    assert "candidates/c003" in rendered


def test_pending_candidate_detail_is_in_flight_only_while_the_search_lives(tmp_path: Path):
    """The journal keeps a candidate `pending` until its result lands, so
    that is the status the in-flight rows (elapsed, tokens so far) key on."""
    from rich.console import Console

    from hillclimb.harness.candidate import utcnow
    from hillclimb.tui.watch import candidate_in_flight

    search_dir = make_run_with_search(tmp_path / "runs", "r")
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_created(
        make_candidate("c003", operator="draft", status="pending", created_at=utcnow(),
                       agent=AgentInfo(name="claude-code", model="sonnet"))
    )
    journal = Journal(search_dir / "journal.jsonl")
    assert candidate_in_flight(journal.candidates["c003"], live=True)
    assert not candidate_in_flight(journal.candidates["c003"], live=False)
    assert not candidate_in_flight(journal.candidates["c001"], live=True)

    def rendered(live: bool, console: bool = False) -> str:
        out = Console(record=True, width=120)
        for r in candidate_detail_renderables(_record(search_dir), journal, "c003", live=live, console=console):
            out.print(r)
        return out.export_text()

    assert "elapsed" in rendered(live=True)
    assert "so far" in rendered(live=True)
    assert "elapsed" not in rendered(live=False)


# --- the live console ---


def test_fresh_lines_grown_shifted_and_replaced():
    from hillclimb.tui.watch import fresh_lines

    seen = ["a", "b", "c", "d"]
    assert fresh_lines([], ["a", "b"]) == (["a", "b"], False)
    assert fresh_lines(seen, seen) == ([], False)
    assert fresh_lines(seen, seen + ["e", "f"]) == (["e", "f"], False)
    # a mirrored tail dropped lines off its head
    assert fresh_lines(seen, ["c", "d", "e"]) == (["e"], False)
    # the file was replaced by something unrelated
    assert fresh_lines(seen, ["x", "y", "z"]) == (["x", "y", "z"], True)
    # a repeated tail: the continuation right after the seen lines wins over
    # a later repeat
    seen = ["tick", "tick", "tick"]
    assert fresh_lines(seen, ["tick", "tick", "tick", "tick", "tick"]) == (["tick", "tick"], False)


def test_complete_lines_waits_for_the_newline(tmp_path: Path):
    from hillclimb.tui.watch import complete_lines

    path = tmp_path / "exec_stdout.log"
    assert complete_lines(path) is None
    path.write_text("")
    assert complete_lines(path) == []
    path.write_text("one\ntwo\nthr")
    assert complete_lines(path) == ["one", "two"]
    path.write_text("one\ntwo\nthree\n")
    assert complete_lines(path) == ["one", "two", "three"]


def _console_text(console) -> str:
    return "\n".join(strip.text for strip in console.lines)


@pytest.mark.asyncio
async def test_running_candidate_detail_has_a_following_console(tmp_path: Path):
    from hillclimb.harness.candidate import utcnow
    from hillclimb.harness.status import CurrentCandidate
    from hillclimb.tui.watch import ConsoleLog

    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(runs_dir, "r")
    live = search_dir / "candidates" / "c003"
    live.mkdir()
    stream = live / "agent_stream.jsonl"
    stream.write_text(_stream_message("reading the contract"))
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_created(
        make_candidate("c003", operator="draft", status="pending", created_at=utcnow(),
                       candidate_dir=str(live), agent=AgentInfo(name="claude-code", model="sonnet"))
    )
    current = CurrentCandidate(candidate_id="c003", operator="draft", phase="agent", candidate_dir=str(live))
    write_status(search_dir, _running_status("circle-packing", "r", [current]))
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.press("enter")
        await pilot.press("o")
        await pilot.pause()
        screen = app.screen
        table = screen.query_one("#candidates")
        table.move_cursor(row=table.get_row_index("c003"))
        await pilot.press("enter")
        await pilot.pause()
        console = screen.query_one("#candidate-console", ConsoleLog)
        detail = screen.query_one("#candidate-detail")
        assert screen._detail_candidate_id == "c003"
        assert str(console.styles.display) == "block"
        assert detail.has_class("with-console")
        assert console.candidate_id == "c003"
        assert console.border_title == "console · agent"
        shown = _console_text(console)
        assert "── agent ──" in shown and "reading the contract" in shown
        assert "epoch" not in shown

        # the operator streams on, the verifier starts: lines are appended, not rewritten
        with stream.open("a") as f:
            f.write(_stream_message("writing solution.py", tokens=2000))
            f.write('{"type": "assistant", "message": {"id": "half"')  # mid-write, no newline yet
        (live / "exec_stdout.log").write_text("epoch 1 done\nepoch 2 done\n")
        current.phase = "exec"
        write_status(search_dir, _running_status("circle-packing", "r", [current]))
        screen.refresh_data()
        await pilot.pause()
        shown = _console_text(console)
        assert shown.count("reading the contract") == 1
        assert "writing solution.py" in shown
        assert "half" not in shown
        assert "── exec stdout ──" in shown and "epoch 2 done" in shown
        assert console.border_title == "console · exec"
        assert console.follow

        # scrolling up pauses following; `f` resumes it
        console.follow = False
        console._update_title()
        assert "paused" in console.border_title
        await pilot.press("f")
        await pilot.pause()
        assert console.follow and "paused" not in console.border_title

        # the result lands: the console goes away and the overview's own tails take over
        done = journal.candidates["c003"]
        done.status = "passing"
        Journal(search_dir / "journal.jsonl").candidate_result(done)
        screen.refresh_data()
        await pilot.pause()
        assert str(console.styles.display) == "none"
        assert not detail.has_class("with-console")
        assert console.candidate_id is None

        # closing the detail hides the pane and the overview with it
        await pilot.press("escape")
        await pilot.pause()
        assert str(detail.styles.display) == "none"
