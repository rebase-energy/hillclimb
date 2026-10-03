"""Policy conformance check: the cheap pre-verifier for an edited
exploration process.

An `OperatorPolicy` that violates its contract (base.py) does not fail
loudly — it stalls a search, proposes a target that does not exist, or
makes `resume` diverge from the run it resumes. Each of those burns a
real budget hour before anyone notices. This module replays recorded
journals (and a synthetic empty one) through the policy WITHOUT any
coding agent, verifier or venv, and reports every contract breach it can see:

- `constructs`        the factory builds the policy (twice — no shared state)
- `starts`            an empty journal with empty slots yields an action, not
                      a hold: a hold with nothing in flight is a stall
- `replay`            two fresh instances fed the same journal propose the
                      same action at every budget point (resume contract)
- `idempotent`        asking one instance twice gives the same answer
                      (`propose` must not consume state)
- `resume`            an instance that saw the journal GROW, result by result
                      as a live search shows it, proposes what one that was
                      shown the finished journal at once proposes — what a
                      resumed search does
- `references`        every target / inspiration id exists, the operator is
                      one this search may run (the climber's own operators
                      included), and operators that need a target carry one
- `read-only`         the journal's candidates and the search dir's files are
                      byte-identical after the policy ran
- `templates`         the prompt override dir lints clean (render.py)

Pure by construction: nothing here writes. `check_policy` is the library
entry; `hillclimb climber check` wraps it over the store's journals and can
follow up with a dummy-coding-agent smoke search.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from hillclimb.harness.evaluation import accept_band
from hillclimb.modules.operators import get_operator, operator_names
from hillclimb.config import Config
from hillclimb.harness.journal import Journal
from hillclimb.harness.loop import PolicyLoop
from hillclimb.modules.policies.base import INJECT_ACTION, TUNE_ACTION, Action, BudgetView, SearchState, OperatorPolicy

# Fractions of the budget still remaining at which every journal is
# probed: fresh, mid-search, and inside the ensemble window.
BUDGET_POINTS: tuple[float, ...] = (1.0, 0.5, 0.05)
# the `resume` check shows a policy the journal at most this many times as it
# grows (each step costs a holdout-blind copy of the prefix)
RESUME_STEPS = 60
# what the harness runs itself: no operator class, no prompt
HARNESS_ACTIONS = frozenset({TUNE_ACTION, INJECT_ACTION})


class _BuiltinOperators:
    """The built-in catalogue, for a check that was handed no operator set."""

    def names(self) -> tuple[str, ...]:
        return tuple(operator_names())

    def get(self, name: str):
        return get_operator(name)


def known_operators(operators=None) -> frozenset[str]:
    """What this search can run: its operators (the climber's `OperatorSet`;
    the built-in catalogue when none is given), plus the harness's own
    `tune` and `inject`."""
    return frozenset((operators or _BuiltinOperators()).names()) | HARNESS_ACTIONS


@dataclass(frozen=True)
class Finding:
    check: str
    ok: bool
    detail: str
    journal: str = ""  # which replayed journal (empty for policy-wide checks)

    def render(self) -> str:
        where = f" [{self.journal}]" if self.journal else ""
        return f"{'ok  ' if self.ok else 'FAIL'} {self.check}{where}: {self.detail}"


@dataclass
class CheckReport:
    policy: str
    params: dict
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(f.ok for f in self.findings)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if not f.ok]

    def render(self) -> str:
        head = f"policy {self.policy} {self.params or '{}'}: " + (
            "conforms" if self.ok else f"{len(self.failures)} contract breach(es)"
        )
        return "\n".join([head, *(f.render() for f in self.findings)])

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "params": self.params,
            "ok": self.ok,
            "findings": [
                {"check": f.check, "ok": f.ok, "detail": f.detail, "journal": f.journal}
                for f in self.findings
            ],
        }


@dataclass(frozen=True)
class JournalCase:
    """One recorded journal to replay: its label (a search ref), the
    journal, the problem's direction and the search's budget."""

    label: str
    journal: Journal
    higher_is_better: bool = True
    total_s: int = 3600
    search_dir: Path | None = None


def _view(case: JournalCase, config: Config, fraction: float) -> SearchState:
    return SearchState(
        journal=case.journal,
        inflight=(),
        budget=BudgetView(
            remaining_s=case.total_s * fraction,
            total_s=case.total_s,
            stop_margin_s=config.budget.stop_margin_s,
        ),
        higher_is_better=case.higher_is_better,
        accept_band=accept_band(config, case.journal),
    )


def _replayed(make_policy: Callable[[], Policy], case: JournalCase, config: Config) -> PolicyLoop:
    """A fresh climber (its policy with its selector, in the loop that asks
    them in order) that has observed the journal — exactly what
    `PolicyLoop.catch_up` does when a search starts."""
    loop = PolicyLoop(make_policy())
    view = _view(case, config, 1.0)
    for candidate in view.journal.candidates.values():
        loop.policy.observe(view, candidate)
    return loop


def _prefix(journal: Journal, n: int) -> Journal:
    """The journal as it stood when its n-th candidate had landed."""
    prefix = Journal.__new__(Journal)
    prefix.backend = _NullBackend()
    prefix.lock = threading.RLock()
    prefix.candidates = dict(list(journal.candidates.items())[:n])
    return prefix


def _grown(make_policy: Callable[[], Policy], case: JournalCase, config: Config) -> PolicyLoop:
    """A fresh climber that watched the journal grow: shown each new result
    through a view of the journal as it stood then — what `PolicyLoop.observe`
    does during a live search. (Long journals grow in `RESUME_STEPS` strides.)"""
    loop = PolicyLoop(make_policy())
    policy = loop.policy
    ids = list(case.journal.candidates)
    stride = max(1, -(-len(ids) // RESUME_STEPS))
    shown = 0
    while shown < len(ids):
        upto = min(len(ids), shown + stride)
        grown = JournalCase(case.label, _prefix(case.journal, upto), case.higher_is_better, case.total_s)
        view = _view(grown, config, 1.0)
        for candidate_id in ids[shown:upto]:
            policy.observe(view, view.journal.candidates[candidate_id])
        shown = upto
    return loop


def _snapshot_journal(journal: Journal) -> list[tuple[str, str]]:
    return [(cid, c.model_dump_json()) for cid, c in journal.candidates.items()]


def _snapshot_tree(root: Path | None) -> dict[str, tuple[int, int]]:
    if root is None or not root.exists():
        return {}
    out: dict[str, tuple[int, int]] = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = Path(dirpath) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            out[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return out


def _describe(action: Action | None) -> str:
    if action is None:
        return "hold"
    parts = [action.operator]
    if action.target_id:
        parts.append(f"-> {action.target_id}")
    if action.inspiration_ids:
        parts.append(f"+{list(action.inspiration_ids)}")
    if action.args:
        parts.append("(" + ", ".join(str(v) for v in action.args.values()) + ")")
    return " ".join(parts)


def _reference_problems(action: Action, journal: Journal, operators=None) -> list[str]:
    problems: list[str] = []
    operators = operators or _BuiltinOperators()
    known = known_operators(operators)
    if action.operator not in known:
        problems.append(f"unknown operator {action.operator!r} (harness runs {sorted(known)})")
    for cid in (action.target_id, *action.inspiration_ids):
        if cid and cid not in journal.candidates:
            problems.append(f"references {cid} which is not in the journal")
    target = journal.candidates.get(action.target_id) if action.target_id else None
    if action.target_id and target is None:
        return problems  # already reported as dangling
    if action.operator == TUNE_ACTION:
        if target is None:
            problems.append("tune without a target_id")
        else:
            if not target.is_scored:
                problems.append(f"tune targets unscored {target.candidate_id}")
            if not target.tunable:
                problems.append(f"tune targets {target.candidate_id} which declares no params.json")
    elif action.operator in known and action.operator not in HARNESS_ACTIONS:
        # the operator's own rule — the same one the harness applies before
        # it creates or spends anything
        reason = operators.get(action.operator).valid_target(target)
        if reason:
            problems.append(reason)
    return problems


def check_policy(
    make_policy: Callable[[], OperatorPolicy],
    cases: Sequence[JournalCase],
    config: Config,
    *,
    budget_points: Sequence[float] = BUDGET_POINTS,
    prompts_dir: Path | None = None,
    operators=None,
) -> CheckReport:
    """Run every conformance check over `cases` (plus a synthetic empty
    journal). `make_policy` must return a NEW instance each call — the
    replay check depends on it. `operators` is the search's operator set
    (`Climber.operator_set()`), so a climber's own operators are known;
    without one the built-in catalogue stands in. Never writes; never runs
    a coding agent."""
    from hillclimb.prompts.render import lint_overrides

    findings: list[Finding] = []
    try:
        first, second = make_policy(), make_policy()
    except Exception as exc:  # noqa: BLE001
        report = CheckReport(policy="?", params={})
        report.findings.append(Finding("constructs", False, f"{type(exc).__name__}: {exc}"))
        return report
    report = CheckReport(policy=getattr(first, "name", type(first).__name__), params=dict(getattr(first, "params", {}) or {}))
    findings.append(
        Finding(
            "constructs",
            first is not second,
            "fresh instance per call" if first is not second else "factory returns the same object — state leaks across resumes",
        )
    )

    empty = JournalCase("empty", Journal(_NullBackend()), True, 3600)
    all_cases = [empty, *cases]

    # starts: the very first decision of a search must be an action
    try:
        action = _replayed(make_policy, empty, config).propose(_view(empty, config, 1.0))
        findings.append(
            Finding(
                "starts",
                action is not None,
                _describe(action) if action is not None else "holds on an empty journal with empty slots — the search would never start",
            )
        )
    except Exception as exc:  # noqa: BLE001
        findings.append(Finding("starts", False, f"propose raised {type(exc).__name__}: {exc}"))

    for case in all_cases:
        before_journal = _snapshot_journal(case.journal)
        before_tree = _snapshot_tree(case.search_dir)
        try:
            a = _replayed(make_policy, case, config)
            b = _replayed(make_policy, case, config)
            live = _grown(make_policy, case, config)
        except Exception as exc:  # noqa: BLE001
            findings.append(Finding("replay", False, f"observe raised {type(exc).__name__}: {exc}", case.label))
            continue
        replay_ok, idem_ok, resume_ok, view_mutated = True, True, True, False
        proposals: list[str] = []
        for fraction in budget_points:
            try:
                view = _view(case, config, fraction)
                handed = _snapshot_journal(view.journal)
                x = a.propose(view)
                # a policy only ever holds its holdout-blind view, so that is
                # where a write lands — the engine's journal is out of reach
                view_mutated = view_mutated or _snapshot_journal(view.journal) != handed
                y = b.propose(_view(case, config, fraction))
                x_again = a.propose(_view(case, config, fraction))
                z = live.propose(_view(case, config, fraction))
            except Exception as exc:  # noqa: BLE001
                findings.append(
                    Finding("replay", False, f"propose raised at {fraction:.0%} budget: {type(exc).__name__}: {exc}", case.label)
                )
                replay_ok = False
                break
            proposals.append(f"{fraction:.0%}: {_describe(x)}")
            if x != y:
                replay_ok = False
                findings.append(
                    Finding(
                        "replay", False,
                        f"two fresh replays disagree at {fraction:.0%} budget: {_describe(x)} vs {_describe(y)}",
                        case.label,
                    )
                )
            if x != x_again:
                idem_ok = False
                findings.append(
                    Finding(
                        "idempotent", False,
                        f"asking twice at {fraction:.0%} budget changed the answer: {_describe(x)} then {_describe(x_again)}",
                        case.label,
                    )
                )
            if x == y and x != z:
                resume_ok = False
                findings.append(
                    Finding(
                        "resume", False,
                        f"a resumed search would diverge at {fraction:.0%} budget: the policy that watched "
                        f"the journal grow proposes {_describe(z)}, one shown the finished journal {_describe(x)}",
                        case.label,
                    )
                )
            if x is not None:
                for problem in _reference_problems(x, case.journal, operators):
                    findings.append(Finding("references", False, f"at {fraction:.0%} budget: {problem}", case.label))
        if replay_ok:
            findings.append(Finding("replay", True, "; ".join(proposals), case.label))
        if idem_ok and replay_ok:
            findings.append(Finding("idempotent", True, "propose is stable", case.label))
        if resume_ok and replay_ok:
            findings.append(Finding("resume", True, "live and resumed state agree", case.label))
        if not any(f.check == "references" and f.journal == case.label for f in findings):
            findings.append(Finding("references", True, "every id resolves", case.label))
        journal_same = _snapshot_journal(case.journal) == before_journal and not view_mutated
        tree_same = _snapshot_tree(case.search_dir) == before_tree
        findings.append(
            Finding(
                "read-only",
                journal_same and tree_same,
                "journal and files untouched" if journal_same and tree_same
                else ("journal candidates were mutated" if not journal_same else "files under the search dir changed"),
                case.label,
            )
        )

    problems = lint_overrides(prompts_dir)
    findings.append(
        Finding(
            "templates",
            not problems,
            "; ".join(problems) if problems else (
                f"overrides in {prompts_dir} lint clean" if prompts_dir is not None and prompts_dir.is_dir()
                else "no prompt overrides"
            ),
        )
    )
    report.findings = findings
    return report


class _NullBackend:
    """An empty, write-rejecting journal backend for the synthetic case."""

    path = None

    def records(self):
        return iter(())

    def append(self, record: dict) -> None:
        raise RuntimeError("climber check journals are read-only")
