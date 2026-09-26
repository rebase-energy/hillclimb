"""Policy conformance check: the cheap pre-verifier for an edited
exploration process.

A `SearchPolicy` that violates its contract (policy.py) does not fail
loudly — it stalls a search, proposes a target that does not exist, or
makes `resume` diverge from the run it resumes. Each of those burns a
real budget hour before anyone notices. This module replays recorded
journals (and a synthetic empty one) through the policy WITHOUT any
agent, verifier or venv, and reports every contract breach it can see:

- `constructs`        the factory builds the policy (twice — no shared state)
- `starts`            an empty journal with empty slots yields an action, not
                      a hold: a hold with nothing in flight is a stall
- `replay`            two fresh instances fed the same journal propose the
                      same action at every budget point (resume contract)
- `idempotent`        asking one instance twice gives the same answer
                      (`propose` must not consume state)
- `references`        every target / inspiration id exists, the operator is
                      one the harness runs, and operators that need a target
                      carry one
- `read-only`         the journal's candidates and the search dir's files are
                      byte-identical after the policy ran
- `templates`         the prompt override dir lints clean (render.py)

Pure by construction: nothing here writes. `check_policy` is the library
entry; `hillclimb climber check` wraps it over the store's journals and can
follow up with a dummy-backend smoke search.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from hillclimb.evaluation import accept_band
from hillclimb.operators import get_operator, operator_names
from hillclimb.config import Config
from hillclimb.journal import Journal
from hillclimb.policy import TUNE_ACTION, Action, BudgetView, PolicyInput, SearchPolicy

# Fractions of the budget still remaining at which every journal is
# probed: fresh, mid-search, and inside the ensemble window.
BUDGET_POINTS: tuple[float, ...] = (1.0, 0.5, 0.05)



def known_operators() -> frozenset[str]:
    """What the harness can run: every registered operator, plus `tune`."""
    return frozenset(operator_names()) | {TUNE_ACTION}


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


def _view(case: JournalCase, config: Config, fraction: float) -> PolicyInput:
    return PolicyInput(
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


def _replayed(make_policy: Callable[[], SearchPolicy], case: JournalCase, config: Config) -> SearchPolicy:
    """A fresh policy that has observed the journal in order — exactly what
    `PolicyLoop.catch_up` does when a search starts."""
    policy = make_policy()
    view = _view(case, config, 1.0)
    for candidate in view.journal.candidates.values():
        policy.observe(view, candidate)
    return policy


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


def _reference_problems(action: Action, journal: Journal) -> list[str]:
    problems: list[str] = []
    known = known_operators()
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
    elif action.operator in known:
        # the operator's own rule — the same one the harness applies before
        # it creates or spends anything
        reason = get_operator(action.operator).valid_target(target)
        if reason:
            problems.append(reason)
    return problems


def check_policy(
    make_policy: Callable[[], SearchPolicy],
    cases: Sequence[JournalCase],
    config: Config,
    *,
    budget_points: Sequence[float] = BUDGET_POINTS,
    prompts_dir: Path | None = None,
) -> CheckReport:
    """Run every conformance check over `cases` (plus a synthetic empty
    journal). `make_policy` must return a NEW instance each call — the
    replay check depends on it. Never writes; never runs an agent."""
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
        except Exception as exc:  # noqa: BLE001
            findings.append(Finding("replay", False, f"observe raised {type(exc).__name__}: {exc}", case.label))
            continue
        replay_ok, idem_ok, view_mutated = True, True, False
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
            if x is not None:
                for problem in _reference_problems(x, case.journal):
                    findings.append(Finding("references", False, f"at {fraction:.0%} budget: {problem}", case.label))
        if replay_ok:
            findings.append(Finding("replay", True, "; ".join(proposals), case.label))
        if idem_ok and replay_ok:
            findings.append(Finding("idempotent", True, "propose is stable", case.label))
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
