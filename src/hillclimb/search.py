from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd

from hillclimb.backends.base import OperatorBackend, OperatorRequest
from hillclimb.baseline import write_baseline
from hillclimb.budget import BudgetManager
from hillclimb.candidate import BackendInfo, Candidate, Trial, utcnow
from hillclimb.config import Config
from hillclimb.control import ControlCommand, apply_prune, read_commands, resync_best
from hillclimb.executor import Executor
from hillclimb.holdout import HoldoutInfo
from hillclimb.journal import Journal
from hillclimb.prompts.render import COMPLEXITY_CUES, render
from hillclimb.scoring import ScoringError, score
from hillclimb.status import CandidateCounts, CurrentCandidate, ScoreRef, StatusWriter
from hillclimb.problem import ProblemSpec
from hillclimb.workspace import create_candidate_workspace

TAIL_CHARS = 2000


class ParkedSearch(Exception):
    """Raised when the backend hits a rate limit; the search can be resumed."""


class StopRequested(Exception):
    """Raised on a graceful stop (control command or SIGTERM); the search can
    be resumed."""


def tail(path: Path, chars: int = TAIL_CHARS) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[-chars:]


class GreedySearcher:
    """Deterministic greedy policy over the candidate tree.

    Policy: (1) baseline at t=0; (2) DEBUG the newest buggy candidate while
    its chain is shallow; (3) DRAFT until `num_drafts` branches have a scored
    solution (complexity cue escalates per draft); (4) otherwise IMPROVE the
    best candidate. Stops on budget margin or max_candidates.
    """

    def __init__(
        self,
        problem: ProblemSpec,
        config: Config,
        journal: Journal,
        backend: OperatorBackend,
        executor: Executor,
        budget: BudgetManager,
        search_dir: Path,
        max_candidates: int = 50,
        log=print,
        holdout: HoldoutInfo | None = None,
        status: StatusWriter | None = None,
    ):
        self.problem = problem
        self.config = config
        self.journal = journal
        self.backend = backend
        self.executor = executor
        self.budget = budget
        self.search_dir = search_dir
        self.max_candidates = max_candidates
        self.log = log
        self.holdout = holdout
        self.status = status
        self._consecutive_failures = 0
        for stale in journal.pending_candidates():
            # a pending candidate at construction time means a previous
            # orchestrator process died mid-operator (crash/kill); its work is
            # unaccounted and must not block decide() forever
            stale.status = "abandoned"
            stale.summary = stale.summary or "orchestrator died mid-operator (crash recovery)"
            stale.finished_at = utcnow()
            journal.candidate_result(stale)
            log(f"  recovered stale pending candidate {stale.candidate_id} -> abandoned")
        existing = journal.selected_candidate(problem.lower_is_better, config.holdout.selection)
        self._selection_id = existing.candidate_id if existing else None  # resume-safe

    @property
    def data_dir(self) -> Path:
        return self.holdout.data_view if self.holdout else self.problem.data_dir

    # --- main loop ---

    def run(self) -> Candidate | None:
        if not self.journal.candidates:
            self.journal.candidate_result(write_baseline(self.problem, self.search_dir))
            self.log("baseline submission written (sample_submission copy)")
        while not self.budget.should_stop() and len(self.journal.candidates) < self.max_candidates:
            self._process_control()
            self._status(current=None)
            operator, target = self.decide()
            self.log(
                f"[{self.budget.remaining_str()} left] {operator}"
                + (f" -> {target.candidate_id}" if target else "")
            )
            self.run_operator(operator, target)
        return self.journal.selected_candidate(
            self.problem.lower_is_better, self.config.holdout.selection
        )

    # --- status reporting ---

    def _status(self, **fields) -> None:
        if self.status is None:
            return
        candidates = self.journal.candidates.values()
        fields.setdefault(
            "candidates",
            CandidateCounts(
                total=len(self.journal.candidates),
                ok=sum(1 for c in candidates if c.status == "ok"),
                buggy=sum(1 for c in candidates if c.status == "buggy"),
                pruned=sum(1 for c in candidates if c.pruned),
            ),
        )
        best = self.journal.best_candidate(self.problem.lower_is_better)
        if best is not None:
            fields.setdefault(
                "best", ScoreRef(candidate_id=best.candidate_id, val_score=best.val_score)
            )
        selected = self.journal.selected_candidate(
            self.problem.lower_is_better, self.config.holdout.selection
        )
        if selected is not None:
            fields.setdefault(
                "selected",
                ScoreRef(
                    candidate_id=selected.candidate_id,
                    val_score=selected.val_score,
                    holdout_score=selected.holdout_score,
                ),
            )
        self.status.update(**fields)

    def _process_control(self) -> None:
        """Apply queued user commands (control/) between operators. Prunes are
        applied before a stop so nothing is left half-processed."""
        commands = read_commands(self.search_dir)
        stop: ControlCommand | None = None
        for path, cmd in sorted(commands, key=lambda pc: pc[1].action != "prune"):
            path.unlink(missing_ok=True)
            if cmd.action == "stop":
                stop = cmd
            elif cmd.action == "prune" and cmd.candidate_id:
                try:
                    pruned = apply_prune(self.journal, cmd.candidate_id, cmd.reason, cmd.source)
                except ValueError as exc:
                    self.log(f"  prune {cmd.candidate_id} rejected: {exc}")
                    continue
                if not pruned:
                    continue
                self.log(f"  pruned {', '.join(pruned)} (by {cmd.source})")
                if self._selection_id in pruned:
                    self._selection_id = resync_best(
                        self.search_dir,
                        self.journal,
                        self.problem.lower_is_better,
                        self.config.holdout.selection,
                    )
        if stop is not None:
            self.journal.control_event("stop", reason=stop.reason, source=stop.source)
            raise StopRequested(f"stop requested by {stop.source}")

    def decide(self) -> tuple[str, Candidate | None]:
        candidates = list(self.journal.candidates.values())
        last = next(
            (c for c in reversed(candidates) if c.status in ("ok", "buggy") and not c.pruned), None
        )
        if last is not None and last.status == "buggy":
            chain = self.journal.debug_chain(last.candidate_id)
            depth = sum(1 for c in chain if c.operator == "debug")
            if depth < self.config.search.max_debug_depth:
                return "debug", last
        if self._should_ensemble():
            return "ensemble", self._ensemble_candidates()[0]
        if self._scored_branches() < self.config.search.num_drafts:
            return "draft", None
        best = self.journal.best_candidate(self.problem.lower_is_better)
        if best is None:
            return "draft", None
        return "improve", best

    # --- ensemble stage ---

    def _in_ensemble_window(self) -> bool:
        # window sits ABOVE the stop margin, else margin swallows it: with a
        # 45m budget, reserve(540s) - margin(300s) left a 240s slot that one
        # improve cycle stepped over entirely
        reserve = self.budget.total_s * self.config.ensemble.reserve_fraction
        return self.budget.remaining() <= reserve + self.budget.stop_margin_s

    def _should_ensemble(self) -> bool:
        cfg = self.config.ensemble
        if not cfg.enabled or not self._in_ensemble_window():
            return False
        attempts = sum(1 for c in self.journal.candidates.values() if c.operator == "ensemble")
        if attempts >= cfg.max_attempts or self._ensemble_succeeded():
            return False
        return len(self._ensemble_candidates()) >= 2

    def _ensemble_succeeded(self) -> bool:
        for candidate in self.journal.candidates.values():
            if candidate.status != "ok":
                continue
            root = self.journal.debug_chain(candidate.candidate_id)[0]
            if root.operator == "ensemble":
                return True
        return False

    def _ensemble_candidates(self) -> list[Candidate]:
        """Top-k scored non-ensemble candidates by the selection rule, deduped
        by script content so near-identical improves don't fill the slots."""
        ranked = self.journal.ranked_candidates(
            self.problem.lower_is_better, self.config.holdout.selection
        )
        picked, seen_hashes = [], set()
        for candidate in ranked:
            if candidate.operator == "ensemble":
                continue
            solution = Path(candidate.workspace) / "solution.py"
            if not solution.exists():
                continue
            digest = hashlib.md5(solution.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            picked.append(candidate)
            if len(picked) >= self.config.ensemble.top_k:
                break
        return picked

    def _scored_branches(self) -> int:
        """Draft branches whose subtree contains at least one scored candidate."""
        count = 0
        for draft in self.journal.drafts():
            frontier = [draft]
            while frontier:
                candidate = frontier.pop()
                if candidate.is_scored:
                    count += 1
                    break
                frontier.extend(self.journal.children(candidate.candidate_id))
        return count

    # --- candidate lifecycle ---

    def run_operator(self, operator: str, target: Candidate | None) -> Candidate:
        candidate_id = self.journal.next_candidate_id()
        parent_solution = (
            Path(target.workspace) / "solution.py"
            if target is not None and operator in ("debug", "improve")
            else None
        )
        workspace = create_candidate_workspace(
            self.search_dir,
            candidate_id,
            self.data_dir,
            self.problem.problem_dir,
            parent_solution,
        )
        ensemble_inputs = None
        if operator == "ensemble":
            ensemble_inputs = self._ensemble_candidates()
            for i, cand in enumerate(ensemble_inputs, 1):
                shutil.copy(Path(cand.workspace) / "solution.py", workspace / f"candidate_{i}.py")
        complexity = self._draft_complexity() if operator == "draft" else None
        prompt = self.build_prompt(operator, target, complexity, ensemble_inputs)
        (workspace / "prompt.md").write_text(prompt)

        # NOTE: no session resume across candidates — Claude Code scopes
        # sessions to the cwd, and every candidate has its own workspace, so
        # --resume can't find a sibling workspace's session. The debug prompt
        # carries the chain's failed-fix history from the journal instead.
        candidate = Candidate(
            candidate_id=candidate_id,
            parent_id=target.candidate_id if target else None,
            operator=operator,
            complexity=complexity,
            debug_depth=(
                sum(1 for c in self.journal.debug_chain(target.candidate_id) if c.operator == "debug") + 1
                if operator == "debug" and target
                else 0
            ),
            workspace=str(workspace),
        )
        self.journal.candidate_created(candidate)

        request = OperatorRequest(
            operator=operator,
            prompt=prompt,
            workspace=workspace,
            timeout_s=min(
                self.config.budget.agent_timeout_s, max(60, int(self.budget.remaining()))
            ),
            model=self.config.model,
        )
        self._status(
            current=CurrentCandidate(
                candidate_id=candidate_id, operator=operator, phase="agent", workspace=str(workspace)
            )
        )
        result = self.backend.invoke(request)
        candidate.backend = BackendInfo(
            name=self.backend.name,
            session_id=result.session_id,
            cost_usd=result.cost_usd,
            num_turns=result.num_turns,
            agent_duration_s=result.duration_s,
            error_kind=result.error_kind,
        )

        if result.error_kind == "rate_limited":
            candidate.status = "parked"
            candidate.summary = f"parked: {result.error_message}"
            candidate.finished_at = utcnow()
            self.journal.candidate_result(candidate)
            raise ParkedSearch(result.error_message)

        if not result.ok:
            # e.g. network down: solution.py may still exist as the parent's
            # copy — executing it would silently re-score the parent
            candidate.status = "abandoned"
            candidate.summary = f"agent call failed ({result.error_kind}): {result.error_message[:150]}"
            candidate.finished_at = utcnow()
            self.journal.candidate_result(candidate)
            self._consecutive_failures += 1
            self.log(f"  agent call failed ({self._consecutive_failures} in a row)")
            if self._consecutive_failures >= 3:
                raise ParkedSearch(
                    f"3 consecutive agent failures; last: {result.error_message[:200]}"
                )
            return candidate
        self._consecutive_failures = 0

        solution = workspace / "solution.py"
        notes = workspace / "notes.md"
        if notes.exists():
            lines = notes.read_text().strip().splitlines()
            candidate.summary = lines[0] if lines else ""

        if not solution.exists():
            candidate.status = "abandoned"
            candidate.summary = candidate.summary or f"agent produced no solution.py ({result.error_kind})"
            candidate.finished_at = utcnow()
            self.journal.candidate_result(candidate)
            return candidate

        exec_timeout = min(
            self.config.budget.exec_timeout_s, max(60, int(self.budget.remaining() - 30))
        )
        self._status(
            current=CurrentCandidate(
                candidate_id=candidate_id, operator=operator, phase="exec", workspace=str(workspace)
            )
        )
        trial_started = utcnow()
        exec_result = self.executor.execute(
            solution,
            workspace,
            exec_timeout,
            verifier=self.problem.verifier,
        )
        trial = Trial(
            returncode=exec_result.returncode,
            duration_s=exec_result.duration_s,
            timed_out=exec_result.timed_out,
            stdout_tail=tail(Path(exec_result.stdout_path)) if exec_result.stdout_path else "",
            submission_ok=exec_result.submission_ok,
            started_at=trial_started,
        )
        candidate.trials.append(trial)

        if exec_result.ok:
            holdout_score, holdout_error = self._score_holdout(workspace)
            # score recorded even when the candidate ends buggy from a holdout
            # contract violation — status/TUI display depend on it
            trial.val_score = exec_result.val_score
            if holdout_error is not None:
                candidate.status = "buggy"
                trial.holdout_error = holdout_error
            else:
                previous_best = self.journal.best_candidate(self.problem.lower_is_better)
                candidate.status = "ok"
                trial.holdout_score = holdout_score
                if previous_best is None or self._improves(exec_result.val_score, previous_best.val_score):
                    candidate.is_best = True
        else:
            candidate.status = "buggy"
        trial.finished_at = utcnow()
        candidate.finished_at = utcnow()
        self.journal.candidate_result(candidate)
        if candidate.status == "ok":
            self._sync_selection()
        self._status(current=None)
        return candidate

    def _sync_selection(self) -> None:
        """Keep best/ pointing at the currently selected candidate. Selection
        is recomputed over the whole tree because rank-blend can shift between
        existing candidates when a new one lands."""
        selected = self.journal.selected_candidate(
            self.problem.lower_is_better, self.config.holdout.selection
        )
        if selected is None or selected.candidate_id == self._selection_id:
            return
        self._selection_id = resync_best(
            self.search_dir, self.journal, self.problem.lower_is_better, self.config.holdout.selection
        )
        scores = f"val_score={selected.val_score}"
        if selected.holdout_score is not None:
            scores += f" holdout={selected.holdout_score:.5g}"
        self.log(f"  new selection: {selected.candidate_id} {scores}")

    def _score_holdout(self, workspace: Path) -> tuple[float | None, str | None]:
        """Score holdout predictions; (score, None) on success, (None, reason)
        on contract violation, (None, None) when holdout is disabled."""
        if self.holdout is None:
            return None, None
        pred_path = workspace / "holdout_predictions.csv"
        if not pred_path.exists():
            return None, "`holdout_predictions.csv` was not written"
        try:
            predictions = pd.read_csv(pred_path)
            answers = pd.read_csv(self.holdout.answers_path)
            value = score(self.problem.metric_name, answers, predictions, self.holdout.id_col)
        except ScoringError as e:
            return None, f"holdout_predictions.csv could not be scored: {e}"
        except Exception as e:
            return None, f"holdout_predictions.csv is unreadable: {e}"
        return value, None

    def _improves(self, score: float, best: float) -> bool:
        return score < best if self.problem.lower_is_better else score > best

    def _draft_complexity(self) -> str:
        index = len(self.journal.drafts())
        return "minimal" if index == 0 else "moderate" if index == 1 else "advanced"

    # --- prompt assembly ---

    def build_prompt(
        self,
        operator: str,
        target: Candidate | None,
        complexity: str | None,
        ensemble_inputs: list[Candidate] | None = None,
    ) -> str:
        holdout_clause = ""
        if self.holdout is not None:
            if self.holdout.strategy == "time-tail":
                if self.holdout.group_col:
                    scope = f"the most recent rows of each `{self.holdout.group_col}` block"
                elif self.holdout.time_cutoff:
                    scope = f"all rows from {self.holdout.time_cutoff} onward"
                else:
                    scope = "the most recent rows"
                split_note = (
                    f"These rows are the chronological TAIL of the training data ({scope}), "
                    "held out by the orchestrator. Treat them as a true forecast: do not "
                    "train on them, and do not use any information from the holdout period."
                )
            else:
                split_note = "These rows were held out at random from the training data."
            # class-columns problems (target column holds class names, e.g.
            # spooky's `author`): predictions must be per-class probability
            # columns in submission format, NOT the raw target column — an
            # agent following the literal column name writes hard labels the
            # log-loss scorer can't grade
            sample_cols = pd.read_csv(self.problem.sample_submission, nrows=0).columns
            if set(self.holdout.target_cols) <= set(sample_cols):
                target_cols_note = ", ".join(f"`{c}`" for c in self.holdout.target_cols)
            else:
                pred_cols = [c for c in sample_cols if c != sample_cols[0]]
                shown = ", ".join(f"`{c}`" for c in pred_cols[:6])
                if len(pred_cols) > 6:
                    shown += f", … ({len(pred_cols)} columns)"
                target_cols_note = (
                    f"the same prediction columns as submission.csv: {shown}"
                )
            holdout_clause = render(
                "holdout_clause",
                holdout_id_col=self.holdout.id_col,
                holdout_target_cols=target_cols_note,
                holdout_split_note=split_note,
            ).rstrip()
        network_note = (
            "Internet access IS available at execution time — this problem's rules "
            "permit fetching external data; cache downloads to files in the "
            "working directory so reruns don't refetch."
            if self.problem.allow_network
            else "Assume no internet access at execution time."
        )
        contract = render(
            "contract",
            metric_name=self.problem.metric_name,
            exec_timeout_min=self.config.budget.exec_timeout_s // 60,
            runtime_pkgs=self._runtime_pkgs(),
            time_remaining=self.budget.remaining_str(),
            holdout_clause=holdout_clause,
            network_note=network_note,
            verifier_clause=self._verifier_clause(),
        )
        direction = "lower is better" if self.problem.lower_is_better else "higher is better"
        if operator == "draft":
            return render(
                "draft",
                description=self.problem.description,
                metric_name=self.problem.metric_name,
                direction=direction,
                data_listing=self._data_listing(),
                complexity_cue=COMPLEXITY_CUES[complexity or "minimal"],
                prior_drafts=self._candidate_summaries(self.journal.drafts()) or "(none yet)",
                contract=contract,
            )
        if operator == "debug":
            assert target is not None
            chain = self.journal.debug_chain(target.candidate_id)
            root, attempts = chain[0], chain[1:]
            last_trial = target.last_trial
            return render(
                "debug",
                parent_summary=root.summary or "(no summary)",
                failure_reason=self._failure_reason(target),
                stderr_tail=tail(Path(target.workspace) / "exec_stderr.log"),
                stdout_tail=last_trial.stdout_tail if last_trial else "",
                debug_history=self._candidate_summaries(attempts) or "(none — this is the first fix attempt)",
                contract=contract,
            )
        if operator == "ensemble":
            assert ensemble_inputs
            table = "\n".join(
                f"- `candidate_{i}.py` — validation {self.problem.metric_name}: "
                f"**{c.val_score:.5g}** ({c.candidate_id}): {c.summary or '(no summary)'}"
                for i, c in enumerate(ensemble_inputs, 1)
            )
            return render(
                "ensemble",
                description=self.problem.description,
                metric_name=self.problem.metric_name,
                direction=direction,
                candidates_table=table,
                contract=contract,
            )
        if operator == "improve":
            assert target is not None
            last_trial = target.last_trial
            return render(
                "improve",
                description=self.problem.description,
                metric_name=self.problem.metric_name,
                direction=direction,
                best_score=target.val_score,
                stdout_tail=last_trial.stdout_tail if last_trial else "",
                sibling_summaries=self._candidate_summaries(self.journal.children(target.candidate_id))
                or "(nothing tried from this solution yet)",
                contract=contract,
            )
        raise ValueError(f"Unknown operator: {operator}")

    def _failure_reason(self, candidate: Candidate) -> str:
        trial = candidate.last_trial
        if trial is None:
            return "The script failed."
        if trial.timed_out:
            return "The script exceeded its execution time limit and was killed."
        if trial.returncode not in (0, None):
            return f"The script crashed (exit code {trial.returncode})."
        problems = []
        if not trial.submission_ok:
            problems.append("`submission.csv` was not written")
        if trial.val_score is None:
            problems.append("no final `val_score: <float>` line was printed")
        if trial.holdout_error:
            problems.append(trial.holdout_error)
        if problems:
            return "The script ran to completion but violated the contract: " + "; ".join(problems) + "."
        return "The script failed."

    def _verifier_clause(self) -> str:
        if self.problem.verifier is None:
            return (
                "- prints exactly one line `val_score: <float>` "
                f"(your validation {self.problem.metric_name}) as the FINAL line of stdout"
            )
        try:
            verifier_name = self.problem.verifier.relative_to(self.problem.problem_dir)
        except ValueError:
            verifier_name = self.problem.verifier.name
        return (
            f"- writes `./submission.csv`; the orchestrator then runs "
            f"`./problem/{verifier_name}` and uses the verifier's final "
            "`val_score: <float>` line as the official validation score. "
            "Do not print your own `val_score:` line."
        )

    def _candidate_summaries(self, candidates: list[Candidate]) -> str:
        lines = []
        for candidate in candidates:
            score = (
                f"val_score={candidate.val_score}"
                if candidate.val_score is not None
                else candidate.status
            )
            lines.append(
                f"- {candidate.candidate_id} ({candidate.operator}, {score}): "
                f"{candidate.summary or '(no summary)'}"
            )
        return "\n".join(lines)

    def _data_listing(self, limit: int = 50) -> str:
        entries = []
        for path in sorted(self.problem.problem_dir.rglob("*")):
            if path.is_file():
                relative = path.relative_to(self.problem.problem_dir)
                entries.append(f"- {relative} ({_human_size(path.stat().st_size)})")
            if len(entries) >= limit:
                entries.append("- ... (truncated)")
                break
        return "\n".join(entries)

    def _runtime_pkgs(self) -> str:
        req = Path(__file__).resolve().parents[2] / "runtime-requirements.txt"
        if not req.exists():
            return "pandas, numpy, scikit-learn"
        pkgs = [
            line.strip()
            for line in req.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return ", ".join(pkgs)


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
