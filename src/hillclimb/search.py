from __future__ import annotations

import shutil
from pathlib import Path

from hillclimb.backends.base import OperatorBackend, OperatorRequest
from hillclimb.baseline import write_baseline
from hillclimb.budget import BudgetManager
from hillclimb.config import Config
from hillclimb.executor import Executor
from hillclimb.journal import Journal
from hillclimb.node import BackendInfo, ExecInfo, Node, utcnow
from hillclimb.prompts.render import COMPLEXITY_CUES, render
from hillclimb.task import TaskSpec
from hillclimb.workspace import create_node_workspace

TAIL_CHARS = 2000


class ParkedRun(Exception):
    """Raised when the backend hits a rate limit; the run can be resumed."""


def tail(path: Path, chars: int = TAIL_CHARS) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[-chars:]


class GreedySearcher:
    """Deterministic greedy policy over the solution tree.

    Policy: (1) baseline at t=0; (2) DEBUG the newest buggy node while its
    chain is shallow; (3) DRAFT until `num_drafts` branches have a scored
    solution (complexity cue escalates per draft); (4) otherwise IMPROVE the
    best node. Stops on budget margin or max_nodes.
    """

    def __init__(
        self,
        task: TaskSpec,
        config: Config,
        journal: Journal,
        backend: OperatorBackend,
        executor: Executor,
        budget: BudgetManager,
        run_dir: Path,
        max_nodes: int = 50,
        log=print,
    ):
        self.task = task
        self.config = config
        self.journal = journal
        self.backend = backend
        self.executor = executor
        self.budget = budget
        self.run_dir = run_dir
        self.max_nodes = max_nodes
        self.log = log
        self._consecutive_failures = 0

    # --- main loop ---

    def run(self) -> Node | None:
        if not self.journal.nodes:
            self.journal.node_result(write_baseline(self.task, self.run_dir))
            self.log("baseline submission written (sample_submission copy)")
        while not self.budget.should_stop() and len(self.journal.nodes) < self.max_nodes:
            operator, target = self.decide()
            self.log(
                f"[{self.budget.remaining_str()} left] {operator}"
                + (f" -> {target.node_id}" if target else "")
            )
            self.run_operator(operator, target)
        return self.journal.best_node(self.task.lower_is_better)

    def decide(self) -> tuple[str, Node | None]:
        nodes = list(self.journal.nodes.values())
        last = next((n for n in reversed(nodes) if n.status in ("ok", "buggy")), None)
        if last is not None and last.status == "buggy":
            chain = self.journal.debug_chain(last.node_id)
            depth = sum(1 for n in chain if n.operator == "debug")
            if depth < self.config.search.max_debug_depth:
                return "debug", last
        if self._scored_branches() < self.config.search.num_drafts:
            return "draft", None
        best = self.journal.best_node(self.task.lower_is_better)
        if best is None:
            return "draft", None
        return "improve", best

    def _scored_branches(self) -> int:
        """Draft branches whose subtree contains at least one scored node."""
        count = 0
        for draft in self.journal.drafts():
            frontier = [draft]
            while frontier:
                node = frontier.pop()
                if node.is_scored:
                    count += 1
                    break
                frontier.extend(self.journal.children(node.node_id))
        return count

    # --- node lifecycle ---

    def run_operator(self, operator: str, target: Node | None) -> Node:
        node_id = self.journal.next_node_id()
        parent_solution = (
            Path(target.workspace) / "solution.py"
            if target is not None and operator in ("debug", "improve")
            else None
        )
        workspace = create_node_workspace(
            self.run_dir, node_id, self.task.data_dir, parent_solution
        )
        complexity = self._draft_complexity() if operator == "draft" else None
        prompt = self.build_prompt(operator, target, complexity)
        (workspace / "prompt.md").write_text(prompt)

        chain_session = None
        if operator == "debug" and target is not None:
            chain_session = self._chain_session(target)

        node = Node(
            node_id=node_id,
            parent_id=target.node_id if target else None,
            operator=operator,
            complexity=complexity,
            debug_depth=(
                sum(1 for n in self.journal.debug_chain(target.node_id) if n.operator == "debug") + 1
                if operator == "debug" and target
                else 0
            ),
            workspace=str(workspace),
        )
        self.journal.node_created(node)

        request = OperatorRequest(
            operator=operator,
            prompt=prompt,
            workspace=workspace,
            timeout_s=min(
                self.config.budget.agent_timeout_s, max(60, int(self.budget.remaining()))
            ),
            model=self.config.model,
            resume_session_id=chain_session,
        )
        result = self.backend.invoke(request)
        node.backend = BackendInfo(
            name=self.backend.name,
            session_id=result.session_id,
            cost_usd=result.cost_usd,
            num_turns=result.num_turns,
            agent_duration_s=result.duration_s,
            error_kind=result.error_kind,
        )

        if result.error_kind == "rate_limited":
            node.status = "parked"
            node.summary = f"parked: {result.error_message}"
            node.finished_at = utcnow()
            self.journal.node_result(node)
            raise ParkedRun(result.error_message)

        if not result.ok:
            # e.g. network down: solution.py may still exist as the parent's
            # copy — executing it would silently re-score the parent
            node.status = "abandoned"
            node.summary = f"agent call failed ({result.error_kind}): {result.error_message[:150]}"
            node.finished_at = utcnow()
            self.journal.node_result(node)
            self._consecutive_failures += 1
            self.log(f"  agent call failed ({self._consecutive_failures} in a row)")
            if self._consecutive_failures >= 3:
                raise ParkedRun(
                    f"3 consecutive agent failures; last: {result.error_message[:200]}"
                )
            return node
        self._consecutive_failures = 0

        solution = workspace / "solution.py"
        notes = workspace / "notes.md"
        if notes.exists():
            lines = notes.read_text().strip().splitlines()
            node.summary = lines[0] if lines else ""

        if not solution.exists():
            node.status = "abandoned"
            node.summary = node.summary or f"agent produced no solution.py ({result.error_kind})"
            node.finished_at = utcnow()
            self.journal.node_result(node)
            return node

        exec_timeout = min(
            self.config.budget.exec_timeout_s, max(60, int(self.budget.remaining() - 30))
        )
        exec_result = self.executor.execute(solution, workspace, exec_timeout)
        node.execution = ExecInfo(
            returncode=exec_result.returncode,
            duration_s=exec_result.duration_s,
            timed_out=exec_result.timed_out,
            stdout_tail=tail(Path(exec_result.stdout_path)) if exec_result.stdout_path else "",
            submission_ok=exec_result.submission_ok,
        )

        if exec_result.ok:
            previous_best = self.journal.best_node(self.task.lower_is_better)
            node.status = "ok"
            node.val_score = exec_result.val_score
            if previous_best is None or self._improves(exec_result.val_score, previous_best.val_score):
                node.is_best = True
                shutil.copy(workspace / "submission.csv", self.run_dir / "best" / "submission.csv")
                shutil.copy(solution, self.run_dir / "best" / "solution.py")
                self.log(f"  new best: {node.node_id} val_score={node.val_score}")
        else:
            node.status = "buggy"
        node.finished_at = utcnow()
        self.journal.node_result(node)
        return node

    def _improves(self, score: float, best: float) -> bool:
        return score < best if self.task.lower_is_better else score > best

    def _draft_complexity(self) -> str:
        index = len(self.journal.drafts())
        return "minimal" if index == 0 else "moderate" if index == 1 else "advanced"

    def _chain_session(self, target: Node) -> str | None:
        """Resume the debug chain's agent session so it remembers prior fixes."""
        for node in reversed(self.journal.debug_chain(target.node_id)):
            if node.backend.session_id:
                return node.backend.session_id
        return None

    # --- prompt assembly ---

    def build_prompt(self, operator: str, target: Node | None, complexity: str | None) -> str:
        contract = render(
            "contract",
            metric_name=self.task.metric_name,
            exec_timeout_min=self.config.budget.exec_timeout_s // 60,
            runtime_pkgs=self._runtime_pkgs(),
            time_remaining=self.budget.remaining_str(),
        )
        direction = "lower is better" if self.task.lower_is_better else "higher is better"
        if operator == "draft":
            return render(
                "draft",
                description=self.task.description,
                metric_name=self.task.metric_name,
                direction=direction,
                data_listing=self._data_listing(),
                complexity_cue=COMPLEXITY_CUES[complexity or "minimal"],
                prior_drafts=self._node_summaries(self.journal.drafts()) or "(none yet)",
                contract=contract,
            )
        if operator == "debug":
            assert target is not None
            chain = self.journal.debug_chain(target.node_id)
            root, attempts = chain[0], chain[1:]
            return render(
                "debug",
                parent_summary=root.summary or "(no summary)",
                failure_reason=self._failure_reason(target),
                stderr_tail=tail(Path(target.workspace) / "exec_stderr.log"),
                stdout_tail=target.execution.stdout_tail,
                debug_history=self._node_summaries(attempts) or "(none — this is the first fix attempt)",
                contract=contract,
            )
        if operator == "improve":
            assert target is not None
            return render(
                "improve",
                description=self.task.description,
                metric_name=self.task.metric_name,
                direction=direction,
                best_score=target.val_score,
                stdout_tail=target.execution.stdout_tail,
                sibling_summaries=self._node_summaries(self.journal.children(target.node_id))
                or "(nothing tried from this solution yet)",
                contract=contract,
            )
        raise ValueError(f"Unknown operator: {operator}")

    def _failure_reason(self, node: Node) -> str:
        execution = node.execution
        if execution.timed_out:
            return "The script exceeded its execution time limit and was killed."
        if execution.returncode not in (0, None):
            return f"The script crashed (exit code {execution.returncode})."
        problems = []
        if not execution.submission_ok:
            problems.append("`submission.csv` was not written")
        if node.val_score is None:
            problems.append("no final `val_score: <float>` line was printed")
        if problems:
            return "The script ran to completion but violated the contract: " + "; ".join(problems) + "."
        return "The script failed."

    def _node_summaries(self, nodes: list[Node]) -> str:
        lines = []
        for node in nodes:
            score = f"val_score={node.val_score}" if node.val_score is not None else node.status
            lines.append(f"- {node.node_id} ({node.operator}, {score}): {node.summary or '(no summary)'}")
        return "\n".join(lines)

    def _data_listing(self, limit: int = 50) -> str:
        entries = []
        for path in sorted(self.task.data_dir.rglob("*")):
            if path.is_file():
                relative = path.relative_to(self.task.data_dir)
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
