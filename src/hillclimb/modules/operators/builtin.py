"""The four built-in operators: draft, debug, improve, ensemble.

Written against `hillclimb.sdk` only, like any climber's own operators — the
templates they render (`draft.md`, `debug.md`, …) can be shadowed by name from
a prompts dir without touching this file.
"""

from __future__ import annotations

from hillclimb.sdk import Candidate, Operator, OperatorContext, Attempt, inspiration_filename

COMPLEXITY_CUES = {
    "minimal": (
        "Produce a MINIMAL, reliable first solution: the simplest sensible model "
        "for this data type (e.g., gradient-boosted trees for tabular data, "
        "TF-IDF + linear model for simple text). No ensembles, no neural networks, "
        "no heavy tuning. Prioritize a correct end-to-end pipeline over cleverness."
    ),
    "moderate": (
        "Produce a MODERATE solution: a solid model with sensible feature "
        "engineering and light hyperparameter tuning. Still no large ensembles."
    ),
    "advanced": (
        "You may produce an ADVANCED solution: stronger models, richer feature "
        "engineering, careful tuning — but it must still run reliably within the "
        "execution time limit."
    ),
}


class Draft(Operator):
    """A new solution from the problem alone. `retrieval` (default on) adds
    the research cue; the first draft of a search also gets memory's proven
    reference solution as a scaffold — later drafts must diverge, so seeding
    them all would fight exploration."""

    name = "draft"
    role = "create"

    def prepare(self, ctx: OperatorContext) -> Attempt:
        problem = ctx.problem
        prior = "\n\n".join(part for part in (ctx.memory.text, ctx.live_experience()) if part)
        research_cue = (
            ctx.render("research_cue", network_note=problem.network_note).rstrip() + "\n"
            # an agent without internet cannot do the research the cue asks for
            if self.params.get("retrieval", True) and ctx.agent_internet
            else ""
        )
        reference = ctx.memory.reference
        wants_reference = reference is not None and reference.exists() and not ctx.journal.drafts()
        starter_cue = ""
        if wants_reference:
            note = f" ({ctx.memory.reference_note})" if ctx.memory.reference_note else ""
            starter_cue = (
                "# Starter reference\n\n"
                f"A proven solution from a previous search is at "
                f"`./reference_solution.py`{note}. Use it as a scaffold: adapt "
                "and improve it for THIS problem — do not resubmit it unchanged.\n"
            )
        prompt = ctx.render(
            "draft",
            description=problem.description,
            metric_name=problem.metric_name,
            direction=problem.direction,
            data_listing=problem.data_listing,
            research_cue=research_cue,
            starter_cue=starter_cue,
            complexity_cue=COMPLEXITY_CUES[ctx.action.args.get("complexity") or "minimal"],
            prior_experience=prior or "(no prior searches recorded)",
            prior_drafts=ctx.summaries(ctx.journal.drafts()) or "(none yet)",
        )
        return Attempt(
            prompt=prompt,
            files={"reference_solution.py": reference} if wants_reference else {},
        )


class Debug(Operator):
    """A fix of a candidate that failed. The chain's earlier failed fixes ride
    in the prompt; where the agent can, the target's agent session is
    continued as well."""

    name = "debug"
    role = "repair"
    needs_target = True

    def valid_target(self, target: Candidate | None) -> str | None:
        reason = super().valid_target(target)
        if reason is None and target.status not in ("failing", "buggy"):
            reason = f"debug targets {target.candidate_id} whose status is {target.status}"
        return reason

    def prepare(self, ctx: OperatorContext) -> Attempt:
        target = ctx.target
        chain = ctx.journal.debug_chain(target.candidate_id)
        root, attempts = chain[0], chain[1:]
        last_replicate = target.last_replicate
        test_result = target.last_trial.unit_tests if target.last_trial else None
        prompt = ctx.render(
            "debug",
            parent_summary=root.summary or "(no summary)",
            failure_reason=ctx.failure_reason(target),
            stderr_tail=(
                test_result.stderr_tail if test_result and test_result.stderr_tail
                else ctx.tail(target, "exec_stderr.log")
            ),
            stdout_tail=(
                test_result.stdout_tail if test_result and test_result.stdout_tail
                else (last_replicate.stdout_tail if last_replicate else "")
            ),
            debug_history=ctx.summaries(attempts) or "(none — this is the first fix attempt)",
        )
        return Attempt(prompt=prompt, copy_parent=True, inherit_params=True, fork_session=True)


class Improve(Operator):
    """A change to a scored candidate. `ablation` (default on) asks the agent
    to measure which components carry the score, and hands the next improve
    of the same target what the last one measured."""

    name = "improve"
    role = "refine"
    needs_target = True

    def valid_target(self, target: Candidate | None) -> str | None:
        reason = super().valid_target(target)
        if reason is None and not target.is_scored:
            reason = f"improve targets unscored {target.candidate_id}"
        return reason

    def prepare(self, ctx: OperatorContext) -> Attempt:
        problem, target = ctx.problem, ctx.target
        last_replicate = target.last_replicate
        live = ctx.live_experience()
        ablation = self.params.get("ablation", True)
        prompt = ctx.render(
            "improve",
            description=problem.description,
            metric_name=problem.metric_name,
            direction=problem.direction,
            best_score=target.val_score,
            stdout_tail=last_replicate.stdout_tail if last_replicate else "",
            sibling_summaries=ctx.summaries(ctx.journal.children(target.candidate_id))
            or "(nothing tried from this solution yet)",
            evaluation_report=ctx.report_section(target),
            live_experience=f"# Discoveries from concurrent searches\n\n{live}\n" if live else "",
            prior_ablations=self._prior_ablations(ctx) if ablation else "",
            ablation_cue=ctx.render("ablation_cue").rstrip() + "\n" if ablation else "",
        )
        return Attempt(prompt=prompt, copy_parent=True, inherit_params=True)

    @staticmethod
    def _prior_ablations(ctx: OperatorContext) -> str:
        """The newest `ablation.md` an earlier improve of this same target
        wrote, so successive improves don't re-measure the same components."""
        for child in reversed(ctx.journal.children(ctx.target.candidate_id, include_pruned=True)):
            body = ctx.read_text(child, "ablation.md")
            if body:
                return (
                    f"# Prior ablation findings for this solution "
                    f"(measured by {child.candidate_id})\n\n{body}\n"
                )
        return ""


class Ensemble(Operator):
    """One solution out of several: the action's inspirations are copied in
    and tabled with their validation scores."""

    name = "ensemble"
    role = "combine"
    needs_target = True

    def prepare(self, ctx: OperatorContext) -> Attempt:
        if not ctx.inspirations:
            raise ValueError("ensemble needs inspiration_ids: the solutions to combine")
        problem = ctx.problem
        table = "\n".join(
            f"- `{inspiration_filename(i)}` — validation {problem.metric_name}: "
            f"**{c.val_score:.5g}** ({c.candidate_id}): {c.summary or '(no summary)'}"
            for i, c in enumerate(ctx.inspirations, 1)
        )
        prompt = ctx.render(
            "ensemble",
            description=problem.description,
            metric_name=problem.metric_name,
            direction=problem.direction,
            candidates_table=table,
        )
        return Attempt(prompt=prompt)


BUILTIN_OPERATORS = (Draft, Debug, Improve, Ensemble)
