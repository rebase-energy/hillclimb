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
    kind = "create"
    templates = ("draft", "research_cue")

    def prepare(self, ctx: OperatorContext) -> Attempt:
        problem = ctx.problem
        prior = "\n\n".join(part for part in (ctx.memory.text, ctx.live_experience()) if part)
        research_cue = (
            ctx.render("research_cue", network_note=problem.network_note).rstrip() + "\n"
            # a coding agent without internet cannot do the research the cue asks for
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
    in the prompt; where the coding agent can, the target's coding agent session is
    continued as well."""

    name = "debug"
    kind = "repair"
    needs_target = True
    templates = ("debug",)

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
    """A change to a scored candidate. `ablation` (default on) asks the coding agent
    to measure which components carry the score, and hands the next improve
    of the same target what the last one measured."""

    name = "improve"
    kind = "refine"
    needs_target = True
    templates = ("improve", "ablation_cue")

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
    kind = "combine"
    needs_target = True
    templates = ("ensemble",)

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

# Every `{{token}}` the templates above carry and what fills it, as (filled
# with, empty when). The guide a climber folder's prompts/README.md shows;
# tests/test_climber_get.py holds it to the templates, both ways.
TOKEN_GUIDE: dict[str, tuple[str, str]] = {
    "description": ("the problem's `description.md`", "never"),
    "metric_name": ("the metric's name, from `problem.yaml`", "never"),
    "direction": ("`higher is better` / `lower is better`", "never"),
    "data_listing": ("the files under the problem's `data/`", "the problem ships no data"),
    "research_cue": (
        "`research_cue.md`: a web-research step before coding, with `{{network_note}}` — "
        "whether the solution itself may reach the internet (`allow_internet_during_solution`)",
        "the operator's `retrieval` param is off, or the coding agent has no internet",
    ),
    "network_note": ("whether the solution may reach the internet at execution time", "never"),
    "starter_cue": (
        "a pointer to `reference_solution.py`: memory's proven solution from an earlier search, copied in",
        "memory has none, or an earlier draft of this search already had it",
    ),
    "complexity_cue": ("the `minimal` / `moderate` / `advanced` cue the policy picked for this draft", "never"),
    "prior_experience": (
        "memory: knowledge cards from earlier searches on this problem, and discoveries of searches running beside this one",
        "`(no prior searches recorded)`",
    ),
    "prior_drafts": ("one line per draft this search already made: id, score, summary", "`(none yet)`"),
    "parent_summary": ("the failing candidate's own summary of its approach (its `notes.md` first line)", "`(no summary)`"),
    "failure_reason": ("why the verifier run did not pass, in the harness's words", "never"),
    "stderr_tail": ("the end of the failing run's stderr (or of its unit-test run)", "it wrote none"),
    "stdout_tail": ("the end of the parent's run output", "it wrote none"),
    "debug_history": ("one line per earlier fix attempt in this debug chain", "`(none — this is the first fix attempt)`"),
    "best_score": ("the parent candidate's validation score", "never"),
    "evaluation_report": (
        "the verifier's breakdown of the parent's score, and where it moved against its own parent",
        "the verifier reports none, or `report.enabled` is off",
    ),
    "sibling_summaries": (
        "one line per attempt already made from this parent, so it is not repeated",
        "`(nothing tried from this solution yet)`",
    ),
    "live_experience": ("what searches running beside this one found so far (polled fresh)", "there are none, or they found nothing yet"),
    "prior_ablations": ("the `ablation.md` an earlier improve of this parent measured", "none measured one, or `ablation` is off"),
    "ablation_cue": ("`ablation_cue.md`: measure which components carry the score before changing one", "the operator's `ablation` param is off"),
    "candidates_table": ("one line per candidate copied in to combine: file name, score, summary", "never"),
    "contract": (
        "the harness's contract: how the solution is run and scored, the output interface, "
        "tunable parameters, the holdout rule — the same for every climber, never a template of yours",
        "never (appended when a template leaves the token out)",
    ),
}
