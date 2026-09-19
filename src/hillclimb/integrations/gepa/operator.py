"""`gepa-reflect`: one reflective mutation of a solution, as an operator.

gepa hands the loop a parent and the evaluation feedback it wants addressed;
this operator turns that into an ordinary attempt — the parent's solution
copied in, the feedback beside it as `feedback.json`, and a prompt that asks
for one focused change. `require_change` makes an agent that hands the parent
back count as "did not propose" instead of a scored duplicate.
"""

from __future__ import annotations

from hillclimb.sdk import Candidate, Operator, OperatorContext, Preparation

OPERATOR_NAME = "gepa-reflect"
FEEDBACK_FILE = "feedback.json"


class GepaReflectOperator(Operator):
    name = OPERATOR_NAME
    role = "refine"
    needs_target = True

    def valid_target(self, target: Candidate | None) -> str | None:
        reason = super().valid_target(target)
        if reason is None and not target.candidate_dir:
            reason = f"{self.name} targets {target.candidate_id} which has no solution to mutate"
        return reason

    def prepare(self, ctx: OperatorContext) -> Preparation:
        feedback = str(ctx.action.args.get("feedback", ""))
        extra = ""
        if ctx.memory.text:
            extra += f"\n## Prior experience\n\n{ctx.memory.text}\n"
        if ctx.memory.reference_note:
            extra += f"\n{ctx.memory.reference_note}\n"
        prompt = ctx.render(
            "gepa_reflect",
            metric=ctx.problem.metric_name,
            direction=ctx.problem.direction,
            feedback=feedback,
            extra=extra,
            remaining=ctx.budget.remaining_str(),
        )
        return Preparation(
            prompt=prompt,
            copy_parent=True,
            inherit_params=True,
            copy_inspirations=False,
            require_change=True,
            texts={FEEDBACK_FILE: feedback},
        )
