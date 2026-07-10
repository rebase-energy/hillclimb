---
concept: geometric-optimization
built_at: '2026-07-10T16:32:51.341998+00:00'
source_claims:
- 1aa35b2e37b3
- 418521a12e1a
- a7837def4474
- d1dd0f6b32ef
- fd273b33c923
source_searches:
- 20260710-115250-memory-graph-demo/circle-packing
- 20260710-155719-credit-demo/circle-packing
---

# Geometric-optimization playbook

Start here: formulate the layout as a constrained nonlinear program over
continuous coordinates (positions, radii, angles) and solve it with a
joint SLSQP pass over all free variables at once, rather than optimizing
objects one at a time. This is the strongest-endorsed technique in the
evidence and should be the default solver for any packing/arrangement
problem with smooth, differentiable constraints.

Prefer:
- Supply an analytic (hand-derived or autodiff) Jacobian for the
  constraint set instead of relying on finite-difference gradients.
  Gradient-based solvers converge faster and more reliably with exact
  derivatives, especially as the number of overlapping/containment
  constraints grows.
- After any local optimization pass, run a feasibility-repair step that
  projects a slightly-infeasible solution back onto the constraint
  manifold (e.g. small overlap/boundary violations). Cheap insurance
  against solver tolerance drift; consider this a suggestion rather than
  a hard requirement since it's untested at scale, but low-cost to add.
- For escaping local optima, use monotonic basin hopping (perturb +
  re-optimize + accept-if-better) rather than greedy sequential
  construction (placing/growing objects one at a time in a fixed order).
  Basin hopping consistently found better arrangements than greedy
  build-up.

Avoid:
- Physics-inspired heuristics that move objects via simulated
  repulsion/attraction ("center movement," force-directed relaxation,
  etc.) as a primary optimization strategy. This underperformed in
  practice (measured below its own authored expectation) — treat any
  physics-simulation approach to geometric layout as a known dead end
  unless paired with a real constrained solver on top of it.

General ordering: get a joint SLSQP solve working end-to-end first (with
numeric gradients if you must), then add the analytic Jacobian, then layer
basin hopping for multi-start/escape, then add feasibility-repair as a
final polish step. Don't invest in physics-style movement heuristics as a
substitute for a real constrained solver.
