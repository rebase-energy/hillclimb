# meta-heilbronn: improve the climber, not the triangles

Your `solution.py` is not a solution to a Heilbronn problem. It is the
**climber** — the search process that decides what an inner hillclimb search
tries next — and it is scored by how far the inner searches it drives climb
on the Heilbronn ladder (`heilbronn-11`, `heilbronn-14`: place N points in
the unit square to maximize the smallest triangle area; see `problem/`'s
sibling problem folders for their descriptions and verifiers).

The verifier runs one real inner search per problem in `problem/grade.yaml`,
each with its own wall-clock budget and the same coding agent that is
writing you, and reports **gap closed**: per problem, the fraction of the
distance from the inner search's floor (its baseline) to the best known
value that the search covered, averaged over problems. 0 means the inner
searches never beat their floor; 1 means they matched the literature.

The starting point is the bundled greedy climber as one file, both of its
policies written out: the selector policy (`Best.schedule`: debug a failing
tip while its chain is shallow, ensemble in the final budget window, draft
until a few branches hold a scored solution, otherwise the best) and the
operator policy (`Greedy.propose`: tune a candidate that declared
parameters, otherwise improve). Everything in that file is yours to change —
the order of those rules, their thresholds (`DEFAULTS`), the number of
drafts, when to give up on a chain, what an operator's prompt says — but the
inner budget is short, so what matters most is what the first few attempts
are spent on.
