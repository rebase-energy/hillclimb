"""GEPA as a hillclimb climber: a `Loop` that lets gepa drive the iteration
over the harness. This file IS the climber (`climber: climbers/gepa/policy.py`
in hillclimb.yaml, `--climber climbers/gepa/policy.py`); the loop and what it
is built from sit beside it — `loop.py` (the `Loop`), `operator.py` (the one
reflective mutation, `gepa-reflect`, rendering `prompts/gepa_reflect.md`),
`evaluator.py` (gepa's scoring view on the harness), `proposer.py`,
`config.py` (the params, validated before any spend) and `driver.py`, the one
module that imports the `gepa` library. Everything but `driver.py` runs
without the optional extra; install it with: pip install 'hillclimb[gepa]'.

gepa drives proposal order, Pareto selection and its checkpoint; everything
that costs or counts is `harness.run(...)`: a reflective mutation is one
`gepa-reflect` attempt, any other text gepa evaluates (a merge) is an
`inject`, and results are cached by `source_hash`. Budgets in every
dimension, the cost ceiling and stop/park bind gepa through the harness.
Its state never meets a holdout value (`holdout_timing = "after"`).
"""

from hillclimb import Climber

from .loop import GepaLoop

climber = Climber(
    loop=GepaLoop,  # the control flow itself, instead of a selector policy and an operator policy
    name="gepa",
)
