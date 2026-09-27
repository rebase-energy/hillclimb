`solution.py` is a one-file climber: it imports from `hillclimb.sdk` (and the
standard library) only, defines exactly one search policy (`propose` +
`observe`, or `POLICY = ...`), and may define `Operator` subclasses whose
`prepare()` returns the prompt an inner agent reads. It is loaded with
`hillclimb run <inner problem> --climber solution.py`; if it does not load,
the verifier fails and this candidate is a debug target.

A verifier run costs `problem/meta.yaml`'s worth of inner searches. Check
the file cheaply before you finish:

    $HILLCLIMB_ENGINE_PYTHON -m hillclimb.cli meta check --climber solution.py
