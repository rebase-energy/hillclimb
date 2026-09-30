"""Memory: what a search knows from other searches, and leaves for the next.

`base.py` is the contract (`Memory`, and the `GraphModule` that indexes one);
`files.py` the built-in (`memory: files`): knowledge cards, claims, credit,
consolidation, papers and skills — YAML files under knowledge/ — and `none`.
A climber's block names its memory like any module and sets its behaviour in
`memory_params`.
"""
