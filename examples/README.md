# Examples: hillclimb from Python

Small scripts that build a climber out of its modules, search a problem with
it, and read what it found. They run on `fitness-landscape`, where a solution
is one point on a terrain and the score is the height there, with Claude Code
writing the solutions. Pass `toy` to run them on the free scripted agent
instead, which takes seconds.

| Script | Shows |
| --- | --- |
| [`nano_climb.py`](nano_climb.py) | the whole thing in one file: a problem, a selector, a policy, two operators, a budget, a search. Start here |
| [`run_and_read.py`](run_and_read.py) | build a climber, `climber.search(problem)`, read the result off it: best, history, spend, a table of candidates |
| [`step_by_step.py`](step_by_step.py) | `climber.start`, then `propose` / `run` / `step` by hand, a move of your own, `finish` |
| [`custom_policy.py`](custom_policy.py) | your own `OperatorPolicy` (what to try next), compared with two others on one budget |
| [`custom_operator.py`](custom_operator.py) | your own `Operator` (how one attempt is made) and the prompt it writes |
| [`custom_selector.py`](custom_selector.py) | your own `SelectorPolicy` (which candidate to expand) under the bundled greedy policy |
| [`custom_agent.py`](custom_agent.py) | `register_agent`: a scripted agent of your own |
| [`define_a_problem.py`](define_a_problem.py) | `Problem(name, score=...)`: a problem of your own from a scoring function, searched |

## Running them

From a checkout of this repository, which is itself a hillclimb dir, with a
logged-in `claude`:

```bash
uv run python examples/nano_climb.py            # Claude Code, a few minutes
uv run python examples/run_and_read.py toy      # the scripted agent, a few seconds
```

`nano_climb.py` defines its own problem (digits of pi), so it has no scripted
stand-in; the others take `toy`.

Anywhere else, after `pip install hillclimb`, make a folder a hillclimb dir
first:

```bash
hillclimb init
python run_and_read.py
```

Each search lands in `runs/` like any other, so `hillclimb watch`,
`hillclimb chart` and `hillclimb surface` (the candidates on the 3D terrain)
show it, live.

## Writing your own

- Keep the code that starts a search under `if __name__ == "__main__":`. A
  search imports the file your classes live in, which runs the file again.
- A module's file imports its contracts from `hillclimb.sdk` (`OperatorPolicy`,
  `Action`, `Operator`, `Attempt`, `SelectorPolicy`, `Selection`, ...).
- The `toy` agent only knows the terrain. A problem of your own needs a real
  coding agent, or a scripted one like the bisector in `define_a_problem.py`.
