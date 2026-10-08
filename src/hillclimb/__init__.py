"""hillclimb: hillclimbing on verifier-defined problems.

The Python front door (lazy — `import hillclimb` stays dependency-light):

    import hillclimb as hc

    climber = hc.Climber(selector_policy=hc.selectors.Best(num_drafts=5), operator_policy=hc.policies.Greedy(), tuner="optuna")
    budget = hc.Budget(wall_clock="10m", evaluations=40)
    climber.search(hc.Problem("heilbronn-11"), budget=budget)
    climber.best, climber.history, climber.to_frame()
    outcome = hc.run("heilbronn-11", climber=climber, budget="10m")   # the same, as a function
    hc.run_spec("run.yaml")

    hc.Climber, hc.ClimberSpec          a climber, composed or as its block
    hc.Problem                          a problem: one that exists, or one defined from a scoring function
    hc.Budget                           what a search may spend: time, evaluations, tokens, cost
    hc.policies / selectors / operators / tuners / memory / loops
                                        the prebuilt building blocks
    hc.run, hc.run_spec                 one search / a run spec, in this process
    hc.open_search                      an earlier search, to read (`SearchOutcome`)
    hc.run_search, hc.Config            the lower-level entry and the config
    hc.Action                           one move: an operator on a candidate
    hc.register_agent                   your own scripted agent, in this process
"""

__version__ = "0.8.2"  # the ONE version: pyproject.toml reads it at build time (`[tool.hatch.version]`)

__all__ = [
    "__version__",
    "Climber",
    "ClimberSpec",
    "Problem",
    "Budget",
    "Config",
    "NotPortableError",
    "SearchOutcome",
    "open_search",
    "run",
    "run_spec",
    "run_search",
    "execute_search",
    "BenchmarkProvider",
    "register_benchmark_provider",
    "register_agent",
    "Action",
    "catalog",
    "policies",
    "selectors",
    "operators",
    "tuners",
    "memory",
    "loops",
]

_LAZY = {
    "Config": ("hillclimb.config", "Config"),
    "Climber": ("hillclimb.climber", "Climber"),
    "Problem": ("hillclimb.problem", "Problem"),
    "Budget": ("hillclimb.harness.budget", "Budget"),
    "ClimberSpec": ("hillclimb.climber", "ClimberSpec"),
    "NotPortableError": ("hillclimb.climber", "NotPortableError"),
    "SearchOutcome": ("hillclimb.results", "SearchOutcome"),
    "open_search": ("hillclimb.results", "open_search"),
    "run": ("hillclimb.api", "run"),
    "run_spec": ("hillclimb.api", "run_spec"),
    "run_search": ("hillclimb.api", "run_search"),
    "execute_search": ("hillclimb.api", "execute_search"),
    "register_agent": ("hillclimb.agents", "register_agent"),
    "Action": ("hillclimb.modules.policies.base", "Action"),
    "BenchmarkProvider": ("hillclimb.benchmark_providers", "BenchmarkProvider"),
    "register_benchmark_provider": (
        "hillclimb.benchmark_providers",
        "register_benchmark_provider",
    ),
}
# the building blocks by name, and the catalog of example problems and
# climbers (`hc.catalog.climber("greedy")`): each a lazy module of its own
_FACADES = ("catalog", "policies", "selectors", "operators", "tuners", "memory", "loops")


def __getattr__(name: str):
    if name in _FACADES:
        import importlib

        return importlib.import_module(f"hillclimb.{name}")
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
