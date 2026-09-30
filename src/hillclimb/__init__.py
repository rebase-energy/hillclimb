"""hillclimb: hillclimbing on verifier-defined problems.

The Python front door (lazy — `import hillclimb` stays dependency-light):

    import hillclimb as hc

    climber = hc.Climber(policy=hc.policies.Greedy(num_drafts=5), tuner="optuna")
    outcome = hc.run("heilbronn-11", climber=climber, budget="10m")
    hc.run_spec("run.yaml")

    hc.Climber, hc.ClimberSpec          a climber, composed or as its block
    hc.policies / selectors / operators / tuners / memory / loops
                                        the prebuilt building blocks
    hc.run, hc.run_spec                 one search / a run spec, in this process
    hc.run_search, hc.Config            the lower-level entry and the config
"""

__version__ = "0.5.0"

__all__ = [
    "__version__",
    "Climber",
    "ClimberSpec",
    "Config",
    "NotPortableError",
    "SearchOutcome",
    "run",
    "run_spec",
    "run_search",
    "execute_search",
    "BenchmarkProvider",
    "register_benchmark_provider",
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
    "ClimberSpec": ("hillclimb.climber", "ClimberSpec"),
    "NotPortableError": ("hillclimb.climber", "NotPortableError"),
    "SearchOutcome": ("hillclimb.api", "SearchOutcome"),
    "run": ("hillclimb.api", "run"),
    "run_spec": ("hillclimb.api", "run_spec"),
    "run_search": ("hillclimb.api", "run_search"),
    "execute_search": ("hillclimb.api", "execute_search"),
    "BenchmarkProvider": ("hillclimb.benchmark_providers", "BenchmarkProvider"),
    "register_benchmark_provider": (
        "hillclimb.benchmark_providers",
        "register_benchmark_provider",
    ),
}
# the building blocks by name: each a lazy module of its own
_FACADES = ("policies", "selectors", "operators", "tuners", "memory", "loops")


def __getattr__(name: str):
    if name in _FACADES:
        import importlib

        return importlib.import_module(f"hillclimb.{name}")
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
