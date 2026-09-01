"""GEPA as a hillclimb search engine (`search.policy: gepa`).

Everything here except driver.py runs without the optional `gepa` package —
the library is imported only when a real search needs the loop (tests inject
a fake driver). Install with: pip install 'hillclimb[gepa]'.
"""

from __future__ import annotations

from hillclimb.integrations.gepa.searcher import GEPASearcher
from hillclimb.search_runner import SearchRunner


def build_gepa_searcher(**deps) -> SearchRunner:
    """The _ENGINES factory: same dependency set build_search_runner passes
    to GreedySearcher, minus the policy (GEPA owns its loop)."""
    return GEPASearcher(**deps)
