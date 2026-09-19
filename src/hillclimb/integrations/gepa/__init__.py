"""GEPA as a hillclimb search engine (`search.policy: gepa`).

Everything here except driver.py runs without the optional `gepa` package —
the library is imported only when a real search needs the loop (tests inject
a fake driver). Install with: pip install 'hillclimb[gepa]'.
"""

from __future__ import annotations

from hillclimb.integrations.gepa.searcher import GEPASearcher
from hillclimb.search_strategy import SearchStrategy


def build_gepa_searcher(**deps) -> SearchStrategy:
    """The _ENGINES factory: same dependency set build_search_strategy passes
    to a policy-driven search, minus the policy (GEPA owns its loop)."""
    return GEPASearcher(**deps)
