"""GEPA as a hillclimb climber (`search.policy: gepa`): a `Loop` that
lets gepa drive the iteration over the harness.

Everything here except driver.py runs without the optional `gepa` package —
the library is imported only when a real search needs the loop (tests inject
a fake driver). Install with: pip install 'hillclimb[gepa]'.
"""

from __future__ import annotations

from hillclimb.integrations.gepa.loop import GepaLoop


def build_gepa_loop(config, *, driver=None, log=print) -> GepaLoop:
    """Validate the config before any spend and hand back the loop (tests
    inject a fake driver; a real search builds the loop from the manifest)."""
    from hillclimb.integrations.gepa.config import validate_gepa_search_config

    return GepaLoop(validate_gepa_search_config(config), driver=driver, log=log)
