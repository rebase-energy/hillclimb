"""How the best is picked on a problem with a hidden split — the selection,
how many candidates get a holdout score, and when — is the climber's."""

from __future__ import annotations

import pytest

from hillclimb.config import Config, ConfigError
from hillclimb.harness.glue import apply_climber_settings
from hillclimb.harness.run import search_selection
from hillclimb.modules.spec import ClimberSpec
from hillclimb.project import scaffold_hillclimb_dir
from tests.catalog_fixture import GREEDY


def test_a_block_carries_how_the_best_is_picked():
    spec = ClimberSpec.model_validate({"operator_policy": f"{GREEDY}:Greedy", "holdout": {"selection": "val", "top_k": 2}})
    assert spec.block()["holdout"] == {"selection": "val", "top_k": 2}
    with pytest.raises(ValueError):
        ClimberSpec.model_validate({"operator_policy": f"{GREEDY}:Greedy", "holdout": {"selection": "best"}})


def test_a_climber_without_holdout_keeps_its_identity_and_one_with_it_is_another():
    from hillclimb.climber import resolve_climber

    plain = resolve_climber(ClimberSpec.model_validate({"operator_policy": f"{GREEDY}:Greedy"}))
    held = resolve_climber(ClimberSpec.model_validate({"operator_policy": f"{GREEDY}:Greedy", "holdout": {"top_k": 2}}))
    # (the recorded identities under tests/fixtures still resolve: see test_climber)
    assert "holdout" not in plain.spec.block()
    assert held.sha256 != plain.sha256


def test_a_search_picks_as_its_climber_says():
    from hillclimb.climber import resolve_climber

    config = Config()
    held = resolve_climber(ClimberSpec.model_validate(
        {"operator_policy": f"{GREEDY}:Greedy", "holdout": {"selection": "holdout", "timing": "after"}}
    ))
    apply_climber_settings(config, held)
    assert config.holdout.selection == "holdout" and config.holdout.timing == "after"
    assert config.holdout.top_k == 5  # unset: the default
    assert held.holdout_timing == "after"


def test_views_read_the_selection_the_search_recorded():
    from types import SimpleNamespace

    meta = SimpleNamespace(climber_spec={"operator_policy": "x.py:Greedy", "holdout": {"selection": "val"}})
    assert search_selection(meta) == "val"
    assert search_selection(SimpleNamespace(climber_spec={})) == "rank-blend"
    assert search_selection(None) == "rank-blend"


def test_no_config_file_sets_it(tmp_path):
    folder = scaffold_hillclimb_dir(tmp_path / "hc")
    (folder / "hillclimb.yaml").write_text("holdout: {top_k: 3}\n")
    with pytest.raises(ConfigError, match="holdout is the climber's"):
        Config.load(start=folder)
    (folder / "hillclimb.yaml").write_text("holdout: {enabled: false}\n")
    with pytest.raises(ConfigError, match="holdout.enabled is the problem's"):
        Config.load(start=folder)
