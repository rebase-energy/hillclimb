"""Write a hillclimb dir's config the way it is split: the run defaults in
runs/config.yaml, the rest in hillclimb.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from hillclimb.config import split_config


def write_config(root: Path, data: dict | None = None) -> None:
    general, runs = split_config(data or {})
    root.mkdir(parents=True, exist_ok=True)
    (root / "hillclimb.yaml").write_text(yaml.safe_dump(general) if general else "")
    (root / "runs").mkdir(exist_ok=True)
    (root / "runs" / "config.yaml").write_text(yaml.safe_dump(runs) if runs else "")
