"""Small valid-by-construction seeds for the Einstein Arena smoke suite."""

from __future__ import annotations


BASELINES: dict[str, str] = {
    "circle-packing": '''"""Trivial valid Einstein Arena seed: 26 zero-radius circles."""
import json
from pathlib import Path

circles = [[0.5, 0.5, 0.0] for _ in range(26)]
Path("submission.json").write_text(json.dumps({"circles": circles}))
''',
    "difference-bases": '''"""Trivial valid Einstein Arena difference basis."""
import json
from pathlib import Path

Path("submission.json").write_text(json.dumps({"set": [0, 1]}))
''',
    "heilbronn-triangles": '''"""Trivial valid Einstein Arena seed: coincident points."""
import json
from pathlib import Path

points = [[0.0, 0.0] for _ in range(11)]
Path("submission.json").write_text(json.dumps({"points": points}))
''',
}
