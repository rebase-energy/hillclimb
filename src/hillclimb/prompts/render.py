from __future__ import annotations

from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent

COMPLEXITY_CUES = {
    "minimal": (
        "Produce a MINIMAL, reliable first solution: the simplest sensible model "
        "for this data type (e.g., gradient-boosted trees for tabular data, "
        "TF-IDF + linear model for simple text). No ensembles, no neural networks, "
        "no heavy tuning. Prioritize a correct end-to-end pipeline over cleverness."
    ),
    "moderate": (
        "Produce a MODERATE solution: a solid model with sensible feature "
        "engineering and light hyperparameter tuning. Still no large ensembles."
    ),
    "advanced": (
        "You may produce an ADVANCED solution: stronger models, richer feature "
        "engineering, careful tuning — but it must still run reliably within the "
        "execution time limit."
    ),
}


def render(template_name: str, **context) -> str:
    """Replace {{key}} tokens. Deliberately not str.format: competition
    descriptions routinely contain literal braces."""
    text = (TEMPLATE_DIR / f"{template_name}.md").read_text()
    for key, value in context.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text
