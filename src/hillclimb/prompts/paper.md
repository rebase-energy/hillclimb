# Distill claims from a paper

The PDF `{{pdf_name}}` in the current directory is a paper the user wants
future hillclimbing searches to draw on{{scope_line}}. Read it (the Read tool
handles PDFs; work through it in page ranges) and distill its *actionable*
content into typed, reusable claims — the method structure, feature tricks,
hyperparameter choices, loss-specific insights, and pitfalls a coding agent
could act on when writing a solution. Not a summary.

Write ONE file in the current directory: `{{claims_filename}}`. Do not modify
any other file.

## Known entities (reuse these slugs — do NOT invent near-duplicates)

{{entities_block}}

## Concept ontology (closed set — classify entities using ONLY these slugs)

{{concepts_block}}

## Output format

```yaml
paper_title: "The paper's actual title"
claims:
  - subject: analog-ensemble          # canonical entity slug (see entities above)
    relation: helps                   # one of: {{relations}}
    object: ""                        # second entity slug for outperforms/requires; else ""
    confidence: 0.7                   # your calibrated belief this transfers to the scoped problems
    evidence: [p4, p7]                # page references in the paper
entities:                             # every claim subject/object must appear here or in known entities
  - slug: analog-ensemble
    kind: technique                   # technique | library | model_family | feature | practice
    aliases: [AnEn, analog ensemble]
    concepts: [probabilistic, forecasting]   # ONLY slugs from the ontology above
proposed_concepts: []                 # RARE: only when nothing in the ontology fits
```

## Rules

- 4-10 claims, quality over quantity. Every claim must be traceable to the
  paper (cite pages as evidence). No claims from your own prior knowledge.
- Prefer the claims a solution author can ACT on: "X with Y-style features
  helps for this metric" beats restating the paper's final score.
- Papers assert, they don't prove transfer: cap confidence at 0.8 unless the
  paper demonstrates the effect across several datasets.
- Reuse existing entity slugs and aliases; only create a new entity when the
  technique is genuinely new to the registry.
- Classify each new entity into 1-3 ontology concepts. Proposing a new
  concept is a last resort; most techniques fit the existing set.
- `outperforms` needs an `object` (what it beat, in the paper's experiments).
  `no_effect`/`hurts` record the paper's negative results — they save budget.
