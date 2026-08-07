# Distill claims from a finished search

A hillclimbing search on **{{problem_id}}** (family: {{family}}, metric:
{{metric}}, {{direction}}, budget {{budget_s}}s) just finished. Your job is to
distill what it taught us into a small set of typed, reusable claims that
future searches on this or similar problems can act on.

Write ONE file in the current directory: `{{claims_filename}}`. Do not modify
any other file.

## What happened in the search

{{search_digest}}

## The selected solution

{{solution_excerpt}}

## Known entities (reuse these slugs — do NOT invent near-duplicates)

{{entities_block}}

## Concept ontology (closed set — classify entities using ONLY these slugs)

{{concepts_block}}

## Output format

```yaml
claims:
  - subject: histgradientboosting     # canonical entity slug (see entities above)
    relation: helps                   # one of: {{relations}}
    object: ""                        # second entity slug for outperforms/requires; else ""
    confidence: 0.8                   # your calibrated belief this transfers to similar problems
    evidence: [c030, c031]            # candidate ids from the search record above
entities:                             # every claim subject/object must appear here or in known entities
  - slug: histgradientboosting
    kind: technique                   # technique | library | model_family | feature | practice
    aliases: [HistGradientBoosting, HGB]
    concepts: [decision-trees, tabular]   # ONLY slugs from the ontology above
proposed_concepts: []                 # RARE: only when nothing in the ontology fits
#  - slug: graph-neural-networks
#    dimension: model-family
#    description: ...
```

## Rules

- 3-8 claims, quality over quantity. A claim must be supported by score
  movements or failures you can point to in the record (cite candidate ids
  as evidence). No speculation.
- Prefer claims that TRANSFER: "X helps on this problem family" beats
  restating a single score.
- Reuse existing entity slugs and aliases; only create a new entity when the
  technique is genuinely new to the registry.
- Classify each new entity into 1-3 ontology concepts. Proposing a new
  concept is a last resort; most techniques fit the existing set.
- `outperforms` needs an `object` (what it beat, in this search). `fails_with`
  means the subject caused failures/errors. `no_effect` records a tried idea
  that did not move the metric — negative results save future budget.
