# Write the playbook for {{concept}} problems

You are consolidating what an auto-hillclimbing system has learned across
many searches into a compact playbook for **{{concept}}** problems. Future
draft agents will read this playbook INSTEAD of the raw claims below, so it
must carry everything worth acting on and nothing else.

Write ONE file in the current directory: `playbook.md`. Do not modify any
other file.

## The evidence: {{n_claims}} accumulated claims

Each line is a typed claim from a finished search. "measured" values are
real track records — how the claim's advice actually paid off when injected
into later searches (these outrank authored confidence).

{{claims_digest}}

## Rules

- Markdown prose, at most {{max_chars}} characters. Aim for 10-20 tight
  lines: a "start here" recommendation, what to prefer, what to avoid, known
  failure modes. No headers deeper than `###`, no preamble about being a
  playbook.
- Weight by evidence: a claim measured well across searches is a directive
  ("use X"); an untested authored claim is a suggestion ("consider X"); a
  claim with a poor measured record should be inverted or dropped.
- Preserve NEGATIVE knowledge — approaches that burned budget without
  scoring are the most expensive lessons here. Keep them as explicit
  "avoid" lines.
- Generalize: write for the CONCEPT, not for any single competition.
  Drop problem-specific numbers; keep transferable technique choices,
  library picks, and ordering advice ("get a working X before trying Y").
